"""Grouped ("ghost-batch") BatchTopK training SAE.

Standard BatchTopK (SAELens) selects ``k * num_samples`` activations over the *entire*
batch — the selection pool is ``batch_size x d_sae``. That makes the optimization batch
size **architectural**: changing it changes which features survive per sample and the EMA
threshold that gets frozen into the exported JumpReLU SAE, not just the gradient noise.

To study the learning-dynamics effect of batch size *independently* of that architectural
effect, this variant splits each batch into fixed-size groups of ``topk_group_size``
samples and runs BatchTopK **within each group** (pool = ``topk_group_size x d_sae``). So
you can grow the optimization batch (more samples per gradient step) while the BatchTopK
selection statistics stay pinned to a fixed reference group size:

    (G*B, d_sae)  --reshape-->  (G, B, d_sae)   # G "ghost" groups of B samples
    top-k taken within each (B, d_sae) group, keeping k*B per group

``topk_group_size <= 0`` (the default) reproduces standard whole-batch BatchTopK exactly,
so this is a no-op unless you opt in.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)

GROUPED_BATCHTOPK_ARCH = "batchtopk_grouped"


def _select_global_batchtopk(acts: torch.Tensor, total_keep: int) -> torch.Tensor:
    """Standard BatchTopK selection over the whole flattened tensor (upstream parity)."""
    flat = acts.flatten()
    topk = torch.topk(flat, total_keep, dim=-1)
    return (
        torch.zeros_like(flat)
        .scatter(-1, topk.indices, topk.values)
        .reshape(acts.shape)
    )


class GroupedBatchTopK(nn.Module):
    """BatchTopK whose top-k pool is a fixed group of ``group_size`` samples.

    ``group_size <= 0`` / ``None`` => whole-batch BatchTopK (identical to upstream).
    If the sample count isn't a multiple of ``group_size`` we fall back to whole-batch
    selection for that step, so a ragged final batch can't crash training.
    """

    def __init__(self, k: float, group_size: int | None = None):
        super().__init__()
        self.k = k
        self.group_size = int(group_size) if group_size else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acts = x.relu()
        num_samples = acts.shape[:-1].numel()
        g = self.group_size
        if g <= 0 or g >= num_samples or num_samples % g != 0:
            return _select_global_batchtopk(acts, int(self.k * num_samples))
        d_sae = acts.shape[-1]
        groups = acts.reshape(num_samples // g, g * d_sae)  # (n_groups, g*d_sae)
        keep = int(self.k * g)
        topk = torch.topk(groups, keep, dim=-1)
        out = torch.zeros_like(groups).scatter(-1, topk.indices, topk.values)
        return out.reshape(acts.shape)


@dataclass
class GroupedBatchTopKTrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """:class:`BatchTopKTrainingSAEConfig` + a fixed-size top-k group.

    Args:
        topk_group_size: Number of samples whose activations share one BatchTopK
            selection. ``0`` (default) = standard whole-batch BatchTopK.
    """

    topk_group_size: int = 0

    @override
    @classmethod
    def architecture(cls) -> str:
        return GROUPED_BATCHTOPK_ARCH


class GroupedBatchTopKTrainingSAE(BatchTopKTrainingSAE):
    """BatchTopK training SAE with a fixed-size (ghost-batch) top-k selection group."""

    cfg: GroupedBatchTopKTrainingSAEConfig  # type: ignore[assignment]

    @override
    def get_activation_fn(self):
        return GroupedBatchTopK(self.cfg.k, self.cfg.topk_group_size)


def register(overwrite: bool = False) -> None:
    """Register the grouped BatchTopK architecture with SAELens (idempotent)."""
    from sae_lens import registry

    already = GROUPED_BATCHTOPK_ARCH in registry.SAE_TRAINING_CLASS_REGISTRY
    if already and not overwrite:
        return
    if already and overwrite:
        registry.SAE_TRAINING_CLASS_REGISTRY.pop(GROUPED_BATCHTOPK_ARCH, None)

    registry.register_sae_training_class(
        GROUPED_BATCHTOPK_ARCH,
        GroupedBatchTopKTrainingSAE,
        GroupedBatchTopKTrainingSAEConfig,
    )
