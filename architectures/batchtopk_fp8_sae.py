"""FP8 BatchTopK training SAE.

A drop-in variant of SAELens' :class:`BatchTopKTrainingSAE` whose two big matmuls —
the encoder (``sae_in @ W_enc``) and the decoder (``feature_acts @ W_dec``) — run in
8-bit floating point according to a configurable :class:`Float8Format`. Everything else
(BatchTopK activation, global threshold EMA, auxiliary dead-neuron loss, decoder-norm
rescaling, activation normalization, saving as a JumpReLU SAE) is inherited unchanged.

Master weights / optimizer state stay in the config dtype (bf16 or fp32); the fp8 cast
happens *inside* the matmul (per-tensor dynamic scaling), which is the standard fp8
training recipe. Pick the format (E4M3, E3M4, E2M5, ...) and backend
(emulated / hardware) via the config to systematically compare formats.

Notes / deliberate simplifications:
  * The auxiliary (dead-neuron) reconstruction loss is left in high precision. It's a
    small corrective term and keeping it stable matters more than its fp8 purity.
  * The BatchTopK threshold EMA is kept in float64 (as upstream) for numerical safety.
  * ``use_sparse_activations`` is not supported with fp8 (the fp8 GEMMs are dense).
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)

from architectures.fp8_formats import get_format
from architectures.fp8_ops import fp8_linear

FP8_BATCHTOPK_ARCH = "batchtopk_fp8"


@dataclass
class BatchTopKFP8TrainingSAEConfig(BatchTopKTrainingSAEConfig):
    """Config for :class:`BatchTopKFP8TrainingSAE`.

    Adds three fp8 knobs on top of :class:`BatchTopKTrainingSAEConfig`:

    Args:
        fp8_format: Which 8-bit float layout to use for the encoder/decoder matmuls.
            One of ``"e6m1"``, ``"e5m2"``, ``"e4m3"``, ``"e3m4"``, ``"e2m5"`` (or any
            ``"eXmY"`` with X+Y==7). More mantissa bits = more precision; more exponent
            bits = more dynamic range.
        fp8_backend: ``"emulated"`` (software fake-quant, works for every format — use
            this for format sweeps), ``"hardware"`` (real ``torch._scaled_mm``, E4M3 /
            E5M2 only — use this for speed), or ``"auto"`` (hardware when available).
        fp8_quantize_grads: Also quantize gradients to fp8 (approximate fully-fp8
            training). Default False = fp8 forward with higher-precision backward.
    """

    fp8_format: str = "e4m3"
    fp8_backend: str = "emulated"
    fp8_quantize_grads: bool = False

    @override
    @classmethod
    def architecture(cls) -> str:
        return FP8_BATCHTOPK_ARCH


class BatchTopKFP8TrainingSAE(BatchTopKTrainingSAE):
    """BatchTopK training SAE with fp8 encoder/decoder matmuls."""

    cfg: BatchTopKFP8TrainingSAEConfig  # type: ignore[assignment]

    def __init__(
        self, cfg: BatchTopKFP8TrainingSAEConfig, use_error_term: bool = False
    ):
        if getattr(cfg, "use_sparse_activations", False):
            raise ValueError(
                "BatchTopKFP8TrainingSAE does not support use_sparse_activations=True "
                "(the fp8 GEMMs operate on dense tensors)."
            )
        # Validate the format string early so misconfigs fail at construction.
        self._fmt = get_format(cfg.fp8_format)
        super().__init__(cfg, use_error_term)

    def _fp8_matmul(self, x: torch.Tensor, W: torch.Tensor) -> torch.Tensor:
        return fp8_linear(
            x,
            W,
            fmt=self._fmt,
            backend=self.cfg.fp8_backend,
            quantize_grads=self.cfg.fp8_quantize_grads,
        )

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        hidden_pre = self.hook_sae_acts_pre(
            self._fp8_matmul(sae_in, self.W_enc) + self.b_enc
        )
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        acts = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            acts = acts * (1.0 / self.W_dec.norm(dim=-1))
        sae_out_pre = self._fp8_matmul(acts, self.W_dec) + self.b_dec
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)


def register(overwrite: bool = False) -> None:
    """Register the fp8 BatchTopK architecture with SAELens (idempotent).

    Safe to call repeatedly (e.g. from a notebook that re-runs cells): if the
    architecture is already registered it's a no-op unless ``overwrite=True``.
    """
    from sae_lens import registry

    already = FP8_BATCHTOPK_ARCH in registry.SAE_TRAINING_CLASS_REGISTRY
    if already and not overwrite:
        return
    if already and overwrite:
        registry.SAE_TRAINING_CLASS_REGISTRY.pop(FP8_BATCHTOPK_ARCH, None)

    registry.register_sae_training_class(
        FP8_BATCHTOPK_ARCH,
        BatchTopKFP8TrainingSAE,
        BatchTopKFP8TrainingSAEConfig,
    )
