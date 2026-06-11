from __future__ import annotations

from dataclasses import dataclass

import torch
from typing_extensions import override

from sae_lens.saes.batchtopk_sae import (
    BatchTopKTrainingSAE,
    BatchTopKTrainingSAEConfig,
)

TE_BATCHTOPK_ARCH = "batchtopk_te_fp8"


@dataclass
class BatchTopKTEFP8TrainingSAEConfig(BatchTopKTrainingSAEConfig):
    fp8_recipe: str = "hybrid"
    fp8_scaling: str = "delayed"
    delayed_amax_history_len: int = 16
    delayed_amax_compute_algo: str = "max"
    margin: int = 0
    fp8_aux_loss: bool = True # to avoid the non TE path

    @override
    @classmethod
    def architecture(cls) -> str:
        return TE_BATCHTOPK_ARCH


class BatchTopKTEFP8TrainingSAE(BatchTopKTrainingSAE):
    cfg: BatchTopKTEFP8TrainingSAEConfig  # type: ignore[assignment]

    def __init__(
        self, cfg: BatchTopKTEFP8TrainingSAEConfig, use_error_term: bool = False
    ):
        if getattr(cfg, "use_sparse_activations", False):
            raise ValueError(
                "BatchTopKTEFP8TrainingSAE does not support use_sparse_activations=True."
            )

        # Lazy import: keeps this module loadable on machines without TE.
        try:
            import transformer_engine.pytorch as te  # noqa: F401
            from transformer_engine.common.recipe import DelayedScaling, Format
        except ImportError as e:
            raise ImportError(
                "BatchTopKTEFP8TrainingSAE requires transformer_engine.pytorch. "
            ) from e

        super().__init__(cfg, use_error_term)

        fmt = {"hybrid": Format.HYBRID, "e4m3": Format.E4M3, "e5m2": Format.E5M2}[
            cfg.fp8_recipe.lower()
        ]
        scaling = cfg.fp8_scaling.lower()
        if scaling == "delayed":
            self._fp8_recipe = DelayedScaling(
                fp8_format=fmt,
                amax_history_len=cfg.delayed_amax_history_len,
                amax_compute_algo=cfg.delayed_amax_compute_algo,
                margin=cfg.margin,
            )
        elif scaling == "current":
            # Per-tensor dynamic ("current") scaling. Newer TE versions expose this
            # as a first-class recipe; on older versions we approximate it with a
            # length-1 amax history using "most_recent" (no smoothing across steps).
            try:
                from transformer_engine.common.recipe import Float8CurrentScaling
                self._fp8_recipe = Float8CurrentScaling(fp8_format=fmt)
            except ImportError:
                import warnings
                warnings.warn(
                    "fp8_scaling='current' requested but this transformer_engine build "
                    "does not expose Float8CurrentScaling; falling back to "
                    "DelayedScaling(amax_history_len=1, 'most_recent').",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._fp8_recipe = DelayedScaling(
                    fp8_format=fmt,
                    amax_history_len=1,
                    amax_compute_algo="most_recent",
                    margin=cfg.margin,
                )
        else:
            raise ValueError(
                f"fp8_scaling must be 'delayed' or 'current', got {cfg.fp8_scaling!r}"
            )
        self._te = te

        master_weight_dtype = self.W_enc.dtype
        device = self.W_enc.device
        has_b_dec = hasattr(self, "b_dec") and "b_dec" in self._parameters
        self.enc_lin = te.Linear(cfg.d_in, cfg.d_sae, bias=True,
                                 params_dtype=master_weight_dtype).to(device)
        self.dec_lin = te.Linear(cfg.d_sae, cfg.d_in, bias=has_b_dec,
                                 params_dtype=master_weight_dtype).to(device)
        with torch.no_grad():
            self.enc_lin.weight.copy_(self.W_enc.t())
            self.dec_lin.weight.copy_(self.W_dec.t())
            self.enc_lin.bias.copy_(self.b_enc)
            if has_b_dec:
                self.dec_lin.bias.copy_(self.b_dec)

        # Drop the base class's W_enc / W_dec / b_enc / b_dec Parameters
        del self._parameters["W_enc"]
        del self._parameters["W_dec"]
        del self._parameters["b_enc"]
        if has_b_dec:
            del self._parameters["b_dec"]
        self._has_b_dec = has_b_dec

    @property
    def W_enc(self) -> torch.Tensor:  # type: ignore[override]
        return self.enc_lin.weight.t()

    @property
    def W_dec(self) -> torch.Tensor:  # type: ignore[override]
        return self.dec_lin.weight.t()

    @property
    def b_enc(self) -> torch.Tensor:  # type: ignore[override]
        return self.enc_lin.bias

    @property
    def b_dec(self) -> torch.Tensor:  # type: ignore[override]
        if not self._has_b_dec:
            raise AttributeError("decoder bias is disabled on this SAE config")
        return self.dec_lin.bias


    def _fp8_ctx(self):
        """key TE context """
        return self._te.fp8_autocast(enabled=True, fp8_recipe=self._fp8_recipe)

    @override
    def encode_with_hidden_pre(
        self, x: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        sae_in = self.process_sae_in(x)
        with self._fp8_ctx():
            hidden_pre = self.enc_lin(sae_in)
        hidden_pre = self.hook_sae_acts_pre(hidden_pre)
        if self.cfg.rescale_acts_by_decoder_norm:
            hidden_pre = hidden_pre * self.W_dec.norm(dim=-1)
        feature_acts = self.hook_sae_acts_post(self.activation_fn(hidden_pre))
        return feature_acts, hidden_pre

    @override
    def decode(self, feature_acts: torch.Tensor) -> torch.Tensor:
        acts = feature_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            acts = acts * (1.0 / self.W_dec.norm(dim=-1))
        with self._fp8_ctx():
            sae_out_pre = self.dec_lin(acts)
        sae_out_pre = self.hook_sae_recons(sae_out_pre)
        sae_out_pre = self.run_time_activation_norm_fn_out(sae_out_pre)
        return self.reshape_fn_out(sae_out_pre, self.d_head)

    @override
    def calculate_topk_aux_loss(
        self,
        sae_in: torch.Tensor,
        sae_out: torch.Tensor,
        hidden_pre: torch.Tensor,
        dead_neuron_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        if not self.cfg.fp8_aux_loss:
            return super().calculate_topk_aux_loss(
                sae_in=sae_in, sae_out=sae_out,
                hidden_pre=hidden_pre, dead_neuron_mask=dead_neuron_mask,
            )

        from sae_lens.saes.topk_sae import calculate_topk_aux_acts

        if dead_neuron_mask is None or (num_dead := int(dead_neuron_mask.sum())) == 0:
            return sae_out.new_tensor(0.0)
        if self.cfg.normalize_activations in ("constant_norm_rescale", "layer_norm"):
            raise ValueError(
                "TopK auxiliary loss does not support activation normalization "
                f"(normalize_activations={self.cfg.normalize_activations!r})."
            )

        residual = (sae_in - sae_out).detach()
        k_aux = sae_in.shape[-1] // 2
        scale = min(num_dead / k_aux, 1.0)
        k_aux = min(k_aux, num_dead)

        auxk_acts = calculate_topk_aux_acts(
            k_aux=k_aux, hidden_pre=hidden_pre, dead_neuron_mask=dead_neuron_mask,
        )
        acts = auxk_acts
        if self.cfg.rescale_acts_by_decoder_norm:
            acts = acts * (1.0 / self.W_dec.norm(dim=-1))
        with self._fp8_ctx():
            recons = self.dec_lin(acts)
        # AuxK reconstruction is W_dec @ z (no bias). dec_lin added b_dec; remove it.
        if self._has_b_dec:
            recons = recons - self.dec_lin.bias

        recons = self.reshape_fn_out(recons, self.d_head)
        auxk_loss = (recons - residual).pow(2).sum(dim=-1).mean()
        return self.cfg.aux_loss_coefficient * scale * auxk_loss

    @override
    def process_state_dict_for_saving_inference(
        self, state_dict: dict[str, "torch.Tensor"]
    ) -> None:
        state_dict["W_enc"] = state_dict["enc_lin.weight"].t().contiguous()
        state_dict["W_dec"] = state_dict["dec_lin.weight"].t().contiguous()
        state_dict["b_enc"] = state_dict["enc_lin.bias"].clone()
        if self._has_b_dec:
            state_dict["b_dec"] = state_dict["dec_lin.bias"].clone()

        for key in list(state_dict.keys()):
            if key.startswith("enc_lin.") or key.startswith("dec_lin."):
                del state_dict[key]

        super().process_state_dict_for_saving_inference(state_dict)


def register(overwrite: bool = False) -> None:
    """Register the TE-fp8 BatchTopK architecture with SAELens"""
    from sae_lens import registry

    already = TE_BATCHTOPK_ARCH in registry.SAE_TRAINING_CLASS_REGISTRY
    if already and not overwrite:
        return
    if already and overwrite:
        registry.SAE_TRAINING_CLASS_REGISTRY.pop(TE_BATCHTOPK_ARCH, None)

    registry.register_sae_training_class(
        TE_BATCHTOPK_ARCH,
        BatchTopKTEFP8TrainingSAE,
        BatchTopKTEFP8TrainingSAEConfig,
    )