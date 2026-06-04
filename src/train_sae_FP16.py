#!/usr/bin/env python3
"""
train_sae_FP16.py — Train Sparse Autoencoder(s) (ReLU/TopK/BatchTopK) on a language model.

Two modes, selected with --sweep:

  --sweep none   (default)  Train a single SAE at one hook. Uses SAETrainingRunner.

  --sweep layers            Train one SAE per layer, all sharing ONE model and ONE
                            forward pass per batch (MultiSAETrainingRunner). Pass
                            --layers "0,4,8" or --layers all (with --n-layers).

  --sweep k                 Train several SAEs at a single hook with different TopK
                            k values (one model, one forward pass). Pass
                            --k-values "32,64,128".

For the sweep modes the model is loaded once and its activations are multiplexed to
every SAE, so an "all layers" job costs ~1 model copy instead of N.

Outputs are saved to:
  <output-dir>/<run_name>/                 (single)
  <run-dir>/L<layer>/                      (single, when --run-dir given)
  <run-dir>/{checkpoints,output}/...       (sweep; per-SAE subdirs keyed by L<layer> or k<k>)

Run name encodes the key hyperparameters so you can identify a run at a glance:
  <model>__<hook|sweep>__<arch>__x<expansion>__<tokens>tok__<dtype>__<timestamp>

This file is a thin FP16 entrypoint: it defines the FP16 precision policy and
delegates all orchestration to the shared `sae_train` package. New precisions
(FP8/FP4) should be added as sibling entrypoints with their own PrecisionPolicy.
"""

from argparse import ArgumentParser

from sae_lens import (
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
    BatchTopKTrainingSAEConfig,
)

from sae_train.cli import build_parser
from sae_train.precision import PrecisionPolicy
from sae_train.runners import dispatch


class FP16Policy(PrecisionPolicy):
    """Standard fp32-master-weights recipe with bf16/fp16 autocast.

    The SAE weights default to float32 (robust: autocast still runs matmuls in
    bf16 for speed/memory while keeping an fp32 master copy + a working
    GradScaler). Setting --sae-dtype bfloat16 trains weights natively in bf16 and
    disables the SAE-forward autocast/GradScaler (no bf16 kernel for unscale).
    """

    name = "fp16"

    def add_dtype_args(self, parser: ArgumentParser) -> None:
        g = parser.add_argument_group("Precision (fp16)")
        g.add_argument("--dtype", default="bfloat16",
                       choices=["bfloat16", "float16", "float32"],
                       help="Activation/buffer dtype (bfloat16 recommended for ROCm)")
        g.add_argument("--sae-dtype", default="float32",
                       choices=["bfloat16", "float16", "float32"],
                       help="Dtype of the SAE weights + optimizer. Separate from "
                            "--dtype (the activation/buffer dtype). Default float32 is "
                            "the robust recipe: autocast still runs matmuls in bf16 "
                            "(fast, low activation memory) while keeping fp32 master "
                            "weights + a working GradScaler. Set bfloat16 to roughly "
                            "halve per-SAE memory when cramming many layers onto one "
                            "GPU — note this disables the SAE-forward autocast/GradScaler "
                            "(they have no bf16 kernel), so weights train natively in bf16.")

    def build_sae_cfg(self, args, d_sae: int, k, training_steps: int):
        base = dict(
            d_in=args.d_in,
            d_sae=d_sae,
            dtype=args.sae_dtype,
            apply_b_dec_to_input=args.apply_b_dec_to_input,
            normalize_activations=args.normalize_activations,
        )
        if args.arch == "topk":
            return TopKTrainingSAEConfig(
                k=int(k), aux_loss_coefficient=args.aux_loss_coeff, **base
            )
        if args.arch == "batchtopk":
            return BatchTopKTrainingSAEConfig(
                k=int(k), aux_loss_coefficient=args.aux_loss_coeff, **base
            )
        l1_warm = args.l1_warm_up_steps or training_steps // 20
        return StandardTrainingSAEConfig(
            l1_coefficient=args.l1_coeff, l1_warm_up_steps=l1_warm, **base
        )

    def resolve_autocast(self, args) -> bool:
        # bf16 SAE weights are incompatible with the AMP GradScaler that SAELens
        # enables whenever autocast=True (its unscale step has no bf16 kernel, and
        # loss scaling is meaningless for bf16). So disable the SAE-forward autocast
        # when weights are bf16 — they already compute in bf16 natively.
        return args.autocast and args.sae_dtype != "bfloat16"


def main() -> None:
    policy = FP16Policy()
    parser = build_parser(policy, description=__doc__)
    args = parser.parse_args()
    dispatch(args, policy)


if __name__ == "__main__":
    main()
