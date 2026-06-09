#!/usr/bin/env python3
"""
train_sae_FP8.py — Train a BatchTopK SAE with 8-bit-float (FP8) encoder/decoder matmuls.

This is the FP8 sibling of train_sae_FP16.py. It reuses all the shared orchestration
(CLI, dataset loading, run naming, sweeps, profiling, W&B wiring) from the `sae_train`
package and only swaps in an FP8 PrecisionPolicy:

  * The SAE is the `batchtopk_fp8` architecture from `efficient_sae/architectures`,
    a BatchTopK SAE whose two big matmuls (encoder `sae_in @ W_enc` and decoder
    `feature_acts @ W_dec`) run in fp8 with per-tensor dynamic scaling. Master weights /
    optimizer state stay in the (high-precision) --sae-dtype; the fp8 cast happens
    inside the GEMM — the standard fp8 training recipe.

  * The fp8 *format* is configurable (--fp8-format e4m3 / e3m4 / e2m5 / e5m2 / ...),
    as is the backend (--fp8-backend emulated | hardware | auto). "emulated" works for
    any format (software fake-quant — use it to compare formats); "hardware" uses real
    torch._scaled_mm fp8 GEMMs and only supports the device-native formats (E4M3/E5M2).

Precision policy: master weights and activations share one dtype (--dtype == --sae-dtype,
both float32 by default) and the SAELens autocast/GradScaler is OFF — the only precision
reduction is the explicit fp8 cast in the matmul. (Mixing a bf16 activation dtype with
fp32 weights without autocast would crash the MSE; keep them equal.)

Currently FP8 supports --arch batchtopk only.
"""

import sys
from argparse import ArgumentParser
from pathlib import Path

# Make the repo-root `architectures` package importable from this src/ entrypoint.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from architectures import BatchTopKFP8TrainingSAEConfig  # noqa: E402  (registers arch on import)
from architectures.fp8_formats import FORMATS, get_format  # noqa: E402

from sae_train.cli import build_parser  # noqa: E402
from sae_train.precision import PrecisionPolicy  # noqa: E402
from sae_train.runners import dispatch  # noqa: E402


class FP8Policy(PrecisionPolicy):
    """FP8 recipe: fp32 (or bf16) master weights + activations, fp8 inside the GEMM.

    Autocast/GradScaler are disabled — the fp8 quantization in the encoder/decoder
    matmuls is the precision policy, and it manages its own scaling. To avoid the
    activation-vs-weight dtype mismatch in the MSE loss, --dtype and --sae-dtype must
    match (both float32 by default; both bfloat16 to roughly halve memory).
    """

    name = "fp8"

    def add_dtype_args(self, parser: ArgumentParser) -> None:
        g = parser.add_argument_group("Precision (fp8)")
        g.add_argument("--dtype", default="float32",
                       choices=["float32", "bfloat16"],
                       help="Activation/buffer + accumulation dtype. Must equal "
                            "--sae-dtype (no autocast in fp8 mode).")
        g.add_argument("--sae-dtype", default="float32",
                       choices=["float32", "bfloat16"],
                       help="Master SAE weight/optimizer dtype. fp8 is applied only "
                            "inside the encoder/decoder matmuls, so master weights stay "
                            "high-precision. Must equal --dtype.")
        g.add_argument("--fp8-format", default="e4m3",
                       help="8-bit float layout for the matmuls: e6m1, e5m2, e4m3, "
                            "e3m4, e2m5 (or any eXmY with X+Y==7). More mantissa bits = "
                            "more precision; more exponent bits = more dynamic range. "
                            f"Known: {sorted(FORMATS)}")
        g.add_argument("--fp8-backend", default="auto",
                       choices=["auto", "emulated", "hardware"],
                       help="emulated: software fake-quant (any format; for sweeps). "
                            "hardware: real torch._scaled_mm fp8 GEMM (E4M3/E5M2 only; "
                            "for speed). auto: hardware when available, else emulated.")
        g.add_argument("--fp8-quantize-grads", action="store_true", default=False,
                       help="Also quantize gradients to fp8 (approximate fully-fp8 "
                            "training). Default: fp8 forward, higher-precision backward.")

    def build_sae_cfg(self, args, d_sae: int, k, training_steps: int):
        if args.dtype != args.sae_dtype:
            sys.exit(
                f"ERROR: fp8 mode needs --dtype == --sae-dtype (got {args.dtype} vs "
                f"{args.sae_dtype}); autocast is disabled so mixed activation/weight "
                "dtypes would crash the MSE loss."
            )
        # Validate the format string up front for a clean error message.
        get_format(args.fp8_format)

        if args.arch != "batchtopk":
            sys.exit(
                f"ERROR: train_sae_FP8.py currently supports --arch batchtopk only "
                f"(got {args.arch!r}). The fp8 BatchTopK SAE lives in "
                "efficient_sae/architectures; add a sibling fp8 class for other archs."
            )

        return BatchTopKFP8TrainingSAEConfig(
            d_in=args.d_in,
            d_sae=d_sae,
            dtype=args.sae_dtype,
            device=args.device,
            apply_b_dec_to_input=args.apply_b_dec_to_input,
            normalize_activations=args.normalize_activations,
            k=int(k),
            aux_loss_coefficient=args.aux_loss_coeff,
            fp8_format=args.fp8_format,
            fp8_backend=args.fp8_backend,
            fp8_quantize_grads=args.fp8_quantize_grads,
        )

    def resolve_autocast(self, args) -> bool:
        # fp8 manages precision explicitly inside the GEMM; no SAELens autocast/GradScaler.
        return False


def main() -> None:
    policy = FP8Policy()
    parser = build_parser(policy, description=__doc__)
    args = parser.parse_args()
    dispatch(args, policy)


if __name__ == "__main__":
    main()
