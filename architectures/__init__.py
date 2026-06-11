"""Workshop for SAE architectural changes — currently FP8 (8-bit float) training.

Layout:
  fp8_formats.py        Float8Format spec + software emulator (any E/M split) + the
                        hardware (torch._scaled_mm) dtype detection. The instrument for
                        systematically comparing fp8 formats (E4M3, E3M4, E2M5, ...).
  fp8_ops.py            fp8_linear(x, W, ...): emulated and hardware fp8 matmul backends.
  batchtopk_fp8_sae.py  BatchTopKFP8TrainingSAE(Config) — a BatchTopK SAE whose encoder
                        and decoder matmuls run in fp8 — plus register().

Importing this package registers the "batchtopk_fp8" architecture with SAELens so it
can be trained through the normal runners (SAETrainingRunner / SyntheticSAERunner).
"""

from architectures.batchtopk_fp8_sae import (
    FP8_BATCHTOPK_ARCH,
    BatchTopKFP8TrainingSAE,
    BatchTopKFP8TrainingSAEConfig,
    register,
)
from architectures.fp8_formats import FORMATS, Float8Format, get_format
from architectures.grouped_batchtopk import (
    GROUPED_BATCHTOPK_ARCH,
    GroupedBatchTopK,
    GroupedBatchTopKTrainingSAE,
    GroupedBatchTopKTrainingSAEConfig,
)
from architectures.grouped_batchtopk import register as register_grouped
from architectures.te_batchtopk_fp8_sae import (
    TE_BATCHTOPK_ARCH,
    BatchTopKTEFP8TrainingSAE,
    BatchTopKTEFP8TrainingSAEConfig,
)
from architectures.te_batchtopk_fp8_sae import register as register_te_fp8

# Register on import so `--arch batchtopk_fp8 / batchtopk_grouped / batchtopk_te_fp8` just work.
register()
register_grouped()
register_te_fp8()

__all__ = [
    "FP8_BATCHTOPK_ARCH",
    "BatchTopKFP8TrainingSAE",
    "BatchTopKFP8TrainingSAEConfig",
    "register",
    "FORMATS",
    "Float8Format",
    "get_format",
    "GROUPED_BATCHTOPK_ARCH",
    "GroupedBatchTopK",
    "GroupedBatchTopKTrainingSAE",
    "GroupedBatchTopKTrainingSAEConfig",
    "register_grouped",
    "TE_BATCHTOPK_ARCH",
    "BatchTopKTEFP8TrainingSAE",
    "BatchTopKTEFP8TrainingSAEConfig",
    "register_te_fp8",
]
