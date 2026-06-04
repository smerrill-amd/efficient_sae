"""Shared, precision-agnostic SAE-training toolkit.

Per-precision entrypoints (train_sae_FP16.py, and later FP8/FP4) define a small
PrecisionPolicy and delegate everything else here:

  utils.py      run-name/tag building, dataset loading, W&B grouping, formatting
  profiling.py  CUDA-synced wall-clock split (LLM forward vs SAE training)
  cli.py        common argparse args + parser builder
  precision.py  PrecisionPolicy ABC (the only precision seam)
  runners.py    run_single / run_multi / dispatch (policy-parameterized)
"""
