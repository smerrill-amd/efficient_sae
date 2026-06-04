"""Shared command-line interface.

`add_common_args` registers every argument that is identical across precisions
(model/data, architecture, sweep, training, hardware, profiling, output). The
precision-specific dtype arguments (`--dtype`, `--sae-dtype`) are NOT added
here — each entrypoint contributes those via its PrecisionPolicy.add_dtype_args,
so a precision only owns the knobs that actually differ.
"""

import argparse
from pathlib import Path

from sae_train.precision import PrecisionPolicy

# Repo root = three levels up from src/sae_train/cli.py
_REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "trained_models"


def add_common_args(p: argparse.ArgumentParser) -> None:
    # ---- Model & data -------------------------------------------------------
    md = p.add_argument_group("Model & Data")
    md.add_argument("--model", required=True,
                    help="HuggingFace / TransformerLens model name "
                         "(e.g. meta-llama/Llama-3.1-8B)")
    md.add_argument("--dataset", required=True,
                    help="HuggingFace dataset path (e.g. HuggingFaceFW/fineweb-edu)")
    md.add_argument("--dataset-config", default=None,
                    help="HuggingFace dataset subset/config name (e.g. sample-10BT)")
    md.add_argument("--hook-name", default=None,
                    help="TransformerLens hook point (e.g. blocks.16.hook_resid_post). "
                         "Required for --sweep none/k; for --sweep layers it is built "
                         "from --hook-template instead.")
    md.add_argument("--d-in", type=int, required=True,
                    help="Hidden dimension at the hook (e.g. 4096 for Llama-8B resid)")
    md.add_argument("--tokenized", action="store_true",
                    help="Dataset is already tokenized")
    md.add_argument("--no-streaming", action="store_true",
                    help="Download full dataset instead of streaming")

    # ---- SAE architecture (dtype args added by the precision policy) --------
    arch = p.add_argument_group("SAE Architecture")
    arch.add_argument("--arch",
                      choices=["relu", "standard", "topk", "batchtopk", "jumprelu"],
                      default="topk",
                      help="SAE architecture: standard/relu (L1 penalty; 'relu' is a "
                           "legacy alias for 'standard'), topk (per-token hard sparsity), "
                           "batchtopk (per-batch hard sparsity), or jumprelu (learned "
                           "per-feature threshold with an L0 penalty)")
    arch.add_argument("--d-sae", type=int, default=None,
                      help="SAE dictionary size. Overrides --dict-mult if set.")
    arch.add_argument("--dict-mult", type=int, default=8,
                      help="Expansion factor (d_sae = d_in × dict-mult, used when --d-sae not set)")
    arch.add_argument("--normalize-activations",
                      choices=["none", "expected_average_only_in",
                                "constant_norm_rescale", "layer_norm"],
                      default="expected_average_only_in")
    arch.add_argument("--apply-b-dec-to-input", action="store_true", default=False)

    # Standard / ReLU-specific
    relu_g = p.add_argument_group("Standard / ReLU SAE (ignored for topk/batchtopk/jumprelu)")
    relu_g.add_argument("--l1-coeff", type=float, default=5.0,
                        help="L1 sparsity coefficient")
    relu_g.add_argument("--l1-warm-up-steps", type=int, default=0,
                        help="Steps to ramp L1 from 0 (0 = 5%% of total steps)")

    # TopK / BatchTopK-specific
    topk_g = p.add_argument_group("TopK / BatchTopK SAE (ignored for standard/jumprelu)")
    topk_g.add_argument("--k", type=int, default=100,
                         help="Number of active features per token")
    topk_g.add_argument("--aux-loss-coeff", type=float, default=1.0 / 32,
                         help="Dead-neuron auxiliary loss coefficient")

    # JumpReLU-specific
    jr_g = p.add_argument_group("JumpReLU SAE (ignored for other archs)")
    jr_g.add_argument("--l0-coeff", type=float, default=1.0,
                      help="L0 sparsity penalty coefficient")
    jr_g.add_argument("--l0-warm-up-steps", type=int, default=0,
                      help="Steps to ramp the L0 penalty from 0 (0 = 5%% of total steps)")

    # ---- Multi-SAE sweep (shared model, one forward pass) -------------------
    sw = p.add_argument_group("Multi-SAE sweep (one model, one forward pass per batch)")
    sw.add_argument("--sweep", choices=["none", "layers", "k"], default="none",
                    help="none: single SAE (default). layers: one SAE per layer in "
                         "--layers. k: several SAEs at --hook-name with --k-values. "
                         "Both sweep modes use MultiSAETrainingRunner (shared model).")
    sw.add_argument("--layers", default=None,
                    help="Sweep=layers: comma list of layer indices (e.g. '0,4,8') "
                         "or 'all' (requires --n-layers).")
    sw.add_argument("--n-layers", type=int, default=None,
                    help="Total model layers, used to expand --layers all.")
    sw.add_argument("--k-values", default=None,
                    help="Sweep=k: comma list of TopK k values (e.g. '32,64,128').")
    sw.add_argument("--hook-template", default="blocks.{layer}.hook_resid_post",
                    help="Sweep=layers: hook pattern with a {layer} placeholder.")

    # ---- Training -----------------------------------------------------------
    tr = p.add_argument_group("Training")
    tr.add_argument("--training-tokens", type=int, default=200_000_000,
                    help="Total tokens to train on")
    tr.add_argument("--batch-size", type=int, default=4096,
                    help="Training batch size in tokens")
    tr.add_argument("--context-size", type=int, default=1024,
                    help="Token context length per prompt")
    tr.add_argument("--lr", type=float, default=5e-5)
    tr.add_argument("--lr-scheduler", default="constant",
                    choices=["constant", "cosineannealing",
                              "cosineannealingwarmrestarts"])
    tr.add_argument("--lr-warm-up-steps", type=int, default=0)
    tr.add_argument("--lr-decay-steps", type=int, default=0,
                    help="LR decay steps (0 = 20%% of total steps)")
    tr.add_argument("--adam-beta1", type=float, default=0.9)
    tr.add_argument("--adam-beta2", type=float, default=0.999)
    tr.add_argument("--dead-feature-window", type=int, default=1000)
    tr.add_argument("--feature-sampling-window", type=int, default=2000)

    # ---- Hardware -----------------------------------------------------------
    hw = p.add_argument_group("Hardware / Efficiency")
    hw.add_argument("--device", default="cuda",
                    help="Device for the SAE (e.g. cuda, cuda:0, cpu)")
    hw.add_argument("--llm-device", default=None,
                    help="Device for the LLM (default: same as --device). "
                         "Set to e.g. cuda:1 to split across GPUs.")
    hw.add_argument("--act-store-device", default=None,
                    help="Device for activation buffer (default: same as --device). "
                         "Set to cpu to save VRAM.")
    hw.add_argument("--autocast", action="store_true", default=True,
                    help="Mixed-precision autocast for the SAE forward")
    hw.add_argument("--autocast-lm", action="store_true", default=True,
                    help="Mixed-precision autocast during activation collection")
    hw.add_argument("--n-batches-in-buffer", type=int, default=32,
                    help="Activation shuffle buffer depth (higher = better shuffle)")
    hw.add_argument("--store-batch-size", type=int, default=32,
                    help="LLM prompts per activation collection batch")
    hw.add_argument("--n-batches-for-norm-estimate", type=int, default=50,
                    help="Batches used to estimate the activation-norm scaling factor. "
                         "SAELens caches ALL of these on-device before training starts, "
                         "and in sweep modes each batch holds every hook — so the default "
                         "of 1000 can stage hundreds of GiB. Lowering this avoids an OOM "
                         "during norm estimation.")
    hw.add_argument("--compile-llm", action="store_true", default=False,
                    help="torch.compile the shared LLM (sweep modes only; the model "
                         "is shared so this is amortized across all SAEs)")

    # ---- Profiling ----------------------------------------------------------
    pf = p.add_argument_group("Profiling (single + sweep modes)")
    pf.add_argument("--no-profile-timing", dest="profile_timing", action="store_false", default=True,
                    help="Disable the wall-clock timing profiler. Profiling (the LLM "
                         "forward vs SAE training split, written to stdout + "
                         "timing_profile.json + W&B profile/* plots) is ON by default; "
                         "this skips it and its small CUDA-sync overhead. Works in both "
                         "single-SAE and sweep modes.")
    pf.add_argument("--profile-steps", type=int, default=0,
                    help="With --profile-timing, stop after this many training batches "
                         "and print the breakdown (0 = profile the whole run).")
    pf.add_argument("--profile-report-every", type=int, default=200,
                    help="With --profile-timing, print a running breakdown every N batches.")

    # ---- Output & logging ---------------------------------------------------
    out = p.add_argument_group("Output & Logging")
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                     help="Root directory for saved models")
    out.add_argument("--run-dir", type=Path, default=None,
                     help="Run directory (e.g. .../Llama-3.1-8B/run1). When set, "
                          "this layer's checkpoint is saved to <run-dir>/L<layer>, "
                          "overriding --output-dir/--run-name for the save path.")
    out.add_argument("--n-checkpoints", type=int, default=5,
                     help="Number of intermediate checkpoints to save (0 disables them)")
    out.add_argument("--no-save-final", action="store_true", default=False,
                     help="Do not save the final SAE at the end of training. Useful for "
                          "throughput benchmarks where you only care about timing. (In "
                          "sweep modes the final SAE is saved by default; single mode does "
                          "not save a final checkpoint by default either way.)")
    out.add_argument("--wandb-project", default="efficient_sae")
    out.add_argument("--wandb-entity", default=None)
    out.add_argument("--wandb-log-frequency", type=int, default=30,
                     help="Steps between wandb metric logs")
    out.add_argument("--eval-every-n-wandb-logs", type=int, default=20,
                     help="Run built-in evals every N wandb logs. In sweep modes evals "
                          "run sequentially per SAE, so raise this when sweeping many SAEs.")
    out.add_argument("--no-wandb", action="store_true",
                     help="Disable Weights & Biases logging")
    out.add_argument("--seed", type=int, default=42)
    out.add_argument("--run-name", default=None,
                     help="Override the auto-generated run name")


def build_parser(policy: PrecisionPolicy, description: str | None = None) -> argparse.ArgumentParser:
    """Build the full parser: common args + the policy's precision-specific dtype args."""
    p = argparse.ArgumentParser(
        description=description,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    add_common_args(p)
    policy.add_dtype_args(p)
    return p
