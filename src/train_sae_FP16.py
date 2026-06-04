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
"""

import argparse
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
    BatchTopKTrainingSAEConfig,
    MultiSAETrainingRunner,
    MultiSAETrainingRunnerConfig,
    LoggingConfig,
)

# Repo root = two levels up from src/train_sae_FP16.py
_REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = _REPO_ROOT / "trained_models"


# ---------------------------------------------------------------------------
# Run-name builder
# ---------------------------------------------------------------------------

_HOOK_ABBREV = {
    ".hook_resid_pre": ".resid_pre",
    ".hook_resid_post": ".resid_post",
    ".hook_mlp_out": ".mlp_out",
    ".hook_attn_out": ".attn_out",
}

_DTYPE_SHORT = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}


def _arch_tag(args) -> str:
    """Short architecture tag for run names / wandb tags."""
    if args.arch == "topk":
        return f"topk-k{args.k}"
    if args.arch == "batchtopk":
        return f"batchtopk-k{args.k}"
    return f"relu-l1{args.l1_coeff}"


def build_one_sae_cfg(args, d_sae: int, k, training_steps: int):
    """Build a single TrainingSAEConfig for the chosen architecture.

    `k` is taken as an explicit argument (rather than args.k) so the k-sweep can
    vary it per SAE while sharing everything else.
    """
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


def build_run_name(args, d_sae: int) -> str:
    model_tag = args.model.split("/")[-1]

    hook_tag = args.hook_name.replace("blocks.", "L")
    for long, short in _HOOK_ABBREV.items():
        hook_tag = hook_tag.replace(long, short)

    arch_tag = _arch_tag(args)

    expansion = d_sae // args.d_in
    tokens_tag = f"{args.training_tokens // 1_000_000}Mt"
    dtype_tag = _DTYPE_SHORT.get(args.dtype, args.dtype)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    return f"{model_tag}__{hook_tag}__{arch_tag}__x{expansion}__{tokens_tag}__{dtype_tag}__{ts}"


def layer_from_hook(hook_name: str):
    """Extract the integer layer index from a hook like 'blocks.12.hook_resid_post'."""
    m = re.search(r"blocks\.(\d+)\.", hook_name)
    return int(m.group(1)) if m else None


def maybe_load_dataset(args):
    """Pre-load the HF dataset when a config/subset name is given.

    SAELens only stores `dataset_path` and calls `load_dataset(path, split="train",
    streaming=...)` with no `name=` argument, so datasets that REQUIRE a config
    (e.g. allenai/c4 needs 'en') can't be loaded through config alone. When
    --dataset-config is set we load it here and hand the dataset object to the
    runner via `override_dataset`. Returns None when no config is given, so the
    default in-runner loading path is used.
    """
    if not args.dataset_config:
        return None
    from datasets import load_dataset

    print(f"Pre-loading dataset {args.dataset} [{args.dataset_config}] "
          f"(streaming={not args.no_streaming}) to pass config name through.")
    return load_dataset(
        args.dataset,
        name=args.dataset_config,
        split="train",
        streaming=not args.no_streaming,
        trust_remote_code=True,
    )


def _fmt_duration(seconds: float) -> str:
    seconds = int(seconds)
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def configure_wandb_env(args, d_sae: int, run_dir: Path | None = None) -> None:
    """Set WANDB_* env vars so runs are grouped and tagged for easy filtering.

    SAELens calls wandb.init() with only project/entity/name/id, so group, tags,
    job-type, and notes are picked up from the environment instead.
    """
    model_tag = args.model.split("/")[-1]
    arch_tag = _arch_tag(args)
    hook_tag = args.hook_name.replace("blocks.", "L")
    for long, short in _HOOK_ABBREV.items():
        hook_tag = hook_tag.replace(long, short)
    expansion = d_sae // args.d_in
    dtype_tag = _DTYPE_SHORT.get(args.dtype, args.dtype)
    tokens_tag = f"{args.training_tokens // 1_000_000}Mt"

    # Group every layer of a single run together. When a run directory is given
    # (e.g. .../Llama-3.1-8B/run1) use "<model>/<run>"; otherwise fall back to
    # grouping by model + architecture.
    if run_dir is not None:
        default_group = f"{run_dir.parent.name}/{run_dir.name}"
    else:
        default_group = f"{model_tag}__{arch_tag}"
    os.environ.setdefault("WANDB_RUN_GROUP", default_group)
    os.environ.setdefault("WANDB_JOB_TYPE", "train-sae")

    tags = [model_tag, args.arch, arch_tag, hook_tag, f"x{expansion}", tokens_tag, dtype_tag]
    os.environ.setdefault("WANDB_TAGS", ",".join(tags))

    notes = (
        f"{args.arch} SAE on {args.model} @ {args.hook_name} "
        f"(d_sae={d_sae}, x{expansion}); dataset={args.dataset}"
        + (f"[{args.dataset_config}]" if args.dataset_config else "")
        + f"; {args.training_tokens:,} tokens, dtype={args.dtype}, lr={args.lr}"
    )
    os.environ.setdefault("WANDB_NOTES", notes)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

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

    # ---- SAE architecture ---------------------------------------------------
    arch = p.add_argument_group("SAE Architecture")
    arch.add_argument("--arch", choices=["relu", "topk", "batchtopk"], default="topk",
                      help="SAE architecture: relu (L1 penalty), topk (per-token hard "
                           "sparsity), or batchtopk (per-batch hard sparsity)")
    arch.add_argument("--d-sae", type=int, default=None,
                      help="SAE dictionary size. Overrides --dict-mult if set.")
    arch.add_argument("--dict-mult", type=int, default=8,
                      help="Expansion factor (d_sae = d_in × dict-mult, used when --d-sae not set)")
    arch.add_argument("--normalize-activations",
                      choices=["none", "expected_average_only_in",
                                "constant_norm_rescale", "layer_norm"],
                      default="expected_average_only_in")
    arch.add_argument("--apply-b-dec-to-input", action="store_true", default=False)
    arch.add_argument("--sae-dtype", default="float32",
                      choices=["bfloat16", "float16", "float32"],
                      help="Dtype of the SAE weights + optimizer. Separate from "
                           "--dtype (the activation/buffer dtype). Default float32 is "
                           "the robust recipe: autocast still runs matmuls in bf16 "
                           "(fast, low activation memory) while keeping fp32 master "
                           "weights + a working GradScaler. Set bfloat16 to roughly "
                           "halve per-SAE memory when cramming many layers onto one "
                           "GPU — note this disables the SAE-forward autocast/GradScaler "
                           "(they have no bf16 kernel), so weights train natively in bf16.")

    # ReLU-specific
    relu_g = p.add_argument_group("ReLU SAE (ignored for topk)")
    relu_g.add_argument("--l1-coeff", type=float, default=5.0,
                        help="L1 sparsity coefficient")
    relu_g.add_argument("--l1-warm-up-steps", type=int, default=0,
                        help="Steps to ramp L1 from 0 (0 = 5%% of total steps)")

    # TopK-specific
    topk_g = p.add_argument_group("TopK SAE (ignored for relu)")
    topk_g.add_argument("--k", type=int, default=100,
                         help="Number of active features per token")
    topk_g.add_argument("--aux-loss-coeff", type=float, default=1.0 / 32,
                         help="Dead-neuron auxiliary loss coefficient")

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
    hw.add_argument("--dtype", default="bfloat16",
                    choices=["bfloat16", "float16", "float32"],
                    help="SAE + activations dtype (bfloat16 recommended for ROCm)")
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

    # ---- Output & logging ---------------------------------------------------
    out = p.add_argument_group("Output & Logging")
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                     help="Root directory for saved models")
    out.add_argument("--run-dir", type=Path, default=None,
                     help="Run directory (e.g. .../Llama-3.1-8B/run1). When set, "
                          "this layer's checkpoint is saved to <run-dir>/L<layer>, "
                          "overriding --output-dir/--run-name for the save path.")
    out.add_argument("--n-checkpoints", type=int, default=5,
                     help="Number of intermediate checkpoints to save")
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

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _dump_resolved_cfg(cfg, title: str) -> None:
    """Print every field SAELens will actually use, so the exact run is recorded
    in the logs (and recoverable) before any heavy work starts."""
    print(f"\n{'='*64}")
    print(f"  {title}")
    print(f"{'='*64}")
    try:
        cfg_dict = cfg.to_dict()
    except Exception:
        import dataclasses
        cfg_dict = (dataclasses.asdict(cfg)
                    if dataclasses.is_dataclass(cfg) else dict(vars(cfg)))
    for key in sorted(cfg_dict):
        print(f"  {key:<34} {cfg_dict[key]}")
    print(f"{'='*64}\n")


# ---------------------------------------------------------------------------
# Single-SAE training (SAETrainingRunner)
# ---------------------------------------------------------------------------

def run_single(args) -> None:
    d_sae = args.d_sae if args.d_sae else args.d_in * args.dict_mult
    training_steps = args.training_tokens // args.batch_size
    lr_decay = args.lr_decay_steps or training_steps // 5

    run_name = args.run_name or build_run_name(args, d_sae)

    # Save location: under a run directory as <run-dir>/L<layer> when --run-dir is
    # given (so a run groups all layers + a config + logs), else the legacy
    # <output-dir>/<run_name> layout.
    layer = layer_from_hook(args.hook_name)
    if args.run_dir is not None:
        layer_tag = f"L{layer}" if layer is not None else run_name
        checkpoint_path = str(args.run_dir / layer_tag)
    else:
        checkpoint_path = str(args.output_dir / run_name)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # ---- Print config summary -----------------------------------------------
    print(f"\n{'='*64}")
    print(f"  run       {run_name}")
    print(f"  model     {args.model}")
    print(f"  hook      {args.hook_name}  (d_in={args.d_in})")
    print(f"  arch      {args.arch}  |  d_sae={d_sae}  (x{d_sae // args.d_in})")
    if args.arch in ("topk", "batchtopk"):
        print(f"  k         {args.k}  |  aux_loss={args.aux_loss_coeff}")
    else:
        print(f"  l1_coeff  {args.l1_coeff}")
    print(f"  dataset   {args.dataset}"
          + (f"  [{args.dataset_config}]" if args.dataset_config else ""))
    print(f"  tokens    {args.training_tokens:,}  ({training_steps:,} steps)")
    print(f"  dtype     act={args.dtype}  sae={args.sae_dtype}  |  lr={args.lr}  |  batch={args.batch_size}")
    print(f"  device    {args.device}"
          + (f"  llm={args.llm_device}" if args.llm_device else "")
          + (f"  acts={args.act_store_device}" if args.act_store_device else ""))
    print(f"  output    {checkpoint_path}")
    print(f"{'='*64}\n")

    sae_cfg = build_one_sae_cfg(args, d_sae, args.k, training_steps)

    # ---- W&B run organization (group / tags / notes via env) ----------------
    if not args.no_wandb:
        configure_wandb_env(args, d_sae, run_dir=args.run_dir)

    logger_cfg = LoggingConfig(
        log_to_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=run_name,
        wandb_log_frequency=args.wandb_log_frequency,
        eval_every_n_wandb_logs=args.eval_every_n_wandb_logs,
    )

    # bf16 SAE weights are incompatible with the AMP GradScaler that SAELens
    # enables whenever autocast=True (its unscale step has no bf16 kernel, and
    # loss scaling is meaningless for bf16). So disable the SAE-forward autocast
    # when weights are bf16 — they already compute in bf16 natively.
    sae_autocast = args.autocast and args.sae_dtype != "bfloat16"

    runner_kwargs = dict(
        model_name=args.model,
        hook_name=args.hook_name,
        dataset_path=args.dataset,
        is_dataset_tokenized=args.tokenized,
        streaming=not args.no_streaming,
        context_size=args.context_size,
        sae=sae_cfg,
        lr=args.lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_name=args.lr_scheduler,
        lr_warm_up_steps=args.lr_warm_up_steps,
        lr_decay_steps=lr_decay,
        train_batch_size_tokens=args.batch_size,
        n_batches_in_buffer=args.n_batches_in_buffer,
        training_tokens=args.training_tokens,
        store_batch_size_prompts=args.store_batch_size,
        n_batches_for_norm_estimate=args.n_batches_for_norm_estimate,
        dead_feature_window=args.dead_feature_window,
        feature_sampling_window=args.feature_sampling_window,
        device=args.device,
        seed=args.seed,
        dtype=args.dtype,
        autocast=sae_autocast,
        autocast_lm=args.autocast_lm,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=checkpoint_path,
        logger=logger_cfg,
    )

    if args.llm_device:
        runner_kwargs["llm_device"] = args.llm_device
    if args.act_store_device:
        runner_kwargs["act_store_device"] = args.act_store_device

    cfg = LanguageModelSAERunnerConfig(**runner_kwargs)
    _dump_resolved_cfg(cfg, "LanguageModelSAERunnerConfig (resolved)")

    override_dataset = maybe_load_dataset(args)

    print("Starting run — watch the 'Training SAE' tqdm bar for live ETA.\n")
    start = time.monotonic()
    SAETrainingRunner(cfg, override_dataset=override_dataset).run()
    elapsed = time.monotonic() - start

    print(f"\nTraining complete in {_fmt_duration(elapsed)}.")
    print(f"Model saved to: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Multi-SAE sweep (MultiSAETrainingRunner — one model, one forward pass)
# ---------------------------------------------------------------------------

def _parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def run_multi(args) -> None:
    d_sae = args.d_sae if args.d_sae else args.d_in * args.dict_mult
    training_steps = args.training_tokens // args.batch_size
    lr_decay = args.lr_decay_steps or training_steps // 5

    # ---- Build the per-SAE configs and hook map -----------------------------
    saes: dict = {}
    hook_names: dict = {}

    if args.sweep == "layers":
        if not args.layers:
            sys.exit("ERROR: --sweep layers requires --layers '0,4,8' or --layers all")
        if args.layers.strip() == "all":
            if not args.n_layers:
                sys.exit("ERROR: --layers all requires --n-layers")
            layers = list(range(args.n_layers))
        else:
            layers = _parse_int_list(args.layers)
        for L in layers:
            name = f"L{L}"
            hook_names[name] = args.hook_template.format(layer=L)
            saes[name] = build_one_sae_cfg(args, d_sae, args.k, training_steps)
        sweep_tag = f"L{layers[0]}-{layers[-1]}-n{len(layers)}"
        arch_tag = _arch_tag(args)
        sweep_desc = f"layers={layers}  k={args.k}"
    else:  # sweep == "k"
        if not args.k_values:
            sys.exit("ERROR: --sweep k requires --k-values '32,64,128'")
        if not args.hook_name:
            sys.exit("ERROR: --sweep k requires --hook-name")
        ks = _parse_int_list(args.k_values)
        layer = layer_from_hook(args.hook_name)
        for kk in ks:
            name = f"k{kk}"
            hook_names[name] = args.hook_name
            saes[name] = build_one_sae_cfg(args, d_sae, kk, training_steps)
        sweep_tag = f"L{layer}-ksweep" if layer is not None else "ksweep"
        arch_tag = f"{args.arch}-k[{min(ks)}-{max(ks)}]"
        sweep_desc = f"hook={args.hook_name}  k_values={ks}"

    # ---- Run name + save locations ------------------------------------------
    model_tag = args.model.split("/")[-1]
    expansion = d_sae // args.d_in
    tokens_tag = f"{args.training_tokens // 1_000_000}Mt"
    dtype_tag = _DTYPE_SHORT.get(args.dtype, args.dtype)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = args.run_name or (
        f"{model_tag}__{sweep_tag}__{arch_tag}__x{expansion}__{tokens_tag}__{dtype_tag}__{ts}"
    )

    if args.run_dir is not None:
        base = args.run_dir
    else:
        base = args.output_dir / run_name
    output_path = str(base)
    checkpoint_path = str(base / "checkpoints")
    Path(output_path).mkdir(parents=True, exist_ok=True)

    if args.tokenized:
        print("NOTE: --tokenized is ignored in sweep modes (V1 multi-SAE runner has no "
              "pretokenized-dataset support).", file=sys.stderr)

    # ---- Print summary ------------------------------------------------------
    print(f"\n{'='*64}")
    print(f"  run       {run_name}")
    print(f"  sweep     {args.sweep}  ({len(saes)} SAEs, shared model + forward pass)")
    print(f"  model     {args.model}")
    print(f"  arch      {args.arch}  |  d_sae={d_sae}  (x{expansion})")
    print(f"  {sweep_desc}")
    print(f"  SAEs      {', '.join(f'{n}->{hook_names[n]}' for n in saes)}")
    print(f"  dataset   {args.dataset}"
          + (f"  [{args.dataset_config}]" if args.dataset_config else ""))
    print(f"  tokens    {args.training_tokens:,}  ({training_steps:,} steps)")
    print(f"  dtype     act={args.dtype}  sae={args.sae_dtype}  |  lr={args.lr}  |  batch={args.batch_size}")
    print(f"  device    {args.device}"
          + (f"  llm={args.llm_device}" if args.llm_device else "")
          + (f"  acts={args.act_store_device}" if args.act_store_device else ""))
    print(f"  output    {output_path}")
    print(f"{'='*64}\n")

    # ---- W&B run organization (single run; metrics namespaced per SAE) -------
    if not args.no_wandb:
        if args.run_dir is not None:
            default_group = f"{args.run_dir.parent.name}/{args.run_dir.name}"
        else:
            default_group = f"{model_tag}__{arch_tag}"
        os.environ.setdefault("WANDB_RUN_GROUP", default_group)
        os.environ.setdefault("WANDB_JOB_TYPE", "train-sae-sweep")
        tags = [model_tag, args.arch, f"sweep-{args.sweep}", f"x{expansion}",
                tokens_tag, dtype_tag]
        os.environ.setdefault("WANDB_TAGS", ",".join(tags))
        os.environ.setdefault(
            "WANDB_NOTES",
            f"multi-SAE {args.sweep} sweep on {args.model}: {sweep_desc}; "
            f"d_sae={d_sae} (x{expansion}); {args.training_tokens:,} tokens, "
            f"dtype={args.dtype}, lr={args.lr}",
        )

    logger_cfg = LoggingConfig(
        log_to_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=run_name,
        wandb_log_frequency=args.wandb_log_frequency,
        eval_every_n_wandb_logs=args.eval_every_n_wandb_logs,
    )

    # bf16 SAE weights are incompatible with the AMP GradScaler that SAELens
    # enables whenever autocast=True (its unscale step has no bf16 kernel, and
    # loss scaling is meaningless for bf16). So disable the SAE-forward autocast
    # when weights are bf16 — they already compute in bf16 natively.
    sae_autocast = args.autocast and args.sae_dtype != "bfloat16"

    runner_kwargs = dict(
        saes=saes,
        hook_names=hook_names,
        model_name=args.model,
        dataset_path=args.dataset,
        streaming=not args.no_streaming,
        context_size=args.context_size,
        lr=args.lr,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        lr_scheduler_name=args.lr_scheduler,
        lr_warm_up_steps=args.lr_warm_up_steps,
        lr_decay_steps=lr_decay,
        train_batch_size_tokens=args.batch_size,
        n_batches_in_buffer=args.n_batches_in_buffer,
        training_tokens=args.training_tokens,
        store_batch_size_prompts=args.store_batch_size,
        n_batches_for_norm_estimate=args.n_batches_for_norm_estimate,
        dead_feature_window=args.dead_feature_window,
        feature_sampling_window=args.feature_sampling_window,
        device=args.device,
        seed=args.seed,
        dtype=args.dtype,
        autocast=sae_autocast,
        autocast_lm=args.autocast_lm,
        compile_llm=args.compile_llm,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=checkpoint_path,
        output_path=output_path,
        save_final_checkpoint=True,
        logger=logger_cfg,
    )

    if args.llm_device:
        runner_kwargs["llm_device"] = args.llm_device
    if args.act_store_device:
        runner_kwargs["act_store_device"] = args.act_store_device

    cfg = MultiSAETrainingRunnerConfig(**runner_kwargs)
    _dump_resolved_cfg(cfg, "MultiSAETrainingRunnerConfig (resolved)")

    override_dataset = maybe_load_dataset(args)

    print(f"Starting {args.sweep} sweep over {len(saes)} SAEs — model loaded once, "
          "activations shared.\n")
    start = time.monotonic()
    MultiSAETrainingRunner(cfg, override_dataset=override_dataset).run()
    elapsed = time.monotonic() - start

    print(f"\nSweep complete in {_fmt_duration(elapsed)}.")
    print(f"Outputs saved under: {output_path}")


# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm not detected — training on CPU will be very slow.",
              file=sys.stderr)

    if args.arch == "batchtopk" and args.normalize_activations == "constant_norm_rescale":
        sys.exit("ERROR: batchtopk does not support normalize_activations="
                 "'constant_norm_rescale'. Use none, expected_average_only_in, or layer_norm.")

    if args.sweep == "none":
        if not args.hook_name:
            sys.exit("ERROR: --hook-name is required for --sweep none")
        run_single(args)
    else:
        run_multi(args)


if __name__ == "__main__":
    main()
