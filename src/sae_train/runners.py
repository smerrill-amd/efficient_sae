"""Precision-agnostic training orchestration.

`run_single` and `run_multi` hold all the boilerplate (run naming, save paths,
config-summary prints, W&B grouping, profiler wiring, the runner invocation and
final report). They are written against a `PrecisionPolicy` and touch precision
only at two seams:

  * policy.build_sae_cfg(...)   when constructing each SAE config
  * policy.resolve_autocast(args) when deciding the SAE-forward autocast

`dispatch` does the shared validation + single-vs-sweep routing so each
entrypoint's main() stays a few lines.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    MultiSAETrainingRunner,
    MultiSAETrainingRunnerConfig,
    LoggingConfig,
)

from sae_train.precision import PrecisionPolicy
from sae_train.profiling import ProfilingComplete, TimingProfiler
from sae_train.utils import (
    DTYPE_SHORT,
    arch_tag,
    build_run_name,
    configure_wandb_env,
    dump_resolved_cfg,
    enable_determinism,
    fmt_duration,
    layer_from_hook,
    maybe_load_dataset,
    parse_int_list,
    set_seed,
)


# ---------------------------------------------------------------------------
# Single-SAE training (SAETrainingRunner)
# ---------------------------------------------------------------------------

def run_single(args, policy: PrecisionPolicy) -> None:
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
    elif args.arch == "jumprelu":
        print(f"  l0_coeff  {args.l0_coeff}")
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

    sae_cfg = policy.build_sae_cfg(args, d_sae, args.k, training_steps)

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

    sae_autocast = policy.resolve_autocast(args)

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
    if args.no_save_final:
        runner_kwargs["save_final_checkpoint"] = False

    cfg = LanguageModelSAERunnerConfig(**runner_kwargs)
    dump_resolved_cfg(cfg, "LanguageModelSAERunnerConfig (resolved)")

    override_dataset = maybe_load_dataset(args)

    profiler = None
    if args.profile_timing:
        profiler = TimingProfiler(
            device=args.device,
            max_steps=args.profile_steps,
            report_every=args.profile_report_every,
            json_path=Path(checkpoint_path) / "timing_profile.json",
            log_wandb=not args.no_wandb,
        )
        profiler.install()
        print("Timing profiler ON: splitting wall-clock between LLM forward and "
              "SAE training (CUDA-synced; slightly slower while profiling).\n")

    print("Starting run — watch the 'Training SAE' tqdm bar for live ETA.\n")
    start = time.monotonic()
    try:
        SAETrainingRunner(cfg, override_dataset=override_dataset).run()
    except ProfilingComplete:
        print(f"\nProfiling budget of {args.profile_steps} batches reached — stopping "
              "early (no final checkpoint saved).")
    elapsed = time.monotonic() - start

    if profiler is not None:
        profiler.report(partial=False)

    print(f"\nTraining complete in {fmt_duration(elapsed)}.")
    print(f"Model saved to: {checkpoint_path}")


# ---------------------------------------------------------------------------
# Multi-SAE sweep (MultiSAETrainingRunner — one model, one forward pass)
# ---------------------------------------------------------------------------

def run_multi(args, policy: PrecisionPolicy) -> None:
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
            layers = parse_int_list(args.layers)
        for L in layers:
            name = f"L{L}"
            hook_names[name] = args.hook_template.format(layer=L)
            saes[name] = policy.build_sae_cfg(args, d_sae, args.k, training_steps)
        sweep_tag = f"L{layers[0]}-{layers[-1]}-n{len(layers)}"
        tag = arch_tag(args)
        sweep_desc = f"layers={layers}  k={args.k}"
    else:  # sweep == "k"
        if not args.k_values:
            sys.exit("ERROR: --sweep k requires --k-values '32,64,128'")
        if not args.hook_name:
            sys.exit("ERROR: --sweep k requires --hook-name")
        ks = parse_int_list(args.k_values)
        layer = layer_from_hook(args.hook_name)
        for kk in ks:
            name = f"k{kk}"
            hook_names[name] = args.hook_name
            saes[name] = policy.build_sae_cfg(args, d_sae, kk, training_steps)
        sweep_tag = f"L{layer}-ksweep" if layer is not None else "ksweep"
        tag = f"{args.arch}-k[{min(ks)}-{max(ks)}]"
        sweep_desc = f"hook={args.hook_name}  k_values={ks}"

    # ---- Run name + save locations ------------------------------------------
    model_tag = args.model.split("/")[-1]
    expansion = d_sae // args.d_in
    tokens_tag = f"{args.training_tokens // 1_000_000}Mt"
    dtype_tag = DTYPE_SHORT.get(args.dtype, args.dtype)
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    run_name = args.run_name or (
        f"{model_tag}__{sweep_tag}__{tag}__x{expansion}__{tokens_tag}__{dtype_tag}__{ts}"
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
            default_group = f"{model_tag}__{tag}"
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

    sae_autocast = policy.resolve_autocast(args)

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
        save_final_checkpoint=not args.no_save_final,
        logger=logger_cfg,
    )

    if args.llm_device:
        runner_kwargs["llm_device"] = args.llm_device
    if args.act_store_device:
        runner_kwargs["act_store_device"] = args.act_store_device

    cfg = MultiSAETrainingRunnerConfig(**runner_kwargs)
    dump_resolved_cfg(cfg, "MultiSAETrainingRunnerConfig (resolved)")

    override_dataset = maybe_load_dataset(args)

    profiler = None
    if args.profile_timing:
        profiler = TimingProfiler(
            device=args.device,
            max_steps=args.profile_steps,
            report_every=args.profile_report_every,
            json_path=Path(output_path) / "logs" / "timing_profile.json",
            log_wandb=not args.no_wandb,
        )
        profiler.install()
        print("Timing profiler ON: splitting wall-clock between LLM forward and "
              "SAE training (CUDA-synced; slightly slower while profiling).\n")

    print(f"Starting {args.sweep} sweep over {len(saes)} SAEs — model loaded once, "
          "activations shared.\n")
    start = time.monotonic()
    try:
        MultiSAETrainingRunner(cfg, override_dataset=override_dataset).run()
    except ProfilingComplete:
        print(f"\nProfiling budget of {args.profile_steps} batches reached — stopping "
              "early (no final checkpoint saved).")
    elapsed = time.monotonic() - start

    if profiler is not None:
        profiler.report(partial=False)

    print(f"\nSweep complete in {fmt_duration(elapsed)}.")
    print(f"Outputs saved under: {output_path}")


# ---------------------------------------------------------------------------
# Dispatch (shared validation + single-vs-sweep routing)
# ---------------------------------------------------------------------------

def dispatch(args, policy: PrecisionPolicy) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    # Seed BEFORE anything builds SAEs / the activation buffer (both use the global
    # torch RNG). Always on so every training run is reproducible by default;
    # --deterministic additionally pins CUDA kernels for bit-identical runs.
    if getattr(args, "deterministic", False):
        enable_determinism()
    set_seed(args.seed)
    print(f"[repro] seed={args.seed}"
          + ("  [deterministic CUDA]" if getattr(args, "deterministic", False) else ""))

    if not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm not detected — training on CPU will be very slow.",
              file=sys.stderr)

    if args.arch == "batchtopk" and args.normalize_activations == "constant_norm_rescale":
        sys.exit("ERROR: batchtopk does not support normalize_activations="
                 "'constant_norm_rescale'. Use none, expected_average_only_in, or layer_norm.")

    if args.sweep == "none":
        if not args.hook_name:
            sys.exit("ERROR: --hook-name is required for --sweep none")
        run_single(args, policy)
    else:
        run_multi(args, policy)
