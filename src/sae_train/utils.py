"""Precision-agnostic helpers shared by every train_sae_FP*.py entrypoint.

Run-name / tag building, dataset loading, W&B grouping, and small formatting
utilities. Nothing here knows about a specific numeric precision — the only
precision-aware code lives in precision.py and the per-precision entrypoints.
"""

import os
import re
from datetime import datetime
from pathlib import Path

# Hook-name abbreviations used in run names / wandb tags.
HOOK_ABBREV = {
    ".hook_resid_pre": ".resid_pre",
    ".hook_resid_post": ".resid_post",
    ".hook_mlp_out": ".mlp_out",
    ".hook_attn_out": ".attn_out",
}

# Maps a torch dtype string to a short tag for run names. Precision-specific
# dtypes (fp8/fp4 variants) can extend this from their entrypoint if desired.
DTYPE_SHORT = {
    "bfloat16": "bf16",
    "float16": "fp16",
    "float32": "fp32",
}


def arch_tag(args) -> str:
    """Short architecture tag for run names / wandb tags."""
    if args.arch == "topk":
        return f"topk-k{args.k}"
    if args.arch == "batchtopk":
        return f"batchtopk-k{args.k}"
    if args.arch == "jumprelu":
        return f"jumprelu-l0{args.l0_coeff}"
    return f"standard-l1{args.l1_coeff}"  # standard / relu


def build_run_name(args, d_sae: int) -> str:
    model_tag = args.model.split("/")[-1]

    hook_tag = args.hook_name.replace("blocks.", "L")
    for long, short in HOOK_ABBREV.items():
        hook_tag = hook_tag.replace(long, short)

    expansion = d_sae // args.d_in
    tokens_tag = f"{args.training_tokens // 1_000_000}Mt"
    dtype_tag = DTYPE_SHORT.get(args.dtype, args.dtype)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    return f"{model_tag}__{hook_tag}__{arch_tag(args)}__x{expansion}__{tokens_tag}__{dtype_tag}__{ts}"


def layer_from_hook(hook_name: str):
    """Extract the integer layer index from a hook like 'blocks.12.hook_resid_post'."""
    m = re.search(r"blocks\.(\d+)\.", hook_name)
    return int(m.group(1)) if m else None


def parse_int_list(spec: str) -> list[int]:
    return [int(x) for x in spec.split(",") if x.strip() != ""]


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


def fmt_duration(seconds: float) -> str:
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
    tag = arch_tag(args)
    hook_tag = args.hook_name.replace("blocks.", "L")
    for long, short in HOOK_ABBREV.items():
        hook_tag = hook_tag.replace(long, short)
    expansion = d_sae // args.d_in
    dtype_tag = DTYPE_SHORT.get(args.dtype, args.dtype)
    tokens_tag = f"{args.training_tokens // 1_000_000}Mt"

    # Group every layer of a single run together. When a run directory is given
    # (e.g. .../Llama-3.1-8B/run1) use "<model>/<run>"; otherwise fall back to
    # grouping by model + architecture.
    if run_dir is not None:
        default_group = f"{run_dir.parent.name}/{run_dir.name}"
    else:
        default_group = f"{model_tag}__{tag}"
    os.environ.setdefault("WANDB_RUN_GROUP", default_group)
    os.environ.setdefault("WANDB_JOB_TYPE", "train-sae")

    tags = [model_tag, args.arch, tag, hook_tag, f"x{expansion}", tokens_tag, dtype_tag]
    os.environ.setdefault("WANDB_TAGS", ",".join(tags))

    notes = (
        f"{args.arch} SAE on {args.model} @ {args.hook_name} "
        f"(d_sae={d_sae}, x{expansion}); dataset={args.dataset}"
        + (f"[{args.dataset_config}]" if args.dataset_config else "")
        + f"; {args.training_tokens:,} tokens, dtype={args.dtype}, lr={args.lr}"
    )
    os.environ.setdefault("WANDB_NOTES", notes)


def dump_resolved_cfg(cfg, title: str) -> None:
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
