#!/usr/bin/env python3
"""
train_sae_FP16.py — Train a Sparse Autoencoder (ReLU/TopK) on a language model.

Outputs are saved to:
  <output-dir>/<run_name>/
    sae_weights.safetensors
    cfg.json
    sparsity.safetensors   (if produced by SAELens)

Run name encodes all key hyperparameters so you can identify any checkpoint at a glance:
  <model>__<hook>__<arch>__x<expansion>__<tokens>tok__<dtype>__<timestamp>
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

import torch
from sae_lens import (
    LanguageModelSAERunnerConfig,
    SAETrainingRunner,
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
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


def build_run_name(args, d_sae: int) -> str:
    model_tag = args.model.split("/")[-1]

    hook_tag = args.hook_name.replace("blocks.", "L")
    for long, short in _HOOK_ABBREV.items():
        hook_tag = hook_tag.replace(long, short)

    if args.arch == "topk":
        arch_tag = f"topk-k{args.k}"
    else:
        arch_tag = f"relu-l1{args.l1_coeff}"

    expansion = d_sae // args.d_in
    tokens_tag = f"{args.training_tokens // 1_000_000}Mt"
    dtype_tag = _DTYPE_SHORT.get(args.dtype, args.dtype)
    ts = datetime.now().strftime("%Y%m%d_%H%M")

    return f"{model_tag}__{hook_tag}__{arch_tag}__x{expansion}__{tokens_tag}__{dtype_tag}__{ts}"


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
    md.add_argument("--hook-name", required=True,
                    help="TransformerLens hook point (e.g. blocks.16.hook_resid_post)")
    md.add_argument("--d-in", type=int, required=True,
                    help="Hidden dimension at the hook (e.g. 4096 for Llama-8B resid)")
    md.add_argument("--tokenized", action="store_true",
                    help="Dataset is already tokenized")
    md.add_argument("--no-streaming", action="store_true",
                    help="Download full dataset instead of streaming")

    # ---- SAE architecture ---------------------------------------------------
    arch = p.add_argument_group("SAE Architecture")
    arch.add_argument("--arch", choices=["relu", "topk"], default="topk",
                      help="SAE architecture: relu (L1 penalty) or topk (hard sparsity)")
    arch.add_argument("--d-sae", type=int, default=None,
                      help="SAE dictionary size. Overrides --dict-mult if set.")
    arch.add_argument("--dict-mult", type=int, default=8,
                      help="Expansion factor (d_sae = d_in × dict-mult, used when --d-sae not set)")
    arch.add_argument("--normalize-activations",
                      choices=["none", "expected_average_only_in",
                                "constant_norm_rescale", "layer_norm"],
                      default="expected_average_only_in")
    arch.add_argument("--apply-b-dec-to-input", action="store_true", default=False)

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

    # ---- Output & logging ---------------------------------------------------
    out = p.add_argument_group("Output & Logging")
    out.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR,
                     help="Root directory for saved models")
    out.add_argument("--n-checkpoints", type=int, default=5,
                     help="Number of intermediate checkpoints to save")
    out.add_argument("--wandb-project", default="efficient_sae")
    out.add_argument("--wandb-entity", default=None)
    out.add_argument("--no-wandb", action="store_true",
                     help="Disable Weights & Biases logging")
    out.add_argument("--seed", type=int, default=42)
    out.add_argument("--run-name", default=None,
                     help="Override the auto-generated run name")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    if not torch.cuda.is_available():
        print("WARNING: CUDA/ROCm not detected — training on CPU will be very slow.",
              file=sys.stderr)

    d_sae = args.d_sae if args.d_sae else args.d_in * args.dict_mult
    training_steps = args.training_tokens // args.batch_size
    lr_decay = args.lr_decay_steps or training_steps // 5

    run_name = args.run_name or build_run_name(args, d_sae)
    checkpoint_path = str(args.output_dir / run_name)
    Path(checkpoint_path).mkdir(parents=True, exist_ok=True)

    # ---- Print config summary -----------------------------------------------
    print(f"\n{'='*64}")
    print(f"  run       {run_name}")
    print(f"  model     {args.model}")
    print(f"  hook      {args.hook_name}  (d_in={args.d_in})")
    print(f"  arch      {args.arch}  |  d_sae={d_sae}  (x{d_sae // args.d_in})")
    if args.arch == "topk":
        print(f"  k         {args.k}  |  aux_loss={args.aux_loss_coeff}")
    else:
        print(f"  l1_coeff  {args.l1_coeff}")
    print(f"  dataset   {args.dataset}"
          + (f"  [{args.dataset_config}]" if args.dataset_config else ""))
    print(f"  tokens    {args.training_tokens:,}  ({training_steps:,} steps)")
    print(f"  dtype     {args.dtype}  |  lr={args.lr}  |  batch={args.batch_size}")
    print(f"  device    {args.device}"
          + (f"  llm={args.llm_device}" if args.llm_device else "")
          + (f"  acts={args.act_store_device}" if args.act_store_device else ""))
    print(f"  output    {checkpoint_path}")
    print(f"{'='*64}\n")

    # ---- SAE config ---------------------------------------------------------
    sae_base = dict(
        d_in=args.d_in,
        d_sae=d_sae,
        apply_b_dec_to_input=args.apply_b_dec_to_input,
        normalize_activations=args.normalize_activations,
    )

    if args.arch == "topk":
        sae_cfg = TopKTrainingSAEConfig(
            k=args.k,
            aux_loss_coefficient=args.aux_loss_coeff,
            **sae_base,
        )
    else:
        l1_warm = args.l1_warm_up_steps or training_steps // 20
        sae_cfg = StandardTrainingSAEConfig(
            l1_coefficient=args.l1_coeff,
            l1_warm_up_steps=l1_warm,
            **sae_base,
        )

    # ---- Logger config ------------------------------------------------------
    logger_cfg = LoggingConfig(
        log_to_wandb=not args.no_wandb,
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        run_name=run_name,
        wandb_log_frequency=30,
        eval_every_n_wandb_logs=20,
    )

    # ---- Runner config ------------------------------------------------------
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
        dead_feature_window=args.dead_feature_window,
        feature_sampling_window=args.feature_sampling_window,
        device=args.device,
        seed=args.seed,
        dtype=args.dtype,
        autocast=args.autocast,
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

    # ---- Train --------------------------------------------------------------
    SAETrainingRunner(cfg).run()

    print(f"\nTraining complete.")
    print(f"Model saved to: {checkpoint_path}")


if __name__ == "__main__":
    main()
