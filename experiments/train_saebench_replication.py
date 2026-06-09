#!/usr/bin/env python3
"""train_saebench_replication.py — replicate the SAEBench BatchTopK SAE suite.

Trains the full SAEBench BatchTopK grid for ONE model in a SINGLE run: all three
widths (4k / 16k / 65k = 2^12 / 2^14 / 2^16) x all six sparsities
(k in [20, 40, 80, 160, 320, 640]) = 18 SAEs, all hooked to the SAME residual
stream layer and sharing ONE model + ONE forward pass per batch
(MultiSAETrainingRunner). This is the efficient way to "do all widths/ks at once":
the LLM is loaded once and its activations are multiplexed to every SAE.

The hyperparameters are taken verbatim from the SAEBench release configs, e.g.
  https://huggingface.co/adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108
    /blob/main/BatchTopK_gemma-2-2b__0108/resid_post_layer_12/trainer_0/config.json
and the SAEBench paper (arXiv:2503.09532, Appendix B):

  Tokens processed        500M               (244,140 steps @ batch 2048)
  Learning rate           3e-4
  LR warmup (from 0)       1,000 steps
  LR decay (to 0)         last 20%           (decay_start=195,312 -> 48,828 steps)
  Dataset                 The Pile           (monology/pile-uncopyrighted)
  Batch size              2,048 tokens
  LLM context length      1,024
  aux (auxk) coefficient  0.03125  (= 1/32)
  BatchTopK threshold     EMA beta=0.999     (-> topk_threshold_lr=1-beta=0.001)
  seed                    0

Per-model specifics (from the configs):
  gemma-2-2b               layer 12, d_in=2304, top_k_aux=1152, refresh_batch=4
  pythia-160m-deduped      layer  8, d_in= 768, top_k_aux= 384, refresh_batch=32

NOTE ON FIDELITY: SAEBench trained with the `dictionary_learning` library; this
script reuses the repo's SAELens stack instead. All *numeric* hyperparameters
above are matched exactly. A few SAELens-vs-dictionary_learning implementation
details cannot be matched 1:1 and use the closest SAELens equivalent:
  * top_k_aux (dead-feature aux-k) is chosen internally by SAELens; the aux loss
    *coefficient* (0.03125) is matched.
  * the BatchTopK inference threshold uses topk_threshold_lr=0.001 (= 1 - 0.999).
  * SAELens reads activations via TransformerLens hooks (blocks.L.hook_resid_post)
    rather than nnsight on the raw HF module.
Treat the result as a faithful SAELens reimplementation of the SAEBench recipe,
not a bit-identical reproduction.

Use the companion shell_scripts/experiments wrapper (experiments/train_saebench.sh)
to launch gemma on GPU 0 and pythia on GPU 1 simultaneously.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from sae_lens import (
    BatchTopKTrainingSAEConfig,
    LoggingConfig,
    MultiSAETrainingRunner,
    MultiSAETrainingRunnerConfig,
)

# Make the repo-root `architectures` package importable (for the optional --fp8 path).
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
from architectures import BatchTopKFP8TrainingSAEConfig  # noqa: E402  (registers fp8 arch)
from architectures.fp8_formats import get_format  # noqa: E402

# --- SAEBench fixed recipe (identical across both models / every SAE) ---------
BATCH_TOKENS = 2048
CONTEXT_SIZE = 1024
TOTAL_STEPS = 244_140                 # == 500M tokens / 2048, per the configs
TRAINING_TOKENS = TOTAL_STEPS * BATCH_TOKENS  # 500,002,920
LR = 3e-4
LR_WARMUP_STEPS = 1_000
DECAY_START = 195_312                 # last 20% -> decay over the remaining steps
LR_DECAY_STEPS = TOTAL_STEPS - DECAY_START  # 48,828
AUX_LOSS_COEFFICIENT = 0.03125        # auxk_alpha = 1/32
THRESHOLD_BETA = 0.999                # BatchTopK EMA threshold beta
TOPK_THRESHOLD_LR = round(1.0 - THRESHOLD_BETA, 6)  # 0.001
SEED = 0
DATASET = "monology/pile-uncopyrighted"   # "The Pile" per the SAEBench paper

# SAEBench width / sparsity grid.
WIDTH_POWS = [12, 14, 16]             # 4k, 16k, 65k
K_VALUES = [20, 40, 80, 160, 320, 640]

# SAEBench buffer geometry (n_ctxs=244, ctx_len=1024) -> ~122 train batches.
N_BATCHES_IN_BUFFER = 122

# --- Per-model specifics (from the released configs) --------------------------
MODEL_SPECS = {
    "gemma": dict(
        model_name="google/gemma-2-2b",
        layer=12,
        d_in=2304,
        store_batch_size_prompts=4,   # refresh_batch_size in the gemma configs
        default_gpu=0,
    ),
    "pythia": dict(
        model_name="EleutherAI/pythia-160m-deduped",
        layer=8,
        d_in=768,
        store_batch_size_prompts=32,  # refresh_batch_size in the pythia configs
        default_gpu=1,
    ),
}


def build_saes(d_in: int, widths: list[int], ks: list[int], device: str,
               sae_dtype: str, fp8: bool = False, fp8_format: str = "e4m3",
               fp8_backend: str = "hardware", fp8_quantize_grads: bool = False
               ) -> dict[str, BatchTopKTrainingSAEConfig]:
    """One BatchTopK config per (width, k), all sharing the same hook.

    With ``fp8=True`` the SAE is the ``batchtopk_fp8`` variant whose encoder/decoder
    matmuls run in 8-bit float — every *other* hyperparameter is byte-for-byte identical
    to the FP16 recipe, so the only change is the GEMM precision.
    """
    saes: dict[str, BatchTopKTrainingSAEConfig] = {}
    for w in widths:
        for k in ks:
            name = f"w{w}_k{k}"
            common = dict(
                d_in=d_in,
                d_sae=w,
                k=k,
                aux_loss_coefficient=AUX_LOSS_COEFFICIENT,
                topk_threshold_lr=TOPK_THRESHOLD_LR,
                # SAELens defaults already match SAEBench/dictionary_learning:
                #   apply_b_dec_to_input=True, normalize_activations="none".
                apply_b_dec_to_input=True,
                normalize_activations="none",
                dtype=sae_dtype,
                device=device,
            )
            if fp8:
                saes[name] = BatchTopKFP8TrainingSAEConfig(
                    fp8_format=fp8_format,
                    fp8_backend=fp8_backend,
                    fp8_quantize_grads=fp8_quantize_grads,
                    **common,
                )
            else:
                saes[name] = BatchTopKTrainingSAEConfig(**common)
    return saes


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", required=True, choices=list(MODEL_SPECS),
                   help="Which SAEBench model suite to train.")
    p.add_argument("--gpu", type=int, default=None,
                   help="CUDA device index (default: gemma->0, pythia->1).")
    p.add_argument("--output-dir", type=Path,
                   default=Path(__file__).resolve().parent / "results",
                   help="Root dir for the run (a saebench_<model>/ subdir is created).")
    p.add_argument("--dataset", default=DATASET, help="HF dataset path (activation source).")
    p.add_argument("--dtype", default="bfloat16",
                   choices=["bfloat16", "float16", "float32"],
                   help="Activation/buffer dtype.")
    p.add_argument("--sae-dtype", default="float32",
                   choices=["bfloat16", "float16", "float32"],
                   help="SAE weight dtype. float32 = robust recipe (autocast matmuls "
                        "in bf16 + fp32 master + GradScaler). Set bfloat16 to ~halve "
                        "SAE memory if the 18-SAE grid OOMs (notably gemma 65k x6).")
    g = p.add_argument_group("FP8 training (optional)")
    g.add_argument("--fp8", action="store_true", default=False,
                   help="Train the batchtopk_fp8 SAE: encoder/decoder matmuls in 8-bit "
                        "float, every other hyperparameter identical to the FP16 recipe. "
                        "Forces autocast OFF and --dtype == --sae-dtype (fp8 manages "
                        "precision in the GEMM; default both float32 = fp32 master).")
    g.add_argument("--fp8-format", default="e4m3",
                   help="fp8 layout for the matmuls (e4m3/e5m2 hardware-native). Default e4m3.")
    g.add_argument("--fp8-backend", default="hardware",
                   choices=["hardware", "emulated", "auto"],
                   help="hardware: real torch._scaled_mm fp8 GEMM (E4M3/E5M2). "
                        "emulated: software fake-quant (any format). Default hardware.")
    g.add_argument("--fp8-quantize-grads", action="store_true", default=False,
                   help="Also quantize gradients to fp8 (approximate fully-fp8 training).")
    p.add_argument("--widths", default=None,
                   help="Comma list of widths to train (default: 4096,16384,65536). "
                        "Drop the 65k width here if you OOM.")
    p.add_argument("--ks", default=None,
                   help="Comma list of k values (default: 20,40,80,160,320,640).")
    p.add_argument("--training-tokens", type=int, default=TRAINING_TOKENS,
                   help="Override the token budget (default: SAEBench's 500M).")
    p.add_argument("--n-checkpoints", type=int, default=0,
                   help="Intermediate checkpoints per SAE (0 = none; final is still saved).")
    p.add_argument("--no-save-final", action="store_true",
                   help="Skip saving the final SAEs (use for a throughput/dry test).")
    p.add_argument("--wandb-project", default=None,
                   help="W&B project (default: saebench-repro-<model>).")
    p.add_argument("--no-wandb", action="store_true", help="Disable W&B logging.")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the planned run + every SAE config, then exit (no training).")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    spec = MODEL_SPECS[args.model]

    gpu = args.gpu if args.gpu is not None else spec["default_gpu"]
    device = f"cuda:{gpu}"
    widths = [int(x) for x in args.widths.split(",")] if args.widths else [2 ** p for p in WIDTH_POWS]
    ks = [int(x) for x in args.ks.split(",")] if args.ks else list(K_VALUES)

    if args.fp8:
        # fp8 manages precision explicitly inside the GEMM (no SAELens autocast), so the
        # activation and master dtypes must match (else the bf16-acts/fp32-weights MSE
        # crashes). Coerce --dtype to the master --sae-dtype (default float32 = fp32
        # master, same as the FP16 baseline). Pass --sae-dtype bfloat16 for a bf16 master.
        get_format(args.fp8_format)  # validate the format string early
        if args.dtype != args.sae_dtype:
            print(f"[fp8] coercing --dtype {args.dtype} -> {args.sae_dtype} "
                  "(fp8 disables autocast; activation/master dtypes must match).")
            args.dtype = args.sae_dtype

    hook_name = f"blocks.{spec['layer']}.hook_resid_post"
    saes = build_saes(spec["d_in"], widths, ks, device, args.sae_dtype,
                      fp8=args.fp8, fp8_format=args.fp8_format,
                      fp8_backend=args.fp8_backend,
                      fp8_quantize_grads=args.fp8_quantize_grads)
    hook_names = {name: hook_name for name in saes}

    # Separate output tree for fp8 so it never clobbers the FP16 baseline results.
    run_suffix = "_fp8" if args.fp8 else ""
    run_dir = args.output_dir / f"saebench_{args.model}{run_suffix}"
    run_dir.mkdir(parents=True, exist_ok=True)
    training_steps = args.training_tokens // BATCH_TOKENS

    print("=" * 70)
    print(f"  SAEBench BatchTopK replication — {args.model} ({spec['model_name']})")
    print(f"  hook        {hook_name}  (d_in={spec['d_in']})")
    print(f"  widths      {widths}")
    print(f"  k values    {ks}")
    print(f"  -> {len(saes)} SAEs, shared model + one forward pass per batch")
    print(f"  tokens      {args.training_tokens:,}  ({training_steps:,} steps)")
    print(f"  lr          {LR}  warmup={LR_WARMUP_STEPS}  decay_steps={LR_DECAY_STEPS} (last 20%)")
    print(f"  aux_coeff   {AUX_LOSS_COEFFICIENT}  topk_threshold_lr={TOPK_THRESHOLD_LR}")
    print(f"  dataset     {args.dataset}")
    print(f"  batch/ctx   {BATCH_TOKENS} tokens / {CONTEXT_SIZE} ctx   seed={SEED}")
    prec = (f"FP8 ({args.fp8_format}, backend={args.fp8_backend}, "
            f"quant_grads={args.fp8_quantize_grads})" if args.fp8 else "FP16/bf16")
    print(f"  precision   {prec}")
    print(f"  device      {device}   acts={args.dtype}  sae={args.sae_dtype}")
    print(f"  output      {run_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[dry-run] planned SAEs:")
        for name, cfg in saes.items():
            fp8_info = (f" fp8={cfg.fp8_format}/{cfg.fp8_backend}"
                        if isinstance(cfg, BatchTopKFP8TrainingSAEConfig) else "")
            print(f"  {name:14s} d_in={cfg.d_in} d_sae={cfg.d_sae} k={cfg.k} "
                  f"aux={cfg.aux_loss_coefficient} thr_lr={cfg.topk_threshold_lr} "
                  f"norm={cfg.normalize_activations} b_dec={cfg.apply_b_dec_to_input}"
                  f"{fp8_info} arch={cfg.architecture()}")
        print("\n[dry-run] no training performed.")
        return

    wandb_project = args.wandb_project or f"saebench-repro-{args.model}{run_suffix}"
    logger_cfg = LoggingConfig(
        log_to_wandb=not args.no_wandb,
        wandb_project=wandb_project,
        run_name=f"saebench_{args.model}{run_suffix}_batchtopk_allwidths_allk",
    )

    # FP8 manages precision explicitly inside the GEMM -> no SAELens autocast/GradScaler.
    # FP16: float32 SAE weights keep the AMP GradScaler path (autocast matmuls in bf16);
    # native bf16 weights disable it (no bf16 kernel for unscale).
    sae_autocast = (not args.fp8) and (args.sae_dtype != "bfloat16")

    cfg = MultiSAETrainingRunnerConfig(
        saes=saes,
        hook_names=hook_names,
        model_name=spec["model_name"],
        dataset_path=args.dataset,
        streaming=True,
        context_size=CONTEXT_SIZE,
        lr=LR,
        adam_beta1=0.9,
        adam_beta2=0.999,
        lr_scheduler_name="constant",      # constant + linear warmup + end decay
        lr_warm_up_steps=LR_WARMUP_STEPS,
        lr_decay_steps=LR_DECAY_STEPS,
        train_batch_size_tokens=BATCH_TOKENS,
        n_batches_in_buffer=N_BATCHES_IN_BUFFER,
        training_tokens=args.training_tokens,
        store_batch_size_prompts=spec["store_batch_size_prompts"],
        device=device,
        llm_device=device,
        seed=SEED,
        dtype=args.dtype,
        autocast=sae_autocast,
        autocast_lm=True,
        n_checkpoints=args.n_checkpoints,
        checkpoint_path=str(run_dir / "checkpoints"),
        output_path=str(run_dir),
        save_final_checkpoint=not args.no_save_final,
        logger=logger_cfg,
    )

    print(f"\nStarting shared-model run over {len(saes)} SAEs on {device} ...\n")
    MultiSAETrainingRunner(cfg).run()
    print(f"\nDone. Outputs under: {run_dir}")


if __name__ == "__main__":
    main()
