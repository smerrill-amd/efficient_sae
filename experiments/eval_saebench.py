#!/usr/bin/env python3
"""eval_saebench.py — run the SAEBench eval suite on OUR trained SAEs.

This evaluates one SAE *config* from a `train_saebench.py` run with the exact
SAEBench eval settings the authors used, so the numbers are directly comparable
to the published baselines on HuggingFace
(`adamkarvonen/sae_bench_results_0125`) and Neuronpedia.

The defaults reproduce the config you flagged on Neuronpedia
(`gemma-2-2b/12-sae_bench_0125-batch_topk-res-64k__trainer_2`):

    model      gemma-2-2b
    hook       blocks.12.hook_resid_post   (layer 12, d_in 2304)
    width      65,536  (2^16)
    arch       batch_topk
    trainer_2  -> target L0 ~84  -> our member ``w65536_k80``

SAEBench trainer index <-> our (width, k), mapped by achieved L0 in the authors'
core results (65k BatchTopK):

    trainer_0  L0~21   w65536_k20
    trainer_1  L0~42   w65536_k40
    trainer_2  L0~84   w65536_k80     <-- the config you sent
    trainer_3  L0~169  w65536_k160
    trainer_4  L0~339  w65536_k320
    trainer_5  L0~696  w65536_k640

CHECKPOINTS
-----------
``--checkpoints final`` (default) evaluates the final exported SAE. ``all``
evaluates every intermediate checkpoint of that member found under
``<run>/checkpoints/<hash>/<step>/<member>/`` *and* the final one — letting you
watch every metric evolve over training. (The training run must have been
launched with intermediate checkpoints, e.g.
``N_CHECKPOINTS=10 WIDTHS=65536 KS=80 ./train_saebench.sh``; with the default
``N_CHECKPOINTS=0`` only the final checkpoint exists.) You can also pass an
explicit comma list of steps, e.g. ``--checkpoints 2441,24414,final``.

All checkpoints selected are passed to each eval together, so the expensive
per-eval setup (model activations, probe datasets) is shared across them.

EVALS
-----
Default suite (`core,sparse_probing,absorption,scr,tpp`) runs with no external
credentials. Opt into the rest via ``--evals``:
  * ``autointerp``  needs an OpenAI key (OPENAI_API_KEY env or openai_api_key.txt)
  * ``ravel``       very slow (~45 min/SAE)
  * ``unlearning``  needs the gated WMDP bio-forget corpus + gemma-2-2b-it
Use ``--evals all`` for everything the environment supports.

Outputs land under ``--output-dir`` as the standard SAEBench tree
``eval_results/<eval_type>/<sae_name>_custom_sae_eval_results.json`` where
``<sae_name>`` encodes the member and training step, e.g.
``mysae_w65536_k80_step_499998720``. The companion notebook
(`notebooks/saebench_compare.ipynb`) reads this tree and compares to the authors.
"""

from __future__ import annotations

import argparse
import gc
import glob
import os
import re
import sys
from pathlib import Path

# Authoritative trainer<->k mapping for the 65k BatchTopK suite (by achieved L0).
TRAINER_TO_K = {0: 20, 1: 40, 2: 80, 3: 160, 4: 320, 5: 640}
DEFAULT_EVALS = ["core", "sparse_probing", "absorption", "scr", "tpp"]
ALL_EVALS = [
    "core", "sparse_probing", "absorption", "scr", "tpp",
    "autointerp", "ravel", "unlearning",
]

# llm batch size / dtype per model (SAEBench MODEL_CONFIGS defaults, tuned for 24GB).
MODEL_DEFAULTS = {
    "gemma-2-2b": dict(batch_size=32, dtype="bfloat16"),
    "pythia-160m-deduped": dict(batch_size=256, dtype="float32"),
    "pythia-70m-deduped": dict(batch_size=512, dtype="float32"),
}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    here = Path(__file__).resolve().parent
    p.add_argument("--run-dir", type=Path, default=here / "results" / "saebench_gemma",
                   help="A train_saebench.py run dir (holds <member>/ and checkpoints/).")
    p.add_argument("--member", default="w65536_k80",
                   help="Which SAE in the run to evaluate (default: w65536_k80 = "
                        "the 65k batch_topk trainer_2 config).")
    p.add_argument("--checkpoints", default="final",
                   help="'final' (default), 'all', or a comma list of steps "
                        "(e.g. 2441,24414,final).")
    p.add_argument("--evals", default=",".join(DEFAULT_EVALS),
                   help=f"Comma list of SAEBench evals, or 'all'. "
                        f"Default: {','.join(DEFAULT_EVALS)}.")
    p.add_argument("--output-dir", type=Path, default=None,
                   help="Where the eval_results/ tree is written "
                        "(default: <run-dir>/saebench_eval/<member>).")
    p.add_argument("--tag", default="mysae",
                   help="Prefix for the SAE identifier in result filenames.")
    p.add_argument("--gpu", type=int, default=0, help="CUDA device index.")
    p.add_argument("--llm-batch-size", type=int, default=None,
                   help="Override the model forward batch size (default: per-model).")
    p.add_argument("--llm-dtype", default=None,
                   help="Override LLM/SAE dtype (default: per-model, bf16 for gemma).")
    p.add_argument("--force-rerun", action="store_true",
                   help="Recompute even if a result JSON already exists.")
    p.add_argument("--save-activations", action="store_true",
                   help="Cache LLM activations to disk (faster reruns, lots of disk).")
    p.add_argument("--dry-run", action="store_true",
                   help="Print the discovered checkpoints + plan, then exit.")
    return p.parse_args()


def _step_of(path: Path) -> int:
    """Training step encoded in a checkpoint path ('final_499998720', '24414')."""
    for part in reversed(path.parts):
        m = re.fullmatch(r"(?:final_)?(\d+)", part)
        if m:
            return int(m.group(1))
    return -1


def discover_checkpoints(run_dir: Path, member: str, which: str) -> list[tuple[int, Path]]:
    """Return [(step, sae_dir)] for the requested checkpoints of ``member``.

    The final exported SAE lives at ``<run>/<member>/`` (already converted to the
    jumprelu inference form). Intermediate checkpoints live at
    ``<run>/checkpoints/<hash>/<step>/<member>/`` and are saved in the raw
    ``batchtopk`` training architecture (converted on load).
    """
    final_dir = run_dir / member
    ckpt_dirs: dict[int, Path] = {}

    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.exists():
        for cfg in glob.glob(str(ckpt_root / "**" / member / "cfg.json"), recursive=True):
            d = Path(cfg).parent
            ckpt_dirs[_step_of(d)] = d

    final_step = -1
    if (final_dir / "cfg.json").exists():
        # Prefer the top-level export's step from the final checkpoint if present.
        final_step = max(ckpt_dirs, default=-1)
        if final_step < 0:
            final_step = _step_of(final_dir)
        ckpt_dirs[final_step] = final_dir  # top-level export wins for the final step

    if not ckpt_dirs:
        raise FileNotFoundError(
            f"No SAE named {member!r} found under {run_dir} "
            f"(looked for <member>/cfg.json and checkpoints/**/<member>/cfg.json)."
        )

    if which == "final":
        step = max(ckpt_dirs)
        return [(step, ckpt_dirs[step])]
    if which == "all":
        return sorted(ckpt_dirs.items())

    wanted = {(-1 if s.strip() == "final" else int(s.strip()))
              for s in which.split(",")}
    if -1 in wanted:
        wanted.discard(-1)
        wanted.add(max(ckpt_dirs))
    missing = wanted - set(ckpt_dirs)
    if missing:
        raise FileNotFoundError(
            f"Requested steps {sorted(missing)} not found for {member}. "
            f"Available: {sorted(ckpt_dirs)}"
        )
    return sorted((s, ckpt_dirs[s]) for s in wanted)


def _ensure_repo_archs_registered() -> None:
    """Register repo-local training architectures (e.g. ``batchtopk_fp8``).

    Intermediate checkpoints of an ``--fp8`` run are saved in the raw
    ``batchtopk_fp8`` training architecture, which only exists once the repo's
    ``architectures`` package is imported. (The final exported SAE is already a
    standard ``jumprelu`` and loads without this.) Idempotent + best-effort.
    """
    repo_root = Path(__file__).resolve().parents[1]
    if str(repo_root) not in sys.path:
        sys.path.insert(0, str(repo_root))
    try:
        import architectures  # noqa: F401  (registers batchtopk_fp8 on import)
    except Exception as e:  # noqa: BLE001
        print(f"  [warn] could not import repo `architectures` ({e}); "
              f"non-standard training archs (e.g. batchtopk_fp8) may fail to load.")


def load_inference_sae(ckpt_dir: Path, cache_dir: Path, device: str, dtype: str):
    """Load an SAE ready for inference.

    Inference-ready architectures (jumprelu/topk/standard) load directly. The raw
    ``batchtopk`` training architecture (intermediate checkpoints) is loaded as a
    TrainingSAE and converted to jumprelu (using the learned threshold); the
    converted SAE is cached so reruns are cheap.
    """
    from sae_lens import SAE
    from sae_lens.saes.sae import TrainingSAE

    try:
        return SAE.load_from_disk(str(ckpt_dir), device=device, dtype=dtype)
    except KeyError:
        pass  # training-only architecture (batchtopk / batchtopk_fp8) -> convert below

    _ensure_repo_archs_registered()  # make batchtopk_fp8 loadable before converting
    cache_dir.mkdir(parents=True, exist_ok=True)
    if not (cache_dir / "cfg.json").exists():
        training_sae = TrainingSAE.load_from_disk(str(ckpt_dir), device=device, dtype=dtype)
        training_sae.save_inference_model(str(cache_dir))
        del training_sae
        gc.collect()
        if __import__("torch").cuda.is_available():
            __import__("torch").cuda.empty_cache()
    return SAE.load_from_disk(str(cache_dir), device=device, dtype=dtype)


def main() -> None:
    args = parse_args()

    # Load secrets (.env) — gemma-2-2b is gated and needs an HF token.
    project_root = Path(__file__).resolve().parents[1]
    env_file = project_root / ".env"
    if env_file.exists():
        for line in env_file.read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))

    eval_types = ALL_EVALS if args.evals.strip() == "all" else \
        [e.strip() for e in args.evals.split(",") if e.strip()]

    run_dir = args.run_dir.resolve()
    out_dir = (args.output_dir or (run_dir / "saebench_eval" / args.member)).resolve()
    ckpts = discover_checkpoints(run_dir, args.member, args.checkpoints)

    print("=" * 70)
    print("  SAEBench evaluation")
    print(f"  run        {run_dir}")
    print(f"  member     {args.member}")
    print(f"  evals      {eval_types}")
    print(f"  checkpoints ({len(ckpts)}):")
    for step, d in ckpts:
        print(f"     step {step:>12}  <- {d}")
    print(f"  output     {out_dir}")
    print("=" * 70)

    if args.dry_run:
        print("\n[dry-run] no evaluation performed.")
        return

    import torch  # noqa: F401 (imported lazily so --dry-run/--help stay fast)
    from sae_bench.custom_saes import run_all_evals_custom_saes as run_all
    from sae_bench.sae_bench_utils import general_utils

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    # Resolve the base model + its eval batch/dtype from the SAE's own config.
    import json
    sample_cfg = json.loads((ckpts[0][1] / "cfg.json").read_text())
    model_name_full = sample_cfg["metadata"]["model_name"]          # e.g. google/gemma-2-2b
    model_name = model_name_full.split("/")[-1]                     # e.g. gemma-2-2b
    defaults = MODEL_DEFAULTS.get(model_name, dict(batch_size=16, dtype="bfloat16"))
    llm_dtype = args.llm_dtype or defaults["dtype"]
    llm_batch_size = args.llm_batch_size or defaults["batch_size"]

    # SAEBench validates model_name against its own MODEL_CONFIGS; register ours.
    run_all.MODEL_CONFIGS.setdefault(
        model_name,
        {"batch_size": llm_batch_size, "dtype": llm_dtype,
         "layers": [sample_cfg["metadata"].get("hook_name", "")], "d_model": sample_cfg["d_in"]},
    )

    torch.set_grad_enabled(False)
    cache_root = out_dir / "inference_saes"

    selected_saes = []
    for step, ckpt_dir in ckpts:
        sae = load_inference_sae(
            ckpt_dir, cache_root / f"{args.member}_step_{step}", device, llm_dtype
        )
        general_utils.load_and_format_sae(args.member, sae, device)
        sae = sae.to(dtype=general_utils.str_to_dtype(llm_dtype))
        sae.cfg.dtype = llm_dtype
        name = f"{args.tag}_{args.member}_step_{step}"
        selected_saes.append((name, sae))
        print(f"  loaded {name:34s} arch="
              f"{sae.cfg.architecture() if callable(sae.cfg.architecture) else sae.cfg.architecture}"
              f"  d_sae={sae.cfg.d_sae}  hook={sae.cfg.metadata.hook_name}")

    api_key = None
    if "autointerp" in eval_types:
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key and Path("openai_api_key.txt").exists():
            api_key = Path("openai_api_key.txt").read_text().strip()

    # The custom-sae wrapper writes to cwd-relative eval_results/<type>/, so run
    # from the output dir to keep all results for this member together.
    out_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(out_dir)

    print(f"\nRunning {eval_types} on {len(selected_saes)} checkpoint(s) "
          f"of {args.member} ({model_name}) ...\n")
    run_all.run_evals(
        model_name=model_name,
        selected_saes=selected_saes,
        llm_batch_size=llm_batch_size,
        llm_dtype=llm_dtype,
        device=device,
        eval_types=eval_types,
        api_key=api_key,
        force_rerun=args.force_rerun,
        save_activations=args.save_activations,
    )
    print(f"\nDone. Results under: {out_dir / 'eval_results'}")


if __name__ == "__main__":
    main()
