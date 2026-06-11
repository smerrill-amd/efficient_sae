#!/usr/bin/env python3
"""compare_features.py — FP8 vs FP16 vs SAEBench feature comparison.

Compares the *learned features* (decoder directions) of three BatchTopK/TopK SAE
families that all live in the SAME residual stream (gemma-2-2b
``blocks.12.hook_resid_post``, width 65,536), so their decoder rows are directly
cosine-comparable:

  * FP16  — our standard BatchTopK k-sweep   (results/saebench_gemma)
  * FP8   — our fp8 BatchTopK k-sweep        (results/saebench_gemma_fp8)
  * SAEBench — the published authors' TopK SAEs
              (canrager/saebench_gemma-2-2b_width-2pow16_date-1109, layer 12)

It produces two analyses, for every k in our k-sweep:

  PART A — "same concept, same feature?"
    For each SAEBench *sparse-probing* concept class we find the single most
    discriminative SAE feature (top-1) using the EXACT selection SAEBench uses
    (`get_top_k_mean_diff_mask`, k=1: argmax |mean_pos - mean_neg| over features).
    We then cosine-compare each pair of SAEs' top-1 feature for that concept —
    i.e. did FP8 / FP16 / SAEBench independently learn the *same direction* for
    the concept's most useful probing feature?

  PART B — "is every feature mirrored in the other SAE?"
    For every feature in SAE A we find its closest counterpart in SAE B
    (max cosine similarity over all of B's decoder rows), in both directions, for
    each SAE pair. The distribution of those max-cosines measures how completely
    one dictionary is reproduced by another.

k PAIRING
---------
Our k-sweep now uses the SAME grid as the published SAEBench TopK trainers,
k={20,40,80,160,320,640}, so each of our k maps 1:1 to the SAEBench trainer with
the identical k (trainer 0..5 = k 20,40,80,160,320,640; printed + recorded in
every output).

CHECKPOINTS / FAIRNESS
----------------------
FP8 and FP16 are compared at the LATEST training step present in BOTH runs (so
the two precisions are matched); override with --fp16-step / --fp8-step. For the
current 100M-token-BUDGET runs that latest common step IS the final 100M model.
SAEBench SAEs are fully trained.

  PART C — "do the matched features ACT the same on real tokens?" (functional)
    Geometry (Parts A/B) compares decoder *directions*; Part C checks *behaviour*
    on a shared held-out token block, on identical inputs:
      * recon_cos — cosine between two SAEs' reconstructions of the SAME activation
        (every pair); the most direct "they do the same thing" number.
      * feat_corr — per-feature activation correlation for index-aligned features
        i<->i (FP16 vs FP8 only: same seed/data/init -> feature i is the same).
      * active_jaccard — per-token overlap of which features fire (FP16 vs FP8).

  Part B also reports BIJECTIVE hardening (self/mutual-match rate, collision rate,
  and a greedy 1-1 matched-cosine), since the raw nearest-cosine is greedy and can
  overstate similarity by mapping many A-features onto one B-feature.

OUTPUTS  (under --output-dir, default results/feature_comparison/<model>)
  k<k>/topk_features.json      Part A raw: per (sae,dataset,class) top-N feature
                               indices + mean-diff scores.
  k<k>/topk_cosine.json        Part A: matched-concept pairwise cosine per concept.
  k<k>/allfeat_match.npz       Part B: per-feature max-cosine + argmax (+ greedy
                               bijective matched-cosine), both directions, every pair.
  k<k>/allfeat_match_summary.json  Part B summary incl. bijective/collision stats.
  k<k>/functional.npz          Part C: recon_cos (all pairs) + feat_corr/active_jaccard
                               (fp16<->fp8) arrays.
  k<k>/functional_summary.json Part C summary (means/medians/fractions).
  k<k>/meta.json               resolved checkpoints, steps, trainer mapping.
  summary.json                 roll-up across all k (means/medians).

The companion notebook (notebooks/feature_comparison.ipynb) reads this tree.
"""

from __future__ import annotations

import argparse
import gc
import glob
import json
import os
import random
import re
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
if str(REPO / "experiments") not in sys.path:
    sys.path.insert(0, str(REPO / "experiments"))

# Our k-sweep grid (now the official SAEBench grid) and the published SAEBench
# TopK trainer k's (65k width) — identical, so nearest_saebench_trainer() is 1:1.
K_SWEEP = [20, 40, 80, 160, 320, 640]
SAEBENCH_TRAINER_K = {0: 20, 1: 40, 2: 80, 3: 160, 4: 320, 5: 640}

# Legacy: the 100M-token *intermediate* step of the old 500M-schedule runs. The new
# runs are 100M-token-BUDGET (final == the 100M model, step ~99999744), so the default
# step selection is "latest step common to FP16 & FP8" (--target-step 0), which resolves
# to that final. Kept here only as a reference constant.
STEP_100M = 100001792

# Per-model wiring: our run dirs, the LLM hook layer / eval settings, and the
# published SAEBench TopK release (gemma-2-2b 65k layer 12; pythia-160m 65k layer 8).
MODELS = {
    "gemma-2-2b": dict(
        fp16_run="saebench_gemma", fp8_run="saebench_gemma_fp8",
        layer=12, llm_batch_size=32, dtype="bfloat16",
        saebench_repo="canrager/saebench_gemma-2-2b_width-2pow16_date-1109",
        saebench_path="gemma-2-2b_topk_width-2pow16_date-1109/"
                      "resid_post_layer_{layer}/trainer_{t}/ae.pt",
        saebench_model="gemma-2-2b",
    ),
    "pythia-160m-deduped": dict(
        fp16_run="saebench_pythia", fp8_run="saebench_pythia_fp8",
        layer=8, llm_batch_size=256, dtype="float32",
        saebench_repo="adamkarvonen/saebench_pythia-160m-deduped_width-2pow16_date-0108",
        saebench_path="TopK_pythia-160m-deduped__0108/"
                      "resid_post_layer_{layer}/trainer_{t}/ae.pt",
        saebench_model="pythia-160m-deduped",
    ),
}
MODEL_ALIASES = {"gemma": "gemma-2-2b", "pythia": "pythia-160m-deduped"}


# --------------------------------------------------------------------------- #
# env / small helpers
# --------------------------------------------------------------------------- #
def _load_env() -> None:
    env_file = REPO / ".env"
    if not env_file.exists():
        return
    for line in env_file.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            k, v = line.split("=", 1)
            os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _step_of(path: Path) -> int:
    for part in reversed(Path(path).parts):
        m = re.fullmatch(r"(?:final_)?(\d+)", part)
        if m:
            return int(m.group(1))
    return -1


def discover_member_checkpoints(run_dir: Path, member: str) -> dict[int, Path]:
    """{step: sae_dir} for ``member`` (final export + intermediate checkpoints)."""
    ckpts: dict[int, Path] = {}
    ckpt_root = run_dir / "checkpoints"
    if ckpt_root.exists():
        for cfg in glob.glob(str(ckpt_root / "**" / member / "cfg.json"), recursive=True):
            d = Path(cfg).parent
            ckpts[_step_of(d)] = d
    final_dir = run_dir / member
    if (final_dir / "cfg.json").exists():
        step = max(ckpts, default=-1)
        if step < 0:
            step = _step_of(final_dir)
        ckpts[step] = final_dir
    return ckpts


def nearest_saebench_trainer(k: int) -> tuple[int, int]:
    """(trainer_index, trainer_k) whose k is closest to ours."""
    t = min(SAEBENCH_TRAINER_K, key=lambda t: abs(SAEBENCH_TRAINER_K[t] - k))
    return t, SAEBENCH_TRAINER_K[t]


# --------------------------------------------------------------------------- #
# SAE loading (lazy: load -> use -> free)
# --------------------------------------------------------------------------- #
def load_our_sae(ckpt_dir: Path, device: str, dtype: str):
    """Load an FP16/FP8 BatchTopK checkpoint as an inference SAE (jumprelu)."""
    import steering_lib as S

    sae = S.load_inference_sae(ckpt_dir, device=device, dtype=dtype)
    sae.eval()
    return sae


def load_saebench_sae(trainer: int, layer: int, device: str, dtype: str,
                      repo: str, path_tmpl: str, model_name: str):
    """Load a published SAEBench TopK SAE for ``trainer`` (65k width, given layer).

    Inlines sae_bench.custom_saes.topk_sae.load_dictionary_learning_topk_sae but
    loads the state dict with strict=False: the dictionary_learning ``ae.pt`` does
    not store ``k`` (it's a constructor arg / registered buffer), which makes the
    packaged strict loader raise on this sae_bench version.
    """
    from huggingface_hub import hf_hub_download
    from sae_bench.custom_saes.topk_sae import TopKSAE

    td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}[dtype]
    filename = path_tmpl.format(layer=layer, t=trainer)
    local_dir = str(REPO / "experiments" / "downloaded_saes")
    path_params = hf_hub_download(repo, filename, local_dir=local_dir)
    path_cfg = hf_hub_download(repo, filename.replace("ae.pt", "config.json"),
                              local_dir=local_dir)
    pt_params = torch.load(path_params, map_location="cpu")
    config = json.loads(Path(path_cfg).read_text())
    k = config["trainer"]["k"]
    assert layer == config["trainer"]["layer"]

    key_map = {"encoder.weight": "W_enc", "decoder.weight": "W_dec",
               "encoder.bias": "b_enc", "bias": "b_dec"}
    pt_params.pop("threshold", None)
    renamed = {key_map.get(kk, kk): v for kk, v in pt_params.items()}
    renamed["W_enc"] = renamed["W_enc"].T  # nn.Linear stores transposed
    renamed["W_dec"] = renamed["W_dec"].T

    sae = TopKSAE(d_in=renamed["b_dec"].shape[0], d_sae=renamed["b_enc"].shape[0],
                  k=k, model_name=model_name, hook_layer=layer,
                  device=torch.device(device), dtype=td, use_threshold=False)
    sae.load_state_dict(renamed, strict=False)  # 'k' buffer comes from ctor
    sae.to(device=device, dtype=td)
    if hasattr(sae, "cfg"):
        sae.cfg.architecture = "topk"
    sae.eval()
    return sae


def sae_decoder(sae) -> torch.Tensor:
    """Decoder matrix [d_sae, d_in] (row i = feature i's residual-space direction)."""
    return sae.W_dec.detach()


def free(*objs):
    for o in objs:
        del o
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# --------------------------------------------------------------------------- #
# Part A: SAEBench sparse-probing top-1 feature per concept class
# --------------------------------------------------------------------------- #
@torch.no_grad()
def meaned_sae_acts(all_llm_BLD: dict, sae, dtype: torch.dtype, sae_batch_size=125):
    """SAE features meaned over (non-masked) sequence positions, per class.

    Replicates sae_bench.activation_collection.get_sae_meaned_activations but is
    robust to the SAE flavour (our jumprelu SAEs and SAEBench TopKSAE both expose
    .encode). Masked tokens were already zeroed in all_llm_BLD.
    """
    import einops

    out = {}
    for cls, all_acts_BLD in all_llm_BLD.items():
        chunks = []
        for i in range(0, len(all_acts_BLD), sae_batch_size):
            acts_BLD = all_acts_BLD[i:i + sae_batch_size].to(sae.W_dec.device)
            acts_BLF = sae.encode(acts_BLD.to(sae.W_dec.dtype))
            activations_BL = einops.reduce(acts_BLD, "B L D -> B L", "sum")
            nonzero_BL = (activations_BL != 0.0).to(dtype=acts_BLF.dtype)
            nonzero_B = einops.reduce(nonzero_BL, "B L -> B", "sum")
            acts_BLF = acts_BLF * nonzero_BL[:, :, None]
            acts_BF = einops.reduce(acts_BLF, "B L F -> B F", "sum") / nonzero_B[:, None]
            chunks.append(acts_BF.to(dtype).cpu())
        out[cls] = torch.cat(chunks, dim=0)
    return out


def top_features_for_concept(sae_BF: dict, cls: str, top_n: int, seed: int):
    """Top-``top_n`` features for concept ``cls`` by |mean_pos - mean_neg|.

    Uses sae_bench's exact balanced positive/negative construction + selection
    criterion (so the top-1 matches what sparse probing would pick).
    Returns (indices[list], scores[list]) sorted by score desc.
    """
    import sae_bench.evals.sparse_probing.probe_training as pt

    torch.manual_seed(seed)
    acts, labels = pt.prepare_probe_data(sae_BF, cls)  # CPU tensors
    import sae_bench.sae_bench_utils.dataset_info as di

    pos = acts[labels == di.POSITIVE_CLASS_LABEL].mean(dim=0)
    neg = acts[labels == di.NEGATIVE_CLASS_LABEL].mean(dim=0)
    diff = (pos - neg).abs()
    n = min(top_n, diff.numel())
    scores, idx = torch.topk(diff, n)
    return idx.tolist(), scores.float().tolist()


def run_part_a(model, layer, hook_name, sae_specs, datasets, cfg, device, seed):
    """Compute top-N probing features per concept for every UNIQUE SAE.

    The model forward happens ONCE per dataset (the residual activations are shared
    across all SAEs / all k), so cost is independent of the number of k's.

    sae_specs: dict sae_id -> callable() -> sae (lazy loader).
    Returns:
      top_features: {sae_id: {dataset: {cls: {"idx": [...], "score": [...]}}}}
      classes_per_dataset: {dataset: [class, ...]}
      decoders: {sae_id: W_dec [d_sae, d_in] on CPU}  (grabbed while each SAE is
                already loaded, so no SAE is loaded twice).
    """
    import sae_bench.sae_bench_utils.activation_collection as ac
    import sae_bench.sae_bench_utils.dataset_info as di
    import sae_bench.sae_bench_utils.dataset_utils as du

    results = {sid: {} for sid in sae_specs}
    classes_per_dataset = {}

    # Load every SAE ONCE and keep it resident (≈10 GB for 17 65k SAEs — trivial on
    # a 192 GB MI300X), so we never re-read 600 MB safetensors per dataset.
    print(f"\n[Part A] loading {len(sae_specs)} SAEs (resident across all datasets) ...")
    saes = {}
    decoders: dict[str, torch.Tensor] = {}
    for sid, loader in sae_specs.items():
        saes[sid] = loader()
        decoders[sid] = sae_decoder(saes[sid]).cpu()
        print(f"    loaded {sid}")

    for dataset_name in datasets:
        print(f"\n[Part A] dataset {dataset_name}")
        train_data, _ = du.get_multi_label_train_test_data(
            dataset_name, cfg["probe_train_set_size"], cfg["probe_test_set_size"], seed
        )
        chosen = di.chosen_classes_per_dataset[dataset_name]
        train_data = du.filter_dataset(train_data, chosen)
        train_data = du.tokenize_data_dictionary(
            train_data, model.tokenizer, cfg["context_length"], device
        )
        classes_per_dataset[dataset_name] = list(train_data.keys())

        # Model forward ONCE per dataset (shared across all SAEs). Keep on GPU.
        all_llm_BLD = ac.get_all_llm_activations(
            train_data, model, cfg["llm_batch_size"], layer, hook_name,
            mask_bos_pad_eos_tokens=True,
        )

        for sid, sae in saes.items():
            sae_BF = meaned_sae_acts(all_llm_BLD, sae, torch.float32,
                                     cfg["sae_batch_size"])
            ds_res = {}
            for cls in train_data.keys():
                idx, score = top_features_for_concept(sae_BF, cls, cfg["top_n"], seed)
                ds_res[cls] = {"idx": idx, "score": score}
            results[sid][dataset_name] = ds_res
            print(f"    {sid:14s} top-1/class: "
                  + ", ".join(f"{c}:{ds_res[c]['idx'][0]}" for c in list(ds_res)[:6])
                  + (" ..." if len(ds_res) > 6 else ""))
            free(sae_BF)

        free(all_llm_BLD, train_data)

    free(*saes.values())
    saes.clear()
    return results, classes_per_dataset, decoders


def cosine(u: torch.Tensor, v: torch.Tensor) -> float:
    u = u.float()
    v = v.float()
    return float(torch.dot(u, v) / (u.norm() * v.norm() + 1e-12))


@torch.no_grad()
def _best_match(vec: torch.Tensor, mat: torch.Tensor) -> tuple[float, int]:
    """Max cosine of ``vec`` ([d_in]) against every row of ``mat`` ([N, d_in])."""
    v = torch.nn.functional.normalize(vec.float(), dim=0)
    m = torch.nn.functional.normalize(mat.float(), dim=1)
    sims = m @ v
    val, idx = sims.max(0)
    return float(val), int(idx)


def part_a_cosines(top_features, classes_per_dataset, decoders):
    """Per concept: how aligned are the SAEs' top-1 probing features?

    For each concept (dataset, class) and each pair of SAEs we record:
      * cos / abscos  — cosine of the two SAEs' OWN top-1 features (the headline).
      * bestmatch_cos — for each direction A->B, the max cosine of A's top-1
        feature against ALL of B's features (and the matched index). This tells us
        whether a low top1-vs-top1 cosine means "the concept direction is absent in
        B" (low best-match too) or merely "it isn't B's #1 probing feature"
        (high best-match).

    decoders: dict name -> W_dec [d_sae, d_in] tensor.
    Returns a list of per-concept dicts.
    """
    names = list(top_features)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    out = []
    for dataset_name, classes in classes_per_dataset.items():
        for cls in classes:
            top1 = {n: top_features[n][dataset_name][cls]["idx"][0] for n in names}
            score = {n: top_features[n][dataset_name][cls]["score"][0] for n in names}
            cos, abscos, best = {}, {}, {}
            for a, b in pairs:
                va = decoders[a][top1[a]]
                vb = decoders[b][top1[b]]
                c = cosine(va, vb)
                cos[f"{a}-{b}"] = c
                abscos[f"{a}-{b}"] = abs(c)
                bv_ab, bi_ab = _best_match(va, decoders[b])
                bv_ba, bi_ba = _best_match(vb, decoders[a])
                best[f"{a}->{b}"] = {"cos": bv_ab, "match_idx": bi_ab}
                best[f"{b}->{a}"] = {"cos": bv_ba, "match_idx": bi_ba}
            out.append({
                "dataset": dataset_name, "class": cls,
                "top1": top1, "meandiff": score,
                "cos": cos, "abscos": abscos, "bestmatch": best,
            })
    return out


# --------------------------------------------------------------------------- #
# Part B: all-feature nearest-neighbour cosine across SAE pairs
# --------------------------------------------------------------------------- #
@torch.no_grad()
def nearest_cosine(A: torch.Tensor, B: torch.Tensor, chunk=8192):
    """For each row of A, max cosine (+argmax) over rows of B. Both [N, d_in].

    Returns (maxcos[N_A], argmax[N_A]) as numpy arrays.
    """
    An = torch.nn.functional.normalize(A.float(), dim=1)
    Bn = torch.nn.functional.normalize(B.float(), dim=1)
    n = An.shape[0]
    maxcos = torch.empty(n, device=A.device)
    argmax = torch.empty(n, dtype=torch.long, device=A.device)
    for i in range(0, n, chunk):
        sims = An[i:i + chunk] @ Bn.t()  # [chunk, N_B]
        v, a = sims.max(dim=1)
        maxcos[i:i + chunk] = v
        argmax[i:i + chunk] = a
    return maxcos.cpu().numpy(), argmax.cpu().numpy().astype(np.int32)


def bijective_stats(a2b_v, a2b_i, b2a_v, b2a_i):
    """Harden the *greedy* nearest-cosine of Part B into a bijective story.

    The greedy max-cosine lets many A-features map to the SAME B-feature, which
    OVERSTATES similarity. These diagnostics quantify how far greedy is from a
    true 1-1 (bijective) matching, and give a permutation-honest matched cosine:

      * self_match_rate — fraction with argmax(i)==i. Meaningful only when A and B
        are the SAME architecture trained from the SAME seed/data (our FP8 vs FP16):
        index i should be the *same* feature, so a high rate is direct evidence the
        precisions learned an index-aligned dictionary (no permutation at all).
      * mutual_rate — fraction of mutual nearest neighbours (A->B->A returns i).
        Permutation-invariant; the honest "these are really paired" number.
      * collision_rate / max_multiplicity — how much greedy doubles up on B. Low
        collisions ⇒ greedy already ≈ bijective, so the Part B plots are sound.
      * greedy_bijective_* — a true 1-1 assignment built from both directions'
        top-1 candidate edges (sort by cosine, assign unique→unique). Its matched-
        cosine mean/median/frac>0.9 is the permutation-honest dictionary overlap.
    """
    n_a, n_b = len(a2b_i), len(b2a_i)
    same_shape = (n_a == n_b)
    self_match = float((a2b_i == np.arange(n_a)).mean()) if same_shape else float("nan")
    mutual = float((b2a_i[a2b_i] == np.arange(n_a)).mean())
    counts = np.bincount(a2b_i, minlength=n_b)
    collision_rate = float(1.0 - (counts > 0).sum() / n_a)  # A-features sharing a B target
    max_mult = int(counts.max()) if counts.size else 0

    # Greedy bijective from the union of both directions' top-1 edges.
    edges = np.concatenate([
        np.stack([a2b_v, np.arange(n_a), a2b_i.astype(np.int64)], axis=1),
        np.stack([b2a_v, b2a_i.astype(np.int64), np.arange(n_b)], axis=1),
    ], axis=0)
    order = np.argsort(-edges[:, 0])  # highest cosine first
    used_a = np.zeros(n_a, dtype=bool)
    used_b = np.zeros(n_b, dtype=bool)
    matched = []
    for e in order:
        c, ai, bi = edges[e]
        ai, bi = int(ai), int(bi)
        if not used_a[ai] and not used_b[bi]:
            used_a[ai] = used_b[bi] = True
            matched.append(c)
    matched = np.asarray(matched, dtype=np.float32)
    cov = float(len(matched) / min(n_a, n_b)) if min(n_a, n_b) else 0.0
    return {
        "self_match_rate": self_match,
        "mutual_rate": mutual,
        "collision_rate": collision_rate,
        "max_multiplicity": max_mult,
        "greedy_bijective_coverage": cov,
        "greedy_bijective_mean": float(matched.mean()) if matched.size else float("nan"),
        "greedy_bijective_median": float(np.median(matched)) if matched.size else float("nan"),
        "greedy_bijective_frac_gt_0.9": float((matched > 0.9).mean()) if matched.size else float("nan"),
    }, matched


def greedy_bijective_pairs(a2b_v, a2b_i, b2a_v, b2a_i):
    """1-1 assignment A<->B from both directions' top-1 edges (sorted by cosine).

    Returns (a_idx, b_idx, cos) arrays of the matched pairs. FP8 and FP16 are NOT
    index-aligned (different precision perturbs the optimization, so feature i in
    one is feature perm(i) in the other) — every functional comparison of matched
    features must go through this permutation, not i<->i.
    """
    n_a, n_b = len(a2b_i), len(b2a_i)
    edges = np.concatenate([
        np.stack([a2b_v, np.arange(n_a), a2b_i.astype(np.int64)], axis=1),
        np.stack([b2a_v, b2a_i.astype(np.int64), np.arange(n_b)], axis=1),
    ], axis=0)
    order = np.argsort(-edges[:, 0])
    used_a = np.zeros(n_a, dtype=bool)
    used_b = np.zeros(n_b, dtype=bool)
    ai_out, bi_out, cos_out = [], [], []
    for e in order:
        c, ai, bi = edges[e]
        ai, bi = int(ai), int(bi)
        if not used_a[ai] and not used_b[bi]:
            used_a[ai] = used_b[bi] = True
            ai_out.append(ai); bi_out.append(bi); cos_out.append(c)
    return (np.asarray(ai_out, dtype=np.int64),
            np.asarray(bi_out, dtype=np.int64),
            np.asarray(cos_out, dtype=np.float32))


def run_part_b(decoders: dict, device: str):
    """Both-direction nearest-cosine for every unordered SAE pair (+ bijective stats)."""
    names = list(decoders)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    out = {}
    summary = {}
    for a, b in pairs:
        A = decoders[a].to(device)
        B = decoders[b].to(device)
        a2b_v, a2b_i = nearest_cosine(A, B)
        b2a_v, b2a_i = nearest_cosine(B, A)
        key = f"{a}__{b}"
        out[f"{key}__A2B_maxcos"] = a2b_v
        out[f"{key}__A2B_argmax"] = a2b_i
        out[f"{key}__B2A_maxcos"] = b2a_v
        out[f"{key}__B2A_argmax"] = b2a_i
        bij, bij_matched = bijective_stats(a2b_v, a2b_i, b2a_v, b2a_i)
        out[f"{key}__bijective_matchcos"] = bij_matched
        summary[key] = {
            "direction": f"{a}->{b} (A2B), {b}->{a} (B2A)",
            "A2B_mean": float(a2b_v.mean()), "A2B_median": float(np.median(a2b_v)),
            "B2A_mean": float(b2a_v.mean()), "B2A_median": float(np.median(b2a_v)),
            "A2B_frac_gt_0.9": float((a2b_v > 0.9).mean()),
            "B2A_frac_gt_0.9": float((b2a_v > 0.9).mean()),
            **bij,
        }
        print(f"    {a:6s} <-> {b:10s}  "
              f"{a}->{b} mean={summary[key]['A2B_mean']:.3f} "
              f"med={summary[key]['A2B_median']:.3f} | "
              f"{b}->{a} mean={summary[key]['B2A_mean']:.3f} | "
              f"mutual={bij['mutual_rate']:.3f} self={bij['self_match_rate']:.3f} "
              f"bij_med={bij['greedy_bijective_median']:.3f}")
    return out, summary


# --------------------------------------------------------------------------- #
# Part C: functional equivalence (do the features ACT the same on real tokens?)
# --------------------------------------------------------------------------- #
@torch.no_grad()
def collect_holdout_acts(model, layer, hook_name, cfg, datasets, device,
                         max_tokens: int, seed: int):
    """A held-out [N, d_in] block of residual activations (real tokens).

    Reuses sae_bench's activation collection on the first probing dataset (text
    only — labels are irrelevant here), flattens all non-masked positions, and
    subsamples to ``max_tokens``. These same activations are fed to every SAE so
    the functional comparison is on identical inputs.
    """
    import sae_bench.sae_bench_utils.activation_collection as ac
    import sae_bench.sae_bench_utils.dataset_info as di
    import sae_bench.sae_bench_utils.dataset_utils as du

    dataset_name = datasets[0]
    train_data, _ = du.get_multi_label_train_test_data(
        dataset_name, cfg["probe_train_set_size"], cfg["probe_test_set_size"], seed
    )
    chosen = di.chosen_classes_per_dataset[dataset_name]
    train_data = du.filter_dataset(train_data, chosen)
    train_data = du.tokenize_data_dictionary(
        train_data, model.tokenizer, cfg["context_length"], device
    )
    all_llm_BLD = ac.get_all_llm_activations(
        train_data, model, cfg["llm_batch_size"], layer, hook_name,
        mask_bos_pad_eos_tokens=True,
    )
    flat = []
    for acts_BLD in all_llm_BLD.values():
        D = acts_BLD.shape[-1]
        x = acts_BLD.reshape(-1, D)
        x = x[x.abs().sum(-1) > 0]  # drop masked (zeroed) positions
        flat.append(x.cpu())
    free(all_llm_BLD, train_data)
    acts = torch.cat(flat, dim=0)
    if acts.shape[0] > max_tokens:
        g = torch.Generator().manual_seed(seed)
        sel = torch.randperm(acts.shape[0], generator=g)[:max_tokens]
        acts = acts[sel]
    return acts  # [N, d_in] on CPU, float32-ish


def _reconstruct(sae, feats):
    """Reconstruction from feature activations, robust to SAE flavour."""
    if hasattr(sae, "decode"):
        return sae.decode(feats)
    rec = feats @ sae.W_dec
    b_dec = getattr(sae, "b_dec", None)
    if b_dec is not None:
        rec = rec + b_dec
    return rec


@torch.no_grad()
def encode_decode(sae, acts_ND, device, chunk=2048):
    """Return (feats [N,d_sae] float32 CPU, recon [N,d_in] float32 CPU)."""
    dt = sae.W_dec.dtype
    feats_chunks, rec_chunks = [], []
    for i in range(0, acts_ND.shape[0], chunk):
        x = acts_ND[i:i + chunk].to(device=device, dtype=dt)
        f = sae.encode(x)
        r = _reconstruct(sae, f)
        feats_chunks.append(f.float().cpu())
        rec_chunks.append(r.float().cpu())
    return torch.cat(feats_chunks), torch.cat(rec_chunks)


def _row_cosine(A, B):
    """Per-row cosine between two [N, D] tensors -> [N] numpy."""
    a = torch.nn.functional.normalize(A.float(), dim=1)
    b = torch.nn.functional.normalize(B.float(), dim=1)
    return (a * b).sum(1).numpy()


def _matched_feature_corr(FA, FB, a_idx, b_idx, min_active=8):
    """Pearson corr across tokens for bijectively-MATCHED features a_idx<->b_idx.

    FA, FB: [N, d_sae] feature activations. a_idx/b_idx: paired feature indices
    (FP8 feature a_idx[m] is matched to FP16 feature b_idx[m]). Returns corr per
    matched pair (nan where either feature is ~never active) and the active mask.
    FP8/FP16 are permuted, so matching (not i<->i) is mandatory.
    """
    A = FA[:, a_idx].float()
    B = FB[:, b_idx].float()
    Ac = A - A.mean(0, keepdim=True)
    Bc = B - B.mean(0, keepdim=True)
    num = (Ac * Bc).sum(0)
    den = Ac.norm(dim=0) * Bc.norm(dim=0) + 1e-12
    corr = (num / den).numpy()
    active = (((A != 0).sum(0).numpy() >= min_active)
              & ((B != 0).sum(0).numpy() >= min_active))
    corr[~active] = np.nan
    return corr, active


def _matched_active_jaccard(FA, FB, a_idx, b_idx):
    """Per-token Jaccard of active sets, restricted to MATCHED features.

    Remaps both SAEs' active patterns onto the shared matched-feature axis
    (position m = the pair (a_idx[m], b_idx[m])) and measures whether the same
    matched feature fires on each token in both SAEs.
    """
    Aon = (FA[:, a_idx] != 0)
    Bon = (FB[:, b_idx] != 0)
    inter = (Aon & Bon).sum(1).float()
    union = (Aon | Bon).sum(1).float().clamp(min=1)
    return (inter / union).numpy()


def run_part_c(sae_specs, roles, acts_ND, device, chunk, matched_pair):
    """Functional equivalence for one k's roles on shared held-out activations.

    On identical inputs:
      * recon_cos (every pair) — cosine between the two SAEs' reconstructions of
        the SAME activation. Permutation-invariant; the most direct "same function"
        number.
      * feat_corr + active_jaccard (matched_pair, e.g. fp16<->fp8) — per-feature
        activation correlation and per-token active-set Jaccard over the BIJECTIVE
        feature matching (FP8/FP16 features are permuted, so we match via decoder
        nearest-cosine, not i<->i).
    """
    feats, recons, decs = {}, {}, {}
    for role, sid in roles.items():
        sae = sae_specs[sid]()
        f, r = encode_decode(sae, acts_ND, device, chunk)
        feats[role], recons[role] = f, r
        decs[role] = sae_decoder(sae).cpu()
        free(sae)

    names = list(roles)
    pairs = [(a, b) for i, a in enumerate(names) for b in names[i + 1:]]
    arrays, summary = {}, {}
    for a, b in pairs:
        key = f"{a}__{b}"
        rc = _row_cosine(recons[a], recons[b])
        arrays[f"{key}__recon_cos"] = rc
        summary[key] = {
            "recon_cos_mean": float(rc.mean()),
            "recon_cos_median": float(np.median(rc)),
            "recon_cos_frac_gt_0.99": float((rc > 0.99).mean()),
        }
        if {a, b} == set(matched_pair):
            # Bijective feature matching from decoder directions (a<->b).
            A = decs[a].to(device); B = decs[b].to(device)
            a2b_v, a2b_i = nearest_cosine(A, B)
            b2a_v, b2a_i = nearest_cosine(B, A)
            ai, bi, mcos = greedy_bijective_pairs(a2b_v, a2b_i, b2a_v, b2a_i)
            free(A, B)
            corr, active = _matched_feature_corr(feats[a], feats[b], ai, bi)
            jac = _matched_active_jaccard(feats[a], feats[b], ai, bi)
            arrays[f"{key}__matched_feat_corr"] = corr
            arrays[f"{key}__matched_cos"] = mcos
            arrays[f"{key}__active_jaccard"] = jac
            cval = corr[~np.isnan(corr)]
            summary[key].update({
                "n_matched": int(len(ai)),
                "feat_corr_mean": float(cval.mean()) if cval.size else float("nan"),
                "feat_corr_median": float(np.median(cval)) if cval.size else float("nan"),
                "feat_corr_frac_gt_0.9": float((cval > 0.9).mean()) if cval.size else float("nan"),
                "n_features_active": int(active.sum()),
                "active_jaccard_mean": float(jac.mean()),
                "active_jaccard_median": float(np.median(jac)),
            })
        print(f"    {a:6s} <-> {b:10s}  recon_cos mean={summary[key]['recon_cos_mean']:.4f} "
              f"med={summary[key]['recon_cos_median']:.4f}"
              + (f" | feat_corr med={summary[key].get('feat_corr_median', float('nan')):.3f}"
                 f" jaccard med={summary[key].get('active_jaccard_median', float('nan')):.3f}"
                 if {a, b} == set(matched_pair) else ""))
        free(rc)
    free(*feats.values(), *recons.values(), *decs.values())
    return arrays, summary


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--model", default="gemma",
                   help="gemma | pythia (or a full name in MODELS).")
    p.add_argument("--width", type=int, default=65536)
    p.add_argument("--ks", default=",".join(map(str, K_SWEEP)),
                   help="Comma list of our k-sweep k's to compare.")
    p.add_argument("--fp16-run", type=Path, default=None,
                   help="Override FP16 run dir (default: per-model in MODELS).")
    p.add_argument("--fp8-run", type=Path, default=None,
                   help="Override FP8 run dir (default: per-model in MODELS).")
    p.add_argument("--target-step", type=int, default=0,
                   help="Pin BOTH precisions to this checkpoint step. Default 0 = "
                        "'latest step common to FP16 & FP8', which for the 100M-budget "
                        f"runs is the final 100M model. Pass {STEP_100M} to target the "
                        "100M *intermediate* step of an old 500M-schedule run instead.")
    p.add_argument("--fp16-step", type=int, default=None,
                   help="Pin FP16 step explicitly (overrides --target-step).")
    p.add_argument("--fp8-step", type=int, default=None,
                   help="Pin FP8 step explicitly (overrides --target-step).")
    p.add_argument("--no-saebench", action="store_true",
                   help="Skip the published SAEBench SAEs (FP8 vs FP16 only).")
    p.add_argument("--parts", default="all", choices=["a", "b", "c", "all"],
                   help="a=probing top-1, b=decoder nearest-cosine (+bijective), "
                        "c=functional equivalence on real tokens, all=everything.")
    p.add_argument("--func-tokens", type=int, default=4096,
                   help="Part C: # held-out token activations for the functional "
                        "comparison (recon cosine / feature corr / active-set Jaccard).")
    p.add_argument("--datasets", default=None,
                   help="Comma list to override the sparse-probing datasets "
                        "(default: all 8). Use a subset for a quick smoke test.")
    p.add_argument("--top-n", type=int, default=20,
                   help="How many top features to record per concept (top-1 is used "
                        "for the headline cosine; more enables top-N analysis).")
    p.add_argument("--probe-train-set-size", type=int, default=4000)
    p.add_argument("--sae-batch-size", type=int, default=512,
                   help="SAE encode batch (only affects speed, not results; SAEBench "
                        "uses 125 — larger is much faster on big GPUs).")
    p.add_argument("--llm-batch-size", type=int, default=None,
                   help="LLM forward batch (default: per-model). Larger = faster "
                        "activation collection on big GPUs.")
    p.add_argument("--output-dir", type=Path, default=None)
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--dtype", default=None)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--force", action="store_true", help="Recompute existing outputs.")
    p.add_argument("--dry-run", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    _load_env()

    model_name = MODEL_ALIASES.get(args.model, args.model)
    if model_name not in MODELS:
        raise SystemExit(f"Unknown model {args.model!r}; choose from "
                         f"{list(MODEL_ALIASES)} or {list(MODELS)}.")
    md = MODELS[model_name]
    dtype = args.dtype or md["dtype"]
    layer = md["layer"]
    hook_name = f"blocks.{layer}.hook_resid_post"
    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    results_root = REPO / "experiments" / "results"
    fp16_run = args.fp16_run or results_root / md["fp16_run"]
    fp8_run = args.fp8_run or results_root / md["fp8_run"]
    out_root = (args.output_dir
                or results_root / "feature_comparison" / model_name)
    out_root.mkdir(parents=True, exist_ok=True)

    # sparse-probing config (SAEBench defaults; train-set size overridable for speed).
    from sae_bench.evals.sparse_probing.eval_config import SparseProbingEvalConfig
    spc = SparseProbingEvalConfig(model_name=model_name)
    datasets = ([d.strip() for d in args.datasets.split(",")]
                if args.datasets else list(spc.dataset_names))
    cfg = dict(
        probe_train_set_size=args.probe_train_set_size,
        probe_test_set_size=spc.probe_test_set_size,
        context_length=spc.context_length,
        sae_batch_size=args.sae_batch_size,
        llm_batch_size=args.llm_batch_size or md["llm_batch_size"],
        top_n=args.top_n,
    )

    # Resolve the FP16/FP8 checkpoints (matched step) for each k.
    members = {k: f"w{args.width}_k{k}" for k in ks}
    resolved = {}  # k -> {"fp16": (step,dir), "fp8": (step,dir), "saebench": (t,k)}
    for k in ks:
        member = members[k]
        fp16 = discover_member_checkpoints(fp16_run, member)
        fp8 = discover_member_checkpoints(fp8_run, member)
        if not fp16 or not fp8:
            print(f"  [skip] k={k}: missing checkpoints "
                  f"(fp16 steps={sorted(fp16)}, fp8 steps={sorted(fp8)})")
            continue
        # Step selection: explicit per-precision > --target-step > latest common.
        fp16_step, fp8_step = args.fp16_step, args.fp8_step
        if args.target_step and args.target_step > 0:
            fp16_step = fp16_step if fp16_step is not None else args.target_step
            fp8_step = fp8_step if fp8_step is not None else args.target_step
        if fp16_step is None or fp8_step is None:
            common = sorted(set(fp16) & set(fp8))
            if common:
                step = common[-1]
                fp16_step = fp16_step if fp16_step is not None else step
                fp8_step = fp8_step if fp8_step is not None else step
            else:
                fp16_step = fp16_step if fp16_step is not None else max(fp16)
                fp8_step = fp8_step if fp8_step is not None else max(fp8)
                print(f"  [warn] k={k}: no common step; using fp16={fp16_step}, "
                      f"fp8={fp8_step} (NOT step-matched).")
        if fp16_step not in fp16 or fp8_step not in fp8:
            print(f"  [skip] k={k}: step not present "
                  f"(want fp16={fp16_step} in {sorted(fp16)}, "
                  f"fp8={fp8_step} in {sorted(fp8)}).")
            continue
        t, tk = nearest_saebench_trainer(k)
        resolved[k] = {
            "fp16": (fp16_step, fp16[fp16_step]),
            "fp8": (fp8_step, fp8[fp8_step]),
            "saebench": (t, tk),
        }

    print("=" * 78)
    print("  FP8 vs FP16 vs SAEBench feature comparison")
    print(f"  model {model_name}  hook {hook_name}  width {args.width}")
    print(f"  parts {args.parts}  saebench {'OFF' if args.no_saebench else 'ON'}")
    print(f"  datasets ({len(datasets)}): {datasets}")
    print(f"  output {out_root}")
    for k in ks:
        if k not in resolved:
            continue
        r = resolved[k]
        sb = "" if args.no_saebench else \
            f"  saebench trainer_{r['saebench'][0]} (k={r['saebench'][1]})"
        print(f"   k={k:5d}  fp16 step {r['fp16'][0]}  fp8 step {r['fp8'][0]}{sb}")
    print("=" * 78)
    if args.dry_run or not resolved:
        if not resolved:
            print("Nothing to do.")
        return

    torch.set_grad_enabled(False)

    # Load the base model once (sae_bench sparse probing uses no_processing).
    # Parts A (probing) and C (functional, needs token activations) both need it.
    model = None
    if args.parts in ("a", "c", "all"):
        from transformer_lens import HookedTransformer
        td = {"bfloat16": torch.bfloat16, "float16": torch.float16,
              "float32": torch.float32}[dtype]
        print(f"\nLoading {model_name} (dtype={dtype}) ...")
        model = HookedTransformer.from_pretrained_no_processing(
            model_name, device=device, dtype=td
        )
        model.eval()

    overall = {"model": model_name, "hook": hook_name, "width": args.width,
               "saebench": (None if args.no_saebench else md["saebench_repo"]),
               "k_pairing": {str(k): {"fp16_step": resolved[k]["fp16"][0],
                                      "fp8_step": resolved[k]["fp8"][0],
                                      "saebench_trainer": resolved[k]["saebench"][0],
                                      "saebench_k": resolved[k]["saebench"][1]}
                             for k in resolved},
               "k_results": {}}

    # ----- per-k SAE id mapping (named role -> unique sae id) ----------------- #
    def k_roles(k):
        r = resolved[k]
        roles = {"fp16": f"fp16_k{k}", "fp8": f"fp8_k{k}"}
        if not args.no_saebench:
            roles["saebench"] = f"saebench_t{r['saebench'][0]}"
        return roles

    # Unique SAE registry across ALL k (so each SAE is loaded/encoded only once).
    sae_specs: dict[str, object] = {}
    for k in resolved:
        r = resolved[k]
        sae_specs.setdefault(f"fp16_k{k}",
                             lambda d=r["fp16"][1]: load_our_sae(d, device, dtype))
        sae_specs.setdefault(f"fp8_k{k}",
                             lambda d=r["fp8"][1]: load_our_sae(d, device, dtype))
        if not args.no_saebench:
            t = r["saebench"][0]
            sae_specs.setdefault(
                f"saebench_t{t}",
                lambda t=t: load_saebench_sae(
                    t, layer, device, dtype, md["saebench_repo"],
                    md["saebench_path"], md["saebench_model"]),
            )

    decoder_cache: dict[str, torch.Tensor] = {}

    def get_decoder(sid):
        if sid not in decoder_cache:
            decoder_cache[sid] = sae_decoder(sae_specs[sid]()).cpu()
        return decoder_cache[sid]

    # ----- Part A: top features for ALL unique SAEs (one model pass / dataset) - #
    top_features, classes = None, None
    if args.parts in ("a", "all"):
        global_feats = out_root / "topk_features.json"
        if global_feats.exists() and not args.force:
            print(f"\n[Part A] loading cached top features {global_feats}")
            blob = json.loads(global_feats.read_text())
            top_features = blob["top_features"]
            classes = blob["classes_per_dataset"]
        else:
            random.seed(args.seed)
            top_features, classes, decs = run_part_a(
                model, layer, hook_name, sae_specs, datasets, cfg, device, args.seed
            )
            decoder_cache.update(decs)
            global_feats.write_text(json.dumps(
                {"classes_per_dataset": classes, "top_features": top_features,
                 "datasets": datasets, "config": cfg}, indent=2))
            print(f"[Part A] wrote {global_feats}")

    # ----- Part C: collect held-out activations ONCE (shared across all k) ----- #
    holdout_acts = None
    if args.parts in ("c", "all"):
        print(f"\n[Part C] collecting {args.func_tokens} held-out token activations "
              f"from '{datasets[0]}' ...")
        holdout_acts = collect_holdout_acts(
            model, layer, hook_name, cfg, datasets, device, args.func_tokens, args.seed
        )
        print(f"[Part C] held-out activations: {tuple(holdout_acts.shape)}")

    # ----- per-k assembly ---------------------------------------------------- #
    for k in ks:
        if k not in resolved:
            continue
        r = resolved[k]
        kdir = out_root / f"k{k}"
        kdir.mkdir(parents=True, exist_ok=True)
        print(f"\n{'#' * 60}\n# k = {k}\n{'#' * 60}")

        meta = {
            "k": k,
            "fp16": {"step": r["fp16"][0], "dir": str(r["fp16"][1])},
            "fp8": {"step": r["fp8"][0], "dir": str(r["fp8"][1])},
            "step_matched": r["fp16"][0] == r["fp8"][0],
        }
        if not args.no_saebench:
            meta["saebench"] = {"trainer": r["saebench"][0], "k": r["saebench"][1],
                                "repo": md["saebench_repo"]}
        (kdir / "meta.json").write_text(json.dumps(meta, indent=2))
        kres = {"meta": meta}
        roles = k_roles(k)

        # ---- Part A cosine assembly for this k ----
        if args.parts in ("a", "all") and top_features is not None:
            tf_view = {role: top_features[sid] for role, sid in roles.items()}
            dec_view = {role: get_decoder(sid) for role, sid in roles.items()}
            cos_rows = part_a_cosines(tf_view, classes, dec_view)
            (kdir / "topk_cosine.json").write_text(json.dumps(
                {"k": k, "meta": meta, "concepts": cos_rows}, indent=2))
            kres["partA"] = _summarize_part_a(cos_rows)
            print("  Part A matched-concept cosine (mean |cos| / mean best-match):")
            for pair, v in kres["partA"]["mean_abscos"].items():
                print(f"     top1-vs-top1 {pair:18s} {v:.3f}")
            for d, v in kres["partA"]["mean_bestmatch"].items():
                print(f"     best-match   {d:18s} {v:.3f}")

        # ---- Part B all-feature nearest cosine for this k ----
        if args.parts in ("b", "all"):
            out_npz = kdir / "allfeat_match.npz"
            if out_npz.exists() and not args.force:
                print(f"  [skip Part B] {out_npz} exists (use --force).")
                if (kdir / "allfeat_match_summary.json").exists():
                    kres["partB"] = json.loads(
                        (kdir / "allfeat_match_summary.json").read_text())
            else:
                dec_view = {role: get_decoder(sid) for role, sid in roles.items()}
                print("  Part B all-feature nearest cosine:")
                arrays, summ = run_part_b(dec_view, device)
                np.savez_compressed(out_npz, **arrays)
                (kdir / "allfeat_match_summary.json").write_text(
                    json.dumps(summ, indent=2))
                kres["partB"] = summ
                free(arrays)

        # ---- Part C functional equivalence for this k ----
        if args.parts in ("c", "all"):
            out_npz = kdir / "functional.npz"
            if out_npz.exists() and not args.force:
                print(f"  [skip Part C] {out_npz} exists (use --force).")
                if (kdir / "functional_summary.json").exists():
                    kres["partC"] = json.loads(
                        (kdir / "functional_summary.json").read_text())
            else:
                print("  Part C functional equivalence (shared held-out tokens):")
                arrays, summ = run_part_c(
                    sae_specs, roles, holdout_acts, device,
                    chunk=cfg["sae_batch_size"], matched_pair=("fp16", "fp8"),
                )
                np.savez_compressed(out_npz, **arrays)
                (kdir / "functional_summary.json").write_text(json.dumps(summ, indent=2))
                kres["partC"] = summ
                free(arrays)

        overall["k_results"][str(k)] = kres

    (out_root / "summary.json").write_text(json.dumps(overall, indent=2))
    print(f"\nDone. Summary -> {out_root / 'summary.json'}")


def _summarize_part_a(cos_rows):
    if not cos_rows:
        return {"n_concepts": 0, "mean_cos": {}, "mean_abscos": {}, "mean_bestmatch": {}}
    pairs = list(cos_rows[0]["cos"].keys())
    dirs = list(cos_rows[0]["bestmatch"].keys())
    mean_cos, mean_abs = {}, {}
    for pr in pairs:
        cs = np.array([row["cos"][pr] for row in cos_rows])
        mean_cos[pr] = float(cs.mean())
        mean_abs[pr] = float(np.abs(cs).mean())
    mean_best = {}
    for d in dirs:
        bs = np.array([row["bestmatch"][d]["cos"] for row in cos_rows])
        mean_best[d] = float(bs.mean())
    return {"n_concepts": len(cos_rows), "mean_cos": mean_cos,
            "mean_abscos": mean_abs, "mean_bestmatch": mean_best}


if __name__ == "__main__":
    main()
