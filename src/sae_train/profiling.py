"""Wall-clock timing profiler shared by all train_sae_FP*.py entrypoints.

The trainer hot loop is exactly two operations per batch:
  1. next(data_provider)  -> pulls activations; the shared LLM forward runs
                             lazily here whenever the shuffle buffer refills.
                             Sweep modes use the multi-hook generator; single-SAE
                             mode uses ActivationsStore.__next__.
  2. trainer.step(batch)  -> per-SAE forward + backward + optimizer.
We monkey-patch those seams with CUDA-synchronized timers to get a true
wall-clock split between "LLM forward / activation generation" and "SAE
training", plus an "other" bucket (norm-est, evals, logging, ckpt).

This is precision-agnostic: it times whatever the SAE forward/backward does,
so it works unchanged for fp16/fp8/fp4.
"""

import json
import time
from pathlib import Path

import torch

from sae_train.utils import fmt_duration


class ProfilingComplete(Exception):
    """Raised to stop training early once --profile-steps batches are timed."""


class TimingProfiler:
    def __init__(self, device: str, max_steps: int = 0, report_every: int = 200,
                 json_path: "str | Path | None" = None, log_wandb: bool = False):
        self.use_cuda = torch.cuda.is_available() and "cuda" in str(device)
        self.max_steps = max_steps
        self.report_every = report_every
        self.json_path = Path(json_path) if json_path else None
        self.log_wandb = log_wandb
        self.data_time = 0.0   # next(data_provider): LLM fwd + buffer refill/shuffle
        self.sae_time = 0.0    # SAETrainer.step summed over all SAEs
        self.n_batches = 0     # batches pulled (incl. norm-estimate warmup)
        self.n_sae_steps = 0   # per-SAE step() calls
        self._t_install = None
        self._wandb_defined = False

    def _sync(self) -> None:
        if self.use_cuda:
            torch.cuda.synchronize()

    def install(self) -> None:
        from sae_lens.training.activations_store import ActivationsStore
        from sae_lens.training.sae_trainer import SAETrainer

        prof = self
        self._t_install = time.perf_counter()

        # --- activation generation (the shared LLM forward happens lazily here) ---
        # Sweep modes pull batches from the multi-hook generator
        # (get_multi_hook_data_loader); single-SAE mode pulls from
        # ActivationsStore.__next__ (single-hook get_data_loader). The two code
        # paths are disjoint, so wrapping both is safe and never double-counts.
        orig_loader = ActivationsStore.get_multi_hook_data_loader

        def timed_loader(store):  # type: ignore[no-untyped-def]
            return prof._wrap_iter(orig_loader(store))

        ActivationsStore.get_multi_hook_data_loader = timed_loader  # type: ignore[method-assign]

        orig_next = ActivationsStore.__next__

        def timed_next(store):  # type: ignore[no-untyped-def]
            prof._sync()
            t0 = time.perf_counter()
            val = orig_next(store)
            prof._sync()
            prof._record_data(time.perf_counter() - t0)
            return val

        ActivationsStore.__next__ = timed_next  # type: ignore[method-assign]

        # --- SAE compute (forward + backward + optimizer), both single & sweep ---
        orig_step = SAETrainer.step

        def timed_step(trainer, batch):  # type: ignore[no-untyped-def]
            prof._sync()
            t0 = time.perf_counter()
            out = orig_step(trainer, batch)
            prof._sync()
            prof.sae_time += time.perf_counter() - t0
            prof.n_sae_steps += 1
            return out

        SAETrainer.step = timed_step  # type: ignore[method-assign]

    def _record_data(self, dt: float) -> None:
        """Accumulate one activation-batch's timing, then handle periodic
        reporting and the --profile-steps early stop. Shared by the single-hook
        (__next__) and multi-hook (generator) data paths."""
        self.data_time += dt
        self.n_batches += 1
        if self.report_every and self.n_batches % self.report_every == 0:
            self.report(partial=True)
        if self.max_steps and self.n_batches >= self.max_steps:
            raise ProfilingComplete

    def _wrap_iter(self, inner):
        prof = self
        while True:
            prof._sync()
            t0 = time.perf_counter()
            try:
                val = next(inner)
            except StopIteration:
                return
            prof._sync()
            dt = time.perf_counter() - t0
            yield val
            prof._record_data(dt)

    def _stats(self, partial: bool) -> dict:
        """Snapshot the current breakdown as a flat, serializable dict."""
        wall = time.perf_counter() - self._t_install
        tracked = self.data_time + self.sae_time
        other = max(wall - tracked, 0.0)

        def pct(x: float, denom: float) -> float:
            return 100.0 * x / denom if denom > 0 else 0.0

        nb = max(self.n_batches, 1)
        return {
            "complete": not partial,
            "n_batches": self.n_batches,
            "n_sae_steps": self.n_sae_steps,
            "approx_n_saes": max(self.n_sae_steps // nb, 1),
            "forward_seconds": self.data_time,
            "sae_seconds": self.sae_time,
            "other_seconds": other,
            "wall_seconds": wall,
            "forward_pct_compute": pct(self.data_time, tracked),
            "sae_pct_compute": pct(self.sae_time, tracked),
            "forward_pct_wall": pct(self.data_time, wall),
            "sae_pct_wall": pct(self.sae_time, wall),
            "other_pct_wall": pct(other, wall),
            "forward_ms_per_batch": 1000 * self.data_time / nb,
            "sae_ms_per_batch": 1000 * self.sae_time / nb,
        }

    def report(self, partial: bool) -> None:
        if self._t_install is None:
            return
        s = self._stats(partial)
        tag = "running" if partial else "FINAL"
        print(
            f"\n{'='*64}\n"
            f"  TIMING PROFILE ({tag}) — {s['n_batches']} batches, "
            f"{s['n_sae_steps']} per-SAE steps (~{s['approx_n_saes']} SAEs)\n"
            f"{'-'*64}"
        )
        print(f"  LLM forward / act-gen : {fmt_duration(s['forward_seconds']):>10}  "
              f"({s['forward_pct_compute']:5.1f}% compute | {s['forward_pct_wall']:5.1f}% wall)")
        print(f"  SAE train (fwd+bwd)   : {fmt_duration(s['sae_seconds']):>10}  "
              f"({s['sae_pct_compute']:5.1f}% compute | {s['sae_pct_wall']:5.1f}% wall)")
        print(f"  other (norm/eval/log) : {fmt_duration(s['other_seconds']):>10}  "
              f"({'':>5}          {s['other_pct_wall']:5.1f}% wall)")
        print(f"  wall                  : {fmt_duration(s['wall_seconds']):>10}")
        if self.n_batches:
            print(f"  per batch: forward {s['forward_ms_per_batch']:6.1f} ms"
                  f"  |  sae {s['sae_ms_per_batch']:6.1f} ms"
                  f"  (sae = sum over ~{s['approx_n_saes']} SAEs)")
        print(f"{'='*64}")

        self._write_json(s)
        self._maybe_log_wandb(s)

    def _write_json(self, stats: dict) -> None:
        """Persist the breakdown to JSON (overwritten on every report, so an
        interrupted run still leaves the latest snapshot on disk)."""
        if self.json_path is None:
            return
        try:
            self.json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.json_path, "w") as f:
                json.dump(stats, f, indent=2)
        except OSError as e:
            print(f"  [profiler] could not write {self.json_path}: {e}")

    def _maybe_log_wandb(self, stats: dict) -> None:
        """Log the running breakdown to W&B against a dedicated `profile/n_batches`
        axis. No-op once the run has finished (e.g. the FINAL report fires after
        the runner calls wandb.finish()), so final totals live only in JSON/stdout.
        commit=False attaches to the current step without advancing W&B's global
        step, so it never disturbs SAELens' explicit step-based logging."""
        if not self.log_wandb:
            return
        try:
            import wandb
        except ImportError:
            return
        if wandb.run is None:
            return
        if not self._wandb_defined:
            wandb.define_metric("profile/n_batches")
            wandb.define_metric("profile/*", step_metric="profile/n_batches")
            self._wandb_defined = True
        wandb.log(
            {
                "profile/n_batches": stats["n_batches"],
                "profile/forward_ms_per_batch": stats["forward_ms_per_batch"],
                "profile/sae_ms_per_batch": stats["sae_ms_per_batch"],
                "profile/forward_pct_compute": stats["forward_pct_compute"],
                "profile/sae_pct_compute": stats["sae_pct_compute"],
                "profile/cumulative_forward_seconds": stats["forward_seconds"],
                "profile/cumulative_sae_seconds": stats["sae_seconds"],
            },
            commit=False,
        )
