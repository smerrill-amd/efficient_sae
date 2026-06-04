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

import time

import torch

from sae_train.utils import fmt_duration


class ProfilingComplete(Exception):
    """Raised to stop training early once --profile-steps batches are timed."""


class TimingProfiler:
    def __init__(self, device: str, max_steps: int = 0, report_every: int = 200):
        self.use_cuda = torch.cuda.is_available() and "cuda" in str(device)
        self.max_steps = max_steps
        self.report_every = report_every
        self.data_time = 0.0   # next(data_provider): LLM fwd + buffer refill/shuffle
        self.sae_time = 0.0    # SAETrainer.step summed over all SAEs
        self.n_batches = 0     # batches pulled (incl. norm-estimate warmup)
        self.n_sae_steps = 0   # per-SAE step() calls
        self._t_install = None

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

    def report(self, partial: bool) -> None:
        if self._t_install is None:
            return
        wall = time.perf_counter() - self._t_install
        tracked = self.data_time + self.sae_time
        other = max(wall - tracked, 0.0)

        def pct(x: float, denom: float) -> float:
            return 100.0 * x / denom if denom > 0 else 0.0

        n_saes = max(self.n_sae_steps // max(self.n_batches, 1), 1)
        tag = "running" if partial else "FINAL"
        print(
            f"\n{'='*64}\n"
            f"  TIMING PROFILE ({tag}) — {self.n_batches} batches, "
            f"{self.n_sae_steps} per-SAE steps (~{n_saes} SAEs)\n"
            f"{'-'*64}"
        )
        print(f"  LLM forward / act-gen : {fmt_duration(self.data_time):>10}  "
              f"({pct(self.data_time, tracked):5.1f}% compute | {pct(self.data_time, wall):5.1f}% wall)")
        print(f"  SAE train (fwd+bwd)   : {fmt_duration(self.sae_time):>10}  "
              f"({pct(self.sae_time, tracked):5.1f}% compute | {pct(self.sae_time, wall):5.1f}% wall)")
        print(f"  other (norm/eval/log) : {fmt_duration(other):>10}  "
              f"({'':>5}          {pct(other, wall):5.1f}% wall)")
        print(f"  wall                  : {fmt_duration(wall):>10}")
        if self.n_batches:
            print(f"  per batch: forward {1000 * self.data_time / self.n_batches:6.1f} ms"
                  f"  |  sae {1000 * self.sae_time / self.n_batches:6.1f} ms"
                  f"  (sae = sum over ~{n_saes} SAEs)")
        print(f"{'='*64}")
