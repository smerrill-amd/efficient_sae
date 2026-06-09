"""FP8 linear operators used inside the SAE forward/backward.

``fp8_linear(x, W, ...)`` computes ``x @ W`` (where ``x`` is ``[*, K]`` and ``W`` is
``[K, N]``) in fp8 according to a :class:`~architectures.fp8_formats.Float8Format`,
via one of two backends:

  * ``"emulated"`` (default) — software fake-quant of ``x`` and ``W`` to the chosen
    format (with a straight-through gradient), then a normal bf16/fp32 matmul. Works
    for *any* format split (E4M3, E3M4, E2M5, ...). This is the instrument for
    comparing formats' numerical effect on feature recovery. Optionally also quantizes
    the gradient (``quantize_grads=True``) to approximate fully-fp8 training.

  * ``"hardware"`` — real on-device fp8 GEMM via ``torch._scaled_mm`` with per-tensor
    dynamic scaling. Only valid for hardware-native formats (E4M3 / E5M2). Backward
    GEMMs also run in fp8 where shapes allow (``_scaled_mm`` needs the contraction and
    output dims divisible by 16), otherwise they transparently fall back to a bf16
    matmul. Use this to measure throughput, not for format sweeps.

  * ``"auto"`` — use ``"hardware"`` when the format has a working on-device dtype,
    else ``"emulated"``.
"""

from __future__ import annotations

import weakref

import torch

from architectures.fp8_formats import (
    Float8Format,
    compute_amax_scale,
    get_format,
    quantize_grad,
    quantize_ste,
)

Backend = str  # "emulated" | "hardware" | "auto"

# ---------------------------------------------------------------------------
# Weight-quantization cache for the hardware backend.
#
# `torch._scaled_mm` needs each operand pre-scaled and cast to fp8, with the
# right-hand operand in column-major layout. For the SAE weight that means a
# full `(amax -> scale -> cast -> column-major copy)` of W on *every* matmul —
# the deep-dive (experiments/fp8_*_breakdown.py) showed this prologue, not the
# GEMM, is what makes the hardware path slow (the GEMM itself is ~1.3-1.8x
# faster than bf16 on MI300X). But a weight only changes when the optimizer
# updates it, which bumps `tensor._version`. Caching the quantized weight keyed
# by `_version` returns *bit-identical* results while skipping the requantize
# whenever W is unchanged (inference/eval, repeated profiling, and the
# forward+backward reuse of fully-fp8 training). During ordinary training the
# weight changes every step, so this correctly misses and re-quantizes.
# ---------------------------------------------------------------------------
# Keyed by id(weight); each entry holds (version, dt, fp8_weight, scale). A weakref
# finalizer evicts the entry when the weight tensor is garbage-collected, so stale
# ids can never be reused. (We can't use a WeakKeyDictionary directly because tensor
# `__eq__` is elementwise, which breaks dict lookups.)
_WEIGHT_FP8_CACHE: dict[int, tuple] = {}


def resolve_backend(backend: Backend, fmt: Float8Format) -> str:
    if backend == "auto":
        return "hardware" if fmt.hardware_dtype is not None else "emulated"
    if backend == "hardware" and fmt.hardware_dtype is None:
        raise ValueError(
            f"fp8 format {fmt.name!r} has no hardware _scaled_mm dtype on this device; "
            "use backend='emulated' (or 'auto') for non-native formats."
        )
    if backend not in ("emulated", "hardware"):
        raise ValueError(f"Unknown fp8 backend {backend!r}")
    return backend


# ---------------------------------------------------------------------------
# Emulated backend
# ---------------------------------------------------------------------------

def _fp8_linear_emulated(
    x: torch.Tensor,
    W: torch.Tensor,
    fmt: Float8Format,
    quantize_grads: bool,
    compute_dtype: torch.dtype,
) -> torch.Tensor:
    xq = quantize_ste(x, fmt)
    wq = quantize_ste(W, fmt)
    out = (xq.to(compute_dtype) @ wq.to(compute_dtype))
    if quantize_grads:
        # Quantize the gradient w.r.t. the matmul output before it feeds the backward
        # GEMMs (a simple stand-in for per-operand gradient quantization).
        out = quantize_grad(out, fmt)
    return out


# ---------------------------------------------------------------------------
# Hardware backend (torch._scaled_mm, per-tensor dynamic scaling)
# ---------------------------------------------------------------------------

def _quantize_per_tensor(t: torch.Tensor, dt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-tensor dynamic-scale quantize ``t`` to fp8 dtype ``dt`` (row-major, contiguous).

    Returns ``(t_fp8, scale)`` where ``scale`` is the fp32 dequant multiplier expected
    by ``torch._scaled_mm``.
    """
    fmt_max = torch.finfo(dt).max
    amax = t.detach().abs().amax().clamp(min=1e-12).float()
    scale = amax / fmt_max
    tq = (t.float() / scale).clamp(-fmt_max, fmt_max).to(dt).contiguous()
    return tq, scale


def _quantize_weight_cached(b: torch.Tensor, dt: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Quantize weight ``b`` ([K,N]) to column-major fp8, reusing a cached result when
    ``b`` is unchanged (same ``_version``). See ``_WEIGHT_FP8_CACHE`` for why."""
    key = id(b)
    ver = b._version
    cached = _WEIGHT_FP8_CACHE.get(key)
    if cached is not None and cached[0] == ver and cached[1] == dt:
        return cached[2], cached[3]

    fmt_max = torch.finfo(dt).max
    amax = b.detach().abs().amax().clamp(min=1e-12).float()
    scale = amax / fmt_max
    # `b` must be column-major for _scaled_mm: make `bq.t()` contiguous.
    bq = (b.float() / scale).clamp(-fmt_max, fmt_max).to(dt)
    bq = bq.t().contiguous().t()
    if key not in _WEIGHT_FP8_CACHE:
        try:
            weakref.finalize(b, _WEIGHT_FP8_CACHE.pop, key, None)
        except TypeError:
            return bq, scale  # not weak-referenceable: compute but don't cache
    _WEIGHT_FP8_CACHE[key] = (ver, dt, bq, scale)
    return bq, scale


def _scaled_mm(a: torch.Tensor, b: torch.Tensor, dt: torch.dtype,
               out_dtype: torch.dtype, cache_weight: bool = False) -> torch.Tensor:
    """fp8 GEMM ``a @ b`` (a:[M,K], b:[K,N]) with per-tensor dynamic scaling.

    Falls back to a plain ``out_dtype`` matmul when ``_scaled_mm``'s shape constraints
    (contraction dim K and output dim N divisible by 16) aren't met. When
    ``cache_weight`` is set, ``b``'s fp8 quantization is memoized by tensor version
    (bit-identical reuse for unchanged weights); ``a`` is always quantized fresh.
    """
    M, K = a.shape
    N = b.shape[1]

    if K % 16 != 0 or N % 16 != 0:
        return a.to(out_dtype) @ b.to(out_dtype)

    aq, sa = _quantize_per_tensor(a, dt)
    if cache_weight:
        bq, sb = _quantize_weight_cached(b, dt)
    else:
        bq, sb = _quantize_per_tensor(b, dt)
        bq = bq.t().contiguous().t()  # column-major for _scaled_mm
    try:
        return torch._scaled_mm(aq, bq, scale_a=sa, scale_b=sb, out_dtype=out_dtype)
    except Exception:
        return a.to(out_dtype) @ b.to(out_dtype)


class _HardwareFp8Linear(torch.autograd.Function):
    """``x @ W`` with on-device fp8 GEMMs in forward and (where possible) backward."""

    @staticmethod
    def forward(ctx, x, W, fmt, quantize_grads, compute_dtype):  # type: ignore[override]
        ctx.save_for_backward(x, W)
        ctx.dt = fmt.hardware_dtype
        ctx.compute_dtype = compute_dtype
        ctx.fp8_backward = quantize_grads
        return _scaled_mm(x, W, ctx.dt, compute_dtype, cache_weight=True)

    @staticmethod
    def backward(ctx, grad_out):  # type: ignore[override]
        x, W = ctx.saved_tensors
        dt, cdt = ctx.dt, ctx.compute_dtype
        if ctx.fp8_backward:
            # grad_x = grad_out @ W^T ; grad_W = x^T @ grad_out
            grad_x = _scaled_mm(grad_out, W.t().contiguous(), dt, cdt)
            grad_W = _scaled_mm(x.t().contiguous(), grad_out, dt, cdt)
        else:
            grad_x = grad_out.to(cdt) @ W.t().to(cdt)
            grad_W = x.t().to(cdt) @ grad_out.to(cdt)
        return grad_x.to(x.dtype), grad_W.to(W.dtype), None, None, None


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def fp8_linear(
    x: torch.Tensor,
    W: torch.Tensor,
    fmt: "str | Float8Format",
    backend: Backend = "emulated",
    quantize_grads: bool = False,
    compute_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Compute ``x @ W`` in fp8.

    Args:
        x: Activations of shape ``[*, K]``.
        W: Weight of shape ``[K, N]`` (note: this is the SAE convention where
            ``sae_in @ W_enc`` and ``feature_acts @ W_dec`` are both ``x @ W``).
        fmt: Float8 format name (``"e4m3"``, ``"e3m4"``, ...) or a :class:`Float8Format`.
        backend: ``"emulated"`` (any format), ``"hardware"`` (E4M3/E5M2 only), or
            ``"auto"``.
        quantize_grads: Also quantize gradients to fp8 (approximate fully-fp8 training).
        compute_dtype: Accumulation/output dtype for the emulated matmul and the
            ``_scaled_mm`` output. Defaults to ``x``'s dtype (falling back to bf16 for
            fp8/odd input dtypes).
    """
    fmt = get_format(fmt)
    backend = resolve_backend(backend, fmt)

    if compute_dtype is None:
        compute_dtype = x.dtype if x.dtype in (torch.float32, torch.bfloat16, torch.float16) else torch.bfloat16

    *batch, K = x.shape
    x2 = x.reshape(-1, K)

    if backend == "hardware":
        out = _HardwareFp8Linear.apply(x2, W, fmt, quantize_grads, compute_dtype)
    else:
        out = _fp8_linear_emulated(x2, W, fmt, quantize_grads, compute_dtype)

    return out.reshape(*batch, W.shape[1]).to(x.dtype)
