"""Configurable FP8 number formats + a software emulator.

This is the heart of the FP8 SAE workshop. The goal is to be able to *systematically*
compare different 8-bit floating point layouts (E4M3, E3M4, E2M5, E5M2, ...) on the
exact same training code, so we can ask questions like "how many mantissa bits does a
BatchTopK SAE need to recover ground-truth features?".

A float8 format is fully described by how it splits its 8 bits:

    1 sign bit + ``exp_bits`` exponent bits + ``man_bits`` mantissa bits = 8

so ``exp_bits + man_bits == 7``. Trading an exponent bit for a mantissa bit trades
*dynamic range* for *precision*:

    E5M2 : huge range, coarse precision   (IEEE-style, hardware-native)
    E4M3 : balanced                        (hardware-native on MI300 / Hopper)
    E3M4 : less range, finer precision
    E2M5 : tiny range, finest precision

Two execution backends are provided (see ``fp8_linear`` in ``fp8_ops``):

  * ``emulated`` — a software fake-quant (round-to-nearest at the binade) that works
    for *any* (exp, man) split. Compute still happens in bf16/fp32, so this isolates
    the **numerical** effect of the format (precision/range) on what the SAE learns.
    This is the apples-to-apples instrument for comparing formats.

  * ``hardware`` — real on-device FP8 matmuls via ``torch._scaled_mm``, only available
    for the two hardware-native formats (E4M3 / E5M2) and only the variant the local
    accelerator implements. This is for measuring *speed*, not for format sweeps.

The emulator treats the all-ones exponent field as a normal number (OCP "fn"-style,
no inf), so e.g. emulated E4M3 max is ~480 vs the OCP-fn 448. The few reserved NaN
encodings are not modelled. This is intentional: with per-tensor dynamic scaling the
absolute max is irrelevant (we scale the tensor's amax onto the format's max), and it
keeps every format on the same simple footing.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import torch


@dataclass(frozen=True)
class Float8Format:
    """An 8-bit float layout: 1 sign + ``exp_bits`` exponent + ``man_bits`` mantissa.

    All derived quantities (bias, exponent range, max/min normal) follow the usual
    IEEE-754 conventions, with the all-ones exponent treated as a normal value.
    """

    name: str
    exp_bits: int
    man_bits: int

    def __post_init__(self) -> None:
        if self.exp_bits < 1 or self.man_bits < 0:
            raise ValueError(f"{self.name}: need exp_bits>=1, man_bits>=0")
        if self.exp_bits + self.man_bits != 7:
            raise ValueError(
                f"{self.name}: exp_bits+man_bits must be 7 (got "
                f"{self.exp_bits}+{self.man_bits}={self.exp_bits + self.man_bits}); "
                "a signed fp8 format has 1 sign bit + 7 exponent/mantissa bits."
            )

    @property
    def bias(self) -> int:
        return 2 ** (self.exp_bits - 1) - 1

    @property
    def exp_max(self) -> int:
        """Largest (unbiased) exponent of a normal number."""
        return (2**self.exp_bits - 1) - self.bias

    @property
    def exp_min(self) -> int:
        """Smallest (unbiased) exponent of a normal number."""
        return 1 - self.bias

    @property
    def max_normal(self) -> float:
        return (2.0 - 2.0 ** (-self.man_bits)) * (2.0**self.exp_max)

    @property
    def min_normal(self) -> float:
        return 2.0**self.exp_min

    @property
    def min_subnormal(self) -> float:
        """Smallest representable positive (subnormal) magnitude."""
        return 2.0 ** (self.exp_min - self.man_bits)

    @property
    def hardware_dtype(self) -> torch.dtype | None:
        """The on-device torch fp8 dtype implementing this format, if any.

        Only E4M3 and E5M2 have hardware fp8 dtypes. Which *variant* (the OCP ``fn``
        form used by NVIDIA, or the ``fnuz`` form used by AMD/MI300) actually works
        with ``torch._scaled_mm`` is probed at runtime; see ``hardware_gemm_dtype``.
        """
        return hardware_gemm_dtype(self.name)

    def describe(self) -> str:
        return (
            f"{self.name} (E{self.exp_bits}M{self.man_bits}): "
            f"range +/-{self.max_normal:.4g}, "
            f"min_normal={self.min_normal:.3g}, "
            f"~{self.man_bits + 1} bits precision, "
            f"hw={'yes' if self.hardware_dtype is not None else 'no'}"
        )


# The canonical signed fp8 formats (sign + 7 bits). exp_bits + man_bits == 7.
FORMATS: dict[str, Float8Format] = {
    "e6m1": Float8Format("e6m1", 6, 1),
    "e5m2": Float8Format("e5m2", 5, 2),
    "e4m3": Float8Format("e4m3", 4, 3),
    "e3m4": Float8Format("e3m4", 3, 4),
    "e2m5": Float8Format("e2m5", 2, 5),
}


def get_format(spec: "str | Float8Format") -> Float8Format:
    """Resolve a format from a name (``"e4m3"``, ``"E4M3"``) or pass a Float8Format
    through. Also accepts arbitrary ``"eXmY"`` strings not in :data:`FORMATS`."""
    if isinstance(spec, Float8Format):
        return spec
    key = spec.lower().strip()
    if key in FORMATS:
        return FORMATS[key]
    # Parse an arbitrary "eXmY" spec on the fly so callers aren't limited to the table.
    if key.startswith("e") and "m" in key:
        try:
            exp_s, man_s = key[1:].split("m", 1)
            return Float8Format(key, int(exp_s), int(man_s))
        except (ValueError, IndexError) as err:
            raise ValueError(f"Could not parse fp8 format spec {spec!r}") from err
    raise ValueError(
        f"Unknown fp8 format {spec!r}. Known: {sorted(FORMATS)} "
        "(or pass an 'eXmY' string with X+Y==7)."
    )


# ---------------------------------------------------------------------------
# Hardware (torch._scaled_mm) dtype detection
# ---------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _scaled_mm_supports(dtype: torch.dtype) -> bool:
    """Probe whether ``torch._scaled_mm`` actually runs for ``dtype`` on this device.

    NVIDIA Hopper implements the OCP ``fn`` fp8 dtypes; AMD MI300 implements the
    ``fnuz`` variants. Rather than hard-code a platform check, we just try a tiny
    matmul once and cache the result.
    """
    if not torch.cuda.is_available() or not hasattr(torch, "_scaled_mm"):
        return False
    try:
        dev = torch.device("cuda")
        a = torch.zeros(16, 16, device=dev).to(dtype)
        b = torch.zeros(16, 16, device=dev).to(dtype).t()
        one = torch.tensor(1.0, device=dev)
        torch._scaled_mm(a, b, scale_a=one, scale_b=one, out_dtype=torch.bfloat16)
        return True
    except Exception:
        return False


@lru_cache(maxsize=None)
def hardware_gemm_dtype(format_name: str) -> torch.dtype | None:
    """Return the torch fp8 dtype usable with ``_scaled_mm`` for this format, or None.

    Tries the OCP ``fn`` variant first (NVIDIA), then the ``fnuz`` variant (AMD/MI300).
    Only E4M3 and E5M2 have hardware dtypes at all; every other split is emulation-only.
    """
    candidates: dict[str, tuple[torch.dtype, ...]] = {
        "e4m3": (torch.float8_e4m3fn, torch.float8_e4m3fnuz),
        "e5m2": (torch.float8_e5m2, torch.float8_e5m2fnuz),
    }
    for dt in candidates.get(format_name.lower(), ()):  # type: ignore[arg-type]
        if _scaled_mm_supports(dt):
            return dt
    return None


# ---------------------------------------------------------------------------
# Software emulation (works for any (exp, man) split)
# ---------------------------------------------------------------------------

def compute_amax_scale(x: torch.Tensor, fmt: Float8Format) -> torch.Tensor:
    """Per-tensor dynamic scale mapping ``x``'s amax onto the format's max normal.

    Returns an fp32 scalar ``scale`` such that ``x / scale`` lands in the format's
    representable range and ``x_q * scale`` dequantizes back. Matches the convention
    used by ``torch._scaled_mm`` (the scale is a *dequantization* multiplier).
    """
    amax = x.detach().abs().amax().clamp(min=1e-12).float()
    return amax / fmt.max_normal


def simulate_fp8(
    x: torch.Tensor,
    fmt: Float8Format,
    scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """Round ``x`` to the given fp8 format (round-to-nearest-even at the binade).

    With ``scale=None`` a per-tensor dynamic scale is computed from ``x``'s amax. The
    result is returned in ``x``'s dtype (values dequantized back to the original scale),
    so this is a "fake quant": it carries the numerical error of the format while
    staying in a normal float dtype for downstream compute.
    """
    if scale is None:
        scale = compute_amax_scale(x, fmt)

    xs = (x.float() / scale)
    sign = torch.sign(xs)
    a = xs.abs()

    # Binade exponent, clamped to the format's normal exponent range. Clamping low to
    # exp_min reproduces the fixed-step "subnormal" region near zero.
    e = torch.floor(torch.log2(a.clamp(min=1e-30))).clamp(fmt.exp_min, fmt.exp_max)
    step = torch.exp2(e - fmt.man_bits)
    q = torch.round(a / step) * step
    q = q.clamp(max=fmt.max_normal)

    # Flush magnitudes below half the smallest subnormal step to zero.
    q = torch.where(a < 0.5 * fmt.min_subnormal, torch.zeros_like(q), q)

    out = sign * q * scale
    return out.to(x.dtype)


class _FakeQuantSTE(torch.autograd.Function):
    """Straight-through fake-quant: forward rounds to fp8, backward is identity."""

    @staticmethod
    def forward(ctx, x: torch.Tensor, fmt: Float8Format) -> torch.Tensor:  # type: ignore[override]
        return simulate_fp8(x, fmt)

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return grad_out, None


def quantize_ste(x: torch.Tensor, fmt: Float8Format) -> torch.Tensor:
    """fp8 fake-quant with a straight-through gradient (used for weights & activations)."""
    return _FakeQuantSTE.apply(x, fmt)


class _QuantizeGrad(torch.autograd.Function):
    """Identity forward; quantizes the gradient to fp8 on the way back.

    Used to *approximately* model gradient quantization in fully-fp8 training: the
    gradient flowing into a matmul's operands is rounded to the format before the
    backward GEMMs. (A faithful implementation would quantize each backward GEMM
    operand separately; this single hook is a deliberately simple stand-in.)
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, fmt: Float8Format) -> torch.Tensor:  # type: ignore[override]
        ctx.fmt = fmt
        return x

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):  # type: ignore[override]
        return simulate_fp8(grad_out, ctx.fmt), None


def quantize_grad(x: torch.Tensor, fmt: Float8Format) -> torch.Tensor:
    return _QuantizeGrad.apply(x, fmt)
