"""quantize_block must compute the int8 scale and the data/scale division in
float32, the way it already did for every fp8 target.

Found by adversarial review (2026-09-02): get_max_min returned a Python int
for integer targets, so `abs_max / 127` on bf16 activations stayed bf16 --
an 8-bit-mantissa scale and an 8-bit-mantissa division feeding a 127-level
rounding. The fp8 branch cast finfo.max to float32, which promoted everything
to f32. Only the int8 arm of the dtype matrix went through the low-precision
path; it is also the only arm that regressed on output quality.
"""
import importlib.util
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
UTIL = ROOT / "tpu_inference" / "kernels" / "quantized_matmul" / "util.py"


def _util():
    """Load the REAL util.py without importing tpu_inference/__init__ (needs vllm)."""
    import sys
    import types
    pytest.importorskip("jax")
    for k in [k for k in sys.modules if k.startswith("tpu_inference")]:
        del sys.modules[k]
    pkg = types.ModuleType("tpu_inference")
    pkg.__path__ = [str(ROOT / "tpu_inference")]
    sys.modules["tpu_inference"] = pkg
    lg = types.ModuleType("tpu_inference.logger")
    lg.init_logger = lambda *a, **k: type("L", (), {
        "__getattr__":
        lambda s, n: (lambda *a, **k: None)
    })()
    sys.modules["tpu_inference.logger"] = lg
    for name, d in (("tpu_inference.kernels",
                     ROOT / "tpu_inference" / "kernels"),
                    ("tpu_inference.kernels.quantized_matmul",
                     ROOT / "tpu_inference" / "kernels" / "quantized_matmul")):
        m = types.ModuleType(name)
        m.__path__ = [str(d)]
        sys.modules[name] = m
    spec = importlib.util.spec_from_file_location(
        "tpu_inference.kernels.quantized_matmul.util", UTIL)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


def _f32_reference(x, dtype_max=127.0):
    xf = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(xf), axis=-1, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / dtype_max).astype(np.float32)
    return np.round(xf / scale).astype(np.int8), scale


def test_int8_scale_is_float32_for_bf16_activations():
    import jax
    import jax.numpy as jnp
    u = _util()
    x = jax.random.normal(jax.random.PRNGKey(0),
                          (16, 256)).astype(jnp.bfloat16)
    q, scale = u.quantize_block(x, axis=-1, target_dtype=jnp.int8)
    assert scale.dtype == jnp.float32, f"int8 scale computed in {scale.dtype}, fp8's is float32"
    assert q.dtype == jnp.int8


def test_int8_codes_track_the_f32_reference():
    """f32 path: scale matches numpy to 1e-6 and codes differ from the numpy
    reference only by round-half ties (|diff| <= 1, well under 1% of codes).
    The old bf16 path: 7.6% of codes differ (measured 2026-09-02, [256x3840]
    N(0,1)), 5.3% more RMS error. The bound below sits between the two."""
    import jax
    import jax.numpy as jnp
    u = _util()
    x = jax.random.normal(jax.random.PRNGKey(1),
                          (32, 512)).astype(jnp.bfloat16)
    q, scale = u.quantize_block(x, axis=-1, target_dtype=jnp.int8)
    q_ref, s_ref = _f32_reference(np.asarray(x.astype(jnp.float32)))
    np.testing.assert_allclose(np.asarray(scale), s_ref, rtol=1e-6)
    d = np.asarray(q).astype(int) - q_ref.astype(int)
    frac = float(np.mean(d != 0))
    assert int(np.abs(d).max(
    )) <= 1, "codes off by more than one level: not a tie-rounding difference"
    assert frac < 0.01, f"{100*frac:.2f}% of int8 codes differ from the f32 reference (bf16 arithmetic leaked in; old path: 7.6%)"


def test_fp8_branch_unchanged():
    import jax
    import jax.numpy as jnp
    u = _util()
    x = jax.random.normal(jax.random.PRNGKey(2), (8, 128)).astype(jnp.bfloat16)
    q, scale = u.quantize_block(x, axis=-1, target_dtype=jnp.float8_e4m3fn)
    assert scale.dtype == jnp.float32 and q.dtype == jnp.float8_e4m3fn


# ---- quantize_array: the Pallas-kernel path (kernel.py matmul_body) ---------


def _f32_ref_kernel_style(x, qd):
    """Rounded, f32 reference for a per-ROW scale over the last axis."""
    import jax.numpy as jnp
    xf = np.asarray(x.astype(jnp.float32))
    dmax = float((jnp.iinfo if not jnp.issubdtype(qd, jnp.floating) else
                  jnp.finfo)(qd).max)
    s = (np.max(np.abs(xf), axis=-1, keepdims=True) / dmax).astype(np.float32)
    q = xf / s
    return (np.round(q).astype(np.int8) if qd == jnp.int8 else np.asarray(
        jnp.asarray(q).astype(qd))), s


def _qa_inputs():
    import jax
    import jax.numpy as jnp
    x = jax.random.normal(jax.random.PRNGKey(7),
                          (256, 3840)).astype(jnp.bfloat16)
    xam = jnp.max(
        jnp.abs(x), axis=-1,
        keepdims=False)[None, :]  # exactly as kernel.py:157-160 builds it
    return x, xam


def test_quantize_array_int8_rounds_and_is_unbiased():
    """MEASURED 2026-09-02: the old path truncated (astype) -> |deq|-|x| bias
    -0.0125, RMS 0.0154, 42% of codes off-by-one vs a rounded f32 reference.
    This code: bias ~0, RMS ~0.0093, <10% codes differing (rounding ties)."""
    import jax.numpy as jnp
    u = _util()
    x, xam = _qa_inputs()
    q, s = u.quantize_array(x, xam, jnp.int8)
    assert s.dtype == jnp.float32
    qref, sref = _f32_ref_kernel_style(x, jnp.int8)
    np.testing.assert_allclose(np.asarray(s), sref, rtol=1e-6)
    xf = np.asarray(x.astype(jnp.float32))
    deq = np.asarray(q.astype(jnp.float32)) * np.asarray(s)
    bias = float(np.mean(np.abs(deq)) - np.mean(np.abs(xf)))
    mism = float(np.mean(np.asarray(q) != qref))
    d = np.asarray(q).astype(int) - qref.astype(int)
    assert abs(
        bias
    ) < 1e-3, f"int8 quantization is biased toward zero by {bias:+.5f}: astype truncation is back"
    assert int(
        np.abs(d).max()
    ) <= 1 and mism < 0.10, f"{100*mism:.1f}% of codes differ (max |d|={int(np.abs(d).max())}); old truncating path: 42%"


def test_quantize_array_fp8_scale_is_f32_and_residual_is_the_bf16_multiply():
    """MEASURED 2026-09-02, fp8 e4m3fn codes vs an f32 reference:
         weak float scale, bf16 multiply (as shipped before)  3.54%
         f32 scale, bf16 multiply (THIS code)                 3.22%
         f32 scale, f32 multiply                              0.01%
    For fp8 the scale precision was never the main error -- the bf16 block
    multiply is. That variant needs an f32 x-block temporary inside the
    Pallas body, which get_vmem_limit does not account for and which cannot
    be sized from CPU, so it is the follow-up, not this change. This test pins
    what IS true: the scale is f32, and the code count is no worse than the
    weak-typed path and improves once the multiply is widened."""
    import jax.numpy as jnp
    u = _util()
    x, xam = _qa_inputs()
    q, s = u.quantize_array(x, xam, jnp.float8_e4m3fn)
    assert s.dtype == jnp.float32, "fp8 scale is not float32 (weak-typed Python float is back)"
    qref, sref = _f32_ref_kernel_style(x, jnp.float8_e4m3fn)
    # AUDIT 2026-09-03: the dtype assertion above CANNOT detect the pre-#45
    # code, because that path also ended in `.astype(jnp.float32)` -- it was
    # the DIVISION that ran at bf16, not the final cast. Compare the scale to
    # the f32 reference instead, which is what the int8 arm already does.
    # MEASURED on this input (256x3840 N(0,1) bf16, PRNGKey(7)):
    #   this code      max rel scale error 1.19e-07,  3.2174% codes differ
    #   pre-#45 code   max rel scale error 3.33e-03,  3.5394% codes differ
    np.testing.assert_allclose(
        np.asarray(s),
        sref,
        rtol=1e-6,
        err_msg=("fp8 scale is bf16-rounded: the weakly-typed Python-float "
                 "divisor is back (this code 1.19e-07, pre-#45 3.33e-03)"))
    mism = float(
        np.mean(
            np.asarray(q.astype(jnp.float32)) != np.asarray(
                jnp.asarray(qref).astype(jnp.float32))))
    assert mism < 0.033, f"{100*mism:.2f}% of fp8 codes differ (this code 3.2174%; the weak-scale path 3.5394%)"


def test_kernel_and_xla_paths_quantize_int8_alike():
    """quantize_array (kernel) and quantize_block (XLA) quantize the SAME
    activations; they must agree to rounding ties, not to a truncation bias."""
    import jax.numpy as jnp
    u = _util()
    x, xam = _qa_inputs()
    qa, sa = u.quantize_array(x, xam, jnp.int8)
    qb, sb = u.quantize_block(x, axis=-1, target_dtype=jnp.int8)
    np.testing.assert_allclose(np.asarray(sa).ravel(),
                               np.asarray(sb).ravel(),
                               rtol=1e-6)
    d = np.asarray(qa).astype(int) - np.asarray(qb).astype(int)
    assert int(np.abs(d).max()) <= 1 and float(
        np.mean(d != 0)
    ) < 0.10, "the two activation-quant paths disagree beyond rounding ties"
