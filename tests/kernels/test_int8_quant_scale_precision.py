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
    import sys, types
    pytest.importorskip("jax")
    for k in [k for k in sys.modules if k.startswith("tpu_inference")]:
        del sys.modules[k]
    pkg = types.ModuleType("tpu_inference"); pkg.__path__ = [str(ROOT / "tpu_inference")]
    sys.modules["tpu_inference"] = pkg
    lg = types.ModuleType("tpu_inference.logger")
    lg.init_logger = lambda *a, **k: type("L", (), {"__getattr__": lambda s, n: (lambda *a, **k: None)})()
    sys.modules["tpu_inference.logger"] = lg
    for name, d in (("tpu_inference.kernels", ROOT / "tpu_inference" / "kernels"),
                    ("tpu_inference.kernels.quantized_matmul", ROOT / "tpu_inference" / "kernels" / "quantized_matmul")):
        m = types.ModuleType(name); m.__path__ = [str(d)]; sys.modules[name] = m
    spec = importlib.util.spec_from_file_location("tpu_inference.kernels.quantized_matmul.util", UTIL)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m; spec.loader.exec_module(m)
    return m


def _f32_reference(x, dtype_max=127.0):
    xf = np.asarray(x, dtype=np.float32)
    amax = np.max(np.abs(xf), axis=-1, keepdims=True)
    scale = np.where(amax == 0, 1.0, amax / dtype_max).astype(np.float32)
    return np.round(xf / scale).astype(np.int8), scale


def test_int8_scale_is_float32_for_bf16_activations():
    import jax, jax.numpy as jnp
    u = _util()
    x = jax.random.normal(jax.random.PRNGKey(0), (16, 256)).astype(jnp.bfloat16)
    q, scale = u.quantize_block(x, axis=-1, target_dtype=jnp.int8)
    assert scale.dtype == jnp.float32, f"int8 scale computed in {scale.dtype}, fp8's is float32"
    assert q.dtype == jnp.int8


def test_int8_codes_track_the_f32_reference():
    """f32 path: scale matches numpy to 1e-6 and codes differ from the numpy
    reference only by round-half ties (|diff| <= 1, well under 1% of codes).
    The old bf16 path: 7.6% of codes differ (measured 2026-09-02, [256x3840]
    N(0,1)), 5.3% more RMS error. The bound below sits between the two."""
    import jax, jax.numpy as jnp
    u = _util()
    x = jax.random.normal(jax.random.PRNGKey(1), (32, 512)).astype(jnp.bfloat16)
    q, scale = u.quantize_block(x, axis=-1, target_dtype=jnp.int8)
    q_ref, s_ref = _f32_reference(np.asarray(x.astype(jnp.float32)))
    np.testing.assert_allclose(np.asarray(scale), s_ref, rtol=1e-6)
    d = np.asarray(q).astype(int) - q_ref.astype(int)
    frac = float(np.mean(d != 0))
    assert int(np.abs(d).max()) <= 1, "codes off by more than one level: not a tie-rounding difference"
    assert frac < 0.01, f"{100*frac:.2f}% of int8 codes differ from the f32 reference (bf16 arithmetic leaked in; old path: 7.6%)"


def test_fp8_branch_unchanged():
    import jax, jax.numpy as jnp
    u = _util()
    x = jax.random.normal(jax.random.PRNGKey(2), (8, 128)).astype(jnp.bfloat16)
    q, scale = u.quantize_block(x, axis=-1, target_dtype=jnp.float8_e4m3fn)
    assert scale.dtype == jnp.float32 and q.dtype == jnp.float8_e4m3fn
