"""Numerics of the online fp8 (e4m3) dense requant (issue #158).

Loads the leaf module `layers/common/quantization/online_fp8_requant.py` BY
FILE PATH so it runs on a CPU-only jax install without importing the
tpu_inference package (which pulls vllm/torchax). Asserts the post-transpose
[in, out] per-output-channel scale is amax/448 within ~1 ULP and the dequant
round-trip is within e4m3 tolerance.
`pytest tests/layers/vllm/test_fp8_online_requant.py` (needs jax; skips
without it).

Negative control (watched via fork_gate / by perturbing E4M3_MAX): a wrong
divisor makes the scale assertion fail.
"""
import importlib.util
from pathlib import Path

import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LEAF = (REPO_ROOT / "tpu_inference" / "layers" / "common" / "quantization" /
        "online_fp8_requant.py")


def _load_leaf():
    spec = importlib.util.spec_from_file_location("online_fp8_requant", LEAF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scale_is_amax_over_448_per_channel():
    m = _load_leaf()
    key = jax.random.PRNGKey(0)
    w = (jax.random.normal(key, (128, 64)) * 3.0).astype(jnp.bfloat16)
    _, scale = m.online_fp8_requant_per_channel(w)
    amax = jnp.max(jnp.abs(w.astype(jnp.float32)), axis=0)
    assert scale.shape == (64, )
    assert float(jnp.max(jnp.abs(scale - amax / 448.0))) < 1e-4


def test_round_trip_within_e4m3_tolerance():
    """AUDIT 2026-09-03: the old comment claimed "a per-tensor or wrong-divisor
    scheme blows past this". MEASURED on the old Gaussian input with the old
    `rel < 0.25` bound, all three defects PASSED:
        shipped (per-channel, dtype-derived divisor)  0.0635
        per-TENSOR scale                              0.0815   <- passed
        wrong divisor 127 (int8 max on an e4m3 target) 0.2486  <- passed
        wrong divisor 57344 (weights overflow to NaN) -inf     <- passed
    The last is the worst: NaN weights make the ratio evaluate to -inf, which
    satisfies any upper bound. Two changes fix all three -- one low-amplitude
    column (which a per-tensor scale collapses to zero: 0.8224) and an
    explicit finiteness check -- plus a bound at the measured value's ~1.25x.
    """
    m = _load_leaf()
    key = jax.random.PRNGKey(1)
    w = (jax.random.normal(key, (256, 32)) * 5.0).astype(jnp.bfloat16)
    # A low-amplitude output column: per-CHANNEL scaling keeps its resolution,
    # a per-TENSOR scale quantizes it to zero.
    w = w.at[:, 0].multiply(1e-3)
    w_fp8, scale = m.online_fp8_requant_per_channel(w)
    rt = w_fp8.astype(jnp.float32) * scale[None, :]
    ref = w.astype(jnp.float32)
    assert bool(jnp.all(jnp.isfinite(rt))), (
        "quantized weights are not finite -- a divisor larger than the target "
        "dtype's max overflows e4m3 to NaN, and the ratio below then reads "
        "-inf, which passes any upper bound")
    rel = float(jnp.max(jnp.abs(rt - ref) / (jnp.abs(ref) + 1e-6)))
    assert rel < 0.08, (
        f"e4m3 per-channel round-trip too lossy: {rel} (this code 0.0635; "
        f"per-tensor 0.8224; divisor-127 0.2486)")


def test_the_divisor_tracks_the_selected_dtype_not_a_hardcoded_448():
    """AUDIT 2026-09-03: this replaces `test_no_empty_scale_ever_created`,
    3 of whose 4 clauses could not fail.
      * `"torch.empty" not in src` -- the leaf is jax-only and never imports
        torch, so the substring cannot appear.
      * `"amax" in src` -- MEASURED: renaming the code's `amax` variable on
        both lines that use it still passed, because a module comment
        mentions "amax/dtype_max".
      * `"quant_dtype_max" in src or "/ E4M3_MAX" in src` -- MEASURED:
        replacing `amax / quant_dtype_max(dtype)` with a hardcoded
        `amax / 448.0`, i.e. reinstating the ti #25 bug that clipped every
        weight for e4m3b11fnuz (max 30) and int8 (max 127), still passed,
        because `def quant_dtype_max(dtype):` remains defined in the file.
    Only the `jnp.empty`/`np.empty` clause could fire; it is kept below.
    Assert the divisor BEHAVIOURALLY instead, on every dtype that matters.
    """
    m = _load_leaf()
    w = (jax.random.normal(jax.random.PRNGKey(2),
                           (64, 8)) * 3.0).astype(jnp.bfloat16)
    amax = jnp.max(jnp.abs(w.astype(jnp.float32)), axis=0)
    for dt in (jnp.float8_e4m3fn, jnp.float8_e4m3b11fnuz, jnp.int8):
        _, scale = m.online_fp8_requant_per_channel(w, dtype=dt)
        expect = jnp.maximum(amax / m.quant_dtype_max(dt), jnp.float32(1e-12))
        assert float(jnp.max(jnp.abs(scale - expect))) < 1e-6, (
            f"{dt.__name__}: the divisor is not the target dtype's max. A "
            f"hardcoded 448 clips every weight for e4m3b11fnuz (max 30) and "
            f"int8 (max 127) -- a wrong-numbers bug rather than a loud one.")
    src = LEAF.read_text()
    assert "jnp.empty" not in src and "np.empty" not in src, (
        "no uninitialized scale may be allocated on any path: the garbage "
        "trap the fail-closed guard names is a scale the loader cannot fill")
