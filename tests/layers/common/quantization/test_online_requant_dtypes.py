"""The online requant path must be correct for EVERY dtype it can emit.

The divisor used to be the literal 448 (e4m3fn's max). That is a
wrong-numbers bug for any other target -- e4m3b11fnuz maxes at 30 and int8
at 127, so a hardcoded 448 would have silently clipped every weight to a
fraction of the representable range and produced a quiet quality loss rather
than a loud failure.

The leaf imports only jax + os, so this runs on a CPU-only jax install.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
LEAF = (ROOT / "tpu_inference" / "layers" / "common" / "quantization" /
        "online_fp8_requant.py")


def _leaf():
    pytest.importorskip("jax", reason="numeric check needs jax")
    spec = importlib.util.spec_from_file_location("_requant_leaf", LEAF)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _weight():
    import jax
    return jax.random.normal(jax.random.PRNGKey(3), (256, 128))


@pytest.mark.parametrize("name", [
    "float8_e4m3fn", "float8_e4m3b11fnuz", "float8_e5m2", "int8",
])
def test_every_dtype_round_trips_within_its_own_resolution(name):
    import jax.numpy as jnp
    r = _leaf()
    dtype = r.ONLINE_QUANT_DTYPES[name]
    w = _weight()

    w_q, scale = r.online_fp8_requant_per_channel(w, dtype=dtype)
    assert w_q.dtype == dtype
    assert w_q.shape == w.shape
    assert scale.shape == (w.shape[1], ), "scale is per-OUTPUT-channel"

    deq = w_q.astype(jnp.float32) * scale[None, :]
    rel = float(jnp.max(jnp.abs(deq - w)) / jnp.max(jnp.abs(w)))
    assert rel < 0.15, (
        f"{name} round-trip relative error {rel:.4f} -- too large to be a "
        f"scaled representation of the weight; the most likely cause is a "
        f"divisor that does not match this dtype's max "
        f"({r.quant_dtype_max(dtype)})")


def test_the_scale_divisor_tracks_the_dtype_not_the_literal_448():
    """The regression guard with teeth: quantizing to a dtype whose max is
    NOT 448 must still use the full target range. If the divisor were
    hardcoded, e4m3b11fnuz (max 30) would use ~1/15th of its range and the
    quantized magnitudes would collapse toward zero."""
    import jax.numpy as jnp
    r = _leaf()
    w = _weight()
    for name in ("float8_e4m3b11fnuz", "int8"):
        dtype = r.ONLINE_QUANT_DTYPES[name]
        dmax = r.quant_dtype_max(dtype)
        w_q, _ = r.online_fp8_requant_per_channel(w, dtype=dtype)
        reach = float(jnp.max(jnp.abs(w_q.astype(jnp.float32))))
        assert reach > 0.5 * dmax, (
            f"{name} only reached {reach} of its max {dmax} -- the scale "
            f"divisor is not tracking the dtype (the hardcoded-448 bug)")


def test_int8_resolves_weights_better_than_e4m3fn():
    """MEASURED, and the reason int8 is wired as a first-class option.

    Per-output-channel int8 carries 8 bits of UNIFORM resolution inside each
    channel's range; e4m3fn spends its 8 bits on an exponent it does not need
    once a per-channel scale has already normalised the range, leaving 3
    mantissa bits. On this weight int8 is ~9x more accurate.

    Combined with v6e doing ~2x bf16 FLOPs in int8 while fp8 is 918 TFLOPs =
    bf16's 918, int8 is the better target on this generation on BOTH axes.
    Asserted loosely (2x) so this documents the direction without becoming a
    brittle numeric pin.
    """
    import jax.numpy as jnp
    r = _leaf()
    w = _weight()

    def rel(dtype):
        w_q, s = r.online_fp8_requant_per_channel(w, dtype=dtype)
        return float(jnp.max(jnp.abs(w_q.astype(jnp.float32) * s[None, :] - w))
                     / jnp.max(jnp.abs(w)))

    assert rel(jnp.int8) < rel(jnp.float8_e4m3fn) / 2.0


def test_default_is_unchanged_and_bad_values_are_rejected_loudly():
    """This is a LEVER, not a behaviour change: unset env keeps e4m3fn."""
    import jax.numpy as jnp
    r = _leaf()
    import os
    saved = os.environ.pop(r.ONLINE_QUANT_DTYPE_ENV, None)
    try:
        assert r.online_quant_dtype() is jnp.float8_e4m3fn
        os.environ[r.ONLINE_QUANT_DTYPE_ENV] = "int8"
        assert r.online_quant_dtype() is jnp.int8
        os.environ[r.ONLINE_QUANT_DTYPE_ENV] = "float8_e4m3b11fnuz"
        assert r.online_quant_dtype() is jnp.float8_e4m3b11fnuz
        # A typo must NOT silently fall back to the default -- that would
        # make a bench arm measure the wrong dtype under the right label.
        os.environ[r.ONLINE_QUANT_DTYPE_ENV] = "fp8"
        with pytest.raises(ValueError, match="not a supported"):
            r.online_quant_dtype()
    finally:
        os.environ.pop(r.ONLINE_QUANT_DTYPE_ENV, None)
        if saved is not None:
            os.environ[r.ONLINE_QUANT_DTYPE_ENV] = saved
