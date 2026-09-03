"""On-the-fly quantized weight requantization for bf16 dense checkpoints.

Leaf module: imports only jax + os, so tests can load it by file path
(bypassing the tpu_inference package __init__ that pulls vllm/torchax) and
verify the numerics on a CPU-only jax install. Issue #158.

This module also owns the ONE definition of "which quantized dtype does the
online path emit", because both online requant paths must answer it the same
way:

  * the flax/nnx lane  -- layers/jax/quantization/fp8.py    (26B-A4B MoE)
  * the vllm/torchax   -- layers/vllm/quantization/fp8.py   (12B dense)

Those are the two model configs under comparison, so a lever that moved only
one of them would make the arms incomparable rather than configurable.
"""
import os
import re

import jax
import jax.numpy as jnp

# e4m3 (float8_e4m3fn) max representable magnitude. Retained as a name
# because it is the historical default and reads in the dequant docs; the
# scale below is derived from the SELECTED dtype, not from this constant.
E4M3_MAX = 448.0

ONLINE_QUANT_DTYPE_ENV = "TPU_ONLINE_QUANT_DTYPE"

# Every dtype the online path can emit. All flow through the same
# amax/dtype_max scale rule below, so adding one here is numerically
# complete rather than partially wired.
ONLINE_QUANT_DTYPES = {
    "float8_e4m3fn": jnp.float8_e4m3fn,
    "float8_e4m3b11fnuz": jnp.float8_e4m3b11fnuz,
    "float8_e5m2": jnp.float8_e5m2,
    "int8": jnp.int8,
}


def _tpu_generation():
    """Best-effort TPU generation, or None when it cannot be determined."""
    try:
        from jax._src import tpu_info as _ti
        for attr in ("generation", "tpu_generation"):
            g = getattr(_ti, attr, None)
            if callable(g):
                return g()
    except Exception:  # noqa: BLE001 -- never fail a load over a probe
        pass
    try:
        kind = getattr(jax.devices()[0], "device_kind", "") or ""
        m = re.search(r"v(\d+)", kind)
        return int(m.group(1)) if m else None
    except Exception:  # noqa: BLE001
        return None


_ANNOUNCED = set()


def _announce(dtype, why):
    """Log the selected dtype once per process. Best-effort: never let a log
    import break a weight load."""
    key = (getattr(dtype, "__name__", str(dtype)), why)
    if key in _ANNOUNCED:
        return
    _ANNOUNCED.add(key)
    try:
        from tpu_inference.logger import init_logger
        init_logger(__name__).info("online quant dtype: %s (%s=%s)", key[0],
                                   ONLINE_QUANT_DTYPE_ENV, why)
    except Exception:  # noqa: BLE001
        pass


def online_quant_dtype():
    """Which quantized dtype the online requant paths emit.

    Defaults to `float8_e4m3fn` -- the historical behaviour -- so this is a
    benchmark LEVER, not a silent change to paths that have barely run. Set
    TPU_ONLINE_QUANT_DTYPE to a key of ONLINE_QUANT_DTYPES, or to "auto" to
    follow what this TPU generation ingests natively.

    "auto" exists because jax/_src/tpu_info.py gates its native-matmul dtype
    table on `generation < 7`: pre-gen-7 lists (F8E5M2, F8E4M3B11FNUZ),
    gen-7+ lists (F8E5M2, F8E4M3FN). So on a v6e the DEFAULT e4m3fn is the
    one fp8 type the hardware must cast.

    On v6e, fp8 is a BANDWIDTH play and not a FLOPs one -- Google's own
    table gives v6e fp8 918 TFLOPs, equal to its bf16 918, while int8 is
    ~2x. That is why int8 is a first-class option here rather than an
    afterthought. See docs/fp8-all-modalities-design-2026-09-01.md.
    """
    # Announce the SELECTED dtype once. The existing engagement marker only
    # proves the online path ran, not WHICH dtype it emitted -- so without
    # this an int8-labelled arm that silently fell back to e4m3fn is
    # indistinguishable from a real int8 arm in the logs. Bank deltas
    # discriminate too, but a log line is the cheap direct proof.
    want = os.environ.get(ONLINE_QUANT_DTYPE_ENV, "").strip()
    if want and want != "auto":
        if want not in ONLINE_QUANT_DTYPES:
            raise ValueError(
                f"{ONLINE_QUANT_DTYPE_ENV}={want!r} is not a supported online "
                f"quantization dtype. Choose one of "
                f"{sorted(ONLINE_QUANT_DTYPES)} or 'auto'.")
        _announce(ONLINE_QUANT_DTYPES[want], want)
        return ONLINE_QUANT_DTYPES[want]
    if want == "auto":
        gen = _tpu_generation()
        if gen is not None and gen < 7:
            _announce(jnp.float8_e4m3b11fnuz, "auto/gen<7")
            return jnp.float8_e4m3b11fnuz
    _announce(jnp.float8_e4m3fn, want or "default")
    return jnp.float8_e4m3fn


def quant_dtype_max(dtype):
    """Largest representable magnitude of `dtype` (works for int and float)."""
    if jnp.issubdtype(dtype, jnp.integer):
        return float(jnp.iinfo(dtype).max)
    return float(jnp.finfo(dtype).max)


def online_fp8_requant_per_channel(weight_in_out, dtype=None):
    """bf16 dense weight [in, out] -> (quantized [in, out], f32 scale [out]).

    Per-output-channel: scale = (max |w| over the input axis) / dtype_max, so
    the largest magnitude in each output column maps to the top of the target
    range. The dequant is `w_q.astype(f32) * scale[None, :]`.

    `dtype` defaults to `online_quant_dtype()`. The divisor is derived from
    the SELECTED dtype -- hardcoding 448 here would silently clip every
    weight for any dtype whose max is not e4m3fn's (e4m3b11fnuz maxes at 30,
    int8 at 127), which is a wrong-numbers bug rather than a loud one.
    """
    dtype = online_quant_dtype() if dtype is None else dtype
    w = weight_in_out.astype(jnp.float32)
    amax = jnp.max(jnp.abs(w), axis=0, keepdims=True)
    scale = jnp.maximum(amax / quant_dtype_max(dtype), jnp.float32(1e-12))
    scaled = w / scale
    # Integers round to nearest; floats truncate toward the representable
    # grid on cast. Rounding an int8 target is worth ~0.5 LSB of error.
    if jnp.issubdtype(dtype, jnp.integer):
        info = jnp.iinfo(dtype)
        scaled = jnp.clip(jnp.round(scaled), info.min, info.max)
    w_q = scaled.astype(dtype)
    return w_q, jnp.squeeze(scale, axis=0)


# Dtypes the TORCHAX path can carry. That path round-trips the quantized array
# back through `torch_view()` -> `torchax.ops.mappings.j2t_dtype`, which needs a
# real torch dtype. `float8_e4m3b11fnuz` is a JAX/ml_dtypes type with NO torch
# equivalent (torch has e4m3fn / e4m3fnuz / e5m2 / e5m2fnuz, not e4m3b11fnuz),
# so selecting it on that path dies at WEIGHT LOAD with
#   RuntimeError: Attempting to convert unknown type: float8_e4m3b11fnuz to torch type
# MEASURED on v6e 2026-09-01: the eval-12b-q-e4m3b11 arm crashlooped ~110s in,
# inside vllm/quantization/fp8.py process_weights_after_loading.
#
# The flax/nnx path never converts to torch, so it is NOT restricted by this.
# Hence a per-path check rather than removing the dtype outright: it stays a
# legitimate option for the 26B, where v6e ingests it natively.
TORCHAX_REPRESENTABLE = frozenset({"float8_e4m3fn", "float8_e5m2", "int8"})


def assert_torchax_representable(dtype):
    """Refuse a JAX-only dtype on the torchax path, EARLY and legibly.

    Called at quant-method selection so the failure lands at engine init with
    an actionable message, instead of ~110s later as a torchax internal error
    whose traceback names j2t_dtype and not the env var the operator set.
    """
    name = getattr(dtype, "__name__", str(dtype))
    if name not in TORCHAX_REPRESENTABLE:
        raise ValueError(
            f"{ONLINE_QUANT_DTYPE_ENV}={name!r} cannot be used on the "
            f"vLLM/torchax path: torch has no {name} dtype, so converting the "
            f"quantized weight back to a torch view fails at weight load. "
            f"Supported here: {sorted(TORCHAX_REPRESENTABLE)}. "
            f"({name} IS valid on the flax_nnx path, which never converts to "
            f"torch.)")
    return dtype
