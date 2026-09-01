"""A JAX-only dtype must be refused on the torchax path, EARLY.

MEASURED on v6e 2026-09-01: `eval-12b-q-e4m3b11` crashlooped ~110s into boot
with

    RuntimeError: Attempting to convert unknown type: float8_e4m3b11fnuz
                  to torch type

from `torchax.ops.mappings.j2t_dtype`, reached via `torch_view()` in
`vllm/quantization/fp8.py process_weights_after_loading`. `float8_e4m3b11fnuz`
is a JAX/ml_dtypes type with no torch equivalent (torch has e4m3fn, e4m3fnuz,
e5m2, e5m2fnuz -- not e4m3b11fnuz).

Two things make that failure expensive out of proportion to the bug:
  * it costs a full model load to discover, and
  * the traceback names j2t_dtype and NEVER names TPU_ONLINE_QUANT_DTYPE, the
    thing the operator actually set.

The dtype stays legal on the flax_nnx path, which never converts to torch --
so this is a per-PATH restriction, not a removal.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
LEAF = (ROOT / "tpu_inference" / "layers" / "common" / "quantization" /
        "online_fp8_requant.py")


def _leaf():
    pytest.importorskip("jax", reason="needs jax dtypes")
    spec = importlib.util.spec_from_file_location("_rq", LEAF)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_e4m3b11fnuz_is_refused_on_the_torchax_path():
    import jax.numpy as jnp
    r = _leaf()
    with pytest.raises(ValueError) as e:
        r.assert_torchax_representable(jnp.float8_e4m3b11fnuz)
    msg = str(e.value)
    # The message must name the ENV VAR the operator set. The real failure did
    # not, which is the half of this bug that cost the most time.
    assert r.ONLINE_QUANT_DTYPE_ENV in msg, (
        "the refusal must name TPU_ONLINE_QUANT_DTYPE -- the torchax error it "
        "replaces named only j2t_dtype, which is not actionable")
    assert "flax" in msg.lower(), (
        "the refusal must say the dtype IS valid on the flax path, or it reads "
        "as 'this dtype is broken' rather than 'wrong path'")


@pytest.mark.parametrize("name", ["float8_e4m3fn", "float8_e5m2", "int8"])
def test_torch_representable_dtypes_are_allowed(name):
    r = _leaf()
    dt = r.ONLINE_QUANT_DTYPES[name]
    assert r.assert_torchax_representable(dt) is dt


def test_the_allowed_set_is_a_strict_subset_of_the_offered_set():
    """If someone adds a dtype to ONLINE_QUANT_DTYPES they must decide whether
    torch can carry it. A guard that allowed everything would be vacuous."""
    r = _leaf()
    offered = set(r.ONLINE_QUANT_DTYPES)
    allowed = set(r.TORCHAX_REPRESENTABLE)
    assert allowed < offered, (
        f"TORCHAX_REPRESENTABLE {allowed} must be a STRICT subset of "
        f"ONLINE_QUANT_DTYPES {offered}; if they are equal the guard cannot "
        f"refuse anything")
    assert "float8_e4m3b11fnuz" in offered - allowed
