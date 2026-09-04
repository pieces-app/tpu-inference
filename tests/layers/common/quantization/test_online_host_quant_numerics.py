"""The host-side quantization must be BIT-IDENTICAL to the device path.

`place_host_quantized` / `quantize_on_host` (online_host_quant, the loader
hook) run quantize_tensor ITSELF under jax.jit on the host device. The
on-device path they replace -- Fp8OnlineLinearMethod.process_weights_after_
loading, now the fallback -- runs the same function eagerly on the placed
kernel. Same function, same float32 abs-max scale, same round-to-nearest for
integer targets (PR #45). The only thing that can differ is jit vs eager,
and this suite pins that to ZERO bits: codes and scales, every online dtype,
a whole zero output column (scale 0 -> scale_inv inf -> code 0), and a 3-D
einsum kernel flattened to the 2-D layout the way the device path did it.

Leaves are imported through a bare `tpu_inference` package stub (the real
package __init__ pulls vllm), as tests/kernels/test_int8_quant_scale_precision.py
does. Placement is the identity here; the loader-level footprint test
(tests/layers/jax/quantization/test_online_quant_load_peak.py) runs the real
assign_and_shard_param with real nnx.Params on a real multi-device mesh.
"""
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
LEAF = ROOT / "tpu_inference" / "layers" / "common" / "quantization" / "online_host_quant.py"
DTYPES = ["int8", "float8_e4m3fn", "float8_e4m3b11fnuz", "float8_e5m2"]


def _leaves():
    pytest.importorskip("jax")
    for k in [k for k in sys.modules if k.startswith("tpu_inference")]:
        del sys.modules[k]
    pkg = types.ModuleType("tpu_inference")
    pkg.__path__ = [str(ROOT / "tpu_inference")]
    sys.modules["tpu_inference"] = pkg
    import tpu_inference.layers.common.quantization as q
    import tpu_inference.layers.common.quantization.online_host_quant as h
    return q, h


class _Param:
    """The three Variable calls the leaf makes; a real nnx.Param has the
    same surface (and the footprint test uses the real one)."""

    def __init__(self):
        self.md = {}
        self.value = None

    def get_metadata(self):
        return dict(self.md)

    def set_metadata(self, k, v):
        self.md[k] = v

    def set_value(self, v):
        self.value = v


# A mesh the host array is NOT on (the leaf only reads `.devices`).
_NOWHERE = types.SimpleNamespace(devices=np.empty((0, ), dtype=object))
_IDENTITY_PUT = lambda x, spec: x  # noqa: E731


def _bf16_weight(shape, seed):
    import jax
    import jax.numpy as jnp
    w = (jax.random.normal(jax.random.PRNGKey(seed), shape) * 2.0).astype(
        jnp.bfloat16)
    return w.at[...,
                3].set(0)  # a whole zero OUTPUT column: the scale-0 branch


def _bits(a):
    a = np.asarray(a)
    return a.view(np.uint8) if a.dtype.itemsize == 1 else a.view(np.uint32)


@pytest.mark.parametrize("name", DTYPES)
def test_host_path_is_bit_identical_to_the_eager_device_path(name):
    import jax.numpy as jnp
    q, h = _leaves()
    dtype = getattr(jnp, name)
    w = _bf16_weight((256, 384), 1)
    # The pre-fix on-device call, verbatim (fp8.py: quantize_tensor(dtype, w2d, axis=0)).
    e_q, e_s = q.quantize_tensor(dtype, w, axis=0)
    assert float(e_s[3]) == 0.0 and int(
        np.asarray(e_q.astype(jnp.float32))[0, 3]
    ) == 0, (
        "the zero column did not take the scale-0 branch; the test input is wrong"
    )

    p = _Param()
    h.request_host_quant(
        p, h.HostQuantRequest(dtype, (256, 384), (None, None), (None, )))
    assert h.place_host_quantized(p, w, mesh=_NOWHERE, put=_IDENTITY_PUT)
    w_s = h.adopt_host_quant_scale(p)

    assert p.value.dtype == dtype and w_s.dtype == jnp.float32
    assert p.value.shape == e_q.shape and w_s.shape == e_s.shape
    assert np.array_equal(
        _bits(p.value),
        _bits(e_q)), f"{name}: codes differ between the host and device paths"
    assert np.array_equal(
        _bits(w_s),
        _bits(e_s)), f"{name}: scales differ between the host and device paths"


def test_three_d_kernel_is_flattened_exactly_as_the_device_path_did():
    """o_proj is JaxEinsum('TNH,NHD->TD') with a [N, H, D] kernel; the device
    path did weight.reshape(prod(in_features), -1) before quantizing."""
    import jax.numpy as jnp
    q, h = _leaves()
    w = _bf16_weight((4, 32, 64), 2)
    e_q, e_s = q.quantize_tensor(jnp.int8, w.reshape(128, 64), axis=0)
    p = _Param()
    h.request_host_quant(
        p, h.HostQuantRequest(jnp.int8, (128, 64), ("model", None), (None, )))
    assert h.place_host_quantized(p, w, mesh=_NOWHERE, put=_IDENTITY_PUT)
    assert p.value.shape == (128, 64)
    assert np.array_equal(_bits(p.value), _bits(e_q))
    assert np.array_equal(_bits(h.adopt_host_quant_scale(p)), _bits(e_s))


def test_place_host_quantized_routes_through_the_jitted_primitive(monkeypatch):
    """AUDIT 2026-09-03: this replaces a pair of source-string assertions
    (`"jax.jit(quantize_tensor" in src` and `"def quantize_tensor" not in src`)
    that could not fail. The first proved only that the LINE exists, not that
    it is on the call path; the second is a negative that holds for almost any
    file. Verified vacuous: with a second, bf16-reduced implementation inlined
    into `place_host_quantized` and the jitted `_quantize` left defined but
    unused, both assertions still passed.

    `_quantize` is a module global, so intercepting it proves the routing.
    """
    import jax.numpy as jnp
    q, h = _leaves()
    seen = []
    real = h._quantize
    monkeypatch.setattr(h, "_quantize", lambda *a:
                        (seen.append(a), real(*a))[1])
    p = _Param()
    w = _bf16_weight((64, 32), 9)
    h.request_host_quant(
        p, h.HostQuantRequest(jnp.int8, (64, 32), (None, None), (None, )))
    assert h.place_host_quantized(p, w, mesh=_NOWHERE, put=_IDENTITY_PUT)
    assert len(seen) == 1, (
        f"place_host_quantized called the jitted primitive {len(seen)} times; "
        "a second implementation inlined here would drift from the device path"
    )
    assert seen[0][0] is jnp.int8, (
        f"the jitted primitive was called with {seen[0][0]!r}, not the "
        "requested dtype")


def test_quantize_on_host_agrees_with_the_eager_device_path():
    """`quantize_on_host` is named in this file's docstring and in the leaf's
    as part of the covered surface, but AUDIT 2026-09-03 found nothing called
    it -- not this suite, not production (weight_utils calls
    place_host_quantized; fp8.py calls request_host_quant/adopt). Cover it
    directly rather than leave a named-but-unexercised entry point.
    """
    import jax.numpy as jnp
    q, h = _leaves()
    w = _bf16_weight((128, 64), 11)
    for dtype in (jnp.int8, jnp.float8_e4m3fn):
        got_q, got_s = h.quantize_on_host(dtype, w)
        exp_q, exp_s = q.quantize_tensor(dtype, w, 0, None)
        assert np.array_equal(_bits(got_q), _bits(exp_q)), dtype
        assert np.array_equal(_bits(got_s), _bits(exp_s)), dtype


def test_specs_reach_put_verbatim_weight_then_scale():
    import jax.numpy as jnp
    q, h = _leaves()
    seen = []
    p = _Param()
    h.request_host_quant(
        p, h.HostQuantRequest(jnp.int8, (64, 32), (None, "model"),
                              ("model", )))
    assert h.place_host_quantized(p,
                                  _bf16_weight((64, 32), 3),
                                  mesh=_NOWHERE,
                                  put=lambda x, spec: (seen.append(
                                      (x.shape, spec)), x)[1])
    assert seen == [((64, 32), (None, "model")), ((32, ), ("model", ))], seen


def test_the_scale_is_parked_once_and_adopted_once():
    import jax.numpy as jnp
    q, h = _leaves()
    p = _Param()
    h.request_host_quant(
        p, h.HostQuantRequest(jnp.int8, (64, 32), (None, None), (None, )))
    assert h.place_host_quantized(p,
                                  _bf16_weight((64, 32), 4),
                                  mesh=_NOWHERE,
                                  put=_IDENTITY_PUT)
    first = h.adopt_host_quant_scale(p)
    assert first is not None and first.shape == (32, )
    assert h.adopt_host_quant_scale(
        p) is None, "adoption must clear the parked reference"
    assert p.md[h.HOST_QUANT_SCALE] is None


def test_without_a_request_nothing_is_touched():
    q, h = _leaves()
    p = _Param()
    assert not h.place_host_quantized(
        p, _bf16_weight((64, 32), 5), mesh=_NOWHERE, put=_IDENTITY_PUT)
    assert p.value is None and p.md == {}
    assert h.adopt_host_quant_scale(p) is None


def test_a_weight_already_on_the_mesh_is_left_to_the_device_path():
    """pathways_dummy builds its dummies ON the TPU; there is nothing to keep
    off the mesh, so the on-device requant stays the path for it."""
    import jax.numpy as jnp
    q, h = _leaves()
    w = _bf16_weight((64, 32), 6)
    on_mesh = types.SimpleNamespace(devices=np.array(
        sorted(w.devices(), key=lambda d: d.id), dtype=object))
    p = _Param()
    h.request_host_quant(
        p, h.HostQuantRequest(jnp.int8, (64, 32), (None, None), (None, )))
    assert not h.place_host_quantized(p, w, mesh=on_mesh, put=_IDENTITY_PUT)
    assert p.value is None and h.adopt_host_quant_scale(p) is None
