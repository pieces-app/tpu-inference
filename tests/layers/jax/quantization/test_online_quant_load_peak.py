"""The loader's peak device footprint for an online-quant model, MEASURED
on the CPU backend, before and after the host-side quantization.

MEASURED 2026-09-02 23:09Z on one v6e (eval-12b-native-int8-w8a16): the
native 12B online-int8 arm died at LOAD with RESOURCE_EXHAUSTED asking for
the 450.00 MiB float32 temporary of one fused gate_up_proj kernel, with
353.97 MiB free on a 31.24 GiB chip. The chip was full because, before this
fix, the loader placed every kernel in bf16, requanted it ON the mesh with
two f32 temporaries alive, deferred the merged kernels' requant to after the
whole stream, and pinned every replaced bf16 Param in load_weights'
params_dict. Peak = bf16 model + int8 copies + f32 temporaries.

This test re-creates that loader sequence on a small synthetic model, using
the REAL assign_and_shard_param / shard_put (weight_utils, imported with
torch/vllm replaced by empty shells -- only the checkpoint-reading paths
need them) and REAL nnx.Params built the way the abstract model is
(nnx.eval_shape under the model mesh), on THREE CPU devices: device 0 is the
host, devices 1-2 form a (1, 2) 'data' x 'model' mesh so the 2-D placement
specs are exercised through general_device_put, not the single-device
shortcut. Bytes are physical: for every array in jax.live_arrays(), its
shard shape times the number of mesh devices it occupies. The old requant is re-enacted line for line from the
pre-fix Fp8OnlineLinearMethod.process_weights_after_loading (that class
cannot be imported without vllm; its shape is pinned by
test_online_host_quant_wiring.py) with jnp.clip hooked to sample the mesh
at the moment the two f32 temporaries are both alive.

What is NOT visible here: the float32 scratch inside the host-side jitted
quantize_tensor is XLA-internal, not a jax.Array. The host bound below is
what Python holds: one kernel's bf16 staging copy plus its int8 and scale
before they are placed.

The measurement runs in a subprocess so the device count is set before jax
initialises, whatever the rest of the pytest session did.
"""
import functools
import json
import os
import pathlib
import subprocess
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
N_DEVICES = 3
N_BLOCKS = 3
# name, kernel shape as the model builds it, 2-D [in, out], kernel spec, 2-D weight spec, scale spec.
# gate_up FIRST and largest: the deferred merged kernel is where the live load died.
LAYERS = [
    ("gate_up", (64, 512), (64, 512), (None, "model"), (None, "model"),
     ("model", )),
    ("down", (256, 64), (256, 64), ("model", None), ("model", None), (None, )),
    ("q", (64, 4, 32), (64, 128), (None, "model", None), (None, "model"),
     ("model", )),
    ("o", (4, 32, 64), (128, 64), ("model", None, None), ("model", None),
     (None, )),
]


def _stub_and_import():
    """The REAL weight_utils (assign_and_shard_param, shard_put) and leaves,
    with torch/torchax/safetensors/vllm as empty shells."""
    for k in [
            k for k in sys.modules
            if k.split(".")[0] in ("tpu_inference", "torch", "torchax",
                                   "safetensors", "vllm")
    ]:
        del sys.modules[k]

    def mod(name, **attrs):
        m = types.ModuleType(name)
        m.__dict__.update(attrs)
        sys.modules[name] = m
        return m

    torch = mod("torch",
                Tensor=type("Tensor", (), {}),
                uint8="uint8",
                uint16="uint16",
                uint32="uint32")
    torch.nn = types.SimpleNamespace(Module=object)
    mod("torchax")
    mod("safetensors", safe_open=None)
    mod("vllm").__path__ = []
    mod("vllm.config",
        ModelConfig=type("ModelConfig", (), {}),
        VllmConfig=type("VllmConfig", (), {}),
        get_current_vllm_config=lambda: None)
    mod("vllm.model_executor").__path__ = []
    mod("vllm.model_executor.model_loader",
        register_model_loader=lambda name: (lambda cls: cls)).__path__ = []
    mod("vllm.model_executor.model_loader.dummy_loader",
        DummyModelLoader=type("DummyModelLoader", (), {}))
    mod("vllm.model_executor.models").__path__ = []
    mod("vllm.model_executor.models.utils",
        AutoWeightsLoader=type("AutoWeightsLoader", (), {}))
    mod("tpu_inference").__path__ = [str(ROOT / "tpu_inference")]
    mod("tpu_inference.utils",
        t2j=lambda t, use_dlpack=False: t,
        get_mesh_shape_product=lambda mesh, axis: 1)
    mod("tpu_inference.logger",
        init_logger=lambda *a, **k: type("L", (), {
            "__getattr__":
            lambda s, n: (lambda *a, **k: None)
        })())
    mod("tpu_inference.models.jax.utils.file_utils")
    import tpu_inference.layers.common.quantization as q
    import tpu_inference.layers.common.quantization.online_host_quant as h
    import tpu_inference.models.jax.utils.weight_utils as wu
    return wu, q, h


def _measure():
    import gc

    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import nnx
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P

    wu, q, h = _stub_and_import()
    devs = jax.devices()
    assert len(devs) >= N_DEVICES, devs
    host, mesh_devs = devs[0], list(devs[1:3])
    host_mesh = Mesh([host], ("cpu", ))
    mesh = Mesh(np.array(mesh_devs).reshape(1, 2), ("data", "model"))

    def phys_of(a, on):
        """Physical bytes `a` occupies on the devices in `on`, from its
        sharding's shard shape. Never via addressable_shards: that
        materialises per-shard Array views which stay registered in
        jax.live_arrays() and would be counted again as separate arrays."""
        devs = a.sharding.device_set & on
        return int(np.prod(a.sharding.shard_shape(
            a.shape))) * a.dtype.itemsize * len(devs)

    MESH_SET, HOST_SET = set(mesh_devs), {host}

    def phys(on):
        on = set(on)
        return sum(phys_of(a, on) for a in jax.live_arrays())

    def shards(a):
        return phys_of(a, MESH_SET | HOST_SET)

    class Block(nnx.Module):

        def __init__(self):
            for name, shape, _, spec, _, _ in LAYERS:
                setattr(
                    self, name,
                    nnx.Param(jnp.zeros(shape, jnp.bfloat16),
                              out_sharding=spec))

    class Model(nnx.Module):

        def __init__(self):
            self.blocks = nnx.List([Block() for _ in range(N_BLOCKS)])

    def abstract_model():
        with jax.set_mesh(mesh):
            return nnx.eval_shape(
                Model)  # ShapeDtypeStruct params, as model_loader builds them

    def checkpoint(li, i):
        """The checkpoint tensor as jax_array_from_reshaped_torch delivers it: bf16 on the host mesh."""
        shape = LAYERS[li][1]
        rng = np.random.default_rng(1000 * li + i)
        with jax.set_mesh(host_mesh):
            return jnp.asarray(rng.standard_normal(
                shape, dtype=np.float32)).astype(jnp.bfloat16)

    def params(model):
        for i, block in enumerate(model.blocks):
            for li, (name, shape, k2d, _, wspec, sspec) in enumerate(LAYERS):
                yield f"{i}.{name}", li, i, k2d, wspec, sspec, block, name

    n_largest = int(np.prod(LAYERS[0][1]))
    out = {
        "n_largest": n_largest,
        "bf16_largest": 2 * n_largest,
        "int8_largest": n_largest,
        "scale_largest": 4 * LAYERS[0][2][1]
    }

    # ------------------------------------------------------------- BEFORE
    # 1. the stream: every kernel placed in bf16 (real assign_and_shard_param, plain path).
    model = abstract_model()
    params_dict, scales, old = {}, {}, {}
    base = phys(mesh_devs)
    bf16_phys = 0
    for key, li, i, k2d, wspec, sspec, block, name in list(params(model)):
        param = getattr(block, name)
        w = checkpoint(li, i)
        with jax.set_mesh(mesh):
            wu.assign_and_shard_param(param, w, key, mesh=mesh)
        del w
        assert param[...].dtype == jnp.bfloat16
        bf16_phys += shards(param[...])
        params_dict[
            key] = param  # JaxAutoWeightsLoader.load_weights: params_dict = dict(named_parameters())
    after_stream = phys(mesh_devs) - base
    # 2. the requant, ON the mesh, as the pre-fix process_weights_after_loading did it.
    peak_clip = 0
    orig_clip = jnp.clip

    def clip_hook(*a, **k):
        nonlocal peak_clip
        peak_clip = max(peak_clip, phys(mesh_devs) - base)
        return orig_clip(*a, **k)

    jnp.clip = clip_hook
    old_q_phys = 0
    peak = after_stream
    for key, li, i, k2d, wspec, sspec, block, name in list(params(model)):
        param = getattr(block, name)
        with jax.set_mesh(mesh):
            weight = param[...]
            src_sharding = getattr(weight, "sharding", None)
            w2d = weight.reshape(k2d)
            w_q, w_s = q.quantize_tensor(jnp.int8, w2d, axis=0)
            try:
                w_q = jax.device_put(w_q, src_sharding)
            except (ValueError, TypeError):
                pass
        delattr(block, name)
        setattr(block, name,
                nnx.Param(w_q))  # the old Param lives on in params_dict
        scales[key] = nnx.Param(w_s)
        old_q_phys += shards(w_q) + shards(w_s)
        del weight, w2d, w_q, w_s
        peak = max(peak, phys(mesh_devs) - base)
    jnp.clip = orig_clip
    out["before"] = {
        "after_stream": after_stream,
        "peak_at_clip": peak_clip,
        "peak": max(peak, peak_clip),
        "end": phys(mesh_devs) - base,
        "bf16": bf16_phys,
        "quantized": old_q_phys
    }
    for key, li, i, k2d, wspec, sspec, block, name in list(params(model)):
        old[key] = (np.asarray(getattr(block, name)[...]).tobytes(),
                    np.asarray(scales[key][...]).tobytes())
    del model, params_dict, scales, param
    gc.collect()

    # -------------------------------------------------------------- AFTER
    model = abstract_model()
    for key, li, i, k2d, wspec, sspec, block, name in list(params(model)):
        h.request_host_quant(getattr(block, name),
                             h.HostQuantRequest(jnp.int8, k2d, wspec, sspec))
    params_dict, scales, new = {}, {}, {}
    base, base_host = phys(mesh_devs), phys([host])
    host_peak = 0
    orig_put = wu.shard_put

    def put_hook(x, spec, mesh=None):
        nonlocal host_peak
        host_peak = max(host_peak, phys([host]) - base_host)
        return orig_put(x, spec, mesh=mesh)

    wu.shard_put = put_hook
    peak = 0
    new_q_phys = 0
    for key, li, i, k2d, wspec, sspec, block, name in list(params(model)):
        param = getattr(block, name)
        params_dict[key] = param
        w = checkpoint(li, i)
        with jax.set_mesh(mesh):
            wu.assign_and_shard_param(param, w, key,
                                      mesh=mesh)  # REAL: host path
        del w
        assert param.get_metadata()["_is_loaded"] is True
        placed = param[...]
        assert placed.dtype == jnp.int8 and placed.shape == k2d, (key,
                                                                  placed.dtype,
                                                                  placed.shape)
        assert placed.sharding.spec == P(*wspec), (key, placed.sharding.spec,
                                                   wspec)
        w_s = h.adopt_host_quant_scale(param)
        assert w_s is not None and w_s.dtype == jnp.float32 and w_s.sharding.spec == P(
            *sspec), key
        # the post-fix process_weights_after_loading: same buffers, fresh Params
        delattr(block, name)
        setattr(block, name, nnx.Param(placed))
        scales[key] = nnx.Param(w_s)
        new_q_phys += shards(placed) + shards(w_s)
        del placed, w_s
        peak = max(peak, phys(mesh_devs) - base)
    wu.shard_put = orig_put
    out["after"] = {
        "peak": peak,
        "end": phys(mesh_devs) - base,
        "quantized": new_q_phys,
        "host_peak": host_peak
    }
    for key, li, i, k2d, wspec, sspec, block, name in list(params(model)):
        new[key] = (np.asarray(getattr(block, name)[...]).tobytes(),
                    np.asarray(scales[key][...]).tobytes())

    # ------------------------------------------------------------ numerics
    eager = {}
    for key, li, i, k2d, *_ in list(params(model)):
        with jax.set_mesh(host_mesh):
            e_q, e_s = q.quantize_tensor(jnp.int8,
                                         checkpoint(li, i).reshape(k2d),
                                         axis=0)
        eager[key] = (np.asarray(e_q).tobytes(), np.asarray(e_s).tobytes())
    out["numerics"] = {
        "host_equals_old_device_path": all(new[k] == old[k] for k in new),
        "host_equals_eager_on_host": all(new[k] == eager[k] for k in new),
        "n": len(new),
    }

    # ---------------------------------------------------- negative controls
    with jax.set_mesh(mesh):
        ctl = nnx.eval_shape(Block)
    plain, req_on_mesh = ctl.gate_up, ctl.down
    with jax.set_mesh(mesh):
        wu.assign_and_shard_param(plain, checkpoint(0, 0), "plain",
                                  mesh=mesh)  # no request
    h.request_host_quant(
        req_on_mesh,
        h.HostQuantRequest(jnp.int8, (256, 64), ("model", None), (None, )))
    with jax.set_mesh(mesh):
        already_there = jnp.asarray(np.ones(
            (256, 64), np.float32)).astype(jnp.bfloat16)  # born ON the mesh
        wu.assign_and_shard_param(req_on_mesh,
                                  already_there,
                                  "on_mesh",
                                  mesh=mesh)
    out["controls"] = {
        "no_request_stays_bf16":
        str(plain[...].dtype),
        "on_mesh_request_stays_bf16":
        str(req_on_mesh[...].dtype),
        "on_mesh_request_parks_no_scale":
        h.adopt_host_quant_scale(req_on_mesh) is None,
    }
    return out


@functools.lru_cache(maxsize=None)
def _result():
    pytest.importorskip("jax")
    pytest.importorskip("flax")
    env = dict(os.environ)
    env["XLA_FLAGS"] = (
        env.get("XLA_FLAGS", "") +
        f" --xla_force_host_platform_device_count={N_DEVICES}").strip()
    env["JAX_PLATFORMS"] = "cpu"
    r = subprocess.run([sys.executable, __file__, "--measure"],
                       env=env,
                       capture_output=True,
                       text=True,
                       timeout=900,
                       cwd=str(ROOT))
    assert r.returncode == 0, f"measurement failed (rc={r.returncode}):\n{r.stdout[-3000:]}\n{r.stderr[-6000:]}"
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_after_the_fix_the_mesh_only_ever_holds_quantized_kernels():
    r = _result()
    a = r["after"]
    assert a["peak"] == a["end"] == a["quantized"], (
        f"mesh peak {a['peak']} B != resident quantized set {a['quantized']} B: something other than "
        f"(w_q, w_s) touched the model mesh during the load")
    assert a["end"] < r["before"][
        "bf16"], "the quantized model must be smaller than the bf16 one it replaces"


def test_before_the_fix_the_whole_bf16_model_stayed_resident_plus_two_f32_temporaries(
):
    """The live failure's arithmetic, reproduced: after the stream every
    kernel is bf16 on the mesh; the requant then needs two float32
    temporaries of the kernel in flight ON TOP of that; and at the end the
    bf16 buffers are still there (pinned by params_dict) next to the int8."""
    r = _result()
    b = r["before"]
    assert b["after_stream"] == b[
        "bf16"], "every kernel should be bf16-resident after the stream"
    assert b["peak_at_clip"] >= b["bf16"] + 2 * 4 * r["n_largest"], (
        f"expected bf16 model ({b['bf16']} B) + two f32 temporaries of the largest kernel "
        f"({2 * 4 * r['n_largest']} B) alive at jnp.clip; measured {b['peak_at_clip']} B"
    )
    assert b["end"] == b["bf16"] + b["quantized"], (
        f"the replaced bf16 Params should stay pinned by params_dict until the loader returns: "
        f"end {b['end']} B vs bf16 {b['bf16']} + quantized {b['quantized']} B")
    assert b["peak"] > 3 * r["after"]["peak"], (
        f"before {b['peak']} B vs after {r['after']['peak']} B: the fix should cut the peak by more than 3x for int8"
    )


def test_host_side_staging_is_one_kernel():
    """What Python holds on the host while a kernel is quantized and placed:
    its bf16 staging copy plus its own int8 and scale, never a second
    kernel. (XLA's scratch inside the jitted quantize_tensor is not a
    jax.Array and is not counted.)"""
    r = _result()
    bound = r["bf16_largest"] + r["int8_largest"] + r["scale_largest"]
    assert 0 < r["after"]["host_peak"] <= bound, (
        f"host peak {r['after']['host_peak']} B exceeds one kernel's staging {bound} B"
    )


def test_host_quantized_kernels_are_bit_identical_to_the_device_path():
    r = _result()
    n = r["numerics"]
    assert n["n"] == N_BLOCKS * len(LAYERS)
    assert n[
        "host_equals_old_device_path"], "int8 codes/scales differ from the pre-fix on-mesh requant"
    assert n[
        "host_equals_eager_on_host"], "int8 codes/scales differ from eager quantize_tensor on the host"


def test_negative_controls_take_the_plain_path():
    r = _result()
    c = r["controls"]
    assert c[
        "no_request_stays_bf16"] == "bfloat16", "a Param without a request must be placed as-is"
    assert c["on_mesh_request_stays_bf16"] == "bfloat16", (
        "an array already on the mesh (pathways_dummy) must be left to the on-device fallback"
    )
    assert c["on_mesh_request_parks_no_scale"]


if __name__ == "__main__" and "--measure" in sys.argv:
    print(json.dumps(_measure()))
