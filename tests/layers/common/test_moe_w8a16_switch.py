"""TPU_ONLINE_QUANT_ACT=0 must give weight-only quantization (W8A16) on the
MoE EXPERT path, the way tests/layers/common/test_w8a16_switch.py pins it for
the dense stack -- and the default must still be W8A8, bit for bit.

Why this exists (2026-09-02): on the 26B-A4B every expert matmul runs through
layers/common/fused_moe_gmm.py gmm_wrapper -> tokamax gmm_v2, and that call
never passed maybe_quantize_lhs. The kernel's default is True, so with
TPU_ONLINE_QUANT_ACT=0 the dense stack went weight-only while the experts --
the bulk of the bytes -- silently stayed W8A8. The 12B measurement that made
W8A16 the quality winner (cap-rate 3/69 vs W8A8's 13/69) had no honest 26B
twin: a lane labelled W8A16 would have been a mixed configuration.

The Mosaic kernel cannot run on CPU. What CAN be pinned here, and is:

  1. the call the wrapper makes: under ACT=0 the activations reach the
     kernel call in bf16 with maybe_quantize_lhs=False; under ACT=1 the call
     is IDENTICAL to the pre-change call (every kwarg, and the flag at the
     kernel's documented default), so the kernel's output cannot differ;
  2. the arithmetic the flag selects, emulated from kernels/megablox/gmm_v2.py
     make_gmm_configs + inner_kernel (weight-only: bf16 lhs, int8 tile widened
     to bf16, f32 accumulation, per-channel scale after the matmul; quantized:
     per-row int8 lhs, int32 accumulation, both scales after), driven through
     the REAL fused_moe_func -> tensor_parallel_gmm -> moe_gmm_local ->
     gmm_wrapper chain with the REAL per-expert per-channel int8 weight
     quantization (layers/common/quantization.quantize_tensor, the primitive
     process_weights/moe_weights.py quantize_moe_weights calls).

The live check for the kernel half is the harness lane
eval-26b-tp4-q-int8-w8a16 (TPU_ONLINE_QUANT_DTYPE=int8 TPU_ONLINE_QUANT_ACT=0
MOE_REQUANTIZE_WEIGHT_DTYPE=int8 on the 26B TP=4 lane).
"""
import importlib.util
import pathlib
import sys
import types
from typing import ClassVar

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
LEAF = ROOT / "tpu_inference" / "layers" / "common" / "fused_moe_gmm.py"
QUANT = (ROOT / "tpu_inference" / "layers" / "common" / "quantization" /
         "__init__.py")


class _Reached(Exception):
    """Sentinel: the experimental fused-RS kernel entry point was called."""


class _Logger:
    """Records every call so the engagement marker can be asserted."""
    records: ClassVar[list] = []

    def __getattr__(self, name):

        def log(msg, *args, **kw):
            _Logger.records.append((name, msg % args if args else msg))

        return log


def _unreachable(name):

    def fn(*a, **k):
        raise AssertionError(f"{name} must not be reached on the TP path")

    return fn


def _dense_gather_reduce_xla(x,
                             indices,
                             topk_weights,
                             reduce_group_size,
                             topk_wgt_zero_nan=False):
    """XLA twin of kernels/sparse_core/dense_gather_reduce (the JAX
    fallback the wrapper itself uses when the Pallas kernel cannot run):
    out[t] = sum_k topk_weights[t, k] * x[indices[t * k + k]]."""
    import jax.numpy as jnp
    tokens = topk_weights.shape[0]
    gathered = x[indices].reshape(tokens, reduce_group_size, x.shape[-1])
    w = topk_weights.reshape(tokens, reduce_group_size, 1)
    return (gathered.astype(jnp.float32) * w.astype(jnp.float32)).sum(axis=1)


def _make_gmm_v2(calls, matmul_operands):
    """Jax-traceable stand-in for tokamax gmm_v2 with the two lhs paths of
    kernels/megablox/gmm_v2.py, recording (at trace time, inside shard_map
    inside jit) the static facts of every call and the dtypes that reach
    the matmul."""
    import jax
    import jax.numpy as jnp

    def gmm_v2(*,
               lhs,
               rhs,
               rhs_scale,
               rhs_bias,
               group_sizes,
               group_offset,
               zero_initialize,
               fuse_act,
               preferred_element_type,
               maybe_quantize_lhs=True):
        scale_shape = None if rhs_scale is None else tuple(rhs_scale.shape)
        calls.append({
            "maybe_quantize_lhs": maybe_quantize_lhs,
            "lhs_dtype": lhs.dtype,
            "rhs_dtype": rhs.dtype,
            "scale_shape": scale_shape,
            "has_bias": rhs_bias is not None,
            "fuse_act": fuse_act,
            "preferred_element_type": preferred_element_type,
            "zero_initialize": zero_initialize,
        })
        assert rhs_bias is None
        assert jnp.issubdtype(rhs.dtype, jnp.integer)
        size_m = lhs.shape[0]
        num_local_groups = rhs.shape[0]
        out_dtype = preferred_element_type
        if out_dtype is None:
            out_dtype = lhs.dtype

        # gm metadata, done densely: which local group owns each row.
        ends = jnp.cumsum(group_sizes)
        gid = jnp.searchsorted(ends, jnp.arange(size_m), side="right")
        gid = jnp.clip(gid - group_offset, 0, num_local_groups - 1)
        w_rows = rhs[gid]  # [m, k, n]

        # make_gmm_configs: lhs is quantized only when asked AND rhs has a scale.
        if maybe_quantize_lhs and rhs_scale is not None:
            # inner_kernel, quantized path (one <=512 block per row here).
            dtype_max = float(jnp.iinfo(jnp.int8).max)
            amax = jnp.max(jnp.abs(lhs), axis=1, keepdims=True)
            scale = amax / dtype_max
            scale_inv = jnp.where(scale == 0, 0, 1 / scale)
            lhs_q = (lhs * scale_inv).astype(jnp.int8)
            matmul_operands.append((lhs_q.dtype, w_rows.dtype))
            acc = jnp.einsum("mk,mkn->mn",
                             lhs_q,
                             w_rows,
                             preferred_element_type=jnp.int32).astype(
                                 jnp.float32)
            acc = acc * scale.astype(jnp.float32)
        else:
            # inner_kernel, "Unquantized matmul path": jnp.matmul(bf16, int8)
            # promotes the int8 tile to the lhs dtype; f32 accumulation.
            w_dense = w_rows.astype(lhs.dtype)
            matmul_operands.append((lhs.dtype, w_dense.dtype))
            acc = jnp.einsum("mk,mkn->mn",
                             lhs,
                             w_dense,
                             preferred_element_type=jnp.float32)
        if rhs_scale is not None:
            assert rhs_scale.shape[1] == 1, "per-channel scale expected"
            acc = acc * rhs_scale[gid][:, 0, 0, :].astype(jnp.float32)
        if fuse_act is not None:
            gate, up = jnp.split(acc, 2, axis=-1)
            act = {
                "gelu": lambda g, u: jax.nn.gelu(g) * u,
                "silu": lambda g, u: jax.nn.silu(g) * u,
            }[fuse_act]
            acc = act(gate, up)
        return acc.astype(out_dtype)

    return gmm_v2


def _old_style(gmm_v2):
    """The PRE-change call shape: gmm_wrapper did not pass maybe_quantize_lhs,
    so the kernel saw its own default. Dropping the kwarg reproduces that."""

    def fn(**kw):
        kw.pop("maybe_quantize_lhs", None)
        return gmm_v2(**kw)

    return fn


def _mod(quant_act: bool, gmm_v2_impl):
    """Load the REAL fused_moe_gmm.py with a controllable envs stub, the REAL
    quantization leaf, and the TPU-only kernels stubbed."""
    pytest.importorskip("jax")
    for k in [
            k for k in sys.modules
            if k.startswith(("tpu_inference", "tokamax"))
    ]:
        del sys.modules[k]

    def pkg(name, path=None):
        m = types.ModuleType(name)
        if path is not None:
            m.__path__ = [str(path)]
        sys.modules[name] = m
        parent, _, leaf = name.rpartition(".")
        if parent and parent in sys.modules:
            setattr(sys.modules[parent], leaf, m)
        return m

    pkg("tpu_inference", ROOT / "tpu_inference")
    envs = pkg("tpu_inference.envs")
    envs.TPU_ONLINE_QUANT_ACT = quant_act
    envs.MOE_APPROX_TOPK = False
    envs.MOE_APPROX_TOPK_RECALL_TARGET = 0.95
    envs.FORCE_MOE_RANDOM_ROUTING = False
    lg = pkg("tpu_inference.logger")
    _Logger.records.clear()
    lg.init_logger = lambda *a, **k: _Logger()
    ut = pkg("tpu_inference.utils")

    def get_mesh_shape_product(mesh, axis):
        axes = (axis, ) if isinstance(axis, str) else tuple(axis)
        prod = 1
        for a in axes:
            prod *= mesh.shape.get(a, 1)
        return prod

    ut.get_mesh_shape_product = get_mesh_shape_product
    pkg("tpu_inference.layers", ROOT / "tpu_inference" / "layers")
    pkg("tpu_inference.layers.common",
        ROOT / "tpu_inference" / "layers" / "common")
    sh = pkg("tpu_inference.layers.common.sharding")
    sh.ShardingAxisName = type(
        "S", (), {
            "MLP_DATA": "data",
            "ATTN_DATA": "data",
            "MLP_TENSOR": "model",
            "EXPERT": "model",
            "EXPERT_DATA": ("data", "model"),
        })
    qs = importlib.util.spec_from_file_location(
        "tpu_inference.layers.common.quantization", QUANT)
    qm = importlib.util.module_from_spec(qs)
    sys.modules[qs.name] = qm
    qs.loader.exec_module(qm)
    pkg("tpu_inference.kernels", ROOT / "tpu_inference" / "kernels")
    pkg("tpu_inference.kernels.collectives")
    pkg("tpu_inference.kernels.collectives.hierrs_sc")
    hw = pkg("tpu_inference.kernels.collectives.hierrs_sc.wrapper")
    hw.hierarchical_reduce_scatter_local = _unreachable(
        "hierarchical_reduce_scatter_local")
    pkg("tpu_inference.kernels.sparse_core")
    dg = pkg("tpu_inference.kernels.sparse_core.dense_gather_reduce")
    dg.dense_gather_reduce = _dense_gather_reduce_xla
    rg = pkg("tpu_inference.kernels.sparse_core.ragged_gather_reduce_v2")
    rg.ragged_gather_reduce = _unreachable("ragged_gather_reduce")
    rv = pkg("tpu_inference.kernels.sparse_core.ragged_gather_v2")
    rv.ragged_gather_v2 = _unreachable("ragged_gather_v2")
    pkg("tpu_inference.kernels.experimental")
    pkg("tpu_inference.kernels.experimental.fused_moe")
    rs = pkg("tpu_inference.kernels.experimental.fused_moe.fused_moe_rs")

    def fused_moe_func_rs(**kw):
        raise _Reached()

    rs.fused_moe_func_rs = fused_moe_func_rs
    for p in ("tokamax", "tokamax._src", "tokamax._src.ops",
              "tokamax._src.ops.experimental",
              "tokamax._src.ops.experimental.gmm_v2"):
        pkg(p)
    leaf = pkg("tokamax._src.ops.experimental.gmm_v2.gmm_v2")
    leaf.gmm_v2 = gmm_v2_impl
    spec = importlib.util.spec_from_file_location(
        "tpu_inference.layers.common.fused_moe_gmm", LEAF)
    m = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m, qm


def _problem(qm, seed=0):
    """Random experts, quantized EXACTLY as the load path quantizes them:
    quantize_moe_weights -> quantize_tensor(int8, w, axis=2, block=K) gives
    a per-expert, per-output-channel f32 abs-max scale with rounding; then
    process_moe_weights transposes to [E, in, out] and lays the scale out as
    [E, in_blocks=1, 1, out] -- the shapes fused_moe_func documents."""
    import jax
    import jax.numpy as jnp
    E, T, D, F, K = 4, 16, 64, 32, 2
    k1, k2, k3, k4 = jax.random.split(jax.random.PRNGKey(seed), 4)
    x = jax.random.normal(k1, (T, D)).astype(jnp.bfloat16)
    w13 = jax.random.normal(k2, (E, 2 * F, D))  # [E, out, in] as staged
    w2 = jax.random.normal(k3, (E, D, F))
    gating = jax.random.normal(k4, (T, E))
    w13_q, w13_s = qm.quantize_tensor(jnp.int8, w13, 2, D)
    w2_q, w2_s = qm.quantize_tensor(jnp.int8, w2, 2, F)
    assert w13_q.dtype == jnp.int8 and w13_s.dtype == jnp.float32
    assert w13_s.shape == (E, 2 * F, 1) and w2_s.shape == (E, D, 1)

    def to_kernel(w_q, w_s):
        w_k = jnp.swapaxes(w_q, 1, 2)
        s_k = jnp.expand_dims(jnp.swapaxes(w_s, 1, 2), 2)
        return w_k, s_k

    w1_k, w1_s_k = to_kernel(w13_q, w13_s)
    w2_k, w2_s_k = to_kernel(w2_q, w2_s)
    assert w1_s_k.shape == (E, 1, 1, 2 * F) and w2_s_k.shape == (E, 1, 1, D)
    return {
        "x": x,
        "gating": gating,
        "w13_q": w13_q,
        "w13_s": w13_s,
        "w2_q": w2_q,
        "w2_s": w2_s,
        "w1": w1_k,
        "w1_scale": w1_s_k,
        "w2": w2_k,
        "w2_scale": w2_s_k,
        "E": E,
        "T": T,
        "D": D,
        "F": F,
        "K": K,
    }


def _mesh():
    import jax
    from jax.sharding import Mesh
    return Mesh(np.array(jax.devices()[:1]).reshape(1, 1), ("data", "model"))


def _forward(m, p, **overrides):
    kw = {
        "hidden_states": p["x"],
        "w1": p["w1"],
        "w2": p["w2"],
        "w1_scale": p["w1_scale"],
        "w2_scale": p["w2_scale"],
        "w1_bias": None,
        "w2_bias": None,
        "gating_output": p["gating"],
        "topk": p["K"],
        "renormalize": True,
        "mesh": _mesh(),
        "use_ep": False,
        "activation": "gelu",
        "scoring_fn": "softmax",
    }
    kw.update(overrides)
    return m.fused_moe_func(**kw)


def _reference(p):
    """f32 MoE over the DEQUANTIZED int8 experts: what W8A16 approximates."""
    import jax
    import jax.numpy as jnp
    x = p["x"].astype(jnp.float32)
    w13 = p["w13_q"].astype(jnp.float32) * p["w13_s"]  # [E, 2F, D]
    w2 = p["w2_q"].astype(jnp.float32) * p["w2_s"]  # [E, D, F]
    probs = jax.nn.softmax(p["gating"], axis=-1)
    tw, ti = jax.lax.top_k(probs, p["K"])
    tw = tw / tw.sum(axis=-1, keepdims=True)
    out = jnp.zeros((p["T"], p["D"]), jnp.float32)
    for e in range(p["E"]):
        h = x @ w13[e].T
        a = jax.nn.gelu(h[:, :p["F"]]) * h[:, p["F"]:]
        y = a @ w2[e].T
        we = (tw * (ti == e)).sum(axis=-1)
        out = out + we[:, None] * y
    return out


def _rel(a, b):
    import jax.numpy as jnp
    a = a.astype(jnp.float32)
    b = b.astype(jnp.float32)
    return float(jnp.max(jnp.abs(a - b)) / jnp.max(jnp.abs(b)))


def test_act0_expert_activations_reach_the_kernel_in_bf16():
    import jax.numpy as jnp
    calls, operands = [], []
    m, qm = _mod(False, _make_gmm_v2(calls, operands))
    _forward(m, _problem(qm))
    assert len(calls) == 2, f"expected GMM1 and GMM2, saw {len(calls)} calls"
    for call in calls:
        assert call["maybe_quantize_lhs"] is False, call
        assert call["lhs_dtype"] == jnp.bfloat16, call
        assert call["rhs_dtype"] == jnp.int8, call
        assert len(call["scale_shape"]) == 4, call
        assert call["scale_shape"][1:3] == (1, 1), call
    # ... and the operands that reach the matmul are bf16 x bf16 (the int8
    # tile widened), never int8 x int8.
    assert operands == [(jnp.bfloat16, jnp.bfloat16)] * 2, operands


def test_act0_announces_itself_once_for_the_evidence_gate():
    """The harness proves an arm engaged by grepping the boot log. W8A8 and
    W8A16 experts print the same requant line, so W8A16 must name itself --
    once -- and W8A8 must not."""
    m, qm = _mod(False, _make_gmm_v2([], []))
    _forward(m, _problem(qm))
    marks = [r for r in _Logger.records if "TPU_ONLINE_QUANT_ACT=0" in r[1]]
    assert marks, "no engagement marker under ACT=0"
    assert all(r[0] == "info_once" for r in marks), marks
    assert "MoE experts serving weight-only" in marks[0][1], marks
    assert "bfloat16" in marks[0][1], marks
    m1, qm1 = _mod(True, _make_gmm_v2([], []))
    _forward(m1, _problem(qm1))
    assert not [r for r in _Logger.records if "TPU_ONLINE_QUANT_ACT=0" in r[1]]


def test_act1_default_quantizes_expert_activations():
    import jax.numpy as jnp
    calls, operands = [], []
    m, qm = _mod(True, _make_gmm_v2(calls, operands))
    _forward(m, _problem(qm))
    assert len(calls) == 2
    for call in calls:
        assert call["maybe_quantize_lhs"] is True, call
        assert call["lhs_dtype"] == jnp.bfloat16, call
    assert operands == [(jnp.int8, jnp.int8)] * 2, operands


def test_act1_call_is_bit_for_bit_the_pre_change_call():
    """Under the default the wrapper must hand the kernel exactly what it
    handed it before this change: the same kwargs, plus the flag at the
    kernel's own default. A kernel that ignores the new kwarg (the
    pre-change call shape) then produces bit-identical output."""
    import jax.numpy as jnp
    calls, _ = [], []
    m, qm = _mod(True, _make_gmm_v2(calls, []))
    p = _problem(qm)
    new = np.asarray(_forward(m, p).astype(jnp.float32))
    m_old, qm_old = _mod(True, _old_style(_make_gmm_v2([], [])))
    old = np.asarray(_forward(m_old, _problem(qm_old)).astype(jnp.float32))
    assert np.array_equal(new, old), "ACT=1 output changed bit-for-bit"
    expected_keys = {
        "maybe_quantize_lhs", "lhs_dtype", "rhs_dtype", "scale_shape",
        "has_bias", "fuse_act", "preferred_element_type", "zero_initialize"
    }
    assert set(calls[0]) == expected_keys
    assert calls[0]["zero_initialize"] is False
    assert calls[0]["fuse_act"] == "gelu" and calls[1]["fuse_act"] is None
    assert calls[0]["preferred_element_type"] == jnp.bfloat16
    assert calls[1]["preferred_element_type"] is None

    # And the switch is not a no-op: under ACT=0 the same kernel-that-
    # ignores-the-kwarg is NOT what the wrapper produces.
    m0, qm0 = _mod(False, _make_gmm_v2([], []))
    w8a16 = np.asarray(_forward(m0, _problem(qm0)).astype(jnp.float32))
    assert not np.array_equal(w8a16, old), (
        "ACT=0 produced the pre-change (W8A8) bits: the switch does nothing")


def test_w8a16_moe_matches_float_reference():
    m, qm = _mod(False, _make_gmm_v2([], []))
    p = _problem(qm)
    out = _forward(m, p)
    rel = _rel(out, _reference(p))
    assert rel < 2e-2, (
        f"W8A16 experts diverge from the dequantized-weight reference: {rel}")


def test_w8a16_and_w8a8_experts_differ_only_by_activation_rounding():
    """The two paths must be DIFFERENT (or the switch does nothing) and close
    (or one of them is broken). Both facts are load-bearing."""
    m8, qm8 = _mod(True, _make_gmm_v2([], []))
    a8 = _forward(m8, _problem(qm8))
    m16, qm16 = _mod(False, _make_gmm_v2([], []))
    a16 = _forward(m16, _problem(qm16))
    d = _rel(a8, a16)
    assert d > 1e-4, "W8A8 and W8A16 experts produced identical outputs"
    assert d < 5e-2, f"W8A8 and W8A16 experts diverge by {d}: one is wrong"


def test_fused_rs_kernel_refuses_w8a16_before_importing_the_kernel():
    """USE_GMM_FUSED_RS_KERNEL builds its own kernel configs and always
    quantizes the activations; under ACT=0 fused_moe_func must refuse,
    naming both switches, before the kernel entry point is reached. Under
    ACT=1 the entry point IS reached (the sentinel fires)."""
    m0, _ = _mod(False, _make_gmm_v2([], []))
    kw = {
        "hidden_states": None,
        "w1": None,
        "w2": None,
        "w1_scale": None,
        "w2_scale": None,
        "w1_bias": None,
        "w2_bias": None,
        "gating_output": None,
        "topk": 2,
        "renormalize": True,
        "mesh": None,
        "use_ep": True,
        "activation": "gelu",
        "scoring_fn": "softmax",
        "use_gmm_fused_rs_kernel": True,
    }
    with pytest.raises(NotImplementedError) as ei:
        m0.fused_moe_func.__wrapped__(**kw)
    msg = str(ei.value)
    assert "TPU_ONLINE_QUANT_ACT" in msg and "USE_GMM_FUSED_RS_KERNEL" in msg
    m1, _ = _mod(True, _make_gmm_v2([], []))
    with pytest.raises(_Reached):
        m1.fused_moe_func.__wrapped__(**kw)


def test_env_reaches_the_moe_kernel_call_site():
    src = LEAF.read_text()
    i = src.index("def gmm_wrapper")
    body = src[i:src.index("\ndef ", i + 10)]
    assert "if maybe_quantize_lhs and not envs.TPU_ONLINE_QUANT_ACT:" in body, (
        "gmm_wrapper does not consult TPU_ONLINE_QUANT_ACT")
    assert "maybe_quantize_lhs=maybe_quantize_lhs" in body, (
        "the flag is computed but not forwarded to gmm_v2")
    j = src.index("def fused_moe_func(")
    fb = src[j:]
    assert fb.index("not envs.TPU_ONLINE_QUANT_ACT") < fb.index(
        "fused_moe_func_rs"), "the fused-RS refusal must precede the import"
