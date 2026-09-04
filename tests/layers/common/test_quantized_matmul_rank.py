"""The quantized matmul must contract the activation's LAST axis at ANY rank.

MEASURED ON v6e 2026-09-01: `xla_quantized_matmul` hardcoded
`dimension_numbers=(((1,), (0,)), ((), ()))` -- rank-2 exactly -- so a rank-3
activation `[1, 1120, 6912]` (the Gemma-4 vision projection, which reaches it
via `pv.unsqueeze(0)`) died with

    dot_general requires contracting dimensions to have the same shape,
    got (1120,) and (6912,)

The weight was innocent: `patch_dense` is a plain 2-D [6912, 3840] kernel and
the correct formulation is identical to a text Linear. This suite drives the
REAL leaf (loaded by file path so it runs on CPU-only jax) at the exact rank
and shapes that failed.

Negative control: restoring the hardcoded dimension_numbers turns the rank-3
cases red while the rank-2 cases stay green -- which is precisely why a
rank-2-only suite certified this code for weeks.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
LEAF = ROOT / "tpu_inference" / "layers" / "common" / "linear.py"


def _mod():
    """Load the REAL linear.py, stubbing only the imports that are unrelated
    to the function under test (tokamax's gmm_v2, the Pallas kernel util, the
    sharding enum, the logger). Testing a local mirror instead would prove
    nothing about the shipped contraction -- the whole defect WAS the shipped
    dimension_numbers."""
    pytest.importorskip("jax", reason="numeric check needs jax")
    import sys
    import types

    def _stub(name, **attrs):
        if name in sys.modules:
            return
        mod = types.ModuleType(name)
        for k, v in attrs.items():
            setattr(mod, k, v)
        sys.modules[name] = mod

    for pkg in ("tokamax", "tokamax._src", "tokamax._src.ops",
                "tokamax._src.ops.experimental",
                "tokamax._src.ops.experimental.gmm_v2"):
        _stub(pkg)
    _stub("tokamax._src.ops.experimental.gmm_v2.gmm_v2",
          gmm_v2=lambda *a, **k: None)

    class _Axis(str):
        pass

    sharding_stub = types.ModuleType("tpu_inference.layers.common.sharding")
    sharding_stub.ShardingAxisName = type(
        "ShardingAxisName", (), {
            n: _Axis(n)
            for n in ("ATTN_DATA", "MLP", "VOCAB", "MODEL", "DATA", "EXPERT",
                      "ATTN_HEAD")
        })
    sys.modules.setdefault("tpu_inference.layers.common.sharding",
                           sharding_stub)

    class _Logger:

        def __getattr__(self, _):
            return lambda *a, **k: None

    logger_stub = types.ModuleType("tpu_inference.logger")
    logger_stub.init_logger = lambda *a, **k: _Logger()
    sys.modules.setdefault("tpu_inference.logger", logger_stub)

    # The ACTIVATION-quant primitive is the REAL one, loaded by path from
    # kernels/quantized_matmul/util.py. An earlier cut of this test
    # hand-rolled it with `jnp.finfo(dtype)` and that stub -- not the shipped
    # code -- was what raised "data type dtype('int8') not compatible with
    # finfo", nearly booking a false INT8 blocker. The real primitive routes
    # through `quantize_block`, which handles integers explicitly
    # (`jnp.round` + `get_max_min`). Stub the primitive, invent the bug.
    # These must be real PACKAGES (a __path__), not bare modules, or util's
    # own `from ...quantized_matmul.tuned_block_sizes import TunedValue`
    # fails with "is not a package".
    for pkg_name, pkg_dir in (
        ("tpu_inference.kernels", ROOT / "tpu_inference" / "kernels"),
        ("tpu_inference.kernels.quantized_matmul",
         ROOT / "tpu_inference" / "kernels" / "quantized_matmul"),
    ):
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(pkg_dir)]
            sys.modules[pkg_name] = pkg
    uspec = importlib.util.spec_from_file_location(
        "tpu_inference.kernels.quantized_matmul.util",
        ROOT / "tpu_inference" / "kernels" / "quantized_matmul" / "util.py")
    umod = importlib.util.module_from_spec(uspec)
    sys.modules["tpu_inference.kernels.quantized_matmul.util"] = umod
    try:
        uspec.loader.exec_module(umod)
    except Exception as e:  # noqa: BLE001
        # Deliberately NOT pytest.skip: this leaf is jax-only and MUST load.
        # A skip here would turn the whole suite vacuously green, which is
        # how a rank-2-only suite certified the broken contraction for weeks.
        raise AssertionError(
            f"real quantized_matmul util failed to load: {e}") from e

    spec = importlib.util.spec_from_file_location("_lin_leaf", LEAF)
    m = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(m)
    except Exception as e:  # noqa: BLE001
        # AUDIT 2026-09-03: this was `pytest.skip(...)`. linear.py is the leaf
        # UNDER TEST, so a load failure is the loudest possible regression --
        # and a skip made the whole file exit 0. Measured: adding one
        # unstubbed `import vllm.x` to linear.py produced "8 skipped", exit 0,
        # gate GREEN, with both rank-3 regression repros silently retired.
        # The util leaf eleven lines above already refuses to skip for exactly
        # this reason; the file under test deserves it at least as much.
        raise AssertionError(
            f"linear.py (the leaf under test) failed to load: {e}. Add a stub "
            f"rather than skipping -- a skip here is a false green.") from e
    return m


def _weight_quant(w):
    """Per-output-channel e4m3 via the REAL common-leaf primitive."""
    import jax.numpy as jnp
    qspec = importlib.util.spec_from_file_location(
        "_q_leaf2", ROOT / "tpu_inference" / "layers" / "common" /
        "quantization" / "__init__.py")
    qmod = importlib.util.module_from_spec(qspec)
    qspec.loader.exec_module(qmod)
    return qmod.quantize_tensor(jnp.float8_e4m3fn, w, axis=0)


# The invariant that matters is SELF-CONSISTENCY, not agreement with a
# hand-rolled reference: a rank-N call must equal the rank-2 call on the same
# data flattened, because the leading axes are pure batch. That holds
# whatever the activation-quant semantics are, so the test cannot be wrong
# about them -- and it is exactly the property the hardcoded
# dimension_numbers violated.


@pytest.mark.parametrize("lead", [(7, ), (1, 5), (2, 3)])
def test_rank_n_equals_rank_2_on_flattened_input(lead):
    import jax
    import jax.numpy as jnp
    m = _mod()
    in_f, out_f = 64, 32
    x = jax.random.normal(jax.random.PRNGKey(0),
                          lead + (in_f, )).astype(jnp.bfloat16)
    w = jax.random.normal(jax.random.PRNGKey(1), (in_f, out_f))
    w_q, w_s = _weight_quant(w)

    out = m.xla_quantized_matmul(x, w_q, w_s)
    assert out.shape == lead + (out_f, ), (
        f"rank {len(lead)+1} activation produced {out.shape}, expected "
        f"{lead + (out_f,)} -- leading axes must be preserved, not fused")

    flat = m.xla_quantized_matmul(x.reshape(-1, in_f), w_q, w_s)
    assert flat.shape == (int(jnp.prod(jnp.array(lead))), out_f)
    diff = float(
        jnp.max(
            jnp.abs(
                out.reshape(-1, out_f).astype(jnp.float32) -
                flat.astype(jnp.float32))))
    assert diff == 0.0, (
        f"rank-{len(lead)+1} result differs from the flattened rank-2 result "
        f"by {diff} -- the leading axes are not pure batch")


@pytest.mark.parametrize(
    "dtype_name",
    [
        "float8_e4m3fn",  # the historical default
        "float8_e4m3b11fnuz",  # what a v6e (<gen 7) ingests without a cast
        "float8_e5m2",
        "int8",  # ~2x bf16 FLOPs on v6e -- the real throughput lever
    ])
def test_every_online_quant_dtype_computes_at_rank_3(dtype_name):
    """Each dtype `Fp8OnlineLinearMethod` can select must survive the whole
    path -- weight quant, activation quant, contraction -- at rank 3.

    This is the CPU half of the BF16/FP8/INT8 bench matrix: if a dtype cannot
    even compute here, no hardware arm using it is measuring what its label
    says. int8 is included deliberately: v6e does ~2x bf16 FLOPs in int8
    while fp8 is 918 TFLOPs = bf16's 918, so int8 -- not fp8 -- is the
    arithmetic lever on this generation.
    """
    import jax
    import jax.numpy as jnp
    m = _mod()
    dtype = getattr(jnp, dtype_name)
    qspec = importlib.util.spec_from_file_location(
        "_q_leaf3", ROOT / "tpu_inference" / "layers" / "common" /
        "quantization" / "__init__.py")
    qmod = importlib.util.module_from_spec(qspec)
    qspec.loader.exec_module(qmod)

    in_f, out_f = 64, 32
    x = jax.random.normal(jax.random.PRNGKey(0),
                          (2, 3, in_f)).astype(jnp.bfloat16)
    w = jax.random.normal(jax.random.PRNGKey(1), (in_f, out_f))
    w_q, w_s = qmod.quantize_tensor(dtype, w, axis=0)
    assert w_q.dtype == dtype

    out = m.xla_quantized_matmul(x, w_q, w_s)
    assert out.shape == (2, 3, out_f)

    ref = jnp.einsum("...i,io->...o", x.astype(jnp.float32), w)
    rel = float(
        jnp.max(jnp.abs(out.astype(jnp.float32) - ref)) /
        jnp.max(jnp.abs(ref)))
    # AUDIT 2026-09-03: this was a single `rel < 0.35` for every dtype --
    # 5x to 45x the measured values, wide enough to accept a grossly wrong
    # result. MEASURED on this exact input (PRNGKey(0)/(1), 2x3x64 @ 64x32):
    #   float8_e4m3fn 0.03553   float8_e4m3b11fnuz 0.03373
    #   float8_e5m2   0.06621   int8               0.00754
    # Bounds are ~1.7x the measured value: still "quantization happened and
    # did not destroy the signal", not a quality bar, but now per-dtype and
    # tight enough that a systematically wrong scale cannot pass.
    bound = {
        "float8_e4m3fn": 0.06,
        "float8_e4m3b11fnuz": 0.06,
        "float8_e5m2": 0.10,
        "int8": 0.02,
    }[dtype_name]
    assert rel < bound, (
        f"{dtype_name} relative error {rel:.4f} exceeds {bound} -- the value "
        f"is not a faithfully quantized version of the reference")


def test_the_exact_shape_that_died_on_hardware():
    """[1, 1120, 6912] x [6912, 3840] -- max_soft_tokens x vision patch dim."""
    import jax
    import jax.numpy as jnp
    m = _mod()
    # AUDIT 2026-09-03: this was `jnp.zeros(...)`. An all-zero activation
    # makes the per-row amax zero, so the run took the degenerate
    # scale == 0 -> scale_inv == 0 branch and the ONLY thing that could fail
    # was the shape. Real data exercises the branch production takes and
    # keeps the shape assertion.
    x = jax.random.normal(jax.random.PRNGKey(4),
                          (1, 8, 6912)).astype(jnp.bfloat16)
    w = jax.random.normal(jax.random.PRNGKey(2), (6912, 64))
    w_q, w_s = _weight_quant(w)
    out = m.xla_quantized_matmul(x, w_q, w_s)
    assert out.shape == (1, 8, 64), (
        "the rank-3 vision-projection shape still collapses -- this is the "
        "exact call that produced 'got (1120,) and (6912,)' on v6e")
    assert bool(jnp.all(
        jnp.isfinite(out))), "the rank-3 projection produced non-finite values"
    assert float(jnp.max(
        jnp.abs(out))) > 0.0, "the rank-3 projection collapsed to all zeros"
