"""TPU_ONLINE_QUANT_ACT=0 must give weight-only quantization (W8A16) that
matches a dequantized-weight bf16 matmul -- and the default must still be W8A8.

Why this exists (2026-09-02): int8 W8A8 hit the 4096-token cap on 13/69
requests vs bf16's 5 and fp8's 6, reproducibly, with a different top-of-output
decision. Hypothesis: per-token int8 ACTIVATION quantization (127 levels)
drops outliers that e4m3's exponent keeps. W8A16 is the experiment that
separates "int8 weights hurt" from "int8 activations hurt". Before it could
run, there was no switch: maybe_quantize_x defaulted True and the batched path
hardcoded `x.dtype.itemsize > 1`.
"""
import importlib.util
import pathlib
import sys
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
LEAF = ROOT / "tpu_inference" / "layers" / "common" / "linear.py"


def _mod(quant_act: bool):
    """Load the REAL linear.py with a controllable envs stub."""
    pytest.importorskip("jax")
    for k in [k for k in sys.modules if k.startswith("tpu_inference")]:
        del sys.modules[k]
    envs = types.ModuleType("tpu_inference.envs")
    envs.TPU_ONLINE_QUANT_ACT = quant_act
    envs.ENABLE_QUANTIZED_MATMUL_KERNEL = False
    sys.modules["tpu_inference.envs"] = envs
    pkg = types.ModuleType("tpu_inference")
    pkg.__path__ = [str(ROOT / "tpu_inference")]
    pkg.envs = envs
    sys.modules["tpu_inference"] = pkg
    for p in ("tokamax", "tokamax._src", "tokamax._src.ops",
              "tokamax._src.ops.experimental",
              "tokamax._src.ops.experimental.gmm_v2"):
        sys.modules[p] = types.ModuleType(p)
    leaf = types.ModuleType("tokamax._src.ops.experimental.gmm_v2.gmm_v2")
    leaf.gmm_v2 = lambda *a, **k: None  # linear.py:19 imports the SUBMODULE, not an attr
    sys.modules["tokamax._src.ops.experimental.gmm_v2.gmm_v2"] = leaf

    class _Axis(str):
        pass

    sh = types.ModuleType("tpu_inference.layers.common.sharding")
    sh.ShardingAxisName = type(
        "S", (), {
            n: _Axis(n)
            for n in ("ATTN_DATA", "MLP", "VOCAB", "MODEL", "DATA", "EXPERT",
                      "ATTN_HEAD")
        })
    sys.modules["tpu_inference.layers.common.sharding"] = sh
    lg = types.ModuleType("tpu_inference.logger")
    lg.init_logger = lambda *a, **k: type("L", (), {
        "__getattr__":
        lambda s, n: (lambda *a, **k: None)
    })()
    sys.modules["tpu_inference.logger"] = lg
    for pkgname, d in (("tpu_inference.kernels",
                        ROOT / "tpu_inference" / "kernels"),
                       ("tpu_inference.kernels.quantized_matmul", ROOT /
                        "tpu_inference" / "kernels" / "quantized_matmul")):
        m = types.ModuleType(pkgname)
        m.__path__ = [str(d)]
        sys.modules[pkgname] = m
    us = importlib.util.spec_from_file_location(
        "tpu_inference.kernels.quantized_matmul.util",
        ROOT / "tpu_inference" / "kernels" / "quantized_matmul" / "util.py")
    um = importlib.util.module_from_spec(us)
    sys.modules[us.name] = um
    us.loader.exec_module(um)
    spec = importlib.util.spec_from_file_location("_lin", LEAF)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _quant_w():
    import jax
    import jax.numpy as jnp
    qs = importlib.util.spec_from_file_location(
        "_q", ROOT / "tpu_inference" / "layers" / "common" / "quantization" /
        "__init__.py")
    q = importlib.util.module_from_spec(qs)
    qs.loader.exec_module(q)
    w = jax.random.normal(jax.random.PRNGKey(1), (64, 32))
    w_q, w_s = q.quantize_tensor(jnp.int8, w, axis=0)
    return w, w_q, w_s


def test_w8a16_matches_dequantized_weight_matmul():
    import jax
    import jax.numpy as jnp
    m = _mod(quant_act=False)
    w, w_q, w_s = _quant_w()
    x = jax.random.normal(jax.random.PRNGKey(0),
                          (1, 8, 64)).astype(jnp.bfloat16)
    out = m.xla_quantized_matmul(x, w_q, w_s, quantize_activation=False)
    w_deq = (w_q.astype(jnp.float32) * w_s[None, :]).astype(jnp.bfloat16)
    ref = jnp.einsum("bpd,dh->bph", x.astype(jnp.float32),
                     w_deq.astype(jnp.float32))
    rel = float(
        jnp.max(jnp.abs(out.astype(jnp.float32) - ref)) /
        jnp.max(jnp.abs(ref)))
    assert rel < 2e-2, f"W8A16 diverges from the dequantized-weight reference: {rel}"


def test_w8a16_and_w8a8_differ_only_by_activation_rounding():
    """The two paths must be DIFFERENT (or the switch does nothing) and close
    (or one of them is broken). Both facts are load-bearing."""
    import jax
    import jax.numpy as jnp
    m = _mod(quant_act=True)
    w, w_q, w_s = _quant_w()
    x = jax.random.normal(jax.random.PRNGKey(0),
                          (1, 8, 64)).astype(jnp.bfloat16)
    a8 = m.xla_quantized_matmul(x, w_q, w_s,
                                quantize_activation=True).astype(jnp.float32)
    a16 = m.xla_quantized_matmul(x, w_q, w_s,
                                 quantize_activation=False).astype(jnp.float32)
    d = float(jnp.max(jnp.abs(a8 - a16)) / jnp.max(jnp.abs(a16)))
    assert d > 1e-4, "W8A8 and W8A16 produced identical outputs: the activation path is not being quantized"
    assert d < 5e-2, f"W8A8 and W8A16 diverge by {d}: one path is wrong, not merely rounded"


def test_env_actually_reaches_the_dense_sharded_matmul():
    """BEHAVIOURAL proof that TPU_ONLINE_QUANT_ACT reaches the matmul.

    AUDIT 2026-09-03: the structural test below was the ONLY gate coverage
    for the dense switch, and it is a source-string match. Two measured
    holes it could not see:
      * deleting the guard from sharded_quantized_matmul turned ONLY that
        one substring test red -- the two numeric tests in this file pass
        `quantize_activation=` explicitly, so they never consult the env;
      * keeping the asserted text verbatim while hardcoding
        `quantize_activation=True` at both call sites -- i.e. ACT=0 silently
        serving W8A8 -- left all five tests green. That is the mislabelled-arm
        failure the MoE twin (test_moe_w8a16_switch.py) has a behavioural
        test for and the dense side did not.

    One forced CPU device is enough: the switch is a host-side branch, and
    the assertion is an exact bit comparison against the two explicit modes.
    """
    import jax
    import jax.numpy as jnp
    import numpy as np
    from jax.sharding import Mesh
    from jax.sharding import PartitionSpec as P
    w, w_q, w_s = _quant_w()
    x = jax.random.normal(jax.random.PRNGKey(0), (8, 64)).astype(jnp.bfloat16)
    mesh = Mesh(
        np.array(jax.devices()[:1]).reshape(1, 1), ("ATTN_DATA", "model"))

    m0 = _mod(quant_act=False)
    with jax.set_mesh(mesh):
        got0 = m0.sharded_quantized_matmul(x,
                                           w_q,
                                           w_s,
                                           P(None, "model"),
                                           mesh=mesh)
    w8a16 = m0.xla_quantized_matmul(x, w_q, w_s, quantize_activation=False)
    w8a8 = m0.xla_quantized_matmul(x, w_q, w_s, quantize_activation=True)
    # The control: the two modes must actually differ, or nothing below binds.
    assert not np.array_equal(np.asarray(w8a16), np.asarray(w8a8)), (
        "W8A16 and W8A8 are bit-identical on this input: the assertions "
        "below cannot distinguish anything")
    assert np.array_equal(np.asarray(got0), np.asarray(w8a16)), (
        "TPU_ONLINE_QUANT_ACT=0 did not reach the matmul: the sharded path "
        "still quantized the activations (a lane labelled W8A16 would serve "
        "W8A8)")

    m1 = _mod(quant_act=True)
    with jax.set_mesh(mesh):
        got1 = m1.sharded_quantized_matmul(x,
                                           w_q,
                                           w_s,
                                           P(None, "model"),
                                           mesh=mesh)
    assert np.array_equal(np.asarray(got1), np.asarray(w8a8)), (
        "the default (TPU_ONLINE_QUANT_ACT=1) stopped quantizing activations")


def test_env_flips_the_sharded_default_without_touching_explicit_callers():
    src = LEAF.read_text()
    i = src.index("def sharded_quantized_matmul")
    body = src[i:src.index("def ", i + 10)]
    assert "if maybe_quantize_x and not envs.TPU_ONLINE_QUANT_ACT:" in body
    assert "envs.TPU_ONLINE_QUANT_ACT" in src[
        src.index("_should_quantize_act ="):][:200], (
            "the batched path hardcoded itemsize>1 and ignored the env")


def test_the_env_defaults_to_w8a8_in_the_shipped_envs_module():
    """AUDIT 2026-09-03: nothing pinned the shipped default. Measured:
    flipping `env_bool("TPU_ONLINE_QUANT_ACT", default=True)` to
    `default=False` left all 309 gate tests green -- every unset deployment
    would silently switch from W8A8 to W8A16, which is a different arm of the
    quality matrix under an unchanged label. The whole switch exists to keep
    those two arms distinguishable, so the default is part of the contract.

    True == W8A8 == the historical behaviour; ACT=0 is the opt-in experiment.
    """
    spec = importlib.util.spec_from_file_location(
        "_w8a16_envs", ROOT / "tpu_inference" / "envs.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.TPU_ONLINE_QUANT_ACT is True, (
        "the shipped default must be W8A8 (the historical behaviour); "
        "weight-only is the opt-in experiment")


def test_env_is_forwarded_to_ray_workers():
    # AUDIT 2026-09-03: this was a raw `plat[i:i+1500]` substring window. The
    # list currently spans ~1100 chars, so the window already overruns into
    # the next method: a few more entries and the names fall OUT of it (false
    # red), while the same name quoted in a nearby comment satisfies it
    # (false green). Parse the actual list instead.
    import ast
    plat = (ROOT / "tpu_inference" / "platforms" /
            "tpu_platform.py").read_text()
    tree = ast.parse(plat)
    node = next(
        (n for n in ast.walk(tree)
         if isinstance(n, (ast.Assign, ast.AnnAssign)) and any(
             getattr(tg, "id", None) == "additional_env_vars" for tg in
             ([n.target] if isinstance(n, ast.AnnAssign) else n.targets))),
        None)
    assert node is not None, "additional_env_vars not found in tpu_platform.py"
    names = {
        e.value
        for e in ast.walk(node.value)
        if isinstance(e, ast.Constant) and isinstance(e.value, str)
    }
    assert "TPU_ONLINE_QUANT_DTYPE" in names, (
        f"TPU_ONLINE_QUANT_DTYPE missing from the Ray passthrough: a "
        f"multi-host int8 lane would serve fp8 on the workers (ti #29). "
        f"Got: {sorted(names)}")
    assert "TPU_ONLINE_QUANT_ACT" in names, (
        f"a multi-host Ray run would silently fall back to W8A8 while the "
        f"lane said W8A16 (the ti #29 shape). Got: {sorted(names)}")


def test_both_w8a16_implementations_agree():
    """TPU_ONLINE_QUANT_ACT=0 has TWO implementations -- the dense
    xla_quantized_matmul and the batched xla_quantized_batched_matmul -- and
    nothing checked they do the same thing. Review 2026-09-02 found the
    batched one had no explicit widen while the dense one did.

    Also the record: ti #38's commit message says "lax.dot_general refuses
    bf16 x int8" and that removing the widen "makes the W8A16 test fail loudly
    on the dtype mismatch". Both are FALSE -- measured on jax 0.11.1, the
    mixed dot_general is accepted and removing the widen left the tests green.
    The widen is for symmetry and explicitness, not correctness, and this test
    pins the property that actually matters: the two paths agree.
    """
    import importlib.util as _ilu

    import jax
    import jax.numpy as jnp
    m = _mod(quant_act=False)
    w, w_q, w_s = _quant_w()  # w_q [64, 32] int8, per-out scale
    x = jax.random.normal(jax.random.PRNGKey(3), (5, 64)).astype(jnp.bfloat16)

    dense = m.xla_quantized_matmul(x, w_q, w_s, quantize_activation=False)

    us = _ilu.spec_from_file_location(
        "tpu_inference.kernels.quantized_matmul.util",
        ROOT / "tpu_inference" / "kernels" / "quantized_matmul" / "util.py")
    um = _ilu.module_from_spec(us)
    import sys as _sys
    _sys.modules[us.name] = um
    us.loader.exec_module(um)
    batched = um.xla_quantized_batched_matmul(x,
                                              w_q,
                                              w_s,
                                              dimension_numbers=(((1, ),
                                                                  (0, )),
                                                                 ((), ())),
                                              quantize_activation=False)

    d = float(
        jnp.max(
            jnp.abs(dense.astype(jnp.float32) - batched.astype(jnp.float32))) /
        jnp.max(jnp.abs(dense.astype(jnp.float32))))
    assert d < 1e-3, f"the dense and batched W8A16 paths disagree by {d}"
