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
import os
import pathlib
import sys
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
LEAF = ROOT / "tpu_inference" / "layers" / "common" / "linear.py"


def _mod(quant_act: bool):
    """Load the REAL linear.py with a controllable envs stub."""
    pytest.importorskip("jax")
    for k in [k for k in sys.modules if k.startswith("tpu_inference")]:
        del sys.modules[k]
    envs = types.ModuleType("tpu_inference.envs"); envs.TPU_ONLINE_QUANT_ACT = quant_act
    envs.ENABLE_QUANTIZED_MATMUL_KERNEL = False
    sys.modules["tpu_inference.envs"] = envs
    pkg = types.ModuleType("tpu_inference"); pkg.__path__ = [str(ROOT / "tpu_inference")]; pkg.envs = envs
    sys.modules["tpu_inference"] = pkg
    for p in ("tokamax", "tokamax._src", "tokamax._src.ops", "tokamax._src.ops.experimental",
              "tokamax._src.ops.experimental.gmm_v2"):
        sys.modules[p] = types.ModuleType(p)
    leaf = types.ModuleType("tokamax._src.ops.experimental.gmm_v2.gmm_v2")
    leaf.gmm_v2 = lambda *a, **k: None  # linear.py:19 imports the SUBMODULE, not an attr
    sys.modules["tokamax._src.ops.experimental.gmm_v2.gmm_v2"] = leaf
    class _Axis(str): pass
    sh = types.ModuleType("tpu_inference.layers.common.sharding")
    sh.ShardingAxisName = type("S", (), {n: _Axis(n) for n in
        ("ATTN_DATA","MLP","VOCAB","MODEL","DATA","EXPERT","ATTN_HEAD")})
    sys.modules["tpu_inference.layers.common.sharding"] = sh
    lg = types.ModuleType("tpu_inference.logger")
    lg.init_logger = lambda *a, **k: type("L", (), {"__getattr__": lambda s, n: (lambda *a, **k: None)})()
    sys.modules["tpu_inference.logger"] = lg
    for pkgname, d in (("tpu_inference.kernels", ROOT/"tpu_inference"/"kernels"),
                       ("tpu_inference.kernels.quantized_matmul", ROOT/"tpu_inference"/"kernels"/"quantized_matmul")):
        m = types.ModuleType(pkgname); m.__path__ = [str(d)]; sys.modules[pkgname] = m
    us = importlib.util.spec_from_file_location("tpu_inference.kernels.quantized_matmul.util",
                                                ROOT/"tpu_inference"/"kernels"/"quantized_matmul"/"util.py")
    um = importlib.util.module_from_spec(us); sys.modules[us.name] = um; us.loader.exec_module(um)
    spec = importlib.util.spec_from_file_location("_lin", LEAF)
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    return m


def _quant_w():
    import jax, jax.numpy as jnp
    qs = importlib.util.spec_from_file_location("_q", ROOT/"tpu_inference"/"layers"/"common"/"quantization"/"__init__.py")
    q = importlib.util.module_from_spec(qs); qs.loader.exec_module(q)
    w = jax.random.normal(jax.random.PRNGKey(1), (64, 32))
    w_q, w_s = q.quantize_tensor(jnp.int8, w, axis=0)
    return w, w_q, w_s


def test_w8a16_matches_dequantized_weight_matmul():
    import jax, jax.numpy as jnp
    m = _mod(quant_act=False)
    w, w_q, w_s = _quant_w()
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 64)).astype(jnp.bfloat16)
    out = m.xla_quantized_matmul(x, w_q, w_s, quantize_activation=False)
    w_deq = (w_q.astype(jnp.float32) * w_s[None, :]).astype(jnp.bfloat16)
    ref = jnp.einsum("bpd,dh->bph", x.astype(jnp.float32), w_deq.astype(jnp.float32))
    rel = float(jnp.max(jnp.abs(out.astype(jnp.float32) - ref)) / jnp.max(jnp.abs(ref)))
    assert rel < 2e-2, f"W8A16 diverges from the dequantized-weight reference: {rel}"


def test_w8a16_and_w8a8_differ_only_by_activation_rounding():
    """The two paths must be DIFFERENT (or the switch does nothing) and close
    (or one of them is broken). Both facts are load-bearing."""
    import jax, jax.numpy as jnp
    m = _mod(quant_act=True)
    w, w_q, w_s = _quant_w()
    x = jax.random.normal(jax.random.PRNGKey(0), (1, 8, 64)).astype(jnp.bfloat16)
    a8 = m.xla_quantized_matmul(x, w_q, w_s, quantize_activation=True).astype(jnp.float32)
    a16 = m.xla_quantized_matmul(x, w_q, w_s, quantize_activation=False).astype(jnp.float32)
    d = float(jnp.max(jnp.abs(a8 - a16)) / jnp.max(jnp.abs(a16)))
    assert d > 1e-4, "W8A8 and W8A16 produced identical outputs: the activation path is not being quantized"
    assert d < 5e-2, f"W8A8 and W8A16 diverge by {d}: one path is wrong, not merely rounded"


def test_env_flips_the_sharded_default_without_touching_explicit_callers():
    src = LEAF.read_text()
    i = src.index("def sharded_quantized_matmul"); body = src[i:src.index("def ", i + 10)]
    assert "if maybe_quantize_x and not envs.TPU_ONLINE_QUANT_ACT:" in body
    assert "envs.TPU_ONLINE_QUANT_ACT" in src[src.index("_should_quantize_act ="):][:200], (
        "the batched path hardcoded itemsize>1 and ignored the env")


def test_env_is_forwarded_to_ray_workers():
    plat = (ROOT / "tpu_inference" / "platforms" / "tpu_platform.py").read_text()
    i = plat.index("additional_env_vars"); block = plat[i:i + 1500]
    assert '"TPU_ONLINE_QUANT_DTYPE"' in block, (
        "TPU_ONLINE_QUANT_DTYPE missing from the Ray passthrough: a multi-host int8 lane would serve fp8 on the workers (ti #29)")
    assert '"TPU_ONLINE_QUANT_ACT"' in block, (
        "a multi-host Ray run would silently fall back to W8A8 while the lane said W8A16 (the ti #29 shape)")
