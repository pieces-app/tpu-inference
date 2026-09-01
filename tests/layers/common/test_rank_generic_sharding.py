"""Two regressions that the rank-generic matmul change introduced today.

Both were found by adversarial review on 2026-09-01, both reproduce on CPU, and
both are INVISIBLE to a single-chip / single-contracting-axis test -- which is
why the suite that shipped with the original change stayed green.

1. `sharded_quantized_matmul` built length-2 PartitionSpecs. JAX pads a short
   spec with None on the TRAILING axes, so for a rank-3 activation the TP axis
   bound to axis 1 (TOKENS) and the out-feature axis was declared replicated.
   shard_map then assembled each shard's block along tokens and returned a
   real-looking, finite, WRONG tensor -- (1, T*tp, N/tp) instead of (1, T, N) --
   with no exception and correct magnitudes.

   It is CORRECT at tp=1, so every single-chip arm and every CPU test passed.

   Worse: before the rank-generic `dimension_numbers` change this call raised
   TypeError inside dot_general, so the mis-assembly was unreachable. Making the
   matmul rank-generic turned a loud crash into a silent wrong answer.

2. `apply_jax` captured `leading = x.shape[:-1]` while the flatten below removes
   ONE AXIS PER CONTRACTING AXIS. Gemma-4's o_proj is JaxEinsum("TNH,NHD->TD")
   with in_features == (num_heads, head_dim), so x is [T, N, H] and TWO axes are
   consumed. The restore then asked for [T, N, D] from a [T, D] result and raised
   on the first forward -- something the PRE-change code did not do.

Test 1 runs in a subprocess because forcing 4 CPU devices requires XLA_FLAGS to
be set before jax is imported.
"""
import ast
import pathlib
import subprocess
import sys
import textwrap

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
LINEAR = ROOT / "tpu_inference" / "layers" / "common" / "linear.py"
FP8 = ROOT / "tpu_inference" / "layers" / "jax" / "quantization" / "fp8.py"


# ------------------------------------------------------------------ finding 1
def test_sharded_quantized_matmul_pads_batch_dims_like_its_sibling():
    """Structural guard: both specs must be padded for extra leading dims.

    `sharded_matmul` has always done this (`batch_dims = (None,)*(x.ndim-2)`);
    `sharded_quantized_matmul` did not. Asserting the padding is present in BOTH
    specs, because padding only one still mis-binds the other.
    """
    src = LINEAR.read_text()
    tree = ast.parse(src)
    fn = next((n for n in ast.walk(tree)
               if isinstance(n, ast.FunctionDef)
               and n.name == "sharded_quantized_matmul"), None)
    assert fn is not None, "sharded_quantized_matmul not found"
    body = ast.get_source_segment(src, fn) or ""

    assert "batch_dims" in body, (
        "sharded_quantized_matmul must pad its PartitionSpecs for extra leading "
        "dims; a length-2 spec binds the TP axis to the TOKEN axis at rank 3")
    for spec in ("x_sharding", "out_sharding"):
        line = next((l for l in body.splitlines()
                     if l.strip().startswith(f"{spec} = P(")), None)
        assert line is not None, f"{spec} assignment not found"
        assert "*batch_dims" in line, (
            f"{spec} is not padded: {line.strip()!r}. Padding only one spec "
            f"still leaves the other mis-bound.")


@pytest.mark.parametrize("tp", [1, 2, 4])
def test_rank3_sharded_result_equals_the_rank2_flattened_result(tp):
    """The behavioural proof, at tp=1/2/4 on forced CPU devices.

    tp=1 passed BEFORE the fix too -- it is included precisely to show that a
    single-chip test could never have caught this.
    """
    pytest.importorskip("jax", reason="needs jax")
    script = textwrap.dedent(f'''
        import os
        os.environ["XLA_FLAGS"] = "--xla_force_host_platform_device_count=4"
        import importlib.util, sys, types
        import numpy as np

        def stub(name, **attrs):
            m = types.ModuleType(name)
            for k, v in attrs.items():
                setattr(m, k, v)
            sys.modules[name] = m
        for p in ("tokamax", "tokamax._src", "tokamax._src.ops",
                  "tokamax._src.ops.experimental",
                  "tokamax._src.ops.experimental.gmm_v2"):
            stub(p)
        stub("tokamax._src.ops.experimental.gmm_v2.gmm_v2",
             gmm_v2=lambda *a, **k: None)

        class _Axis(str):
            pass
        sh = types.ModuleType("tpu_inference.layers.common.sharding")
        sh.ShardingAxisName = type("S", (), {{n: _Axis(n) for n in
            ("ATTN_DATA","MLP","VOCAB","MODEL","DATA","EXPERT","ATTN_HEAD")}})
        sys.modules["tpu_inference.layers.common.sharding"] = sh
        lg = types.ModuleType("tpu_inference.logger")
        lg.init_logger = lambda *a, **k: type("L", (), {{"__getattr__":
            lambda s, n: (lambda *a, **k: None)}})()
        sys.modules["tpu_inference.logger"] = lg

        ROOT = {str(ROOT)!r}
        for pkg, d in (("tpu_inference.kernels", ROOT + "/tpu_inference/kernels"),
                       ("tpu_inference.kernels.quantized_matmul",
                        ROOT + "/tpu_inference/kernels/quantized_matmul")):
            m = types.ModuleType(pkg); m.__path__ = [d]; sys.modules[pkg] = m
        us = importlib.util.spec_from_file_location(
            "tpu_inference.kernels.quantized_matmul.util",
            ROOT + "/tpu_inference/kernels/quantized_matmul/util.py")
        um = importlib.util.module_from_spec(us)
        sys.modules["tpu_inference.kernels.quantized_matmul.util"] = um
        us.loader.exec_module(um)

        spec = importlib.util.spec_from_file_location("_lin", {str(LINEAR)!r})
        lin = importlib.util.module_from_spec(spec); spec.loader.exec_module(lin)

        import jax, jax.numpy as jnp
        from jax.sharding import Mesh, PartitionSpec as P
        qs = importlib.util.spec_from_file_location(
            "_q", ROOT + "/tpu_inference/layers/common/quantization/__init__.py")
        qm = importlib.util.module_from_spec(qs); qs.loader.exec_module(qm)

        tp = {tp}
        devs = np.array(jax.devices()[:tp]).reshape(1, tp)
        mesh = Mesh(devs, ("ATTN_DATA", "model"))
        in_f, out_f, T = 64, 32, 8
        x = jax.random.normal(jax.random.PRNGKey(0), (1, T, in_f)).astype(jnp.bfloat16)
        w = jax.random.normal(jax.random.PRNGKey(1), (in_f, out_f))
        w_q, w_s = qm.quantize_tensor(jnp.float8_e4m3fn, w, axis=0)
        with mesh:
            out = lin.sharded_quantized_matmul(x, w_q, w_s, P(None, "model"), mesh=mesh)
            flat = lin.sharded_quantized_matmul(
                x.reshape(1 * T, in_f), w_q, w_s, P(None, "model"), mesh=mesh)
        print("SHAPE", out.shape, flat.shape)
        ok = tuple(out.shape) == (1, T, out_f)
        same = bool(jnp.array_equal(out.reshape(-1, out_f).astype(jnp.float32),
                                    flat.astype(jnp.float32)))
        print("RESULT", ok, same)
    ''')
    r = subprocess.run([sys.executable, "-c", script], capture_output=True,
                       text=True, timeout=600)
    if "RESULT" not in r.stdout:
        # Deliberately NOT a skip. A skip here is a false green -- the whole
        # point of this test is a wrong ANSWER, so "the repro did not run" must
        # be red, not silence.
        raise AssertionError(
            f"sharded repro failed to produce a RESULT line.\n"
            f"stdout: {r.stdout[-600:]}\nstderr: {r.stderr[-1200:]}")
    ok, same = r.stdout.strip().splitlines()[-1].split()[1:3]
    assert ok == "True" and same == "True", (
        f"tp={tp}: rank-3 sharded result is wrong. {r.stdout.strip()}\n"
        f"A length-2 out_spec binds the TP axis to the TOKEN axis, so the "
        f"result is finite, correctly-scaled and MIS-ASSEMBLED.")


# ------------------------------------------------------------------ finding 2
@pytest.mark.parametrize("cls", ["Fp8TensorwiseLinearMethod",
                                 "Fp8BlockwiseLinearMethod"])
def test_apply_jax_restores_one_axis_per_contracting_axis(cls):
    """`x.shape[:-1]` is only right when there is exactly ONE contracting axis.

    Gemma-4's o_proj ("TNH,NHD->TD") has two, so the restore must consume
    len(in_features) axes. Structural, because constructing the real config
    needs a JaxEinsum and a live mesh.
    """
    src = FP8.read_text()
    tree = ast.parse(src)
    node = next((n for n in ast.walk(tree)
                 if isinstance(n, ast.ClassDef) and n.name == cls), None)
    assert node is not None, f"{cls} not found"
    fn = next((f for f in node.body
               if isinstance(f, ast.FunctionDef) and f.name == "apply_jax"), None)
    assert fn is not None, f"{cls}.apply_jax not found"
    body = ast.get_source_segment(src, fn) or ""

    assert "leading = x.shape[:-1]" not in body, (
        f"{cls}.apply_jax drops exactly one axis. o_proj "
        f'(JaxEinsum("TNH,NHD->TD")) contracts TWO, so this raises "cannot '
        f'reshape array of shape (T, D) into (T, N, D)" on the first forward.')
    assert "len(self.linear_config.in_features)" in body, (
        f"{cls}.apply_jax must consume one leading axis per CONTRACTING axis, "
        f"i.e. len(self.linear_config.in_features)")
