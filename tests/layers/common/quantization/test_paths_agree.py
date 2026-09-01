"""The two online quantization paths must produce the SAME weights.

There are two of them, and they serve the two model configs that get compared
against each other:

  * `quantization/__init__.py::quantize_tensor`        -> flax/nnx  (26B MoE)
  * `online_fp8_requant.py::online_fp8_requant_...`    -> vllm/torchax (12B dense)

They used to disagree, because the flax one reduced in the TENSOR's dtype
while the torchax one cast to float32 first. On a bf16 checkpoint that is a
bf16-rounded amax and a bf16-rounded quotient; the trailing `.astype(float32)`
cannot recover what the rounding already threw away.

MEASURED before the fix, [4096,1024] bf16 weight:

    dtype     scale rel-diff   recon err flax   recon err torchax
    e4m3fn        0.22%            0.19484          0.18115
    int8          0.28%            0.04591          0.02826

The cost landed ~8x harder on int8 -- finer resolution makes a fixed relative
scale error a larger share of total error -- so the defect SYSTEMATICALLY
understated int8 against fp8 on the flax path. That is the exact comparison
the dtype matrix exists to make, on the half of the matrix that runs flax.
Two instruments disagreeing by more than the effect under test is not a
measurement.
"""
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
COMMON = ROOT / "tpu_inference" / "layers" / "common" / "quantization"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _paths():
    pytest.importorskip("jax", reason="numeric check needs jax")
    return (_load("_q_leaf", COMMON / "__init__.py"),
            _load("_r_leaf", COMMON / "online_fp8_requant.py"))


@pytest.mark.parametrize("name", [
    "float8_e4m3fn", "float8_e4m3b11fnuz", "float8_e5m2", "int8",
])
def test_both_paths_quantize_a_bf16_weight_identically(name):
    import jax
    import jax.numpy as jnp
    q, r = _paths()
    dtype = r.ONLINE_QUANT_DTYPES[name]

    # bf16 specifically: that is what the checkpoints hold, and it is the only
    # input dtype for which the two paths ever disagreed. An f32 input hides
    # the bug completely.
    w = jax.random.normal(jax.random.PRNGKey(7),
                          (512, 128)).astype(jnp.bfloat16)

    w_flax, s_flax = q.quantize_tensor(dtype, w, axis=0)
    w_tx, s_tx = r.online_fp8_requant_per_channel(w, dtype=dtype)

    assert w_flax.dtype == w_tx.dtype == dtype
    assert s_flax.shape == s_tx.shape

    s_rel = float(jnp.max(jnp.abs(s_flax - s_tx)) / jnp.max(jnp.abs(s_tx)))
    assert s_rel == 0.0, (
        f"{name}: the two paths disagree on the scale by {s_rel:.8f}. The "
        f"26B (flax) and 12B (torchax) arms of the dtype matrix would then "
        f"differ by the instrument as well as by the dtype.")

    assert jnp.array_equal(w_flax.astype(jnp.float32),
                           w_tx.astype(jnp.float32)), (
        f"{name}: same weight, same dtype, different quantized values")


def test_the_reduction_happens_in_f32_not_the_input_dtype():
    """Guard with teeth: a bf16 input must give the same scale as its own f32
    upcast. That is false whenever the reduction runs in the input dtype, and
    it is the property the trailing .astype(float32) cannot provide."""
    import jax
    import jax.numpy as jnp
    q, _ = _paths()
    w32 = jax.random.normal(jax.random.PRNGKey(11), (512, 128))
    wbf = w32.astype(jnp.bfloat16)

    for dtype in (jnp.float8_e4m3fn, jnp.int8):
        _, s_from_bf16 = q.quantize_tensor(dtype, wbf, axis=0)
        # The reference: the SAME values, already widened. Any difference is
        # the reduction's precision, not the data's.
        _, s_from_f32 = q.quantize_tensor(dtype, wbf.astype(jnp.float32),
                                          axis=0)
        assert jnp.array_equal(s_from_bf16, s_from_f32), (
            f"{dtype.__name__}: reducing a bf16 tensor gives a different "
            f"scale than reducing its exact f32 upcast -- the reduction is "
            f"still running in the input dtype")
