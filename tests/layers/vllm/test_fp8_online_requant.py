"""Numerics of the online fp8 (e4m3) dense requant (issue #158).

Loads the leaf module `layers/common/quantization/online_fp8_requant.py` BY
FILE PATH so it runs on a CPU-only jax install without importing the
tpu_inference package (which pulls vllm/torchax). Asserts the per-output-
channel scale is amax/448 within ~1 ULP and the dequant round-trip is within
e4m3 tolerance. `pytest tests/layers/vllm/test_fp8_online_requant.py`
(needs jax; skips without it).

Negative control (watched via fork_gate / by perturbing E4M3_MAX): a wrong
divisor makes the scale assertion fail.
"""
import importlib.util
from pathlib import Path

import pytest

jnp = pytest.importorskip("jax.numpy")
import jax  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[3]
LEAF = (REPO_ROOT / "tpu_inference" / "layers" / "common" / "quantization" /
        "online_fp8_requant.py")


def _load_leaf():
    spec = importlib.util.spec_from_file_location("online_fp8_requant", LEAF)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_scale_is_amax_over_448_per_channel():
    m = _load_leaf()
    key = jax.random.PRNGKey(0)
    w = (jax.random.normal(key, (64, 128)) * 3.0).astype(jnp.bfloat16)
    _, scale = m.online_fp8_requant_per_channel(w)
    amax = jnp.max(jnp.abs(w.astype(jnp.float32)), axis=-1)
    assert scale.shape == (64, )
    assert float(jnp.max(jnp.abs(scale - amax / 448.0))) < 1e-4


def test_round_trip_within_e4m3_tolerance():
    m = _load_leaf()
    key = jax.random.PRNGKey(1)
    w = (jax.random.normal(key, (32, 256)) * 5.0).astype(jnp.bfloat16)
    w_fp8, scale = m.online_fp8_requant_per_channel(w)
    rt = w_fp8.astype(jnp.float32) * scale[:, None]
    ref = w.astype(jnp.float32)
    rel = jnp.max(jnp.abs(rt - ref) / (jnp.abs(ref) + 1e-6))
    # e4m3 has ~3 mantissa bits; per-channel max-scaled round-trip stays well
    # under 2^-2. A per-tensor or wrong-divisor scheme blows past this.
    assert float(rel) < 0.25


def test_no_empty_scale_ever_created():
    """The garbage trap the fail-closed guard names is a torch.empty() scale
    the bf16 loader cannot fill. The online method must compute the scale, so
    the leaf never allocates an uninitialized one."""
    src = LEAF.read_text()
    assert "torch.empty" not in src
    assert "amax" in src and "/ E4M3_MAX" in src.replace(" ", " ")

