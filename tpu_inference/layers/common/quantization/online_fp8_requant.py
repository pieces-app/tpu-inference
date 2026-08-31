"""On-the-fly fp8 (e4m3) weight requantization for bf16 dense checkpoints.

Leaf module: imports only jax, so tests can load it by file path (bypassing
the tpu_inference package __init__ that pulls vllm/torchax) and verify the
numerics on a CPU-only jax install. Issue #158.
"""
import jax.numpy as jnp

# e4m3 (float8_e4m3fn) max representable magnitude.
E4M3_MAX = 448.0


def online_fp8_requant_per_channel(weight_out_in):
    """bf16 dense weight [out, in] -> (e4m3 weight [out, in], f32 scale [out]).

    Per-output-channel: scale = (max |w| over the input axis) / 448, so the
    largest magnitude in each output row maps to the top of the e4m3 range.
    The dequant is `w_fp8.astype(f32) * scale[:, None]`.
    """
    w = weight_out_in.astype(jnp.float32)
    amax = jnp.max(jnp.abs(w), axis=-1, keepdims=True)
    scale = jnp.maximum(amax / E4M3_MAX, jnp.float32(1e-12))
    w_fp8 = (w / scale).astype(jnp.float8_e4m3fn)
    return w_fp8, jnp.squeeze(scale, axis=-1)
