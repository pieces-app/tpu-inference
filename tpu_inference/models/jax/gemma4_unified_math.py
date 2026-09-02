# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Pure-JAX math for the Gemma-4 *Unified* encoder-free vision embedder.

Kept in its own leaf with NO tpu_inference/vllm imports on purpose: this is
the only genuinely new arithmetic in the native 12B port, and it must be
testable on a CPU-only jax install against an independent NumPy reference.
Everything else in the port is assembly of layers that already serve the 26B.

Reference: vllm/model_executor/models/gemma4_unified.py
(Gemma4UnifiedVisionEmbedder) -- pipeline is
    raw patches -> LN1 -> Dense(+bias) -> LN2 -> +factorized posemb -> LN3
with NO pooler and NO vision transformer. The Unified checkpoint carries exactly
ten vision tensors; there is nothing else to run.
"""
import jax
import jax.numpy as jnp

# The Gemma4ImageProcessor pads pixel_position_ids with -1 for patches beyond
# the real image. A padded patch gets a ZERO positional embedding (both axes
# masked), and is stripped after projection by the caller.
POSITIONS_PAD_VALUE = -1


def factorized_posemb(pos_embedding: jax.Array,
                      positions_xy: jax.Array) -> jax.Array:
    """Factorized 2-D positional embedding.

    Args:
      pos_embedding: [mm_posemb_size, 2, D] -- axis 1 selects the x/y table.
      positions_xy:  [..., 2] int, (x, y) per patch, -1 for padding.

    Returns [..., D]: table_x[x] + table_y[y], with each axis contribution
    zeroed where that axis position is -1. Clamping to 0 before the gather
    keeps the index in range; the mask is what removes the contribution --
    the two are both required, which is why the negative control in the test
    drops the mask and expects divergence.
    """
    clamped = jnp.maximum(positions_xy, 0)
    valid = positions_xy != POSITIONS_PAD_VALUE
    out = jnp.zeros(positions_xy.shape[:-1] + (pos_embedding.shape[-1], ),
                    pos_embedding.dtype)
    for axis in range(2):
        table = pos_embedding[:, axis, :]  # [S, D]
        gathered = jnp.take(table, clamped[..., axis], axis=0)  # [..., D]
        mask = valid[..., axis][..., None].astype(gathered.dtype)
        out = out + gathered * mask
    return out


def layer_norm(x: jax.Array, weight: jax.Array, bias: jax.Array,
               eps: float) -> jax.Array:
    """torch.nn.LayerNorm semantics: biased variance over the last axis,
    statistics in float32, affine applied, result cast back to x.dtype."""
    xf = x.astype(jnp.float32)
    mean = jnp.mean(xf, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(xf - mean), axis=-1, keepdims=True)
    y = (xf - mean) * jax.lax.rsqrt(var + eps)
    y = y * weight.astype(jnp.float32) + bias.astype(jnp.float32)
    return y.astype(x.dtype)


def unified_vision_embed(pixel_values: jax.Array, positions_xy: jax.Array, *,
                         ln1_w, ln1_b, dense_w, dense_b, ln2_w, ln2_b,
                         pos_embedding, pos_norm_w, pos_norm_b,
                         eps: float) -> jax.Array:
    """The whole embedder as one pure function, for the reference test.

    dense_w is [patch_dim, mm_embed_dim] (einsum/JAX layout, i.e. the torch
    [out, in] weight already transposed).

    NOTE eps: these are torch.nn.LayerNorm layers, whose default eps is 1e-5.
    That is NOT the text stack's rms_norm_eps (1e-6). Passing the RMSNorm eps
    here would be a silent numerical mismatch against the reference.
    """
    h = layer_norm(pixel_values.astype(pos_embedding.dtype), ln1_w, ln1_b, eps)
    h = jnp.einsum("bpd,dh->bph", h, dense_w) + dense_b
    h = layer_norm(h, ln2_w, ln2_b, eps)
    h = h + factorized_posemb(pos_embedding, positions_xy)
    return layer_norm(h, pos_norm_w, pos_norm_b, eps)
