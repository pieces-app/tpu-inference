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
"""fp32 attention logits for the Gemma-4 vision tower on the torchax path.

WHAT WAS WRONG
--------------
``Gemma4VisionAttention`` dispatches to transformers' ``sdpa`` interface,
which calls ``torch.nn.functional.scaled_dot_product_attention``.  Under
torchax that function is not a fused kernel: it is
``torchax.ops.jtorch._sdpa_reference``, a plain transcription that does

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

with **bf16 operands**, so the QK^T product is *materialised in bf16*.  Every
real SDPA backend -- torch's own CPU/flash/mem-efficient kernels, and the
flax path's ``sharded_flash_attention`` -- keeps the scores in fp32 and only
casts the attention *output* back down.

Gemma-4's vision attention runs with ``scaling = 1.0`` (the RMSNorm on q and
k replaces the ``1/sqrt(head_dim)`` factor), so the logits are the raw
64-term dot products and are numerically large.  Rounding them to bf16 (8
mantissa bits) perturbs each one by ~0.2 % of its magnitude, and that error
is then *exponentiated* by the softmax.

MEASURED, on CPU, at the real E4B tower shape (12 heads, head_dim 64,
10 080 query positions, 252 padded), against an fp64 reference:

    torch eager SDPA (bf16 in)                      cos 0.999959
    torchax _sdpa_reference (bf16 logits)  <-- today cos 0.999784
    bf16 operands, fp32 logits             <-- this  cos 0.999958

i.e. the bf16 logit buffer accounts for ~5x the attention error, and nothing
else does: with fp32 logits the torchax path lands on torch eager's own
accuracy.  Feeding fp32 *operands* to the matmul is exactly equivalent to
asking for an fp32 output from bf16 operands (measured identical to 9
decimals), and it is the only formulation expressible in torch ops.

WHAT IT DOES *NOT* EXPLAIN -- OFF BY DEFAULT
--------------------------------------------
The same CPU differential, run end to end on google/gemma-4-E4B-it's own
vision weights (fetched from ``model.safetensors``) at all three live grids
(126x78, 117x84, 144x69), says this is NOT the E4B image degeneracy:

* torchax in fp32 reproduces torch eager in fp32 to cos 1.00000000 at every
  one of the 16 layers -- the mask, the RoPE, the clamps and the pooling are
  identical, there is no semantic difference to find;
* in bf16 the torchax tower tracks the eager tower to cos >= 0.99962 at
  every layer, i.e. inside the bf16 noise the two share;
* turning THIS patch on moves the tower output by ~2e-8 in cosine, because
  the trained tower's scores are small (q_norm 0.4062 * k_norm 1.2344) and
  the bf16 error is dominated by the projections both paths share.

So it ships behind ``PIECES_GEMMA4_VISION_ATTN_FP32``, default OFF: a
one-env-var arm for a chip run that wants the torchax lane's last numerical
difference from every other backend removed, not a fix for the degeneracy.

COST
----
Casting the operands to fp32 does not add TPU work: XLA:TPU's default
precision for an f32 dot is a single bf16 pass with fp32 accumulation, which
is what the MXU already did -- only the *output* buffer changes dtype.  The
score matrix would double in bytes, so the attention is computed in blocks
along the query axis (``PIECES_GEMMA4_VISION_ATTN_CHUNK``, default 1024):
one block's fp32 scores are ~5x SMALLER than today's dense bf16 matrix, so
this is a memory improvement as well as a numerical one.

SCOPE
-----
Instance-level replacement of ``Gemma4VisionAttention.forward`` on the
modules under a Gemma-4 vision tower, and nothing else.  Qwen3-VL, the
Gemma-4 audio tower and the language model are untouched, and
``config._attn_implementation`` stays ``"sdpa"`` so transformers' mask
builder keeps producing the same 4D boolean mask (a custom name there would
make ``_preprocess_mask_arguments`` skip mask creation entirely).
"""

from types import MethodType

import torch
import torch.nn as nn

from tpu_inference import envs
from tpu_inference.logger import init_logger

logger = init_logger(__name__)

# ``Gemma4VisionAttention`` under a Gemma-4 vision tower, by class name: the
# module tree is transformers', and importing the class here would pull the
# whole modeling file into the CPU gate.
VISION_ATTENTION_CLASS = "Gemma4VisionAttention"


def fp32_logit_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    chunk: int,
) -> torch.Tensor:
    """Bidirectional attention with the scores kept in fp32.

    Args:
        query, key, value: ``(batch, heads, length, head_dim)``, the layout
          ``Gemma4VisionAttention`` hands the attention interface.
        attention_mask: ``None``, or the 4D mask transformers' ``sdpa_mask``
          builds -- ``(batch, 1, q_length, kv_length)``, ``True`` = attend.
          A float additive mask is also accepted (added to the scores).
        scaling: the interface's ``scaling`` (1.0 for this tower).
        chunk: query-axis block size; ``<= 0`` computes the whole matrix.

    Returns:
        ``(batch, heads, length, head_dim)`` in ``value``'s dtype.
    """
    q_len = query.shape[-2]
    block = q_len if chunk is None or chunk <= 0 else min(chunk, q_len)
    key_f32 = key.float()
    outputs = []
    for start in range(0, q_len, block):
        stop = min(start + block, q_len)
        # fp32 scores. On TPU this is still one bf16 MXU pass with fp32
        # accumulation (XLA's DEFAULT precision for an f32 dot); what
        # changes is that the product is no longer rounded back to bf16.
        scores = torch.matmul(query[:, :, start:stop].float(),
                              key_f32.transpose(-2, -1))
        if scaling != 1.0:
            scores = scores * scaling
        if attention_mask is not None:
            mask = attention_mask
            if mask.shape[-2] != 1:
                mask = mask[:, :, start:stop, :]
            if mask.dtype == torch.bool:
                scores = torch.where(
                    mask, scores,
                    torch.full_like(scores,
                                    torch.finfo(torch.float32).min))
            else:
                scores = scores + mask.float()
        probs = torch.softmax(scores, dim=-1).to(value.dtype)
        outputs.append(torch.matmul(probs, value))
    if len(outputs) == 1:
        return outputs[0]
    return torch.cat(outputs, dim=-2)


def _vision_attention_forward(
    self,
    hidden_states: torch.Tensor,
    position_embeddings: torch.Tensor = None,
    attention_mask: torch.Tensor | None = None,
    position_ids: torch.LongTensor | None = None,
    **kwargs,
):
    """Replaces ``Gemma4VisionAttention.forward`` on the torchax path.

    Provenance, for review against the upstream source
    (transformers/models/gemma4/modeling_gemma4.py,
    ``Gemma4VisionAttention.forward``): the projections, the two RMSNorms,
    the multidimensional RoPE, the transposes, the reshape and ``o_proj``
    are copied verbatim.  The single change is the last step: instead of
    ``ALL_ATTENTION_FUNCTIONS.get_interface(...)`` -- which under torchax is
    ``_sdpa_reference`` and materialises the scores in bf16 -- the scores
    are computed in fp32 by ``fp32_logit_attention``.  ``self.scaling`` and
    the mask are passed through unchanged, and ``attn_weights`` is ``None``
    exactly as the sdpa interface returns it.
    """
    from transformers.models.gemma4.modeling_gemma4 import \
        apply_multidimensional_rope

    # [upstream forward, verbatim]
    input_shape = hidden_states.shape[:-1]
    hidden_shape = (*input_shape, -1, self.head_dim)

    cos, sin = position_embeddings

    query_states = self.q_proj(hidden_states).view(hidden_shape)
    query_states = self.q_norm(query_states)
    query_states = apply_multidimensional_rope(query_states, cos, sin,
                                               position_ids)
    query_states = query_states.transpose(1, 2)

    key_states = self.k_proj(hidden_states).view(hidden_shape)
    key_states = self.k_norm(key_states)
    key_states = apply_multidimensional_rope(key_states, cos, sin,
                                             position_ids)
    key_states = key_states.transpose(1, 2)

    value_states = self.v_proj(hidden_states).view(hidden_shape)
    value_states = self.v_norm(value_states)
    value_states = value_states.transpose(1, 2)

    # [replaces the ALL_ATTENTION_FUNCTIONS dispatch] Same math, fp32 scores.
    # This tower is MHA (num_key_value_heads == num_attention_heads), so the
    # interface's repeat_kv is a no-op here; refuse the GQA case rather than
    # attend against the wrong heads.
    if key_states.shape[1] != query_states.shape[1]:
        raise ValueError(
            "gemma4 vision fp32-logit attention expects MHA, got "
            f"{query_states.shape[1]} query heads and {key_states.shape[1]} "
            "key/value heads")
    attn_output = fp32_logit_attention(
        query_states,
        key_states,
        value_states,
        attention_mask,
        self.scaling,
        self._pieces_attn_chunk,
    )

    # [upstream forward, verbatim] -- the sdpa interface transposes before
    # returning, so the transpose that belongs to it is inlined here.
    attn_output = attn_output.transpose(1, 2).contiguous()
    attn_output = attn_output.reshape(*input_shape, -1).contiguous()
    attn_output = self.o_proj(attn_output)
    return attn_output, None


# The marker `apply_gemma4_vision_attention_patch` reads to stay idempotent:
# a bound method falls back to its function's attributes, so an already
# patched module answers True and an unpatched one answers False.
_vision_attention_forward._pieces_fp32_logits = True


def _vision_attention_modules(vision_tower: nn.Module) -> list[nn.Module]:
    return [
        m for m in vision_tower.modules()
        if type(m).__name__ == VISION_ATTENTION_CLASS
    ]


def apply_gemma4_vision_attention_patch(vision_tower: nn.Module,
                                        chunk: int) -> int:
    """Install ``_vision_attention_forward`` on the tower's attentions.

    Returns the number of modules patched.  Idempotent: a module whose
    ``forward`` is already ours is left alone.
    """
    patched = 0
    for module in _vision_attention_modules(vision_tower):
        if getattr(module.forward, "_pieces_fp32_logits", False):
            continue
        module._pieces_attn_chunk = chunk
        module.forward = MethodType(_vision_attention_forward, module)
        patched += 1
    return patched


def maybe_apply_gemma4_vision_attention_patch(vllm_model: nn.Module) -> int:
    """Patch the Gemma-4 vision tower's attention, when there is one."""
    if not envs.PIECES_GEMMA4_VISION_ATTN_FP32:
        # Default. The CPU differential (see the module docstring) shows the
        # bf16 scores are not what makes E4B images degenerate, so nothing
        # changes on the lane unless the flag is set.
        return 0
    vision_tower = getattr(vllm_model, "vision_tower", None)
    if vision_tower is None:
        return 0
    chunk = envs.PIECES_GEMMA4_VISION_ATTN_CHUNK
    patched = apply_gemma4_vision_attention_patch(vision_tower, chunk)
    if patched:
        logger.info(
            "[gemma4-patch] vision attention: fp32 scores on %d modules "
            "(query chunk=%s). torchax's _sdpa_reference materialises the "
            "QK^T product in bf16; every other backend keeps it in fp32.",
            patched, chunk or "off")
    return patched
