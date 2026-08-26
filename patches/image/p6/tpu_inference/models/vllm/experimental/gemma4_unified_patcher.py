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
"""Gemma4 *Unified* (encoder-free) fidelity patches for the torchax path.

Why this exists — measured on gemma-4-12b-it (Gemma4UnifiedForConditionalGeneration):

The Unified variant has no vision tower. Raw pixel patches go through
``Gemma4UnifiedVisionEmbedder``: LayerNorm -> Dense -> LayerNorm ->
+factorized posemb -> LayerNorm. On the TPU path this runs in bf16 under
torchax, whose ``aten.native_layer_norm`` lowering computes mean/var in the
*input dtype* (``jnp.mean``/``jnp.var`` on bf16), unlike PyTorch native
eager, which accumulates LayerNorm statistics in fp32 even for bf16 tensors.

For screenshot-class images this is catastrophic on near-flat patches
(uniform background, faint fine text): patch std ~= 0.005 is the same order
as the bf16 quantization step at pixel magnitude ~1.0, so the bf16
mean/variance estimate — and with it rstd — is badly wrong, and LayerNorm
*amplifies* the error by ~1/std. Measured against the fp32 reference on a
real screenshot (CPU-JAX, tools/gemma4_unified_vision_diff/):

    stage                torch-eager bf16     torchax bf16
    LN1 out              max|d| 1.3           max|d| 13.1
    LN2 out              max|d| 1.8           max|d| 438.4   (p95 across
    soft tokens (S7)     rel 0.31%, cos 1.0   rel 4.4%, cos 0.994  tokens!)

i.e. >=5% of the image's soft tokens are effectively noise. Running the
embedder in fp32 under torchax matches the fp32 eager reference to ~1e-5
relative. The embedder is ~27M params and runs once per image — fp32
compute here is free relative to the LM forward.

The patch replaces ``Gemma4UnifiedVisionEmbedder.forward`` with a
numerically-identical fp32 version (weights stay bf16 in HBM; they are
upcast inline, which XLA folds into cheap converts) and casts the result
back to the original activation dtype at the end, so everything downstream
(``embed_vision`` — whose RMSNorm is already fp32-internal — and the merge
into text embeddings) is unchanged.
"""

from types import MethodType

import torch
import torch.nn as nn
import torch.nn.functional as F

from tpu_inference.logger import init_logger

logger = init_logger(__name__)

try:
    from vllm.model_executor.models.gemma4_unified import (
        Gemma4UnifiedForConditionalGeneration, Gemma4UnifiedVisionEmbedder)
    _HAVE_GEMMA4_UNIFIED = True
except ImportError:  # pragma: no cover - older vLLM without the model
    _HAVE_GEMMA4_UNIFIED = False


def _f32_vision_embedder_forward(
    self,
    pixel_values: torch.Tensor,
    pixel_position_ids: torch.Tensor,
) -> torch.Tensor:
    """fp32 re-statement of Gemma4UnifiedVisionEmbedder.forward.

    Identical math to upstream (LN1 -> Dense -> LN2 -> +posemb -> LN3),
    with every op computed in fp32. The output is cast back to the
    original parameter dtype so downstream consumers see no change.
    """
    f32 = torch.float32
    out_dtype = self.pos_embedding.dtype

    ln1_w = self.patch_ln1.weight.to(f32)
    ln1_b = self.patch_ln1.bias.to(f32)
    hidden_states = F.layer_norm(
        pixel_values.to(f32),
        self.patch_ln1.normalized_shape,
        ln1_w,
        ln1_b,
        self.patch_ln1.eps,
    )

    # ColumnParallelLinear applied functionally: full (out, in) weight +
    # bias. On the tpu-inference torchax path parallelism is expressed via
    # GSPMD sharding annotations on the weights, so F.linear is exact.
    hidden_states = F.linear(
        hidden_states,
        self.patch_dense.weight.to(f32),
        self.patch_dense.bias.to(f32) if self.patch_dense.bias is not None
        else None,
    )

    hidden_states = F.layer_norm(
        hidden_states,
        self.patch_ln2.normalized_shape,
        self.patch_ln2.weight.to(f32),
        self.patch_ln2.bias.to(f32),
        self.patch_ln2.eps,
    )

    # Factorized 2D posemb, upstream math with an fp32 table view.
    pos_table = self.pos_embedding.to(f32)
    clamped_pos = pixel_position_ids.clamp(min=0).long()
    valid_mask = pixel_position_ids != -1
    pos_embs = torch.zeros(
        *pixel_position_ids.shape[:-1],
        pos_table.shape[-1],
        device=pixel_position_ids.device,
        dtype=f32,
    )
    for i in range(2):
        axis_pe = pos_table[:, i, :][clamped_pos[..., i]]
        mask = valid_mask[..., i].unsqueeze(-1).to(f32)
        pos_embs = pos_embs + (axis_pe * mask)

    hidden_states = hidden_states + pos_embs
    hidden_states = F.layer_norm(
        hidden_states,
        self.pos_norm.normalized_shape,
        self.pos_norm.weight.to(f32),
        self.pos_norm.bias.to(f32),
        self.pos_norm.eps,
    )
    return hidden_states.to(out_dtype)


def apply_gemma4_unified_patches(vllm_model: nn.Module) -> None:
    embedder = getattr(vllm_model, "vision_embedder", None)
    if embedder is None:
        return
    embedder.forward = MethodType(_f32_vision_embedder_forward, embedder)
    logger.info(
        "[gemma4-unified-patch] vision_embedder.forward now computes in "
        "fp32 under torchax (bf16 LayerNorm statistics corrupt near-flat "
        "patches; see gemma4_unified_patcher.py).")


def maybe_apply_gemma4_unified_patches(vllm_model: nn.Module) -> None:
    if not _HAVE_GEMMA4_UNIFIED:
        return
    if not isinstance(vllm_model, Gemma4UnifiedForConditionalGeneration):
        return
    if not isinstance(getattr(vllm_model, "vision_embedder", None),
                      Gemma4UnifiedVisionEmbedder):
        # Already wrapped/patched or unexpected layout; leave untouched.
        logger.warning(
            "[gemma4-unified-patch] vision_embedder is not the expected "
            "Gemma4UnifiedVisionEmbedder; skipping fp32 patch.")
        return
    apply_gemma4_unified_patches(vllm_model)
