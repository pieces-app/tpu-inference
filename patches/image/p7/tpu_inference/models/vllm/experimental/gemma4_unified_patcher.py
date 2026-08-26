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
"""Gemma4 *Unified* (encoder-free) fidelity patch for the torchax path.

Why this exists — measured on gemma-4-12b-it
(``Gemma4UnifiedForConditionalGeneration``):

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

i.e. >=5% of the image's soft tokens are effectively noise. Computing the
LayerNorm statistics in fp32 matches the fp32 eager reference to ~1e-5
relative. These are three norms over a ~1120-token sequence, run once per
image — the fp32 cost is nil next to the LM forward.

SCOPE: this patch replaces ONLY the ``forward`` of the embedder's three
``nn.LayerNorm`` submodules. The embedder's own ``forward`` — and with it
every tensor shape, dtype bridge, and parallel-linear call in the vLLM
implementation — is left exactly as upstream wrote it. An earlier version
re-stated the whole embedder forward (including a functional
``F.linear`` for the ``ColumnParallelLinear``) and died in the vision path
with an einsum shape mismatch; re-deriving upstream's plumbing is both
unnecessary and fragile, since the measured defect is entirely in the
LayerNorm statistics.
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

_LN_ATTRS = ("patch_ln1", "patch_ln2", "pos_norm")


def _f32_layer_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """``nn.LayerNorm.forward`` with fp32 statistics.

    Same normalized_shape/eps/affine parameters as the module it replaces;
    only the arithmetic precision changes. The result is cast back to the
    input dtype so the surrounding graph is shape- and dtype-identical to
    the unpatched module.
    """
    orig_dtype = x.dtype
    f32 = torch.float32
    weight = self.weight.to(f32) if self.weight is not None else None
    bias = self.bias.to(f32) if self.bias is not None else None
    out = F.layer_norm(
        x.to(f32),
        self.normalized_shape,
        weight,
        bias,
        self.eps,
    )
    return out.to(orig_dtype)


def apply_gemma4_unified_patches(vllm_model: nn.Module) -> None:
    embedder = getattr(vllm_model, "vision_embedder", None)
    if embedder is None:
        return

    patched = []
    for attr in _LN_ATTRS:
        norm = getattr(embedder, attr, None)
        if not isinstance(norm, nn.LayerNorm):
            logger.warning(
                "[gemma4-unified-patch] %s is not an nn.LayerNorm (%s); "
                "skipping fp32 statistics for it.", attr, type(norm).__name__)
            continue
        norm.forward = MethodType(_f32_layer_norm_forward, norm)
        patched.append(attr)

    if patched:
        logger.info(
            "[gemma4-unified-patch] vision embedder LayerNorms %s now compute "
            "statistics in fp32 under torchax (bf16 mean/var corrupts "
            "near-flat image patches; see gemma4_unified_patcher.py).",
            ", ".join(patched))


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
            "Gemma4UnifiedVisionEmbedder; skipping fp32 LayerNorm patch.")
        return
    apply_gemma4_unified_patches(vllm_model)
