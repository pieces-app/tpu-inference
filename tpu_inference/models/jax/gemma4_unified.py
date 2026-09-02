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
"""Gemma-4 *Unified* (encoder-free) multimodal model, JAX-native.

Serves google/gemma-4-12b-it (Gemma4UnifiedForConditionalGeneration) on the
flax_nnx path instead of the vLLM/torchax fallback.

WHY THIS EXISTS. The Unified architecture was absent from the flax_nnx
registry, so the loader logged "Resolved MODEL_IMPL_TYPE 'auto' to 'flax_nnx'"
and on the very next line fell back to torchax. Two consequences, both
measured on v6e 2026-09-01:
  * MTP was IMPOSSIBLE: the Gemma4-MTP proposer shares the target's
    embed_tokens into the drafter and requires a flax_nnx target to find it
    (spec_decode/jax/eagle3.py) -- eval-12b-mtp refused at load with
    "(draft=True, target=False)".
  * Audio worked ONLY on the fallback, because Gemma4ForConditionalGeneration
    (the tower variant) skips audio weights unconditionally.
So on the 12B one could have audio OR speculative decoding, never both. This
class has both.

WHAT THE UNIFIED CHECKPOINT ACTUALLY IS (677 tensors, read from the
safetensors header, NOT assumed):
  664  model.language_model.layers.*   48 layers, heterogeneous attention --
       40 sliding (GQA 16:8, head_dim 256) + 8 full-attention at
       5/11/17/23/29/35/41/47 (16 heads x 512, ONE kv head, attention_k_eq_v
       so NO v_proj). Gemma4Model already handles this topology; the 31B runs
       it on v6e today.
   10  model.vision_embedder.*  a linear patch embedder. NO vision
       transformer: patch_ln1 -> patch_dense(+bias) -> patch_ln2 ->
       +factorized 2-D posemb -> pos_norm. patch_dense is [3840, 6912],
       6912 = model_patch_size(48)^2 * 3.
    1  model.embed_vision.embedding_projection   [3840, 3840]
    1  model.embed_audio.embedding_projection    [3840, 640] -- the ENTIRE
       audio path. Frames of 640 raw samples (audio_samples_per_token=640)
       are RMS-normed and projected straight into text space; the text stack
       does the audio understanding.
    2  model.language_model.{embed_tokens, norm}

NO CUSTOM KERNELS. Every op here is one the 26B flax path already runs (RPA v3
attention, RMSNorm, SwiGLU, einsum) plus three LayerNorms and a gather.

The multimodal front end is DELIBERATELY NOT QUANTIZED (quant_config=None):
the ten vision tensors are ~24 MB, quantizing them buys nothing, and the
torchax lane excludes them for the same reason (ti #22).
"""
import functools
from typing import Any, Callable, Iterable, List, Optional, Tuple, TypedDict

import jax
import jax.numpy as jnp
import torch
from flax import nnx
from jax.sharding import Mesh, NamedSharding, PartitionSpec
from transformers import PretrainedConfig
from vllm.config import VllmConfig
from vllm.model_executor.models.gemma4_unified import \
    Gemma4UnifiedForConditionalGeneration as PtGemma4Unified
from vllm.model_executor.models.utils import WeightsMapper

from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.linear import JaxEinsum
from tpu_inference.layers.jax.norm import JaxLayerNorm
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.gemma4 import Gemma4ForCausalLM, Gemma4Model
from tpu_inference.models.jax.gemma4_mm import (POSITIONS_PAD_VALUE,
                                                 Gemma4ImagePixelInputs,
                                                 Gemma4MultimodalEmbedder,
                                                 init_fn)
from tpu_inference.models.jax.gemma4_unified_math import factorized_posemb
from tpu_inference.models.jax.jax_intermediate_tensor import \
    JaxIntermediateTensors
from tpu_inference.models.jax.utils.multi_modal_utils import \
    merge_multimodal_embeddings
from tpu_inference.models.jax.utils.weight_utils import (
    JaxAutoWeightsLoader, LoadableWithIterator, StandardWeightLoader)

logger = init_logger(__name__)

# torch.nn.LayerNorm's default. NOT the text stack's rms_norm_eps (1e-6): the
# three embedder norms are LayerNorms with bias, and the reference constructs
# them with no eps argument. Using 1e-6 here would be a silent mismatch.
_LAYERNORM_EPS = 1e-5


class Gemma4AudioInputs(TypedDict):
    """Raw per-frame audio features from Gemma4UnifiedAudioFeatureExtractor."""
    type: str
    input_features_padded: jax.Array  # (bn, T, 640) -- feature_size 640
    input_features_mask: jax.Array  # (bn, T) bool


class Gemma4UnifiedVisionEmbedder(JaxModule):
    """Encoder-free vision embedder: patches -> LN -> Dense -> LN -> +pos -> LN.

    Mirrors vllm/model_executor/models/gemma4_unified.py
    Gemma4UnifiedVisionEmbedder op-for-op. No pooler: one embedding per
    VALID patch, padding stripped by the caller.
    """

    def __init__(self,
                 config: PretrainedConfig,
                 dtype: jnp.dtype,
                 rng: nnx.Rngs,
                 prefix: str = ""):
        patch_dim = config.model_patch_size**2 * 3
        mm_embed_dim = config.mm_embed_dim
        self.patch_dim = patch_dim
        self.mm_embed_dim = mm_embed_dim

        self.patch_ln1 = JaxLayerNorm(patch_dim,
                                      epsilon=_LAYERNORM_EPS,
                                      param_dtype=dtype,
                                      rngs=rng)
        self.patch_dense = JaxEinsum(
            "bpd,dh->bph",
            (patch_dim, mm_embed_dim),
            rngs=rng,
            bias_shape=(mm_embed_dim, ),
            param_dtype=dtype,
            kernel_init=nnx.with_partitioning(init_fn, (None, None)),
            bias_init=nnx.with_partitioning(nnx.initializers.zeros, (None, )),
            quant_config=None,
            prefix=f"{prefix}.patch_dense",
        )
        self.patch_ln2 = JaxLayerNorm(mm_embed_dim,
                                      epsilon=_LAYERNORM_EPS,
                                      param_dtype=dtype,
                                      rngs=rng)
        # [mm_posemb_size, 2, D]: axis 1 selects the x / y table. Loaded
        # as-is (3-D, default loader is identity -- weight_utils
        # jax_array_from_reshaped_torch only auto-transposes 2-D).
        self.pos_embedding = nnx.Param(
            jnp.zeros((config.mm_posemb_size, 2, mm_embed_dim), dtype=dtype))
        self.pos_norm = JaxLayerNorm(mm_embed_dim,
                                     epsilon=_LAYERNORM_EPS,
                                     param_dtype=dtype,
                                     rngs=rng)

    def __call__(self, pixel_values: jax.Array,
                 pixel_position_ids: jax.Array) -> jax.Array:
        # ORDER MATTERS and is asserted by a source-structure test:
        # ln1 -> dense -> ln2 -> +posemb -> pos_norm.
        h = self.patch_ln1(pixel_values.astype(self.pos_embedding.value.dtype))
        h = self.patch_dense(h)
        h = self.patch_ln2(h)
        h = h + factorized_posemb(self.pos_embedding.value, pixel_position_ids)
        return self.pos_norm(h)


class Gemma4UnifiedModel(JaxModule):
    """Container mirroring the checkpoint layout:
        model.language_model.*   text backbone (Gemma4Model, reused as-is)
        model.vision_embedder.*  encoder-free patch embedder
        model.embed_vision.*     3840 -> 3840 (RMSNorm w/o scale + projection)
        model.embed_audio.*      640  -> 3840 (same class; the whole audio path)
    """

    def __init__(self,
                 vllm_config: VllmConfig,
                 rng: nnx.Rngs,
                 mesh: Mesh,
                 prefix: str = "model"):
        model_config = vllm_config.model_config
        hf = model_config.hf_config
        vision_config = hf.vision_config
        audio_config = getattr(hf, "audio_config", None)
        text_hidden = hf.text_config.hidden_size
        dtype = model_config.dtype

        self.language_model = Gemma4Model(
            vllm_config=vllm_config,
            rng=rng,
            mesh=mesh,
            prefix=prefix + ".language_model",
        )
        self.vision_embedder = Gemma4UnifiedVisionEmbedder(
            config=vision_config,
            dtype=dtype,
            rng=rng,
            prefix=f"{prefix}.vision_embedder",
        )
        # Gemma4MultimodalEmbedder = RMSNorm(no scale) -> projection. The
        # vLLM reference uses the SAME class for both connectors, sized by
        # output_proj_dims (3840 vision / 640 audio). Its docstring claims a
        # post-projection norm; its CODE builds a pre-projection norm -- the
        # code is what the checkpoint matches (only embedding_projection.weight
        # exists), and the JAX class mirrors the code.
        self.embed_vision = Gemma4MultimodalEmbedder(
            vision_hidden_size=(getattr(vision_config, "output_proj_dims",
                                        None) or vision_config.mm_embed_dim),
            text_hidden_size=text_hidden,
            dtype=dtype,
            rng=rng,
            quant_config=None,
            prefix=f"{prefix}.embed_vision",
            rms_norm_eps=vision_config.rms_norm_eps,
        )
        if audio_config is not None:
            self.embed_audio = Gemma4MultimodalEmbedder(
                vision_hidden_size=(getattr(audio_config, "output_proj_dims",
                                            None) or audio_config.hidden_size),
                text_hidden_size=text_hidden,
                dtype=dtype,
                rng=rng,
                quant_config=None,
                prefix=f"{prefix}.embed_audio",
                rms_norm_eps=audio_config.rms_norm_eps,
            )
        else:
            self.embed_audio = None


class Gemma4UnifiedForConditionalGeneration(JaxModule, LoadableWithIterator):
    packed_modules_mapping = Gemma4ForCausalLM.packed_modules_mapping
    WeightLoader = StandardWeightLoader
    supports_multimodal = True
    # The tower variant shards a ViT across the VIT_BATCH axis and routes
    # images through MMEncoderJITManager. This front end is a few matmuls;
    # it runs through the plain, MODALITY-GENERIC run_embed_multimodal path
    # (models/common/model_loader.py), which is also the only path that
    # delivers AUDIO kwargs. Both flags stay False on purpose; turning the
    # cudagraph manager on later is an optimisation, not a correctness item.
    supports_encoder_tp_data = False
    supports_encoder_cudagraph = False
    _processor_factory = getattr(PtGemma4Unified, "_processor_factory", None)

    def __init__(self, vllm_config: VllmConfig, rng_key: jax.Array,
                 mesh: Mesh) -> None:
        self.vllm_config = vllm_config
        rng = nnx.Rngs(rng_key)
        self.mesh = mesh

        self.model = Gemma4UnifiedModel(vllm_config=vllm_config,
                                        rng=rng,
                                        mesh=mesh,
                                        prefix="model")
        model_config = vllm_config.model_config
        hf = model_config.hf_config
        vision_config = hf.vision_config
        self.image_token_id = getattr(hf, "image_token_id", 258880)
        self.audio_token_id = getattr(hf, "audio_token_id", 258881)
        # Max patches per image == the positional table size (1120). There is
        # no pooler, so this is also the max soft tokens per image.
        self.max_soft_tokens = vision_config.mm_posemb_size
        self.patch_pixels = vision_config.model_patch_size**2 * 3

        self.final_logit_softcapping = getattr(hf.text_config,
                                               "final_logit_softcapping",
                                               None)

        if not hf.tie_word_embeddings:
            if self.model.language_model.is_last_rank:
                from tpu_inference.layers.jax.linear import JaxLmHead
                self.lm_head = JaxLmHead(
                    hidden_size=hf.text_config.hidden_size,
                    vocab_size=model_config.get_vocab_size(),
                    param_dtype=model_config.dtype,
                    kernel_init=nnx.with_partitioning(init_fn,
                                                      ("model", None)),
                    rngs=rng,
                    prefix="lm_head",
                )
            else:
                from tpu_inference.layers.jax.pp_utils import PPMissingLayer
                self.lm_head = PPMissingLayer()

    # ------------------------------------------------------------ weights
    def load_weights(self, weights: Iterable[Tuple[str, Any]]):
        hf = self.vllm_config.model_config.hf_config
        # Fail CLOSED both ways on audio: the tower variant silently dropped a
        # 752-tensor audio encoder (ti #34). Here the checkpoint's audio path
        # is ONE tensor and we serve it, so a config that declares audio must
        # have produced an embed_audio, and vice versa.
        declares_audio = getattr(hf, "audio_config", None) is not None
        if declares_audio != (self.model.embed_audio is not None):
            raise ValueError(
                f"audio_config declared={declares_audio} but embed_audio "
                f"built={self.model.embed_audio is not None}; refusing to "
                "load a model whose audio capability disagrees with its "
                "config.")
        mapper = WeightsMapper(orig_to_new_prefix={"model.lm_head.": "lm_head."})
        loader = JaxAutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head"] if not hasattr(self, "lm_head") else []),
            # ONLY quantization statistics. NO "audio_tower"/"embed_audio":
            # the whole point of this class is that it loads them.
            skip_substrs=[".input_max", ".input_min", ".output_max",
                          ".output_min"],
        )
        return loader.load_weights(mapper.apply(weights))

    # ------------------------------------------------------------ embeddings
    def embed_input_ids(self,
                        input_ids: jax.Array,
                        multimodal_embeddings: Optional[jax.Array] = None,
                        **kwargs) -> jax.Array:
        inputs_embeds = self.model.language_model.embed_tokens(input_ids)
        target_dtype = inputs_embeds.dtype
        inputs_embeds = (inputs_embeds *
                         self.model.language_model.embedding_scale).astype(
                             target_dtype)
        if multimodal_embeddings is not None and multimodal_embeddings.shape[
                0] > 0:
            # BOTH placeholder ids: image AND audio soft tokens are merged.
            inputs_embeds = merge_multimodal_embeddings(
                input_ids, inputs_embeds, multimodal_embeddings,
                [self.image_token_id, self.audio_token_id])
        return inputs_embeds.astype(target_dtype)

    @jax.jit
    def get_single_image_embedding(
            self, pixel_values: jax.Array,
            pixel_position_ids: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """(b, P, 6912), (b, P, 2) -> ((b, P, hidden), (b, P) valid mask)."""
        embedded = self.model.vision_embedder(pixel_values, pixel_position_ids)
        projected = self.model.embed_vision(embedded)
        valid = jnp.logical_not(
            jnp.all(pixel_position_ids == POSITIONS_PAD_VALUE, axis=-1))
        projected = jax.lax.with_sharding_constraint(
            projected, NamedSharding(self.mesh, PartitionSpec(None, None,
                                                              None)))
        return projected, valid

    @jax.jit
    def get_audio_embedding(self, input_features: jax.Array) -> jax.Array:
        """(b, T, 640) raw frames -> (b, T, hidden). That is the whole path."""
        return self.model.embed_audio(input_features)

    def _parse_and_validate_image_input(
            self, **kwargs: object) -> Optional[Gemma4ImagePixelInputs]:
        pixel_values = kwargs.pop("pixel_values", None)
        pixel_position_ids = kwargs.pop("pixel_position_ids", None)
        assert kwargs.pop("image_embeds", None) is None, \
            "Gemma4 does not support image_embeds."
        if pixel_values is None:
            return None
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values.contiguous().view(
                torch.int16).numpy().view(jnp.bfloat16)
            pixel_values = jnp.asarray(pixel_values)
        if isinstance(pixel_position_ids, torch.Tensor):
            pixel_position_ids = jnp.asarray(
                pixel_position_ids.to(torch.int32).contiguous().numpy())
        return Gemma4ImagePixelInputs(type="pixel_values",
                                      pixel_values=pixel_values,
                                      pixel_position_ids=pixel_position_ids)

    def _parse_and_validate_audio_input(
            self, **kwargs: object) -> Optional[Gemma4AudioInputs]:
        feats = kwargs.pop("input_features_padded", None)
        mask = kwargs.pop("input_features_mask", None)
        if feats is None:
            return None
        if isinstance(feats, torch.Tensor):
            if feats.dtype == torch.bfloat16:
                feats = jnp.asarray(
                    feats.contiguous().view(torch.int16).numpy().view(
                        jnp.bfloat16))
            else:
                feats = jnp.asarray(feats.to(torch.float32).contiguous().numpy())
        if isinstance(mask, torch.Tensor):
            mask = jnp.asarray(mask.to(torch.bool).contiguous().numpy())
        if mask is None:
            # All dims but the feature dim: (bn, T) for the documented rank-3
            # input AND (T,) for the single-item rank-2 input that
            # _process_audio_input expands. `shape[:2]` was right only for
            # rank-3; for rank-2 it produced a (T, 640) mask that the gather
            # `emb[i][mask[i]]` cannot index -- a runtime-only crash on the
            # branch nothing exercised (review 2026-09-02).
            mask = jnp.ones(feats.shape[:-1], dtype=bool)
        return Gemma4AudioInputs(type="input_features",
                                 input_features_padded=feats,
                                 input_features_mask=mask)

    def _process_image_input(
            self, image_input: Gemma4ImagePixelInputs) -> list[jax.Array]:
        pv = image_input["pixel_values"]
        pp = image_input["pixel_position_ids"]
        if pv.ndim == 2:
            pv = jnp.expand_dims(pv, 0)
        if pp.ndim == 2:
            pp = jnp.expand_dims(pp, 0)
        # One image per call: the processor pads every image to the same
        # patch count, so the JIT shape is fixed and the cache holds one
        # entry. Padding (-1 positions) is stripped HERE, outside the jit,
        # because the valid count differs per image.
        out: list[jax.Array] = []
        for i in range(pv.shape[0]):
            proj, valid = self.get_single_image_embedding(pv[i:i + 1],
                                                          pp[i:i + 1])
            out.append(proj[0][valid[0]])
        return out

    def _process_audio_input(
            self, audio_input: Gemma4AudioInputs) -> list[jax.Array]:
        feats = audio_input["input_features_padded"]
        mask = audio_input["input_features_mask"]
        if feats.ndim == 2:
            feats = jnp.expand_dims(feats, 0)
            mask = jnp.expand_dims(mask, 0)
        # `.weight`, NOT `.kernel`: JaxEinsum (layers/jax/linear.py) aliases the
        # nnx.Einsum param to `weight` and then `delattr(self, 'kernel')` so the
        # HF-style name matches. Reading `.kernel` raised AttributeError and
        # killed the whole EngineCore -- MEASURED 2026-09-02 17:06Z, native 12B,
        # on the first audio request (text and vision were already serving).
        # Nothing caught it earlier because this line only runs on live audio.
        target_dtype = self.model.embed_audio.embedding_projection.weight.value.dtype
        emb = self.get_audio_embedding(feats.astype(target_dtype))
        return [emb[i][mask[i]] for i in range(emb.shape[0])]

    def embed_multimodal(self, **kwargs) -> List[jax.Array]:
        # The runner batches one modality per call
        # (multimodal_manager.py group_and_batch_mm_kwargs).
        image_input = self._parse_and_validate_image_input(**dict(kwargs))
        if image_input is not None:
            return self._process_image_input(image_input)
        audio_input = self._parse_and_validate_audio_input(**dict(kwargs))
        if audio_input is not None:
            if self.model.embed_audio is None:
                raise ValueError(
                    "received audio inputs but this checkpoint declares no "
                    "audio_config -- refusing rather than embedding garbage")
            return self._process_audio_input(audio_input)
        return []

    def get_input_modality(self, mm_kwargs: dict[str, Any]) -> str:
        if "pixel_values_videos" in mm_kwargs:
            raise NotImplementedError("Video not yet supported.")
        if "input_features_padded" in mm_kwargs:
            return "audio"
        return "image"

    # ------------------------------------------------------------ forward
    def __call__(
        self,
        kv_caches: List[jax.Array],
        input_ids: jax.Array,
        attention_metadata: Any,
        inputs_embeds: Optional[jax.Array] = None,
        _input_positions=None,
        _layer_name_to_kv_cache=None,
        _lora_metadata=None,
        intermediate_tensors: JaxIntermediateTensors | None = None,
        is_first_rank: bool = True,
        is_last_rank: bool = True,
        *args,
    ) -> Tuple[List[jax.Array], jax.Array | JaxIntermediateTensors,
               List[jax.Array], Optional[jax.Array]]:
        multimodal_embeddings = getattr(attention_metadata,
                                        "multimodal_embeddings", None)
        if inputs_embeds is None:
            inputs_embeds = self.embed_input_ids(input_ids,
                                                 multimodal_embeddings)
        if not is_first_rank and intermediate_tensors is not None:
            inputs_embeds = intermediate_tensors["hidden_states"]
        layer_name_to_kv_cache = dict(
            _layer_name_to_kv_cache) if _layer_name_to_kv_cache else None
        is_multimodal = ((input_ids == self.image_token_id) |
                         (input_ids == self.audio_token_id)
                         ) if input_ids is not None else None
        kv_caches, x, expert_indices = self.model.language_model(
            kv_caches,
            input_ids,
            attention_metadata,
            inputs_embeds,
            layer_name_to_kv_cache=layer_name_to_kv_cache,
            is_multimodal=is_multimodal,
        )
        if not is_last_rank:
            x = JaxIntermediateTensors(tensors={"hidden_states": x})
        return kv_caches, x, [], expert_indices

    def compute_logits(self, hidden_states: jax.Array) -> jax.Array:
        if hasattr(self, "lm_head"):
            logits = self.lm_head(hidden_states)
        else:
            logits = self.model.language_model.embed_tokens.decode(
                hidden_states)
        if self.final_logit_softcapping is not None:
            logits = jnp.tanh(logits / self.final_logit_softcapping
                              ) * self.final_logit_softcapping
        return logits

    def precompile_vision_encoder(self, run_compilation_fn: Callable) -> None:
        # Pre-patchified input: the processor pads every image to
        # mm_posemb_size patches of model_patch_size^2*3 pixels, so ONE shape
        # covers every image. (The tower variant iterates image_shapes from
        # additional_config; here there is nothing to iterate.)
        from tpu_inference import utils
        dtype_str = str(self.vllm_config.model_config.dtype).split(".")[-1]
        jax_dtype = utils.get_jax_dtype_from_str_dtype(dtype_str)
        pv = jnp.ones((1, self.max_soft_tokens, self.patch_pixels),
                      dtype=jax_dtype)
        pp = jnp.ones((1, self.max_soft_tokens, 2), dtype=jnp.int32)
        run_compilation_fn("vision_encoder",
                           self.get_single_image_embedding,
                           pv,
                           pp,
                           image_shape=[self.max_soft_tokens,
                                        self.patch_pixels])
