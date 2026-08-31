# Copyright 2025 Google LLC
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

from dataclasses import InitVar, dataclass
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax import nnx
from flax.typing import Sharding
from jax import lax
from jax.sharding import Mesh
from jax.sharding import PartitionSpec as P

from tpu_inference import envs, utils
from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ragged_paged_attention
from tpu_inference.layers.common.attention_metadata import AttentionMetadata
from tpu_inference.layers.common.quantization import quantize_kv
from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.rope_interface import apply_rope

KVCache = Tuple[jax.Array, jax.Array]


def _rpa_block_size_kwargs() -> dict[str, tuple[int, int, int, int]]:
    """Optional RPA v3 block-size overrides from env, for the call site to
    forward with ``**``.

    Each ``RPA_V3_*_BLOCK_SIZES`` env var is a comma-separated 4-tuple
    ``(bq_sz, bkv_sz, bq_csz, bkv_csz)``. A key is included only when its env
    var is non-empty, so the default call forwards nothing and stays identical
    to the stock invocation (and never passes a v3-only kwarg to a kernel whose
    signature doesn't accept it).
    """
    env_to_kwarg = {
        "d_block_sizes": envs.RPA_V3_DECODE_BLOCK_SIZES,
        "p_block_sizes": envs.RPA_V3_PREFILL_BLOCK_SIZES,
        "m_block_sizes": envs.RPA_V3_MIXED_BLOCK_SIZES,
    }
    return {k: tuple(v) for k, v in env_to_kwarg.items() if v}


@dataclass(kw_only=True)
class Attention(nnx.Module):
    """An implementation of attention.

    This module performs the attention mechanism for a transformer model,
    including query, key, and value projections, application of Rotary
    Position Embeddings (RoPE), and management of a KV cache for efficient
    autoregressive generation. It supports both prefill and generation
    (decode) modes and handles tensor sharding for distributed computation.

    Attributes:
        mesh: The JAX device mesh for distributed computation.
    """
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rope_theta: float
    rope_scaling: dict[str, Any]
    dtype: jnp.dtype
    mesh: Mesh
    kv_cache_dtype: str

    dnh_sharding: Sharding = ()
    dkh_sharding: Sharding = ()
    nhd_sharding: Sharding = ()

    activation_q_td: P = P(ShardingAxisName.ATTN_DATA)
    query_tnh: P = P(ShardingAxisName.ATTN_DATA)
    keyvalue_skh: P = P(ShardingAxisName.ATTN_DATA)

    attn_o_tnh: P = P(ShardingAxisName.ATTN_DATA)
    rngs: InitVar[nnx.Rngs]

    random_init: bool = False
    attention_chunk_size: int | None = None
    rope_input_ordering: str = "split"

    _q_scale: float = 1.0
    _k_scale: float = 1.0
    _v_scale: float = 1.0

    kv_cache_quantized_dtype = None

    def __post_init__(self, rngs: nnx.Rngs):
        """Initializes the weight kernels for Q, K, V, and O projections."""
        N = self.num_attention_heads
        K = self.num_key_value_heads
        D = self.hidden_size
        H = self.head_dim

        self.kernel_q_proj_DNH = create_param(rngs, (D, N, H),
                                              self.dnh_sharding,
                                              self.dtype,
                                              random_init=self.random_init)
        self.kernel_k_proj_DKH = create_param(rngs, (D, K, H),
                                              self.dkh_sharding,
                                              self.dtype,
                                              random_init=self.random_init)
        self.kernel_v_proj_DKH = create_param(rngs, (D, K, H),
                                              self.dkh_sharding,
                                              self.dtype,
                                              random_init=self.random_init)
        self.kernel_o_proj_NHD = create_param(rngs, (N, H, D),
                                              self.nhd_sharding,
                                              self.dtype,
                                              random_init=self.random_init)

        if self.kv_cache_dtype != "auto":
            self.kv_cache_quantized_dtype = utils.get_jax_dtype_from_str_dtype(
                self.kv_cache_dtype)

    def __call__(self,
                 x,
                 is_prefill,
                 kv_cache: KVCache,
                 attention_metadata: AttentionMetadata,
                 use_attention_rope: bool = True):
        """Performs the forward pass of the attention module.

        This method computes the attention output by projecting the input `x`
        to queries, keys, and values, applying RoPE, performing scaled
        dot-product attention, and projecting the result back to the model
        dimension. It updates and utilizes a KV cache.

        Args:
            x: The input tensor of shape `(seq_len, d_model)`.
            is_prefill: Whether the operation mode is prefill (otherwise it is generate).
            kv_cache: The key-value cache for storing past attention states.
            attention_metadata: Metadata for attention, such as input positions.
            use_attention_rope: Whether to use RoPE.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(batch_size, seq_len, d_model)`.
        """
        md = attention_metadata
        x_SD = jnp.asarray(x, self.dtype)
        x_q_TD = lax.with_sharding_constraint(x, self.activation_q_td)
        H = self.head_dim
        with jax.named_scope("q_proj"):
            q_TNH = jnp.einsum('TD,DNH -> TNH', x_q_TD,
                               self.kernel_q_proj_DNH.value)
            if use_attention_rope:
                q_TNH = apply_rope(q_TNH, md.input_positions, H,
                                   self.rope_theta, self.rope_scaling,
                                   self.rope_input_ordering)
            q_TNH = lax.with_sharding_constraint(q_TNH, self.query_tnh)
        with jax.named_scope("k_proj"):
            k_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_k_proj_DKH.value)
            if use_attention_rope:
                k_SKH = apply_rope(k_SKH, md.input_positions, H,
                                   self.rope_theta, self.rope_scaling,
                                   self.rope_input_ordering)
            k_SKH = lax.with_sharding_constraint(k_SKH, self.keyvalue_skh)

        with jax.named_scope("v_proj"):
            v_SKH = jnp.einsum('SD,DKH -> SKH', x_SD,
                               self.kernel_v_proj_DKH.value)

        q_scale = k_scale = v_scale = None
        if self.kv_cache_quantized_dtype:
            # TODO(kyuyeunk/jacobplatin): Enable w8a8 when VREG spill issue is resolved.
            # q_scale = self._q_scale
            k_scale = self._k_scale
            v_scale = self._v_scale
            k_SKH, v_SKH = quantize_kv(self.kv_cache_quantized_dtype, k_SKH,
                                       v_SKH, k_scale, v_scale)

        with jax.named_scope("attn_op"):
            new_kv_cache, outputs_TNH = self.attention(
                is_prefill,
                kv_cache,
                q_TNH,
                k_SKH,
                v_SKH,
                attention_metadata,
                self.mesh,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
            )

        with jax.named_scope("o_proj"):
            o_TD = jnp.einsum('TNH,NHD -> TD', outputs_TNH,
                              self.kernel_o_proj_NHD.value)
        return new_kv_cache, o_TD

    def attention(
        self,
        is_prefill: bool,
        kv_cache: KVCache,
        q_TNH: jax.Array,
        k_SKH: jax.Array,
        v_SKH: jax.Array,
        attention_metadata: AttentionMetadata,
        mesh: Mesh,
        q_scale: float | None = None,
        k_scale: float | None = None,
        v_scale: float | None = None,
    ) -> Tuple[KVCache, jax.Array]:
        """Performs scaled dot-product attention and updates the KV cache.

        This function handles the core attention logic, which varies between
        prefill and generation modes. In prefill, it computes self-attention
        over the input sequence with a causal mask. In generation, it attends
        to the full history of keys and values stored in the cache.

        Args:
            is_prefill: A boolean indicating if the mode is 'prefill'.
            kv_cache: The key-value cache to be updated and used.
            q_TNH: Query tensor of shape `(query_seq, num_attention_heads, head_dim)`.
            k_SKH: Key tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            v_SKH: Value tensor of shape `(kv_seq, num_key_value_heads, head_dim)`.
            attention_metadata: Metadata containing sequence lengths.
            mesh: The JAX device mesh (unused in this specific function but
                kept for potential future use or API consistency).
            q_scale: Quantization scale for q.
            k_scale: Quantization scale for k.
            v_scale: Quantization scale for v.

        Returns:
            A tuple containing:
                - The updated KV cache.
                - The attention output tensor of shape
                  `(seq, num_q_heads, head_dim)`.
        """
        md = attention_metadata
        kv_cache_spec = P(ShardingAxisName.ATTN_DATA, None, "model")
        in_specs = (
            self.query_tnh,  # q
            self.keyvalue_skh,  # k
            self.keyvalue_skh,  # v
            kv_cache_spec,  # kv_cache
            P(ShardingAxisName.ATTN_DATA),  # md.seq_lens
            P(ShardingAxisName.ATTN_DATA),  # page_indices_flat
            P(ShardingAxisName.ATTN_DATA),  # query_start_loc
            P(ShardingAxisName.ATTN_DATA),  # distribution
        )

        # Gemma-4 blockwise-bidirectional image attention. The runner builds
        # md.mm_bidi_ranges (i32[max_num_seqs, 2]) only when TPURunner._init_mm_bidi
        # enabled it; the torchax path forwards it (flash_attn.py) and the v3
        # RPA kernel consumes it, but this flax_nnx path historically dropped
        # it, so tower models (31B / 26B-A4B) served a causal-only mask under
        # TPU_MM_BIDI_ATTENTION=force. Thread it as an extra replicated operand
        # exactly as layers/common/attention_interface.py does. When it is None
        # (text-only, or the feature disabled at startup) nothing changes.
        has_mm_bidi = md.mm_bidi_ranges is not None
        if has_mm_bidi and envs.USE_BATCHED_RPA_KERNEL:
            # Invariant, not runtime policy: _init_mm_bidi refuses to build the
            # ranges under the batched RPA kernel, so reaching here means that
            # gate was bypassed. Fail loud rather than serve a wrong mask.
            raise NotImplementedError(
                "mm_bidi_ranges (PrefixLM blockwise attention) is not supported "
                "by the batched RPA kernel; TPURunner._init_mm_bidi should have "
                "disabled it when USE_BATCHED_RPA_KERNEL=1.")
        if has_mm_bidi:
            in_specs = in_specs + (P(ShardingAxisName.ATTN_DATA), )  # mm_bidi_ranges

        out_specs = (self.attn_o_tnh, kv_cache_spec)

        def _ragged_paged_attention(*args):
            # mm_bidi_ranges rides as the last operand when present (so it is
            # sharded/replicated by the shard_map like every other array),
            # then peeled back off here and forwarded to the kernel as a
            # keyword, exactly as layers/common/attention_interface.py does.
            if has_mm_bidi:
                *args, mm_ranges = args
            else:
                mm_ranges = None
            # Optional operator block-size overrides, read at the call site
            # (outside the kernel, so the kernel stays self-contained) and
            # forwarded only when set. Each env var is a comma-separated 4-tuple
            # (bq_sz, bkv_sz, bq_csz, bkv_csz); an empty list (the default) is
            # omitted entirely, so the stock call is unchanged and no v3-only
            # kwarg is passed to a kernel that doesn't accept it.
            block_size_kwargs = _rpa_block_size_kwargs()
            if mm_ranges is not None:
                block_size_kwargs["mm_bidi_ranges"] = mm_ranges
            return ragged_paged_attention(
                *args,
                sm_scale=q_TNH.shape[-1]**-0.5,
                q_scale=q_scale,
                k_scale=k_scale,
                v_scale=v_scale,
                **block_size_kwargs,
            )

        output_TNH, kv_cache = jax.jit(
            jax.shard_map(
                _ragged_paged_attention,
                mesh=mesh,
                in_specs=in_specs,
                out_specs=out_specs,
                check_vma=False,
            ))(
                *(
                    q_TNH,
                    k_SKH,
                    v_SKH,
                    kv_cache,
                    md.seq_lens,
                    md.block_tables,
                    md.query_start_loc,
                    md.request_distribution,
                ) + ((md.mm_bidi_ranges, ) if has_mm_bidi else ())
            )
        return kv_cache, output_TNH
