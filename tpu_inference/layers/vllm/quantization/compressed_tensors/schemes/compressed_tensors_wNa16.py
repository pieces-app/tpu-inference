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
"""Weight-only int4 (W4A16 / "wNa16") scheme for compressed-tensors
pack-quantized checkpoints on the torchax path.

Covers checkpoints whose quantization_config is
    weights: {num_bits: 4, type: int, symmetric: true, strategy: group}
    input_activations: null
    format: pack-quantized
e.g. Google's official ``gemma-4-*-it-qat-w4a16-ct`` QAT exports (group 32)
and RedHatAI's ``*-INT4`` GPTQ exports (group 128, actorder "static" — no
``weight_g_idx`` tensors are serialized for that mode, verified against the
checkpoints; only ``actorder: group`` serializes g_idx and is rejected below).

Storage layout after load: weights stay PACKED as ``jnp.int4`` ``[in, out]``
plus bf16 group scales ``[in // group_size, out]``. By default,
dequantization to bf16 happens per-forward inside
``sharded_quantized_matmul``'s XLA path (the 2D scale triggers blockwise
weight dequant and disables activation quantization), mirroring the AWQ
scheme's dequant-then-einsum approach. With
``ENABLE_QUANTIZED_MATMUL_KERNEL=1`` the scale is instead formatted to 4D,
which routes the matmul to the tokamax ``gmm_v2`` fused kernel: weights
stay int4-packed in HBM and are dequantized tile-wise in VMEM
(``maybe_quantize_lhs=False`` keeps activations bf16 — W4A16 semantics
either way). The memory win (int4-resident weights -> larger KV arena)
holds on both routes; the kernel route removes the per-forward bf16 weight
materialization that made the XLA route memory-bound.
"""

from typing import Callable, Optional

import jax
import jax.numpy as jnp
import torch
from compressed_tensors import CompressionFormat
from compressed_tensors.quantization import (ActivationOrdering,
                                             QuantizationArgs,
                                             QuantizationStrategy,
                                             QuantizationType)
from jax.sharding import PartitionSpec
from torch.nn.parameter import Parameter
from torchax.interop import jax_view, torch_view
from torchax.ops.mappings import t2j
from vllm.model_executor.layers.quantization.compressed_tensors.schemes import \
    CompressedTensorsScheme
from vllm.model_executor.layers.quantization.utils.quant_utils import \
    unpack_quantized_values_into_int32
from vllm.model_executor.parameter import (BasevLLMParameter,
                                           ChannelQuantScaleParameter,
                                           GroupQuantScaleParameter,
                                           PackedvLLMParameter)
from vllm.scalar_type import scalar_types

from tpu_inference.layers.common.linear import sharded_quantized_matmul
from tpu_inference.layers.common.process_weights.linear_weights import (
    LinearWeights, process_linear_weights, shard_linear_weights,
    to_parameter_list)
from tpu_inference.layers.common.utils import \
    slice_sharded_tensor_for_concatenation
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig
from tpu_inference.logger import init_logger

P = PartitionSpec
logger = init_logger(__name__)

WNA16_SUPPORTED_BITS = (4, )


def is_wNa16_group(weight_quant: Optional[QuantizationArgs],
                   input_quant: Optional[QuantizationArgs],
                   quant_format: Optional[str]) -> bool:
    """True for weight-only int4 group quantization in pack-quantized format.

    Self-contained predicate (does not rely on upstream helper availability):
    static int weights, group strategy, no input activation quantization.
    """
    if weight_quant is None or input_quant is not None:
        return False
    if quant_format != CompressionFormat.pack_quantized.value:
        return False
    return (weight_quant.type == QuantizationType.INT
            and not weight_quant.dynamic
            and weight_quant.strategy == QuantizationStrategy.GROUP.value
            and weight_quant.num_bits in WNA16_SUPPORTED_BITS)


class VllmCompressedTensorsWNA16(CompressedTensorsScheme):
    """Weight-only int4 group-quantized linear scheme (W4A16), torchax path."""

    def __init__(
        self,
        weight_quant: QuantizationArgs,
        linear_config: VllmQuantLinearConfig,
    ):
        self.pack_factor = 32 // weight_quant.num_bits
        self.strategy = weight_quant.strategy
        self.num_bits = weight_quant.num_bits
        self.symmetric = weight_quant.symmetric
        self.group_size = (-1 if weight_quant.group_size is None else
                           weight_quant.group_size)
        # Only ``actorder: group`` serializes per-column g_idx tensors that
        # must be honored at dequant time; "weight"/"static" reorder only
        # inside groups at quantization time and need no runtime handling.
        self.has_g_idx = weight_quant.actorder == ActivationOrdering.GROUP

        if self.num_bits not in WNA16_SUPPORTED_BITS:
            raise NotImplementedError(
                f"WNA16 TPU scheme supports num_bits in {WNA16_SUPPORTED_BITS}"
                f", got {self.num_bits}.")
        if not self.symmetric:
            raise NotImplementedError(
                "WNA16 TPU scheme supports symmetric quantization only "
                "(no weight_zero_point).")
        if self.group_size <= 0:
            raise NotImplementedError(
                "WNA16 TPU scheme supports group strategy only (channelwise "
                "int4 would need a mixed-dtype matmul on the XLA path).")
        if self.has_g_idx:
            raise NotImplementedError(
                "WNA16 TPU scheme does not support actorder=group "
                "checkpoints (weight_g_idx handling not implemented).")

        self.wtype = scalar_types.uint4
        self.weight_quant = weight_quant
        self.linear_config = linear_config

    @classmethod
    def get_min_capability(cls) -> int:
        # Capability gating is a CUDA concept; the TPU platform does not
        # consult it. Match the most permissive value.
        return 0

    def create_weights(
        self,
        layer: torch.nn.Module,
        output_size: int,
        input_size: int,
        output_partition_sizes: list[int],
        input_size_per_partition: int,
        params_dtype: torch.dtype,
        weight_loader: Callable,
        **kwargs,
    ):
        # Identical layout to VllmCompressedTensorsW4A8Fp8.create_weights:
        # the on-disk tensors (weight_packed / weight_scale / weight_shape)
        # are the same pack-quantized format.
        output_size_per_partition = sum(output_partition_sizes)

        group_size = self.group_size
        row_parallel = input_size != input_size_per_partition
        partition_scales = not row_parallel

        scales_and_zp_size = input_size // group_size
        if partition_scales:
            assert input_size_per_partition % group_size == 0
            scales_and_zp_size = input_size_per_partition // group_size

        weight = PackedvLLMParameter(
            data=torch.empty(
                output_size_per_partition,
                input_size_per_partition // self.pack_factor,
                dtype=torch.int32,
            ),
            input_dim=1,
            output_dim=0,
            packed_dim=1,
            packed_factor=self.pack_factor,
            weight_loader=weight_loader,
        )

        weight_scale_args = {
            "data":
            torch.empty(
                output_size_per_partition,
                scales_and_zp_size,
                dtype=params_dtype,
            ),
            "weight_loader":
            weight_loader,
        }
        if partition_scales:
            weight_scale = GroupQuantScaleParameter(output_dim=0,
                                                    input_dim=1,
                                                    **weight_scale_args)
        else:
            weight_scale = ChannelQuantScaleParameter(output_dim=0,
                                                      **weight_scale_args)

        weight_shape = BasevLLMParameter(data=torch.empty(2,
                                                          dtype=torch.int64),
                                         weight_loader=weight_loader)

        layer.register_parameter("weight_packed", weight)
        layer.register_parameter("weight_scale", weight_scale)
        layer.register_parameter("weight_shape", weight_shape)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        unpacked_weights = unpack_quantized_values_into_int32(
            layer.weight_packed, self.wtype, packed_dim=1)
        uint_weight = t2j(unpacked_weights, use_dlpack=False)
        delattr(layer, "weight_packed")
        weight_scale = t2j(layer.weight_scale, use_dlpack=False)
        delattr(layer, "weight_scale")

        if getattr(layer, "bias",
                   None) is not None and not layer.skip_bias_add:
            if layer.return_bias:
                logger.warning_once("Bias might return incorrect value.")
            bias = t2j(layer.bias, use_dlpack=False)
            delattr(layer, "bias")
        else:
            bias = None

        @jax.jit
        def process_wNa16_linear_weights(
            uint_weight: jax.Array,
            weight_scale: jax.Array,
            bias: jax.Array | None,
        ) -> LinearWeights:
            # Stored nibbles are offset-binary (value + 8); recenter to
            # signed int4 and keep the weights 4-bit resident in HBM.
            weight = (uint_weight - 8).astype(jnp.int4)
            weight = jnp.transpose(weight)  # -> [in, out]

            # -> [in // group_size, out]. With enable_kernel False the 2D
            # scale steers sharded_quantized_matmul into its blockwise
            # weight-dequant XLA path (materializes a bf16 weight copy per
            # forward). With ENABLE_QUANTIZED_MATMUL_KERNEL=1 the scale is
            # reformatted to 4D (format_linear_scale), which routes to the
            # tokamax gmm_v2 fused kernel: weights stay int4-packed in HBM
            # and are dequantized tile-wise in VMEM. Activation quantization
            # stays off on both routes (apply passes maybe_quantize_x=False,
            # i.e. gmm_v2 maybe_quantize_lhs=False) — W4A16 semantics.
            weight_scale = jnp.transpose(weight_scale)

            return process_linear_weights(
                LinearWeights(
                    weight=weight,
                    weight_scale=weight_scale,
                    zero_point=None,
                    bias=bias,
                ),
                fused=self.linear_config.fuse_matmuls,
                output_sizes=self.linear_config.output_sizes,
                reorder_size=self.linear_config.n_shards,
                enable_kernel=self.linear_config.
                enable_quantized_matmul_kernel,
            )

        weights = process_wNa16_linear_weights(uint_weight, weight_scale, bias)
        weights = torch_view(
            shard_linear_weights(
                weights,
                mesh=self.linear_config.mesh,
                weight_p_spec=self.linear_config.weight_sharding,
                bias_p_spec=self.linear_config.bias_sharding,
            ))

        if self.linear_config.fuse_matmuls:
            layer.weight = Parameter(weights.weight, requires_grad=False)
            layer.weight_scale = Parameter(weights.weight_scale,
                                           requires_grad=False)
            if bias is not None:
                layer.bias = Parameter(weights.bias, requires_grad=False)
        else:
            layer.weight = to_parameter_list(weights.weight)
            layer.weight_scale = to_parameter_list(weights.weight_scale)
            if bias is not None:
                layer.bias = to_parameter_list(weights.bias)

    def apply_weights(self, layer: torch.nn.Module, x: torch.Tensor,
                      bias: Optional[torch.Tensor]) -> torch.Tensor:
        with jax.named_scope(layer._get_name()):
            if self.linear_config.fuse_matmuls:
                return self._apply_fused(layer, x, bias)
            else:
                return self._apply_split(layer, x, bias)

    def _apply_fused(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        x_jax = jax_view(x)
        weight_jax = jax_view(layer.weight)
        weight_scale_jax = jax_view(layer.weight_scale)

        outs = sharded_quantized_matmul(
            x_jax,
            weight_jax,
            weight_scale_jax,
            self.linear_config.weight_sharding,
            mesh=self.linear_config.mesh,
            defer_all_reduce=self.linear_config.defer_all_reduce,
            # W4A16: activations stay bf16.
            maybe_quantize_x=False)

        if bias is not None and not layer.skip_bias_add:
            outs += jax_view(bias)
        outs = slice_sharded_tensor_for_concatenation(
            outs, self.linear_config.output_sizes, self.linear_config.n_shards)
        return torch_view(jnp.concatenate(outs, axis=-1))

    def _apply_split(self, layer: torch.nn.Module, x: torch.Tensor,
                     bias: Optional[torch.Tensor]) -> torch.Tensor:
        assert isinstance(layer.weight, torch.nn.ParameterList)

        x_jax = jax_view(x)
        outs = []
        for i, (weight, weight_scale) in enumerate(
                zip(layer.weight, layer.weight_scale)):
            weight_jax = jax_view(weight)
            weight_scale_jax = jax_view(weight_scale)

            out = sharded_quantized_matmul(
                x_jax,
                weight_jax,
                weight_scale_jax,
                self.linear_config.weight_sharding,
                mesh=self.linear_config.mesh,
                defer_all_reduce=self.linear_config.defer_all_reduce,
                maybe_quantize_x=False)

            if bias is not None and not layer.skip_bias_add:
                out += jax_view(bias[i])
            outs.append(out)
        return torch_view(jnp.concatenate(outs, axis=-1))
