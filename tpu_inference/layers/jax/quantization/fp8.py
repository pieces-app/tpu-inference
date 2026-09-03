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

import functools
import math
from functools import partial
from typing import Iterable, Optional, Sequence

import jax
import jax.numpy as jnp
from flax import nnx
from jax.sharding import NamedSharding
from jax.sharding import PartitionSpec as P

from tpu_inference.layers.common.linear import sharded_quantized_batched_matmul
from tpu_inference.layers.common.moe import MoEBackend, moe_apply
from tpu_inference.layers.common.process_weights.linear_weights import \
    shard_linear_weights
from tpu_inference.layers.common.process_weights.moe_weights import (
    FusedMoEWeights, process_quantized_moe_weights)
from tpu_inference.layers.common.quantization import fp8 as common_fp8
from tpu_inference.layers.common.quantization import quantize_tensor
from tpu_inference.layers.common.utils import \
    reorder_concatenated_tensor_for_sharding
from tpu_inference.layers.common.quantization.online_fp8_requant import (
    ONLINE_QUANT_DTYPE_ENV, online_quant_dtype)
from tpu_inference.layers.common.quantization.online_host_quant import (
    HostQuantRequest, adopt_host_quant_scale, request_host_quant)
from tpu_inference.layers.common.utils import cpu_mesh, cpu_mesh_context
from tpu_inference.layers.jax import JaxModule
from tpu_inference.layers.jax.base import create_param
from tpu_inference.layers.jax.linear import (JaxEinsum,
                                             JaxMergedColumnParallelLinear)
from tpu_inference.layers.jax.moe.moe import JaxMoE, JaxRoutedExperts
from tpu_inference.layers.jax.quantization import QuantizeMethodBase
from tpu_inference.layers.jax.quantization.configs import (QuantizationConfig,
                                                           QuantLinearConfig)
from tpu_inference.layers.jax.quantization.unquantized import (
    UnquantizedFusedMoEMethod, UnquantizedLinearMethod)
from tpu_inference.logger import init_logger
from tpu_inference.models.jax.utils.weight_utils import (
    assign_and_shard_param, jax_array_from_reshaped_torch,
    load_nnx_param_from_reshaped_torch)

logger = init_logger(__name__)

# TODO (jacobplatin): remove once we support all backends
FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS = [
    MoEBackend.GMM_EP, MoEBackend.GMM_TP
]


class Fp8TensorwiseLinearMethod(QuantizeMethodBase,
                                common_fp8.Fp8LinearMethod):
    """Tensor-wise Fp8 method for JAX Linear layer."""

    def __init__(self, layer: JaxEinsum, linear_config: QuantLinearConfig):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)

        self.einsum_str = layer.einsum_str

        self.output_shape = linear_config.out_features
        self.batch_features = linear_config.batch_features
        self.batch_sharding = linear_config.batch_sharding
        out_features = math.prod(self.output_shape)
        in_features = math.prod(linear_config.in_features)
        self.weight_sharding = linear_config.weight_sharding
        if self.batch_features:
            # Batched case: keep original weight sharding for the full
            # 3D weight (matches kernel_shape).
            self.kernel_shape = layer.kernel_shape
        else:
            self.kernel_shape = (in_features, out_features)

        self.in_features = in_features

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)

        layer.weight = create_param(rngs,
                                    shape=self.kernel_shape,
                                    dtype=jnp.float8_e4m3fn,
                                    sharding=self.weight_sharding)

        layer.weight.set_metadata(
            'weight_loader',
            partial(load_nnx_param_from_reshaped_torch,
                    permute_dims=(1, 0),
                    param_name=layer.prefix + ".weight"))

        # Scale is always per-output-channel (1D).
        scale_sharding = None
        if self.batch_features:
            # For batched weights, the output dim sharding comes from
            # the weight's non-contracting, non-batch axis.
            if self.batch_sharding:
                scale_sharding = None  # replicated scale for simplicity
        elif isinstance(self.weight_sharding, P) and len(
                self.weight_sharding) > 0:
            scale_sharding = P(self.weight_sharding[0])
        elif isinstance(self.weight_sharding,
                        (tuple, list)) and len(self.weight_sharding) > 0:
            scale_sharding = (self.weight_sharding[0], )

        layer.weight_scale = create_param(rngs,
                                          shape=(out_features, ),
                                          dtype=jnp.float32,
                                          sharding=scale_sharding)
        layer.weight_scale.set_metadata(
            'weight_loader',
            partial(load_nnx_param_from_reshaped_torch,
                    reshape_dims=(out_features, ),
                    permute_dims=None,
                    param_name=layer.prefix + ".weight_scale"))

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        bias = layer.bias[...] if layer.bias is not None else None

        if self.batch_features:
            # Batched case: use dot_general with batch dims.
            out = sharded_quantized_batched_matmul(
                x,
                layer.weight[...],
                layer.weight_scale[...],
                einsum_str=self.einsum_str,
                weight_sharding=self.weight_sharding,
                mesh=self.linear_config.mesh)
            if bias is not None:
                out += bias
            return out

        # Preserve the leading (batch/token) axes. `out.shape[:-1] +
        # shape` was a NO-OP that silently FUSED them: after the
        # flatten `out` is already 2-D, so the restore returned
        # [N, out] regardless of the caller's rank. Live in the
        # 26B/31B flax lanes at B>1 and invisible to any B=1 test.
        # NOT x.shape[:-1]: the flatten below removes ONE axis per CONTRACTING
        # axis, and a kernel can have several. Gemma-4's o_proj is
        # JaxEinsum("TNH,NHD->TD") with in_features == (num_heads, head_dim),
        # so x is [T, N, H] and TWO axes are consumed. Capturing only
        # x.shape[:-1] there restores [T, N, D] from a [T, D] result and raises
        # "cannot reshape array of shape (1024, 3840) into (1024, 16, 3840)" on
        # the FIRST forward -- which the pre-change code did not do.
        # Found by adversarial review 2026-09-01; affects every o_proj in the
        # tree (gemma4, gemma4_mtp, qwen2, qwen3, qwen3_dflash, gemma4_mm).
        leading = x.shape[:-len(self.linear_config.in_features)]
        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)
        out = self._apply_fused(x,
                                layer.weight[...],
                                layer.weight_scale[...],
                                bias=bias)
        out = out.reshape(tuple(leading) + tuple(self.output_shape))
        return out


class Fp8TensorwiseMergedLinearMethod(Fp8TensorwiseLinearMethod):
    """Tensorwise Fp8 for ``JaxMergedColumnParallelLinear`` (fused gate_up/qkv).

    ``_route`` delivers each sub-projection's ``weight`` and ``weight_scale``
    as separate shards (shard_id 0, 1, …).  This class installs
    shard-accumulating loaders on both params: shards are buffered in
    ``_merged_shards`` metadata and concatenated once all arrive, mirroring
    ``Fp8BlockwiseMergedLinearMethod`` for the blockwise case.
    """

    @staticmethod
    def _load_merged_shard(param,
                           torch_tensor,
                           shard_id=-1,
                           *,
                           reshape_dims,
                           permute_dims,
                           param_name,
                           n_shards=1,
                           output_sizes=None):
        shards = param.get_metadata("_merged_shards")
        with cpu_mesh_context():
            if shard_id == -1:
                merged = jax_array_from_reshaped_torch(
                    torch_tensor,
                    reshape_dims=reshape_dims,
                    permute_dims=permute_dims)
            else:
                shards[shard_id] = torch_tensor
                if any(s is None for s in shards):
                    return
                merged = jnp.concatenate([
                    jax_array_from_reshaped_torch(t,
                                                  reshape_dims=reshape_dims,
                                                  permute_dims=permute_dims)
                    for t in shards
                ],
                                         axis=-1)
            # REVIEW-CONFIRMED DEFECT (2026-09-01, 3/3): the merged kernel was
            # stored as plain [gate | up], but the inherited apply path
            # (common/quantization/fp8.py _apply_fused ->
            # slice_sharded_tensor_for_concatenation) DE-INTERLEAVES the output
            # as if the kernel were interleaved by shard. Identity at
            # n_shards=1 -- every arm so far -- and silently wrong columns at
            # TP>1 for any compressed-tensors fp8 checkpoint. Store interleaved,
            # exactly as UnquantizedMergedLinearMethod does. The per-output-
            # channel scale is 1-D over the same axis, so it gets the same
            # reorder; that is why this happens BEFORE assign for both.
            if output_sizes is not None and n_shards > 1:
                merged = reorder_concatenated_tensor_for_sharding(
                    merged, output_sizes, n_shards, dim=merged.ndim - 1)
        assign_and_shard_param(param, merged, param_name=param_name)

    def create_weights_jax(self, layer: JaxMergedColumnParallelLinear,
                           *weight_args, rngs, **extra_weight_attrs):
        assert isinstance(layer, JaxMergedColumnParallelLinear)
        super().create_weights_jax(layer,
                                   *weight_args,
                                   rngs=rngs,
                                   **extra_weight_attrs)
        n_proj = len(layer.output_sizes)
        layer.weight.set_metadata("_merged_shards", [None] * n_proj)
        layer.weight.set_metadata(
            "weight_loader",
            functools.partial(self._load_merged_shard,
                              reshape_dims=None,
                              permute_dims=(1, 0),
                              param_name=layer.prefix + ".weight",
                              n_shards=self.linear_config.n_shards,
                              output_sizes=self.linear_config.output_sizes))
        layer.weight_scale.set_metadata("_merged_shards", [None] * n_proj)
        layer.weight_scale.set_metadata(
            "weight_loader",
            functools.partial(self._load_merged_shard,
                              reshape_dims=(-1, ),
                              permute_dims=None,
                              param_name=layer.prefix + ".weight_scale",
                              n_shards=self.linear_config.n_shards,
                              output_sizes=self.linear_config.output_sizes))


class Fp8BlockwiseLinearMethod(QuantizeMethodBase, common_fp8.Fp8LinearMethod):
    """Block-wise Fp8 method for JAX Linear layer."""

    def __init__(self,
                 quant_config: "Fp8Config",
                 layer: JaxEinsum,
                 linear_config: QuantLinearConfig,
                 weight_scale_name: str = "weight_scale_inv"):
        common_fp8.Fp8LinearMethod.__init__(self, linear_config)
        self.quant_config = quant_config
        # Checkpoint-dependent name of the dequant scale: DeepSeek-style fp8
        # checkpoints serialize "weight_scale_inv" while compressed-tensors
        # uses "weight_scale". Both are the same quantity (the multiplier in
        # x ~= q * scale); only the serialized name differs.
        self.weight_scale_name = weight_scale_name
        self.einsum_str = layer.einsum_str

        self.out_features = linear_config.out_features
        self.in_features = math.prod(linear_config.in_features)
        self.batch_features = linear_config.batch_features
        self.batch_sharding = linear_config.batch_sharding
        self.weight_sharding = linear_config.weight_sharding
        self.bias_sharding = linear_config.bias_sharding
        if self.batch_features:
            # Batched case: keep original weight sharding for the full
            # 3D weight (matches kernel_shape).
            self.kernel_shape = layer.kernel_shape
        else:
            self.kernel_shape = (self.in_features,
                                 math.prod(self.out_features))

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxEinsum)

        out_features = sum(self.linear_config.output_sizes)
        kernel_init = layer.kernel_init

        if self.batch_features:
            # Batched case: create weight with the original 3D kernel shape
            # so the weight loader can populate it directly after transpose.
            # Weight stays in FP8 and is used with sharded_quantized_batched_matmul.
            param_dtype = jnp.float8_e4m3
            layer.weight = nnx.Param(
                nnx.initializers.uniform()(rngs.params(), self.kernel_shape,
                                           param_dtype),
                weight_loader=partial(load_nnx_param_from_reshaped_torch,
                                      permute_dims=None,
                                      param_name=layer.prefix + ".weight"),
                eager_sharding=False)
            layer.weight.set_metadata('out_sharding', self.weight_sharding)

            # Per-output-channel scale (1D, covers the free weight dim).
            scale_param = nnx.Param(jnp.ones((out_features, ),
                                             dtype=layer.dtype),
                                    weight_loader=partial(
                                        load_nnx_param_from_reshaped_torch,
                                        permute_dims=None,
                                        param_name=layer.prefix + "." +
                                        self.weight_scale_name,
                                    ),
                                    eager_sharding=False)
            scale_param.set_metadata('out_sharding', ())
            setattr(layer, self.weight_scale_name, scale_param)
            return

        # Follow upstream limitation that only float8_e4m3 is supported.
        # https://github.com/vllm-project/vllm/blob/2a99c5a6c86daef8c766ba2dbf05c385b192c64b/vllm/model_executor/layers/quantization/fp8.py#L283-L284
        param_dtype = jnp.float8_e4m3
        layer.weight = nnx.Param(
            kernel_init(rngs.params(), self.kernel_shape, param_dtype),
            weight_loader=partial(load_nnx_param_from_reshaped_torch,
                                  permute_dims=(1, 0),
                                  param_name=layer.prefix + ".weight"),
            eager_sharding=False)
        layer.weight.set_metadata('out_sharding', self.weight_sharding)

        # Block-wise quantization scale
        block_n, block_k = self.quant_config.weight_block_size[
            0], self.quant_config.weight_block_size[1]
        scale_param = nnx.Param(kernel_init(
            rngs.params(),
            [(self.in_features + block_k - 1) // block_k,
             (out_features + block_n - 1) // block_n],
            layer.dtype,
        ),
                                weight_loader=partial(
                                    load_nnx_param_from_reshaped_torch,
                                    permute_dims=(1, 0),
                                    param_name=layer.prefix + "." +
                                    self.weight_scale_name,
                                ),
                                eager_sharding=False)
        scale_param.set_metadata('out_sharding', self.weight_sharding)
        setattr(layer, self.weight_scale_name, scale_param)

        # Force the parameters to be loaded onto CPU, such that in `process_weights_after_loading`
        # we can process the weights on CPU to avoid OOM on device.
        layer.weight.set_metadata('mesh', cpu_mesh())
        scale_param.set_metadata('mesh', cpu_mesh())
        if layer.bias is not None:
            layer.bias.set_metadata('mesh', cpu_mesh())

    def process_weights_after_loading(self, layer: JaxEinsum) -> bool:
        assert isinstance(layer, JaxEinsum)
        assert self.quant_config.weight_block_size is not None

        if self.batch_features:
            # Batched case: weight stays in FP8. No blockwise processing
            # needed — the batched matmul uses dot_general with FP8 natively.
            return True

        scale_param = getattr(layer, self.weight_scale_name)
        if not layer.weight.get_metadata(
                "_is_loaded", False) or not scale_param.get_metadata(
                    "_is_loaded", False):
            # Weight and scale could spread across multiple files,
            # so we only process once both of them are loaded.
            return False

        # Do the re-quant process on CPU to avoid OOM on device.
        with cpu_mesh_context():
            weight = layer.weight[...]
            weight_scale_inv = scale_param[...]
            bias = layer.bias[...] if getattr(layer, 'bias',
                                              None) is not None else None
            if bias is not None:
                bias = bias.reshape(-1)
            weights = common_fp8.process_blockwise_fp8_linear_weights(
                weight,
                weight_scale_inv,
                bias=bias,
                weight_block_size=tuple(self.quant_config.weight_block_size),
                requant_block_size=self.linear_config.requant_block_size,
                output_sizes=tuple(self.linear_config.output_sizes),
                requant_weight_dtype=self.linear_config.requant_weight_dtype,
                fuse_matmuls=self.linear_config.fuse_matmuls,
                n_shards=self.linear_config.n_shards,
                enable_kernel=self.linear_config.enable_quantized_matmul_kernel
            )
            delattr(layer, 'weight')
            delattr(layer, self.weight_scale_name)
            delattr(layer, 'bias')

        # Put onto the device.
        weights = shard_linear_weights(
            weights,
            mesh=None,
            weight_p_spec=self.linear_config.weight_sharding,
            bias_p_spec=self.linear_config.bias_sharding,
        )
        if self.linear_config.fuse_matmuls:
            layer.weight = nnx.Param(weights.weight)
            setattr(layer, self.weight_scale_name,
                    nnx.Param(weights.weight_scale))
            layer.bias = nnx.Param(weights.bias) if bias is not None else None
        else:
            raise NotImplementedError(
                "Fp8 block-wise linear method only supports fuse_matmuls.")

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array) -> jax.Array:
        if self.batch_features:
            # Batched case: use dot_general with FP8 and batch dims.
            out = sharded_quantized_batched_matmul(
                x,
                layer.weight[...],
                getattr(layer, self.weight_scale_name)[...],
                einsum_str=self.einsum_str,
                weight_sharding=self.weight_sharding,
                mesh=self.linear_config.mesh)
            return out

        if not self.linear_config.fuse_matmuls:
            raise NotImplementedError(
                "Fp8 block-wise linear method only supports fuse_matmuls.")
        weight = layer.weight[...]
        scale = getattr(layer, self.weight_scale_name)[...]
        bias = layer.bias[...] if layer.bias is not None else None
        # Preserve the leading (batch/token) axes. `out.shape[:-1] +
        # shape` was a NO-OP that silently FUSED them: after the
        # flatten `out` is already 2-D, so the restore returned
        # [N, out] regardless of the caller's rank. Live in the
        # 26B/31B flax lanes at B>1 and invisible to any B=1 test.
        # NOT x.shape[:-1]: the flatten below removes ONE axis per CONTRACTING
        # axis, and a kernel can have several. Gemma-4's o_proj is
        # JaxEinsum("TNH,NHD->TD") with in_features == (num_heads, head_dim),
        # so x is [T, N, H] and TWO axes are consumed. Capturing only
        # x.shape[:-1] there restores [T, N, D] from a [T, D] result and raises
        # "cannot reshape array of shape (1024, 3840) into (1024, 16, 3840)" on
        # the FIRST forward -- which the pre-change code did not do.
        # Found by adversarial review 2026-09-01; affects every o_proj in the
        # tree (gemma4, gemma4_mtp, qwen2, qwen3, qwen3_dflash, gemma4_mm).
        leading = x.shape[:-len(self.linear_config.in_features)]
        if len(x.shape) > 2:
            x = x.reshape(-1, self.in_features)
        out = self._apply_fused(x, weight, scale, bias=bias)
        out = out.reshape(tuple(leading) + tuple(self.out_features))
        return out


class Fp8BlockwiseMergedLinearMethod(Fp8BlockwiseLinearMethod):
    """Block-wise Fp8 method for ``JaxMergedColumnParallelLinear`` layers.

    Mirrors ``UnquantizedMergedLinearMethod`` (see unquantized.py): a merged
    column-parallel linear (e.g. fused ``gate_up_proj``) holds one kernel,
    but a checkpoint that ships native (unfused) FP8 weights still stores
    each projection (``gate_proj``, ``up_proj``) as a separate tensor with
    its own block-wise scale. ``create_weights_jax`` attaches a
    ``weight_loader`` that accumulates each projection's ``weight`` and
    ``weight_scale_inv`` tensor (by ``shard_id``) and, once all projections
    for a given param have arrived, concatenates them in declaration order.

    Unlike the unquantized merge loader, this does not interleave shards for
    TP sharding here: ``process_weights_after_loading``
    (``process_blockwise_fp8_linear_weights``, inherited unchanged from
    ``Fp8BlockwiseLinearMethod``) consumes the plain concatenated
    ``[proj0, proj1, ...]`` layout and performs the requant + interleave
    itself, given ``output_sizes``/``n_shards``.
    """

    @staticmethod
    def _load_merged_shard(param: nnx.Param,
                           torch_tensor,
                           shard_id: int = -1,
                           *,
                           permute_dims,
                           param_name: str):
        shards = param.get_metadata("_merged_shards")
        with cpu_mesh_context():
            if shard_id == -1:
                merged = jax_array_from_reshaped_torch(
                    torch_tensor, permute_dims=permute_dims)
            else:
                shards[shard_id] = torch_tensor
                if any(s is None for s in shards):
                    return
                merged = jnp.concatenate([
                    jax_array_from_reshaped_torch(t, permute_dims=permute_dims)
                    for t in shards
                ],
                                         axis=-1)
        assign_and_shard_param(param, merged, param_name=param_name)

    def create_weights_jax(self, layer: JaxModule, *weight_args, rngs,
                           **extra_weight_attrs):
        assert isinstance(layer, JaxMergedColumnParallelLinear)
        super().create_weights_jax(layer,
                                   *weight_args,
                                   rngs=rngs,
                                   **extra_weight_attrs)
        n_proj = len(layer.output_sizes)
        layer.weight.set_metadata("_merged_shards", [None] * n_proj)
        layer.weight.set_metadata(
            "weight_loader",
            functools.partial(self._load_merged_shard,
                              permute_dims=(1, 0),
                              param_name=layer.prefix + ".weight"))
        layer.weight_scale_inv.set_metadata("_merged_shards", [None] * n_proj)
        layer.weight_scale_inv.set_metadata(
            "weight_loader",
            functools.partial(self._load_merged_shard,
                              permute_dims=(1, 0),
                              param_name=layer.prefix + ".weight_scale_inv"))


class Fp8FusedMoEMethod(QuantizeMethodBase):
    """
    Fp8 method for JAXMoE layer.

    TODO (jacobplatin): support weight loading -- currently, model-dependent.
    """

    def __init__(self, weight_block_size: Optional[Sequence[int]], *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.extra_backend_kwargs = {}
        self.weight_block_size = None if weight_block_size is None else tuple(
            weight_block_size)
        self.block_quant: bool = self.weight_block_size is not None
        self.weight_scale_name = ("weight_scale_inv"
                                  if self.block_quant else "weight_scale")

    def load_weights(self, *, layer: JaxMoE, original_load_weights_fn,
                     weights: Iterable) -> set:
        """Load scale paramters and delegate the weight paramters to `original_load_weights_fn`"""

        # Remaining non-scale parameters will be loaded using original load_weights function.
        remaining_weights = dict()
        cnt = 0
        for torch_name, torch_weight in weights:
            torch_name: str = torch_name.split(
                layer.prefix)[-1]  # ".0.down_proj.weight" for example
            names = torch_name.split(".")
            assert len(
                names
            ) == 3, f"Expected param name to be .<expert_id>.<param_name>.weight, got {torch_name}"
            expert_id, _, _ = names
            expert_id = int(expert_id)
            jax_param_name = ""
            if torch_name.endswith("up_proj." + self.weight_scale_name):
                jax_param_name = "kernel_up_proj_EDF_" + self.weight_scale_name
            elif torch_name.endswith("down_proj." + self.weight_scale_name):
                jax_param_name = "kernel_down_proj_EFD_" + self.weight_scale_name
            elif torch_name.endswith("gate_proj." + self.weight_scale_name):
                jax_param_name = "kernel_gating_EDF_" + self.weight_scale_name
            else:
                remaining_weights[torch_name] = torch_weight
                continue
            cnt += 1
            jax_param = getattr(layer, jax_param_name, None)

            assert isinstance(jax_param, nnx.Param)

            # Here we rely on `jax_array_from_reshaped_torch` to load weights
            # onto CPU and prepend a leading dimension for expert_id, because
            # later in `process_weights_after_loading` the sharded experts
            # will be concatenated altogether then put onto the device.
            jax_weight = jax_array_from_reshaped_torch(torch_weight,
                                                       reshape_dims=(1, ) +
                                                       torch_weight.shape)
            jax_param._weights_to_load[expert_id] = jax_weight

        logger.debug(
            f"Loaded {cnt} weight scales for {layer.prefix} MoE layer.")

        loaded_names = original_load_weights_fn(remaining_weights.items(),
                                                mesh=cpu_mesh())
        for param_name in {
                "kernel_gating_EDF_" + self.weight_scale_name,
                "kernel_up_proj_EDF_" + self.weight_scale_name,
                "kernel_down_proj_EFD_" + self.weight_scale_name,
        }:
            param = getattr(layer, param_name)
            if all(w is not None for w in param._weights_to_load):
                loaded_names.add(param_name)

        return loaded_names

    def create_weights_jax(self, layer: JaxMoE, *weight_args, rngs,
                           **extra_weight_attrs) -> None:
        """
        Create the quant method-specific weights.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to create weights for.
        """

        # TODO (#1681): support other backends
        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            # vLLM reference here:
            # https://github.com/vllm-project/vllm/blob/9bdb06b/vllm/model_executor/layers/quantization/fp8.py#L763
            if not self.block_quant:
                # Tensorwise (per-channel) FP8: weights are fp8, scales are per
                # output channel with shape [E, N_out, 1] (CT format).
                for param_name in [
                        "kernel_gating_EDF", "kernel_up_proj_EDF",
                        "kernel_down_proj_EFD"
                ]:
                    param = getattr(layer, param_name, None)
                    assert isinstance(
                        param, nnx.Param
                    ), f"Expected nnx.Param for {param_name}, got {type(param)}"
                    init_fn = param.init_fn
                    E, K, N = param[...].shape
                    value = init_fn(rngs.params(), (E, K, N),
                                    jnp.float8_e4m3fn)
                    param.set_raw_value(value)
                    # Placeholder shape [E, N, 1]: N is the output-channel count.
                    # Actual per-expert scale loaded by load_weights as [1, N, 1]
                    # and concatenated to [E, N, 1] in process_weights_after_loading.
                    scale_value = jnp.zeros((E, N, 1),
                                            dtype=jnp.float32,
                                            device=jax.devices('cpu')[0])
                    setattr(
                        layer, f"{param_name}_{self.weight_scale_name}",
                        nnx.Param(scale_value,
                                  _weights_to_load=[None for _ in range(E)]))
            else:
                assert len(
                    self.weight_block_size
                ) == 2, f"Expected 2D block size, got {self.weight_block_size}"
                block_n, block_k = self.weight_block_size

                # re-create the weights to be in fp8 type
                for param_name in [
                        "kernel_gating_EDF", "kernel_up_proj_EDF",
                        "kernel_down_proj_EFD"
                ]:
                    param = getattr(layer, param_name, None)
                    assert isinstance(
                        param, nnx.Param
                    ), f"Expected nnx.Param for {param_name}, got {type(param)}"
                    init_fn = param.init_fn
                    E, K, N = param[...].shape
                    value = init_fn(rngs.params(), (E, K, N),
                                    jnp.float8_e4m3fn)
                    param.set_raw_value(value)

                    scale_value = jnp.zeros((E, (K + block_k - 1) // block_k,
                                             (N + block_n - 1) // block_n),
                                            device=jax.devices('cpu')[0])
                    setattr(
                        layer, f"{param_name}_{self.weight_scale_name}",
                        nnx.Param(scale_value,
                                  _weights_to_load=[None for _ in range(E)]))
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

    def process_weights_after_loading(self, layer: JaxMoE) -> bool:
        """
        Process weights after loading.

        Please see https://github.com/vllm-project/tpu-inference/blob/bb1a88/tpu_inference/layers/common/moe.py#L39
        for more information on the expected weights per MoE backend.

        Args:
            layer: The layer to process.
        """
        # TODO (#1681): support other backends

        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            gating_scale_name = f"kernel_gating_EDF_{self.weight_scale_name}"
            up_scale_name = f"kernel_up_proj_EDF_{self.weight_scale_name}"
            down_scale_name = f"kernel_down_proj_EFD_{self.weight_scale_name}"

            if any(
                    any(w is None for w in param._weights_to_load) for param in
                [
                    getattr(layer, gating_scale_name),
                    getattr(layer, up_scale_name),
                    getattr(layer, down_scale_name), layer.kernel_gating_EDF,
                    layer.kernel_up_proj_EDF, layer.kernel_down_proj_EFD
                ]):
                # If weights for a module is spread across multiple files, this function may be called
                # more than once. We only want to process the weights once all of them are loaded.
                return False

            with cpu_mesh_context():
                w_gate = jnp.concatenate(
                    layer.kernel_gating_EDF._weights_to_load, axis=0)
                w_up = jnp.concatenate(
                    layer.kernel_up_proj_EDF._weights_to_load, axis=0)
                s_gate = jnp.concatenate(getattr(
                    layer, gating_scale_name)._weights_to_load,
                                         axis=0)
                s_up = jnp.concatenate(getattr(layer,
                                               up_scale_name)._weights_to_load,
                                       axis=0)
                w2_weight = jnp.concatenate(
                    layer.kernel_down_proj_EFD._weights_to_load, axis=0)
                w2_weight_scale = jnp.concatenate(getattr(
                    layer, down_scale_name)._weights_to_load,
                                                  axis=0)

                # Fuse the weights into w13: [Gate, Up]. w2 is expected to be
                # (num_experts, hidden_size, intermediate_size), w13 is expected to
                # be (num_experts, 2 * intermediate_size, hidden_size,)
                w13_weight = jnp.concatenate([w_gate, w_up], axis=1)
                w13_weight_scale = jnp.concatenate([s_gate, s_up], axis=1)

            weight_block_size = None
            if self.weight_block_size is not None:
                weight_block_size = tuple(self.weight_block_size)

            # TODO (jacobplatin): we should support bias
            input_weights = FusedMoEWeights(w13_weight=w13_weight,
                                            w13_weight_scale=w13_weight_scale,
                                            w13_bias=None,
                                            w2_weight=w2_weight,
                                            w2_weight_scale=w2_weight_scale,
                                            w2_bias=None)

            # Shard MoE weights to TPU before requantization so that
            # process_quantized_moe_weights runs on TPU instead of CPU.
            weights = process_quantized_moe_weights(
                input_weights,
                moe_backend=layer.moe_backend,
                mesh=layer.mesh,
                activation=layer.activation,
                weight_block_size=weight_block_size,
                source_mesh=cpu_mesh(),
            )

            del layer.kernel_gating_EDF
            del layer.kernel_up_proj_EDF
            delattr(layer, gating_scale_name)
            delattr(layer, up_scale_name)

            # process_quantized_moe_weights applies with_sharding_constraint via
            # _get_moe_weight_shardings for all weight fields; the arrays are
            # already correctly sharded on return — no shard_put needed.
            layer.kernel_gating_upproj_EDF = nnx.Param(weights.w13_weight)
            layer.kernel_down_proj_EFD = nnx.Param(weights.w2_weight)
            setattr(layer,
                    f"kernel_gating_upproj_EDF_{self.weight_scale_name}",
                    nnx.Param(weights.w13_weight_scale))
            setattr(layer, f"kernel_down_proj_EFD_{self.weight_scale_name}",
                    nnx.Param(weights.w2_weight_scale))
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return True

    def apply_jax(self, layer: JaxModule, x: jax.Array, *,
                  router_logits: jax.Array) -> jax.Array:
        """
        Run the forward pass of the MoE layer.

        Args:
            layer: The layer to apply the quantization method to.
            x: The input to the layer.

        Returns:
            The MoE output.
        """
        assert isinstance(layer, (JaxMoE, JaxRoutedExperts))

        x_TD = jnp.asarray(x, layer.dtype)
        x_TD = jax.lax.with_sharding_constraint(
            x_TD,
            jax.sharding.NamedSharding(layer.mesh,
                                       P(*layer.activation_ffw_td)))

        # Fused weight backends
        if layer.moe_backend in FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS:
            # router_logits is of shape TE -- we don't return the indices

            if layer.moe_backend == MoEBackend.FUSED_MOE:
                w13_weight = layer.kernel_gating_upproj_E2DF[...]
            else:
                w13_weight = layer.kernel_gating_upproj_EDF[...]
            w2_weight = layer.kernel_down_proj_EFD[...]
            w13_weight_scale = getattr(
                layer,
                f"kernel_gating_upproj_EDF_{self.weight_scale_name}")[...]

            w2_weight_scale = getattr(
                layer, f"kernel_down_proj_EFD_{self.weight_scale_name}")[...]

            # TODO (jacobplatin/bzgoogle): we should support bias
            weights = FusedMoEWeights(
                w13_weight=w13_weight,
                w13_weight_scale=w13_weight_scale,
                w13_bias=None,
                w2_weight=w2_weight,
                w2_weight_scale=w2_weight_scale,
                w2_bias=None,
            )
        else:
            raise NotImplementedError(
                f"Unsupported moe backend: {layer.moe_backend}! Currently supported: {FP8_QUANT_METHOD_SUPPORTED_MOE_BACKENDS}"
            )

        return moe_apply(layer, x_TD, router_logits, weights,
                         layer.moe_backend, layer.mesh,
                         self.extra_backend_kwargs)


class Fp8Config(QuantizationConfig):

    ACTIVATION_SCHEMES = ["dynamic", "static"]

    def __init__(self, hf_quant_config: dict):
        # Replicating upstream https://github.com/vllm-project/vllm/blob/77c09e1130661197ccac2d968a28cd4a557922d5/vllm/model_executor/layers/quantization/fp8.py#L167-L175

        quant_method = self.get_from_keys(hf_quant_config, ["quant_method"])
        self.is_checkpoint_fp8_serialized = "fp8" in quant_method
        activation_scheme = self.get_from_keys(hf_quant_config,
                                               ["activation_scheme"])
        ignored_layers = self.get_from_keys(hf_quant_config,
                                            ["ignored_layers"], None)
        weight_block_size = self.get_from_keys(hf_quant_config,
                                               ["weight_block_size"], None)
        # `ignored_layers` lists exact leaf module names (exact match);
        # `modules_to_not_convert` lists container/module names meant to
        # skip everything nested under them (needs substring match) — see
        # `QuantizationConfig.is_layer_skipped`.
        self.skip_with_substr = False
        if not ignored_layers:
            ignored_layers = self.get_from_keys(hf_quant_config,
                                                ["modules_to_not_convert"],
                                                None)
            self.skip_with_substr = True

        if activation_scheme not in self.ACTIVATION_SCHEMES:
            raise ValueError(
                f"Unsupported activation scheme {activation_scheme}")
        self.activation_scheme = activation_scheme
        self.ignored_layers = ignored_layers or []
        if weight_block_size is not None:
            if not self.is_checkpoint_fp8_serialized:
                raise ValueError(
                    "The block-wise quantization only supports fp8-serialized "
                    "checkpoint for now.")
            if len(weight_block_size) != 2:
                raise ValueError(
                    "The quantization block size of weight must have 2 "
                    f"dimensions, but got {len(weight_block_size)} dimensions")
            if activation_scheme != "dynamic":
                raise ValueError("The block-wise quantization only supports "
                                 "dynamic activation scheme for now, but got "
                                 f"{activation_scheme} activation scheme.")
        self.weight_block_size = weight_block_size

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(layer, enable_sp=False)
            if self.is_layer_skipped(prefix,
                                     ignored_layers=self.ignored_layers,
                                     skip_with_substr=self.skip_with_substr):
                return UnquantizedLinearMethod(linear_config)
            if self.weight_block_size is not None:
                if isinstance(layer, JaxMergedColumnParallelLinear):
                    return Fp8BlockwiseMergedLinearMethod(
                        self, layer, linear_config)
                return Fp8BlockwiseLinearMethod(self, layer, linear_config)
            else:
                return Fp8TensorwiseLinearMethod(layer, linear_config)
        elif isinstance(layer, (JaxRoutedExperts, JaxMoE)):
            if self.is_layer_skipped(prefix,
                                     ignored_layers=self.ignored_layers,
                                     skip_with_substr=self.skip_with_substr):
                return UnquantizedFusedMoEMethod(layer)
            return Fp8FusedMoEMethod(self.weight_block_size)
        return None


# The dtype selection lives in the COMMON leaf so that BOTH online requant
# paths -- this flax/nnx one (26B-A4B MoE) and the vllm/torchax one (12B
# dense) -- answer it identically. A lever that moved only one of them would
# make the two arms incomparable rather than configurable.
_online_fp8_dtype = online_quant_dtype

# The lane tooling greps this line as one half of the engagement proof (bank
# writes are the other half). Emitted by BOTH the host-quantized path and the
# on-device fallback, so the proof does not depend on which path ran.
_ONLINE_DENSE_MARKER = (
    "VLLM_FP8_ONLINE_DENSE=1: serving dense on-the-fly fp8 "
    "(e4m3, per-output-channel) -- requanted from the bf16 "
    "checkpoint at load; experts stay on MOE_REQUANTIZE.")


class Fp8OnlineLinearMethod(Fp8TensorwiseLinearMethod):
    """On-the-fly per-output-channel e4m3 for a BF16 checkpoint (flax lane).

    The native (flax_nnx) mirror of the torchax VllmFp8OnlineLinearMethod
    (issue #158, fork PR #17 + the axis/env fix #18). Reached ONLY when
    --quantization fp8 is set, the checkpoint is NOT fp8-serialized, and
    VLLM_FP8_ONLINE_DENSE=1; every other path keeps the clean fail-closed
    NotImplementedError.

    Discipline inherited from the torchax method: create_weights_jax is a
    NO-OP so the model's own bf16 kernel and its default loaders survive --
    no param replacement, and critically NO empty weight_scale param (an
    uninitialized scale is the garbage the fail-closed guard exists to
    prevent). The requant happens after load, and apply_jax is inherited
    unchanged so BOTH lanes land on the same xla_quantized_matmul --
    arm-to-arm numerical comparability by construction.
    """

    def create_weights_jax(self, layer: JaxEinsum, *weight_args, rngs,
                           **extra_weight_attrs):
        # Keep the model-created bf16 kernel and its default loader. Merged
        # layers (gate_up / qkv) must still merge+interleave at load, so
        # delegate to the unquantized merged path; per-output-channel scales
        # are order-equivariant, so quantizing the already-interleaved
        # kernel needs no reorder logic. (Blockwise would NOT commute --
        # explicit non-goal.)
        from tpu_inference.layers.jax.quantization.unquantized import (
            UnquantizedLinearMethod, UnquantizedMergedLinearMethod)
        if isinstance(layer, JaxMergedColumnParallelLinear):
            # The delegated create_weights_jax builds
            # functools.partial(self._load_merged_tensor, ...) -- and that is
            # a staticmethod of UnquantizedMergedLinearMethod, NOT of this
            # class. Delegating the function without binding the method it
            # reads off `self` raised
            #   AttributeError: 'Fp8OnlineLinearMethod' object has no
            #   attribute '_load_merged_tensor'
            # at MODEL CONSTRUCTION for every gate_up_proj, killing all four
            # flax dense-quant arms of the 26B identically (fp8, e4m3b11fnuz,
            # int8, allint8) ~70s into boot -- measured 2026-09-01 23:07Z.
            # Never reachable on the 12B, whose torchax path has its own
            # merged loader, which is why #24's CPU gate did not see it.
            self._load_merged_tensor = (
                UnquantizedMergedLinearMethod._load_merged_tensor)
            UnquantizedMergedLinearMethod.create_weights_jax(
                self, layer, *weight_args, rngs=rngs, **extra_weight_attrs)
        else:
            UnquantizedLinearMethod.create_weights_jax(
                self, layer, *weight_args, rngs=rngs, **extra_weight_attrs)

        # Ask the loader to quantize this kernel on the HOST and place only
        # (w_q, w_s). MEASURED 2026-09-02 23:09Z: requanting after placement
        # put the whole bf16 checkpoint, the int8 copies AND two f32
        # temporaries on one v6e at once and the 12B int8 arm died at load
        # (online_host_quant has the arithmetic). The request rides on the
        # Param's metadata, like weight_loader, because the loader only
        # knows the Param; process_weights_after_loading adopts the parked
        # scale. kernel_shape / weight_sharding are the 2-D [in, out] layout
        # and spec the apply path assumes, so the placement matches the
        # forward. Batched kernels keep the fail-closed refusal below.
        if not self.batch_features:
            request_host_quant(
                layer.weight,
                HostQuantRequest(dtype=_online_fp8_dtype(),
                                 kernel_shape=tuple(self.kernel_shape),
                                 weight_spec=tuple(self.weight_sharding),
                                 scale_spec=(self.weight_sharding[1], )))

    def process_weights_after_loading(self, layer: JaxEinsum) -> bool:
        assert isinstance(layer, JaxEinsum)
        if not layer.weight.get_metadata("_is_loaded", False):
            # The kernel can arrive across multiple files; requant once.
            return False
        if self.batch_features:
            raise NotImplementedError(
                "VLLM_FP8_ONLINE_DENSE does not support batched (3D) linear "
                "weights; no Gemma-4 dense layer has true batch dims. "
                "Refusing rather than guessing a scale layout.")

        w_s = adopt_host_quant_scale(layer.weight)
        if w_s is not None:
            # The loader quantized this kernel on the HOST and placed only
            # (w_q, w_s): no bf16 kernel reached the mesh. Re-wrap the SAME
            # device buffers as fresh Params (no copy, no metadata) -- the
            # post-load state the on-device path below produces. The old
            # Param that JaxAutoWeightsLoader.load_weights still holds in
            # its params_dict now pins an int8 buffer that is shared with
            # the new Param, not a bf16 one that nothing else uses.
            w_q = layer.weight[...]
            delattr(layer, 'weight')
            layer.weight = nnx.Param(w_q)
            layer.weight_scale = nnx.Param(w_s)
            logger.info_once(
                "online quant: kernels quantized on the host before "
                "placement; no bf16 kernel reached the model mesh.")
            logger.info_once(_ONLINE_DENSE_MARKER)
            return True

        # FALLBACK: a bf16 kernel that reached the mesh unquantized. Only a
        # loader that hands assign_and_shard_param an array ALREADY on the
        # model mesh gets here (pathways_dummy builds its dummies on the
        # TPU); every checkpoint path is served above. This is the pre-fix
        # path, unchanged: it holds the bf16 kernel and two f32 temporaries
        # on the mesh at once. Requant WHERE THE PARAM LIVES.
        #
        # This used to open `with cpu_mesh_context():` around a param that
        # `assign_and_shard_param` has already committed to the TPU model
        # mesh, so the first op (jnp.max inside quantize_tensor) raised
        # "Received incompatible devices for jitted computation" -- the flax
        # online-fp8 lane could never complete a weight load, which is why no
        # flax fp8 arm has ever served a token. The blockwise sibling gets
        # away with cpu_mesh_context because IT pins its params to cpu_mesh
        # in create_weights_jax; this method deliberately does not create
        # params at all, so it must not borrow that discipline.
        weight = layer.weight[...]
        src_sharding = getattr(weight, "sharding", None)
        w2d = weight.reshape(math.prod(self.linear_config.in_features), -1)
        # The same primitive the MoE requant path already trusts; axis=0 is
        # the per-OUTPUT-channel reduction for an [in, out] kernel
        # (numerically amax/448, matching the torchax leaf after #18).
        w_q, w_s = quantize_tensor(_online_fp8_dtype(), w2d, axis=0)
        delattr(layer, 'weight')

        # Sharding is derived from the SOURCE ARRAY, not from
        # linear_config.mesh -- that attribute is None on the JAX lane, and
        # NamedSharding(None, spec) raises TypeError before any token is
        # served. The quantized weight has the source's shape, so the
        # source's sharding applies unchanged; the per-output scale is
        # 1-D over the OUT axis and is left replicated (it is tiny, and a
        # wrong spec here is the Fp8Tensorwise wart of sharding a per-output
        # scale along the INPUT axis).
        if src_sharding is not None:
            try:
                w_q = jax.device_put(w_q, src_sharding)
            except (ValueError, TypeError):
                pass  # shape/mesh mismatch: leave placement to the runtime
        layer.weight = nnx.Param(w_q)
        layer.weight_scale = nnx.Param(w_s)
        logger.info_once(_ONLINE_DENSE_MARKER)
        return True


class Fp8OnlineConfig(Fp8Config):
    """Config that dispatches the online-dense method for bf16 checkpoints.

    Constructed ONLY by get_tpu_quantization_config when the flag is set
    (see layers/jax/quantization/__init__.py); it never reads a checkpoint
    quantization_config, because there is none -- which is exactly why the
    stock Fp8Config raised KeyError here before.
    """

    def __init__(self, hf_quant_config: dict | None = None):
        # Bypass Fp8Config.__init__: nothing to read from the checkpoint.
        self.is_checkpoint_fp8_serialized = False
        self.activation_scheme = "dynamic"
        self.ignored_layers = []
        self.skip_with_substr = False
        self.weight_block_size = None

    def get_quant_method(self, layer: JaxModule,
                         prefix: str) -> Optional[QuantizeMethodBase]:
        from tpu_inference.layers.jax.quantization.unquantized import (
            UnquantizedFusedMoEMethod)
        if isinstance(layer, (JaxRoutedExperts, JaxMoE)):
            # Experts keep the orthogonal, already-serving requant path
            # (MOE_REQUANTIZE_WEIGHT_DTYPE) -- never double-quantized here.
            return UnquantizedFusedMoEMethod(layer)
        if isinstance(layer, JaxEinsum):
            linear_config = QuantLinearConfig(layer, enable_sp=False)
            # Skip the router (routing quality is disproportionately
            # sensitive, no HBM payoff) AND every multimodal projection.
            # MEASURED 2026-09-01 on the torchax twin of this method: a
            # quantized mm projection booted and gated fine, then died on
            # the first request with `dot_general ... got (1120,) and
            # (6912,)` -- max_soft_tokens against the vision patch dim,
            # because those kernels are not the [in, out] 2-D text-stack
            # layout a per-output-channel scale assumes. The vision tower is
            # ~2% of a 26B checkpoint; the HBM case for fp8 is the text
            # stack.
            low = prefix.lower()
            if any(s in low for s in ("router", "vision", "audio", "mm_",
                                      "multi_modal", "multimodal",
                                      "embed_vision", "embed_audio")):
                return UnquantizedLinearMethod(linear_config)
            return Fp8OnlineLinearMethod(layer, linear_config)
        return None
