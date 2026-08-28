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
"""CPU tests for the weight-only int4 (W4A16 / wNa16) compressed-tensors
scheme on the torchax path.

The first test validates the load-path unpack + dequant math BIT-EXACTLY
against the reference dequantization from the ``compressed-tensors`` pip
package (the library that produced the target checkpoints). The remaining
tests mirror ``test_compressed_tensors_w4a8_fp8.py`` at the layer level,
using the same public wNa16 checkpoint WITHOUT the fp8-activation override
that file needs — this is the natural W4A16 path.
"""

import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from compressed_tensors.compressors import pack_to_int32, unpack_from_int32
from compressed_tensors.quantization import QuantizationArgs
from compressed_tensors.quantization.lifecycle.forward import dequantize
from jax.sharding import PartitionSpec
from torchax.interop import torch_view
from torchax.ops.mappings import j2t, t2j
from vllm.config import set_current_vllm_config
from vllm.distributed.parallel_state import (ensure_model_parallel_initialized,
                                             init_distributed_environment)
from vllm.engine.arg_utils import EngineArgs
from vllm.model_executor.layers.linear import (ColumnParallelLinear,
                                               LinearBase,
                                               MergedColumnParallelLinear,
                                               QKVParallelLinear,
                                               RowParallelLinear)
from vllm.model_executor.layers.quantization.compressed_tensors.compressed_tensors import \
    CompressedTensorsLinearMethod
from vllm.model_executor.layers.quantization.utils.quant_utils import (
    pack_quantized_values_into_int32, quantize_weights,
    unpack_quantized_values_into_int32)
from vllm.model_executor.model_loader import get_model as vllm_get_model
from vllm.scalar_type import scalar_types

from tests.layers.common import utils as test_utils
from tpu_inference.layers.vllm.quantization import get_tpu_quantization_config
from tpu_inference.layers.vllm.quantization.compressed_tensors.compressed_tensors import \
    VllmCompressedTensorsConfig
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import \
    VllmCompressedTensorsWNA16
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig

P = PartitionSpec

torch.manual_seed(42)

# Public weight-only int4 group-128 checkpoint (same one the sibling W4A8
# test uses; here WITHOUT injecting activation quantization).
MODELS = [
    "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16",
]


@pytest.mark.parametrize("group_size", [32, 128])
def test_dequant_matches_compressed_tensors_reference(group_size):
    """Load-path unpack+dequant must match the compressed-tensors package
    reference bit-exactly.

    Builds a random symmetric int4 group-quantized weight, packs it with the
    reference ``pack_to_int32`` (what produced weight_packed on disk), then
    compares:
      reference: compressed_tensors.dequantize(q, scale, args)
      ours:      vllm unpack -> (uint4 - 8) -> int4 -> grouped scale multiply
                 (the exact math in
                 VllmCompressedTensorsWNA16.process_weights_after_loading +
                 xla_quantized_matmul's blockwise dequant branch)
    """
    out_features, in_features = 64, 256
    n_groups = in_features // group_size

    q = torch.randint(-8, 8, (out_features, in_features), dtype=torch.int8)
    scale = (torch.rand(out_features, n_groups, dtype=torch.float32) + 0.01)

    args = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=group_size,
    )

    # Reference path: what the checkpoint producer serialized and what the
    # reference consumer computes.
    packed = pack_to_int32(q, num_bits=4)
    assert packed.dtype == torch.int32
    assert packed.shape == (out_features, in_features // 8)
    ref_roundtrip = unpack_from_int32(packed,
                                      num_bits=4,
                                      shape=torch.Size(
                                          (out_features, in_features)))
    torch.testing.assert_close(ref_roundtrip, q, rtol=0, atol=0)
    ref_dequant = dequantize(q, scale, args=args, dtype=torch.float32)

    # Our load path: vllm's unpack (nibbles are offset-binary uint4),
    # recenter by -8, grouped scale multiply.
    ours_uint = unpack_quantized_values_into_int32(packed,
                                                   scalar_types.uint4,
                                                   packed_dim=1)
    ours_q = t2j(ours_uint, use_dlpack=False) - 8
    # Confirm the recentered nibbles equal the reference int4 values exactly.
    np.testing.assert_array_equal(np.asarray(ours_q), q.numpy())

    ours_int4 = ours_q.astype(jnp.int4)
    w = jnp.transpose(ours_int4)  # [in, out], as stored in HBM
    s = jnp.transpose(t2j(scale, use_dlpack=False))  # [groups, out]
    # Blockwise dequant exactly as xla_quantized_matmul's 2D-scale branch:
    in_blocks, out_blocks = s.shape[0], s.shape[1]
    block_in = in_features // in_blocks
    block_out = out_features // out_blocks
    ours_dequant = (w.reshape(in_blocks, block_in, out_blocks,
                              block_out).astype(jnp.float32) *
                    s[:, jnp.newaxis, :, jnp.newaxis]).reshape(
                        in_features, out_features)

    np.testing.assert_array_equal(
        np.asarray(ours_dequant),
        ref_dequant.T.numpy(),
        err_msg="wNa16 load-path dequant diverges from the "
        "compressed-tensors reference")


def ref_w4a16(x: torch.Tensor, w_float: torch.Tensor,
              b: Optional[torch.Tensor]):
    out = torch.einsum('bd,fd->bf', x.to(torch.float32),
                       w_float.to(torch.float32))
    if b is not None:
        out += b.to(torch.float32)
    return out.to(x.dtype)


def initialize_layer_weights(layer: torch.nn.Module) -> torch.Tensor:
    assert isinstance(layer, LinearBase)
    scheme = layer.scheme
    assert isinstance(scheme, VllmCompressedTensorsWNA16)
    quant_config = scheme.linear_config
    assert isinstance(quant_config, VllmQuantLinearConfig)

    group_size = scheme.group_size

    weight_list = []
    weight_ref_list = []
    weight_scale_list = []
    for output_size in quant_config.output_sizes:
        weight = torch.rand(
            (output_size, layer.input_size), dtype=torch.bfloat16) / 10

        # Transpose to group along input_size (dim 0 of weight.T).
        weight_ref_t, weight_q_t, weight_scale_t, _ = quantize_weights(
            weight.T, scalar_types.int4, group_size=group_size)

        # Offset to uint4 [0, 15] range: the on-disk packing convention.
        weight_uint4 = weight_q_t.T + 8
        packed_weight_ = pack_quantized_values_into_int32(weight_uint4,
                                                          scalar_types.uint4,
                                                          packed_dim=1)

        weight_list.append(packed_weight_)
        weight_ref_list.append(weight_ref_t.T)
        weight_scale_list.append(weight_scale_t.T)

    weight_packed = torch.concatenate(weight_list)
    weight_ref = torch.concatenate(weight_ref_list)
    weight_scale = torch.concatenate(weight_scale_list)

    assert layer.weight_packed.data.shape == weight_packed.shape
    assert layer.weight_scale.data.shape == weight_scale.shape

    layer.weight_packed.data = weight_packed
    layer.weight_scale.data = weight_scale.to(layer.weight_scale.data.dtype)

    if layer.bias is not None:
        layer.bias.data = torch.rand_like(layer.bias.data)
    return weight_ref


def return_ref_and_layer_output(layer: torch.nn.Module, batch_size: int = 16):
    weight_ref = initialize_layer_weights(layer)
    assert isinstance(layer, LinearBase)
    quant_method = layer.quant_method
    assert isinstance(quant_method, CompressedTensorsLinearMethod)

    input_tensor = torch.rand(
        batch_size, layer.input_size, dtype=torch.bfloat16) / 10
    input_tensor = input_tensor.to('cpu')

    ref_output = ref_w4a16(input_tensor, weight_ref, layer.bias)

    with torchax.default_env():
        quant_method.process_weights_after_loading(layer)

        # The memory win is the point: weights must be int4-resident.
        stored = layer.weight
        stored_list = list(stored) if isinstance(
            stored, torch.nn.ParameterList) else [stored]
        for w in stored_list:
            assert torchax.interop.jax_view(w).dtype == jnp.int4

        jax_input_tensor = torch_view(t2j(input_tensor, use_dlpack=False))
        layer_output = layer(jax_input_tensor)
        layer_output = j2t(layer_output.to(torch.float32)).to(torch.bfloat16)

    return ref_output, layer_output


@pytest.fixture(autouse=True)
def mock_get_pp_group():
    with patch("tpu_inference.distributed.jax_parallel_state.get_pp_group",
               return_value=MagicMock(is_first_rank=True,
                                      is_last_rank=True,
                                      rank_in_group=0,
                                      world_size=1)):
        yield


@pytest.fixture(autouse=True)
def setup_environment():
    engine_args = EngineArgs(
        model=MODELS[0],
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()

    with set_current_vllm_config(vllm_config):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            1,
            0,
            local_rank=0,
            distributed_init_method=f"file://{temp_file}",
            backend="gloo")
        ensure_model_parallel_initialized(1, 1)


@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
    test_utils.get_spmd_mesh(min(4, jax.local_device_count()))
])
@pytest.mark.parametrize("model", MODELS)
def test_quant_override(model, mesh):
    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    assert isinstance(quant_config, VllmCompressedTensorsConfig)
    assert quant_config.vllm_config == vllm_config
    assert quant_config.mesh == mesh


@pytest.mark.parametrize("mesh", [
    test_utils.get_spmd_mesh(1),
])
@pytest.mark.parametrize("model", MODELS)
def test_loading_model(model, mesh):
    """Natural-path load of a real wNa16 checkpoint: every LinearBase must
    select the WNA16 scheme (this is the exact selection that raised
    NotImplementedError before this scheme existed)."""
    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16

    vllm_config.quant_config = get_tpu_quantization_config(vllm_config, mesh)
    vllm_config.device_config.device = "cpu"

    with set_current_vllm_config(vllm_config):
        vllm_model = vllm_get_model(vllm_config=vllm_config)
    layers = test_utils.find_all_layer_type(vllm_model, LinearBase)
    assert len(layers) > 0
    quantized = [l for l in layers if hasattr(l, "scheme")]
    assert len(quantized) > 0
    for layer in quantized:
        assert isinstance(layer.scheme, VllmCompressedTensorsWNA16)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_row_parallel_linear(model, bias, num_devices, enable_sp):
    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = RowParallelLinear(
            input_size=1024,
            output_size=2048,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("num_devices", [1, min(4, jax.local_device_count())])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_column_parallel_linear(model, bias, num_devices, enable_sp):
    mesh = test_utils.get_spmd_mesh(num_devices)
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.compilation_config.pass_config.enable_sp = enable_sp

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = ColumnParallelLinear(
            input_size=1024,
            output_size=2048,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_qkv_parallel_linear(model, bias, fuse_matmuls):
    mesh = test_utils.get_spmd_mesh(1)
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = QKVParallelLinear(
            hidden_size=1024,
            head_size=64,
            total_num_heads=16,
            total_num_kv_heads=4,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.scheme.linear_config.fuse_matmuls = fuse_matmuls

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("bias", [False, True])
@pytest.mark.parametrize("fuse_matmuls", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_merged_column_parallel_linear(model, bias, fuse_matmuls):
    mesh = test_utils.get_spmd_mesh(1)
    dtype = torch.bfloat16

    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()

    vllm_config.model_config.dtype = dtype
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        linear_layer = MergedColumnParallelLinear(
            input_size=1024,
            output_sizes=[2048] * 2,
            bias=bias,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
        linear_layer.scheme.linear_config.fuse_matmuls = fuse_matmuls

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("model", MODELS)
def test_weight_only_non_wNa16_fails_closed_with_clear_error(model):
    """A weight-only config that is NOT 4-bit/group/pack-quantized
    (channelwise, 2/3/5/6/7/8-bit, non-pack formats) must refuse with a
    clear NotImplementedError -- not die in _is_dynamic_token_w8a8's
    input_quant.num_bits dereference (bare AttributeError), which is the
    misleading-crash mechanism this scheme's own docstring documents.
    Modeled by forcing is_wNa16_group to reject the real W4A16 config."""
    mesh = test_utils.get_spmd_mesh(1)
    engine_args = EngineArgs(
        model=model,
        max_model_len=64,
        max_num_batched_tokens=64,
        max_num_seqs=4,
    )
    vllm_config = engine_args.create_engine_config()
    vllm_config.model_config.dtype = torch.bfloat16
    quant_config = get_tpu_quantization_config(vllm_config, mesh)
    with set_current_vllm_config(vllm_config):
        with patch(
                "tpu_inference.layers.vllm.quantization.compressed_tensors"
                ".compressed_tensors.is_wNa16_group",
                return_value=False), \
             pytest.raises(NotImplementedError, match="Weight-only"):
            RowParallelLinear(
                input_size=1024,
                output_size=2048,
                bias=False,
                params_dtype=torch.bfloat16,
                return_bias=False,
                quant_config=quant_config,
            )


def test_w4a16_moe_dispatch_fails_closed():
    """get_moe_method must not silently route a weight-only int4 (W4A16)
    checkpoint's RoutedExperts through the activation-quantizing W4A8
    method: linears would run true W4A16 while experts run numerics the
    export was never calibrated for. Fail-closed, like the fp8
    non-serialized guard one dispatch table over."""
    from vllm.model_executor.layers.fused_moe import RoutedExperts

    from tpu_inference.layers.vllm.quantization.compressed_tensors \
        .compressed_tensors_moe.compressed_tensors_moe import \
        VllmCompressedTensorsMoEMethod

    weight_quant = QuantizationArgs(num_bits=4,
                                    type="int",
                                    strategy="group",
                                    group_size=128,
                                    symmetric=True)
    quant_config = MagicMock()
    quant_config.get_scheme_dict.return_value = {
        "weights": weight_quant,
        "input_activations": None,
    }
    quant_config._is_fp8_w8a8.return_value = False
    layer = MagicMock(spec=RoutedExperts)

    with pytest.raises(NotImplementedError, match="W4A16"):
        VllmCompressedTensorsMoEMethod.get_moe_method(
            quant_config, layer, "model.layers.0.mlp.experts")

    # Positive control: a genuine w4a8 config (input activations quantized)
    # still reaches the W4A8 method through the same dispatch.
    #
    # moe_config has to be set explicitly. MagicMock(spec=RoutedExperts) takes
    # its allowed attributes from dir(RoutedExperts), and moe_config is assigned
    # in __init__ rather than declared on the class, so a spec'd mock raises
    # AttributeError on the dispatch's `layer.moe_config` read. The guard above
    # returns before that line, which is why only the positive control reached
    # it -- the negative case would pass either way.
    layer.moe_config = MagicMock()
    quant_config.get_scheme_dict.return_value = {
        "weights": weight_quant,
        "input_activations": MagicMock(num_bits=8),
    }
    with patch(
            "tpu_inference.layers.vllm.quantization.compressed_tensors"
            ".compressed_tensors_moe.compressed_tensors_moe_w4a8"
            ".VllmCompressedTensorsW4A8MoEMethod") as ctor:
        VllmCompressedTensorsMoEMethod.get_moe_method(
            quant_config, layer, "model.layers.0.mlp.experts")
        ctor.assert_called_once()
