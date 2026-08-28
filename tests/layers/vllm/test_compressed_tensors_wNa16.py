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

Hardening (review findings): TP>1 cases never silently collapse to one
device (forced CPU host devices, else a VISIBLE skip); layer fixtures are
signed and give every quantization group a materially distinct scale;
deterministic worst-case int4 bit patterns are asserted bit-exactly; and
the forward runs at two token counts in one process to catch trace-cached
state reuse.
"""

import os

# Must run before the JAX backend initializes: give single-chip CPU hosts
# real multi-device coverage. The old `min(4, jax.local_device_count())`
# parametrization, evaluated at collection time, silently collapsed every
# tensor-parallel case to one device on such hosts — the suite "passed"
# with ZERO TP coverage. JAX honors this flag on CPU; if some earlier
# import already initialized the backend, the TP>1 cases below skip
# VISIBLY instead of quietly running at TP=1 (see require_devices).
_FORCED_HOST_DEVICES = 4
_xla_flags = os.environ.get("XLA_FLAGS", "")
if "xla_force_host_platform_device_count" not in _xla_flags:
    os.environ["XLA_FLAGS"] = (
        f"{_xla_flags} "
        f"--xla_force_host_platform_device_count={_FORCED_HOST_DEVICES}"
    ).strip()

import tempfile
from typing import Optional
from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch
import torchax
from compressed_tensors import CompressionFormat
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
from tpu_inference.layers.vllm.quantization.compressed_tensors.schemes.compressed_tensors_wNa16 import (
    VllmCompressedTensorsWNA16, is_wNa16_group)
from tpu_inference.layers.vllm.quantization.configs import \
    VllmQuantLinearConfig

P = PartitionSpec

torch.manual_seed(42)

# Public weight-only int4 group-128 checkpoint (same one the sibling W4A8
# test uses; here WITHOUT injecting activation quantization).
MODELS = [
    "nm-testing/SmolLM-1.7B-Instruct-quantized.w4a16",
]


def require_devices(num_devices: int):
    """Return an SPMD mesh over exactly ``num_devices`` devices, or skip
    VISIBLY.

    Never falls back to fewer devices: a TP=4 case that cannot get 4
    devices must show up as SKIPPED in the run summary, not silently pass
    as TP=1 (which is what the old collection-time
    ``min(4, jax.local_device_count())`` did)."""
    available = jax.local_device_count()
    if available < num_devices:
        pytest.skip(
            f"needs {num_devices} local devices for TP={num_devices}, host "
            f"has {available} (XLA_FLAGS --xla_force_host_platform_device_"
            f"count={_FORCED_HOST_DEVICES} was set too late to take effect "
            "in this process)")
    mesh = test_utils.get_spmd_mesh(num_devices)
    assert mesh.devices.size == num_devices, (
        "mesh helper returned fewer devices than requested — TP coverage "
        "would silently degrade")
    return mesh


# Per-group magnitude exponents (gain = 2**e): a fixed APERIODIC pattern over
# {1x, 2x, 4x, 8x}. Adjacent groups always differ by >= 2x, and for every
# cyclic group offset k (1 <= k < n_groups, n_groups <= 32) at least 62% of
# the groups change scale by >= 2x -- so any group misalignment, not just
# off-by-one, is a large output error. The range is deliberately bounded at
# 8x: an exponential 2**g ramp across 16 groups (this checkpoint is group 64,
# so a 1024-wide input has 16 groups) pushed outputs into the thousands,
# where bf16 rounding of the per-shard partial sums that TP>1 all-reduces
# (1 ulp = 16..32) swamped the 0.05 absolute tolerance on elements that
# cancel -- a fixture artifact that masqueraded as a TP=4 bug.
_GROUP_GAIN_EXPONENTS = (2, 1, 3, 0, 1, 3, 2, 0, 1, 3, 2, 3, 1, 2, 0, 1, 0, 2,
                         1, 3, 1, 3, 2, 1, 0, 3, 0, 2, 1, 3, 0, 2)


def group_gains(n_groups: int) -> torch.Tensor:
    assert n_groups <= len(_GROUP_GAIN_EXPONENTS), n_groups
    exponents = torch.tensor(_GROUP_GAIN_EXPONENTS[:n_groups],
                             dtype=torch.float32)
    return 2.0**exponents


def assert_group_scales_distinct(weight_scale_t: torch.Tensor) -> None:
    """``weight_scale_t``: [n_groups, out]. Every adjacent pair of groups
    must differ by ~2x or more (either direction) in median scale, or a
    group-indexing regression can hide inside the 5% forward tolerance."""
    med = weight_scale_t.abs().float().median(dim=1).values
    ratio = med[1:] / med[:-1]
    assert bool(((ratio > 1.9) | (ratio < 0.53)).all()), (
        "fixture regression: adjacent per-group scales are not materially "
        f"distinct: {ratio.tolist()}")


def make_group_structured_weight(output_size: int, input_size: int,
                                 group_size: int) -> torch.Tensor:
    """Signed, zero-mean weights whose magnitude follows the fixed
    aperiodic 1x/2x/4x/8x pattern above from one input group to the next.

    Two review findings shaped this draw. Uniform ``torch.rand(...) / 10``
    fixtures gave (a) per-group scales identical to within ~1%, so a
    group-INDEXING regression (scales offset by one group) moved outputs by
    less than the 5% forward tolerance and passed; and (b) all-positive
    weights, so symmetric quantization never emitted a negative int4 code
    and the entire [-8, 0) half of the format went unexercised. Group-
    distinct magnitudes turn any scale/group misalignment into a >= 2x
    error on most groups, and the zero-mean draw populates both signs."""
    base = (torch.rand(
        (output_size, input_size), dtype=torch.float32) - 0.5) / 5.0
    n_groups = input_size // group_size
    gains = group_gains(n_groups).repeat_interleave(group_size)
    return (base * gains).to(torch.bfloat16)


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


@pytest.mark.parametrize("pattern", ["all_16_codes", "alternating_extremes"])
def test_signed_int4_codes_bit_exact(pattern):
    """Deterministic worst-case bit patterns through pack -> unpack ->
    recenter -> grouped dequant, asserted bit-exactly against the
    compressed-tensors reference.

    ``all_16_codes`` tiles every int4 code including the asymmetric edge
    code -8; ``alternating_extremes`` alternates -8/7 in both nibble
    phases, so every packed int32 is a worst-case bit pattern. The layer
    fixtures used to be all-positive, so the negative half of the format
    was never decoded at all."""
    out_features, in_features, group_size = 8, 128, 32

    if pattern == "all_16_codes":
        q = torch.arange(-8, 8, dtype=torch.int8).repeat(
            out_features, in_features // 16)
    else:
        q = torch.empty((out_features, in_features), dtype=torch.int8)
        q[:, 0::2], q[:, 1::2] = -8, 7
        q[1::2, 0::2], q[1::2, 1::2] = 7, -8  # both nibble phases
    assert int(q.min()) == -8 and int(q.max()) == 7

    scale = (torch.rand(out_features,
                        in_features // group_size,
                        dtype=torch.float32) + 0.01)
    args = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=group_size,
    )

    packed = pack_to_int32(q, num_bits=4)
    ref_dequant = dequantize(q, scale, args=args, dtype=torch.float32)

    ours_uint = unpack_quantized_values_into_int32(packed, scalar_types.uint4,
                                                   packed_dim=1)
    ours_q = t2j(ours_uint, use_dlpack=False) - 8
    np.testing.assert_array_equal(
        np.asarray(ours_q),
        q.numpy(),
        err_msg="negative int4 codes corrupted by unpack/recenter")

    ours_int4 = ours_q.astype(jnp.int4)
    w = jnp.transpose(ours_int4)
    s = jnp.transpose(t2j(scale, use_dlpack=False))
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
        err_msg=f"wNa16 dequant diverges from the compressed-tensors "
        f"reference on the {pattern} pattern")


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
        weight = make_group_structured_weight(output_size, layer.input_size,
                                              group_size)

        # Transpose to group along input_size (dim 0 of weight.T).
        weight_ref_t, weight_q_t, weight_scale_t, _ = quantize_weights(
            weight.T, scalar_types.int4, group_size=group_size)

        # Fixture self-checks (the review findings this fixture exists
        # for). Both int4 sign halves must actually occur:
        assert int(weight_q_t.min()) < 0 < int(weight_q_t.max()), (
            "fixture regression: quantized codes are single-signed, the "
            "negative int4 half is unexercised")
        # ...and adjacent groups must have materially distinct scales, or
        # group-indexing regressions hide inside the 5% forward tolerance:
        assert_group_scales_distinct(weight_scale_t)

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


@pytest.mark.parametrize("num_devices", [1, 4], ids=["tp1", "tp4"])
@pytest.mark.parametrize("model", MODELS)
def test_quant_override(model, num_devices):
    mesh = require_devices(num_devices)
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


@pytest.mark.parametrize("model", MODELS)
def test_loading_model(model):
    """Natural-path load of a real wNa16 checkpoint: every LinearBase must
    select the WNA16 scheme (this is the exact selection that raised
    NotImplementedError before this scheme existed)."""
    mesh = require_devices(1)
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
@pytest.mark.parametrize("num_devices", [1, 4], ids=["tp1", "tp4"])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_row_parallel_linear(model, bias, num_devices, enable_sp):
    mesh = require_devices(num_devices)
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
@pytest.mark.parametrize("num_devices", [1, 4], ids=["tp1", "tp4"])
@pytest.mark.parametrize("enable_sp", [False, True])
@pytest.mark.parametrize("model", MODELS)
def test_column_parallel_linear(model, bias, num_devices, enable_sp):
    mesh = require_devices(num_devices)
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
@pytest.mark.parametrize("num_devices", [1, 4], ids=["tp1", "tp4"])
@pytest.mark.parametrize("model", MODELS)
def test_qkv_parallel_linear(model, bias, fuse_matmuls, num_devices):
    """TP>1 + fuse_matmuls=True is the only path that exercises the
    n_shards>1 fused-tensor reorder (reorder_concatenated_tensor_for_
    sharding at load, undone by slice_sharded_tensor_for_concatenation at
    forward). At TP=1 the reorder is the identity, so single-device runs
    cannot see a reorder/slice mismatch."""
    mesh = require_devices(num_devices)
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
@pytest.mark.parametrize("num_devices", [1, 4], ids=["tp1", "tp4"])
@pytest.mark.parametrize("model", MODELS)
def test_merged_column_parallel_linear(model, bias, fuse_matmuls,
                                       num_devices):
    """See test_qkv_parallel_linear: TP>1 makes the fused reorder real."""
    mesh = require_devices(num_devices)
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


def _stored_dequant_bit_exact(layer) -> torch.Tensor:
    """Load a group-structured weight into ``layer`` (already created with
    the WNA16 scheme), run process_weights_after_loading, and require the
    STORED int4 weights x STORED scales to reproduce q * bf16(scale)
    EXACTLY, group by group, in the production layout.

    Returns the bf16 dequantized reference weight ``[out, in]`` so callers
    can also check the forward."""
    scheme = layer.scheme
    assert isinstance(scheme, VllmCompressedTensorsWNA16)
    dtype = torch.bfloat16
    in_features, out_features = layer.input_size, layer.output_size
    group_size = scheme.group_size
    n_groups = in_features // group_size

    weight = make_group_structured_weight(out_features, in_features,
                                          group_size)
    weight_ref_t, weight_q_t, weight_scale_t, _ = quantize_weights(
        weight.T, scalar_types.int4, group_size=group_size)
    assert int(weight_q_t.min()) < 0 < int(weight_q_t.max())
    assert_group_scales_distinct(weight_scale_t)

    layer.weight_packed.data = pack_quantized_values_into_int32(
        weight_q_t.T + 8, scalar_types.uint4, packed_dim=1)
    layer.weight_scale.data = weight_scale_t.T.to(dtype)

    quant_method = layer.quant_method
    assert isinstance(quant_method, CompressedTensorsLinearMethod)
    with torchax.default_env():
        quant_method.process_weights_after_loading(layer)
        stored_w = layer.weight
        stored_s = layer.weight_scale
        w_list = list(stored_w) if isinstance(
            stored_w, torch.nn.ParameterList) else [stored_w]
        s_list = list(stored_s) if isinstance(
            stored_s, torch.nn.ParameterList) else [stored_s]
    assert len(w_list) == 1 and len(s_list) == 1

    w_jax = torchax.interop.jax_view(w_list[0])
    s_jax = torchax.interop.jax_view(s_list[0])
    assert w_jax.dtype == jnp.int4
    assert w_jax.shape == (in_features, out_features)
    assert s_jax.shape == (n_groups, out_features)

    w_np = np.asarray(w_jax.astype(jnp.float32))  # [in, out]
    s_np = np.asarray(s_jax.astype(jnp.float32))  # [groups, out]
    ours = (w_np.reshape(n_groups, group_size, out_features) *
            s_np[:, None, :]).reshape(in_features, out_features)

    ref_q = weight_q_t.to(torch.float32).numpy()  # [in, out]
    ref_s = weight_scale_t.to(dtype).to(torch.float32).numpy()  # [g, out]
    ref = (ref_q.reshape(n_groups, group_size, out_features) *
           ref_s[:, None, :]).reshape(in_features, out_features)

    np.testing.assert_array_equal(
        ours,
        ref,
        err_msg="stored int4 weights + stored scales no longer reproduce "
        "the per-group reference dequantization — group indexing or layout "
        "broke in the load path")
    return weight_ref_t.T


@pytest.mark.parametrize("model", MODELS)
def test_stored_weights_dequant_bit_exact_per_group(model):
    """Push a group-structured weight through the REAL load path
    (create_weights -> weight_packed/weight_scale -> process_weights_after_
    loading), then dequantize the STORED int4 weights with the STORED
    scales in the production layout and require exact equality with
    q * bf16(scale), group by group.

    With the group-distinct fixture, a scale tensor offset by any number
    of groups is a >= 2x error on most groups here. The old uniform
    fixture kept exactly that bug inside the layer tests' 5% tolerance."""
    mesh = require_devices(1)
    dtype = torch.bfloat16
    out_features, in_features = 512, 1024

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
        layer = RowParallelLinear(
            input_size=in_features,
            output_size=out_features,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    _stored_dequant_bit_exact(layer)




def _rebuild_layer_with_group_size(layer, group_size: int):
    """Swap the checkpoint-derived scheme (group 64 for the test model) for
    a directly constructed WNA16 scheme with ``group_size`` and re-create
    the on-disk-format parameters, keeping the layer's linear_config."""
    old_scheme = layer.scheme
    assert isinstance(old_scheme, VllmCompressedTensorsWNA16)
    weight_quant = QuantizationArgs(
        num_bits=4,
        type="int",
        symmetric=True,
        strategy="group",
        group_size=group_size,
    )
    scheme = VllmCompressedTensorsWNA16(
        weight_quant=weight_quant, linear_config=old_scheme.linear_config)
    for name in ("weight_packed", "weight_scale", "weight_shape"):
        if hasattr(layer, name):
            delattr(layer, name)
    scheme.create_weights(
        layer,
        output_size=layer.output_size,
        input_size=layer.input_size,
        output_partition_sizes=[layer.output_size],
        input_size_per_partition=layer.input_size,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )
    layer.scheme = scheme
    return scheme


@pytest.mark.parametrize("group_size", [32, 128])
@pytest.mark.parametrize("model", MODELS)
def test_layer_path_other_group_sizes(model, group_size):
    """The layer tests above all inherit the test checkpoint's group size
    (64). The scheme's real targets differ: Google's gemma-4 QAT wNa16
    exports are group 32 and RedHatAI's GPTQ exports are group 128 — run
    the full layer path (create_weights -> load -> process -> forward,
    plus the stored-dequant bit-exact check) at both."""
    mesh = require_devices(1)
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
        layer = RowParallelLinear(
            input_size=1024,
            output_size=512,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    scheme = _rebuild_layer_with_group_size(layer, group_size)
    assert scheme.group_size == group_size

    weight_ref = _stored_dequant_bit_exact(layer)

    x = torch.rand(8, layer.input_size, dtype=dtype) / 10
    ref = ref_w4a16(x, weight_ref, None)
    with torchax.default_env():
        out = layer(torch_view(t2j(x, use_dlpack=False)))
        out = j2t(out.to(torch.float32)).to(dtype)
    torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("model", MODELS)
def test_forward_alternating_extreme_codes(model):
    """End-to-end forward with the worst-case alternating [-8, 7] code
    pattern packed into a real layer, against an integer-math reference.

    The all-positive fixtures never emitted a single negative int4 code,
    so a sign-handling bug in the unpack -> recenter -> dequant chain
    (e.g. dropping the sign of the recentered nibble) was invisible to
    every layer test in this file."""
    mesh = require_devices(1)
    dtype = torch.bfloat16
    out_features, in_features = 256, 512

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
        layer = RowParallelLinear(
            input_size=in_features,
            output_size=out_features,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )
    group_size = layer.scheme.group_size

    q = torch.empty((out_features, in_features), dtype=torch.int8)
    q[:, 0::2], q[:, 1::2] = -8, 7
    q[1::2, 0::2], q[1::2, 1::2] = 7, -8  # both nibble phases
    scale = (torch.rand(out_features,
                        in_features // group_size,
                        dtype=torch.float32) * 0.05 + 0.01).to(dtype)

    # Reference-packed, exactly as serialized on disk (bit-compatibility
    # with vllm's unpack is proven by test_signed_int4_codes_bit_exact).
    layer.weight_packed.data = pack_to_int32(q, num_bits=4)
    layer.weight_scale.data = scale

    w_float = (q.to(torch.float32) * scale.to(torch.float32).
               repeat_interleave(group_size, dim=1))
    x = ((torch.rand(4, in_features, dtype=torch.float32) - 0.5) /
         5).to(dtype)
    ref = torch.einsum('bd,fd->bf', x.to(torch.float32), w_float)

    with torchax.default_env():
        layer.quant_method.process_weights_after_loading(layer)
        out = layer(torch_view(t2j(x, use_dlpack=False)))
        out = j2t(out.to(torch.float32))
    torch.testing.assert_close(out, ref, rtol=0.05, atol=0.05)


@pytest.mark.parametrize("model", MODELS)
def test_forward_two_shapes_one_process(model):
    """Run the SAME processed layer at two distinct token counts in one
    process, then the first shape again, each against its own reference.

    Guards the bug class PR #3/#5 belonged to: state captured or cached at
    trace time leaking into a later trace (or corrupting the earlier one)
    when a second batch/sequence shape compiles in the same process. A
    single forward at a single shape can never see it."""
    mesh = require_devices(1)
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
        layer = RowParallelLinear(
            input_size=1024,
            output_size=2048,
            bias=True,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    weight_ref = initialize_layer_weights(layer)
    bias_ref = layer.bias.data.clone()

    inputs, refs = {}, {}
    for batch in (16, 3):
        x = (torch.rand(batch, layer.input_size, dtype=dtype) - 0.5) / 5
        inputs[batch] = x
        refs[batch] = ref_w4a16(x, weight_ref, bias_ref)

    quant_method = layer.quant_method
    assert isinstance(quant_method, CompressedTensorsLinearMethod)
    with torchax.default_env():
        quant_method.process_weights_after_loading(layer)
        for batch in (16, 3, 16):  # shape A, shape B, then A again
            out = layer(torch_view(t2j(inputs[batch], use_dlpack=False)))
            out = j2t(out.to(torch.float32)).to(dtype)
            torch.testing.assert_close(
                out,
                refs[batch],
                rtol=0.05,
                atol=0.05,
                msg=lambda m, b=batch: (
                    f"forward at batch={b} diverged from its reference "
                    f"after another shape was traced in the same process "
                    f"(trace-cached state reuse): {m}"))


def _weight_args(**overrides):
    base = dict(num_bits=4,
                type="int",
                symmetric=True,
                strategy="group",
                group_size=64)
    base.update(overrides)
    return QuantizationArgs(**base)


def test_is_wNa16_group_predicate_screens_configs():
    """The dispatch predicate must accept exactly the weight-only int4
    group pack-quantized shape and screen out the config families that the
    scheme cannot serve, so they reach get_scheme's fail-closed
    NotImplementedError instead of crashing somewhere misleading."""
    pack = CompressionFormat.pack_quantized.value
    ok = _weight_args()
    assert is_wNa16_group(ok, None, pack)

    # Screened here (fall through to the explicit weight-only refusal):
    assert not is_wNa16_group(
        _weight_args(strategy="channel", group_size=None), None,
        pack)  # channelwise int4
    assert not is_wNa16_group(_weight_args(num_bits=8), None, pack)  # w8
    assert not is_wNa16_group(ok, None, "int-quantized")  # non-pack format
    assert not is_wNa16_group(ok, None, None)  # missing format
    assert not is_wNa16_group(None, None, pack)  # no weight quant at all
    assert not is_wNa16_group(ok, _weight_args(num_bits=8),
                              pack)  # activation quant present -> not wNa16

    # NOT screened here: these are wNa16-shaped, so they reach the scheme
    # constructor, which must refuse them itself (next test).
    assert is_wNa16_group(_weight_args(symmetric=False), None, pack)
    assert is_wNa16_group(_weight_args(actorder="group"), None, pack)


def test_scheme_ctor_fails_closed_on_unsupported_configs():
    """Every unsupported config family must die in the constructor with a
    clear NotImplementedError naming the limitation — never load weights
    and produce garbage, never crash later in an unrelated dereference."""
    cases = [
        (_weight_args(symmetric=False), "symmetric"),
        (_weight_args(actorder="group"), "actorder"),
        (_weight_args(num_bits=8), "num_bits"),
        (_weight_args(strategy="channel", group_size=None), "group strategy"),
    ]
    for weight_quant, match in cases:
        with pytest.raises(NotImplementedError, match=match):
            VllmCompressedTensorsWNA16(weight_quant=weight_quant,
                                       linear_config=MagicMock())


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


# ---------------------------------------------------------------------------
# gmm_v2 fused-kernel routing (ENABLE_QUANTIZED_MATMUL_KERNEL gate)
# ---------------------------------------------------------------------------


def _spy_gmm_v2(calls: list):
    """Jax-traceable stand-in for tokamax ``gmm_v2`` with the contract the
    linear wrapper relies on: int4 rhs ``[1, k, n]``, 4D rhs_scale
    ``[1, num_blocks, 1, n]``, blockwise dequant in lhs dtype, matmul with
    f32 accumulation. Records the static call facts the route assertions
    need (at trace time, inside shard_map). CPU tests cannot execute the
    real Mosaic kernel; real-kernel numerics are validated on TPU hardware
    (temp-0 canary vs the XLA path)."""

    def fake_gmm_v2(*, lhs, rhs, group_sizes, rhs_scale, rhs_bias,
                    group_offset, zero_initialize, preferred_element_type,
                    maybe_quantize_lhs):
        calls.append({
            "maybe_quantize_lhs": maybe_quantize_lhs,
            "lhs_dtype": lhs.dtype,
            "rhs_dtype": rhs.dtype,
            "scale_shape": tuple(rhs_scale.shape),
        })
        w = rhs[0]  # [k, n] int4, per shard
        scale = rhs_scale[0]  # [num_blocks, 1, n]
        k, n = w.shape
        num_blocks = scale.shape[0]
        group = k // num_blocks
        w_deq = (w.reshape(num_blocks, group, n).astype(lhs.dtype) *
                 scale.astype(lhs.dtype)).reshape(k, n)
        return jax.lax.dot_general(
            lhs,
            w_deq,
            dimension_numbers=(((1, ), (0, )), ((), ())),
            preferred_element_type=jnp.float32).astype(preferred_element_type)

    return fake_gmm_v2


def _stored_scales(layer) -> list:
    stored = layer.weight_scale
    return list(stored) if isinstance(stored, torch.nn.ParameterList) else [
        stored
    ]


@pytest.mark.parametrize("model", MODELS)
def test_kernel_route_not_taken_by_default(model, monkeypatch):
    """Without ENABLE_QUANTIZED_MATMUL_KERNEL the scheme must keep its
    original XLA dequant route: 2D stored scales, gmm_v2 never invoked.
    Guards the env-gating of the c1 change (default behavior unchanged)."""
    monkeypatch.delenv("ENABLE_QUANTIZED_MATMUL_KERNEL", raising=False)
    mesh = require_devices(1)
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
        linear_layer = ColumnParallelLinear(
            input_size=1024,
            output_size=2048,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    calls = []
    monkeypatch.setattr("tpu_inference.layers.common.linear.gmm_v2",
                        _spy_gmm_v2(calls))

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)

    assert calls == [], (
        "gmm_v2 must not be reached when the env gate is unset")
    for s in _stored_scales(linear_layer):
        assert torchax.interop.jax_view(s).ndim == 2, (
            "default route must keep the 2D blockwise scale that steers "
            "sharded_quantized_matmul onto its XLA path")


@pytest.mark.parametrize("layer_cls",
                         [ColumnParallelLinear, RowParallelLinear],
                         ids=["column", "row"])
@pytest.mark.parametrize("num_devices", [1, 4], ids=["tp1", "tp4"])
@pytest.mark.parametrize("model", MODELS)
def test_gmm_v2_kernel_route_env_gated(model, num_devices, layer_cls,
                                       monkeypatch):
    """ENABLE_QUANTIZED_MATMUL_KERNEL=1 must steer wNa16 onto the fused
    gmm_v2 route with W4A16 semantics intact: 4D stored scale, int4 rhs
    (weights stay packed), lhs bf16 and NEVER quantized
    (maybe_quantize_lhs=False), and layer numerics unchanged vs the
    dequantized reference."""
    monkeypatch.setenv("ENABLE_QUANTIZED_MATMUL_KERNEL", "1")
    mesh = require_devices(num_devices)
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
        linear_layer = layer_cls(
            input_size=1024,
            output_size=2048,
            bias=False,
            params_dtype=dtype,
            return_bias=False,
            quant_config=quant_config,
        )

    calls = []
    monkeypatch.setattr("tpu_inference.layers.common.linear.gmm_v2",
                        _spy_gmm_v2(calls))

    ref_output, layer_output = return_ref_and_layer_output(linear_layer)
    torch.testing.assert_close(ref_output, layer_output, rtol=0.05, atol=0.05)

    assert calls, "env gate set but the gmm_v2 kernel route was not taken"
    for call in calls:
        # W4A16: activations must never be quantized on the kernel route.
        assert call["maybe_quantize_lhs"] is False
        assert call["lhs_dtype"] == jnp.bfloat16
        # Weights must reach the kernel still int4-packed (the memory win).
        assert call["rhs_dtype"] == jnp.int4
        # tokamax contract: rhs_scale [size_group=1, num_blocks, 1, n].
        assert len(call["scale_shape"]) == 4
        assert call["scale_shape"][0] == 1 and call["scale_shape"][2] == 1

    for s in _stored_scales(linear_layer):
        assert torchax.interop.jax_view(s).ndim == 4, (
            "kernel route requires the 4D scale emitted by "
            "format_linear_scale")


# ---------------------------------------------------------------------------
# XLA-path dequant expression (w4tax fix a'): bf16 leg must be bit-identical
# to the historical f32 leg
# ---------------------------------------------------------------------------


def _f32_leg_reference(x, w_q, w_scale):
    """The pre-a' expression of xla_quantized_matmul's 2D-scale branch,
    verbatim: f32 product, degenerate 4D reshape, cast to x.dtype, then the
    same dot."""
    in_features, out_features = w_q.shape
    in_blocks, out_blocks = w_scale.shape
    w = (w_q.reshape(in_blocks, in_features // in_blocks, out_blocks,
                     out_features // out_blocks).astype(jnp.float32) *
         w_scale[:, None, :, None]).reshape(in_features,
                                            out_features).astype(x.dtype)
    out = jax.lax.dot_general(x,
                              w,
                              dimension_numbers=(((1, ), (0, )), ((), ())),
                              preferred_element_type=jnp.float32)
    return out.astype(x.dtype)


@pytest.mark.parametrize("in_features,out_features,group_size",
                         [(1024, 2048, 32), (3840, 1024, 128),
                          (256, 512, 64)])
def test_xla_dequant_bf16_leg_bit_exact_vs_f32_leg(in_features, out_features,
                                                   group_size):
    """xla_quantized_matmul's 2D-scale branch now dequantizes in the
    activation dtype (no f32 leg, no degenerate trailing dim for the
    per-output-channel wNa16 layout). int4 codes are exact in bf16 and the
    code x bf16-scale product has <= 16 significant bits, so the single
    rounding to bf16 lands on the same value as the f32 route: the layer
    output must be BIT-IDENTICAL, over wide scale magnitudes and both
    signs, eager and jitted."""
    from tpu_inference.layers.common.linear import xla_quantized_matmul

    key = jax.random.PRNGKey(1234)
    k_w, k_s, k_x = jax.random.split(key, 3)
    codes = jax.random.randint(k_w, (in_features, out_features), -8, 8,
                               dtype=jnp.int32)
    w_q = codes.astype(jnp.int4)
    assert int(codes.min()) == -8 and int(codes.max()) == 7
    # log-uniform scales across 2**-20 .. 2**6: exercises bf16 exponent
    # range, not just the ~1e-2 magnitudes real checkpoints happen to have.
    w_scale = (2.0**jax.random.uniform(k_s, (in_features // group_size,
                                             out_features),
                                       minval=-20,
                                       maxval=6)).astype(jnp.bfloat16)
    x = jax.random.normal(k_x, (16, in_features), jnp.bfloat16)

    ref = _f32_leg_reference(x, w_q, w_scale)
    got = xla_quantized_matmul(x, w_q, w_scale, quantize_activation=False)
    got_jit = jax.jit(lambda x, w, s: xla_quantized_matmul(
        x, w, s, quantize_activation=False))(x, w_q, w_scale)

    np.testing.assert_array_equal(
        np.asarray(got, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        err_msg="bf16-leg dequant is not bit-identical to the f32 leg")
    np.testing.assert_array_equal(
        np.asarray(got_jit, dtype=np.float32),
        np.asarray(ref, dtype=np.float32),
        err_msg="jitted bf16-leg dequant is not bit-identical to the f32 leg")
    assert not bool(jnp.isnan(got).any())


def test_xla_dequant_f32_scale_keeps_f32_route():
    """When the scale dtype differs from the activation dtype (e.g. f32
    scales), the f32 product is kept so the scale is NOT rounded to bf16
    before the multiply: output must equal the historical expression
    bit-for-bit (and would NOT, in general, equal a bf16-rounded-scale
    variant)."""
    from tpu_inference.layers.common.linear import xla_quantized_matmul

    key = jax.random.PRNGKey(7)
    k_w, k_s, k_x = jax.random.split(key, 3)
    in_features, out_features, group_size = 512, 256, 64
    w_q = jax.random.randint(k_w, (in_features, out_features), -8, 8,
                             dtype=jnp.int32).astype(jnp.int4)
    # f32 scales with mantissa bits beyond bf16 precision.
    w_scale = jax.random.uniform(k_s, (in_features // group_size,
                                       out_features),
                                 jnp.float32,
                                 minval=0.01,
                                 maxval=0.5)
    assert not bool(
        jnp.array_equal(w_scale, w_scale.astype(jnp.bfloat16).astype(
            jnp.float32)))
    x = jax.random.normal(k_x, (16, in_features), jnp.bfloat16)

    ref = _f32_leg_reference(x, w_q, w_scale)
    got = xla_quantized_matmul(x, w_q, w_scale, quantize_activation=False)
    np.testing.assert_array_equal(np.asarray(got, dtype=np.float32),
                                  np.asarray(ref, dtype=np.float32))
