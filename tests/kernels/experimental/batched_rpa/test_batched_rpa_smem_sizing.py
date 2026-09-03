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
"""SMEM sizing regression tests for the batched RPA kernel.

Locks the arithmetic behind the 2026-08-27 incident: gemma-4-26B-A4B on
v6e-8 (TP=8, max-model-len 16384 -> auto page_size 16, max-num-seqs 16,
bf16 activations, fp8 KV) died at warmup compile with

    RESOURCE_EXHAUSTED: ... Ran out of memory in memory space smem.
    Used 1.10M of 1.00M smem. ...
    op_name=".../rpa_metadata_schedule/pallas_call"
    Shape: s32[163840]{0:T(128)}  (640.0K scratch)

Root cause: max_steps_ub used `max(1, steps_fit // num_lanes) * num_lanes`,
which rounds UP to num_lanes (128) steps whenever fewer than 128 steps
actually fit in the SMEM budget. At the incident config the MIXED-mode fit
was 105 steps; rounding up to 128 made the dma_kv_new scratch alone
128 * 2 * 160 * 4 = 163840 words (640KiB) and the total schedule scratch
1.101MiB > 1MiB. These tests pin the numbers so the sizing formulas cannot
silently drift, and verify the trace-time fallback gate.
"""

import types
from unittest import mock

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.pallas import tpu as pltpu

from tpu_inference.kernels.experimental.batched_rpa import configs, wrapper
from tpu_inference.kernels.experimental.batched_rpa.tuned_params import \
    calculate_block_sizes

# Per-TensorCore v6e values, mirroring jax._src.tpu_info for TPU_V6E.
_V6E = types.SimpleNamespace(
    generation=6,
    num_lanes=128,
    num_sublanes=8,
    mxu_column_size=256,
    smem_capacity_bytes=1024 * 1024,
    vmem_capacity_bytes=128 * 1024 * 1024,
)


@pytest.fixture(autouse=True)
def _fake_v6e():
    with mock.patch.object(pltpu, "get_tpu_info", return_value=_V6E):
        yield


def _serve_cfgs(page_size: int, pages_per_seq: int) -> configs.ServingConfigs:
    return configs.ServingConfigs(
        num_seqs=16,
        page_size=page_size,
        total_q_tokens=1024,
        num_page_indices=16 * pages_per_seq,
        dtype_q=jnp.bfloat16,
        dtype_kv=jnp.float8_e5m2,
        dtype_out=jnp.bfloat16,
    )


def _model_cfgs(num_q_heads: int, num_kv_heads: int) -> configs.ModelConfigs:
    return configs.ModelConfigs(
        num_q_heads=num_q_heads,
        num_kv_heads=num_kv_heads,
        head_dim=256,
        mask_value=-1e9,
        sliding_window=1024,
    )


def _cfgs_for(model, serve, mode):
    decode_bs, prefill_bs = calculate_block_sizes(
        model, serve, vmem_limit_bytes=_V6E.vmem_capacity_bytes)
    blocks = decode_bs if mode == configs.RpaCase.DECODE else prefill_bs
    return configs.RpaConfigs(
        block=blocks,
        model=model,
        serve=serve,
        vmem_limit_bytes=_V6E.vmem_capacity_bytes,
        mode=mode,
    )


# ---------------------------------------------------------------------------
# Incident regression lock: gemma-4-26B-A4B TP=8 shard, 16k ctx, page 16.
# Sliding-attention layers shard to num_q_heads=2, num_kv_heads=1,
# head_dim=256 at TP=8 (global layers use the hd64 kernel instead).
# ---------------------------------------------------------------------------


class TestIncidentConfigTp8:

    def _mixed(self):
        return _cfgs_for(_model_cfgs(2, 1), _serve_cfgs(16, 1024),
                         configs.RpaCase.MIXED)

    def _decode(self):
        return _cfgs_for(_model_cfgs(2, 1), _serve_cfgs(16, 1024),
                         configs.RpaCase.DECODE)

    def test_mixed_mode_arithmetic(self):
        cfgs = self._mixed()
        # Auto-tuned prefill blocks at this geometry on v6e.
        assert cfgs.block.batch_size == 2
        assert cfgs.block.bkv_sz == 2560
        assert cfgs.bkv_p == 160
        assert cfgs.bkv_p_cache == 160
        assert cfgs.bkv_p_new == 160
        # 28 + 12*160 + 16*160 = 4508 bytes per step per lane, 2 lanes.
        assert cfgs.smem_bytes_per_step == 9016
        # (16 + 17 + 16*1024 + 3 + 2 + 1) * 4
        assert cfgs.smem_fixed_bytes == 65692
        # (1MiB - 32KiB - 65692) // 9016
        assert cfgs.smem_steps_fit == 105
        assert not cfgs.fits_smem_budget

    def test_mixed_mode_old_formula_reproduces_the_oom(self):
        """The pre-fix rounding produced exactly the observed allocation."""
        cfgs = self._mixed()
        old_ub = max(1, cfgs.smem_steps_fit // _V6E.num_lanes) * _V6E.num_lanes
        assert old_ub == 128  # rounded UP past the 105-step fit
        # The s32[163840] (640KiB) scratch named in the XLA error message
        # is the dma_kv_new leaf: steps * lanes * bkv_p_new * struct_size.
        dma_kv_new_words = (old_ub * cfgs.batch_size * cfgs.bkv_p_new *
                            cfgs.dma_kv_new_size)
        assert dma_kv_new_words == 163840
        # Total schedule scratch: 1.101 MiB > the 1 MiB SMEM capacity.
        assert old_ub * cfgs.smem_bytes_per_step == 1154048

    def test_decode_mode_also_over_budget(self):
        cfgs = self._decode()
        assert cfgs.block.batch_size == 8
        assert cfgs.block.bkv_sz == 3840
        assert cfgs.smem_bytes_per_step == 23392
        assert cfgs.smem_steps_fit == 40
        assert not cfgs.fits_smem_budget

    def test_max_steps_ub_now_raises_instead_of_over_allocating(self):
        for cfgs in (self._mixed(), self._decode()):
            with pytest.raises(configs.SmemBudgetExceededError):
                _ = cfgs.max_steps_ub


# ---------------------------------------------------------------------------
# TP=4 prediction: q=4, kv=2, head_dim=256 per shard at 16k ctx / page 16.
# The decode-mode schedule is also over budget, so TP=4 would have hit the
# same compile OOM. (The mixed-mode schedule fits at TP=4.)
# ---------------------------------------------------------------------------


class TestTp4Prediction:

    def test_decode_over_budget(self):
        cfgs = _cfgs_for(_model_cfgs(4, 2), _serve_cfgs(16, 1024),
                         configs.RpaCase.DECODE)
        assert cfgs.block.bkv_sz == 3584
        assert cfgs.smem_steps_fit == 43
        assert not cfgs.fits_smem_budget

    def test_mixed_fits(self):
        cfgs = _cfgs_for(_model_cfgs(4, 2), _serve_cfgs(16, 1024),
                         configs.RpaCase.MIXED)
        assert cfgs.block.bkv_sz == 1536
        assert cfgs.smem_steps_fit == 174
        assert cfgs.fits_smem_budget


# ---------------------------------------------------------------------------
# Configs that fit must be byte-identical with the historical formula.
# ---------------------------------------------------------------------------


class TestFittingConfigsUnchanged:

    @pytest.mark.parametrize(
        "model,serve,mode",
        [
            # TP=1 full model at 16k ctx / page 16.
            (_model_cfgs(16, 8), _serve_cfgs(16, 1024), configs.RpaCase.DECODE
             ),
            (_model_cfgs(16, 8), _serve_cfgs(16, 1024), configs.RpaCase.MIXED),
            # Upstream-CI-like: page 256 (vllm-project/tpu-inference#3280
            # pins batched-RPA perf jobs to block size 256, which is why
            # upstream CI never hits the overflow).
            (_model_cfgs(2, 1), _serve_cfgs(256, 32), configs.RpaCase.DECODE),
            (_model_cfgs(2, 1), _serve_cfgs(256, 32), configs.RpaCase.MIXED),
        ],
    )
    def test_max_steps_ub_matches_old_formula(self, model, serve, mode):
        cfgs = _cfgs_for(model, serve, mode)
        assert cfgs.fits_smem_budget
        fit = cfgs.smem_steps_fit
        assert fit >= _V6E.num_lanes
        old_ub = max(1, fit // _V6E.num_lanes) * _V6E.num_lanes
        assert cfgs.max_steps_ub == old_ub
        # And the allocation honestly fits the budget it was derived from.
        assert cfgs.max_steps_ub * cfgs.smem_bytes_per_step <= (
            _V6E.smem_capacity_bytes - 32 * 1024 - cfgs.smem_fixed_bytes)


# ---------------------------------------------------------------------------
# Wrapper-level trace-time fallback gate.
# ---------------------------------------------------------------------------


def _incident_inputs():
    """Abstract inputs matching the incident config (TP=8 shard)."""
    total_tokens = 1024
    q = jax.ShapeDtypeStruct((total_tokens, 2, 256), jnp.bfloat16)
    # Real serving quantizes k/v to the KV-cache dtype before the kernel.
    k = jax.ShapeDtypeStruct((total_tokens, 1, 256), jnp.float8_e5m2)
    v = jax.ShapeDtypeStruct((total_tokens, 1, 256), jnp.float8_e5m2)
    # HEAD_ALONG_SUBLANE fp8 cache: (pages, page, align(2,4)//4, 4, 256).
    kv_cache = jax.ShapeDtypeStruct((16 * 1024, 16, 1, 4, 256),
                                    jnp.float8_e5m2)
    kv_lens = jax.ShapeDtypeStruct((16, ), jnp.int32)
    page_indices = jax.ShapeDtypeStruct((16 * 1024, ), jnp.int32)
    cu_q_lens = jax.ShapeDtypeStruct((17, ), jnp.int32)
    distribution = jax.ShapeDtypeStruct((3, ), jnp.int32)
    return (q, k, v, kv_cache, kv_lens, page_indices, cu_q_lens, distribution)


class TestWrapperFallback:

    def test_falls_back_to_v3_at_incident_config(self):
        args = _incident_inputs()

        v3_mock = mock.MagicMock(
            side_effect=lambda q, k, v, kv_cache, *a, **kw: (q, kv_cache))
        with mock.patch(
                "tpu_inference.kernels.ragged_paged_attention.v3.kernel"
                ".ragged_paged_attention", v3_mock):
            out, new_cache = jax.eval_shape(
                lambda *a: wrapper.ragged_paged_attention(a[0],
                                                          a[1],
                                                          a[2],
                                                          a[3],
                                                          a[4],
                                                          a[5],
                                                          a[6],
                                                          a[7],
                                                          sliding_window=1024),
                *args)

        assert v3_mock.call_count == 1
        kwargs = v3_mock.call_args.kwargs
        # Caller kwargs are forwarded verbatim (byte-identical with the
        # USE_BATCHED_RPA_KERNEL=0 path, which passes exactly these).
        assert kwargs["sliding_window"] == 1024
        assert kwargs["mask_value"] is None
        assert kwargs["out_dtype"] is None
        assert kwargs["vmem_limit_bytes"] is None
        assert out.shape == (1024, 2, 256)
        assert new_cache.shape == (16 * 1024, 16, 1, 4, 256)

    def test_seq_along_lane_raises_actionable_error(self):
        (q, k, v, _, kv_lens, page_indices, cu_q_lens,
         distribution) = _incident_inputs()
        # SEQ_ALONG_LANE fp8 cache: (pages, kv*2, hd/pack, pack, page).
        kv_cache = jax.ShapeDtypeStruct((16 * 1024, 2, 64, 4, 16),
                                        jnp.float8_e5m2)

        with pytest.raises(configs.SmemBudgetExceededError):
            jax.eval_shape(
                lambda *a: wrapper.ragged_paged_attention(
                    *a, kv_layout=configs.KVLayout.SEQ_ALONG_LANE), q, k, v,
                kv_cache, kv_lens, page_indices, cu_q_lens, distribution)
