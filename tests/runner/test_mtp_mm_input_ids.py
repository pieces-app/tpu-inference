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
"""Regression tests for the mm + speculative-decoding input_ids nulling bug.

The multimodal path used to rebind `input_ids` to None in `_execute_model`
(via `_get_input_ids_embeds`), which killed the engine with
`assert input_ids is not None` (tpu_runner.py:2116 in the deployed wheel,
measured 2026-08-27) as soon as an image-chunk step coincided with scheduled
draft tokens. The fix binds the model-forward view to `model_input_ids` and
keeps the raw token ids live for spec decode.
"""

from unittest.mock import MagicMock, patch

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from vllm.config import (CacheConfig, ModelConfig, ParallelConfig,
                         SchedulerConfig, SpeculativeConfig, VllmConfig)

from tpu_inference.layers.common.sharding import ShardingAxisName
from tpu_inference.runner.tpu_runner import TPUModelRunner


class TestMtpMmInputIdsPreserved:

    def setup_method(self):
        # Mock JAX dependencies
        self.mock_devices = [MagicMock(coords=i) for i in range(1)]
        self.mock_rng_key = MagicMock()
        device_array = np.array(jax.devices()[:1]).reshape(1, 1, 1, -1)
        self.mock_mesh = jax.make_mesh(device_array.shape,
                                       ('data', 'attn_dp', 'expert', 'model'))
        with patch('jax.devices', return_value=self.mock_devices), \
             patch('jax.make_mesh', return_value=self.mock_mesh), \
             patch('jax.random.key', return_value=self.mock_rng_key), \
             patch('tpu_inference.runner.tpu_runner.get_model', return_value=MagicMock()), \
             patch('tpu_inference.runner.tpu_runner.make_optimized_mesh', return_value=self.mock_mesh):

            model_config = ModelConfig(tokenizer_mode="auto",
                                       trust_remote_code=False,
                                       seed=0,
                                       dtype='bfloat16')
            cache_config = CacheConfig(
                block_size=16,
                gpu_memory_utilization=0.9,
                cache_dtype="auto",
            )
            scheduler_config = SchedulerConfig(max_num_seqs=16,
                                               max_model_len=1024,
                                               is_encoder_decoder=False)
            parallel_config = ParallelConfig(
                pipeline_parallel_size=1,
                tensor_parallel_size=1,
            )
            speculative_config = SpeculativeConfig(
                model='ngram',
                num_speculative_tokens=5,
                prompt_lookup_max=4,
            )
            vllm_config = VllmConfig(
                model_config=model_config,
                cache_config=cache_config,
                scheduler_config=scheduler_config,
                parallel_config=parallel_config,
                speculative_config=speculative_config,
                observability_config={},
                additional_config={},
            )

            self.runner = TPUModelRunner(vllm_config,
                                         devices=self.mock_devices)

    def _run_execute_model(self, is_multimodal: bool):
        """Drives _execute_model with a mocked forward pass and returns
        (raw_input_ids, embeds, model_fn_mock)."""
        runner = self.runner
        num_tokens = 8
        hidden_size = 4

        raw_input_ids = jnp.arange(num_tokens, dtype=jnp.int32) + 100
        embeds = jnp.ones((num_tokens, hidden_size), dtype=jnp.float32)

        prepared = (
            raw_input_ids,  # input_ids
            jnp.zeros(num_tokens, dtype=jnp.int32),  # input_positions
            MagicMock(),  # attn_metadata
            MagicMock(),  # sampling_metadata
            jnp.arange(2, dtype=jnp.int32),  # logits_indices
            MagicMock(),  # spec_decode_metadata (draft tokens scheduled)
            None,  # logits_indices_selector
            8,  # padded_num_reqs
            {
                0: []
            },  # req_ids_dp
            0,  # padded_num_scheduled_tokens_per_dp_rank
            None,  # tokens_indices_selector
            None,  # shared_attn_metadata
        )

        scheduler_output = MagicMock()
        scheduler_output.total_num_scheduled_tokens = num_tokens

        runner.persistent_batch_manager = MagicMock()
        runner.get_mrope_input_positions_fn = MagicMock()
        runner.enable_continue_decode = False
        runner.is_multimodal_model = is_multimodal

        # Multimodal machinery: image tokens scheduled in this step.
        runner.mm_manager = MagicMock()
        runner.mm_manager.gather_mm_embeddings.return_value = (
            [jnp.ones((2, hidden_size), dtype=jnp.float32)],
            jnp.zeros(num_tokens, dtype=jnp.bool_),
        )

        # The embed merge is a pure elementwise substitution: same length,
        # positions aligned with raw_input_ids.
        runner.embed_input_ids_fn = MagicMock(return_value=embeds)
        runner.state_leaves = MagicMock()
        runner.kv_caches = MagicMock()
        runner.lora_utils = MagicMock()

        model_fn = MagicMock(return_value=(MagicMock(),
                                           jnp.zeros((num_tokens,
                                                      hidden_size)), None,
                                           None))
        runner.model_fn = model_fn
        runner.compute_logits_fn = MagicMock(return_value=jnp.zeros((2, 16)))
        # Shadow the jitted staticmethod with a passthrough.
        runner._select_from_array_fn = MagicMock(
            side_effect=lambda array, idx, mesh, pcp=1: array)

        with patch.object(runner, '_prepare_inputs', return_value=prepared):
            result = runner._execute_model(scheduler_output)

        assert result is None  # state stashed for sample_tokens
        return raw_input_ids, embeds, model_fn

    def test_execute_model_preserves_raw_input_ids_for_mm(self):
        """mm batch: the model forward gets the embeds AND the ids, and the
        raw token ids MUST survive into execute_model_state so spec decode
        can score scheduled draft tokens. On the unfixed tree the state held
        None and the engine died at the assert."""
        raw_input_ids, embeds, model_fn = self._run_execute_model(
            is_multimodal=True)

        # (a) mm forward contract: both operands. The ids arg was None here
        # until the PLE id-track fix -- the model needs the token ids to
        # build the per-layer embedding track on an image step, and the two
        # operands together are one static jit signature (the "backbone with
        # embeds" primer builds the same pair).
        model_fn.assert_called_once()
        call_args = model_fn.call_args[0]
        assert call_args[2] is raw_input_ids  # model_input_ids
        assert call_args[4] is embeds  # inputs_embeds

        # (b) the raw token ids are preserved for spec decode. This is the
        # regression assertion: it fails (state input_ids is None) without
        # the model_input_ids fix.
        assert self.runner.execute_model_state is not None
        assert self.runner.execute_model_state.input_ids is raw_input_ids

    def test_execute_model_text_only_path_unchanged(self):
        """Positive control: text-only spec decode is bit-for-bit untouched --
        the forward still receives the raw ids and no embeds."""
        raw_input_ids, _, model_fn = self._run_execute_model(
            is_multimodal=False)

        model_fn.assert_called_once()
        call_args = model_fn.call_args[0]
        assert call_args[2] is raw_input_ids  # model_input_ids == raw ids
        assert call_args[4] is None  # inputs_embeds

        assert self.runner.execute_model_state is not None
        assert self.runner.execute_model_state.input_ids is raw_input_ids

    def test_extract_draft_token_ids_cpu(self):
        """The scoring math the fix re-enables runs on the preserved array
        (worked example from speculative_decoding_manager.get_spec_decode_metadata)."""
        rng = np.random.default_rng(0)
        input_ids_np = rng.integers(0, 32000, size=209).astype(np.int32)
        # cu_num_scheduled_tokens [4, 104, 107, 207, 209],
        # num_draft_tokens        [3,   0,   2,   0,   1]:
        final_logits_indices = np.array(
            [0, 1, 2, 3, 103, 104, 105, 106, 206, 207, 208], dtype=np.int32)
        target_logits_indices = np.array([0, 1, 2, 5, 6, 9], dtype=np.int32)

        expected = input_ids_np[final_logits_indices][target_logits_indices +
                                                      1]

        sharding = jax.sharding.NamedSharding(
            self.runner.mesh,
            jax.sharding.PartitionSpec(ShardingAxisName.ATTN_DATA))
        got = self.runner._extract_draft_token_ids(
            jax.device_put(jnp.asarray(input_ids_np), sharding),
            jax.device_put(jnp.asarray(final_logits_indices), sharding),
            jax.device_put(jnp.asarray(target_logits_indices), sharding))

        np.testing.assert_array_equal(np.asarray(got), expected)

    def _call_sample_from_logits(self, input_ids):
        """Calls _sample_from_logits on the spec-decode branch with the heavy
        internals mocked; returns the _extract_draft_token_ids spy."""
        runner = self.runner
        vocab = 16

        tpu_sampling_metadata = MagicMock()
        tpu_sampling_metadata.do_sampling = False
        tpu_sampling_metadata.logprobs = False

        spec_decode_metadata = MagicMock()
        logits = jnp.zeros((4, vocab), dtype=jnp.bfloat16)

        runner.rng_params_for_sampling = jax.random.key(0)
        runner._select_from_array_fn = MagicMock(
            side_effect=lambda array, idx, mesh, pcp=1: array)
        runner.rejection_sampler = MagicMock(
            return_value=jnp.zeros((2, 6), dtype=jnp.int32))

        extract_spy = MagicMock(return_value=jnp.zeros(6, dtype=jnp.int32))

        with patch.object(runner, '_extract_draft_token_ids', extract_spy), \
             patch('tpu_inference.runner.tpu_runner.sample',
                   return_value=(jnp.zeros((2, 1), dtype=jnp.int32), None)), \
             patch('tpu_inference.runner.tpu_runner.compute_prompt_logprobs',
                   return_value=None), \
             patch('tpu_inference.runner.tpu_runner.extract_last_sampled_tokens',
                   return_value=(MagicMock(), MagicMock())), \
             patch.object(runner.speculative_decoding_manager,
                          'propose_draft_token_ids'), \
             patch('tpu_inference.runner.tpu_runner.runner_utils.host_extract_sampled_tokens',
                   return_value=[]):
            runner._sample_from_logits(
                scheduler_output=MagicMock(),
                attn_metadata=MagicMock(),
                tpu_sampling_metadata=tpu_sampling_metadata,
                input_ids=input_ids,
                hidden_states=jnp.zeros((4, 4)),
                logits=logits,
                aux_hidden_states=None,
                spec_decode_metadata=spec_decode_metadata,
                kv_connector_output=None,
                logits_indices_selector=None,
                padded_num_reqs=8,
            )
        return extract_spy

    def test_sample_from_logits_spec_path_uses_raw_ids(self):
        """With the raw ids preserved, the spec-decode scoring branch runs
        (no engine death) and extracts draft ids from exactly that array."""
        raw_input_ids = jnp.arange(8, dtype=jnp.int32)

        extract_spy = self._call_sample_from_logits(raw_input_ids)

        extract_spy.assert_called_once()
        assert extract_spy.call_args[0][0] is raw_input_ids

    def test_sample_from_logits_null_input_ids_diagnosed(self):
        """If a future regression nulls input_ids again, the guard must die
        with an explicit diagnosis (RuntimeError), not a bare assert that
        vanishes under python -O."""
        with pytest.raises(RuntimeError,
                           match="multimodal path must not null"):
            self._call_sample_from_logits(None)
