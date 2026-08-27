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

from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import jax
import jax.numpy as jnp

from tpu_inference.runner.grammar_bitmask_rows import scatter_grammar_bitmask
from tpu_inference.utils import device_array

if TYPE_CHECKING:
    from vllm.v1.core.sched.output import GrammarOutput
    from vllm.v1.core.sched.output import SchedulerOutput as VllmSchedulerOutput

    from tpu_inference.runner.tpu_runner import TPUModelRunner


class StructuredDecodingManager:

    def __init__(self, runner: "TPUModelRunner"):
        self.runner = runner

    @jax.jit(static_argnums=(0, ))
    def structured_decode_fn(self, require_struct_decoding: jax.Array,
                             grammar_bitmask: jax.Array, logits: jax.Array,
                             arange: jax.Array) -> jax.Array:
        return jax.lax.cond(
            jnp.any(require_struct_decoding),
            lambda: self._apply_grammar_bitmask_kernel(
                logits, grammar_bitmask, require_struct_decoding, arange),
            lambda: logits)

    @jax.jit(static_argnums=(0, ))
    def _apply_grammar_bitmask_kernel(self, logits: jax.Array,
                                      grammar_bitmask: jax.Array,
                                      require_struct_decoding: jax.Array,
                                      arange: jax.Array) -> jax.Array:

        # Unpack the bitmask for the entire batch at once.
        # grammar_bitmask: (B, N) where B=logits.shape[0] (one row per
        # sampled logits position: num_reqs without speculative decoding,
        # sum(1 + num_draft_tokens) padded with it), N=cdiv(vocab_size, 32)
        # arange: (32,)
        # (B, N, 1) and (1, 1, 32) broadcast to (B, N, 32)
        unpacked_bitmask = jnp.right_shift(grammar_bitmask[:, :, None],
                                           arange[None, None, :]) & 1 == 0

        # Reshape to (B, vocab_size) and apply to logits.
        # (B, N * 32) -> (B, vocab_size)
        unpacked_bitmask = unpacked_bitmask.reshape(
            logits.shape[0], -1)[:, :self.runner.vocab_size]

        masked_logits = jnp.where(unpacked_bitmask, -jnp.inf, logits)

        return jnp.where(require_struct_decoding, masked_logits, logits)

    def prepare_structured_decoding_input(
        self,
        logits: jax.Array,
        grammar_output: "GrammarOutput",
        scheduler_output: Optional["VllmSchedulerOutput"] = None,
        req_ids_dp: Optional[Dict[int, List[str]]] = None,
    ) -> Tuple[jax.Array, jax.Array, jax.Array]:
        """Scatter the scheduler's grammar bitmask onto the logits rows.

        The scheduler emits ``1 + len(scheduled_spec_decode_tokens[req])``
        rows per structured request (one per verified draft position plus
        the bonus position), packed contiguously in
        ``structured_output_request_ids`` (scheduler) order. Under
        speculative decoding ``logits`` is likewise one row per sampled
        position in batch order, so each verified position must receive its
        own row -- see ``grammar_bitmask_rows.py`` for the full contract.
        Without speculative decoding every request has exactly one row and
        this reduces to the one-row-per-request mapping.
        """
        grammar_bitmask = grammar_output.grammar_bitmask
        assert grammar_bitmask is not None
        num_rows, _ = logits.shape

        if scheduler_output is not None:
            spec_tokens = scheduler_output.scheduled_spec_decode_tokens
        elif self.runner.speculative_config is not None:
            raise ValueError(
                "prepare_structured_decoding_input needs scheduler_output "
                "when speculative decoding is enabled: the grammar bitmask "
                "has (1 + num_draft_tokens) rows per request.")
        else:
            spec_tokens = None

        dp_size = self.runner.dp_size
        if dp_size > 1:
            if req_ids_dp is None:
                raise ValueError(
                    "prepare_structured_decoding_input needs req_ids_dp when "
                    "dp_size > 1: logits rows are laid out per DP rank.")
            batch_req_ids_per_rank = [
                req_ids_dp[rank] for rank in range(dp_size)
            ]
        else:
            num_reqs = self.runner.input_batch.num_reqs
            batch_req_ids_per_rank = [
                self.runner.input_batch.req_ids[:num_reqs]
            ]
        padded_rows_per_rank = num_rows // dp_size

        bitmask_cpu = self.runner.grammar_bitmask_cpu
        require_cpu = self.runner.require_structured_out_cpu
        if num_rows > bitmask_cpu.shape[0]:
            raise ValueError(
                f"logits has {num_rows} rows but grammar_bitmask_cpu was "
                f"allocated with {bitmask_cpu.shape[0]}; the buffer must "
                "cover max_num_reqs * (1 + num_speculative_tokens).")

        # Reset pre-allocated tensors
        bitmask_cpu.fill(0)
        require_cpu.fill(0)

        # It's not guaranteed that all requests (or, under speculative
        # decoding, all logits rows) require structured output, so
        # require_cpu is a bool tensor marking the rows that do.
        scatter_grammar_bitmask(
            grammar_output.structured_output_request_ids,
            grammar_bitmask,
            spec_tokens,
            batch_req_ids_per_rank,
            padded_rows_per_rank,
            bitmask_cpu,
            require_cpu,
        )

        (require_structured_out_cpu,
         grammar_bitmask_cpu, structured_decode_arange) = device_array(
             self.runner.mesh,
             (require_cpu[:num_rows], bitmask_cpu[:num_rows],
              self.runner.structured_decode_arange))

        return (require_structured_out_cpu, grammar_bitmask_cpu,
                structured_decode_arange)
