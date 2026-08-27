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
"""Row mapping between the scheduler's grammar bitmask and the logits rows.

This module is deliberately vllm-free (the test loads this module by file path, because importing
    the tpu_inference package pulls in vllm.logger) (numpy only, no vllm, no jax) so
the mapping can be unit-tested on a CPU box without a vLLM install
(``tests/runner/test_guided_bitmask_specdecode_rows.py``).

Layout contract (vLLM ``vllm/v1/structured_output/__init__.py`` @ d626108b,
serial fill path L281-357, and the GPU consumer
``vllm/v1/structured_output/utils.py::apply_grammar_bitmask`` L113-141):

* ``S = grammar_output.structured_output_request_ids`` is in SCHEDULER order
  (insertion order of ``scheduler_output.num_scheduled_tokens``; never
  sorted -- see #1563). Requests without a grammar, and structured requests
  still inside a prefill chunk, are absent from ``S`` and own ZERO rows.
* ``K(rid) = len(scheduler_output.scheduled_spec_decode_tokens.get(rid, ()))``
  -- ``-1`` (padded / grammar-invalid) drafts still count.
* Source rows: request ``S[r]`` owns ``1 + K(S[r])`` contiguous rows starting
  at ``row_start(r) = sum_{j<r} (1 + K(S[j]))``. Row ``p < K`` masks the
  logits position that verifies draft ``p``; row ``p == K`` is the bonus
  position. ``grammar_bitmask.shape[0] == sum_r (1 + K(S[r]))`` (trailing
  trim only).
* Destination rows: under speculative decoding the runner's ``logits`` are
  per-POSITION rows, ``1 + K(b)`` contiguous rows per batch entry ``b`` in
  ``input_batch.req_ids`` order (``speculative_decoding_manager.py``
  ``get_spec_decode_metadata``), each DP rank padded to
  ``padded_logits_length`` rows. EVERY batch entry contributes ``1 + K`` to
  the offset, grammar or not. Without speculative decoding ``K == 0``
  everywhere and the destination row is the batch index (today's behaviour).
"""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np


def num_spec_tokens(spec_tokens: Mapping[str, Sequence[int]] | None,
                    req_id: str) -> int:
    """``K(req_id)``: scheduled draft count, ``-1`` padding included."""
    if not spec_tokens:
        return 0
    return len(spec_tokens.get(req_id, ()))


def source_row_starts(
    structured_output_request_ids: Sequence[str],
    spec_tokens: Mapping[str, Sequence[int]] | None,
) -> tuple[dict[str, int], int]:
    """Map each structured request id to its first bitmask row.

    Returns ``(row_start_by_req_id, total_rows)`` where ``total_rows`` is the
    number of rows the scheduler must have emitted (``sum(1 + K)``).
    """
    starts: dict[str, int] = {}
    cumulative = 0
    for req_id in structured_output_request_ids:
        starts[req_id] = cumulative
        cumulative += 1 + num_spec_tokens(spec_tokens, req_id)
    return starts, cumulative


def logits_row_starts(
    batch_req_ids_per_rank: Sequence[Sequence[str]],
    spec_tokens: Mapping[str, Sequence[int]] | None,
    padded_rows_per_rank: int,
) -> dict[str, int]:
    """Map each batch request id to its first logits row.

    ``batch_req_ids_per_rank[rank]`` is the runner batch order for that DP
    rank (``req_ids_dp[rank]``; for ``dp_size == 1`` this is just
    ``input_batch.req_ids[:num_reqs]``). Rank ``r`` owns logits rows
    ``[r * padded_rows_per_rank, (r + 1) * padded_rows_per_rank)``.
    """
    starts: dict[str, int] = {}
    for rank, req_ids in enumerate(batch_req_ids_per_rank):
        cumulative = rank * padded_rows_per_rank
        for req_id in req_ids:
            starts[req_id] = cumulative
            cumulative += 1 + num_spec_tokens(spec_tokens, req_id)
        if cumulative > (rank + 1) * padded_rows_per_rank:
            raise ValueError(
                f"DP rank {rank} needs {cumulative - rank * padded_rows_per_rank} "
                f"logits rows but is padded to {padded_rows_per_rank}")
    return starts


def scatter_grammar_bitmask(
    structured_output_request_ids: Sequence[str],
    grammar_bitmask: np.ndarray,
    spec_tokens: Mapping[str, Sequence[int]] | None,
    batch_req_ids_per_rank: Sequence[Sequence[str]],
    padded_rows_per_rank: int,
    dst_bitmask: np.ndarray,
    dst_require: np.ndarray,
) -> list[int]:
    """Scatter the scheduler-ordered bitmask onto logits rows.

    Mirrors ``apply_grammar_bitmask`` in upstream vLLM exactly: rows are
    consumed from ``grammar_bitmask`` in scheduler order with stride
    ``1 + K``, and written to ``dst_bitmask`` / ``dst_require`` at the
    request's logits rows. A structured request id that is not in the batch
    still advances the source cursor by ``1 + K`` (mirrors utils.py L141).

    ``dst_bitmask`` (``(rows, vocab_words)``) and ``dst_require``
    (``(rows, 1)`` bool) are written in place; callers zero them first.
    Returns the list of destination rows written (for tests / debugging).
    """
    src_starts, total_rows = source_row_starts(structured_output_request_ids,
                                               spec_tokens)
    if grammar_bitmask.shape[0] < total_rows:
        raise ValueError(
            f"grammar_bitmask has {grammar_bitmask.shape[0]} rows but the "
            f"(1 + num_spec_tokens)-per-request layout requires {total_rows} "
            f"for {len(structured_output_request_ids)} structured requests")
    dst_starts = logits_row_starts(batch_req_ids_per_rank, spec_tokens,
                                   padded_rows_per_rank)

    written: list[int] = []
    for req_id in structured_output_request_ids:
        rows = 1 + num_spec_tokens(spec_tokens, req_id)
        dst = dst_starts.get(req_id)
        if dst is None:
            continue
        src = src_starts[req_id]
        if dst + rows > dst_bitmask.shape[0]:
            raise ValueError(
                f"request {req_id!r} needs logits rows [{dst}, {dst + rows}) "
                f"but the bitmask buffer only has {dst_bitmask.shape[0]} rows")
        dst_bitmask[dst:dst + rows] = grammar_bitmask[src:src + rows]
        dst_require[dst:dst + rows] = True
        written.extend(range(dst, dst + rows))
    return written
