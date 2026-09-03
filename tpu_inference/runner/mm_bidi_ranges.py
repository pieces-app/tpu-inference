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
"""The per-request PrefixLM blockwise-bidirectional image spans.

Lifted verbatim out of ``TPUModelRunner._prepare_inputs`` so the span
arithmetic can be exercised without a runner, a mesh or a chip: the whole
point of ``mm_bidi_ranges`` is a pair of integers per request, and until
now the only way to see those integers was to boot a TPU.

Layout contract (unchanged): the returned array is
``(max_num_reqs, 2) int32``, indexed by the SAME row as ``seq_lens``
(``dp_rank * max_num_reqs_per_dp_rank + i``), holding a half-open
``[start, end)`` token range in absolute in-sequence positions.
``(0, 0)`` means "no block" — the RPA v3 kernel reads that as an empty
range and the request keeps a purely causal mask.

This module imports numpy and nothing else on purpose: the fork's CPU gate
installs jax/flax/numpy with no vllm and no torch, and loads it by path.
"""

from typing import Any, Iterable, List, Sequence, Tuple

import numpy as np

# One format string, referenced by both the runner and its test, so the
# wording of the fallback warning cannot drift away from what is asserted.
MULTI_BLOCK_WARNING = (
    "mm-bidi: request %s has %d image blocks but the kernel operand holds "
    "one range per request — falling back to CAUSAL-ONLY attention for this "
    "request's images (degraded fine-text fidelity).")

# The live probe's line. `bidi=on` means the range below is what the
# attention kernel receives; `bidi=off` means the runner built no operand at
# all and every image token attends causally on BOTH the flax and the
# torchax path (see TPURunner._init_mm_bidi for which gate said no).
SPAN_DEBUG_PREFIX = "[mm-debug] spans="


def image_spans_for_request(mm_features: Any) -> List[Tuple[int, int]]:
    """The INCLUSIVE ``[first, last]`` span of every image block in a request.

    Mirrors vLLM's GPU runner (``gpu_model_runner.py``: ``req_doc_ranges``):
    audio features are skipped, and ``PlaceholderRange.extract_embeds_range``
    is preferred so a placeholder run with ``is_embed`` holes contributes one
    span per contiguous embedded stretch rather than one span across the
    holes.
    """
    spans: List[Tuple[int, int]] = []
    for feat in mm_features or ():
        if getattr(feat, "modality", None) == "audio":
            continue
        pos_info = feat.mm_position
        if hasattr(pos_info, "extract_embeds_range"):
            spans.extend((int(r[0]), int(r[1]))
                         for r in pos_info.extract_embeds_range())
        else:
            spans.append((int(pos_info.offset),
                          int(pos_info.offset) + int(pos_info.length) - 1))
    return spans


def format_span_line(req_id: Any, row: int, spans: Sequence[Tuple[int, int]],
                     emitted: Tuple[int, int] | None, enabled: bool) -> str:
    """One line per request for `PIECES_MM_DEBUG`; see SPAN_DEBUG_PREFIX."""
    blocks = ",".join(f"[{s},{e}]" for s, e in spans) or "-"
    if not enabled:
        carried = "none(bidi=off)"
    elif emitted is None:
        carried = "none(causal-fallback)"
    else:
        carried = f"[{emitted[0]},{emitted[1]})"
    return (f"{SPAN_DEBUG_PREFIX}{blocks} req={req_id} row={row} "
            f"blocks={len(spans)} carried={carried}")


def build_mm_bidi_ranges(
    max_num_reqs: int,
    rows: Iterable[Tuple[int, Any, Any]],
    *,
    enabled: bool,
    logger: Any = None,
    debug: bool = False,
) -> np.ndarray | None:
    """Build the ``(max_num_reqs, 2)`` operand for the scheduled requests.

    Args:
      max_num_reqs: rows in the operand (the persistent-batch capacity).
      rows: ``(row_index, req_id, mm_features)`` for each scheduled request,
        with ``row_index`` already offset by the DP rank.
      enabled: ``TPURunner.mm_bidi_enabled``. When False no operand is built
        (the attention path then sees ``None``, its documented "causal only"
        input) — but the debug lines are still emitted, which is the only way
        to tell "no images" from "images, mask inert" on a live server.
      logger: receives the multi-block fallback warning and the debug lines.
      debug: emit one `[mm-debug] spans=` line per request that carries an
        image block.
    """
    out = (np.zeros((max_num_reqs, 2), dtype=np.int32) if enabled else None)
    for row, req_id, mm_features in rows:
        spans = image_spans_for_request(mm_features)
        emitted: Tuple[int, int] | None = None
        if enabled and len(spans) == 1:
            start, last = spans[0]
            out[row, 0] = start
            out[row, 1] = last + 1
            emitted = (start, last + 1)
        elif len(spans) > 1:
            # KNOWN LIMITATION: the kernel operand carries one [start, end)
            # per request, so a multi-image (or multi-frame video) request
            # keeps causal-only attention over its image blocks and will
            # show the same fine-text degradation this feature fixes.
            # Warn per request-shape, not once globally, so it cannot hide
            # behind an earlier single-image warning.
            if enabled and logger is not None:
                logger.warning(MULTI_BLOCK_WARNING, req_id, len(spans))
        if debug and spans and logger is not None:
            logger.info("%s",
                        format_span_line(req_id, row, spans, emitted, enabled))
    return out
