"""Differential test for the (1 + K)-rows-per-request grammar bitmask mapping.

Under speculative decoding vLLM's scheduler emits ``1 + K(req)`` bitmask rows
per structured request (``vllm/v1/structured_output/__init__.py`` @ d626108b
L281-357) and the GPU runner scatters them onto per-position logits rows
(``vllm/v1/structured_output/utils.py::apply_grammar_bitmask`` L113-141).
``tpu_inference/runner/grammar_bitmask_rows.py`` must produce the SAME
row-to-row mapping, so this test ports the upstream index arithmetic verbatim
as an oracle and compares against it on randomized batches, then proves the
K=0 layout is byte-identical to the pre-fix (one-row-per-request) algorithm.

Runs on CPU with numpy only. It deliberately does NOT import the
``tpu_inference`` package (its ``__init__`` pulls in ``vllm.logger``); the
helper module is loaded straight from its file.
"""

from __future__ import annotations

import importlib.util
import pathlib
import random

import numpy as np
import pytest

_HELPER = (pathlib.Path(__file__).resolve().parents[2] / "tpu_inference" /
           "runner" / "grammar_bitmask_rows.py")
_spec = importlib.util.spec_from_file_location("grammar_bitmask_rows", _HELPER)
rows = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(rows)

VOCAB_WORDS = 3  # cdiv(vocab, 32) stand-in; content is opaque to the mapping


# --------------------------------------------------------------------------
# Oracle 1: upstream apply_grammar_bitmask (d626108b utils.py L113-141),
# index arithmetic only. dp_size == 1 layout: logits row for batch entry b is
# b + sum_{j<b} K(j); request r in scheduler order consumes 1 + K(r) rows.
# --------------------------------------------------------------------------
def upstream_rows(batch_req_ids, struct_ids, spec_tokens, grammar_bitmask,
                  num_logits_rows):
    struct_out_req_batch_indices = {}
    cumulative_offset = 0
    struct_set = set(struct_ids)
    for batch_index, req_id in enumerate(batch_req_ids):
        logit_index = batch_index + cumulative_offset
        cumulative_offset += len(spec_tokens.get(req_id, ()))
        if req_id in struct_set:
            struct_out_req_batch_indices[req_id] = logit_index

    sorted_bitmask = np.full((num_logits_rows, grammar_bitmask.shape[1]),
                             -1,
                             dtype=grammar_bitmask.dtype)
    out_indices = []
    cumulative_index = 0
    for req_id in struct_ids:
        k = len(spec_tokens.get(req_id, ()))
        if (logit_idx := struct_out_req_batch_indices.get(req_id)) is not None:
            for i in range(1 + k):
                bitmask_index = logit_idx + i
                sorted_bitmask[bitmask_index] = grammar_bitmask[cumulative_index
                                                                + i]
                out_indices.append(bitmask_index)
        cumulative_index += 1 + k
    return sorted_bitmask, out_indices


# --------------------------------------------------------------------------
# Oracle 2: the pre-fix TPU algorithm (tpu-inference a6785016
# structured_decoding_manager.py L69-88): one row per request at batch index.
# --------------------------------------------------------------------------
def old_tpu_rows(batch_req_ids, struct_ids, grammar_bitmask, num_rows):
    req_id_to_index = {r: i for i, r in enumerate(batch_req_ids)}
    bitmask = np.zeros((num_rows, grammar_bitmask.shape[1]), dtype=np.int32)
    require = np.zeros((num_rows, 1), dtype=np.bool_)
    cumulative_mask_idx = 0
    for req_id in struct_ids:
        if req_id in req_id_to_index:
            b = req_id_to_index[req_id]
            bitmask[b] = grammar_bitmask[cumulative_mask_idx]
            require[b] = True
        cumulative_mask_idx += 1
    return bitmask, require


def _random_case(rng, allow_spec=True):
    num_reqs = rng.randint(1, 12)
    batch = [f"req-{rng.randint(0, 10_000)}-{i}" for i in range(num_reqs)]
    kmax = rng.randint(0, 4) if allow_spec else 0
    spec_tokens = {}
    for r in batch:
        if kmax and rng.random() < 0.7:
            k = rng.randint(1, kmax)
            drafts = [rng.randint(0, 99) for _ in range(k)]
            if rng.random() < 0.3:  # async-scheduling style -1 padding
                drafts = drafts + [-1] * rng.randint(1, 2)
            spec_tokens[r] = drafts
    # scheduler order = batch order (num_scheduled_tokens insertion order),
    # but structured ids are a SUBSET; also inject ids that are not in the
    # batch (finished / evicted between schedule and sample) to exercise the
    # source-cursor advance.
    struct_ids = [r for r in batch if rng.random() < 0.6]
    ghosts = []
    for _ in range(rng.randint(0, 2)):
        g = f"ghost-{rng.randint(0, 999)}"
        ghosts.append(g)
        if kmax and rng.random() < 0.5:
            spec_tokens[g] = [rng.randint(0, 99) for _ in range(rng.randint(1, kmax))]
    struct_ids = struct_ids + ghosts
    rng.shuffle(struct_ids)  # scheduler order is NOT sorted (#1563)
    total = sum(1 + len(spec_tokens.get(r, ())) for r in struct_ids)
    # Distinct payload per row so any row swap is detectable.
    grammar_bitmask = (np.arange(total * VOCAB_WORDS, dtype=np.int32).reshape(
        total, VOCAB_WORDS) + 1000 * rng.getrandbits(1))
    needed = sum(1 + len(spec_tokens.get(r, ())) for r in batch)
    padded = needed + rng.randint(0, 5)  # runner pads to a bucket
    return batch, struct_ids, spec_tokens, grammar_bitmask, needed, padded


@pytest.mark.parametrize("seed", range(300))
def test_matches_upstream_apply_grammar_bitmask(seed):
    rng = random.Random(seed)
    batch, struct_ids, spec_tokens, gb, needed, padded = _random_case(rng)
    if not struct_ids:
        pytest.skip("no structured requests drawn")

    oracle, out_indices = upstream_rows(batch, struct_ids, spec_tokens, gb,
                                        padded)

    dst_bitmask = np.zeros((padded, VOCAB_WORDS), dtype=np.int32)
    dst_require = np.zeros((padded, 1), dtype=np.bool_)
    written = rows.scatter_grammar_bitmask(struct_ids, gb, spec_tokens,
                                          [batch], padded, dst_bitmask,
                                          dst_require)

    assert sorted(written) == sorted(out_indices)
    assert len(set(written)) == len(written), "row written twice"
    for idx in out_indices:
        np.testing.assert_array_equal(dst_bitmask[idx], oracle[idx])
    assert dst_require[out_indices].all()
    untouched = np.setdiff1d(np.arange(padded), out_indices)
    assert not dst_require[untouched].any()
    assert not dst_bitmask[untouched].any()
    # padding rows past the real batch are never targeted
    assert all(i < needed for i in written)


@pytest.mark.parametrize("seed", range(100))
def test_k0_is_byte_identical_to_pre_fix_algorithm(seed):
    rng = random.Random(1_000 + seed)
    batch, struct_ids, _, gb, needed, padded = _random_case(rng,
                                                            allow_spec=False)
    if not struct_ids:
        pytest.skip("no structured requests drawn")
    assert needed == len(batch)
    old_bitmask, old_require = old_tpu_rows(batch, struct_ids, gb, padded)

    for spec_tokens in (None, {}):
        dst_bitmask = np.zeros((padded, VOCAB_WORDS), dtype=np.int32)
        dst_require = np.zeros((padded, 1), dtype=np.bool_)
        rows.scatter_grammar_bitmask(struct_ids, gb, spec_tokens, [batch],
                                    padded, dst_bitmask, dst_require)
        assert dst_bitmask.tobytes() == old_bitmask.tobytes()
        assert dst_require.tobytes() == old_require.tobytes()


def test_row_p_verifies_draft_p_and_bonus_is_last():
    # Two requests, K=2 and K=0, grammar on both, scheduler order reversed
    # relative to batch order. Request A occupies logits rows [0,3): row 0
    # verifies draft 0, row 1 verifies draft 1, row 2 is the bonus position.
    # Request B (no drafts) is at row 3.
    batch = ["A", "B"]
    struct_ids = ["B", "A"]
    spec = {"A": [7, 8]}
    gb = np.array([[100], [200], [201], [202]], dtype=np.int32)  # B, A0, A1, Abonus
    dst = np.zeros((8, 1), dtype=np.int32)
    req = np.zeros((8, 1), dtype=np.bool_)
    rows.scatter_grammar_bitmask(struct_ids, gb, spec, [batch], 8, dst, req)
    assert dst[:4, 0].tolist() == [200, 201, 202, 100]
    assert req[:4, 0].tolist() == [True] * 4 and not req[4:].any()


def test_request_without_grammar_still_offsets_by_1_plus_k():
    # Middle request has drafts but NO grammar: it must still shift the
    # following request's logits rows (mirrors utils.py L117-119).
    batch = ["A", "NOGRAMMAR", "C"]
    spec = {"A": [1], "NOGRAMMAR": [1, 2, 3], "C": [5, 6]}
    gb = np.array([[1], [2], [3], [4], [5]], dtype=np.int32)  # A: 2 rows, C: 3 rows
    dst = np.zeros((16, 1), dtype=np.int32)
    req = np.zeros((16, 1), dtype=np.bool_)
    written = rows.scatter_grammar_bitmask(["A", "C"], gb, spec, [batch], 16,
                                          dst, req)
    assert written == [0, 1, 6, 7, 8]
    assert dst[[0, 1, 6, 7, 8], 0].tolist() == [1, 2, 3, 4, 5]


def test_dp_ranks_are_offset_by_padded_rows_per_rank():
    ranks = [["A"], ["B", "C"]]
    spec = {"A": [1, 2], "C": [3]}
    gb = np.array([[10], [11], [12], [20], [30], [31]], dtype=np.int32)
    padded_per_rank = 4
    dst = np.zeros((8, 1), dtype=np.int32)
    req = np.zeros((8, 1), dtype=np.bool_)
    written = rows.scatter_grammar_bitmask(["A", "B", "C"], gb, spec, ranks,
                                          padded_per_rank, dst, req)
    assert written == [0, 1, 2, 4, 5, 6]
    assert dst[:, 0].tolist() == [10, 11, 12, 0, 20, 30, 31, 0]


def test_rank_overflow_and_short_bitmask_raise():
    with pytest.raises(ValueError):
        rows.logits_row_starts([["A", "B"]], {"A": [1, 2, 3]}, 4)
    dst = np.zeros((8, 1), dtype=np.int32)
    req = np.zeros((8, 1), dtype=np.bool_)
    with pytest.raises(ValueError):
        rows.scatter_grammar_bitmask(["A"], np.zeros((1, 1), np.int32),
                                    {"A": [1]}, [["A"]], 8, dst, req)
