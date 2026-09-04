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
"""Async token substitution must leave a resumed request's real tokens alone.

Live failure (lane eval-26b-1chip-mtp, Gemma-4 26B-A4B + its MTP drafter,
one v6e chip, ``--max-model-len 6144 --max-num-seqs 8``, 2026-09-04
11:17:06Z, kv_cache_usage 92-100 %, 15 preemptions in 20 s)::

    tpu_runner.py:2660 _prepare_async_token_substitution_indices
        assert num_scheduled_tokens_per_req[i] <= max_num_spec_tokens + 1
    AssertionError

The dumped SchedulerOutput (pod-previous.log line 1, ``dump_input.py:79``)
had four cached requests and no resumed ones: three decodes at 4 scheduled
tokens each (1 + three ``-1`` draft placeholders) and
``chatcmpl-9be5c9c61c40f679-961d365f`` at 60 scheduled tokens with NO draft
entry, ``num_computed_tokens=3007``, ``num_output_tokens=115``: a request
preempted earlier, resumed, past the end of its prompt, and recomputing its
own already-generated output tokens in chunks.

Root cause, two halves of one bookkeeping gap:

* ``_update_placeholder`` under spec decode rebuilt
  ``placeholder_req_id_to_index`` from ``spec_decode_metadata.req_ids_dp``,
  i.e. EVERY request in the batch, including rows whose sampled token had
  been discarded because they had not reached their frontier (a prefill
  chunk, or a recompute chunk). Those rows got no placeholder tokens
  appended, but the map said they had.
* ``_prepare_async_token_substitution_indices`` excused a mapped request
  only while ``num_computed_tokens < num_prompt_tokens``. Past the prompt a
  recompute chunk is real token ids in ``token_ids_cpu``, not placeholders,
  and its length is the chunk size, so the ``<= K + 1`` assertion fired. A
  chunk of ``<= K + 1`` tokens would not have asserted: it would have been
  silently overwritten with the previous step's sampled/draft tokens for
  that slot.

The fix: the map carries only the requests that actually received
placeholders; the substitution and the rejection-count subtraction skip a
request the scheduler resumed in the current step (``update_states``
rebuilt its row from real token ids); ``_modify_prev_results`` skips it
too; and a request re-added to the persistent batch takes its output
history from the scheduler's ``all_token_ids`` (the sampled tokens of the
step in flight at preemption were delivered by the scheduler but never
written by ``_modify_prev_results``, which skips a request that has left
the batch, so the placeholder zeros would otherwise be recomputed as
tokens).

CPU-only: numpy + jax[cpu]. The runner methods are compiled from
``tpu_runner.py`` by source (``ast``) and the batch manager is loaded by
file behind stand-in modules; needs neither vllm, torch nor a TPU.

NEGATIVE CONTROLS on the pre-fix tree (``git stash`` of the two source
files with this test kept), see the PR body for the run:
  * ``test_dumped_batch_no_longer_trips_the_assert`` -> AssertionError at
    the live line (the crash reproduced);
  * ``test_map_carries_only_rows_that_received_placeholders`` -> the map
    holds the recompute row;
  * ``test_over_inclusive_map_is_a_named_error_not_a_bare_assert`` -> the
    bare AssertionError at the same line;
  * ``test_update_states_rebuilds_a_resumed_history_from_the_scheduler``
    -> ``[101, 102, 103, 0, 0]``: the placeholder zeros survive the resume;
  * the ``resumed_req_ids`` tests -> TypeError (the parameter did not
    exist); ``_resumed_req_ids`` -> not defined;
  * ``test_normal_decode_path_is_unchanged`` and
    ``test_update_states_without_all_token_ids_keeps_the_old_truncation``
    pass on both trees: that is their point.
"""

from __future__ import annotations

import ast
import contextlib
import dataclasses
import importlib.util
import logging
import pathlib
import sys
import types
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, cast

import jax
import jax.numpy as jnp
import numpy as np
import pytest

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "tpu_inference" / "runner" / "tpu_runner.py"
_PBM = _ROOT / "tpu_inference" / "runner" / "persistent_batch_manager.py"

# The batch the engine died on: pod-previous.log line 1. List order is the
# persistent-batch order used below; the dump's dict order differs.
DUMP_REQ_IDS = [
    "chatcmpl-858b14c7db0e9fcc-9928d9f2",
    "chatcmpl-903cf1ad6c2a58e9-b4e9bb11",
    "chatcmpl-94e04bcc492c2eef-a0ed6ff4",
    "chatcmpl-9be5c9c61c40f679-961d365f",
]
RECOMPUTE = DUMP_REQ_IDS[3]
DUMP_NUM_COMPUTED = [3020, 3020, 3321, 3007]
DUMP_NUM_OUTPUT = [75, 75, 376, 115]
DUMP_NUM_SCHEDULED = [4, 4, 4, 60]
DUMP_SPEC_TOKENS = {req_id: [-1, -1, -1] for req_id in DUMP_REQ_IDS[:3]}
DUMP_NUM_SPEC_TO_SCHEDULE = 3
MAX_MODEL_LEN = 6144
MAX_NUM_SEQS = 8
# The prompt of the recompute request is not in the dump; anything
# <= num_computed_tokens (3007) puts it past its prompt, which is the case
# the old `is_prefill` guard did not cover.
RECOMPUTE_NUM_PROMPT = 3007 - 55


# --------------------------------------------------------------------------
# Loading the code under test
# --------------------------------------------------------------------------
@dataclasses.dataclass
class _SpecDecodeMetadata:
    req_ids_dp: dict


def _runner_defs(names):
    tree = ast.parse(_RUNNER.read_text(), filename=str(_RUNNER))
    found = {}
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == "TPUModelRunner":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name in names:
                    found[item.name] = item
        elif isinstance(node, ast.FunctionDef) and node.name in names:
            found[node.name] = node
    missing = [n for n in names if n not in found]
    assert not missing, f"not defined in tpu_runner.py: {missing}"
    return [found[n] for n in names]


def _load_runner_methods(*names, **extra_globals):
    """Compile the named ``TPUModelRunner`` methods / module functions of
    ``tpu_runner.py`` as plain functions. The file name is kept so a failure
    points at the real line."""
    module = ast.Module(body=_runner_defs(names), type_ignores=[])
    ast.fix_missing_locations(module)
    ns: dict = {
        "np": np,
        "jax": jax,
        "jnp": jnp,
        "Any": Any,
        "Optional": Optional,
        "List": List,
        "Dict": Dict,
        "cast": cast,
        "SpecDecodeMetadata": _SpecDecodeMetadata,
    }
    ns.update(extra_globals)
    exec(compile(module, str(_RUNNER), "exec"), ns)
    return tuple(ns[n] for n in names)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _load_pbm():
    """``PersistentBatchManager`` from the installed package when it imports
    (the in-image gate has vllm), else from the file behind stand-ins that are
    removed again afterwards."""
    try:
        from tpu_inference.runner.persistent_batch_manager import \
            PersistentBatchManager
        return PersistentBatchManager
    except ImportError:
        pass

    class _Cls:
        pass

    stubs = {
        "vllm":
        _module("vllm"),
        "vllm.v1":
        _module("vllm.v1"),
        "vllm.v1.core":
        _module("vllm.v1.core"),
        "vllm.v1.core.sched":
        _module("vllm.v1.core.sched"),
        "vllm.v1.core.sched.output":
        _module("vllm.v1.core.sched.output", SchedulerOutput=_Cls),
        "tpu_inference":
        _module("tpu_inference"),
        "tpu_inference.logger":
        _module("tpu_inference.logger", init_logger=logging.getLogger),
        "tpu_inference.runner":
        _module("tpu_inference.runner"),
        "tpu_inference.runner.input_batch":
        _module("tpu_inference.runner.input_batch",
                CachedRequestState=_Cls,
                InputBatch=_Cls),
    }
    for name, mod in stubs.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(stubs[parent], child, mod)
    injected = [n for n in stubs if n not in sys.modules]
    sys.modules.update({n: stubs[n] for n in injected})
    try:
        spec = importlib.util.spec_from_file_location("_pbm_under_test", _PBM)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        for n in injected:
            sys.modules.pop(n, None)
    return mod.PersistentBatchManager


# --------------------------------------------------------------------------
# The slice of the runner the methods read
# --------------------------------------------------------------------------
class _Batch:
    """The ``InputBatch`` fields the three runner methods read."""

    def __init__(self,
                 req_ids,
                 *,
                 num_computed,
                 num_prompt,
                 num_output,
                 max_num_reqs=MAX_NUM_SEQS):
        n = len(req_ids)
        self.req_ids = list(req_ids)
        self.req_id_to_index = {r: i for i, r in enumerate(req_ids)}
        self.num_computed_tokens_cpu = np.zeros(max_num_reqs, np.int32)
        self.num_prompt_tokens = np.zeros(max_num_reqs, np.int32)
        self.num_tokens_no_spec = np.zeros(max_num_reqs, np.int32)
        self.num_tokens = np.zeros(max_num_reqs, np.int32)
        self.token_ids_cpu = np.zeros((max_num_reqs, MAX_MODEL_LEN), np.int32)
        self.num_computed_tokens_cpu[:n] = num_computed
        self.num_prompt_tokens[:n] = num_prompt
        self.num_tokens_no_spec[:n] = np.add(num_prompt, num_output)
        self.num_tokens[:n] = self.num_tokens_no_spec[:n]


def _runner(batch,
            *,
            num_spec_tokens,
            placeholder_map=None,
            dp_size=1,
            requests=None):
    return SimpleNamespace(
        input_batch=batch,
        dp_size=dp_size,
        max_num_reqs=MAX_NUM_SEQS,
        max_model_len=MAX_MODEL_LEN,
        speculative_config=(None
                            if num_spec_tokens is None else SimpleNamespace(
                                num_speculative_tokens=num_spec_tokens)),
        _pre_async_results=SimpleNamespace(
            placeholder_req_id_to_index=dict(placeholder_map or {})),
        requests=requests or {},
        mesh=None,
        maybe_forbid_compile=contextlib.nullcontext(),
    )


def _state(req_id, num_output):
    return SimpleNamespace(req_id=req_id,
                           output_token_ids=list(range(1, num_output + 1)))


def _previous_step_map(update_placeholder, *, num_spec_tokens):
    """Replay the step BEFORE the crash: three decodes at their frontier,
    the recompute row discarded (its chunk ended short of its num_tokens).
    Returns (runner, map, states)."""
    batch = _Batch(DUMP_REQ_IDS,
                   num_computed=[3016, 3016, 3317, 2960],
                   num_prompt=[2945, 2945, 2945, RECOMPUTE_NUM_PROMPT],
                   num_output=[71, 71, 372, 115])
    states = {
        r: _state(r, n)
        for r, n in zip(DUMP_REQ_IDS, [71, 71, 372, 115])
    }
    runner = _runner(batch, num_spec_tokens=num_spec_tokens)
    request_seq_lens = [(i, states[r], int(batch.num_tokens_no_spec[i]))
                        for i, r in enumerate(DUMP_REQ_IDS[:3])]
    placeholder_map = update_placeholder(
        runner,
        [3],  # discard_sampled_tokens_req_indices: the recompute row
        request_seq_lens,
        SimpleNamespace(scheduled_spec_decode_tokens=DUMP_SPEC_TOKENS),
        None,
        _SpecDecodeMetadata(req_ids_dp={0: list(DUMP_REQ_IDS)}),
    )
    return runner, placeholder_map, states


def _pre_fix_substitution(runner, req_ids_dp, scheduled_tokens_per_dp_rank,
                          padded_per_rank, dp_size):
    """The pre-fix algorithm (pieces/main dea0d018, tpu_runner.py
    L2606-2672), as the oracle for the path the fix must not change."""
    cur_dp, pre_dp = {}, {}
    spec = runner.speculative_config is not None
    for rank in range(dp_size):
        cur_dp[rank], pre_dp[rank] = [], []
        acc = padded_per_rank * rank
        for i, req_id in enumerate(req_ids_dp[rank]):
            n = scheduled_tokens_per_dp_rank[rank][i]
            acc += n
            idx = runner.input_batch.req_id_to_index[req_id]
            if (runner.input_batch.num_computed_tokens_cpu[idx]
                    < runner.input_batch.num_prompt_tokens[idx]):
                continue
            pmap = runner._pre_async_results.placeholder_req_id_to_index
            if req_id not in pmap:
                continue
            if not spec:
                assert n == 1
                cur_dp[rank].append(acc - 1)
                pre_dp[rank].append(pmap[req_id])
            else:
                k = runner.speculative_config.num_speculative_tokens
                assert n <= k + 1
                for j in range(n):
                    cur_dp[rank].append(acc - n + j)
                    pre_dp[rank].append(pmap[req_id] * (k + 1) + j)
    return cur_dp, pre_dp


# --------------------------------------------------------------------------
# 1. The dumped batch
# --------------------------------------------------------------------------
@pytest.mark.parametrize("num_spec_tokens", [DUMP_NUM_SPEC_TO_SCHEDULE, 4])
def test_map_carries_only_rows_that_received_placeholders(num_spec_tokens):
    (update_placeholder, ) = _load_runner_methods("_update_placeholder")
    runner, placeholder_map, states = _previous_step_map(
        update_placeholder, num_spec_tokens=num_spec_tokens)

    assert placeholder_map == {r: i for i, r in enumerate(DUMP_REQ_IDS[:3])}
    assert RECOMPUTE not in placeholder_map
    # The three decodes got 1 + 3 placeholder tokens; the recompute row got
    # none, and its history was not touched.
    for i in range(3):
        assert runner.input_batch.num_tokens_no_spec[i] == (
            runner.input_batch.num_prompt_tokens[i] + 71 +
            4 if i < 2 else runner.input_batch.num_prompt_tokens[i] + 372 + 4)
        assert states[DUMP_REQ_IDS[i]].output_token_ids[-4:] == [0, 0, 0, 0]
    assert runner.input_batch.num_tokens_no_spec[3] == (RECOMPUTE_NUM_PROMPT +
                                                        115)
    assert states[RECOMPUTE].output_token_ids == list(range(1, 116))


@pytest.mark.parametrize("num_spec_tokens", [DUMP_NUM_SPEC_TO_SCHEDULE, 4])
def test_dumped_batch_no_longer_trips_the_assert(num_spec_tokens):
    """The exact step that killed the engine, driven through the real map
    construction of the previous step and the real index derivation."""
    update_placeholder, prepare = _load_runner_methods(
        "_update_placeholder", "_prepare_async_token_substitution_indices")
    _, placeholder_map, _ = _previous_step_map(update_placeholder,
                                               num_spec_tokens=num_spec_tokens)

    batch = _Batch(DUMP_REQ_IDS,
                   num_computed=DUMP_NUM_COMPUTED,
                   num_prompt=[2945, 2945, 2945, RECOMPUTE_NUM_PROMPT],
                   num_output=DUMP_NUM_OUTPUT)
    runner = _runner(batch,
                     num_spec_tokens=num_spec_tokens,
                     placeholder_map=placeholder_map)

    # Positional call, as `_prepare_inputs` did before the fix: on the
    # pre-fix tree this is the AssertionError from the log.
    cur, pre = prepare(runner, {0: list(DUMP_REQ_IDS)},
                       {0: list(DUMP_NUM_SCHEDULED)}, 128, 1)

    k1 = num_spec_tokens + 1
    assert cur == {0: list(range(12))}
    assert pre == {0: [i * k1 + j for i in range(3) for j in range(4)]}


def test_over_inclusive_map_is_a_named_error_not_a_bare_assert():
    """With the recompute row wrongly mapped (the pre-fix map), the derivation
    refuses with the request id and the counts instead of a bare assert."""
    (prepare,
     ) = _load_runner_methods("_prepare_async_token_substitution_indices")
    batch = _Batch(DUMP_REQ_IDS,
                   num_computed=DUMP_NUM_COMPUTED,
                   num_prompt=[2945, 2945, 2945, RECOMPUTE_NUM_PROMPT],
                   num_output=DUMP_NUM_OUTPUT)
    runner = _runner(batch,
                     num_spec_tokens=3,
                     placeholder_map={
                         r: i
                         for i, r in enumerate(DUMP_REQ_IDS)
                     })
    with pytest.raises(RuntimeError) as excinfo:
        prepare(runner, {0: list(DUMP_REQ_IDS)}, {0: list(DUMP_NUM_SCHEDULED)},
                128, 1)
    message = str(excinfo.value)
    assert RECOMPUTE in message
    assert "60 tokens" in message
    assert "num_computed_tokens=3007" in message


# --------------------------------------------------------------------------
# 2. Resumed in this step: the row is real tokens whatever its length
# --------------------------------------------------------------------------
@pytest.mark.parametrize("num_spec_tokens", [None, 3])
def test_same_step_resume_is_not_substituted(num_spec_tokens):
    """R reached its frontier last step (mapped), is resumed this step with a
    full prefix-cache hit (num_computed = num_tokens - 1 >= num_prompt) and
    exactly 1 scheduled token: a slice the old guards could not tell from a
    decode. A is a normal decode next to it."""
    (prepare,
     ) = _load_runner_methods("_prepare_async_token_substitution_indices")
    k_t = 0 if num_spec_tokens is None else 3
    batch = _Batch(["A", "R"],
                   num_computed=[100 + 20, 100 + 30 - 1],
                   num_prompt=[100, 100],
                   num_output=[20 + 1 + k_t, 30])
    runner = _runner(batch,
                     num_spec_tokens=num_spec_tokens,
                     placeholder_map={
                         "A": 0,
                         "R": 1
                     })

    cur, pre = prepare(runner, {0: ["A", "R"]}, {0: [1 + k_t, 1]},
                       64,
                       1,
                       resumed_req_ids=frozenset({"R"}))

    if num_spec_tokens is None:
        assert cur == {0: [0]} and pre == {0: [0]}
    else:
        assert cur == {0: [0, 1, 2, 3]}
        assert pre == {0: [0, 1, 2, 3]}  # slot 0 * (K + 1) + j


def test_rejection_subtraction_skips_a_resumed_row():
    subtract, = _load_runner_methods(
        "_subtract_num_rejected_tokens",
        device_array=lambda mesh, arrays: jax.tree.map(jnp.asarray, arrays),
    )
    (fn, ) = _load_runner_methods("_subtract_num_rejected_tokens_fn")
    batch = _Batch(["A", "R"],
                   num_computed=[124, 129],
                   num_prompt=[100, 100],
                   num_output=[24, 30])
    runner = _runner(batch,
                     num_spec_tokens=3,
                     placeholder_map={
                         "A": 0,
                         "R": 1
                     })
    num_rejected = np.zeros(MAX_NUM_SEQS, np.int32)
    num_rejected[:2] = [2, 1]
    runner._pre_async_results.spec_decode_num_rejected_tokens = jnp.asarray(
        num_rejected)
    seq_lens = jnp.asarray(np.array([124, 129] + [0] * 6, np.int32))
    positions = jnp.asarray(np.arange(16, dtype=np.int32))
    # Bind the module-level jitted kernel the method calls.
    subtract.__globals__["_subtract_num_rejected_tokens_fn"] = fn

    out_seq_lens, out_positions = subtract(
        runner,
        seq_lens,
        positions,
        {0: ["A", "R"]},
        {0: [4, 1]},
        resumed_req_ids=frozenset({"R"}),
    )

    out_seq_lens = np.asarray(out_seq_lens)
    out_positions = np.asarray(out_positions)
    assert out_seq_lens[0] == 124 - 2  # A: its 2 rejections applied
    assert out_seq_lens[1] == 129  # R: untouched
    assert list(out_positions[:4]) == [0 - 2, 1 - 2, 2 - 2, 3 - 2]
    assert out_positions[4] == 4  # R's single position untouched
    assert list(out_positions[5:]) == list(range(5, 16))


def test_modify_prev_results_skips_a_same_step_resumed_row():
    a_state = SimpleNamespace(req_id="A",
                              output_token_ids=[101, 102, 0, 0, 0, 0])
    r_state = SimpleNamespace(req_id="R", output_token_ids=[201, 202, 203])
    sampled = [[7, 8], [9]]  # what the previous step sampled for A and R
    runner_utils = SimpleNamespace(
        host_extract_sampled_tokens=lambda *args: [list(x) for x in sampled])
    (modify, ) = _load_runner_methods("_modify_prev_results",
                                      runner_utils=runner_utils)

    batch = _Batch(["A", "R"],
                   num_computed=[12, 0],
                   num_prompt=[10, 10],
                   num_output=[6, 3])
    batch.token_ids_cpu[0, :16] = list(range(1, 11)) + a_state.output_token_ids
    batch.token_ids_cpu[1, :13] = list(range(1, 11)) + r_state.output_token_ids
    runner = _runner(batch,
                     num_spec_tokens=3,
                     requests={
                         "A": a_state,
                         "R": r_state
                     })
    runner._pre_async_results = SimpleNamespace(
        req_ids=["A", "R"],
        next_tokens=None,
        request_seq_lens=[(0, a_state, 16), (1, r_state, 13)],
        discard_sampled_tokens_req_indices=[],
        logits_indices_selector=None,
        spec_decode_metadata=None,
        scheduler_output=SimpleNamespace(scheduled_spec_decode_tokens={
            "A": [-1, -1, -1],
            "R": [-1, -1, -1]
        }),
        is_continue_decode=False,
    )

    modify(runner, frozenset({"R"}))

    # A: the 4 placeholders became the 2 sampled tokens.
    assert a_state.output_token_ids == [101, 102, 7, 8]
    assert list(batch.token_ids_cpu[0, 10:14]) == [101, 102, 7, 8]
    assert batch.num_tokens_no_spec[0] == 14 and batch.num_tokens[0] == 14
    # R: rebuilt by the same-step resume, nothing written over it.
    assert r_state.output_token_ids == [201, 202, 203]
    assert list(batch.token_ids_cpu[1, 10:13]) == [201, 202, 203]
    assert batch.num_tokens_no_spec[1] == 13 and batch.num_tokens[1] == 13


# --------------------------------------------------------------------------
# 3. The path that must not change
# --------------------------------------------------------------------------
@pytest.mark.parametrize("dp_size", [1, 2])
@pytest.mark.parametrize("num_spec_tokens", [None, 3])
def test_normal_decode_path_is_unchanged(dp_size, num_spec_tokens):
    """All-decode batches (full K, a max_model_len-shortened draft list, a
    row that is not mapped, a prefill chunk) give the pre-fix indices."""
    (prepare,
     ) = _load_runner_methods("_prepare_async_token_substitution_indices")
    req_ids = ["A", "B", "C", "D"]
    if num_spec_tokens is None:
        scheduled = [1, 1, 1, 50]
    else:
        scheduled = [4, 3, 4, 50]  # D is a prefill chunk
    batch = _Batch(req_ids,
                   num_computed=[120, 130, 140, 20],
                   num_prompt=[100, 100, 100, 100],
                   num_output=[24, 34, 44, 0])
    per_rank = MAX_NUM_SEQS // dp_size
    if dp_size == 1:
        req_ids_dp = {0: req_ids}
        sched_dp = {0: scheduled}
        pmap = {"A": 0, "B": 1, "D": 3}  # C is not mapped
    else:
        req_ids_dp = {0: ["A", "C"], 1: ["B", "D"]}
        sched_dp = {
            0: [scheduled[0], scheduled[2]],
            1: [scheduled[1], scheduled[3]]
        }
        pmap = {"A": 0, "B": per_rank + 0, "D": per_rank + 1}
    runner = _runner(batch,
                     num_spec_tokens=num_spec_tokens,
                     placeholder_map=pmap,
                     dp_size=dp_size)

    # Positional call: this test must pass on the pre-fix tree as well.
    got = prepare(runner, req_ids_dp, sched_dp, 64, dp_size)
    want = _pre_fix_substitution(runner, req_ids_dp, sched_dp, 64, dp_size)
    assert got == want
    # And something was actually substituted.
    assert sum(len(v) for v in got[0].values()) > 0


def test_map_keeps_the_per_rank_index_for_kept_rows():
    (update_placeholder, ) = _load_runner_methods("_update_placeholder")
    batch = _Batch(["A", "X", "B"],
                   num_computed=[120, 20, 130],
                   num_prompt=[100, 100, 100],
                   num_output=[20, 0, 30])
    runner = _runner(batch, num_spec_tokens=3, dp_size=2)
    states = {r: _state(r, n) for r, n in zip(["A", "X", "B"], [20, 0, 30])}
    placeholder_map = update_placeholder(
        runner,
        [1],  # X: a prefill chunk, discarded
        [(0, states["A"], 120), (2, states["B"], 130)],
        SimpleNamespace(scheduled_spec_decode_tokens={
            "A": [-1, -1, -1],
            "B": [-1, -1]
        }),
        None,
        _SpecDecodeMetadata(req_ids_dp={
            0: ["A", "X"],
            1: ["B"]
        }),
    )
    per_rank = MAX_NUM_SEQS // 2
    assert placeholder_map == {"A": 0, "B": per_rank + 0}
    assert states["A"].output_token_ids[-4:] == [0, 0, 0, 0]
    assert states["B"].output_token_ids[-3:] == [0, 0, 0]
    assert states["X"].output_token_ids == []


# --------------------------------------------------------------------------
# 4. The resumed history
# --------------------------------------------------------------------------
@dataclasses.dataclass
class _ReqState:
    req_id: str
    prompt_token_ids: list
    output_token_ids: list
    num_computed_tokens: int = 0
    block_ids: object = None
    mamba_state_slot: object = None
    mm_features: list = dataclasses.field(default_factory=list)

    @property
    def num_prompt_tokens(self):
        return len(self.prompt_token_ids)

    @property
    def num_tokens(self):
        return self.num_prompt_tokens + len(self.output_token_ids)


class _PbmBatch:
    """What ``update_states`` reads from ``InputBatch``. ``add_request`` is
    the token copy of ``input_batch.py`` L348-383."""

    def __init__(self):
        self.req_ids: list = []
        self.req_id_to_index: dict = {}
        self.token_ids_cpu = np.zeros((MAX_NUM_SEQS, MAX_MODEL_LEN), np.int32)
        self.num_tokens = np.zeros(MAX_NUM_SEQS, np.int32)
        self.num_tokens_no_spec = np.zeros(MAX_NUM_SEQS, np.int32)
        self.num_prompt_tokens = np.zeros(MAX_NUM_SEQS, np.int32)
        self.num_computed_tokens_cpu = np.zeros(MAX_NUM_SEQS, np.int32)
        self.mamba_state_indices_cpu = np.zeros(MAX_NUM_SEQS, np.int32)
        self.block_table = SimpleNamespace(add_row=lambda *a, **k: None,
                                           append_row=lambda *a, **k: None)
        self.request_distribution = [0, 0, 0]
        self.has_mamba_layers = False

    @property
    def num_reqs(self):
        return len(self.req_id_to_index)

    def add_request(self, request, req_index=None, dp_rank=0):
        if req_index is None:
            req_index = len(self.req_ids)
        if req_index == len(self.req_ids):
            self.req_ids.append(request.req_id)
        else:
            self.req_ids[req_index] = request.req_id
        self.req_id_to_index[request.req_id] = req_index
        n_prompt = len(request.prompt_token_ids)
        self.num_prompt_tokens[req_index] = n_prompt
        self.token_ids_cpu[req_index, :n_prompt] = request.prompt_token_ids
        end = n_prompt + len(request.output_token_ids)
        self.token_ids_cpu[req_index, n_prompt:end] = request.output_token_ids
        self.num_tokens[req_index] = request.num_tokens
        self.num_tokens_no_spec[req_index] = request.num_tokens
        self.num_computed_tokens_cpu[req_index] = request.num_computed_tokens

    def remove_request(self, req_id, *, free_mamba_slot=True):
        req_index = self.req_id_to_index.pop(req_id, None)
        if req_index is not None:
            self.req_ids[req_index] = None
        return req_index

    def condense(self, empty_req_indices):
        raise AssertionError("no hole expected in these scenarios")

    def release_mamba_slot(self, slot):
        pass

    def swap_states(self, i, j):
        raise AssertionError("no reorder expected in these scenarios")


def _resume_output(req_id, *, num_computed, num_output_tokens, all_token_ids,
                   num_scheduled):
    """The scheduler output of the resume step for one cached request."""
    req_data = SimpleNamespace(
        req_ids=[req_id],
        resumed_req_ids={req_id},
        new_token_ids=[[]],
        all_token_ids=all_token_ids,
        new_block_ids=[([7, 8], )],
        num_computed_tokens=[num_computed],
        num_output_tokens=[num_output_tokens],
    )
    return SimpleNamespace(
        scheduled_cached_reqs=req_data,
        scheduled_new_reqs=[],
        scheduled_spec_decode_tokens={},
        num_scheduled_tokens={req_id: num_scheduled},
        total_num_scheduled_tokens=num_scheduled,
        finished_req_ids=[],
        free_encoder_mm_hashes=[],
        preempted_req_ids=[],
    )


def _evicted_request():
    """R after the preemption step: three real outputs, then the 1 + K
    placeholder zeros the frontier step appended; the tokens that step
    sampled were never written because R had left the batch."""
    prompt = list(range(1, 11))
    return _ReqState(req_id="R",
                     prompt_token_ids=prompt,
                     output_token_ids=[101, 102, 103, 0, 0, 0, 0],
                     num_computed_tokens=13)


def test_update_states_rebuilds_a_resumed_history_from_the_scheduler():
    PersistentBatchManager = _load_pbm()
    req_state = _evicted_request()
    batch = _PbmBatch()
    manager = PersistentBatchManager({"R": req_state}, batch, {}, False,
                                     SimpleNamespace(), True)
    # The scheduler delivered the two tokens the in-flight step accepted
    # (AsyncScheduler._update_request_with_output, is_stale=True) before it
    # resumed R, and it sends the full list because R was not scheduled in
    # the previous step.
    delivered = req_state.prompt_token_ids + [101, 102, 103, 104, 105]
    scheduler_output = _resume_output("R",
                                      num_computed=0,
                                      num_output_tokens=5,
                                      all_token_ids={"R": delivered},
                                      num_scheduled=15)

    changed = manager.update_states(scheduler_output, None)

    assert changed
    assert req_state.output_token_ids == [101, 102, 103, 104, 105]
    assert req_state.num_computed_tokens == 0
    assert req_state.block_ids == ([7, 8], )
    idx = batch.req_id_to_index["R"]
    assert list(batch.token_ids_cpu[idx, :15]) == delivered
    assert batch.num_tokens[idx] == 15 and batch.num_tokens_no_spec[idx] == 15
    assert batch.num_computed_tokens_cpu[idx] == 0
    assert batch.num_prompt_tokens[idx] == 10


def test_update_states_without_all_token_ids_keeps_the_old_truncation():
    """A scheduler that does not send the list (the field absent, or the
    request not in it) gets exactly the pre-fix behaviour: the runner's own
    history, truncated to num_output_tokens."""
    PersistentBatchManager = _load_pbm()
    for all_token_ids in ({}, None):
        req_state = _evicted_request()
        batch = _PbmBatch()
        manager = PersistentBatchManager({"R": req_state}, batch, {}, False,
                                         SimpleNamespace(), True)
        scheduler_output = _resume_output("R",
                                          num_computed=0,
                                          num_output_tokens=5,
                                          all_token_ids=all_token_ids,
                                          num_scheduled=15)
        if all_token_ids is None:
            del scheduler_output.scheduled_cached_reqs.all_token_ids

        manager.update_states(scheduler_output, None)

        assert req_state.output_token_ids == [101, 102, 103, 0, 0]
        idx = batch.req_id_to_index["R"]
        assert list(batch.token_ids_cpu[idx, 10:15]) == [101, 102, 103, 0, 0]


def test_update_states_no_outputs_leaves_the_history_alone():
    """num_output_tokens == 0 (preempted before any token was delivered, or
    the in-flight output dropped): nothing to rebuild; the old truncation
    empties the placeholders."""
    PersistentBatchManager = _load_pbm()
    req_state = _ReqState(req_id="R",
                          prompt_token_ids=list(range(1, 11)),
                          output_token_ids=[0, 0, 0, 0],
                          num_computed_tokens=10)
    batch = _PbmBatch()
    manager = PersistentBatchManager({"R": req_state}, batch, {}, False,
                                     SimpleNamespace(), True)
    scheduler_output = _resume_output("R",
                                      num_computed=0,
                                      num_output_tokens=0,
                                      all_token_ids={"R": list(range(1, 11))},
                                      num_scheduled=10)

    manager.update_states(scheduler_output, None)

    assert req_state.output_token_ids == []
    idx = batch.req_id_to_index["R"]
    assert batch.num_tokens[idx] == 10


# --------------------------------------------------------------------------
# 5. The helper the call sites use
# --------------------------------------------------------------------------
def test_resumed_req_ids_helper_tolerates_missing_fields():
    (helper, ) = _load_runner_methods("_resumed_req_ids")
    assert helper(SimpleNamespace()) == frozenset()
    assert helper(SimpleNamespace(
        scheduled_cached_reqs=SimpleNamespace())) == frozenset()
    assert helper(
        SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids={"R", "S"}))) == frozenset({"R", "S"})
    assert helper(
        SimpleNamespace(scheduled_cached_reqs=SimpleNamespace(
            resumed_req_ids=["R"]))) == frozenset({"R"})


def test_call_sites_pass_the_resumed_set():
    """`_prepare_inputs` hands the set to both derivations and both
    `_modify_prev_results` calls that have a batch pass it."""
    tree = ast.parse(_RUNNER.read_text(), filename=str(_RUNNER))
    calls: dict = {}
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "self"):
            calls.setdefault(node.func.attr, []).append(node)
    for name in ("_prepare_async_token_substitution_indices",
                 "_subtract_num_rejected_tokens"):
        (call, ) = calls[name]
        assert any(kw.arg == "resumed_req_ids" for kw in call.keywords), name
    modify_calls = calls["_modify_prev_results"]
    with_set = [c for c in modify_calls if c.args]
    assert len(modify_calls) == 3 and len(with_set) == 2
