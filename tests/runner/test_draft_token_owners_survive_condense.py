"""Draft rows must keep their owners across a batch condense (async MTP + grammar).

Root cause of the ``Unexpected: grammar rejected tokens [...]`` terminations on
the MTP lanes with JSON-schema structured output (eval-26b-mtp 2026-09-03:
15/138 requests; eval-e4b-mtp 2026-09-02: 12/138; zero without MTP):

* Under async scheduling the engine core defers sampling of a step that has
  structured-output requests until the previous step's tokens have advanced
  the grammar (``vllm/v1/engine/core.py`` ``step_with_batch_queue``). It then
  calls ``take_draft_token_ids()``, validates those drafts against each
  request's grammar and pads the invalid tail with ``-1``
  (``scheduler.update_draft_token_ids_in_output``), builds the ``(1 + K)``
  bitmask rows from that prefix (``StructuredOutputManager.grammar_bitmask``)
  and only then calls ``sample_tokens``.
* With the ``UniProcExecutor`` (forced on single-host TPU) ``execute_model``
  runs inline even with ``non_block=True``, so by the time
  ``take_draft_token_ids()`` runs, the NEXT step's ``update_states`` has
  already removed the finished requests and condensed the batch
  (``InputBatch.condense``: the last request moves into the freed slot).
* ``SpeculativeDecodingManager.take_draft_token_ids`` zipped the cached draft
  rows -- laid out by batch index AT PROPOSAL TIME -- with the LIVE
  ``input_batch.req_ids``, so the request moved into the hole was handed the
  finished request's drafts. The scheduler validated the wrong drafts under
  its grammar, the ``(1 + K)`` rows were built from that prefix, and the
  recovery token the greedy rejection sampler takes at the first mismatch
  (``argmax`` of the masked target row) was sampled under a row built for the
  wrong prefix: legal in that state, illegal in the real one. That is the
  repeated-token shape in the pod logs (``[236779, 236779]``: quote, quote).

The rows themselves are laid out correctly (``test_guided_bitmask_specdecode_rows``
proves the mapping); it is the owner of each row that went stale.

CPU-only: numpy + jax[cpu]. Prefers the installed ``tpu_inference`` (the
in-image gate has vllm); on a box without vllm it loads the runner modules by
file path behind stand-in modules for the imports that do not resolve.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import pathlib
import sys
import types
from types import SimpleNamespace
from unittest.mock import patch

import jax
import jax.numpy as jnp
import numpy as np

_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUNNER = _ROOT / "tpu_inference" / "runner"
_SAMPLE = _ROOT / "tpu_inference" / "layers" / "jax" / "sample"


# --------------------------------------------------------------------------
# Loading the code under test
# --------------------------------------------------------------------------
@dataclasses.dataclass
class _DraftTokenIds:
    req_ids: list
    draft_token_ids: list


@dataclasses.dataclass
class _SpecDecodeMetadata:
    draft_lengths: object
    target_logits_indices: object
    bonus_logits_indices: object
    final_logits_indices: object
    draft_lengths_cpu: object = dataclasses.field(init=False, default=None)
    req_indices_dp: dict = dataclasses.field(init=False, default_factory=dict)
    req_ids_dp: dict = dataclasses.field(init=False, default_factory=dict)


def _device_array(mesh, *args, sharding=None, **kwargs):
    arrays = args[0] if len(args) == 1 else args
    return jax.tree.map(jnp.asarray, arrays)


def _module(name, **attrs):
    mod = types.ModuleType(name)
    mod.__dict__.update(attrs)
    return mod


def _load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _load_under_test():
    """(SpeculativeDecodingManager, StructuredDecodingManager, greedy sampler,
    parse_output), from the installed package when it imports, else from the
    files behind stand-in modules that are removed again afterwards."""
    try:
        from tpu_inference.layers.jax.sample.rejection_sampler import (
            RejectionSampler, _greedy_rejection_sample_with_segment)
        from tpu_inference.runner.speculative_decoding_manager import \
            SpeculativeDecodingManager
        from tpu_inference.runner.structured_decoding_manager import \
            StructuredDecodingManager
        return (SpeculativeDecodingManager, StructuredDecodingManager,
                _greedy_rejection_sample_with_segment,
                RejectionSampler.parse_output)
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
        _module("vllm.v1.core.sched.output",
                SchedulerOutput=_Cls,
                GrammarOutput=_Cls),
        "vllm.v1.outputs":
        _module("vllm.v1.outputs", DraftTokenIds=_DraftTokenIds),
        "vllm.v1.spec_decode":
        _module("vllm.v1.spec_decode"),
        "vllm.v1.spec_decode.ngram_proposer":
        _module("vllm.v1.spec_decode.ngram_proposer", NgramProposer=_Cls),
        "tpu_inference":
        _module("tpu_inference"),
        "tpu_inference.layers":
        _module("tpu_inference.layers"),
        "tpu_inference.layers.common":
        _module("tpu_inference.layers.common"),
        "tpu_inference.layers.common.sharding":
        _module("tpu_inference.layers.common.sharding",
                ShardingAxisName=SimpleNamespace(ATTN_DATA=("data",
                                                            "attn_dp"))),
        "tpu_inference.layers.common.binary_search":
        _module("tpu_inference.layers.common.binary_search",
                topk_mask=None,
                topp_mask=None),
        "tpu_inference.layers.jax":
        _module("tpu_inference.layers.jax"),
        "tpu_inference.layers.jax.sample":
        _module("tpu_inference.layers.jax.sample"),
        "tpu_inference.layers.jax.sample.sampling_metadata":
        _module("tpu_inference.layers.jax.sample.sampling_metadata",
                TPUSupportedSamplingMetadata=_Cls),
        "tpu_inference.runner":
        _module("tpu_inference.runner"),
        "tpu_inference.runner.utils":
        _module("tpu_inference.runner.utils",
                SpecDecodeMetadata=_SpecDecodeMetadata),
        "tpu_inference.spec_decode":
        _module("tpu_inference.spec_decode"),
        "tpu_inference.spec_decode.jax":
        _module("tpu_inference.spec_decode.jax"),
        "tpu_inference.spec_decode.jax.eagle3":
        _module("tpu_inference.spec_decode.jax.eagle3", Eagle3Proposer=_Cls),
        "tpu_inference.utils":
        _module("tpu_inference.utils", device_array=_device_array),
    }
    # The real row mapping, under the name structured_decoding_manager imports.
    stubs["tpu_inference.runner.grammar_bitmask_rows"] = _load_file(
        "tpu_inference.runner.grammar_bitmask_rows",
        _RUNNER / "grammar_bitmask_rows.py")
    for name, mod in stubs.items():
        parent, _, child = name.rpartition(".")
        if parent:
            setattr(stubs[parent], child, mod)
    injected = [n for n in stubs if n not in sys.modules]
    sys.modules.update({n: stubs[n] for n in injected})
    try:
        sdm = _load_file("_sdm_under_test",
                         _RUNNER / "speculative_decoding_manager.py")
        stm = _load_file("_stm_under_test",
                         _RUNNER / "structured_decoding_manager.py")
        rs = _load_file("_rs_under_test", _SAMPLE / "rejection_sampler.py")
    finally:
        for n in injected:
            sys.modules.pop(n, None)
    return (sdm.SpeculativeDecodingManager, stm.StructuredDecodingManager,
            rs._greedy_rejection_sample_with_segment,
            rs.RejectionSampler.parse_output)


# --------------------------------------------------------------------------
# The slice of the runner the managers read
# --------------------------------------------------------------------------
class _FakeInputBatch:
    """What the managers read from ``InputBatch``. ``condense`` is ported
    from ``input_batch.py`` L595-624: the last request moves into the hole."""

    def __init__(self):
        self._req_ids: list = []
        self.req_id_to_index: dict = {}

    @property
    def req_ids(self):
        return self._req_ids

    @property
    def num_reqs(self):
        return len(self.req_id_to_index)

    def add_request(self, req_id):
        idx = self.num_reqs
        if idx == len(self._req_ids):
            self._req_ids.append(req_id)
        else:
            self._req_ids[idx] = req_id
        self.req_id_to_index[req_id] = idx

    def remove_request(self, req_id):
        idx = self.req_id_to_index.pop(req_id)
        self._req_ids[idx] = None
        return idx

    def condense(self, empty_req_indices):
        last = self.num_reqs + len(empty_req_indices) - 1
        empty_req_indices = sorted(empty_req_indices, reverse=True)
        while empty_req_indices:
            while last in empty_req_indices:
                last -= 1
            empty = empty_req_indices.pop()
            if empty >= last:
                break
            req_id = self._req_ids[last]
            self._req_ids[empty] = req_id
            self._req_ids[last] = None
            self.req_id_to_index[req_id] = empty

    def finish(self, req_id):
        """``PersistentBatchManager.update_states`` for a finished request."""
        self.condense([self.remove_request(req_id)])


def _fake_runner(batch, num_spec, max_num_reqs, vocab_size=32):
    rows = max_num_reqs * (1 + num_spec)
    return SimpleNamespace(
        input_batch=batch,
        speculative_config=SimpleNamespace(num_speculative_tokens=num_spec,
                                           method="mtp",
                                           use_eagle=lambda: True),
        max_num_reqs=max_num_reqs,
        dp_size=1,
        drafter=object(),
        requests={},
        mesh=jax.make_mesh((1, 1, 1, 1),
                           ("data", "attn_dp", "expert", "model")),
        vocab_size=vocab_size,
        grammar_bitmask_cpu=np.zeros((rows, (vocab_size + 31) // 32),
                                     dtype=np.int32),
        require_structured_out_cpu=np.zeros((rows, 1), dtype=np.bool_),
        structured_decode_arange=np.arange(0, 32, dtype=np.int32),
    )


def _propose(mgr, runner, drafts_by_req):
    """Run ``propose_draft_token_ids`` on the eagle/MTP path exactly as
    ``_sample_from_logits`` does, with the drafter's device output stubbed:
    one row per batch index, padded to ``max_num_reqs`` rows."""
    batch = runner.input_batch
    num_spec = runner.speculative_config.num_speculative_tokens
    rows = np.zeros((runner.max_num_reqs, num_spec), dtype=np.int32)
    for req_id, drafts in drafts_by_req.items():
        rows[batch.req_id_to_index[req_id]] = drafts
    md = SimpleNamespace(
        req_indices_dp={0: list(range(batch.num_reqs))},
        req_ids_dp={0: list(batch.req_ids[:batch.num_reqs])},
    )
    with patch.object(mgr,
                      "propose_eagle3_draft_token_ids",
                      return_value=jnp.asarray(rows)):
        mgr.propose_draft_token_ids(None,
                                    None,
                                    None,
                                    None, [],
                                    None,
                                    None,
                                    async_scheduling=True,
                                    spec_decode_metadata=md,
                                    scheduler_output=None,
                                    input_ids=jnp.zeros((1, ), jnp.int32),
                                    hidden_states=None)


def _owners(draft_token_ids):
    return {
        req_id: [int(t) for t in toks]
        for req_id, toks in zip(draft_token_ids.req_ids,
                                draft_token_ids.draft_token_ids)
        if req_id is not None
    }


# --------------------------------------------------------------------------
# 1. Owner mapping across a condense (the defect, in isolation)
# --------------------------------------------------------------------------
def test_take_returns_the_owners_the_drafts_were_proposed_for():
    SDM, _, _, _ = _load_under_test()
    batch = _FakeInputBatch()
    for req_id in ("A", "B", "C"):
        batch.add_request(req_id)
    runner = _fake_runner(batch, num_spec=2, max_num_reqs=4)
    mgr = SDM(runner)
    drafts = {"A": [10, 11], "B": [20, 21], "C": [30, 31]}

    _propose(mgr, runner, drafts)  # sample_tokens(N-1)
    batch.finish("A")  # execute_model(N): update_states -> C moves to slot 0
    assert batch.req_ids[:batch.num_reqs] == ["C", "B"]

    owners = _owners(mgr.take_draft_token_ids())  # deferred bitmask for N
    assert owners["C"] == [30, 31], owners
    assert owners["B"] == [20, 21], owners
    assert owners["A"] == [10, 11], owners  # finished: the scheduler skips it
    assert mgr.take_draft_token_ids() is None
    assert mgr._draft_token_ids is None and mgr._draft_req_ids is None


def test_take_without_a_batch_change_is_the_live_order():
    SDM, _, _, _ = _load_under_test()
    batch = _FakeInputBatch()
    for req_id in ("A", "B", "C"):
        batch.add_request(req_id)
    runner = _fake_runner(batch, num_spec=2, max_num_reqs=4)
    mgr = SDM(runner)
    drafts = {"A": [10, 11], "B": [20, 21], "C": [30, 31]}
    _propose(mgr, runner, drafts)
    out = mgr.take_draft_token_ids()
    assert list(out.req_ids) == ["A", "B", "C"]
    assert _owners(out) == drafts


# --------------------------------------------------------------------------
# 2. The failure class end to end: a grammar-invalid recovery token
# --------------------------------------------------------------------------
LBRACE, QUOTE, KEY_K, KEY_J, COLON, DIGIT, RBRACE, NL, EOS = range(9)
VOCAB = 9
_ALLOWED = [
    {LBRACE},  # 0: start
    {QUOTE, NL},  # 1: after {
    {KEY_K, KEY_J},  # 2: inside the key string
    {QUOTE},  # 3: after the key
    {COLON, NL},  # 4: after the closing quote
    {DIGIT, NL},  # 5: after :
    {RBRACE, NL},  # 6: after the value
    {EOS, NL},  # 7: after }
]
_TERMINATED = len(_ALLOWED)


class ToyJsonGrammar:
    """``{"k": d}`` / ``{"j": d}``; newlines allowed between structural
    tokens, never inside the key string; EOS terminates. Exposes the xgrammar
    surface vLLM's backend drives (``backend_xgrammar.py`` L157-217)."""

    def __init__(self):
        self._states = [0]

    @property
    def state(self):
        return self._states[-1]

    def is_terminated(self):
        return self.state == _TERMINATED

    def accept_token(self, tok):
        if self.is_terminated() or tok not in _ALLOWED[self.state]:
            return False
        if tok == NL:
            self._states.append(self.state)
        elif tok == EOS:
            self._states.append(_TERMINATED)
        else:
            self._states.append(self.state + 1)
        return True

    def rollback(self, n):
        del self._states[len(self._states) - n:]

    def fill_next_token_bitmask(self, bitmask, idx):
        word = 0
        for tok in _ALLOWED[self.state]:
            word |= 1 << tok
        bitmask[idx, 0] = word

    # vllm.v1.structured_output.backend_xgrammar.XgrammarGrammar
    def accept_tokens(self, tokens):
        if self.is_terminated():
            return True
        for tok in tokens:
            if not self.accept_token(tok):
                return False
            if self.is_terminated():
                break
        return True

    def validate_tokens(self, tokens):
        accepted = []
        for tok in tokens:
            if self.accept_token(tok):
                accepted.append(tok)
            else:
                break
        if accepted:
            self.rollback(len(accepted))
        return accepted


def _fill(grammar, bitmask, idx):
    """``StructuredOutputManager._fill_bitmasks`` (structured_output/__init__.py
    L200-211): a terminated grammar gets the all-allowed row."""
    if grammar.is_terminated():
        bitmask[idx].fill(-1)
    else:
        grammar.fill_next_token_bitmask(bitmask, idx)


def scheduler_validate_and_pad(grammar, drafts, num_placeholders):
    """``Scheduler.update_draft_token_ids_in_output`` (scheduler.py
    L2260-2328) for one request: trim to the scheduled draft count, keep the
    grammar-valid prefix, pad with -1."""
    valid = grammar.validate_tokens(list(drafts[:num_placeholders]))
    return valid + [-1] * (num_placeholders - len(valid))


def scheduler_grammar_bitmask(grammars, structured_ids, spec_tokens):
    """``StructuredOutputManager.grammar_bitmask`` serial path
    (structured_output/__init__.py L281-345, post PR #8): per request one row
    per scheduled draft, advancing past real drafts and not past -1, then the
    bonus row, then rollback."""
    total = sum(1 + len(spec_tokens.get(r, ())) for r in structured_ids)
    bitmask = np.zeros((total, 1), dtype=np.int32)
    idx = 0
    for req_id in structured_ids:
        grammar = grammars[req_id]
        advanced = 0
        for tok in spec_tokens.get(req_id, ()):
            _fill(grammar, bitmask, idx)
            if tok != -1 and not grammar.is_terminated():
                assert grammar.accept_token(tok), (req_id, tok)
                advanced += 1
            idx += 1
        _fill(grammar, bitmask, idx)
        idx += 1
        if advanced:
            grammar.rollback(advanced)
    return bitmask


def _logits(rows_by_req, batch, num_spec, padded_rows):
    """Per-position logits in batch order, ``1 + K`` rows per request
    (``get_spec_decode_metadata``), padded to a bucket."""
    logits = np.zeros((padded_rows, VOCAB), dtype=np.float32)
    for req_id, rows in rows_by_req.items():
        base = batch.req_id_to_index[req_id] * (1 + num_spec)
        for p, prefs in enumerate(rows):
            for tok, score in prefs.items():
                logits[base + p, tok] = score
    return logits


def test_condense_between_propose_and_take_makes_the_recovery_token_illegal():
    """Step N-1 batch [A, B, C]; A finishes; step N batch is [C, B].

    C sits after ``{``; its own drafts are [QUOTE, KEY_K]. The target agrees
    on QUOTE and prefers KEY_J at the next position, so the greedy rejection
    sampler must emit the recovery token KEY_J from the row masked for
    "inside the key string". Handed A's drafts instead ([RBRACE, RBRACE],
    illegal after ``{``), the scheduler pads them to [-1, -1] and every row of
    C is built from the after-``{`` state; under that mask KEY_J is illegal,
    QUOTE wins, and C's step output is [QUOTE, QUOTE] -- accepted by the
    sampler, rejected by the grammar. That is the pod-log signature.
    """
    SDM, STM, greedy_rejection_sample, parse_output = _load_under_test()
    num_spec, max_num_reqs = 2, 4
    batch = _FakeInputBatch()
    for req_id in ("A", "B", "C"):
        batch.add_request(req_id)
    runner = _fake_runner(batch, num_spec, max_num_reqs, vocab_size=VOCAB)
    spec_mgr = SDM(runner)
    struct_mgr = STM(runner)

    grammars = {"B": ToyJsonGrammar(), "C": ToyJsonGrammar()}
    assert grammars["C"].accept_tokens([LBRACE])
    assert grammars["B"].accept_tokens([LBRACE, QUOTE, KEY_K, QUOTE, COLON])

    # sample_tokens(N-1): the drafter proposes for every batch row.
    drafts = {
        "A": [RBRACE, RBRACE],  # A is about to finish; its drafts are noise
        "B": [DIGIT, RBRACE],
        "C": [QUOTE, KEY_K],
    }
    _propose(spec_mgr, runner, drafts)

    # execute_model(N): update_states removes A and condenses; the forward
    # pass verifies each request's OWN drafts (the placeholder substitution
    # is keyed by request id).
    batch.finish("A")
    assert batch.req_ids[:batch.num_reqs] == ["C", "B"]

    # Deferred grammar work for N: take the drafts, validate per request,
    # build the (1 + K) rows in scheduler order.
    taken = spec_mgr.take_draft_token_ids()
    live = {"B", "C"}
    structured_ids = ["B", "C"]  # scheduler order != batch order
    scheduled = {}
    for req_id, req_drafts in zip(taken.req_ids, taken.draft_token_ids):
        if req_id in live:
            scheduled[req_id] = scheduler_validate_and_pad(
                grammars[req_id], [int(t) for t in req_drafts], num_spec)
    bitmask = scheduler_grammar_bitmask(grammars, structured_ids, scheduled)
    grammar_output = SimpleNamespace(
        structured_output_request_ids=structured_ids, grammar_bitmask=bitmask)
    scheduler_output = SimpleNamespace(scheduled_spec_decode_tokens=scheduled)

    # sample_tokens(N): mask the per-position logits with the real scatter +
    # kernel, then greedy rejection sampling on each request's own drafts.
    padded_rows = max_num_reqs * (1 + num_spec)
    logits = _logits(
        {
            "C": [
                {
                    QUOTE: 5.0,
                    KEY_J: 3.0,
                    NL: 1.0
                },  # after {
                {
                    KEY_J: 5.0,
                    KEY_K: 4.0,
                    QUOTE: 3.0,
                    NL: 1.0
                },  # after {"
                {
                    QUOTE: 5.0
                },  # after {"k
            ],
            "B": [
                {
                    DIGIT: 5.0,
                    NL: 1.0
                },  # after :
                {
                    RBRACE: 5.0,
                    NL: 1.0
                },  # after the digit
                {
                    EOS: 5.0,
                    NL: 1.0
                },  # after }
            ],
        },
        batch,
        num_spec,
        padded_rows)
    require, packed, arange = struct_mgr.prepare_structured_decoding_input(
        jnp.asarray(logits), grammar_output, scheduler_output=scheduler_output)
    masked = np.asarray(
        struct_mgr.structured_decode_fn(require, packed, jnp.asarray(logits),
                                        arange))

    order = batch.req_ids[:batch.num_reqs]
    target_rows = [
        b * (1 + num_spec) + p for b in range(len(order))
        for p in range(num_spec)
    ]
    bonus_rows = [b * (1 + num_spec) + num_spec for b in range(len(order))]
    verified_drafts = np.array([t for r in order for t in drafts[r]],
                               dtype=np.int32)
    draft_lengths = np.array([num_spec] * len(order), dtype=np.int32)
    output = greedy_rejection_sample(
        jnp.asarray(verified_drafts),
        jnp.asarray(masked[target_rows]),
        jnp.asarray(draft_lengths),
        jnp.asarray(np.argmax(masked[bonus_rows], axis=-1).astype(np.int32)),
    )
    sampled = parse_output(output, VOCAB, draft_lengths, len(order),
                           len(target_rows), 1, {0: list(range(len(order)))})
    sampled = dict(zip(order, sampled))

    # update_from_output(N): the grammar must accept every emitted token.
    assert sampled["B"] == [DIGIT, RBRACE, EOS], sampled
    assert grammars["B"].accept_tokens(sampled["B"])
    assert sampled["C"] == [
        QUOTE, KEY_J
    ], ("C's recovery token was sampled under a row built for another "
        "request's drafts", sampled, scheduled)
    assert grammars["C"].accept_tokens(
        sampled["C"]), ("grammar rejected tokens", sampled["C"])
