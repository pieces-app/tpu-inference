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
"""Both backbone primers must cover the SAME (num_tokens, num_reqs) grid.

`padded_num_reqs` is a META field of `AttentionMetadata`
(`layers/common/attention_metadata.py`: `meta_fields=["padded_num_reqs",
"pcp_cache_pages"]`), so it is a jit STATIC argument -- every value is its own
compiled program, exactly like `num_tokens`.  `_precompile_backbone_helper`
passes it straight through as `padded_num_reqs=num_reqs`.

`_precompile_backbone_with_inputs_embeds` used to build its dummies INSIDE the
`for num_reqs in attn_num_reqs_paddings` loop and call the helper OUTSIDE it,
so the multimodal primer covered exactly ONE request padding per token bucket
(whatever value the loop variable was left holding) while
`_precompile_backbone_text_only` covered the whole ladder.  A multimodal step
at any other request padding therefore traced and XLA-compiled the backbone
inside the serving loop: a TTFT spike on the image request, a
persistent-compilation-cache WRITE instead of a hit, or a hard death under
VLLM_XLA_CHECK_RECOMPILATION=1.

HOW THIS MEASURES IT.  The primers cannot be imported here -- the CPU gate has
jax[cpu]+flax and no vllm/torch, and `compilation_manager` imports `vllm.envs`
at module scope.  So the two primer methods (plus the two small helpers they
call for their ladders) are lifted out of the SHIPPED source with `ast`,
unparsed into a standalone module and executed against fakes: fake
`NamedSharding`/`PartitionSpec`/`jnp`, a fake runner, a `_create_dummy_tensor`
that records a shape/dtype/sharding record instead of allocating, and a
`_precompile_backbone_helper` that records `(num_tokens, num_reqs)` instead of
lowering anything.  The loop structure under test is the real one, byte for
byte; only the leaves are stubbed.

`test_the_enumerator_sees_the_pre_fix_defect` is the in-file positive control:
it runs the same enumerator over a synthetic function with the OLD shape (call
hoisted out of the request-padding loop) and asserts the coverage comes back
deficient.  Without it a harness that silently recorded nothing would make
every assertion below vacuously green.

NEGATIVE CONTROL (applied to the tree, run, then restored byte-identically):
re-indenting `_precompile_backbone_with_inputs_embeds` back to the pre-fix
shape -> this file fails.
"""
import ast
import importlib.util
import itertools
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CM = ROOT / "tpu_inference" / "runner" / "compilation_manager.py"
RUNNER_UTILS = ROOT / "tpu_inference" / "runner" / "utils.py"
ENVS = ROOT / "tpu_inference" / "envs.py"

TEXT_PRIMER = "_precompile_backbone_text_only"
EMBEDS_PRIMER = "_precompile_backbone_with_inputs_embeds"
LADDER = "_mm_embeds_primer_req_paddings"
PCP_LADDER = "_pcp_cache_page_buckets"

TOKEN_PADDINGS = [16, 32, 64, 128]
REQ_PADDINGS = [8, 16, 32, 64]


# --------------------------------------------------------------------------
# the fakes
# --------------------------------------------------------------------------
class _PartitionSpec(tuple):
    """Structural stand-in for jax.sharding.PartitionSpec."""

    def __new__(cls, *axes):
        return super().__new__(cls, axes)

    def __repr__(self):
        return f"P{tuple(self)}"


class _NamedSharding:

    def __init__(self, mesh, spec):
        self.mesh, self.spec = mesh, spec

    def __eq__(self, other):
        return (isinstance(other, _NamedSharding) and self.mesh is other.mesh
                and tuple(self.spec) == tuple(other.spec))

    def __hash__(self):
        return hash((id(self.mesh), tuple(self.spec)))

    def __repr__(self):
        return f"NamedSharding({self.spec!r})"


class _ShardingAxisName:
    BATCH = "batch"
    ATTN_DATA = "attn_data"
    MODEL = "model"
    PREFILL_CONTEXT = "prefill_context"


class _Jnp:
    int32 = "int32"
    bfloat16 = "bfloat16"


class _Dummy:
    """What `_create_dummy_tensor` returns here: shape/dtype/sharding only."""

    def __init__(self, shape, dtype, sharding):
        self.shape, self.dtype, self.sharding = tuple(shape), dtype, sharding

    def key(self):
        return (self.shape, self.dtype, self.sharding)

    def __repr__(self):
        return f"Dummy{self.shape}:{self.dtype}@{self.sharding}"


class _JaxIntermediateTensors:

    def __init__(self, tensors):
        self.tensors = tensors


class _Envs:
    """Only the flags the lifted code reads."""

    def __init__(self, all_req_paddings=True, bucketized=False, custom=()):
        self.MM_EMBEDS_PRIMER_ALL_REQ_PADDINGS = all_req_paddings
        self.ATTN_BUCKETIZED_NUM_REQS = bucketized
        self.ATTN_CUSTOM_NUM_REQS_BUCKETS = list(custom)


class _ModelConfig:

    def __init__(self, hidden_size, dtype, hf_config):
        self._hidden_size = hidden_size
        self.dtype = dtype
        self.hf_config = hf_config

    def get_hidden_size(self):
        return self._hidden_size


class _Runner:

    def __init__(self,
                 *,
                 hidden_size=64,
                 hf_config=None,
                 uses_mrope=False,
                 is_first_rank=True,
                 is_last_rank=True,
                 prefill_cp_size=1,
                 token_paddings=None,
                 req_paddings=None):
        self.mesh = object()
        self.rank = 0
        self.uses_mrope = uses_mrope
        self.is_first_rank = is_first_rank
        self.is_last_rank = is_last_rank
        self.num_tokens_paddings = list(token_paddings or TOKEN_PADDINGS)
        self.attn_num_reqs_paddings = list(req_paddings or REQ_PADDINGS)
        self.max_num_blocks_per_req = 32
        cfg = _ModelConfig(hidden_size, "bfloat16", hf_config
                           or _HfConfig(None))
        self.model_config = cfg
        self.vllm_config = _VllmConfig(cfg, prefill_cp_size)


class _HfConfig:

    def __init__(self, vision_config):
        self.vision_config = vision_config


class _VisionConfig:

    def __init__(self, out_hidden_size, deepstack_visual_indexes):
        self.out_hidden_size = out_hidden_size
        self.deepstack_visual_indexes = deepstack_visual_indexes


class _ShardingConfig:

    def __init__(self, prefill_cp_size):
        self.prefill_cp_size = prefill_cp_size


class _VllmConfig:

    def __init__(self, model_config, prefill_cp_size):
        self.model_config = model_config
        self.sharding_config = _ShardingConfig(prefill_cp_size)


# --------------------------------------------------------------------------
# lifting the shipped source
# --------------------------------------------------------------------------
def _functions(src: str, names):
    """The named `def`s from `src`, wherever they sit (module or class)."""
    found = {}
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            assert node.name not in found, f"{node.name} defined twice"
            found[node.name] = node
    return found


def _build(src: str, names, extra_globals):
    """exec the lifted `def`s in a namespace of fakes; return the functions."""
    fns = _functions(src, names)
    missing = set(names) - set(fns)
    assert not missing, f"not found in the shipped source: {sorted(missing)}"
    module_src = "\n\n".join(ast.unparse(fns[n]) for n in names)
    ns = {
        "NamedSharding": _NamedSharding,
        "PartitionSpec": _PartitionSpec,
        "ShardingAxisName": _ShardingAxisName,
        "jnp": _Jnp,
        "JaxIntermediateTensors": _JaxIntermediateTensors,
        "List": list,
        "Optional": object,
        "logger": _NullLogger(),
    }
    ns.update(extra_globals)
    exec(compile(module_src, "<lifted>", "exec"), ns)  # noqa: S102
    return {n: ns[n] for n in names}


class _NullLogger:

    def info(self, *a, **k):
        pass

    def warning(self, *a, **k):
        pass


class _Recorder:
    """The `self` the lifted primers run against."""

    def __init__(self, runner, envs):
        self.runner = runner
        self._envs = envs
        self.compiled = []  # (name, num_tokens, num_reqs, pcp)
        self.dummies = []  # every _create_dummy_tensor call
        self.operands = []  # the full kwargs of every helper call

    # --- stubbed leaves -------------------------------------------------
    def _create_dummy_tensor(self, shape, dtype, sharding=None):
        d = _Dummy(shape, dtype, sharding)
        self.dummies.append(d)
        return d

    def _precompile_backbone_helper(self,
                                    name,
                                    *,
                                    input_ids,
                                    positions,
                                    inputs_embeds,
                                    intermediate_tensors=None,
                                    is_first_rank=True,
                                    is_last_rank=True,
                                    num_reqs,
                                    pcp_cache_pages=0):
        # Derive num_tokens exactly the way the real helper does.
        if input_ids is not None:
            num_tokens = input_ids.shape[0]
        else:
            num_tokens = inputs_embeds.shape[0]
        self.compiled.append((name, num_tokens, num_reqs, pcp_cache_pages))
        self.operands.append(
            dict(name=name,
                 input_ids=input_ids,
                 positions=positions,
                 inputs_embeds=inputs_embeds,
                 num_reqs=num_reqs,
                 pcp_cache_pages=pcp_cache_pages))

    # --- coverage -------------------------------------------------------
    def grid(self):
        return {(t, r) for _, t, r, _ in self.compiled}


def _run(primer_names, runner, envs, src=None, extra_names=()):
    src = CM.read_text() if src is None else src
    names = list(
        dict.fromkeys(
            list(primer_names) + [LADDER, PCP_LADDER] + list(extra_names)))
    fns = _build(src, names, {
        "envs": envs,
        "pcp_cache_page_buckets": lambda n: [0, 1, 2],
    })
    rec = _Recorder(runner, envs)
    for name, fn in fns.items():
        setattr(rec, name, fn.__get__(rec, _Recorder))
    for name in primer_names:
        fns[name](rec)
    return rec


def _both(runner=None, envs=None, src=None):
    runner = runner or _Runner()
    envs = envs or _Envs()
    text = _run([TEXT_PRIMER], runner, envs, src=src)
    embeds = _run([EMBEDS_PRIMER], runner, envs, src=src)
    return text, embeds


# --------------------------------------------------------------------------
# the harness has teeth
# --------------------------------------------------------------------------
def test_the_lift_found_both_primers_and_both_ladders():
    """A lift that silently finds nothing makes every check below vacuous."""
    fns = _functions(CM.read_text(),
                     {TEXT_PRIMER, EMBEDS_PRIMER, LADDER, PCP_LADDER})
    assert set(fns) == {TEXT_PRIMER, EMBEDS_PRIMER, LADDER, PCP_LADDER}


def test_the_harness_records_a_non_empty_grid_for_both_primers():
    text, embeds = _both()
    assert len(text.compiled) == len(TOKEN_PADDINGS) * len(REQ_PADDINGS)
    assert len(embeds.compiled) == len(TOKEN_PADDINGS) * len(REQ_PADDINGS)


PRE_FIX_SHAPE = '''
def _precompile_backbone_with_inputs_embeds_prefix(self) -> None:
    """The pre-fix shape, verbatim in structure: the dummies are built inside
    the request-padding loop and the helper call sits outside it."""
    hidden_size = self.runner.model_config.get_hidden_size()
    dtype = self.runner.model_config.dtype
    for num_tokens in self.runner.num_tokens_paddings:
        for num_reqs in self.runner.attn_num_reqs_paddings:
            sharding = NamedSharding(
                self.runner.mesh,
                PartitionSpec(ShardingAxisName.ATTN_DATA, None))
            inputs_embeds = self._create_dummy_tensor(
                (num_tokens, hidden_size), dtype, sharding=sharding)
        ids_sharding = NamedSharding(
            self.runner.mesh, PartitionSpec(ShardingAxisName.BATCH))
        input_ids = self._create_dummy_tensor(
            (num_tokens, ), jnp.int32, ids_sharding)
        positions = self._create_dummy_tensor(
            (num_tokens, ), jnp.int32, ids_sharding)
        self._precompile_backbone_helper(
            "backbone with embeds",
            input_ids=input_ids,
            positions=positions,
            inputs_embeds=inputs_embeds,
            num_reqs=num_reqs)
'''


def test_the_enumerator_sees_the_pre_fix_defect():
    """POSITIVE CONTROL. Feed the enumerator the old loop shape and it must
    report deficient coverage -- one request padding per token bucket, the
    last one on the ladder. If this passes silently the whole file is a
    no-op."""
    runner, envs = _Runner(), _Envs()
    src = CM.read_text() + "\n" + PRE_FIX_SHAPE
    broken = _run(["_precompile_backbone_with_inputs_embeds_prefix"],
                  runner,
                  envs,
                  src=src)
    text = _run([TEXT_PRIMER], runner, envs)
    assert broken.grid() != text.grid()
    assert broken.grid() == {(t, REQ_PADDINGS[-1]) for t in TOKEN_PADDINGS}
    assert len(broken.grid()) == len(TOKEN_PADDINGS)
    assert len(text.grid()) == len(TOKEN_PADDINGS) * len(REQ_PADDINGS)


# --------------------------------------------------------------------------
# the property
# --------------------------------------------------------------------------
def test_both_primers_cover_the_same_num_tokens_num_reqs_grid():
    """THE FIX. Anything the text-only primer primes, the multimodal primer
    primes too -- otherwise an image request at that request padding compiles
    the backbone inside the serving loop."""
    text, embeds = _both()
    assert embeds.grid() == text.grid(), (
        "grid mismatch; text-only primes "
        f"{sorted(text.grid() - embeds.grid())} that the embeds primer does "
        "not")


def test_the_grid_is_the_full_cartesian_product():
    text, embeds = _both()
    expected = set(itertools.product(TOKEN_PADDINGS, REQ_PADDINGS))
    assert text.grid() == expected
    assert embeds.grid() == expected


@pytest.mark.parametrize("uses_mrope", [False, True])
@pytest.mark.parametrize("is_first_rank", [True, False])
def test_the_grid_holds_across_mrope_and_pipeline_rank(uses_mrope,
                                                       is_first_rank):
    """The primer has four shape branches; none of them may drop a padding."""
    runner = _Runner(uses_mrope=uses_mrope, is_first_rank=is_first_rank)
    text, embeds = _both(runner=runner)
    assert embeds.grid() == text.grid()


def test_the_deepstack_second_hidden_size_also_walks_the_whole_ladder():
    """Deepstack models compile the embeds primer at two hidden sizes; both
    have to cover the ladder, not just the last one."""
    runner = _Runner(hidden_size=64,
                     hf_config=_HfConfig(_VisionConfig(32, [1, 2])))
    embeds = _run([EMBEDS_PRIMER], runner, _Envs())
    by_h = {}
    for op in embeds.operands:
        by_h.setdefault(op["inputs_embeds"].shape[1], set()).add(
            (op["inputs_embeds"].shape[0], op["num_reqs"]))
    assert set(by_h) == {64, 32 * 3}, sorted(by_h)
    full = set(itertools.product(TOKEN_PADDINGS, REQ_PADDINGS))
    for h, grid in by_h.items():
        assert grid == full, f"hidden size {h} covers only {sorted(grid)}"


def test_the_embeds_primer_keeps_the_ids_operand_the_text_primer_uses():
    """#61's pin, measured rather than parsed: the ids dummy the multimodal
    primer passes has the same shape, dtype and sharding as the text-only
    primer's at the same token bucket. Shape AND sharding are part of the jit
    signature."""
    text, embeds = _both()
    text_ids = {
        op["input_ids"].shape[0]: op["input_ids"].key()
        for op in text.operands
    }
    for op in embeds.operands:
        ids = op["input_ids"]
        assert ids is not None, "the multimodal primer dropped input_ids"
        assert ids.key() == text_ids[ids.shape[0]], (
            f"ids operand differs from the text primer's at "
            f"{ids.shape[0]} tokens: {ids.key()} vs {text_ids[ids.shape[0]]}")


def test_the_embeds_operand_is_present_and_tracks_its_token_bucket():
    embeds = _run([EMBEDS_PRIMER], _Runner(), _Envs())
    for op in embeds.operands:
        assert op["inputs_embeds"] is not None
        assert op["inputs_embeds"].shape[0] == op["input_ids"].shape[0]


def test_the_text_primer_still_passes_no_embeds():
    """The two step kinds stay two distinct jit signatures."""
    text = _run([TEXT_PRIMER], _Runner(), _Envs())
    assert all(op["inputs_embeds"] is None for op in text.operands)
    assert all(op["input_ids"] is not None for op in text.operands)


def test_the_dummies_are_built_once_per_token_bucket_not_per_request_padding():
    """The dummies do not depend on `num_reqs`, so widening the grid must not
    widen the allocation count: the request ladder wraps the CALL only.
    Pre-fix, `inputs_embeds` was allocated once per (token, req) pair and all
    but the last thrown away."""
    embeds = _run([EMBEDS_PRIMER], _Runner(), _Envs())
    assert len(embeds.compiled) == len(TOKEN_PADDINGS) * len(REQ_PADDINGS)
    # ids + positions + embeds per token bucket, and nothing per request
    # padding.
    assert len(embeds.dummies) == 3 * len(TOKEN_PADDINGS)


# --------------------------------------------------------------------------
# what it costs, and the escape hatch
# --------------------------------------------------------------------------
def test_the_env_off_restores_the_pre_fix_coverage_exactly():
    runner = _Runner()
    off = _run([EMBEDS_PRIMER], runner, _Envs(all_req_paddings=False))
    assert off.grid() == {(t, REQ_PADDINGS[-1]) for t in TOKEN_PADDINGS}
    assert len(off.compiled) == len(TOKEN_PADDINGS)


def test_the_env_defaults_to_on_in_the_shipped_envs_module():
    """Default ON: the grid is free unless request bucketizing is enabled
    (see the next test), so correctness is the default and the hatch is the
    opt-in."""
    spec = importlib.util.spec_from_file_location("_lifted_envs", ENVS)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    assert mod.MM_EMBEDS_PRIMER_ALL_REQ_PADDINGS is True


def test_the_default_deployment_pays_no_extra_compiles():
    """`get_attn_req_paddings` returns a SINGLE padding unless
    ATTN_BUCKETIZED_NUM_REQS is set, so on a default run the full grid and the
    pre-fix coverage are the same one-wide ladder: zero extra compiles."""
    fns = _build(
        RUNNER_UTILS.read_text(), ["get_attn_req_paddings"], {
            "envs": _Envs(bucketized=False),
            "get_req_paddings": lambda a, b: [8, 16, 32, 64],
        })
    ladder = fns["get_attn_req_paddings"](min_req_size=8, max_req_size=64)
    assert ladder == [64], ladder

    runner = _Runner(req_paddings=ladder)
    on = _run([EMBEDS_PRIMER], runner, _Envs(all_req_paddings=True))
    off = _run([EMBEDS_PRIMER], runner, _Envs(all_req_paddings=False))
    assert on.grid() == off.grid()
    assert len(on.compiled) == len(off.compiled) == len(TOKEN_PADDINGS)


def test_bucketizing_is_what_makes_the_grid_cost_anything():
    """And when it does, the multiplier is the ladder length -- the same one
    the text-only primer already pays."""
    fns = _build(
        RUNNER_UTILS.read_text(), ["get_attn_req_paddings"], {
            "envs": _Envs(bucketized=True),
            "get_req_paddings": lambda a, b: [8, 16, 32, 64],
        })
    ladder = fns["get_attn_req_paddings"](min_req_size=8, max_req_size=64)
    assert ladder == [8, 16, 32, 64], ladder

    runner = _Runner(req_paddings=ladder)
    on = _run([EMBEDS_PRIMER], runner, _Envs(all_req_paddings=True))
    off = _run([EMBEDS_PRIMER], runner, _Envs(all_req_paddings=False))
    assert len(off.compiled) == len(TOKEN_PADDINGS)
    assert len(on.compiled) == len(TOKEN_PADDINGS) * len(ladder)
    text = _run([TEXT_PRIMER], runner, _Envs())
    assert len(text.compiled) == len(on.compiled)


# --------------------------------------------------------------------------
# the one asymmetry this change does NOT close
# --------------------------------------------------------------------------
def test_the_pcp_cache_page_ladder_is_a_separate_uncovered_gap():
    """NOT FIXED HERE, pinned so it cannot be forgotten. The text-only primer
    walks `_pcp_cache_page_buckets()`; the embeds primer takes the helper's
    `pcp_cache_pages=0` default. With prefill context parallelism off (the
    default, and every arm the fork runs) that ladder is `[0]` and the two
    agree, so this file's grid claim holds. With PCP on the multimodal primer
    still under-covers on the cache-page axis -- a distinct defect on a
    distinct axis, left for its own change."""
    runner = _Runner(prefill_cp_size=1)
    text = _run([TEXT_PRIMER], runner, _Envs())
    embeds = _run([EMBEDS_PRIMER], runner, _Envs())
    assert {c[3] for c in text.compiled} == {0}
    assert {c[3] for c in embeds.compiled} == {0}
    assert {c[1:] for c in text.compiled} == {c[1:] for c in embeds.compiled}

    pcp_runner = _Runner(prefill_cp_size=4)
    pcp_text = _run([TEXT_PRIMER], pcp_runner, _Envs())
    pcp_embeds = _run([EMBEDS_PRIMER], pcp_runner, _Envs())
    assert {c[3] for c in pcp_text.compiled} == {0, 1, 2}
    # AUDIT 2026-09-03: `assert {c[3] for c in pcp_embeds.compiled} == {0}`
    # stood here, i.e. the test asserted THE DEFECT IS STILL PRESENT.
    # Measured: wrapping the embeds primer's helper call in
    # `self._pcp_cache_page_buckets()` -- the obvious fix -- turned this test
    # RED. A test that fails when the bug it documents is fixed is a tax on
    # the next change, and it gets "fixed" by deleting the line rather than
    # reading it. Assert the invariant that holds in BOTH states instead, and
    # move the gap itself to an xfail that goes green on its own when closed.
    assert {c[3]
            for c in pcp_embeds.compiled} <= {c[3]
                                              for c in pcp_text.compiled
                                              }, ("the embeds primer covers a "
                                                  "cache-page bucket the text "
                                                  "primer does not")
    # the (num_tokens, num_reqs) grid -- what this change is about -- still
    # matches even there.
    assert pcp_embeds.grid() == pcp_text.grid()


@pytest.mark.xfail(
    reason="the embeds primer takes _precompile_backbone_helper's "
    "pcp_cache_pages=0 default instead of walking _pcp_cache_page_buckets(); "
    "a distinct defect on a distinct axis, left for its own change",
    strict=False)
def test_the_embeds_primer_walks_the_pcp_cache_page_ladder_too():
    """The gap above, as a test that turns green when the gap is closed."""
    pcp_runner = _Runner(prefill_cp_size=4)
    pcp_embeds = _run([EMBEDS_PRIMER], pcp_runner, _Envs())
    assert {c[3] for c in pcp_embeds.compiled} == {0, 1, 2}
