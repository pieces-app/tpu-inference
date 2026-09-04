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
"""The `model_fn` operand pair, and the primers that have to match it.

A multimodal step now hands the model BOTH `input_ids` and `inputs_embeds`
(`TPUModelRunner._get_input_ids_embeds`), because Gemma-4 E2B/E4B need the
real token ids to build the per-layer-embedding id-track on an image
prefill; withholding them collapsed that track to embedding slot 0 for the
entire prompt.  See `tests/models/vllm/experimental/
test_gemma4_ple_image_prefill.py` for the numbers and
`test_gemma4_ple_reference_parity.py` for the transformers oracle.

WHY THIS FILE EXISTS: the operand pair is a JIT SIGNATURE, not a value.
`None` and `int32[num_tokens]` are different pytree/aval structures, so
every step kind must be primed with the structure it will actually run:

    text step        (input_ids,  None  )  <- _precompile_backbone_text_only
    multimodal step  (input_ids, embeds )  <- _precompile_backbone_with_inputs_embeds

Both are static across a run: the runner has exactly two returns, one per
kind, and neither is conditional on the model.  A primer that builds the
wrong pair reports SUCCESS and then makes request #1 compile a 12B/26B
model inside the serving loop -- minutes of TTFT, a persistent-cache miss,
or a hard death under VLLM_XLA_CHECK_RECOMPILATION=1.  That is exactly the
failure `test_primers_carry_mm_bidi_ranges.py` was written for, one operand
over.

Source-structure tests: exercising the primers needs a live runner, a mesh
and a TPU. They need neither jax nor vllm, so they run anywhere.

NEGATIVE CONTROLS, each mutation applied to the tree and reverted:
  * `return None, inputs_embeds` restored in `_get_input_ids_embeds`
      -> 2 failed (here and in the PLE image-prefill file)
  * `input_ids=None` restored in the embeds primer          -> 1 failed
  * the embeds primer's dummy built with `jnp.bfloat16`     -> 2 failed
  * the embeds primer's ids sharded ATTN_DATA not BATCH     -> 1 failed
  * the flax out-of-vocab `jnp.where` deleted               -> 1 failed
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
RUNNER = ROOT / "tpu_inference" / "runner" / "tpu_runner.py"
CM = ROOT / "tpu_inference" / "runner" / "compilation_manager.py"
FLAX_GEMMA4 = ROOT / "tpu_inference" / "models" / "jax" / "gemma4.py"
FLAX_GEMMA4_MM = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mm.py"
FLAX_MTP = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mtp.py"

TEXT_PRIMER = "_precompile_backbone_text_only"
EMBEDS_PRIMER = "_precompile_backbone_with_inputs_embeds"


def _fn(path: pathlib.Path, name: str) -> tuple[str, ast.FunctionDef]:
    src = path.read_text()
    found = [
        n for n in ast.walk(ast.parse(src))
        if isinstance(n, ast.FunctionDef) and n.name == name
    ]
    assert len(found) == 1, f"{name} in {path.name}: found {len(found)}"
    return src, found[0]


def _method(path: pathlib.Path, class_name: str,
            method_name: str) -> tuple[str, ast.FunctionDef]:
    src = path.read_text()
    for node in ast.walk(ast.parse(src)):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            found = [
                n for n in node.body
                if isinstance(n, ast.FunctionDef) and n.name == method_name
            ]
            assert len(found) == 1, f"{class_name}.{method_name}"
            return src, found[0]
    raise AssertionError(f"{class_name} not found in {path.name}")


def _ids_build(path: pathlib.Path, fn_name: str) -> str:
    """The single `input_ids = ...` assignment inside `fn_name`."""
    src, fn = _fn(path, fn_name)
    assigns = [
        n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "input_ids" for t in n.targets)
    ]
    assert len(assigns) == 1, (
        f"{fn_name}: expected exactly one input_ids build, got "
        f"{len(assigns)}")
    return " ".join(ast.get_source_segment(src, assigns[0]).split())


# --------------------------------------------------------------------- #
# The runner's half of the contract
# --------------------------------------------------------------------- #


def test_the_multimodal_branch_returns_the_token_ids():
    """`_get_input_ids_embeds` must return the ids on BOTH branches.

    This is the whole fix. Returning `None` here is what made the PLE
    id-track degenerate on every image step, on both model paths.
    """
    _, fn = _fn(RUNNER, "_get_input_ids_embeds")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 2, "expected one return per step kind"
    for ret in returns:
        assert isinstance(ret.value, ast.Tuple), ast.dump(ret)
        first, _second = ret.value.elts
        assert isinstance(first, ast.Name) and first.id == "input_ids", (
            "the ids operand is None on one branch -- the model cannot "
            "rebuild the PLE id-track from embeddings")


def test_exactly_one_branch_carries_embeds():
    """The two returns are the two step kinds; a third would be a third
    signature with no primer behind it.

    AUDIT 2026-09-03: `sorted(has_embeds) == [False, True]` is satisfied by
    BOTH the fixed `(input_ids, inputs_embeds)/(input_ids, None)` and the
    pre-#61 `(None, inputs_embeds)/(input_ids, None)` -- measured: it survives
    the exact reverted code, so it read as a second guard on the fix and was
    not one. Assert the two return SHAPES together, which pins the branch
    count and the operands at once.
    """
    _, fn = _fn(RUNNER, "_get_input_ids_embeds")
    returns = [n for n in ast.walk(fn) if isinstance(n, ast.Return)]
    assert len(returns) == 2, "expected one return per step kind"
    shapes = {(ast.unparse(r.value.elts[0]), ast.unparse(r.value.elts[1]))
              for r in returns}
    assert shapes == {
        ("input_ids", "inputs_embeds"), ("input_ids", "None")
    }, (f"_get_input_ids_embeds returns {sorted(shapes)}; the multimodal "
        f"branch must carry BOTH operands and the text branch the ids alone")


def test_the_forward_call_takes_the_model_side_name():
    """Position 2 of `model_fn` is the model-side ids, position 4 the
    embeds. (The raw buffer keeps its own name for spec decode; see
    test_mtp_mm_input_ids_source.py.)"""
    src, fn = _fn(RUNNER, "_execute_model")
    calls = [
        n for n in ast.walk(fn) if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Attribute) and n.func.attr == "model_fn"
    ]
    assert len(calls) == 1
    args = calls[0].args
    assert isinstance(args[2], ast.Name) and args[2].id == "model_input_ids"
    assert isinstance(args[4], ast.Name) and args[4].id == "inputs_embeds"


# --------------------------------------------------------------------- #
# The primers' half
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("primer", [TEXT_PRIMER, EMBEDS_PRIMER])
def test_every_backbone_primer_builds_a_token_ids_operand(primer):
    build = _ids_build(CM, primer)
    assert "_create_dummy_tensor" in build, build
    assert "num_tokens" in build, build
    assert "jnp.int32" in build, build


@pytest.mark.parametrize("primer", [TEXT_PRIMER, EMBEDS_PRIMER])
def test_the_primers_pass_the_ids_they_build(primer):
    src, fn = _fn(CM, primer)
    body = ast.get_source_segment(src, fn)
    assert "input_ids=input_ids" in body, (
        f"{primer} does not hand its input_ids to _precompile_backbone_helper")
    assert "input_ids=None" not in body, (
        f"{primer} still primes the ids-free signature -- the runtime never "
        f"hits that graph, so every real step of that kind recompiles")


def test_the_two_primers_build_the_same_ids_operand():
    """Same shape, same dtype, same sharding.

    The runtime array is the SAME buffer on both step kinds (it comes out of
    the one `metadata["input_ids"]` slice in `_prepare_inputs`), so the two
    primers must not disagree about it. jit specialises on sharding as well
    as shape and dtype.
    """
    text = _ids_build(CM, TEXT_PRIMER)
    embeds = _ids_build(CM, EMBEDS_PRIMER)
    for token in ("(num_tokens, )", "jnp.int32"):
        assert token in text, text
        assert token in embeds, embeds
    # Both must name a BATCH-partitioned sharding, and neither may fall back
    # to the positions' ATTN_DATA one.
    src_text, fn_text = _fn(CM, TEXT_PRIMER)
    src_embeds, fn_embeds = _fn(CM, EMBEDS_PRIMER)
    for name, build, src, fn in (
        (TEXT_PRIMER, text, src_text, fn_text),
        (EMBEDS_PRIMER, embeds, src_embeds, fn_embeds),
    ):
        sharding_name = build.rstrip(")").split(",")[-1].strip()
        assert sharding_name, name
        assigned = [
            n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(
                isinstance(t, ast.Name) and t.id == sharding_name
                for t in n.targets)
        ]
        assert len(assigned) == 1, f"{name}: {sharding_name} not built once"
        spec = " ".join(ast.get_source_segment(src, assigned[0]).split())
        assert "ShardingAxisName.BATCH" in spec, (
            f"{name} builds its input_ids with {spec!r}; the runtime array is "
            f"BATCH-partitioned, and a sharding mismatch is a jit cache miss")
        # AUDIT 2026-09-03: `assert body  # the segment resolved` stood here.
        # `body` is a non-empty source string by the time this line runs (the
        # loop above already indexed into it), so it could never be falsy.
        # The resolution it meant to guard is asserted where it happens.


def test_the_embeds_primer_is_the_only_multimodal_entry_point():
    """If a second primer started passing inputs_embeds it would need the
    same treatment; assert there is still exactly one."""
    src = CM.read_text()
    tree = ast.parse(src)
    callers = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "_precompile_backbone_helper"):
            continue
        embeds = [k for k in node.keywords if k.arg == "inputs_embeds"]
        assert len(embeds) == 1
        value = embeds[0].value
        if isinstance(value, ast.Constant) and value.value is None:
            continue
        callers.add(node.lineno)
    assert len(callers) == 1, (
        f"expected one primer to pass inputs_embeds, found {len(callers)}")


# --------------------------------------------------------------------- #
# What must NOT change
# --------------------------------------------------------------------- #


def test_the_text_only_primer_still_primes_the_ids_only_pair():
    """Text steps are untouched by this change: same operands, same
    signature, so the 12B/26B/31B text graphs are the ones already in the
    persistent compilation cache."""
    src, fn = _fn(CM, TEXT_PRIMER)
    body = ast.get_source_segment(src, fn)
    assert "inputs_embeds=None" in body


def test_the_non_ple_variants_have_no_id_track_to_get_wrong():
    """26B/31B/12B set `hidden_size_per_layer_input == 0`: the flax PLE
    compute returns None before it looks at input_ids at all, so the extra
    operand cannot move a number for them. (The 12B checkpoint header in
    tests/fixtures carries zero `per_layer` tensors.)"""
    src, fn = _fn(FLAX_GEMMA4, "compute_per_layer_inputs")
    body = ast.get_source_segment(src, fn)
    guard = body.index("hidden_size_per_layer_input == 0")
    uses_ids = body.index("input_ids is None")
    assert guard < uses_ids, (
        "the PLE-disabled early return must come before anything reads "
        "input_ids, or a non-PLE model would take a different path")
    fixture = ROOT / "tests" / "fixtures" / "gemma-4-12b-it.safetensors-header.json"
    if fixture.exists():
        import json
        header = json.loads(fixture.read_text())
        assert not [k for k in header if "per_layer" in k], (
            "the 12B checkpoint grew per-layer tensors; it is no longer a "
            "non-PLE control")


def test_the_mtp_drafter_has_no_per_layer_embeddings():
    """The drafter is not a PLE model and never receives inputs_embeds: it
    embeds the ids itself and concatenates the backbone hidden state. So the
    drafter's own jit signature is untouched by any of this."""
    src = FLAX_MTP.read_text()
    assert "per_layer" not in src, (
        "the MTP drafter grew a per-layer path; it now needs the same "
        "id-track reasoning as the backbone")
    _, fn = _method(FLAX_MTP, "Gemma4MultiTokenPredictor", "__call__")
    args = {a.arg for a in fn.args.args}
    assert "inputs_embeds" not in args, args
    assert "input_ids" in args, args


def test_the_flax_id_selection_keeps_both_rewrites():
    """Two rewrites feed the lookup, and both are load-bearing.

    `is_multimodal -> 0` is the reference's placeholder rule (transformers
    rewrites those positions to pad_token_id, which is 0); `id >=
    vocab_size_per_layer_input -> 0` is the narrower-PLE-table guard vLLM
    applies in `get_per_layer_inputs`. Passing the real ids only helps if
    both survive.
    """
    src, fn = _fn(FLAX_GEMMA4, "compute_per_layer_inputs")
    body = " ".join(ast.get_source_segment(src, fn).split())
    assert "jnp.where(is_multimodal, 0, input_ids)" in body, body
    assert ("ple_input_ids < self.vocab_size_per_layer_input, ple_input_ids, "
            "0)" in body), body
    assert "self.embed_tokens_per_layer(ple_input_ids)" in body, body


def test_the_flax_multimodal_mask_is_derived_from_the_ids():
    """The mask that redirects placeholder positions to slot 0 comes from
    `input_ids`, so it is only real once the ids are passed. If this went
    back to a constant None the id-track would be right for text and wrong
    for the image span."""
    src, fn = _method(FLAX_GEMMA4_MM, "Gemma4ForConditionalGeneration",
                      "__call__")
    body = " ".join(ast.get_source_segment(src, fn).split())
    assert "is_multimodal = (input_ids == self.image_token_id" in body, body
    assert "is_multimodal=is_multimodal" in body
