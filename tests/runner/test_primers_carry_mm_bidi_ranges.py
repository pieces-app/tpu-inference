"""EVERY AttentionMetadata a primer builds must carry `mm_bidi_ranges`.

`mm_bidi_ranges` is a DATA field of the registered dataclass, so `None` vs an
array is a different pytree treedef -- a hard jit cache miss, not a value
difference. `_prepare_inputs` allocates it on EVERY step once
`mm_bidi_enabled` (images or not), so a primer that leaves it None primes a
graph the runtime can never hit.

The symptom is nasty because precompilation REPORTS SUCCESS. Request #1 then
traces and XLA-compiles a 12B/26B model inside the serving loop -- minutes of
TTFT, and a persistent-cache miss the warm bank cannot cover -- or, with
VLLM_XLA_CHECK_RECOMPILATION=1, the engine dies on the first request with
ForbidCompile.

The target primer was fixed for this; SIX other construction sites were not:
eagle3, mtp, two in the dflash helpers (the four named in the 2026-09-01
adversarial review), plus two in `_precompile_continue_decode` that the review
did not list.

The trigger needs an mm-bidi-eligible arch (Gemma-4 Unified, or any arch under
TPU_MM_BIDI_ATTENTION=force), spec decode, and `--disable_chunked_mm_input` --
which EVERY lane in the isolation harness already sets.

This is a source-structure test: exercising the primers needs a live runner,
mesh and TPU. It asserts the property over ALL sites rather than the six that
were broken, so the next primer added cannot omit it silently -- which is
exactly how these six did.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CM = ROOT / "tpu_inference" / "runner" / "compilation_manager.py"
META = ROOT / "tpu_inference" / "layers" / "common" / "attention_metadata.py"


def _attention_metadata_calls():
    """Every `AttentionMetadata(...)` construction, with its line number."""
    src = CM.read_text()
    tree = ast.parse(src)
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "AttentionMetadata"):
            kwargs = {k.arg for k in node.keywords if k.arg}
            out.append((node.lineno, kwargs))
    return sorted(out)


def test_the_collector_finds_the_call_sites():
    """A collector that silently finds nothing makes every check below
    vacuously green -- the same shape as the skip that hid the rank bug."""
    calls = _attention_metadata_calls()
    assert len(calls) >= 7, (
        f"expected at least 7 AttentionMetadata constructions in "
        f"compilation_manager.py, found {len(calls)} at "
        f"{[ln for ln, _ in calls]}")


def test_mm_bidi_ranges_is_a_data_field_of_the_dataclass():
    """The premise. If this ever becomes metadata/static, the treedef argument
    above stops holding and this whole test is measuring nothing."""
    assert "mm_bidi_ranges" in META.read_text(), (
        "mm_bidi_ranges is no longer a field of AttentionMetadata -- "
        "re-derive whether the primers still need it")


@pytest.mark.parametrize("lineno,kwargs", _attention_metadata_calls())
def test_every_primer_metadata_carries_mm_bidi_ranges(lineno, kwargs):
    assert "mm_bidi_ranges" in kwargs, (
        f"AttentionMetadata at compilation_manager.py:{lineno} omits "
        f"mm_bidi_ranges. None vs an array is a different pytree treedef, so "
        f"this primes a graph the runtime never hits: precompilation reports "
        f"success and request #1 compiles a 12B/26B model inside the serving "
        f"loop (or dies with ForbidCompile).")


def test_the_dummy_has_exactly_one_definition():
    """It lived inline in the target primer, and six sites were missed. One
    definition is what makes 'add a primer' safe by default."""
    src = CM.read_text()
    assert src.count("def _dummy_mm_bidi_ranges") == 1, (
        "there must be exactly one definition of the dummy builder")
    # No site should re-roll the array itself.
    assert src.count("np.zeros((self.runner.max_num_reqs, 2)") <= 1, (
        "a call site is building the dummy inline instead of calling "
        "_dummy_mm_bidi_ranges -- that is how the six sites drifted apart")
