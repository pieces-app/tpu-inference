"""ALLOW_AUDIO_WEIGHT_SKIP=1 must also require audio:0 at the API.

MEASURED 2026-09-02: `eval-e2b-base` (18:27Z) and `eval-e4b-int8` (17:09Z) set
the hatch, booted clean without an audio tower, passed the readiness gate, and
then died on the first audio request:

    AssertionError: Expected number of multimodal embeddings to match number
    of input items: 1, but got len(mm_embeddings)=0

That assertion is inside the model-execute path, so it kills EngineCore: HTTP
500, container restart, and the whole benchmark arm lost (0 of 69 requests).
The lanes that also passed --limit-mm-per-prompt audio:0 (eval-e4b-base,
eval-e4b-mtp) survived and PASSed, which is what isolated the cause.

The hatch's own message promises "serve text+vision only". This makes the
serving config say it, at load, where the fix is one flag.
"""
import ast
import pathlib
import types

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
MM = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mm.py"


def _guard():
    """Compile ONLY the real helper out of the real source (the module itself
    imports vllm, which the CPU gate does not have)."""
    tree = ast.parse(MM.read_text())
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef)
               and n.name == "require_audio_disabled_at_api"), None)
    assert fn is not None, "require_audio_disabled_at_api is gone from gemma4_mm.py"
    mod = ast.Module(body=[fn], type_ignores=[])
    ns = {}
    exec(compile(ast.fix_missing_locations(mod), str(MM), "exec"), ns)
    return ns["require_audio_disabled_at_api"]


def _cfg(limits, accessor=False):
    """`limits` as vLLM actually stores it: MMDummyOptions objects with a
    .count, not ints. A dict of ints is ALSO accepted (older shape). With
    accessor=True the config exposes get_limit_per_prompt(modality) -> int,
    which is what vLLM at the pin provides (vllm/config/multimodal.py:487)."""
    mm = types.SimpleNamespace(limit_per_prompt=limits)
    if accessor and isinstance(limits, dict):
        mm.get_limit_per_prompt = lambda m: (getattr(limits[
            m], "count", limits[m]) if m in limits else 999)
    return types.SimpleNamespace(model_config=types.SimpleNamespace(
        multimodal_config=mm))


class _Opt:
    """Stand-in for vllm.config.multimodal.AudioDummyOptions: an object with a
    .count that is NEVER == 0 as an object. This is the shape that killed two
    lanes on 2026-09-02."""

    def __init__(self, count):
        self.count = count


def test_real_vllm_shape_count_zero_is_accepted():
    _guard()(AUDIO, _cfg({
        "image": _Opt(1),
        "audio": _Opt(0),
        "video": _Opt(0)
    }), "Gemma4ForConditionalGeneration")
    _guard()(AUDIO, _cfg({
        "image": _Opt(1),
        "audio": _Opt(0)
    }, accessor=True), "Gemma4ForConditionalGeneration")


def test_real_vllm_shape_count_one_is_refused():
    with pytest.raises(ValueError):
        _guard()(AUDIO, _cfg({
            "image": _Opt(1),
            "audio": _Opt(1)
        }), "Gemma4ForConditionalGeneration")
    with pytest.raises(ValueError):
        _guard()(AUDIO,
                 _cfg({
                     "image": _Opt(1),
                     "audio": _Opt(1)
                 }, accessor=True), "Gemma4ForConditionalGeneration")


AUDIO = object()  # any non-None audio_config


def test_audio_zero_is_accepted():
    _guard()(AUDIO, _cfg({
        "image": 1,
        "audio": 0,
        "video": 0
    }), "Gemma4ForConditionalGeneration")


def test_audio_one_is_refused():
    with pytest.raises(ValueError) as e:
        _guard()(AUDIO, _cfg({
            "image": 1,
            "audio": 1,
            "video": 0
        }), "Gemma4ForConditionalGeneration")
    assert "limit-mm-per-prompt" in str(e.value)


def test_unset_limits_are_refused():
    """The measured failures had no audio limit at all -- the default admits
    audio, so an absent limit must be refused, not treated as zero."""
    for limits in ({}, None, {"image": 1}):
        with pytest.raises(ValueError):
            _guard()(AUDIO, _cfg(limits), "Gemma4ForConditionalGeneration")


def test_a_checkpoint_without_audio_is_untouched():
    """26B-A4B declares audio_config: null and must not be constrained."""
    _guard()(None, _cfg({}), "Gemma4ForConditionalGeneration")
    _guard()(None, _cfg(None), "Gemma4ForConditionalGeneration")


def test_the_guard_is_actually_called_from_load_weights():
    """A helper nobody calls is not a guard (the ti #39 lesson)."""
    tree = ast.parse(MM.read_text())
    called = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "require_audio_disabled_at_api"
    ]
    assert called, "require_audio_disabled_at_api is defined but never called"
    fns = []
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and any(
                isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
                and n.func.id == "require_audio_disabled_at_api"
                for n in ast.walk(fn)):
            fns.append(fn.name)
    assert "load_weights" in fns, f"called, but not from load_weights (from {fns})"
