"""Dropping a checkpoint's audio tower must be REFUSED, not defaulted into.

`Gemma4ForConditionalGeneration` (the native flax_nnx path) has no audio tower,
and its loader passes
    skip_substrs=["audio_tower", "embed_audio", ...]

That is CORRECT for a checkpoint with no audio -- `gemma-4-26B-A4B-it` declares
`audio_config: null` and ships zero audio tensors.

It is SILENTLY WRONG for one that has audio. VERIFIED against the staged
artifact 2026-09-01: `google/gemma-4-E4B-it` declares
`audio_config: gemma4_audio` and carries a real encoder --

    744  model.audio_tower.layers
      5  model.audio_tower.subsample_conv_projection
      2  model.audio_tower.output_proj
      1  model.embed_audio.embedding_projection
    ---
    752  tensors

-- and the static skip list drops all 752. The model then LOADS CLEAN, serves
text and vision correctly, and mishandles audio with no error, no warning and
no failed gate. An entire modality discarded by a list literal.

So the loader now refuses when the config CLAIMS audio and this class cannot
serve it. ALLOW_AUDIO_WEIGHT_SKIP=1 is the escape hatch, because text+vision
from an audio-capable checkpoint is a legitimate thing to want -- but it has to
be SAID.

Source-structure test: constructing the real model needs a live mesh and a
checkpoint, which CPU-only jax cannot provide.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
MM = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mm.py"
ENVS = ROOT / "tpu_inference" / "envs.py"


def _load_weights_src():
    src = MM.read_text()
    tree = ast.parse(src)
    for cls in ast.walk(tree):
        if (isinstance(cls, ast.ClassDef)
                and cls.name == "Gemma4ForConditionalGeneration"):
            for f in cls.body:
                if isinstance(f, ast.FunctionDef) and f.name == "load_weights":
                    return ast.get_source_segment(src, f) or ""
    pytest.fail("Gemma4ForConditionalGeneration.load_weights not found")


def test_the_skip_still_exists():
    """Premise: if the class ever GROWS an audio tower, this whole guard
    should be deleted rather than left asserting something untrue."""
    assert "audio_tower" in _load_weights_src(), (
        "the audio skip is gone -- if an audio tower was implemented, delete "
        "this test with it")


def test_load_refuses_when_the_checkpoint_declares_audio():
    body = _load_weights_src()
    assert "audio_config" in body, (
        "load_weights never consults audio_config, so it cannot tell a "
        "checkpoint with no audio (26B, correct to skip) from one with 752 "
        "audio tensors (E4B, catastrophic to skip)")
    assert "raise" in body, (
        "load_weights must REFUSE when the config declares audio it cannot "
        "serve; skipping silently discards the modality")
    # the refusal must be reachable BEFORE the loader runs
    i_cfg, i_raise = body.index("audio_config"), body.index("raise")
    i_loader = body.index("JaxAutoWeightsLoader")
    assert i_cfg < i_raise < i_loader, (
        "the audio check must run BEFORE JaxAutoWeightsLoader, or the weights "
        "are already dropped by the time it fires")


def test_there_is_a_stated_escape_hatch():
    body = _load_weights_src()
    assert "ALLOW_AUDIO_WEIGHT_SKIP" in body, (
        "text+vision from an audio-capable checkpoint is legitimate; it needs "
        "an opt-in rather than being impossible")
    assert "ALLOW_AUDIO_WEIGHT_SKIP" in ENVS.read_text(), (
        "the escape hatch must be a declared env var, not an undeclared read")


def test_the_hatch_defaults_to_refusing():
    envs_src = ENVS.read_text()
    i = envs_src.index('"ALLOW_AUDIO_WEIGHT_SKIP"')
    window = envs_src[i:i + 200]
    assert "default=False" in window, (
        "ALLOW_AUDIO_WEIGHT_SKIP must default to False -- a default of True "
        "restores exactly the silent behaviour this guard exists to stop")


def _load_weights_fn_with_audio_guard():
    import ast
    tree = ast.parse(MM.read_text())
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef) and fn.name == "load_weights":
            if any(
                    isinstance(n, ast.Name) and n.id == "audio_cfg"
                    for n in ast.walk(fn)):
                return fn
    raise AssertionError("no load_weights binds audio_cfg")


def test_the_guard_is_a_real_if_with_the_right_polarity_and_a_raise():
    """AST, not substrings: the earlier test was satisfied by the comment
    that mentions audio_config above the raise (review 2026-09-02). This one
    fails on a dead guard (`audio_cfg = None`), an inverted hatch
    (`envs.ALLOW_AUDIO_WEIGHT_SKIP` without `not`), or a removed raise."""
    import ast
    fn = _load_weights_fn_with_audio_guard()
    binds = [
        n for n in ast.walk(fn) if isinstance(n, ast.Assign) and any(
            isinstance(t, ast.Name) and t.id == "audio_cfg" for t in n.targets)
    ]
    assert len(binds) == 1 and isinstance(binds[0].value, ast.Call), (
        "audio_cfg must be bound exactly once, from a call (getattr on the config), not a constant"
    )
    guards = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If) or not isinstance(
                node.test, ast.BoolOp) or not isinstance(
                    node.test.op, ast.And):
            continue
        ops = node.test.values
        has_cfg = any(
            isinstance(o, ast.Compare) and isinstance(o.left, ast.Name)
            and o.left.id == "audio_cfg" and any(
                isinstance(c, ast.IsNot) for c in o.ops) for o in ops)
        has_hatch = any(
            isinstance(o, ast.UnaryOp) and isinstance(o.op, ast.Not)
            and isinstance(o.operand, ast.Attribute)
            and o.operand.attr == "ALLOW_AUDIO_WEIGHT_SKIP" for o in ops)
        raises = any(isinstance(n, ast.Raise) for n in node.body)
        if has_cfg and has_hatch:
            guards.append(raises)
    assert guards, "no `if audio_cfg is not None and not envs.ALLOW_AUDIO_WEIGHT_SKIP:` guard in load_weights"
    assert all(guards), "the audio guard exists but its body does not raise"
