"""The audio-mask fallback must match the features' non-feature dims for BOTH
input ranks the model accepts.

`_parse_and_validate_audio_input` built `jnp.ones(feats.shape[:2])`, which is
(bn, T) for the documented rank-3 input and (T, 640) for the rank-2 single
item that `_process_audio_input` explicitly expands. That mask then fails the
gather `emb[i][mask[i]]` -- an engine death on a branch no probe exercised
(review 2026-09-02; the live audio canary always supplies a mask).
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
SRC = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_unified.py"


def _fallback_expr():
    tree = ast.parse(SRC.read_text())
    fn = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
              and n.name == "_parse_and_validate_audio_input")
    for node in ast.walk(fn):
        if (isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "mask" for t in node.targets)
                and isinstance(node.value, ast.Call) and ast.unparse(node.value.func).endswith("ones")):
            return ast.unparse(node.value)
    raise AssertionError("no `mask = ...ones(...)` fallback in _parse_and_validate_audio_input")


def test_fallback_mask_uses_all_but_the_feature_dim():
    expr = _fallback_expr()
    assert "shape[:-1]" in expr, f"fallback mask shape is not feats.shape[:-1]: {expr}"
    assert "shape[:2]" not in expr


@pytest.mark.parametrize("shape", [(2, 37, 640), (37, 640)])
def test_the_rule_gives_the_gatherable_shape(shape):
    """Behavioural: apply the exact rule to both ranks and run the gather the
    model performs after its own expand_dims."""
    jnp = pytest.importorskip("jax").numpy
    feats = jnp.zeros(shape, jnp.bfloat16)
    mask = jnp.ones(feats.shape[:-1], dtype=bool)
    if feats.ndim == 2:
        feats = jnp.expand_dims(feats, 0); mask = jnp.expand_dims(mask, 0)
    emb = jnp.zeros(feats.shape[:2] + (3840,), jnp.bfloat16)   # what get_audio_embedding returns
    out = [emb[i][mask[i]] for i in range(emb.shape[0])]
    assert all(o.shape == (feats.shape[1], 3840) for o in out)


def test_the_old_rule_breaks_rank_2():
    """Negative control for the property, not for the code: shape[:2] on a
    rank-2 input yields a mask the gather cannot use."""
    jnp = pytest.importorskip("jax").numpy
    feats = jnp.zeros((37, 640), jnp.bfloat16)
    mask = jnp.ones(feats.shape[:2], dtype=bool)          # the old rule
    feats = jnp.expand_dims(feats, 0); mask = jnp.expand_dims(mask, 0)
    emb = jnp.zeros((1, 37, 3840), jnp.bfloat16)
    with pytest.raises(Exception):
        _ = emb[0][mask[0]]
