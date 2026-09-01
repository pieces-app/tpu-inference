"""T1: the Gemma4-MTP drafter must suppress the DRAFT checkpoint's
generation_config.suppress_tokens (issue #155/#158 lineage; plan M0a).

Why this matters: `google/gemma-4-12B-it-assistant` ships
suppress_tokens=[258883, 258882] (eoa/eoi placeholders; VERIFIED against the
downloaded artifact 2026-09-01) while the 26B assistant ships none. The torch
drafter suppresses; this flax drafter did not, so a drafted placeholder id
would corrupt subsequent draft steps and reach target verification with no
embedding behind it.

Structure over import: the model module pulls vllm/torchax, so these tests
exercise the SUPPRESSION LOGIC by source-pinned reimplementation plus AST
assertions that the shipped code carries that logic at the right points.
The dtype-min/argmax semantics are checked numerically on jax[cpu]; the
placement (both compute_logits branches, after softcap/scatter) is checked
structurally -- the pair is what makes this non-vacuous.

Negative control: revert the suppression hunks -> the AST tests go red
(watched). `pytest tests/models/jax/test_gemma4_mtp_suppress.py`
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
MTP = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mtp.py"
EAGLE3 = ROOT / "tpu_inference" / "spec_decode" / "jax" / "eagle3.py"

def _jnp():
    """jax only where numerics need it -- the AST/structure tests must run
    (and be able to FAIL) on any interpreter, including a bare gate image."""
    return pytest.importorskip("jax.numpy", reason="numeric check needs jax")


def _fn(tree, name, cls=None):
    for node in ast.walk(tree):
        if cls and isinstance(node, ast.ClassDef) and node.name == cls:
            for sub in node.body:
                if isinstance(sub, ast.FunctionDef) and sub.name == name:
                    return sub
        if cls is None and isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    return None


# ---------------------------------------------------------------- numerics
def _suppress(logits, ids):
    """The shipped _apply_suppress semantics, mirrored for numeric check."""
    jnp = _jnp()
    if not ids:
        return logits
    sup = jnp.asarray(ids, dtype=jnp.int32)
    return logits.at[:, sup].set(jnp.finfo(logits.dtype).min)


def test_suppressed_columns_are_dtype_min_and_never_argmax():
    jnp = _jnp()
    # Craft logits whose UNsuppressed argmax is a suppressed id.
    logits = jnp.zeros((2, 16), dtype=jnp.float32).at[:, 7].set(50.0)
    out = _suppress(logits, (7, 9))
    assert float(out[0, 7]) == float(jnp.finfo(jnp.float32).min)
    assert float(out[0, 9]) == float(jnp.finfo(jnp.float32).min)
    assert int(jnp.argmax(out[0])) != 7, "argmax still lands on a suppressed id"
    assert int(jnp.argmax(logits[0])) == 7, "control: unsuppressed argmax IS 7"


def test_empty_set_is_identity():
    jnp = _jnp()
    logits = jnp.arange(32, dtype=jnp.float32).reshape(2, 16)
    assert bool(jnp.array_equal(_suppress(logits, ()), logits))


def test_sparse_path_masks_selected_ids():
    jnp = _jnp()
    # indices [batch, k] with a suppressed id winning on score
    indices = jnp.asarray([[3, 7, 11]], dtype=jnp.int32)
    logits = jnp.asarray([[1.0, 9.0, 2.0]], dtype=jnp.float32)
    sup = jnp.asarray((7, ), dtype=jnp.int32)
    hit = (indices[..., None] == sup[None, None, :]).any(axis=-1)
    masked = jnp.where(hit, jnp.finfo(logits.dtype).min, logits)
    best = jnp.argmax(masked, axis=-1, keepdims=True)
    tok = int(jnp.take_along_axis(indices, best, axis=-1).squeeze(-1)[0])
    assert tok != 7 and tok == 11


# ---------------------------------------------------------------- structure
def test_init_parses_and_validates_suppress_tokens():
    src = MTP.read_text()
    tree = ast.parse(src)
    init = _fn(tree, "__init__", cls="Gemma4MTPForCausalLM")
    assert init is not None
    body = ast.get_source_segment(src, init) or ""
    assert "try_get_generation_config" in body, (
        "__init__ must read the DRAFT checkpoint's generation_config")
    assert "suppress_tokens" in body
    assert "_suppress_token_ids" in body
    assert body.count("raise ValueError") >= 2, (
        "malformed and out-of-vocab ids must both refuse (fail-closed)")
    assert "vocab_size" in body, "ids must be range-checked against the vocab"


def test_both_compute_logits_branches_suppress():
    src = MTP.read_text()
    tree = ast.parse(src)
    cl = _fn(tree, "compute_logits", cls="Gemma4MTPForCausalLM")
    assert cl is not None
    body = ast.get_source_segment(src, cl) or ""
    # centroid branch (early return) and dense branch (final return)
    returns = [n for n in ast.walk(cl) if isinstance(n, ast.Return)]
    assert len(returns) >= 2, "expected the centroid and dense returns"
    assert body.count("_apply_suppress") >= 2, (
        "BOTH the masked_embedding branch and the dense branch must apply "
        "suppression -- a suppressed id can be centroid-selected")


def test_suppression_is_applied_after_softcapping():
    src = MTP.read_text()
    cl = _fn(ast.parse(src), "compute_logits", cls="Gemma4MTPForCausalLM")
    body = ast.get_source_segment(src, cl) or ""
    cap = body.find("final_logit_softcapping")
    sup = body.rfind("_apply_suppress")
    assert cap != -1 and sup > cap, (
        "dtype-min must be written AFTER softcapping (torch applies "
        "suppression after the logits processor; routing -inf through tanh "
        "would change the value)")


def test_apply_suppress_uses_static_empty_branch():
    """The empty-set path must be a Python-level `if`, so the no-suppress
    HLO stays byte-identical and existing 26B/31B warm banks are untouched."""
    src = MTP.read_text()
    fn = _fn(ast.parse(src), "_apply_suppress", cls="Gemma4MTPForCausalLM")
    assert fn is not None, "_apply_suppress helper missing"
    ifs = [n for n in ast.walk(fn) if isinstance(n, ast.If)]
    assert ifs, "expected a static `if self._suppress_token_ids:`"
    assert any(isinstance(n.test, ast.Attribute)
               and n.test.attr == "_suppress_token_ids" for n in ifs), (
        "the guard must be the plain attribute (a traced/jnp condition would "
        "bake the masking into every graph)")


def test_no_hardcoded_placeholder_ids_anywhere():
    """The mechanism is generic: ids live in checkpoint configs only."""
    for p in (MTP, EAGLE3):
        src = p.read_text()
        code = "\n".join(l for l in src.splitlines()
                         if not l.strip().startswith("#"))
        for bad in ("258882", "258883"):
            assert bad not in code, f"{p.name} hardcodes {bad}"


def test_eagle3_gemma4_mtp_embedding_share_is_fail_closed():
    src = EAGLE3.read_text()
    tree = ast.parse(src)
    raises = [n for n in ast.walk(tree)
              if isinstance(n, ast.Raise) and "embed" in
              (ast.get_source_segment(src, n) or "").lower()]
    assert raises, (
        "a missing target-embedding share for Gemma4-MTP must RAISE, not "
        "warn -- the drafter is unusable without it")
