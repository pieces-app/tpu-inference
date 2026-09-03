"""The native Gemma-4 Unified (12B) port: what it must get right on CPU.

Constructing the real class needs a VllmConfig, a mesh and a checkpoint, so
these are (a) NUMERIC tests of the only new arithmetic (the encoder-free
vision embedder, via the pure-JAX leaf), each with a negative control, and
(b) SOURCE-STRUCTURE tests pinning the decisions that would fail silently on
hardware: audio is LOADED not skipped, both placeholder ids are merged, the
LayerNorm eps is torch's 1e-5 (not rms_norm_eps), the op order, and that
every one of the checkpoint's 677 tensors maps somewhere.
"""
import ast
import importlib.util
import json
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_unified.py"
MATH = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_unified_math.py"
LOADER = ROOT / "tpu_inference" / "models" / "common" / "model_loader.py"
FIXTURE = ROOT / "tests" / "fixtures" / "gemma-4-12b-it.safetensors-header.json"


def _math():
    pytest.importorskip("jax")
    spec = importlib.util.spec_from_file_location("_um", MATH)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)  # pure jax: no stubs, no skip
    return m


# ------------------------------------------------------------ numeric
def _np_reference(pv, pos, P, eps, mask_padding=True):
    """Independent NumPy implementation of the reference pipeline."""

    def ln(x, w, b):
        m = x.mean(-1, keepdims=True)
        v = ((x - m)**2).mean(-1, keepdims=True)
        return (x - m) / np.sqrt(v + eps) * w + b

    h = ln(pv, P["ln1_w"], P["ln1_b"])
    h = h @ P["dense_w"] + P["dense_b"]
    h = ln(h, P["ln2_w"], P["ln2_b"])
    pe = np.zeros(h.shape, np.float32)
    for ax in range(2):
        idx = np.maximum(pos[..., ax], 0)
        g = P["pos_embedding"][:, ax, :][idx]
        if mask_padding:
            g = g * (pos[..., ax] != -1)[..., None]
        pe = pe + g
    return ln(h + pe, P["pos_norm_w"], P["pos_norm_b"])


def _case(seed=0, b=2, p=6, patch_dim=12, mm=8, S=5):
    rng = np.random.default_rng(seed)
    P = dict(ln1_w=rng.normal(1, .1, patch_dim),
             ln1_b=rng.normal(0, .1, patch_dim),
             dense_w=rng.normal(0, .3, (patch_dim, mm)),
             dense_b=rng.normal(0, .1, mm),
             ln2_w=rng.normal(1, .1, mm),
             ln2_b=rng.normal(0, .1, mm),
             pos_embedding=rng.normal(0, .5, (S, 2, mm)),
             pos_norm_w=rng.normal(1, .1, mm),
             pos_norm_b=rng.normal(0, .1, mm))
    P = {k: v.astype(np.float32) for k, v in P.items()}
    pv = rng.normal(0, 1, (b, p, patch_dim)).astype(np.float32)
    pos = rng.integers(0, S, (b, p, 2)).astype(np.int32)
    pos[:, -2:, :] = -1  # two padded patches per image
    pos[0, 1, 1] = -1  # and one half-padded axis
    return P, pv, pos


def test_vision_embedder_matches_independent_reference():
    import jax.numpy as jnp
    m = _math()
    P, pv, pos = _case()
    got = np.asarray(
        m.unified_vision_embed(jnp.asarray(pv),
                               jnp.asarray(pos),
                               **{
                                   k: jnp.asarray(v)
                                   for k, v in P.items()
                               },
                               eps=1e-5))
    want = _np_reference(pv, pos, P, 1e-5)
    assert np.max(np.abs(got - want)) < 1e-4, np.max(np.abs(got - want))


def test_negative_control_padding_mask_is_load_bearing():
    """Drop the -1 mask from the reference and it must DIVERGE: proves the
    test is sensitive to the one piece of logic that is easy to get wrong."""
    import jax.numpy as jnp
    m = _math()
    P, pv, pos = _case()
    got = np.asarray(
        m.unified_vision_embed(jnp.asarray(pv),
                               jnp.asarray(pos),
                               **{
                                   k: jnp.asarray(v)
                                   for k, v in P.items()
                               },
                               eps=1e-5))
    unmasked = _np_reference(pv, pos, P, 1e-5, mask_padding=False)
    assert np.max(np.abs(got - unmasked)) > 1e-2, (
        "removing the padding mask did not change the reference -- the test "
        "cannot detect a missing mask")


def test_factorized_posemb_zero_for_fully_padded_patch():
    import jax.numpy as jnp
    m = _math()
    P, _, _ = _case()
    pos = jnp.asarray(np.array([[[-1, -1], [2, 3]]], np.int32))
    pe = np.asarray(m.factorized_posemb(jnp.asarray(P["pos_embedding"]), pos))
    assert np.all(pe[0, 0] == 0), "a fully padded patch must get a ZERO posemb"
    assert not np.all(pe[0, 1] == 0)


# ------------------------------------------------------------ structure
def test_registered_in_the_flax_registry():
    src = LOADER.read_text()
    assert '_MODEL_REGISTRY[\n        "Gemma4UnifiedForConditionalGeneration"]' in src or \
           '_MODEL_REGISTRY["Gemma4UnifiedForConditionalGeneration"]' in src, (
        "Gemma4UnifiedForConditionalGeneration is not registered -- the 12B "
        "would still fall back to torchax, and MTP would still be impossible")


def _load_weights_src():
    src = MODEL.read_text()
    tree = ast.parse(src)
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef
                      ) and c.name == "Gemma4UnifiedForConditionalGeneration":
            for f in c.body:
                if isinstance(f, ast.FunctionDef) and f.name == "load_weights":
                    return ast.get_source_segment(src, f)
    pytest.fail("load_weights not found")


def test_audio_is_loaded_not_skipped():
    body = _load_weights_src()
    assert "audio" not in body.split("skip_substrs")[1].split("]")[0].lower(
    ), ("the skip list mentions audio -- this class exists to LOAD the audio "
        "projection, the tower variant's skip is exactly the bug (ti #34)")
    for stat in (".input_max", ".input_min", ".output_max", ".output_min"):
        assert stat in body


def test_both_placeholder_ids_are_merged():
    src = MODEL.read_text()
    i = src.index("def embed_input_ids")
    j = src.index("def get_single_image_embedding")
    body = src[i:j]
    assert "self.image_token_id" in body and "self.audio_token_id" in body, (
        "embed_input_ids must merge BOTH image and audio soft tokens")


def test_layernorm_eps_is_torch_default_not_rms_eps():
    src = MODEL.read_text()
    assert "_LAYERNORM_EPS = 1e-5" in src
    i = src.index("class Gemma4UnifiedVisionEmbedder")
    j = src.index("class Gemma4UnifiedModel")
    emb = src[i:j]
    assert emb.count("epsilon=_LAYERNORM_EPS"
                     ) == 3, "three LayerNorms, all torch-default eps"
    assert "rms_norm_eps" not in emb, "rms_norm_eps is the TEXT stack's eps; wrong here"


def test_embedder_op_order():
    src = MODEL.read_text()
    i = src.index("class Gemma4UnifiedVisionEmbedder")
    j = src.index("class Gemma4UnifiedModel")
    call = src[i:j].split("def __call__")[1]
    order = [
        call.index(t)
        for t in ("self.patch_ln1(", "self.patch_dense(", "self.patch_ln2(",
                  "factorized_posemb(", "self.pos_norm(")
    ]
    assert order == sorted(
        order), "op order must be ln1 -> dense -> ln2 -> +posemb -> pos_norm"


def test_multimodal_front_end_is_not_quantized():
    src = MODEL.read_text()
    i = src.index("class Gemma4UnifiedVisionEmbedder")
    j = src.index("class Gemma4UnifiedForConditionalGeneration")
    assert src[i:j].count("quant_config=None") >= 3, (
        "vision_embedder / embed_vision / embed_audio must pass quant_config=None "
        "(the rank-3 vision death of 2026-09-01 came from quantizing these)")


def test_every_checkpoint_tensor_maps_onto_the_module_tree():
    h = json.loads(FIXTURE.read_text())
    h.pop("__metadata__", None)
    assert len(h) == 677
    prefixes = ("model.language_model.", "model.vision_embedder.",
                "model.embed_vision.", "model.embed_audio.", "lm_head.")
    skip = (".input_max", ".input_min", ".output_max", ".output_min")
    unmapped = [k for k in h if not k.startswith(prefixes)]
    assert not unmapped, f"checkpoint names outside the module tree: {unmapped[:5]}"
    assert not any(any(s in k for s in skip)
                   for k in h), "fixture has quant stats?"
    non_text = sorted(k for k in h
                      if not k.startswith("model.language_model."))
    assert non_text == [
        "model.embed_audio.embedding_projection.weight",
        "model.embed_vision.embedding_projection.weight",
        "model.vision_embedder.patch_dense.bias",
        "model.vision_embedder.patch_dense.weight",
        "model.vision_embedder.patch_ln1.bias",
        "model.vision_embedder.patch_ln1.weight",
        "model.vision_embedder.patch_ln2.bias",
        "model.vision_embedder.patch_ln2.weight",
        "model.vision_embedder.pos_embedding",
        "model.vision_embedder.pos_norm.bias",
        "model.vision_embedder.pos_norm.weight",
    ], "the 11 non-text tensors must be exactly the ones this class declares"
    assert not any("audio_tower" in k for k in h), "Unified has no audio tower"
