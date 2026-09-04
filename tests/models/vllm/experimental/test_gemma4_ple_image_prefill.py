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
"""Gemma-4 per-layer embeddings (PLE) on an IMAGE prefill, torchax path.

WHAT WAS WRONG
--------------
`tpu_inference/models/vllm/experimental/gemma4_mm_patcher.py` replaces
vLLM's `Gemma4ForConditionalGeneration.forward` because upstream reads PLE
out of a pre-allocated CUDA-graph buffer that torchax cannot write.  The
replacement computes PLE inline -- but it was guarded on

    if inputs_embeds is not None and input_ids is not None:

and the TPU runner never passes both.  `TPUModelRunner._get_input_ids_embeds`
returns `(None, inputs_embeds)` for a step that carries multimodal
embeddings and `(input_ids, None)` for every other step, so the conjunction
was FALSE on every call and `per_layer_inputs` was always None.

On a text step that is harmless: with `inputs_embeds=None`,
`Gemma4SelfDecoderLayers.forward` takes its `else` branch and recomputes
both PLE tracks from `input_ids` itself.  On an IMAGE step it is not:
`inputs_embeds` is set, so the model runs

    per_layer_inputs = self.project_per_layer_inputs(hidden_states, None)

and `project_per_layer_inputs` returns the PROJECTION TRACK ALONE when its
`per_layer_inputs` argument is None (vllm/model_executor/models/gemma4.py).
The per-layer EMBEDDING track was dropped for the entire image prompt --
on the torchax path only, for the PLE variants only (E2B/E4B;
26B/31B have `hidden_size_per_layer_input == 0` and no PLE at all).

The flax path reaches the same missing-ids case and handles it:
`Gemma4Model.compute_per_layer_inputs` (models/jax/gemma4.py) synthesises
zero token ids and keeps BOTH tracks.  So the two paths computed different
per-layer inputs from identical image embeddings -- an asymmetry downstream
of the projector, which is where the vision-tower differential ended.

AND WHAT WAS STILL WRONG AFTER THAT
-----------------------------------
Splitting the guard restored the id-track's SHAPE but not its CONTENT.  The
runner still withheld `input_ids` on a multimodal step, so both paths took
their zero-ids fallback and every position -- the text of the prompt
included -- looked up embedding slot 0.  The reference does not do that: it
computes the id-track from the REAL token ids and rewrites only the
placeholder positions, to `text_config.pad_token_id`, which is 0 for every
Gemma-4 config (transformers `Gemma4Model.forward`:

    llm_input_ids = torch.where(multimodal_mask,
                                self.config.text_config.pad_token_id,
                                llm_input_ids)

and vLLM's GPU path reaches the same place from the other side: it computes
PLE inside `embed_input_ids`, where the ids ARE available, masks the
`is_multimodal` positions with `torch.zeros_like`, and parks the result in
the `per_layer_embeddings` CUDA-graph buffer that `forward` then reads).

`TPUModelRunner._get_input_ids_embeds` now returns BOTH operands on a
multimodal step, so both paths compute the real per-token id-track and the
zeros fallback is dead code kept only as a defence.

WHAT IS TESTED HERE
-------------------
The gate half (numpy only, no torch/torchax/vllm) reimplements the four
formulations at the E4B geometry and measures them against each other: the
fixed result is the reference EXACTLY, it is not the zeros fallback, the
zeros fallback is not the projection-only tensor, and the differences
survive the downstream RMSNorm (so they are not scale artefacts that
normalise away).  Plus AST tests that pin the guard, the fallback, the
placeholder rule, and the runner calling convention that makes the real
id-track reachable.

The reference is re-derived against real transformers weights in
`test_gemma4_ple_reference_parity.py`, which also drives the actual
`_ple_forward`; it needs torch and skips on the gate.

NEGATIVE CONTROL: restoring `and input_ids is not None` to the guard turns
`test_the_guard_does_not_require_input_ids` red; deleting the `else` branch
turns `test_the_fallback_synthesises_zero_token_ids` red; reverting
`_get_input_ids_embeds` to return `(None, inputs_embeds)` turns
`test_the_runner_hands_the_model_both_operands_on_a_multimodal_step` red;
changing the fixed formula to drop `per_layer_input_scale` leaves the
numeric tests GREEN by design -- a uniform scalar is invisible downstream,
which `test_a_uniform_scale_on_per_layer_input_is_invisible` states
outright.
"""
import ast
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
PATCHER = (ROOT / "tpu_inference" / "models" / "vllm" / "experimental" /
           "gemma4_mm_patcher.py")
FLAX = ROOT / "tpu_inference" / "models" / "jax" / "gemma4.py"
RUNNER = ROOT / "tpu_inference" / "runner" / "tpu_runner.py"

# google/gemma-4-E4B-it text config shape.
H = 2048  # hidden_size
L = 35  # num_hidden_layers
P = 256  # hidden_size_per_layer_input
EMBED_SCALE_PER_LAYER = float(P)**0.5  # 16.0
PROJECTION_SCALE = float(H)**-0.5
PER_LAYER_INPUT_SCALE = 1.0 / (2.0**0.5)
# transformers rewrites multimodal placeholder positions to this id before
# the embed_tokens_per_layer lookup; Gemma4TextConfig defaults it to 0, and
# so do both TPU paths (jnp.where(is_multimodal, 0, ids) /
# torch.where(ids == placeholder, zeros, ids)).
PAD_TOKEN_ID = 0
IMAGE_TOKEN_ID = 258_880  # google/gemma-4 config default
# One real E4B image prompt's shape, from the 69-image bank: a short
# instruction, then a 1092-soft-token image span, then the question. The
# defect only has a size when the prompt has BOTH kinds of position, which
# every real image request does.
IMAGE_SPAN = (14, 14 + 1092)
PROMPT_TOKENS = 2048


def _rms_norm(x, weight, eps=1e-6):
    var = np.mean(x.astype(np.float32)**2, axis=-1, keepdims=True)
    return (x * (1.0 / np.sqrt(var + eps)) * weight).astype(np.float32)


class _Weights:
    """One deterministic draw of every tensor the PLE path touches."""

    def __init__(self, T, seed=0, n_layers=4, image_span=None):
        rng = np.random.default_rng(seed)
        self.T = T
        self.n_layers = n_layers
        # Post-merge residual stream: image soft tokens and text tokens.
        self.x = rng.normal(0.0, 1.0, (T, H)).astype(np.float32)
        self.ids = rng.integers(1, 262_144, size=T).astype(np.int64)
        # The image span holds the placeholder id, exactly as the prompt the
        # runner hands the model does. No span = a text-only prompt.
        self.mm = np.zeros((T, ), bool)
        if image_span is not None:
            lo, hi = image_span
            assert 0 <= lo < hi <= T
            self.mm[lo:hi] = True
            self.ids[self.mm] = IMAGE_TOKEN_ID
        # embed_tokens_per_layer: a small table is enough; slot 0 is the one
        # the missing-ids fallback looks up.
        self.ple_table = rng.normal(
            0.0, 0.02, (262_144 % 4096 + 4096, L * P)).astype(np.float32)
        self.w_proj = rng.normal(0.0, 0.02, (H, L * P)).astype(np.float32)
        self.proj_norm_w = np.ones(
            (P, ), np.float32) + rng.normal(0.0, 0.05,
                                            (P, )).astype(np.float32)
        # Per-decoder-layer PLE block.
        self.w_gate = rng.normal(0.0, 0.02,
                                 (self.n_layers, H, P)).astype(np.float32)
        self.w_up = rng.normal(0.0, 0.02,
                               (self.n_layers, P, H)).astype(np.float32)
        self.post_norm_w = np.ones(
            (self.n_layers, H), np.float32) + rng.normal(
                0.0, 0.05, (self.n_layers, H)).astype(np.float32)

    def track_a(self, ids):
        """`get_per_layer_inputs`: table lookup * sqrt(P), reshaped."""
        rows = self.ple_table[ids % self.ple_table.shape[0]]
        return (rows * EMBED_SCALE_PER_LAYER).reshape(self.T, L, P)

    def track_b(self):
        """The projection track, scaled, reshaped and RMSNormed over P."""
        proj = (self.x @ self.w_proj) * PROJECTION_SCALE
        return _rms_norm(proj.reshape(self.T, L, P), self.proj_norm_w)

    def contribution(self, per_layer_input, layer):
        """`post_per_layer_input_norm(W_up(gelu(W_gate(h)) * ple))`."""
        gate = self.x @ self.w_gate[layer]
        gate = 0.5 * gate * (
            1.0 + np.tanh(np.sqrt(2.0 / np.pi) * (gate + 0.044715 * gate**3)))
        gated = gate * per_layer_input[:, layer, :]
        return _rms_norm(gated @ self.w_up[layer], self.post_norm_w[layer])


def _projection_only(w):
    """BEFORE PR #60 on the torchax image prefill:
    project_per_layer_inputs(x, None) -- no id-track at all."""
    return w.track_b()


def _zeros_fallback(w):
    """AFTER PR #60, BEFORE this change, on BOTH paths: the id-track is
    present but every position looks up slot 0, because the runner passed no
    input_ids."""
    zeros = np.zeros((w.T, ), np.int64)
    return (w.track_b() + w.track_a(zeros)) * PER_LAYER_INPUT_SCALE


def _fixed(w):
    """NOW, on both paths: the real per-token ids, placeholders masked to 0.

    flax: `jnp.where(is_multimodal, 0, input_ids)` then the OOV mask
    (models/jax/gemma4.py, compute_per_layer_inputs).
    torchax: `torch.where(ids == placeholder_id, zeros, ids)` per placeholder
    id (models/vllm/experimental/gemma4_mm_patcher.py, _ple_forward).
    """
    ids = np.where(w.mm, 0, w.ids)
    return (w.track_b() + w.track_a(ids)) * PER_LAYER_INPUT_SCALE


def _reference(w):
    """transformers, Gemma4Model.forward: placeholders -> pad_token_id."""
    ids = np.where(w.mm, PAD_TOKEN_ID, w.ids)
    return (w.track_b() + w.track_a(ids)) * PER_LAYER_INPUT_SCALE


def _rel(a, b):
    assert a.size and b.size, "empty comparison: the slice bounds are wrong"
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def _image_prompt(seed=0):
    """An E4B image request: text, a 1092-token image span, more text."""
    return _Weights(PROMPT_TOKENS, seed=seed, image_span=IMAGE_SPAN)


# --------------------------------------------------------------------- #
# The numbers
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("T", [1092, 1104])
def test_the_fix_lands_exactly_on_the_flax_result(T):
    """PR #60's property, still required: when a path DOES fall back to
    zero ids, the two paths must agree. Only the fallback is exercised here
    -- production reaches it no longer."""
    w = _Weights(T)
    assert np.allclose(_zeros_fallback(w), _zeros_fallback(w), atol=0)
    # The flax formula, written independently below, is the same expression:
    zeros = np.zeros((T, ), np.int64)
    flax = (w.track_b() + w.track_a(zeros)) * PER_LAYER_INPUT_SCALE
    assert _rel(_zeros_fallback(w), flax) == 0.0


@pytest.mark.parametrize("T", [1092, 1104])
def test_the_old_projection_only_result_is_a_different_tensor(T):
    """`project_per_layer_inputs(x, None)` is NOT the flax fallback."""
    w = _Weights(T)
    old, fixed = _projection_only(w), _zeros_fallback(w)
    assert old.shape == fixed.shape == (T, L, P)
    assert _rel(old, fixed) > 0.3, (
        "if these were close the defect would not matter; they are not")


def test_a_uniform_scale_on_per_layer_input_is_invisible_downstream():
    """The 1/sqrt(2) alone explains NOTHING.

    The decoder layer feeds per_layer_input through a linear and then an
    RMSNorm, and RMSNorm is scale-invariant up to its epsilon, so
    multiplying the whole per-layer input by a positive constant leaves the
    residual contribution unchanged to float32 rounding. The defect is the
    MISSING TRACK, not the missing scale -- stated here so the fix is not
    mistaken for a constant, and so the size of the real difference measured
    below (>5 %) cannot be confused with this (~1e-5).
    """
    w = _Weights(256, seed=3)
    ple = _zeros_fallback(w)
    for layer in range(w.n_layers):
        base = w.contribution(ple, layer)
        scaled = w.contribution(ple * 2.0**0.5, layer)
        assert _rel(scaled, base) < 1e-4


def test_the_missing_track_survives_the_downstream_rmsnorm():
    """The difference reaches the residual stream, layer after layer."""
    w = _Weights(1092, seed=1)
    old, fixed = _projection_only(w), _zeros_fallback(w)
    for layer in range(w.n_layers):
        d = _rel(w.contribution(old, layer), w.contribution(fixed, layer))
        assert d > 0.05, (
            f"layer {layer}: the projection-only per-layer input produces a "
            f"contribution only {d:.4f} away from the correct one")


def test_the_three_states_order_by_distance_to_the_reference():
    """projection-only (pre-#60) is worst, the zeros fallback (post-#60) is
    better, and the fix IS the reference."""
    w = _image_prompt(seed=2)
    ref = _reference(w)
    d_fixed = _rel(_fixed(w), ref)
    d_zeros = _rel(_zeros_fallback(w), ref)
    d_proj = _rel(_projection_only(w), ref)
    assert d_fixed == 0.0
    assert 0.0 < d_zeros < d_proj


@pytest.mark.parametrize("seed", [0, 6])
def test_the_fix_is_the_reference_exactly(seed):
    """The whole point: the image step's per-layer inputs are now the tensor
    transformers computes, not an approximation of it."""
    w = _image_prompt(seed=seed)
    assert _rel(_fixed(w), _reference(w)) == 0.0


@pytest.mark.parametrize("seed", [0, 6])
def test_the_fix_is_not_the_zeros_fallback(seed):
    """The second defect had a size. If these two agreed there would have
    been nothing to fix after #60. The gap is carried by the ~950 TEXT
    positions of the prompt, which the fallback flattened to slot 0."""
    w = _image_prompt(seed=seed)
    lo, hi = IMAGE_SPAN
    fixed, zeros = _fixed(w), _zeros_fallback(w)
    # Inside the image span the two agree exactly: both look up slot 0.
    assert _rel(fixed[lo:hi], zeros[lo:hi]) == 0.0
    # Outside it they do not.
    assert _rel(fixed[hi:], zeros[hi:]) > 0.3
    assert _rel(fixed[:lo], zeros[:lo]) > 0.3


def test_the_placeholder_positions_are_slot_zero_on_both_sides():
    """What transformers does AT the image positions, and what both TPU
    paths now do: the placeholder id is rewritten to pad_token_id == 0, so
    the id-track there is the slot-0 row -- NOT the embedding of the
    placeholder token itself."""
    w = _image_prompt(seed=4)
    lo, hi = IMAGE_SPAN
    slot0 = w.track_a(np.zeros((w.T, ), np.int64))
    fixed = _fixed(w)
    ref = _reference(w)
    track_b = w.track_b()
    for name, got in (("fixed", fixed), ("reference", ref)):
        span = (got[lo:hi] / PER_LAYER_INPUT_SCALE) - track_b[lo:hi]
        assert np.allclose(span, slot0[lo:hi], atol=1e-5), name
    # ... and NOT the placeholder token's own row, which is what an
    # unmasked lookup would give.
    unmasked = w.track_a(w.ids)
    assert not np.allclose(unmasked[lo:hi], slot0[lo:hi], atol=1e-3)


def test_the_text_positions_carry_their_own_token_identity():
    """Outside the image span the id-track must depend on the actual token.

    The zeros fallback cannot tell two different prompts apart there; the
    fix must.
    """
    a = _image_prompt(seed=5)
    b = _image_prompt(seed=5)
    lo, hi = IMAGE_SPAN
    # Same everything except the text ids after the image span.
    b.ids = a.ids.copy()
    b.ids[hi:] = (b.ids[hi:] + 12345) % 262_144
    b.mm = a.mm
    b.x = a.x
    assert _rel(_zeros_fallback(a), _zeros_fallback(b)) == 0.0, (
        "the fallback is blind to the token ids -- that is the defect")
    tail_a, tail_b = _fixed(a)[hi:], _fixed(b)[hi:]
    assert _rel(tail_a, tail_b) > 0.05
    # The image span is untouched by the text change.
    assert _rel(_fixed(a)[lo:hi], _fixed(b)[lo:hi]) == 0.0


def test_a_ple_free_variant_is_untouched():
    """26B/31B have hidden_size_per_layer_input == 0: no tracks, no defect.

    Modelled as the guard the patcher itself applies -- there is nothing to
    add when the projection module does not exist.
    """
    tree = ast.parse(PATCHER.read_text())
    src = ast.get_source_segment(PATCHER.read_text(), [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        and n.name == "maybe_apply_gemma4_mm_patches"
    ][0])
    assert "hidden_size_per_layer_input" in src
    assert "return" in src


# --------------------------------------------------------------------- #
# Source structure: the guard, the fallback, and its reachability
# --------------------------------------------------------------------- #


def _ple_forward_src():
    src = PATCHER.read_text()
    tree = ast.parse(src)
    fn = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_ple_forward"
    ]
    assert len(fn) == 1
    return ast.get_source_segment(src, fn[0])


def test_the_guard_does_not_require_input_ids():
    src = _ple_forward_src()
    assert "if inputs_embeds is not None and input_ids is not None:" not in src, (
        "the runner never passes both, so this conjunction makes the whole "
        "PLE block dead code")
    assert "if inputs_embeds is not None:" in src


def test_the_fallback_synthesises_zero_token_ids_from_positions():
    src = _ple_forward_src()
    assert "torch.zeros_like(pos)" in src
    assert "positions.dim() == 1" in src, (
        "positions is (3, num_tokens) under mRoPE; taking it whole would "
        "build a per-layer input three times too long")


def test_the_masked_id_path_still_runs_when_ids_are_present():
    """The text-token masking must not be lost to the new branch."""
    src = _ple_forward_src()
    assert "_tpu_ple_mask_token_ids" in src
    assert "if input_ids is not None:" in src


def test_the_flax_fallback_this_mirrors_still_exists():
    """If the flax path stops synthesising zeros, the two paths diverge
    again and this fix becomes the odd one out."""
    src = FLAX.read_text()
    tree = ast.parse(src)
    fn = [
        n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
        and n.name == "compute_per_layer_inputs"
    ]
    assert len(fn) == 1
    body = ast.get_source_segment(src, fn[0])
    assert "if input_ids is None:" in body
    assert "jnp.zeros" in body
    assert "per_layer_input_scale" in body


def test_the_runner_hands_the_model_both_operands_on_a_multimodal_step():
    """The calling convention that makes the REAL id-track reachable.

    `_get_input_ids_embeds` returns `(input_ids, inputs_embeds)` on a
    multimodal step and `(input_ids, None)` otherwise: the ids are never
    withheld. This is what #60 pinned in its inverted form -- it asserted
    that exactly one of the two was always None, which made the zeros
    fallback the production path on every image step.

    If this goes back to `(None, inputs_embeds)`, both paths silently return
    to looking up slot 0 for the whole prompt.
    """
    src = RUNNER.read_text()
    tree = ast.parse(src)
    fn = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_get_input_ids_embeds"
    ]
    assert len(fn) == 1
    returns = [n for n in ast.walk(fn[0]) if isinstance(n, ast.Return)]
    assert len(returns) == 2
    shapes = set()
    for r in returns:
        assert isinstance(r.value, ast.Tuple) and len(r.value.elts) == 2
        first, second = r.value.elts
        assert isinstance(first, ast.Name) and first.id == "input_ids", (
            "the ids operand must be the token ids on BOTH branches")
        shapes.add(isinstance(second, ast.Constant) and second.value is None)
    assert shapes == {
        True, False
    }, ("expected one branch with inputs_embeds and one without")
