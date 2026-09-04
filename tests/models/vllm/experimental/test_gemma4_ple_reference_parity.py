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
"""Gemma-4 PLE on a multimodal step, measured against TRANSFORMERS itself.

`test_gemma4_ple_image_prefill.py` reimplements the arithmetic in numpy so
the CPU gate (jax + numpy only) can measure it.  This file does the other
half: it builds a real, tiny `Gemma4TextModel` from transformers, runs the
fork's ACTUAL `_ple_forward` against it, and asserts the per-layer inputs
handed to the decoder stack are the tensor transformers computes -- bit for
bit, not "close".

WHAT THE REFERENCE DOES AT PLACEHOLDER POSITIONS
------------------------------------------------
transformers `Gemma4Model.forward`
(transformers/models/gemma4/modeling_gemma4.py, the `per_layer_inputs`
block just after `get_placeholder_mask`):

    image_mask, video_mask, audio_mask = self.get_placeholder_mask(...)
    multimodal_mask = image_mask | video_mask | audio_mask
    llm_input_ids = torch.where(multimodal_mask,
                                self.config.text_config.pad_token_id,
                                llm_input_ids)
    ...
    per_layer_inputs = self.language_model.get_per_layer_inputs(
        llm_input_ids, llm_inputs_embeds)

So the id-track is looked up from the REAL token ids everywhere, and the
image/video/audio spans are rewritten to `pad_token_id` -- which
`Gemma4TextConfig` defaults to 0 -- NOT to the placeholder token's own row,
and NOT to zeros for the whole prompt.  `test_the_reference_rewrites_only_
the_placeholder_span_to_pad` pins both halves of that against the live
transformers class, and `test_pad_token_id_is_zero_so_masking_to_zero_is_
the_same_rule` pins the constant the two TPU paths hard-code.

vLLM's GPU path lands in the same place from the other side: it computes
PLE inside `embed_input_ids` (where the ids are always available), masks
the `is_multimodal` positions with `torch.zeros_like`, and parks the result
in the `per_layer_embeddings` CUDA-graph buffer that `forward` reads.

NEGATIVE CONTROLS, each mutation applied to `gemma4_mm_patcher.py` and
reverted:
  * `ple_input_ids = torch.zeros_like(input_ids)` -- the pre-fix behaviour,
    ids present but ignored                                 -> 2 failed
  * the `_tpu_ple_mask_token_ids` loop made a no-op, so the image span
    looks up the placeholder token's own row                -> 2 failed

SKIPS on the CPU gate: needs torch + transformers, which the gate image
(jax + flax + numpy) does not install.
"""
import pathlib
import sys
import types

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

from transformers.models.gemma4.configuration_gemma4 import \
    Gemma4TextConfig  # noqa: E402
from transformers.models.gemma4.modeling_gemma4 import \
    Gemma4TextModel  # noqa: E402

ROOT = pathlib.Path(__file__).resolve().parents[4]
PATCHER = (ROOT / "tpu_inference" / "models" / "vllm" / "experimental" /
           "gemma4_mm_patcher.py")

# Tiny E4B-shaped text config: PLE active, everything else small enough to
# construct in a second on CPU.
V = 512  # vocab_size (and vocab_size_per_layer_input)
H = 32  # hidden_size
L = 3  # num_hidden_layers
P = 8  # hidden_size_per_layer_input
T = 24  # prompt length
IMAGE_TOKEN_ID = 300  # in-vocab ON PURPOSE: see the test that says why
IMAGE_SPAN = (6, 18)


def _load_patcher():
    """Compile `gemma4_mm_patcher.py` with stand-ins for its vllm imports.

    Compiled from source rather than imported through
    `spec_from_file_location`: the bytecode cache is keyed on (mtime, size),
    and two equal-length edits inside one filesystem-timestamp tick hand
    back the STALE .pyc -- which read GREEN for a negative control while
    PR #60's tests were being written.
    """
    saved = {k: sys.modules.get(k) for k in _STUB_NAMES}
    try:
        for name, attrs in _STUBS:
            module = types.ModuleType(name)
            for key, value in attrs.items():
                setattr(module, key, value)
            sys.modules[name] = module
        module = types.ModuleType("_gemma4_mm_patcher_under_test")
        module.__file__ = str(PATCHER)
        exec(compile(PATCHER.read_text(), str(PATCHER), "exec"),
             module.__dict__)
        return module
    finally:
        for key, value in saved.items():
            if value is None:
                sys.modules.pop(key, None)
            else:
                sys.modules[key] = value


class _UpstreamBase:
    """Stands in for vLLM's Gemma4ForConditionalGeneration."""


_STUBS = [
    ("vllm", {}),
    ("vllm.model_executor", {}),
    ("vllm.model_executor.models", {}),
    ("vllm.model_executor.models.gemma4_mm", {
        "Gemma4ForConditionalGeneration": _UpstreamBase
    }),
    ("vllm.sequence", {
        "IntermediateTensors": object
    }),
    ("tpu_inference", {}),
    ("tpu_inference.logger", {
        "init_logger":
        lambda name: types.SimpleNamespace(info=lambda *a, **k: None)
    }),
    ("tpu_inference.models", {}),
    ("tpu_inference.models.vllm", {}),
    ("tpu_inference.models.vllm.experimental", {}),
    ("tpu_inference.models.vllm.experimental.gemma4_vision_attention", {
        "maybe_apply_gemma4_vision_attention_patch": lambda model: None
    }),
]
_STUB_NAMES = [name for name, _ in _STUBS]


@pytest.fixture(scope="module")
def patcher():
    return _load_patcher()


@pytest.fixture(scope="module")
def reference():
    """A tiny transformers Gemma4 text model plus one image prompt."""
    torch.manual_seed(0)
    config = Gemma4TextConfig(
        vocab_size=V,
        vocab_size_per_layer_input=V,
        hidden_size=H,
        intermediate_size=4 * H,
        num_hidden_layers=L,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=16,
        hidden_size_per_layer_input=P,
    )
    model = Gemma4TextModel(config).eval()
    lo, hi = IMAGE_SPAN
    ids = torch.randint(1, V, (T, ))
    is_mm = torch.zeros(T, dtype=torch.bool)
    is_mm[lo:hi] = True
    ids[is_mm] = IMAGE_TOKEN_ID
    # The post-merge residual stream: the runner's `inputs_embeds`, with the
    # vision encoder's output already scattered into the image span.
    inputs_embeds = torch.randn(T, H)
    return types.SimpleNamespace(config=config,
                                 model=model,
                                 ids=ids,
                                 is_mm=is_mm,
                                 inputs_embeds=inputs_embeds)


def _reference_id_track(ref):
    """transformers `Gemma4Model.forward`, the two lines that matter."""
    llm_input_ids = torch.where(ref.is_mm, ref.config.pad_token_id, ref.ids)
    with torch.no_grad():
        return ref.model.get_per_layer_inputs(llm_input_ids, None)


def _run_ple_forward(patcher, ref, input_ids):
    """Drive the fork's real `_ple_forward` and capture what it hands down.

    The language model is a shim that reproduces vLLM's
    `Gemma4SelfDecoderLayers.get_per_layer_inputs`
    (vllm/model_executor/models/gemma4.py) over the transformers weights:
    the same out-of-vocab mask, the same scaled lookup, the same reshape.
    """
    captured = {}

    class _LanguageModelInner:

        def get_per_layer_inputs(self, input_ids):
            mask = torch.logical_and(
                input_ids >= 0, input_ids
                < ref.config.vocab_size_per_layer_input)
            tokens = torch.where(mask, input_ids, torch.zeros_like(input_ids))
            return ref.model.embed_tokens_per_layer(tokens).reshape(
                *input_ids.shape, L, P)

        def __call__(self, ids, positions, per_layer_inputs=None, **kwargs):
            captured["per_layer_inputs"] = per_layer_inputs
            return torch.zeros(1)

    fake_self = types.SimpleNamespace(
        language_model=types.SimpleNamespace(model=_LanguageModelInner()),
        config=types.SimpleNamespace(text_config=types.SimpleNamespace(
            num_hidden_layers=L, hidden_size_per_layer_input=P)),
        _tpu_ple_mask_token_ids=(IMAGE_TOKEN_ID, ),
        _clear_mm_prefix_for_full_attn_layers=lambda: None,
    )
    with torch.no_grad():
        patcher._ple_forward(fake_self, input_ids, torch.arange(T), None,
                             ref.inputs_embeds)
    assert "per_layer_inputs" in captured, "the forward was never reached"
    return captured["per_layer_inputs"]


# --------------------------------------------------------------------- #
# What the reference does
# --------------------------------------------------------------------- #


def test_pad_token_id_is_zero_so_masking_to_zero_is_the_same_rule():
    """Both TPU paths mask placeholder positions to literal 0. That is only
    the reference's rule because Gemma-4's pad_token_id IS 0; if a config
    ever moved it, `jnp.where(is_multimodal, 0, ids)` would stop matching.
    """
    assert Gemma4TextConfig().pad_token_id == 0


def test_the_reference_rewrites_only_the_placeholder_span_to_pad(reference):
    """The id-track at image positions is the PAD row -- not the placeholder
    token's own row, and not zeros everywhere."""
    lo, hi = IMAGE_SPAN
    ref_track = _reference_id_track(reference)
    with torch.no_grad():
        pad_row = reference.model.get_per_layer_inputs(
            torch.zeros(T, dtype=torch.long), None)
        placeholder_row = reference.model.get_per_layer_inputs(
            reference.ids, None)
        real_row = reference.model.get_per_layer_inputs(reference.ids, None)

    # inside the span: the pad row
    assert torch.equal(ref_track[lo:hi], pad_row[lo:hi])
    # the placeholder id is IN vocab here, so an unmasked lookup would have
    # returned a different (and wrong) row -- this is what the mask buys.
    assert not torch.allclose(placeholder_row[lo:hi], pad_row[lo:hi])
    # outside the span: each token's own row, NOT the pad row
    assert torch.equal(ref_track[:lo], real_row[:lo])
    assert torch.equal(ref_track[hi:], real_row[hi:])
    assert not torch.allclose(ref_track[hi:], pad_row[hi:])


# --------------------------------------------------------------------- #
# The torchax path, driven for real
# --------------------------------------------------------------------- #


def test_the_real_ids_path_is_the_reference(patcher, reference):
    """`_ple_forward` with the ids the runner now passes reproduces
    transformers' per-layer id-track exactly."""
    got = _run_ple_forward(patcher, reference, reference.ids)
    assert torch.equal(got, _reference_id_track(reference))


def test_the_combined_per_layer_input_is_the_reference(patcher, reference):
    """... and so is the tensor after the projection track is folded in,
    which is what the decoder layers actually consume."""
    got = _run_ple_forward(patcher, reference, reference.ids)
    with torch.no_grad():
        got_full = reference.model.project_per_layer_inputs(
            reference.inputs_embeds, got)
        ref_full = reference.model.project_per_layer_inputs(
            reference.inputs_embeds, _reference_id_track(reference))
    assert torch.equal(got_full, ref_full)


def test_the_zero_ids_fallback_is_not_the_reference(patcher, reference):
    """The state of every image step before this fix: no ids, so every
    position reads the pad row. It differs from the reference on the TEXT
    positions and agrees on the image span -- the exact shape of the defect.
    """
    lo, hi = IMAGE_SPAN
    fallback = _run_ple_forward(patcher, reference, None)
    ref_track = _reference_id_track(reference)
    assert torch.equal(fallback[lo:hi], ref_track[lo:hi])
    assert not torch.allclose(fallback[:lo], ref_track[:lo])
    assert not torch.allclose(fallback[hi:], ref_track[hi:])


def test_a_text_step_never_enters_the_block(patcher, reference):
    """With `inputs_embeds=None` the guard is false and PLE stays None: the
    language model recomputes both tracks from input_ids itself. This is why
    text steps are byte-identical across all three states of this code."""
    captured = {}

    class _LanguageModelInner:

        def __call__(self, ids, positions, per_layer_inputs=None, **kwargs):
            captured["per_layer_inputs"] = per_layer_inputs
            return torch.zeros(1)

        def get_per_layer_inputs(self, ids):  # pragma: no cover - must not run
            raise AssertionError("PLE computed on a text step")

    fake_self = types.SimpleNamespace(
        language_model=types.SimpleNamespace(model=_LanguageModelInner()),
        config=types.SimpleNamespace(text_config=types.SimpleNamespace(
            num_hidden_layers=L, hidden_size_per_layer_input=P)),
        _tpu_ple_mask_token_ids=(IMAGE_TOKEN_ID, ),
        _clear_mm_prefix_for_full_attn_layers=lambda: None,
    )
    with torch.no_grad():
        patcher._ple_forward(fake_self, reference.ids, torch.arange(T), None,
                             None)
    assert captured["per_layer_inputs"] is None


def test_mrope_positions_do_not_change_the_fallback_length(patcher, reference):
    """positions is (3, T) under mRoPE; the fallback takes row 0, so the
    zeros it builds are still T long."""
    captured = {}

    class _LanguageModelInner:

        def get_per_layer_inputs(self, input_ids):
            assert input_ids.shape == (T, ), input_ids.shape
            return ref_lookup(input_ids)

        def __call__(self, ids, positions, per_layer_inputs=None, **kwargs):
            captured["per_layer_inputs"] = per_layer_inputs
            return torch.zeros(1)

    def ref_lookup(input_ids):
        return reference.model.embed_tokens_per_layer(input_ids).reshape(
            *input_ids.shape, L, P)

    fake_self = types.SimpleNamespace(
        language_model=types.SimpleNamespace(model=_LanguageModelInner()),
        config=types.SimpleNamespace(text_config=types.SimpleNamespace(
            num_hidden_layers=L, hidden_size_per_layer_input=P)),
        _tpu_ple_mask_token_ids=(IMAGE_TOKEN_ID, ),
        _clear_mm_prefix_for_full_attn_layers=lambda: None,
    )
    mrope_positions = torch.arange(3 * T).reshape(3, T)
    with torch.no_grad():
        patcher._ple_forward(fake_self, None, mrope_positions, None,
                             reference.inputs_embeds)
    assert captured["per_layer_inputs"].shape == (T, L, P)


# --------------------------------------------------------------------- #
# The flax path's id selection, against the same oracle
# --------------------------------------------------------------------- #


def _flax_ple_ids(input_ids, is_multimodal, vocab_size_per_layer_input):
    """`Gemma4Model.compute_per_layer_inputs`' id selection, in numpy.

    Kept in step with the source by
    `test_gemma4_ple_image_prefill.py::test_the_flax_fallback_this_mirrors_
    still_exists` and by the AST pins in
    `tests/runner/test_mm_step_model_fn_operands.py`.
    """
    import numpy as np
    ids = np.where(is_multimodal, 0, input_ids)
    return np.where(ids < vocab_size_per_layer_input, ids, 0)


def test_the_flax_id_selection_reproduces_the_reference_ids(reference):
    """Different expression, same tensor: `jnp.where(is_multimodal, 0, ids)`
    is `torch.where(multimodal_mask, pad_token_id, ids)` when pad is 0."""
    flax_ids = _flax_ple_ids(reference.ids.numpy(), reference.is_mm.numpy(),
                             reference.config.vocab_size_per_layer_input)
    tf_ids = torch.where(reference.is_mm, reference.config.pad_token_id,
                         reference.ids)
    assert torch.equal(torch.from_numpy(flax_ids).to(tf_ids.dtype), tf_ids)


def test_the_flax_id_selection_yields_the_reference_track(reference):
    """And feeding those ids through the reference lookup gives the
    reference id-track, so the two TPU paths agree with each other too."""
    flax_ids = torch.from_numpy(
        _flax_ple_ids(reference.ids.numpy(), reference.is_mm.numpy(),
                      reference.config.vocab_size_per_layer_input)).long()
    with torch.no_grad():
        got = reference.model.get_per_layer_inputs(flax_ids, None)
    assert torch.equal(got, _reference_id_track(reference))
