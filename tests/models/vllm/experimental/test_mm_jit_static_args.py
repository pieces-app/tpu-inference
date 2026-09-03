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
"""The static/dynamic split for a jitted multimodal submodule.

Before this, ``patch_mm_model`` handed every module named in
``JITTED_MM_MODULE_KEYS`` the same ``static_argnums=(3,)`` /
``static_argnames=("grid_thw",)`` -- Qwen3-VL's mapping. Index 3 is
``grid_thw`` only because ``Qwen3_VisionTransformer.forward(self, x,
grid_thw, *, encoder_metadata=None)`` puts it second (plus the two leading
``(params, buffers)`` arguments ``JittableModule`` prepends). On Gemma-4's
``Gemma4VisionEncoder.forward(self, inputs_embeds, attention_mask,
pixel_position_ids=None, **kwargs)`` index 3 is ``attention_mask``.

These tests pin the derivation, and the behavioural ones are the negative
control: with the old fixed mapping a Gemma-4-shaped positional call marks
``attention_mask`` static and jax rejects it; with the derived mapping the
same call traces every array and matches the un-jitted result.

Runtime note: the behavioural tests use plain ``jax.jit`` over a stand-in
that reproduces ``JittableModule``'s call shape (``functional_call(params,
buffers, *args, **kwargs)`` invoked as ``jitted(params, buffers, ...)``),
because the CPU gate has neither torch nor torchax. The last test runs the
same comparison through the real ``torchax.interop.JittableModule`` and is
skipped when torchax is not importable.
"""

import importlib.util
import inspect
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = pathlib.Path(__file__).resolve().parents[4]
SIGNATURE_SRC = (ROOT / "tpu_inference" / "models" / "vllm" / "experimental" /
                 "mm_jit_signature.py")
PATCHER_SRC = (ROOT / "tpu_inference" / "models" / "vllm" / "experimental" /
               "model_patcher.py")

# The exact dict model_patcher.py used to hard-code for every module.
OLD_FIXED_MAPPING = {
    "static_argnums": (3, ),
    "static_argnames": ("grid_thw", ),
}


def _signature_module():
    """Load mm_jit_signature.py by path: stdlib-only, no vllm/torch import."""
    spec = importlib.util.spec_from_file_location("_mm_jit_signature",
                                                  SIGNATURE_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MM = _signature_module()


# --------------------------------------------------------------- signatures
# Stand-ins that carry the real forward signatures. Only the parameter names,
# kinds and order matter to the derivation, so these are faithful without
# pulling vLLM or transformers into a CPU runner.
class _Qwen3VisionLike:
    """Mirrors vllm.model_executor.models.qwen3_vl.Qwen3_VisionTransformer."""

    def forward(self, x, grid_thw, *, encoder_metadata=None):
        return x


class _Gemma4VisionEncoderLike:
    """Mirrors transformers ...gemma4.modeling_gemma4.Gemma4VisionEncoder."""

    def forward(self,
                inputs_embeds,
                attention_mask,
                pixel_position_ids=None,
                **kwargs):
        return inputs_embeds


class _GridThwFirst:

    def forward(self, grid_thw, x):
        return x


class _GridThwKeywordOnly:

    def forward(self, x, *, grid_thw=None):
        return x


def test_qwen3_vl_signature_reproduces_the_old_fixed_mapping():
    """The derivation must be byte-identical to what Qwen3-VL got before."""
    got = MM.compute_mm_jit_static_args(_Qwen3VisionLike().forward)
    assert got == OLD_FIXED_MAPPING


def test_gemma4_vision_encoder_declares_no_static_argument():
    """Gemma-4's encoder has no grid argument, so nothing may be static."""
    assert MM.compute_mm_jit_static_args(
        _Gemma4VisionEncoderLike().forward) == {}


def test_static_index_tracks_the_position_in_the_signature():
    """A different position must give a different argnum, not Qwen's 3."""
    assert MM.compute_mm_jit_static_args(_GridThwFirst().forward) == {
        "static_argnums": (2, ),
        "static_argnames": ("grid_thw", ),
    }


def test_keyword_only_static_argument_gets_no_argnum():
    """A keyword-only parameter has no positional index to address."""
    assert MM.compute_mm_jit_static_args(_GridThwKeywordOnly().forward) == {
        "static_argnums": (),
        "static_argnames": ("grid_thw", ),
    }


def test_unintrospectable_forward_falls_back_to_no_static_args():
    assert MM.compute_mm_jit_static_args(len) == {}


def test_offset_matches_the_jittable_module_call_contract():
    """JittableModule prepends exactly (params, buffers) before *args.

    ``jittable_call`` builds ``jitted(self.params, self.buffers, *args,
    **kwargs)`` over ``functools.partial(self.functional_call, method_name)``.
    Binding the method name leaves ``(params, buffers, *args, **kwargs)``, so
    the module's first positional forward parameter is at index 2.
    """

    def functional_call(method_or_name, params, buffers, *args, **kwargs):
        return method_or_name, params, buffers, args, kwargs

    bound = inspect.signature(functional_call).parameters
    leading = [
        name for name, parameter in bound.items()
        if parameter.kind is inspect.Parameter.POSITIONAL_OR_KEYWORD
    ]
    # method_or_name is bound away by functools.partial before jitting.
    assert leading == ["method_or_name", "params", "buffers"]
    assert MM.JITTABLE_MODULE_POSITIONAL_OFFSET == len(leading) - 1


# ------------------------------------------------------------- behavioural
def _jittable_call_shape(forward):
    """A stand-in for what JittableModule hands to jax.jit.

    ``functional_call`` with the method name already bound, i.e.
    ``(params, buffers, *args, **kwargs)``, invoked as
    ``jitted(params, buffers, *args, **kwargs)``.
    """
    seen = []

    def functional_call(params, buffers, *args, **kwargs):
        seen.append(
            [type(a).__name__ for a in args] +
            [f"{k}={type(v).__name__}" for k, v in sorted(kwargs.items())])
        return forward(*args, **kwargs)

    return functional_call, seen


def _gemma4_like_forward(inputs_embeds,
                         attention_mask,
                         pixel_position_ids=None,
                         **kwargs):
    """A Gemma-4-shaped encoder body: the mask must affect the output."""
    mask = attention_mask[:, None, :].astype(inputs_embeds.dtype)
    scores = jnp.einsum("bqd,bkd->bqk", inputs_embeds, inputs_embeds)
    scores = jnp.where(mask > 0, scores, -1e9)
    weights = jax.nn.softmax(scores, axis=-1)
    out = jnp.einsum("bqk,bkd->bqd", weights, inputs_embeds)
    if pixel_position_ids is not None:
        out = out + pixel_position_ids[..., :1].astype(out.dtype) * 0.01
    return out


def test_old_fixed_mapping_marks_gemma4_attention_mask_static():
    """Negative control: Qwen's index 3 lands on Gemma-4's attention_mask."""
    functional_call, _ = _jittable_call_shape(_gemma4_like_forward)
    jitted = jax.jit(functional_call, **OLD_FIXED_MAPPING)

    rng = np.random.default_rng(0)
    embeds = jnp.asarray(rng.standard_normal((2, 4, 3)), dtype=jnp.float32)
    mask = jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.bool_)
    positions = jnp.asarray(rng.integers(0, 4, (2, 4, 2)), dtype=jnp.int32)

    with pytest.raises(ValueError, match="[Nn]on-hashable static argument"):
        jitted({}, {}, embeds, mask, positions)


def test_derived_mapping_traces_every_gemma4_array():
    """The fix: no static args, so the same positional call traces cleanly."""
    derived = MM.compute_mm_jit_static_args(_Gemma4VisionEncoderLike().forward)
    assert derived == {}

    functional_call, seen = _jittable_call_shape(_gemma4_like_forward)
    jitted = jax.jit(functional_call, **derived)

    rng = np.random.default_rng(0)
    embeds = jnp.asarray(rng.standard_normal((2, 4, 3)), dtype=jnp.float32)
    mask = jnp.asarray([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=jnp.bool_)
    positions = jnp.asarray(rng.integers(0, 4, (2, 4, 2)), dtype=jnp.int32)

    got = jitted({}, {}, embeds, mask, positions)
    assert all(name.endswith("Tracer") for name in seen[-1]), seen[-1]
    np.testing.assert_allclose(
        np.asarray(got),
        np.asarray(_gemma4_like_forward(embeds, mask, positions)),
        rtol=1e-6,
        atol=1e-6)


def test_derived_mapping_equals_the_unjitted_module_on_random_inputs():
    """Patched == unpatched for a Gemma-4-shaped encoder, keyword call.

    Keyword arguments are how vLLM's ``_process_image_input`` calls
    ``vt.encoder``; this pins that the derived (empty) mapping leaves the
    numerics alone across several random draws and shapes.
    """
    derived = MM.compute_mm_jit_static_args(_Gemma4VisionEncoderLike().forward)
    functional_call, _ = _jittable_call_shape(_gemma4_like_forward)
    jitted = jax.jit(functional_call, **derived)

    rng = np.random.default_rng(7)
    for batch, tokens, dim in ((1, 3, 2), (2, 5, 4), (3, 8, 6)):
        embeds = jnp.asarray(rng.standard_normal((batch, tokens, dim)),
                             dtype=jnp.float32)
        mask = jnp.asarray(rng.integers(0, 2, (batch, tokens)),
                           dtype=jnp.bool_)
        mask = mask.at[:, 0].set(True)  # never an all-masked row
        positions = jnp.asarray(rng.integers(0, tokens, (batch, tokens, 2)),
                                dtype=jnp.int32)

        want = _gemma4_like_forward(embeds, mask, positions)
        got = jitted({}, {},
                     inputs_embeds=embeds,
                     attention_mask=mask,
                     pixel_position_ids=positions)
        np.testing.assert_allclose(np.asarray(got),
                                   np.asarray(want),
                                   rtol=1e-6,
                                   atol=1e-6)


def test_derived_mapping_keeps_qwen_grid_thw_static():
    """Qwen3-VL must still get a concrete grid_thw inside the trace."""
    kinds = []

    def qwen_like_forward(x, grid_thw, *, encoder_metadata=None):
        kinds.append(type(grid_thw).__name__)
        # A static tuple can drive Python control flow; a tracer cannot.
        total = sum(t * h * w for t, h, w in grid_thw)
        return x * float(total)

    derived = MM.compute_mm_jit_static_args(_Qwen3VisionLike().forward)
    assert derived == OLD_FIXED_MAPPING

    functional_call, _ = _jittable_call_shape(qwen_like_forward)
    jitted = jax.jit(functional_call, **derived)

    x = jnp.ones((2, 3), dtype=jnp.float32)
    grid = ((1, 2, 2), )
    got = jitted({}, {}, x, grid)
    assert kinds == ["tuple"]
    np.testing.assert_allclose(np.asarray(got), np.full((2, 3), 4.0))

    # And as a keyword, which static_argnames covers.
    got_kw = jitted({}, {}, x, grid_thw=grid)
    np.testing.assert_allclose(np.asarray(got_kw), np.full((2, 3), 4.0))


# -------------------------------------------------- stand-in fidelity
def _positional_names(forward):
    return [
        name
        for name, parameter in inspect.signature(forward).parameters.items()
        if parameter.kind in (inspect.Parameter.POSITIONAL_ONLY,
                              inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def test_gemma4_stand_in_matches_the_real_encoder_signature():
    """Skipped without transformers; pins the stand-in when it is present."""
    modeling = pytest.importorskip(
        "transformers.models.gemma4.modeling_gemma4")
    real = modeling.Gemma4VisionEncoder.forward
    assert _positional_names(real)[1:] == _positional_names(
        _Gemma4VisionEncoderLike.forward)[1:]
    assert MM.compute_mm_jit_static_args(real) == {}


def test_qwen3_vl_stand_in_matches_the_real_transformer_signature():
    """Skipped without vLLM; pins the stand-in when it is present."""
    qwen3_vl = pytest.importorskip("vllm.model_executor.models.qwen3_vl")
    real = qwen3_vl.Qwen3_VisionTransformer.forward
    assert _positional_names(real)[1:] == _positional_names(
        _Qwen3VisionLike.forward)[1:]
    # Unbound here, so drop the leading `self` the bound method would hide.
    assert MM.compute_mm_jit_static_args(real) == {
        "static_argnums": (OLD_FIXED_MAPPING["static_argnums"][0] + 1, ),
        "static_argnames": ("grid_thw", ),
    }


# ------------------------------------------------------------------ source
def test_patcher_derives_static_args_instead_of_hard_coding_them():
    """Guard the revert: no fixed static_argnums literal in the patcher."""
    source = PATCHER_SRC.read_text(encoding="utf-8")
    assert "compute_mm_jit_static_args(target_module.forward)" in source
    body = source.split("def patch_mm_model", 1)[1]
    assert '"static_argnums"' not in body
    assert '"static_argnames"' not in body


# ------------------------------------------------------- real torchax path
def test_real_jittable_module_matches_eager_for_a_gemma4_like_module():
    """Same claim through torchax.interop.JittableModule when available."""
    torch = pytest.importorskip("torch")
    interop = pytest.importorskip("torchax.interop")
    torchax = pytest.importorskip("torchax")

    class Gemma4LikeEncoder(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(6, 6, bias=False)

        def forward(self,
                    inputs_embeds,
                    attention_mask,
                    pixel_position_ids=None,
                    **kwargs):
            hidden = self.proj(inputs_embeds)
            mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
            hidden = hidden * mask
            if pixel_position_ids is not None:
                hidden = hidden + pixel_position_ids[..., :1].to(
                    hidden.dtype) * 0.01
            return hidden

    torch.manual_seed(0)
    module = Gemma4LikeEncoder().eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)

    embeds = torch.randn(2, 5, 6)
    mask = torch.ones(2, 5, dtype=torch.bool)
    mask[1, 3:] = False
    positions = torch.arange(5).view(1, 5, 1).expand(2, 5, 2).contiguous()

    with torch.no_grad():
        want = module(inputs_embeds=embeds,
                      attention_mask=mask,
                      pixel_position_ids=positions).numpy()

    derived = MM.compute_mm_jit_static_args(module.forward)
    assert derived == {}

    env = torchax.default_env()
    with env:
        on_jax = module.to("jax")
        jittable = interop.JittableModule(on_jax, extra_jit_args=derived)
        with torch.no_grad():
            got = jittable(
                inputs_embeds=embeds.to("jax"),
                attention_mask=mask.to("jax"),
                pixel_position_ids=positions.to("jax"),
            )
        got = np.asarray(interop.jax_view(got))
    np.testing.assert_allclose(got, want, rtol=1e-5, atol=1e-5)
