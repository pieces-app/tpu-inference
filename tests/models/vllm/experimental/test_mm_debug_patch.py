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
"""PIECES_MM_DEBUG on the torchax path: the recorder hooks and the census.

``mm_debug_patch.py`` is torch-free, so it is driven here with plain-Python
stand-ins that keep the two properties the real objects have: ``__call__``
resolves ``self.forward`` through the instance (``torch.nn.Module`` and
``torchax.interop.JittableModule`` both do), and the vLLM
``_process_image_input`` flow is patch_embedder -> encoder(kwargs) ->
strip padding -> embed_vision(inputs_embeds=...) per image. The last test
repeats the claim through the real ``JittableModule`` when torchax is
importable (skipped on the CPU gate).
"""

import ast
import importlib.util
import pathlib

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = pathlib.Path(__file__).resolve().parents[4]
EXPERIMENTAL = ROOT / "tpu_inference" / "models" / "vllm" / "experimental"
PATCH_SRC = EXPERIMENTAL / "mm_debug_patch.py"
PATCHER_SRC = EXPERIMENTAL / "model_patcher.py"
STATS_SRC = ROOT / "tpu_inference" / "models" / "common" / "mm_debug_stats.py"


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


STATS = _load("_mm_debug_stats", STATS_SRC)
PATCH = _load("_mm_debug_patch", PATCH_SRC)

PP, H, D = 6, 5, 4
_RNG = np.random.default_rng(1)
W_PATCH = jnp.asarray(_RNG.standard_normal((PP, H)) * 0.3, jnp.float32)
W_ENC = jnp.asarray(_RNG.standard_normal((H, H)) * 0.3, jnp.float32)
W_PROJ = jnp.asarray(_RNG.standard_normal((H, D)) * 0.3, jnp.float32)


# ---------------------------------------------------------------- stand-ins
class _Mod:
    """nn.Module-like: __call__ -> self.forward, instance attribute first."""

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)

    def named_modules(self):
        yield "", self
        for name, child in getattr(self, "_children", {}).items():
            for sub, module in child.named_modules():
                yield f"{name}.{sub}".strip("."), module


class _Linear(_Mod):
    """HF nn.Linear stand-in (no quant_method)."""

    def forward(self, x):
        return x


class _ReplicatedLinear(_Mod):
    """vLLM LinearBase stand-in carrying the tpu_inference quant method."""

    class VllmUnquantizedLinearMethod:
        pass

    def __init__(self):
        self.quant_method = self.VllmUnquantizedLinearMethod()

    def forward(self, x):
        return x


class _Output:

    def __init__(self, last_hidden_state):
        self.last_hidden_state = last_hidden_state


class _PatchEmbedder(_Mod):

    def forward(self, pixel_values, pixel_position_ids, pad):
        return jnp.tanh(pixel_values @ W_PATCH)


class _Encoder(_Mod):

    def __init__(self, w_enc=W_ENC):
        self.w_enc = w_enc
        self._children = {
            "layers.0.self_attn.q_proj": _ReplicatedLinear(),
            "layers.0.self_attn.k_proj": _ReplicatedLinear(),
            "layers.0.mlp.fc1": _Linear(),
        }

    def forward(self,
                inputs_embeds,
                attention_mask,
                pixel_position_ids=None,
                **kwargs):
        hidden = jnp.tanh(inputs_embeds @ self.w_enc)
        return _Output(hidden * attention_mask[..., None].astype(hidden.dtype))


class JittableModule(_Mod):
    """Mirrors torchax.interop.JittableModule: wraps ``_model``, lies about
    ``__class__``, and its ``forward`` is the eager entry to the inner call."""

    def __init__(self, model):
        self._model = model
        self._children = {"_model": model}

    @property
    def __class__(self):
        return self._model.__class__

    def forward(self, *args, **kwargs):
        return self._model(*args, **kwargs)


class _Config:
    _attn_implementation = "sdpa"


class _Tower(_Mod):

    def __init__(self, encoder):
        self.config = _Config()
        self.patch_embedder = _PatchEmbedder()
        self.encoder = encoder
        self._children = {
            "patch_embedder": self.patch_embedder,
            "encoder": self.encoder,
        }


class _EmbedVision(_Mod):

    def __init__(self):
        self.embedding_projection = _ReplicatedLinear()
        self._children = {"embedding_projection": self.embedding_projection}

    def forward(self, inputs_embeds):
        return inputs_embeds @ W_PROJ


class _FakeGemma4:
    """vLLM Gemma4ForConditionalGeneration's vision flow, minus vLLM."""

    def __init__(self, encoder=None):
        self.vision_tower = _Tower(encoder or JittableModule(_Encoder()))
        self.embed_vision = _EmbedVision()

    def _process_image_input(self, image_input):
        pixel_values = image_input["pixel_values"]
        pixel_position_ids = image_input["pixel_position_ids"]
        vt = self.vision_tower
        per_image = []
        for pv, pp in zip(pixel_values, pixel_position_ids):
            pad = jnp.all(pp == -1, axis=-1)
            inputs_embeds = vt.patch_embedder(pv[None], pp[None], pad[None])
            encoder_outputs = vt.encoder(inputs_embeds=inputs_embeds,
                                         attention_mask=~pad[None],
                                         pixel_position_ids=pp[None])
            hidden = encoder_outputs.last_hidden_state[0]
            valid = hidden[~pad]
            proj = self.embed_vision(inputs_embeds=valid[None])[0]
            per_image.append(proj)
        return per_image

    def encoder_cudagraph_forward(self, inputs, path="default", **kwargs):
        pv = inputs["pixel_values"]
        pp = inputs["pixel_position_ids"]
        pad = jnp.all(pp == -1, axis=-1)
        inputs_embeds = self.vision_tower.patch_embedder(pv, pp, pad)
        hidden = self.vision_tower.encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=~pad,
            pixel_position_ids=pp).last_hidden_state
        flat = hidden.reshape(-1, hidden.shape[-1])
        return self.embed_vision(inputs_embeds=flat[None])[0]


def _images(patch_counts=(6, 4), pad_last_of_first=2):
    pvs, pps = [], []
    for i, n in enumerate(patch_counts):
        pv = _RNG.standard_normal((n, PP)).astype(np.float32)
        pp = _RNG.integers(0, 3, (n, 2)).astype(np.int32)
        if i == 0 and pad_last_of_first:
            pp[-pad_last_of_first:] = -1
            pv[-pad_last_of_first:] = 1e6
        pvs.append(jnp.asarray(pv))
        pps.append(jnp.asarray(pp))
    return {"pixel_values": pvs, "pixel_position_ids": pps}


def _fields(line):
    assert line.startswith(STATS.LINE_PREFIX + " "), line
    return dict(
        tok.split("=", 1)
        for tok in line[len(STATS.LINE_PREFIX) + 1:].split(" "))


def _install(model, log):
    return PATCH.install_mm_debug(model,
                                  to_jax=lambda t: t,
                                  log=log.append,
                                  emit=STATS.emit_mm_debug_stats)


# -------------------------------------------------------------- install
def test_install_hooks_the_four_members_and_is_idempotent():
    model = _FakeGemma4()
    log = []
    assert _install(model, log) == [
        "vision_tower.encoder",
        "embed_vision",
        "_process_image_input",
        "encoder_cudagraph_forward",
    ]
    assert _install(model, log) == []
    assert getattr(model.vision_tower.encoder.forward, "_mm_debug_wrapped")
    assert getattr(model._process_image_input, "_mm_debug_wrapped")


def test_model_without_vision_members_installs_nothing():

    class _Text:
        pass

    log = []
    assert _install(_Text(), log) == []
    assert log == []


# ------------------------------------------------------------ one line
def test_one_line_per_process_image_input_with_every_key():
    model = _FakeGemma4()
    log = []
    _install(model, log)
    out = model._process_image_input(_images())
    jax.block_until_ready(out)
    assert len(log) == 1, log
    f = _fields(log[0])
    assert f["path"] == "torchax"
    assert f["site"] == "_process_image_input"
    assert f["n_images"] == "2"
    for tensor in ("pv", "enc", "tower", "proj"):
        for key in STATS.STAT_KEYS:
            assert f"{tensor}.{key}" in f, (tensor, key, log[0])
        assert f[f"{tensor}.nan"] == "0"
    # Two images of different sizes: chunks are listed, not merged blindly.
    assert f["pv.shape"] == f"[(6,{PP}),(4,{PP})]"
    assert f["enc.shape"] == f"[(1,6,{H}),(1,4,{H})]"
    # embed_vision saw only valid rows: 4 + 4.
    assert f["tower.shape"] == f"2x(1,4,{H})"
    assert f["proj.shape"] == f"2x(1,4,{D})"
    assert f["soft_tokens"] == "8"
    # The padding rows of image 0 hold 1e6 and are masked out of pv.
    assert float(f["pv.max"]) < 100.0


def test_hooks_leave_the_outputs_unchanged():
    images = _images()
    clean = _FakeGemma4()
    want = clean._process_image_input(images)
    hooked = _FakeGemma4()
    _install(hooked, [])
    got = hooked._process_image_input(images)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert np.array_equal(np.asarray(g), np.asarray(w))


def test_nan_born_inside_the_encoder_is_counted_downstream_but_not_in_pv():
    """The hypothesis the live run tests: clean pixels in, garbage out of
    the jitted tower. A NaN weight in the encoder must show up as
    enc/tower/proj NaNs with pv.nan=0."""
    w_bad = W_ENC.at[0, 0].set(jnp.nan)
    model = _FakeGemma4(encoder=JittableModule(_Encoder(w_bad)))
    log = []
    _install(model, log)
    jax.block_until_ready(model._process_image_input(_images()))
    f = _fields(log[0])
    assert f["pv.nan"] == "0"
    assert int(f["enc.nan"]) > 0
    assert int(f["tower.nan"]) > 0
    assert int(f["proj.nan"]) > 0


def test_encoder_cudagraph_forward_logs_from_inside_jit():
    """Defensive coverage for the MM-encoder JIT manager's traced path: the
    hook sees tracers, the line is still produced when the program runs."""
    model = _FakeGemma4()
    log = []
    _install(model, log)
    pv = jnp.asarray(_RNG.standard_normal((2, 6, PP)).astype(np.float32))
    pp = jnp.asarray(_RNG.integers(0, 3, (2, 6, 2)).astype(np.int32))
    inputs = {"pixel_values": pv, "pixel_position_ids": pp}

    jitted = jax.jit(lambda v: model.encoder_cudagraph_forward(v))
    text = jax.make_jaxpr(lambda v: model.encoder_cudagraph_forward(v))(
        inputs).pretty_print()
    assert "debug_callback" in text
    out = jitted(inputs)
    jax.block_until_ready(out)
    assert len(log) == 1
    f = _fields(log[0])
    assert f["site"] == "encoder_cudagraph_forward"
    assert f["n_images"] == "2"
    assert f["soft_tokens"] == "12"
    assert f["pv.shape"] == f"(2,6,{PP})"
    assert f["enc.shape"] == f"(2,6,{H})"


# --------------------------------------------------------------- census
def test_census_reports_class_quant_method_wrapper_and_attention():
    model = _FakeGemma4()
    line = PATCH.describe_linears(model.vision_tower)
    assert "root=_Tower" in line
    # type() on purpose: the JittableModule stand-in lies about __class__.
    assert "encoder=JittableModule" in line
    assert "attn_impl=sdpa" in line
    assert "_ReplicatedLinear[VllmUnquantizedLinearMethod]=2" in line
    assert "_Linear[-]=1" in line
    assert "total=3" in line

    lines = PATCH.tower_census_lines(model)
    assert [line.split(":")[0] for line in lines
            ] == ["census vision_tower", "census embed_vision"]
    assert "_ReplicatedLinear[VllmUnquantizedLinearMethod]=1" in lines[1]


# ------------------------------------------------------------------ source
def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def test_patcher_installs_the_hooks_only_under_the_flag():
    tree = ast.parse(PATCHER_SRC.read_text())
    parents = _parents(tree)
    entry = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                 and n.name == "apply_model_specific_patches")
    calls = [
        n for n in ast.walk(entry) if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "_install_torchax_mm_debug"
    ]
    assert len(calls) == 1
    node = calls[0]
    guards = []
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            guards.append(ast.unparse(node.test))
    assert guards == ["envs.PIECES_MM_DEBUG"], guards


def test_patch_module_imports_neither_torch_nor_vllm():
    tree = ast.parse(PATCH_SRC.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    # tpu_inference appears only as the lazy default for `emit`.
    assert roots <= {"collections", "typing", "jax", "tpu_inference"}, roots
    assert "torch" not in roots and "vllm" not in roots


# ------------------------------------------------------- real torchax
def test_real_jittable_module_instance_forward_override_is_observed():
    """Same claim through torch + torchax.interop.JittableModule: replacing
    ``forward`` on the instance is what ``__call__`` runs, and
    ``jax_view(t.detach())`` hands the recorder the jitted region's output."""
    torch = pytest.importorskip("torch")
    torchax = pytest.importorskip("torchax")
    interop = pytest.importorskip("torchax.interop")

    class ModelOutput:
        """transformers BaseModelOutputWithPast stand-in: keys()/values() and
        keyword construction, registered as a pytree exactly the way
        patch_mm_model registers REGISTER_MM_MODULE_CUSTOM_PYTREE_CLASSES."""

        def __init__(self, last_hidden_state=None):
            self.last_hidden_state = last_hidden_state

        def keys(self):
            return ["last_hidden_state"]

        def values(self):
            return [self.last_hidden_state]

    jax.tree_util.register_pytree_node(
        ModelOutput,
        lambda obj: (obj.values(), obj.keys()),
        lambda keys, children: ModelOutput(**dict(zip(keys, children))),
    )

    class Encoder(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.proj = torch.nn.Linear(H, H, bias=False)

        def forward(self,
                    inputs_embeds,
                    attention_mask,
                    pixel_position_ids=None,
                    **kwargs):
            hidden = torch.tanh(self.proj(inputs_embeds))
            return ModelOutput(hidden *
                               attention_mask.unsqueeze(-1).to(hidden.dtype))

    class EmbedVision(torch.nn.Module):

        def __init__(self):
            super().__init__()
            self.embedding_projection = torch.nn.Linear(H, D, bias=False)

        def forward(self, inputs_embeds):
            return self.embedding_projection(inputs_embeds)

    class Model:

        def __init__(self, encoder, embed_vision):
            self.vision_tower = type("Tower", (), {})()
            self.vision_tower.encoder = encoder
            self.embed_vision = embed_vision

        def _process_image_input(self, image_input):
            out = []
            for pv, pp in zip(image_input["pixel_values"],
                              image_input["pixel_position_ids"]):
                pad = (pp == -1).all(dim=-1)
                enc = self.vision_tower.encoder(
                    inputs_embeds=pv.unsqueeze(0),
                    attention_mask=(~pad).unsqueeze(0),
                    pixel_position_ids=pp.unsqueeze(0)).last_hidden_state
                # squeeze, not [0]: torchax's narrow-slice View has no
                # boolean indexing (a stand-in detail, not the claim).
                enc = enc.squeeze(0)
                out.append(self.embed_vision(inputs_embeds=enc[~pad]))
            return out

    torch.manual_seed(0)
    encoder = Encoder().eval()
    embed_vision = EmbedVision().eval()
    for p in list(encoder.parameters()) + list(embed_vision.parameters()):
        p.requires_grad_(False)

    pv = torch.randn(6, H)
    pp = torch.randint(0, 3, (6, 2))
    pp[-2:] = -1

    def to_jax(t):
        return interop.jax_view(t.detach())

    env = torchax.default_env()
    with env:
        model = Model(interop.JittableModule(encoder.to("jax")),
                      embed_vision.to("jax"))
        log = []
        hooks = PATCH.install_mm_debug(model,
                                       to_jax=to_jax,
                                       log=log.append,
                                       emit=STATS.emit_mm_debug_stats)
        assert hooks == [
            "vision_tower.encoder", "embed_vision", "_process_image_input"
        ]
        with torch.no_grad():
            out = model._process_image_input({
                "pixel_values": [pv.to("jax")],
                "pixel_position_ids": [pp.to("jax")],
            })
        got = np.asarray(interop.jax_view(out[0]))
    assert got.shape == (4, D)
    assert len(log) == 1, log
    f = _fields(log[0])
    assert f["enc.shape"] == f"(1,6,{H})"
    assert f["tower.shape"] == f"(4,{H})"
    assert f["proj.shape"] == f"(4,{D})"
    assert f["soft_tokens"] == "4"
    assert f["enc.nan"] == "0" and f["proj.nan"] == "0"


# ------------------------------------- the pooler, at Gemma-4 E4B scale
#
# The lane serves --mm-processor-kwargs '{"max_soft_tokens": 1120}', so the
# HF Gemma4 image processor emits MAX_PATCHES = 1120 * 3^2 = 10080 patches
# per image, right-padded with (-1, -1) positions. The pooler's output
# buffer is therefore always 1120 slots; how many of them are VALID depends
# on the image's patch grid, and that valid count -- not 1120 -- is what the
# processor put in the prompt as <image> placeholders.
#
# _Pooler mirrors transformers Gemma4VisionPooler._avg_pool_by_positions.
# Verified against the real module (transformers 5.16.1) for the grids
# below: (1, 1120, D) pooled, 1092 / 1092 / 1104 valid.
MAX_SOFT_TOKENS = 1120
POOL_K = 3
MAX_PATCHES = MAX_SOFT_TOKENS * POOL_K**2  # 10080
# (image WxH, cols x rows after the processor's aspect-preserving resize,
#  soft tokens the prompt gets). Three real images from the iso-cases-wide
# set the eval-e4b lanes serve.
E4B_GRIDS = [
    ("3402x2158", 126, 78, 1092),
    ("2946x2070", 117, 84, 1092),
    ("1840x872", 144, 69, 1104),
]


def _pool_by_positions(hidden, pos, length):
    """transformers Gemma4VisionPooler._avg_pool_by_positions, in jnp.

    The index math is copied verbatim -- ``max_x // k`` as the row stride is
    where the valid-slot count comes from. The reduction is written as a
    segment sum rather than the reference one-hot matmul (which would
    allocate a 10080x1120 weight matrix); the two agree exactly, and
    ``test_segment_form_matches_the_reference_one_hot_matmul`` pins that.
    """
    k = int((hidden.shape[1] // length)**0.5)
    clamped = jnp.clip(pos, 0, None)
    max_x = clamped[..., 0].max(axis=-1, keepdims=True) + 1
    kernel_idxs = clamped // k
    idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    pooled = jnp.stack([
        jax.ops.segment_sum(hidden[b], idx[b], num_segments=length)
        for b in range(hidden.shape[0])
    ]) / (k * k)
    mask = jnp.zeros((hidden.shape[0], length),
                     bool).at[jnp.arange(hidden.shape[0])[:, None],
                              idx].set(True)
    return pooled, mask


def _pool_by_positions_reference(hidden, pos, length):
    """The reference one-hot matmul, for the equivalence check only."""
    k = int((hidden.shape[1] // length)**0.5)
    clamped = jnp.clip(pos, 0, None)
    max_x = clamped[..., 0].max(axis=-1, keepdims=True) + 1
    kernel_idxs = clamped // k
    idx = kernel_idxs[..., 0] + (max_x // k) * kernel_idxs[..., 1]
    weights = jax.nn.one_hot(idx, length, dtype=jnp.float32) / (k * k)
    pooled = jnp.einsum("bnl,bnh->blh", weights, hidden)
    mask = jnp.logical_not((weights == 0).all(axis=1))
    return pooled, mask


class _Pooler(_Mod):

    def forward(self, hidden_states, pixel_position_ids, padding_positions,
                output_length):
        hidden_states = jnp.where(padding_positions[..., None], 0.0,
                                  hidden_states)
        if hidden_states.shape[1] == output_length:
            return hidden_states, jnp.logical_not(padding_positions)
        return _pool_by_positions(hidden_states, pixel_position_ids,
                                  output_length)


class _PooledTower(_Tower):

    def __init__(self, encoder):
        super().__init__(encoder)
        self.pooler = _Pooler()
        self._children["pooler"] = self.pooler


class _FakeGemma4Pooled(_FakeGemma4):
    """vllm gemma4_mm.py Gemma4ForConditionalGeneration._process_image_input:
    encode the padded patches, pool per image to shape[1] // k^2 slots, drop
    the invalid slots, project the packed rows for every image at once."""

    def __init__(self, encoder=None):
        super().__init__(encoder)
        self.vision_tower = _PooledTower(encoder or JittableModule(_Encoder()))

    def _process_image_input(self, image_input):
        vt = self.vision_tower
        valid_states, valid_lens = [], []
        for pv, pp in zip(image_input["pixel_values"],
                          image_input["pixel_position_ids"]):
            pad = jnp.all(pp == -1, axis=-1)
            inputs_embeds = vt.patch_embedder(pv[None], pp[None], pad[None])
            hidden = vt.encoder(inputs_embeds=inputs_embeds,
                                attention_mask=~pad[None],
                                pixel_position_ids=pp[None]).last_hidden_state
            output_length = hidden.shape[1] // (POOL_K**2)
            pooled, valid_mask = vt.pooler(
                hidden_states=hidden,
                pixel_position_ids=pp[None],
                padding_positions=pad[None],
                output_length=output_length,
            )
            packed = pooled[valid_mask]
            valid_states.append(packed)
            valid_lens.append(int(packed.shape[0]))
        flat = jnp.concatenate(valid_states, axis=0)
        proj = self.embed_vision(inputs_embeds=flat[None])[0]
        out, offset = [], 0
        for n in valid_lens:
            out.append(proj[offset:offset + n])
            offset += n
        return out


def _e4b_image(cols, rows):
    """One image the way the HF Gemma4 image processor emits it: real (x, y)
    patch positions row-major, right-padded to MAX_PATCHES with (-1, -1)."""
    ys, xs = jnp.meshgrid(jnp.arange(rows), jnp.arange(cols), indexing="ij")
    real = jnp.stack([xs.reshape(-1), ys.reshape(-1)], axis=-1)
    pad = jnp.full((MAX_PATCHES - real.shape[0], 2), -1, real.dtype)
    pos = jnp.concatenate([real, pad], axis=0).astype(jnp.int32)
    pv = jnp.asarray(
        _RNG.standard_normal((MAX_PATCHES, PP)).astype(np.float32))
    return pv, pos


def test_segment_form_matches_the_reference_one_hot_matmul():
    pv, pos = _e4b_image(9, 6)
    pos = pos[:54][None]
    hidden = jnp.asarray(_RNG.standard_normal((1, 54, H)), jnp.float32)
    got_pooled, got_mask = _pool_by_positions(hidden, pos, 6)
    want_pooled, want_mask = _pool_by_positions_reference(hidden, pos, 6)
    assert np.array_equal(np.asarray(got_mask), np.asarray(want_mask))
    np.testing.assert_allclose(np.asarray(got_pooled),
                               np.asarray(want_pooled),
                               rtol=1e-5,
                               atol=1e-5)


@pytest.mark.parametrize("name,cols,rows,soft", E4B_GRIDS)
def test_pooler_buffer_is_1120_slots_and_the_valid_count_is_the_answer(
        name, cols, rows, soft):
    """The torchax call path, at the real scale. 10080 padded patches pool
    to a 1120-slot buffer for EVERY image; the number of valid slots -- and
    so the number of embeddings the projector and the LM see -- is 1092 or
    1104 depending on the grid. 1120 is the buffer, not the answer."""
    pv, pos = _e4b_image(cols, rows)
    hidden = jnp.asarray(_RNG.standard_normal((1, MAX_PATCHES, H)),
                         jnp.float32)
    pad = jnp.all(pos == -1, axis=-1)[None]
    pooled, valid_mask = _Pooler()(hidden_states=hidden,
                                   pixel_position_ids=pos[None],
                                   padding_positions=pad,
                                   output_length=MAX_PATCHES // POOL_K**2)
    assert pooled.shape == (1, MAX_SOFT_TOKENS, H), name
    assert int(valid_mask.sum()) == soft, name
    assert pooled[valid_mask].shape == (soft, H), name


def test_line_reports_the_padded_pooler_buffer_and_the_valid_soft_tokens():
    """``tower`` is the pooler's (1, 1120, H) buffer -- the same tensor at
    the same point the flax path reports -- and ``soft_tokens`` is the 1092
    valid rows the projector and the language model actually get."""
    model = _FakeGemma4Pooled()
    log = []
    assert _install(model, log) == [
        "vision_tower.encoder",
        "vision_tower.pooler",
        "embed_vision",
        "_process_image_input",
        "encoder_cudagraph_forward",
    ]
    pv, pos = _e4b_image(126, 78)
    out = model._process_image_input({
        "pixel_values": [pv],
        "pixel_position_ids": [pos]
    })
    jax.block_until_ready(out)
    assert len(log) == 1, log
    f = _fields(log[0])
    assert f["tower.shape"] == f"(1,{MAX_SOFT_TOKENS},{H})"
    assert f["soft_tokens"] == "1092"
    assert f["proj.shape"] == f"(1,1092,{D})"
    assert f["pv.shape"] == f"({MAX_PATCHES},{PP})"
    assert out[0].shape == (1092, D)


def test_tower_stats_skip_the_invalid_pooler_slots():
    """The mask has to travel with the buffer: the 28 slots that are not
    valid for this grid must not enter the statistics, or the torchax and
    flax lines describe different elements again."""

    class _SpikedPooler(_Pooler):

        def forward(self, hidden_states, pixel_position_ids, padding_positions,
                    output_length):
            pooled, mask = super().forward(hidden_states, pixel_position_ids,
                                           padding_positions, output_length)
            return jnp.where(mask[..., None], pooled, 1e9), mask

    model = _FakeGemma4Pooled()
    model.vision_tower.pooler = _SpikedPooler()
    log = []
    _install(model, log)
    pv, pos = _e4b_image(126, 78)
    jax.block_until_ready(
        model._process_image_input({
            "pixel_values": [pv],
            "pixel_position_ids": [pos]
        }))
    f = _fields(log[0])
    assert f["tower.shape"] == f"(1,{MAX_SOFT_TOKENS},{H})"
    assert float(f["tower.maxabs"]) < 1e6, f["tower.maxabs"]


def test_pooler_hook_leaves_the_embeddings_unchanged():
    pv, pos = _e4b_image(126, 78)
    images = {"pixel_values": [pv], "pixel_position_ids": [pos]}
    want = _FakeGemma4Pooled()._process_image_input(images)
    hooked = _FakeGemma4Pooled()
    _install(hooked, [])
    got = hooked._process_image_input(images)
    assert len(got) == len(want)
    for g, w in zip(got, want):
        assert np.array_equal(np.asarray(g), np.asarray(w))


def test_tower_without_a_pooler_still_reports_the_projector_input():
    """Towers with no pooler to hook (the Unified patcher's shape) keep the
    old fallback rather than losing the field."""
    model = _FakeGemma4()
    assert not hasattr(model.vision_tower, "pooler")
    log = []
    assert "vision_tower.pooler" not in _install(model, log)
    jax.block_until_ready(model._process_image_input(_images()))
    f = _fields(log[0])
    assert f["tower.shape"] == f"2x(1,4,{H})"


# ------------------------------------------- PIECES_MM_DEBUG_LAYERS (A/L)
class _VisionAttention(_Mod):
    """Class name ends in "Attention"; returns (output, attn_weights)."""

    def forward(self, hidden, **kwargs):
        return jnp.tanh(hidden @ W_ENC), None


class _VisionEncoderLayer(_Mod):
    """Class name ends in "EncoderLayer"; returns the hidden states."""

    def __init__(self):
        self.self_attn = _VisionAttention()
        self._children = {"self_attn": self.self_attn}

    def forward(self, hidden, **kwargs):
        attn, _ = self.self_attn(hidden)
        return hidden + attn


class _LayeredEncoder(_Mod):

    def __init__(self, n_layers=2):
        self.layers = [_VisionEncoderLayer() for _ in range(n_layers)]
        self._children = {
            f"layers.{i}": layer
            for i, layer in enumerate(self.layers)
        }

    def forward(self,
                inputs_embeds,
                attention_mask,
                pixel_position_ids=None,
                **kwargs):
        hidden = inputs_embeds
        for layer in self.layers:
            hidden = layer(hidden)
        return _Output(hidden * attention_mask[..., None].astype(hidden.dtype))


def _layered_model():
    return _FakeGemma4(encoder=_LayeredEncoder())


def test_per_layer_off_emits_one_line_and_no_layer_fields():
    """Negative control: the flag off changes nothing about the line."""
    log = []
    model = _layered_model()
    _install(model, log)
    model._process_image_input(_images())
    assert len(log) == 1, log
    fields = _fields(log[0])
    assert not [k for k in fields if k.startswith(("A0.", "L0."))]


def test_per_layer_on_emits_a_second_line_with_every_layer():
    log = []
    model = _layered_model()
    PATCH.install_mm_debug(model,
                           to_jax=lambda t: t,
                           log=log.append,
                           emit=STATS.emit_mm_debug_stats,
                           per_layer=True)
    model._process_image_input(_images())
    assert len(log) == 2, log
    main, layers = _fields(log[0]), _fields(log[1])
    assert main["site"] == "_process_image_input"
    assert layers["site"] == "_process_image_input:layers"
    # Two layers, each with an attention output and a layer output.
    for name in ("A0", "A1", "L0", "L1"):
        assert f"{name}.std" in layers, (name, sorted(layers))
        assert f"{name}.nan" in layers
    assert "A2.std" not in layers
    # The recorded tensors are the encoder's own, so a layer's stats must
    # differ from the attention output it is built from.
    assert layers["A0.std"] != layers["L0.std"]


def test_per_layer_hooks_leave_the_encoder_output_unchanged():
    """The hooks record; they must not alter what the tower returns."""
    images = _images()  # _RNG advances per call, so build the input ONCE
    clean = _layered_model()._process_image_input(images)
    hooked_model = _layered_model()
    PATCH.install_mm_debug(hooked_model,
                           to_jax=lambda t: t,
                           log=[].append,
                           emit=STATS.emit_mm_debug_stats,
                           per_layer=True)
    hooked = hooked_model._process_image_input(images)
    assert len(clean) == len(hooked)
    for a, b in zip(clean, hooked):
        assert np.array_equal(np.asarray(a), np.asarray(b))


def test_per_layer_hooks_reach_inside_a_jittable_module():
    """The encoder is wrapped by patch_mm_model before the hooks go in."""
    log = []
    model = _FakeGemma4(encoder=JittableModule(_LayeredEncoder()))
    PATCH.install_mm_debug(model,
                           to_jax=lambda t: t,
                           log=log.append,
                           emit=STATS.emit_mm_debug_stats,
                           per_layer=True)
    model._process_image_input(_images())
    assert len(log) == 2, log
    assert "L1.std" in _fields(log[1])


def test_the_patcher_passes_the_layers_flag_through():
    src = PATCHER_SRC.read_text()
    assert "per_layer=envs.PIECES_MM_DEBUG_LAYERS" in src


def test_the_flax_tower_emits_the_same_two_names_under_the_same_flag():
    """A one-sided instrument cannot produce a differential."""
    flax_src = (ROOT / "tpu_inference" / "models" / "jax" /
                "gemma4_mm.py").read_text()
    assert "envs.PIECES_MM_DEBUG_LAYERS" in flax_src
    assert 'f"A{index}"' in flax_src or '"A"' in flax_src
    assert "layer_sink" in flax_src and "attn_sink" in flax_src
    # ... and the sinks stay None with the flag off, so the jaxpr is unchanged.
    tree = ast.parse(flax_src)
    fn = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.FunctionDef) and n.name == "_new_layer_sinks")
    body = ast.unparse(fn)
    assert "envs.PIECES_MM_DEBUG and envs.PIECES_MM_DEBUG_LAYERS" in body
    assert "return (None, None)" in body
