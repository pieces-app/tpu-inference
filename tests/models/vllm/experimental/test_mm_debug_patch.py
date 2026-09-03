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
