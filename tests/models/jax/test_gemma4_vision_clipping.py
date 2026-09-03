"""The Gemma-4 vision tower's activation clipping: the flax/torchax differential.

WHAT WAS WRONG
--------------
On the same 69-image bank, E4B on the torchax path (MODEL_IMPL_TYPE=vllm)
answered degenerately on ~20 requests where the flax path (flax_nnx) did on 3.
Pixel values were bit-identical per request and the soft-token counts agreed,
but the ENCODER OUTPUT differed (enc.std 49.80 torchax vs 47.89 native).

The cause is activation clipping.  transformers' ``Gemma4ClippableLinear``
clamps a projection's input and its output against four scalars that ship in
the checkpoint; ``google/gemma-4-E4B-it`` sets ``use_clipped_linears: true``
and carries 448 finite BF16 clamp tensors for the vision tower (fixture
below).  vLLM's ``AutoWeightsLoader`` loads registered buffers, so the torchax
lane clips.  The flax loader skipped those four names outright and the flax
modules had no clamp to receive them, so the native lane did not.

WHICH SIDE IS THE REFERENCE
---------------------------
transformers' implementation, running the checkpoint's own numbers, is.  The
flax path was the deviating one -- even though it produced the better answers
on 2026-09-03.  These tests pin the reference and the fix.

WHAT IS TESTED HERE
-------------------
The gate half (jax + numpy, no torch/vllm) is a layer-by-layer differential
between an independent NumPy transcription of transformers'
``Gemma4VisionEncoderLayer`` and a JAX stack built on the model's own
``gemma4_vision_clip`` leaf: with the clamps honoured the two agree to fp32
rounding, and with them dropped they part company at LAYER 0.  Plus AST tests
that pin the loader's skip list, which projections are clipped, and the
load-completeness check -- the parts that cannot be exercised without a mesh
and a checkpoint.

The torch half (``test_transformers_tower_*``) drives the REAL
``transformers.Gemma4VisionModel`` and is importorskipped; it does not run on
the CPU gate (which installs no torch).  Its local output is pasted in the PR.

TPU-ONLY, NOT COVERED HERE: the flax tower's attention is the Pallas
``sharded_flash_attention`` kernel and its 128-multiple query padding.  This
file substitutes dense attention with the same segment-id mask, sm_scale=1.0
and fp32 softmax, so it proves the clipping arithmetic and the module wiring,
not the kernel.  ``PIECES_MM_DEBUG=1`` on hardware is what closes that gap.
"""
import ast
import importlib.util
import json
import math
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
MODEL = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mm.py"
LEAF = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_vision_clip.py"
FIXTURE = ROOT / "tests" / "fixtures" / "gemma-4-E4B-it.vision-clamps.json"

CLAMP_NAMES = (".input_min", ".input_max", ".output_min", ".output_max")


def _leaf():
    """Load the pure-jax leaf by path: importing the package would pull vllm."""
    pytest.importorskip("jax")
    spec = importlib.util.spec_from_file_location("_g4vclip", LEAF)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def _tree():
    return ast.parse(MODEL.read_text())


def _classdef(tree, name):
    for n in ast.walk(tree):
        if isinstance(n, ast.ClassDef) and n.name == name:
            return n
    raise AssertionError(f"{name} not found in {MODEL}")


def _funcdef(node, name):
    for n in ast.walk(node):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return n
    raise AssertionError(
        f"{name} not found under {getattr(node, 'name', node)}")


def _assignments(fn):
    """{attribute name: the ast.Call it is assigned} for `self.x = f(...)`."""
    out = {}
    for n in ast.walk(fn):
        if not isinstance(n, ast.Assign) or not isinstance(n.value, ast.Call):
            continue
        for t in n.targets:
            if (isinstance(t, ast.Attribute) and isinstance(t.value, ast.Name)
                    and t.value.id == "self"):
                out[t.attr] = n.value
    return out


def _call_name(call):
    f = call.func
    return f.id if isinstance(f, ast.Name) else getattr(f, "attr", "")


# --------------------------------------------------------------- the checkpoint
def test_the_checkpoint_ships_finite_vision_clamps():
    """448 live numbers, some tighter than +-2.  The skip was dropping data.

    If these were all +-inf the skip would have been harmless and this whole
    change unnecessary; they are not.
    """
    clamps = json.loads(FIXTURE.read_text())["clamps"]
    assert len(clamps) == 448, len(clamps)
    values = np.array(list(clamps.values()), dtype=np.float64)
    assert np.isfinite(values).all(), "a clamp that is +-inf would be a no-op"

    # 16 layers x {q,k,v,o,gate,up,down}_proj x 4 bounds
    projections = {k.rsplit(".", 1)[0] for k in clamps}
    assert len(projections) == 112, sorted(projections)[:4]
    assert {k.rsplit(".", 1)[1]
            for k in clamps
            } == {"input_min", "input_max", "output_min", "output_max"}

    # The tightest bound is the attention output projection's input: clipping
    # there bites on ordinary activations, it is not a numerical backstop.
    o_in = [
        abs(v) for k, v in clamps.items()
        if k.endswith(("o_proj.input_max", "o_proj.input_min"))
    ]
    assert min(o_in) < 2.0, min(o_in)
    assert max(abs(values)) < 100.0


# --------------------------------------------------------------- source shape
def test_load_weights_no_longer_skips_the_vision_clamps():
    """The four clamp names must not appear in load_weights' skip list.

    Negative control: the pre-fix list carried all four and this fails on it.
    """
    fn = _funcdef(_classdef(_tree(), "Gemma4ForConditionalGeneration"),
                  "load_weights")
    skipped = {
        c.value
        for c in ast.walk(fn)
        if isinstance(c, ast.Constant) and isinstance(c.value, str)
    }
    for suffix in CLAMP_NAMES:
        assert suffix not in skipped, (
            f"{suffix} is still skipped: the checkpoint's clamp would be "
            "dropped and the encoder would run unclipped")
    # ...while the audio names, which this class genuinely cannot serve, stay.
    assert {"audio_tower", "embed_audio"} <= skipped


def test_every_clipped_projection_is_clipped_and_the_patch_embedder_is_not():
    """Exactly the seven projections transformers clips, and no others.

    transformers builds q/k/v/o and gate/up/down as Gemma4ClippableLinear and
    the patch embedder's input_proj and the multimodal embedder's projection
    as plain nn.Linear.  A clamp on the wrong module would corrupt activations
    the reference never touches.
    """
    tree = _tree()
    clipped_cls = "Gemma4VisionClippedEinsum"

    attn = _assignments(
        _funcdef(_classdef(tree, "Gemma4VisionFlashAttention"), "__init__"))
    mlp = _assignments(_funcdef(_classdef(tree, "Gemma4VisionMLP"),
                                "__init__"))
    for name, calls in (("q_proj", attn), ("k_proj", attn), ("v_proj", attn),
                        ("o_proj", attn), ("gate_proj", mlp), ("up_proj", mlp),
                        ("down_proj", mlp)):
        call = calls[name]
        assert _call_name(call) == clipped_cls, (name, _call_name(call))
        kw = {k.arg for k in call.keywords}
        assert "use_clipped_linears" in kw, name

    embedder = _assignments(
        _funcdef(_classdef(tree, "Gemma4VisionPatchEmbedder"), "__init__"))
    assert _call_name(embedder["input_proj"]) == "JaxEinsum"
    projector = _assignments(
        _funcdef(_classdef(tree, "Gemma4MultimodalEmbedder"), "__init__"))
    assert _call_name(projector["embedding_projection"]) == "JaxEinsum"


def test_load_weights_verifies_every_clamp_landed():
    """A clamp left at its init is +-inf, i.e. silently unclipped again."""
    fn = _funcdef(_classdef(_tree(), "Gemma4ForConditionalGeneration"),
                  "load_weights")
    called = {_call_name(c) for c in ast.walk(fn) if isinstance(c, ast.Call)}
    assert "_verify_vision_clamps_loaded" in called


def test_the_clipped_einsum_falls_back_to_the_plain_projection():
    """use_clipped_linears=False must not add params or change arithmetic.

    E2B/12B variants that do not set the flag must keep byte-identical
    behaviour, so the class has to have a real off switch.
    """
    cls = _classdef(_tree(), "Gemma4VisionClippedEinsum")
    call = _funcdef(cls, "__call__")
    first = call.body[0]
    assert isinstance(first, ast.If), ast.dump(first)[:80]
    assert isinstance(first.body[0], ast.Return)
    init = _funcdef(cls, "__init__")
    guarded = [n for n in init.body if isinstance(n, ast.If)]
    assert guarded, "the clamp params must be created only when clipping"
    created = {
        t.attr
        for n in ast.walk(guarded[0]) if isinstance(n, ast.Assign)
        for t in n.targets if isinstance(t, ast.Attribute)
    }
    assert {"input_min", "input_max", "output_min", "output_max"} <= created


# --------------------------------------------------------------- leaf numerics
def _np_clippable_linear(x, w, lo_in, hi_in, lo_out, hi_out):
    """transformers' Gemma4ClippableLinear.forward, in NumPy.

    torch: clamp(x, input_min, input_max) -> linear -> clamp(., output_min,
    output_max), with w in (out, in) layout.
    """
    x = np.clip(x, lo_in, hi_in)
    y = x @ w.T
    return np.clip(y, lo_out, hi_out)


def test_clamp_activation_matches_the_reference_clamp():
    clip = _leaf()
    import jax.numpy as jnp
    rng = np.random.default_rng(3)
    x = rng.standard_normal((7, 13)).astype(np.float32) * 4.0
    lo, hi = -1.75, 2.25
    got = np.asarray(clip.clamp_activation(jnp.asarray(x), lo, hi))
    np.testing.assert_array_equal(got, np.clip(x, lo, hi))
    # it must actually bite on this data, or the test proves nothing
    assert (np.abs(x - got) > 0).mean() > 0.2


def test_clamp_activation_reads_the_bound_in_the_activation_dtype():
    """transformers clamps a bf16 activation against bf16 buffers."""
    clip = _leaf()
    import jax.numpy as jnp
    x = jnp.asarray([-4.0, 0.5, 4.0], jnp.bfloat16)
    out = clip.clamp_activation(x, -1.3, 1.3)
    assert out.dtype == jnp.bfloat16
    # 1.3 is not representable in bf16; both sides round it the same way
    bound = np.asarray(jnp.asarray(1.3, jnp.bfloat16), np.float32)
    np.testing.assert_allclose(np.asarray(out, np.float32),
                               np.clip([-4.0, 0.5, 4.0], -bound, bound))


def test_neutral_clamps_are_a_no_op():
    """An unloaded clamp must degrade to the old behaviour, not to zeros."""
    clip = _leaf()
    import jax.numpy as jnp
    lo, hi = clip.neutral_clamps()
    assert float(lo) == -math.inf and float(hi) == math.inf
    x = jnp.asarray(
        np.random.default_rng(4).standard_normal(64) * 1e4, jnp.float32)
    np.testing.assert_array_equal(np.asarray(clip.clamp_activation(x, lo, hi)),
                                  np.asarray(x))


def test_unloaded_clamps_names_exactly_the_unfilled_clamps():
    clip = _leaf()
    loaded = {"a.input_min", "a.input_max", "a.output_min", "b.weight"}
    params = [("a.weight", 1), ("a.input_min", 1), ("a.input_max", 1),
              ("a.output_min", 1), ("a.output_max", 1), ("b.weight", 1)]
    named = [(n, n) for n, _ in params]
    missing = clip.unloaded_clamps(named, lambda p: p in loaded)
    assert list(missing) == ["a.output_max"]
    # nothing to report when they all arrived
    assert not clip.unloaded_clamps(named, lambda p: True)
    # a model with no clamps at all is not an error
    assert not clip.unloaded_clamps([("b.weight", "b.weight")],
                                    lambda p: False)


def test_clamp_suffixes_are_the_four_transformers_registers():
    assert tuple(_leaf().CLAMP_SUFFIXES) == CLAMP_NAMES


# ------------------------------------------------- the encoder-layer differential
HID, INTER, NHEAD, HDIM, EPS, THETA = 48, 128, 4, 12, 1e-6, 100.0


def _weights(layers, seed=0, clip_at=1.25):
    rng = np.random.default_rng(seed)

    def n(shape, s):
        return (rng.standard_normal(shape) * s).astype(np.float32)

    ws = []
    for _ in range(layers):
        d = {}
        for nm, o, i in (("q", NHEAD * HDIM, HID), ("k", NHEAD * HDIM, HID),
                         ("v", NHEAD * HDIM, HID), ("o", HID, NHEAD * HDIM),
                         ("gate", INTER, HID), ("up", INTER,
                                                HID), ("down", HID, INTER)):
            d[nm] = n((o, i), 1.0 / math.sqrt(i))
            # asymmetric, finite, and tight enough to bite -- as in the ckpt
            d[nm + "_clamp"] = (-clip_at, clip_at * 0.9, -clip_at * 1.1,
                                clip_at)
        for nm, dim in (("ln_in", HID), ("ln_post_attn", HID),
                        ("ln_pre_ff", HID), ("ln_post_ff", HID),
                        ("q_norm", HDIM), ("k_norm", HDIM)):
            d[nm] = (1.0 + 0.1 * rng.standard_normal(dim)).astype(np.float32)
        ws.append(d)
    return ws


def _grid(w, h, seed=1):
    rng = np.random.default_rng(seed)
    valid = w * h
    total = ((valid + 9) // 9) * 9  # >= one all-padding pooled slot
    x = np.zeros((1, total, HID), np.float32)
    x[0, :valid] = rng.standard_normal((valid, HID)).astype(np.float32)
    pos = np.full((1, total, 2), -1, np.int64)
    ys, xs = np.divmod(np.arange(valid), w)
    pos[0, :valid, 0], pos[0, :valid, 1] = xs, ys
    return x, pos


def _np_layer(h, pos, d, clipped):
    """transformers' Gemma4VisionEncoderLayer.forward, in NumPy.

    Independent of the model source: explicit matmuls, torch's fp32 RMSNorm,
    the concatenated per-dimension cos/sin of Gemma4VisionRotaryEmbedding and
    rotate_half, eager attention with an additive padding mask and fp32
    softmax.
    """

    def rms(x, scale):
        v = (x.astype(np.float64)**2).mean(-1, keepdims=True)
        y = x.astype(np.float64) * (v + EPS)**-0.5
        return (y * scale if scale is not None else y).astype(np.float32)

    def lin(x, key):
        w = d[key]
        if not clipped:
            return x @ w.T
        lo_i, hi_i, lo_o, hi_o = d[key + "_clamp"]
        return _np_clippable_linear(x, w, lo_i, hi_i, lo_o, hi_o)

    B, T, _ = h.shape
    residual = h
    x = rms(h, d["ln_in"])

    # cos/sin: per spatial dim, inv_freq over head_dim//2, doubled by cat
    spatial = HDIM // 2
    inv = 1.0 / (THETA**(np.arange(0, spatial, 2, dtype=np.float64) / spatial))
    cos, sin = [], []
    for ax in range(2):
        f = pos[..., ax][..., None] * inv  # (B,T,spatial/2)
        e = np.concatenate([f, f], -1)  # (B,T,spatial)
        cos.append(np.cos(e))
        sin.append(np.sin(e))
    cos = np.concatenate(cos, -1)[:, :, None, :]  # (B,T,1,HDIM)
    sin = np.concatenate(sin, -1)[:, :, None, :]

    def rope(t):
        parts = []
        for ax in range(2):
            s = slice(ax * spatial, (ax + 1) * spatial)
            p, c, si = t[..., s], cos[..., s], sin[..., s]
            half = spatial // 2
            rot = np.concatenate([-p[..., half:], p[..., :half]], -1)
            parts.append(p * c + rot * si)
        return np.concatenate(parts, -1)

    q = rms(lin(x, "q").reshape(B, T, NHEAD, HDIM), d["q_norm"])
    k = rms(lin(x, "k").reshape(B, T, NHEAD, HDIM), d["k_norm"])
    v = rms(lin(x, "v").reshape(B, T, NHEAD, HDIM), None)
    q, k = rope(q), rope(k)
    q, k, v = (t.transpose(0, 2, 1, 3) for t in (q, k, v))

    valid = pos[..., 0] != -1  # (B,T)
    logits = np.einsum("bnth,bnsh->bnts", q.astype(np.float64),
                       k.astype(np.float64))  # scaling = 1.0
    allowed = valid[:, None, :, None] & valid[:, None, None, :]
    allowed |= (~valid)[:, None, :, None] & (~valid)[:, None, None, :]
    logits = np.where(allowed, logits, -np.inf)
    logits -= logits.max(-1, keepdims=True)
    p = np.exp(logits)
    p /= p.sum(-1, keepdims=True)
    a = np.einsum("bnts,bnsh->bnth", p, v.astype(np.float64))
    a = a.transpose(0, 2, 1, 3).reshape(B, T, NHEAD * HDIM).astype(np.float32)

    a = rms(lin(a, "o"), d["ln_post_attn"]) + residual
    y = rms(a, d["ln_pre_ff"])
    g = lin(y, "gate")
    g = 0.5 * g * (
        1.0 + np.tanh(math.sqrt(2.0 / math.pi) *
                      (g + 0.044715 * g**3)))  # gelu tanh
    y = lin(g * lin(y, "up"), "down")
    return rms(y, d["ln_post_ff"]) + a


def _jax_layer(h, pos, d, clipped):
    """The fork's Gemma4VisionEncoderLayer arithmetic, using the real leaf.

    The projections go through gemma4_vision_clip.clamp_activation -- the
    function Gemma4VisionClippedEinsum calls -- so a change to the leaf shows
    up here.  Attention is the dense stand-in for sharded_flash_attention
    (TPU Pallas): same segment-id mask, sm_scale=1.0, fp32 softmax.
    """
    import jax
    import jax.numpy as jnp
    clip = _leaf()

    def rms(x, scale):
        v = jnp.mean(jnp.square(x.astype(jnp.float32)), -1, keepdims=True)
        y = x.astype(jnp.float32) * jax.lax.rsqrt(v + EPS)
        if scale is not None:
            y = y * jnp.asarray(scale, jnp.float32)
        return y.astype(x.dtype)

    def lin(x, key, spec):
        kernel = jnp.asarray(d[key].T)  # flax layout: (in, out)
        if not clipped:
            return jnp.einsum(spec, x, kernel)
        lo_i, hi_i, lo_o, hi_o = d[key + "_clamp"]
        x = clip.clamp_activation(x, lo_i, hi_i)
        return clip.clamp_activation(jnp.einsum(spec, x, kernel), lo_o, hi_o)

    B, T, _ = h.shape
    pos = jnp.asarray(pos, jnp.int32)
    x = rms(h, d["ln_in"])

    def heads(t):
        return t.reshape(B, T, NHEAD, HDIM)

    q = rms(heads(lin(x, "q", "...d,df->...f")), d["q_norm"])
    k = rms(heads(lin(x, "k", "...d,df->...f")), d["k_norm"])
    v = rms(heads(lin(x, "v", "...d,df->...f")), None)

    ndim = 2
    c_per_dim = 2 * (HDIM // (2 * ndim))
    half_c = c_per_dim // 2
    inv = 1.0 / (THETA**(jnp.arange(0, c_per_dim, 2, dtype=jnp.float32) /
                         c_per_dim))
    freqs = jnp.expand_dims(pos[..., None] * inv, axis=2)
    cos, sin = jnp.cos(freqs), jnp.sin(freqs)

    def rope(t):
        r = t.reshape(B, T, NHEAD, ndim, 2, half_c)
        x1, x2 = r[..., 0, :], r[..., 1, :]
        o1, o2 = x1 * cos - x2 * sin, x2 * cos + x1 * sin
        return jnp.stack([o1, o2], -2).reshape(B, T, NHEAD, ndim * c_per_dim)

    q, k = rope(q), rope(k)
    q, k, v = (jnp.transpose(t, (0, 2, 1, 3)) for t in (q, k, v))
    seg = jnp.where(pos[..., 0] != -1, 1, 2)
    logits = jnp.einsum("bnth,bnsh->bnts", q.astype(jnp.float32),
                        k.astype(jnp.float32))
    logits = jnp.where(seg[:, None, :, None] == seg[:, None, None, :], logits,
                       -jnp.inf)
    p = jax.nn.softmax(logits, -1)
    a = jnp.einsum("bnts,bnsh->bnth", p, v.astype(jnp.float32))
    a = jnp.transpose(a, (0, 2, 1, 3)).reshape(B, T, NHEAD * HDIM)

    a = rms(lin(a, "o", "...d,df->...f"), d["ln_post_attn"]) + h
    y = rms(a, d["ln_pre_ff"])
    g = jax.nn.gelu(lin(y, "gate", "...d,df->...f"), approximate=True)
    y = lin(g * lin(y, "up", "...d,df->...f"), "down", "...f,fd->...d")
    return rms(y, d["ln_post_ff"]) + a


def _stack(layer_fn, x, pos, ws, clipped):
    import jax.numpy as jnp
    h = x if layer_fn is _np_layer else jnp.asarray(x)
    out = []
    for d in ws:
        h = layer_fn(h, pos, d, clipped)
        out.append(np.asarray(h, np.float64))
    return out


def _rel(a, b, valid):
    a, b = a[valid], b[valid]
    return float(np.abs(a - b).max() / max(np.abs(b).max(), 1e-12))


def _cos(a, b, valid):
    a, b = a[valid].ravel(), b[valid].ravel()
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


LAYERS = 4


def test_honouring_the_clamps_reproduces_the_reference_encoder():
    """With clipping on, the fork's arithmetic IS transformers' arithmetic.

    This is the differential's control: everything else in the layer (RoPE,
    RMSNorm, the padding mask, the MLP's tanh gelu) already agrees, so any
    residual gap is fp32 accumulation, not semantics.
    """
    pytest.importorskip("jax")
    ws = _weights(LAYERS)
    x, pos = _grid(9, 6)
    valid = pos[..., 0] != -1
    ref = _stack(_np_layer, x, pos, ws, clipped=True)
    got = _stack(_jax_layer, x, pos, ws, clipped=True)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert _rel(g, r, valid) < 2e-4, (i, _rel(g, r, valid))
        assert _cos(g, r, valid) > 1 - 1e-8, (i, _cos(g, r, valid))


def test_dropping_the_clamps_diverges_from_layer_zero():
    """The measured defect: no clipping and the encoder is a different function.

    Not a rounding difference and not a late drift -- it is there in layer 0
    and it compounds.  The magnitudes below are the CPU restatement of what
    PIECES_MM_DEBUG measured on hardware (enc.std 49.80 clipped vs 47.89
    unclipped, only ~4 % apart in std while the vectors themselves diverge).
    """
    pytest.importorskip("jax")
    ws = _weights(LAYERS)
    x, pos = _grid(9, 6)
    valid = pos[..., 0] != -1
    ref = _stack(_np_layer, x, pos, ws, clipped=True)  # the reference
    unclipped = _stack(_jax_layer, x, pos, ws,
                       clipped=False)  # the old flax path

    first = _rel(unclipped[0], ref[0], valid)
    assert first > 1e-2, f"clipping must bite in layer 0, got {first}"
    # and it does not wash out
    assert _cos(unclipped[-1], ref[-1], valid) < 0.999
    # while the summary statistic barely moves -- which is why enc.std alone
    # under-reported the defect on hardware
    s_ref = ref[-1][valid].std()
    s_unc = unclipped[-1][valid].std()
    assert abs(s_unc - s_ref) / s_ref < 0.25, (s_ref, s_unc)


def test_the_clamps_are_what_diverges_and_nothing_else():
    """Turn clipping off on BOTH sides and the two implementations coincide."""
    pytest.importorskip("jax")
    ws = _weights(LAYERS)
    x, pos = _grid(9, 6)
    valid = pos[..., 0] != -1
    ref = _stack(_np_layer, x, pos, ws, clipped=False)
    got = _stack(_jax_layer, x, pos, ws, clipped=False)
    assert _rel(got[-1], ref[-1], valid) < 2e-4


# ------------------------------------------------------------ the torch half
# Skipped on the CPU gate (no torch/transformers there); run locally in the
# torch venv and pasted into the PR.
def _transformers_tower(layers, clipped):
    torch = pytest.importorskip("torch")
    pytest.importorskip("transformers")
    from transformers.models.gemma4.configuration_gemma4 import \
        Gemma4VisionConfig
    from transformers.models.gemma4.modeling_gemma4 import Gemma4VisionModel
    cfg = Gemma4VisionConfig(hidden_size=HID,
                             intermediate_size=INTER,
                             num_hidden_layers=layers,
                             num_attention_heads=NHEAD,
                             num_key_value_heads=NHEAD,
                             head_dim=HDIM,
                             patch_size=4,
                             pooling_kernel_size=3,
                             position_embedding_size=256,
                             rms_norm_eps=EPS,
                             use_clipped_linears=clipped,
                             standardize=False)
    cfg._attn_implementation = "eager"
    m = Gemma4VisionModel(cfg).to(torch.float32).eval()
    return torch, m


def _load_into_tower(torch, m, ws):
    sd = dict(m.state_dict())
    for i, d in enumerate(ws):
        p = f"encoder.layers.{i}."
        for key, nm in (("q", "self_attn.q_proj"), ("k", "self_attn.k_proj"),
                        ("v", "self_attn.v_proj"), ("o", "self_attn.o_proj"),
                        ("gate", "mlp.gate_proj"), ("up", "mlp.up_proj"),
                        ("down", "mlp.down_proj")):
            sd[p + nm + ".linear.weight"] = torch.as_tensor(d[key])
            if p + nm + ".input_min" in sd:
                lo_i, hi_i, lo_o, hi_o = d[key + "_clamp"]
                for suffix, val in (("input_min", lo_i), ("input_max", hi_i),
                                    ("output_min", lo_o), ("output_max",
                                                           hi_o)):
                    sd[p + nm + "." + suffix] = torch.tensor(float(val))
        for key, nm in (("ln_in", "input_layernorm"),
                        ("ln_post_attn", "post_attention_layernorm"),
                        ("ln_pre_ff", "pre_feedforward_layernorm"),
                        ("ln_post_ff", "post_feedforward_layernorm"),
                        ("q_norm", "self_attn.q_norm"), ("k_norm",
                                                         "self_attn.k_norm")):
            sd[p + nm + ".weight"] = torch.as_tensor(d[key])
    m.load_state_dict(sd, strict=True)


def _run_tower(torch, m, x, pos):
    caps = []
    hooks = [
        layer.register_forward_hook(lambda mod, a, o: caps.append(
            (o[0] if isinstance(o, tuple) else o).detach().double().numpy()))
        for layer in m.encoder.layers
    ]
    with torch.no_grad():
        m.encoder(inputs_embeds=torch.as_tensor(x),
                  attention_mask=torch.as_tensor(pos[..., 0] != -1),
                  pixel_position_ids=torch.as_tensor(pos).long())
    for h in hooks:
        h.remove()
    return caps


def test_transformers_tower_agrees_when_the_clamps_are_honoured():
    torch, m = _transformers_tower(LAYERS, clipped=True)
    ws = _weights(LAYERS)
    _load_into_tower(torch, m, ws)
    x, pos = _grid(9, 6)
    valid = pos[..., 0] != -1
    ref = _run_tower(torch, m, x, pos)
    got = _stack(_jax_layer, x, pos, ws, clipped=True)
    for i, (r, g) in enumerate(zip(ref, got)):
        assert _rel(g, r, valid) < 2e-4, (i, _rel(g, r, valid))


def test_transformers_tower_disagrees_with_the_unclipped_flax_path():
    """The defect, against the real reference implementation."""
    torch, m = _transformers_tower(LAYERS, clipped=True)
    ws = _weights(LAYERS)
    _load_into_tower(torch, m, ws)
    x, pos = _grid(9, 6)
    valid = pos[..., 0] != -1
    ref = _run_tower(torch, m, x, pos)
    old = _stack(_jax_layer, x, pos, ws, clipped=False)
    assert _rel(old[0], ref[0], valid) > 1e-2
    assert _cos(old[-1], ref[-1], valid) < 0.999
