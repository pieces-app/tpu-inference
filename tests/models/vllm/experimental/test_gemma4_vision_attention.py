"""The Gemma-4 vision tower's attention scores: bf16 vs fp32 under torchax.

WHAT WAS WRONG
--------------
``Gemma4VisionAttention`` dispatches to transformers' ``sdpa`` interface,
i.e. ``torch.nn.functional.scaled_dot_product_attention``.  On the torchax
lane (``MODEL_IMPL_TYPE=vllm``) that is not a fused kernel: torchax lowers
it to ``torchax.ops.jtorch._sdpa_reference``, whose first line is

    attn_weight = query @ key.transpose(-2, -1) * scale_factor

with bf16 operands -- so the QK^T product is **materialised in bf16**.  Every
other backend keeps the scores in fp32: torch's own CPU/flash kernels do, and
so does the flax path's ``sharded_flash_attention``.

This tower makes that unusually expensive.  ``Gemma4VisionAttention`` sets
``self.scaling = 1.0`` (the q/k RMSNorms carry the scale instead), and
google/gemma-4-E4B-it's own ``q_norm``/``k_norm`` are constant scalars --
0.4062 and 1.2344 at layer 0 -- so a score is 64 * 0.50 * <unit dot>, tens
of units wide.  Rounding that to bf16's 8 mantissa bits perturbs it by
~0.2 % of its magnitude, and the softmax then exponentiates the error.

MEASURED on CPU at the real tower shape (12 heads, head_dim 64, 10 080
query positions, 252 padded), against an fp64 reference:

    torch eager SDPA (bf16 in)                       cos 0.999959
    torchax _sdpa_reference (bf16 scores)  <-- today  cos 0.999784
    bf16 operands, fp32 scores             <-- fixed  cos 0.999958

WHAT IS TESTED HERE
-------------------
The gate half (jax + numpy, no torch/torchax/vllm) is a numerical
differential between the two formulations at the real head geometry and the
checkpoint's real norm scales: the bf16-score form is several times further
from the fp64 reference than the fp32-score form, which lands on it.  Plus
the chunking equivalence (the fix computes the scores in query blocks to
keep the fp32 matrix small, and that must not change the answer), and AST
tests that pin the fix, its gating and its packaging -- the parts that
cannot be exercised without torch on the runner.

TPU-ONLY, NOT COVERED HERE: that XLA:TPU's default precision for an f32 dot
is a single bf16 pass with fp32 accumulation, i.e. that the fix costs no MXU
work on the chip.  ``eval-e4b-torchax`` with ``PIECES_MM_DEBUG=1`` is what
closes that gap; the boot log's ``[gemma4-patch] vision attention: fp32
scores on N modules`` line says the patch is live.
"""
import ast
import pathlib

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
PATCH = (ROOT / "tpu_inference" / "models" / "vllm" / "experimental" /
         "gemma4_vision_attention.py")
MM_PATCHER = (ROOT / "tpu_inference" / "models" / "vllm" / "experimental" /
              "gemma4_mm_patcher.py")
ENVS = ROOT / "tpu_inference" / "envs.py"
P8_DOCKERFILE = ROOT / "patches" / "image" / "p8" / "Dockerfile"
P8_ASSEMBLE = ROOT / "patches" / "image" / "p8" / "assemble.sh"

# google/gemma-4-E4B-it, vision_config + the layer-0 norm scales read out of
# model.safetensors (both are constant vectors in the checkpoint).
NHEAD, HDIM = 12, 64
Q_NORM_RMS, K_NORM_RMS = 0.4062, 1.2344


def _fn(tree, name):
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) \
                and node.name == name:
            return node
    raise AssertionError(f"{name} not found")


def _src(path, name):
    return ast.get_source_segment(path.read_text(),
                                  _fn(ast.parse(path.read_text()), name))


# --------------------------------------------------------------------- #
# The numerical differential: the two formulations, at the real geometry.
# --------------------------------------------------------------------- #


def _inputs(n_query, n_pad, seed=0):
    """q/k/v as the tower produces them: RMSNorm'd to the checkpoint's scales.

    ``Gemma4RMSNorm`` normalises each head's vector to unit RMS and then
    multiplies by the learned scale, so a component's RMS is exactly that
    scale.  v carries no scale (``with_scale=False``).
    """
    rng = np.random.default_rng(seed)

    def heads(scale):
        x = rng.standard_normal((1, NHEAD, n_query, HDIM))
        x /= np.sqrt((x**2).mean(-1, keepdims=True))
        return (x * scale).astype(np.float32)

    q, k, v = heads(Q_NORM_RMS), heads(K_NORM_RMS), heads(1.0)
    valid = np.ones((1, n_query), bool)
    if n_pad:
        valid[:, n_query - n_pad:] = False
    # transformers' sdpa_mask for this tower: bidirectional, AND the padding
    # mask, which is indexed by kv only -- so the mask is constant down the
    # query axis.
    mask = np.broadcast_to(valid[:, None, None, :], (1, 1, n_query, n_query))
    return q, k, v, mask, valid


def _attention(q, k, v, mask, score_dtype, out_dtype, chunk=0):
    """The vision tower's attention, with the score dtype as the knob.

    ``score_dtype=bfloat16`` is ``_sdpa_reference``; ``float32`` is the fix.
    """
    import jax
    import jax.numpy as jnp
    q, k, v = (jnp.asarray(x, out_dtype) for x in (q, k, v))
    m = jnp.asarray(mask)
    n = q.shape[-2]
    block = n if chunk <= 0 else min(chunk, n)
    parts = []
    for start in range(0, n, block):
        stop = min(start + block, n)
        s = jnp.einsum("bhqd,bhkd->bhqk", q[:, :,
                                            start:stop].astype(score_dtype),
                       k.astype(score_dtype))
        s = jnp.where(m[:, :, start:stop, :], s,
                      jnp.asarray(jnp.finfo(score_dtype).min, score_dtype))
        p = jax.nn.softmax(s, axis=-1).astype(out_dtype)
        parts.append(jnp.einsum("bhqk,bhkd->bhqd", p, v))
    return np.asarray(jnp.concatenate(parts, axis=-2), np.float64)


def _reference(q, k, v, mask):
    """fp64 softmax attention -- the answer both formulations approximate."""
    s = np.einsum("bhqd,bhkd->bhqk", q.astype(np.float64),
                  k.astype(np.float64))
    s = np.where(mask, s, -np.inf)
    s -= s.max(-1, keepdims=True)
    p = np.exp(s)
    p /= p.sum(-1, keepdims=True)
    return np.einsum("bhqk,bhkd->bhqd", p, v.astype(np.float64))


def _defect(a, b):
    """1 - cosine, i.e. how far `a` is from `b`. 0.0 is agreement."""
    a, b = np.asarray(a, np.float64).ravel(), np.asarray(b, np.float64).ravel()
    return 1.0 - float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


@pytest.mark.parametrize("n_query,n_pad", [(504, 12), (1008, 24)])
def test_bf16_scores_are_several_times_further_from_the_reference(
        n_query, n_pad):
    """The defect: bf16 scores, as _sdpa_reference builds them."""
    jnp = pytest.importorskip("jax.numpy")
    q, k, v, mask, _ = _inputs(n_query, n_pad)
    ref = _reference(q, k, v, mask)
    bf16_scores = _defect(
        _attention(q, k, v, mask, jnp.bfloat16, jnp.bfloat16), ref)
    fp32_scores = _defect(_attention(q, k, v, mask, jnp.float32, jnp.bfloat16),
                          ref)
    assert fp32_scores > 0.0, "the reference must not be reproduced exactly"
    assert bf16_scores > 3 * fp32_scores, (
        f"bf16 scores {bf16_scores:.3e} vs fp32 scores {fp32_scores:.3e}: "
        "the score dtype no longer costs accuracy, so this differential "
        "no longer measures anything")


@pytest.mark.parametrize("n_query,n_pad", [(504, 12), (1008, 24)])
def test_fp32_scores_reach_full_bf16_accuracy(n_query, n_pad):
    """With fp32 scores the only error left is the bf16 inputs and output."""
    jnp = pytest.importorskip("jax.numpy")
    q, k, v, mask, _ = _inputs(n_query, n_pad)
    ref = _reference(q, k, v, mask)
    # Everything in fp32 is the floor: bf16 q/k/v cannot beat it.
    floor = _defect(_attention(q, k, v, mask, jnp.float32, jnp.float32), ref)
    fp32_scores = _defect(_attention(q, k, v, mask, jnp.float32, jnp.bfloat16),
                          ref)
    assert floor < 1e-12
    assert fp32_scores < 1e-4


@pytest.mark.parametrize("chunk", [128, 512, 0])
def test_query_chunking_does_not_change_the_answer(chunk):
    """PIECES_GEMMA4_VISION_ATTN_CHUNK is a memory knob, not a numeric one."""
    jnp = pytest.importorskip("jax.numpy")
    q, k, v, mask, _ = _inputs(1008, 24)
    dense = _attention(q, k, v, mask, jnp.float32, jnp.bfloat16, chunk=0)
    blocked = _attention(q, k, v, mask, jnp.float32, jnp.bfloat16, chunk=chunk)
    assert np.array_equal(dense, blocked)


def test_padded_keys_are_excluded_and_padded_queries_still_attend():
    """The mask semantics the fix must preserve.

    transformers' ``padding_mask_function`` indexes the padding mask by
    ``kv_idx`` only, so a padded *query* row still attends to the valid keys
    (its output is dropped later by the pooler); what must never happen is a
    padded *key* entering any row's softmax.
    """
    jnp = pytest.importorskip("jax.numpy")
    n, pad = 504, 12
    q, k, v, mask, valid = _inputs(n, pad)
    # Poison the padded keys: if they leaked into the softmax the valid rows
    # would move.
    k_poisoned = k.copy()
    v_poisoned = v.copy()
    k_poisoned[:, :, n - pad:, :] *= 50.0
    v_poisoned[:, :, n - pad:, :] = 1e3
    clean = _attention(q, k, v, mask, jnp.float32, jnp.float32)
    poisoned = _attention(q, k_poisoned, v_poisoned, mask, jnp.float32,
                          jnp.float32)
    assert np.array_equal(clean[:, :, valid[0], :], poisoned[:, :,
                                                             valid[0], :])
    # ... and the padded query rows are computed, not left as garbage/NaN.
    assert np.isfinite(clean[:, :, ~valid[0], :]).all()


# --------------------------------------------------------------------- #
# AST: the fix itself, its gating and its packaging.
# --------------------------------------------------------------------- #


def test_both_score_operands_are_cast_to_fp32_before_the_matmul():
    """The whole fix: `.float()` on both sides of the score matmul."""
    src = _src(PATCH, "fp32_logit_attention")
    assert "key_f32 = key.float()" in src
    assert "query[:, :, start:stop].float()" in src
    assert "torch.matmul(query[:, :, start:stop].float()," in src


def test_the_softmax_runs_on_the_fp32_scores_and_only_probs_are_cast_down():
    src = _src(PATCH, "fp32_logit_attention")
    assert "torch.softmax(scores, dim=-1).to(value.dtype)" in src
    # The scores must not be narrowed before the softmax.
    assert ".to(torch.bfloat16)" not in src
    assert "scores.bfloat16()" not in src


def test_the_bool_mask_keeps_where_true():
    """A bool mask is True = attend; inverting it would mask everything."""
    src = _src(PATCH, "fp32_logit_attention")
    assert "torch.where(\n                    mask, scores," in src.replace(
        "\r\n", "\n")


def test_gqa_is_refused_rather_than_silently_mis_attended():
    src = _src(PATCH, "_vision_attention_forward")
    assert "key_states.shape[1] != query_states.shape[1]" in src
    assert "raise ValueError" in src


def test_the_patch_is_gated_on_its_env_and_on_a_vision_tower():
    src = _src(PATCH, "maybe_apply_gemma4_vision_attention_patch")
    assert "envs.PIECES_GEMMA4_VISION_ATTN_FP32" in src
    assert 'getattr(vllm_model, "vision_tower", None)' in src
    assert "envs.PIECES_GEMMA4_VISION_ATTN_CHUNK" in src


def test_the_patch_never_touches_the_attn_implementation_name():
    """transformers' mask builder keys off `_attn_implementation`.

    ``_preprocess_mask_arguments`` returns the *unbuilt* 2D mask for any
    implementation name it does not know, so renaming it away from "sdpa"
    would silently hand the tower a mask of the wrong rank.
    """
    tree = ast.parse(PATCH.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert node.attr != "_attn_implementation", (
                "the patch must leave config._attn_implementation alone")


def test_only_gemma4_vision_attention_modules_are_patched():
    src = PATCH.read_text()
    assert 'VISION_ATTENTION_CLASS = "Gemma4VisionAttention"' in src
    assert "type(m).__name__ == VISION_ATTENTION_CLASS" in src


def test_the_gemma4_patcher_applies_it_before_the_no_ple_early_return():
    """26B/31B have a vision tower but no PLE buffer to replace."""
    tree = ast.parse(MM_PATCHER.read_text())
    body = _fn(tree, "maybe_apply_gemma4_mm_patches").body
    called_at = ple_guard_at = None
    for i, stmt in enumerate(body):
        text = ast.dump(stmt)
        if "maybe_apply_gemma4_vision_attention_patch" in text:
            called_at = i
        if (isinstance(stmt, ast.If) and "ple_dim" in ast.dump(stmt.test)
                and any(isinstance(s, ast.Return) for s in stmt.body)):
            ple_guard_at = i
    assert called_at is not None, "the vision attention patch is never applied"
    assert ple_guard_at is not None, "the no-PLE early return is gone"
    assert called_at < ple_guard_at, (
        "the vision attention patch runs after the no-PLE early return, so "
        "the variants without PLE (26B/31B) would never get it")


def test_the_qwen_patchers_are_untouched():
    for name in ("qwen3_vl_patcher.py", "qwen3_omni_patcher.py"):
        text = (PATCH.parent / name).read_text()
        assert "gemma4_vision_attention" not in text
        assert "fp32_logit_attention" not in text


def test_both_envs_are_declared_and_registered():
    text = ENVS.read_text()
    for name in ("PIECES_GEMMA4_VISION_ATTN_FP32",
                 "PIECES_GEMMA4_VISION_ATTN_CHUNK"):
        assert f"    {name}: " in text, f"{name} missing from the dataclass"
        assert f'"{name}":' in text, f"{name} missing from the env mapping"
    assert 'env_bool("PIECES_GEMMA4_VISION_ATTN_FP32", default=False)' in text
    assert 'os.getenv("PIECES_GEMMA4_VISION_ATTN_CHUNK", "1024")' in text


def test_the_new_and_changed_modules_ship_in_the_p8_image():
    """A fix that is not in the image never reaches a chip."""
    for path in (P8_DOCKERFILE, P8_ASSEMBLE):
        text = path.read_text()
        for f in ("models/vllm/experimental/gemma4_vision_attention.py",
                  "models/vllm/experimental/gemma4_mm_patcher.py"):
            assert f in text, f"{f} not copied by {path.name}"
    assert "fp32_logit_attention" in P8_DOCKERFILE.read_text(), (
        "the Dockerfile must verify the fix landed, not just the file")


def test_the_patch_module_parses_and_imports_no_transformers_at_module_scope():
    """The census/CPU gate must be able to read this file without torch."""
    tree = ast.parse(PATCH.read_text())
    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            name = getattr(node, "module", None) or ""
            assert "transformers" not in name and "vllm" not in name, (
                f"module-scope import of {name}")


def test_the_forward_keeps_the_upstream_pre_attention_arithmetic():
    """Everything before the attention call is transformers', verbatim."""
    src = _src(PATCH, "_vision_attention_forward")
    for expected in (
            "hidden_shape = (*input_shape, -1, self.head_dim)",
            "cos, sin = position_embeddings",
            "query_states = self.q_proj(hidden_states).view(hidden_shape)",
            "query_states = self.q_norm(query_states)",
            "key_states = self.k_norm(key_states)",
            "value_states = self.v_norm(value_states)",
            "apply_multidimensional_rope(query_states, cos, sin",
            "attn_output = self.o_proj(attn_output)",
    ):
        assert expected in src, f"missing upstream line: {expected}"
    assert "self.scaling" in src, "the interface's scaling must be passed on"
