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
"""TPU test: the PALLAS RPA v3 kernel with mm_bidi_ranges.

Complements ``test_rpa_mm_bidi_ref.py``, which only pins the pure-JAX
reference. This exercises the parts that only exist in the Pallas kernel
and are therefore invisible to a reference-only test:

  * the ``effective_kv_len`` extension (in-block queries must FETCH the
    forward KV blocks, not merely be unmasked for them) — covered by
    ``blk`` ranges that extend past the query block boundary;
  * the static operand shim in ``_ragged_paged_attention_kernel``
    (``has_mm_bidi`` inserting ``None``) and the resulting
    ``input_output_aliases`` index shift — covered by asserting that a
    zero-range operand reproduces the no-operand result BITWISE, which
    fails loudly if q/kv_cache aliasing lands on the wrong operand;
  * the int16 narrowing of the span arithmetic on v6+.

The oracle is written independently of the implementation: it builds a
dense [q, kv] boolean mask and runs a plain fp32 softmax attention, rather
than re-deriving the kernel's streaming/flash formulation.

Requires a TPU (Pallas has no CPU lowering); skipped otherwise.
"""
import numpy as np
import pytest

jax = pytest.importorskip("jax")
import jax.numpy as jnp  # noqa: E402

pytestmark = pytest.mark.skipif(
    jax.default_backend() != "tpu",
    reason="Pallas RPA kernel requires a TPU backend",
)

# Imported lazily inside the helper: on a machine without vLLM installed the
# tpu_inference package import chain fails, and a module-level import would
# turn a clean skip into a COLLECTION ERROR.


def _dense_oracle(q, k, v, kv_len, q_len, sm_scale, sliding_window, blk):
    """Independent dense-mask attention: AND(sw, OR(causal, blockwise))."""
    kv_gap = kv_len - q_len
    q_pos = (kv_gap + np.arange(q_len))[:, None]
    kv_pos = np.arange(kv_len)[None, :]

    allow = q_pos >= kv_pos  # causal
    if blk is not None and blk != (0, 0):
        s, e = blk
        inside = ((q_pos >= s) & (q_pos < e) & (kv_pos >= s) & (kv_pos < e))
        allow = allow | inside
    if sliding_window is not None:
        allow = allow & (q_pos < kv_pos + sliding_window)

    scores = np.einsum("qhd,khd->hqk", q.astype(np.float32),
                       k.astype(np.float32)) * sm_scale
    scores = np.where(allow[None], scores, -3.0e38)
    scores -= scores.max(axis=-1, keepdims=True)
    p = np.exp(scores)
    p /= p.sum(axis=-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v.astype(np.float32))


def _run_kernel(q,
                k,
                v,
                kv_len,
                mm,
                *,
                sliding_window,
                page_size=16,
                pages_per_seq=16,
                max_num_seqs=1):
    from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
        ragged_paged_attention

    num_kv_heads = k.shape[1]
    head_dim = q.shape[2]
    total_pages = max_num_seqs * pages_per_seq
    kv_cache = jnp.zeros(
        (total_pages, page_size, num_kv_heads * 2, 1, head_dim),
        dtype=jnp.bfloat16)
    out, _ = ragged_paged_attention(
        jnp.asarray(q, jnp.bfloat16),
        jnp.asarray(k, jnp.bfloat16),
        jnp.asarray(v, jnp.bfloat16),
        kv_cache,
        jnp.asarray([kv_len] + [0] * (max_num_seqs - 1), jnp.int32),
        jnp.arange(max_num_seqs * pages_per_seq, dtype=jnp.int32),
        jnp.asarray([0, q.shape[0]] + [q.shape[0]] * (max_num_seqs - 1),
                    jnp.int32),
        jnp.asarray([0, 1, 1], jnp.int32),
        jnp.asarray(mm, jnp.int32) if mm is not None else None,
        sm_scale=head_dim**-0.5,
        sliding_window=sliding_window,
        out_dtype=jnp.float32,
    )
    return np.asarray(out)[:q.shape[0]]


@pytest.mark.parametrize("sliding_window", [None, 32])
@pytest.mark.parametrize(
    "blk",
    [
        (0, 0),  # sentinel -> pure causal
        (16, 80),  # block spans multiple bq/bkv blocks (KV extension)
        (8, 120),  # block wider than the sliding window
        (96, 128),  # block at the tail
    ],
)
def test_kernel_matches_dense_oracle(sliding_window, blk):
    rng = np.random.default_rng(7)
    q_len = kv_len = 128
    num_heads, head_dim = 4, 128

    q = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)

    got = _run_kernel(q,
                      k,
                      v,
                      kv_len,
                      np.array([blk], np.int32),
                      sliding_window=sliding_window)
    want = _dense_oracle(q, k, v, kv_len, q_len, head_dim**-0.5,
                         sliding_window, blk)

    # bf16 inputs through a flash-style accumulation: loose but far tighter
    # than the ~O(1) differences a wrong mask produces.
    np.testing.assert_allclose(got, want, rtol=2e-2, atol=2e-2)


@pytest.mark.parametrize("sliding_window", [None, 32])
def test_zero_range_operand_is_bitwise_identical_to_no_operand(sliding_window):
    """Static shim + aliasing-index shift must not perturb anything.

    A zero (0, 0) range is semantically a no-op, so passing the operand
    must reproduce the no-operand result EXACTLY. Any drift here means the
    extra scalar-prefetch operand shifted `input_output_aliases` onto the
    wrong buffer.
    """
    rng = np.random.default_rng(11)
    q_len = kv_len = 128
    num_heads, head_dim = 4, 128
    q = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)

    without = _run_kernel(q, k, v, kv_len, None, sliding_window=sliding_window)
    with_zero = _run_kernel(q,
                            k,
                            v,
                            kv_len,
                            np.array([[0, 0]], np.int32),
                            sliding_window=sliding_window)
    np.testing.assert_array_equal(without, with_zero)


def test_block_actually_changes_in_block_tokens_only():
    """In-block queries must change; post-block queries must not."""
    rng = np.random.default_rng(13)
    q_len = kv_len = 128
    num_heads, head_dim = 4, 128
    q = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)

    causal = _run_kernel(q, k, v, kv_len, None, sliding_window=64)
    bidi = _run_kernel(q,
                       k,
                       v,
                       kv_len,
                       np.array([[16, 80]], np.int32),
                       sliding_window=64)

    # Inside the block, queries gain forward context -> must differ.
    assert np.abs(causal[16:80] - bidi[16:80]).max() > 1e-2
    # After the block, the mask is unchanged -> must match.
    np.testing.assert_allclose(causal[80:], bidi[80:], rtol=2e-2, atol=2e-2)
