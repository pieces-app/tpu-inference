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
"""CPU test: mm_bidi_ranges (PrefixLM blockwise) mask semantics of the RPA
v3 reference implementation against an oracle dense-mask attention built the
HF Gemma-4 way: AND(sliding_window, OR(causal, blockwise)).

The Pallas kernel mirrors the reference's mask math line for line (same
q_span/k_span formulation, same OR-then-AND composition), so this pins the
semantics; kernel-vs-reference equivalence is covered by the existing RPA
kernel tests plus on-TPU validation.

Run: pytest tests/kernels/test_rpa_mm_bidi_ref.py  (works on CPU jax)
"""
import importlib.util
import logging
import sys
import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# On CPU dev machines without a vLLM install, stub the two vllm symbols the
# tpu_inference import chain touches so the pure-JAX kernel module loads.
if importlib.util.find_spec("vllm") is None:  # pragma: no cover - dev only
    vllm_stub = types.ModuleType("vllm")
    vllm_logger_stub = types.ModuleType("vllm.logger")

    class _VllmLogger(logging.Logger):
        pass

    def init_logger(name):
        return logging.getLogger(name)

    vllm_logger_stub._VllmLogger = _VllmLogger
    vllm_logger_stub.init_logger = init_logger
    vllm_stub.logger = vllm_logger_stub
    sys.modules.setdefault("vllm", vllm_stub)
    sys.modules.setdefault("vllm.logger", vllm_logger_stub)

from tpu_inference.kernels.ragged_paged_attention.v3.kernel import \
    ref_ragged_paged_attention

jax.config.update("jax_platform_name", "cpu")


def _oracle(q, k, v, kv_len, q_len, sm_scale, sliding_window, blk):
    """Dense attention with the HF Gemma-4 mask composition."""
    kv_gap = kv_len - q_len
    q_pos = kv_gap + np.arange(q_len)[:, None]  # absolute q positions
    kv_pos = np.arange(kv_len)[None, :]
    causal = q_pos >= kv_pos
    if blk is not None:
        s, e = blk
        in_block = ((q_pos >= s) & (q_pos < e) & (kv_pos >= s) & (kv_pos < e))
        causal = causal | in_block
    mask = causal
    if sliding_window is not None:
        mask = mask & (q_pos < kv_pos + sliding_window)

    scores = np.einsum("qhd,khd->hqk", q, k) * sm_scale
    scores = np.where(mask[None], scores, -1e30)
    p = np.exp(scores - scores.max(-1, keepdims=True))
    p /= p.sum(-1, keepdims=True)
    return np.einsum("hqk,khd->qhd", p, v)


@pytest.mark.parametrize("sliding_window", [None, 16])
@pytest.mark.parametrize(
    "blk",
    [
        None,  # no block -> pure causal
        (0, 0),  # sentinel -> pure causal
        (8, 40),  # block inside the query span
        (2, 70),  # block longer than the sliding window
    ],
)
def test_ref_mm_bidi_matches_dense_oracle(sliding_window, blk):
    rng = np.random.default_rng(0)
    q_len = kv_len = 96  # fresh prefill: q covers the whole sequence
    num_q_heads, num_kv_heads, head_dim = 4, 2, 128
    page_size = 16
    max_num_seqs = 2
    pages_per_seq = 8
    sm_scale = head_dim**-0.5

    q = rng.standard_normal((q_len, num_q_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((q_len, num_kv_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((q_len, num_kv_heads, head_dim)).astype(np.float32)

    total_pages = max_num_seqs * pages_per_seq
    kv_cache = np.zeros((total_pages, page_size, num_kv_heads * 2, 1, head_dim),
                        dtype=np.float32)
    kv_lens = np.array([kv_len, 0], dtype=np.int32)
    page_indices = np.arange(max_num_seqs * pages_per_seq, dtype=np.int32)
    cu_q_lens = np.array([0, q_len, q_len], dtype=np.int32)
    distribution = np.array([0, 1, 1], dtype=np.int32)

    mm = None
    if blk is not None:
        mm = np.zeros((max_num_seqs, 2), dtype=np.int32)
        mm[0] = blk

    out, _ = ref_ragged_paged_attention(
        jnp.asarray(q),
        jnp.asarray(k),
        jnp.asarray(v),
        jnp.asarray(kv_cache),
        jnp.asarray(kv_lens),
        jnp.asarray(page_indices),
        jnp.asarray(cu_q_lens),
        jnp.asarray(distribution),
        jnp.asarray(mm) if mm is not None else None,
        sm_scale=sm_scale,
        sliding_window=sliding_window,
        out_dtype=jnp.float32,
    )
    got = np.asarray(out)[:q_len]

    q_rep = np.repeat(q, 1, axis=1)
    k_rep = np.repeat(k, num_q_heads // num_kv_heads, axis=1)
    v_rep = np.repeat(v, num_q_heads // num_kv_heads, axis=1)
    blk_eff = None if blk is None or blk == (0, 0) else blk
    want = _oracle(
        q.transpose(0, 1, 2),
        k_rep,
        v_rep,
        kv_len,
        q_len,
        sm_scale,
        sliding_window,
        blk_eff,
    )

    np.testing.assert_allclose(got, want, rtol=2e-3, atol=2e-3)


def test_bidi_actually_changes_output():
    """Sanity: a (8, 40) block must differ from pure causal."""
    rng = np.random.default_rng(1)
    q_len = kv_len = 64
    num_heads, head_dim, page_size = 2, 128, 16
    q = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    k = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    v = rng.standard_normal((q_len, num_heads, head_dim)).astype(np.float32)
    kv_cache = np.zeros((8, page_size, num_heads * 2, 1, head_dim),
                        dtype=np.float32)
    kv_lens = np.array([kv_len], dtype=np.int32)
    page_indices = np.arange(8, dtype=np.int32)
    cu_q_lens = np.array([0, q_len], dtype=np.int32)
    distribution = np.array([0, 1, 1], dtype=np.int32)

    def run(mm):
        out, _ = ref_ragged_paged_attention(
            jnp.asarray(q), jnp.asarray(k), jnp.asarray(v),
            jnp.asarray(kv_cache), jnp.asarray(kv_lens),
            jnp.asarray(page_indices), jnp.asarray(cu_q_lens),
            jnp.asarray(distribution),
            jnp.asarray(mm) if mm is not None else None,
            sm_scale=head_dim**-0.5,
            sliding_window=32,
            out_dtype=jnp.float32)
        return np.asarray(out)[:q_len]

    causal = run(None)
    bidi = run(np.array([[8, 40]], dtype=np.int32))
    # Tokens inside the block see future context -> must change.
    assert np.abs(causal[8:40] - bidi[8:40]).max() > 1e-3
    # Tokens after the block end see only the past -> unchanged.
    np.testing.assert_allclose(causal[40:], bidi[40:], rtol=1e-5, atol=1e-5)
