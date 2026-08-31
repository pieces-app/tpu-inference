"""The flax_nnx JAX attention path must forward md.mm_bidi_ranges (issue #156).

Gemma-4 blockwise-bidirectional image attention is built by the runner
(`TPURunner._init_mm_bidi`) as `md.mm_bidi_ranges` (i32[max_num_seqs, 2]) and
consumed by the v3 RPA kernel. The torchax path forwards it
(`layers/vllm/backends/flash_attn.py`) and the generic
`layers/common/attention_interface.py` threads it as a replicated shard_map
operand. The flax_nnx path (`layers/jax/attention/attention.py::Attention.attention`)
historically did NOT — it called `ragged_paged_attention` with no
`mm_bidi_ranges`, so tower models (31B / 26B-A4B) served a causal-only mask
under `TPU_MM_BIDI_ATTENTION=force` and the operand was silently dropped.

Two tests, mirroring the fp8-guard file's dual structure:

* `test_source_threads_mm_bidi_ranges` — dependency-free structural guard
  (runs on any CPython, and is the arm the fleet CPU gate + fork_gate
  negative-control exercise): the source peels the operand
  (`*args, mm_ranges = args`), forwards it as the `mm_bidi_ranges=` kwarg, and
  appends it to the shard_map operands under `has_mm_bidi`. Negative control:
  drop any of those and this goes red (verified 2026-08-31).

* `test_forward_pass_forwards_mm_bidi_ranges` — behavioral spy that needs the
  fork's jax stack (skips without it; runs in the in-image gate with forced
  host devices). Monkeypatches `ragged_paged_attention` in the attention
  module with a trace-time capturing stub and asserts the kwarg arrives
  non-None when `md.mm_bidi_ranges` is set, and is absent when it is None.
"""

import py_compile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[4]
ATTN_PATH = (REPO_ROOT / "tpu_inference" / "layers" / "jax" / "attention" /
             "attention.py")


# ---------------------------------------------------------------------------
# Structural guard — no jax needed.
# ---------------------------------------------------------------------------
def test_attention_compiles():
    py_compile.compile(str(ATTN_PATH), doraise=True)


def test_source_threads_mm_bidi_ranges():
    src = ATTN_PATH.read_text()
    required = {
        "has_mm_bidi flag": "has_mm_bidi = md.mm_bidi_ranges is not None",
        "operand peel": "*args, mm_ranges = args",
        "kwarg forward": 'block_size_kwargs["mm_bidi_ranges"] = mm_ranges',
        "in_specs append": "in_specs = in_specs + (P(ShardingAxisName.ATTN_DATA), )",
        "operand append": "(md.mm_bidi_ranges, ) if has_mm_bidi else ()",
    }
    missing = [name for name, frag in required.items() if frag not in src]
    assert not missing, (
        f"the flax_nnx attention path no longer threads mm_bidi_ranges: "
        f"missing {missing}. Tower models would drop the blockwise image mask "
        f"(issue #156)."
    )


# ---------------------------------------------------------------------------
# Behavioral spy — needs the fork's jax/flax stack; skips elsewhere.
# ---------------------------------------------------------------------------
@pytest.fixture
def jax_stack():
    jax = pytest.importorskip("jax")
    jnp = pytest.importorskip("jax.numpy")
    pytest.importorskip("flax")
    import numpy as np
    from flax import nnx
    from jax.sharding import Mesh
    from tpu_inference.layers.common.attention_interface import \
        get_kv_cache_shape
    from tpu_inference.layers.common.attention_metadata import AttentionMetadata
    from tpu_inference.layers.jax.attention import attention as attn_mod
    from tpu_inference.layers.jax.attention.attention import Attention
    return dict(jax=jax, jnp=jnp, np=np, nnx=nnx, Mesh=Mesh,
               get_kv_cache_shape=get_kv_cache_shape,
               AttentionMetadata=AttentionMetadata, attn_mod=attn_mod,
               Attention=Attention)


def _run_forward(s, with_mm_bidi, monkeypatch):
    jax, jnp, np, nnx = s["jax"], s["jnp"], s["np"], s["nnx"]
    mesh = s["Mesh"](np.array(jax.devices()[:1]).reshape(1, 1, 1, -1),
                     axis_names=("data", "attn_dp", "expert", "model"))
    captured = {}

    def _spy(*args, **kwargs):
        # Runs once at trace time; record whether the mask operand arrived.
        captured["seen"] = "mm_bidi_ranges" in kwargs
        captured["value_is_none"] = kwargs.get("mm_bidi_ranges") is None
        out = jnp.zeros_like(args[0])          # q_TNH-shaped output
        return out, args[3]                    # (output, kv_cache)

    monkeypatch.setattr(s["attn_mod"], "ragged_paged_attention", _spy)

    hidden, nheads = 1024, 8
    hd = hidden // nheads
    with jax.set_mesh(mesh):
        attention = s["Attention"](hidden_size=hidden, num_attention_heads=nheads,
                                   num_key_value_heads=nheads, head_dim=hd,
                                   rope_theta=10000.0, rope_scaling={},
                                   dtype=jnp.bfloat16, mesh=mesh,
                                   random_init=True, rngs=nnx.Rngs(42),
                                   kv_cache_dtype="auto")
        seq_len, block_size, num_blocks = 64, 16, 8
        x = jnp.ones((seq_len, hidden), dtype=jnp.bfloat16)
        kv = jnp.zeros(s["get_kv_cache_shape"](num_blocks, block_size, nheads, hd,
                                               jnp.bfloat16), dtype=jnp.bfloat16)
        md_kwargs = dict(
            input_positions=jnp.arange(seq_len, dtype=jnp.int32),
            block_tables=jnp.arange(seq_len // block_size, dtype=jnp.int32),
            seq_lens=jnp.array([seq_len], dtype=jnp.int32),
            query_start_loc=jnp.array([0, seq_len], dtype=jnp.int32),
            request_distribution=jnp.array([0, 0, 1], dtype=jnp.int32),
        )
        if with_mm_bidi:
            md_kwargs["mm_bidi_ranges"] = jnp.array([[0, 4]], dtype=jnp.int32)
        md = s["AttentionMetadata"](**md_kwargs)
        attention(x, is_prefill=True, kv_cache=kv, attention_metadata=md)
    return captured


def test_forward_pass_forwards_mm_bidi_ranges(jax_stack, monkeypatch):
    seen = _run_forward(jax_stack, with_mm_bidi=True, monkeypatch=monkeypatch)
    assert seen.get("seen") is True and seen.get("value_is_none") is False, (
        "attention() did not forward md.mm_bidi_ranges to the kernel")


def test_forward_pass_omits_mm_bidi_when_unset(jax_stack, monkeypatch):
    seen = _run_forward(jax_stack, with_mm_bidi=False, monkeypatch=monkeypatch)
    assert not seen.get("seen", False), (
        "attention() passed mm_bidi_ranges when metadata had none")
