"""Regression test for the gemma4_mm compute_logits engine kill (fixed in p9).

Upstream vLLM's Gemma4Unified `compute_logits` cached the suppressed-token
index tensor in module state. Under `jax.jit` (torchax on TPU) the first
trace stores a trace-bound tensor, and the AOT warmup `.lower()` at the
SECOND batch bucket silently reuses it — a poisoned lowering. The engine
then dies with `UnexpectedTracerError` on the first batch that pads to that
bucket (observed live 2026-08-26: C=12 killed gemma-4-12B in seconds, while
C<=8 stayed safe only because MIN_NUM_SEQS=8 pads every small batch to the
one un-poisoned first shape).

This models both code shapes on CPU jax in under a second — no TPU needed —
so it can gate any re-vendor of the patched file. It intentionally does NOT
import vllm: the failure is a jax-tracing property of the caching pattern,
and the string-level guard on the vendored file lives in the p9 Dockerfile.
"""

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

SUPPRESS = [2, 3]  # stand-in for generation_config.suppress_tokens (int32[2])


def _make(cached: bool):
    """Model of compute_logits: `cached` mirrors upstream's module-state
    cache; `not cached` mirrors the p9 fix (rebuild every call)."""

    class Model:

        def __init__(self):
            self._cache = None

        def compute_logits(self, logits):
            if cached:
                if self._cache is None:
                    self._cache = jnp.asarray(np.array(SUPPRESS))
                idx = self._cache
            else:
                idx = jnp.asarray(np.array(SUPPRESS))
            return logits.at[:, idx].set(-np.inf)

    m = Model()
    return m, jax.jit(m.compute_logits)


def _suppressed_correctly(result) -> bool:
    r = np.asarray(result)
    return bool(
        np.isinf(r[:, SUPPRESS[0]]).all()
        and np.isinf(r[:, SUPPRESS[1]]).all() and not np.isinf(r[:, 0]).any())


def _warmup(fn, batch):
    fn.lower(jax.ShapeDtypeStruct((batch, 16), jnp.float32)).compile()


def test_module_state_cache_dies_on_second_bucket_after_warmup():
    """The production sequence: warmup lowers buckets [8, 16], serving at
    C<=8 works, the first batch padding to 16 raises. The second lower()
    does NOT raise — the poisoning is silent until serve time."""
    _, fn = _make(cached=True)
    _warmup(fn, 8)
    _warmup(fn, 16)  # silently reuses the tracer cached by the first lower
    assert _suppressed_correctly(fn(jnp.zeros((8, 16))))
    with pytest.raises(jax.errors.UnexpectedTracerError):
        fn(jnp.zeros((16, 16)))


def test_module_state_cache_dies_on_shape_change_without_warmup():
    """SKIP_JAX_PRECOMPILE variant: repeat same-shape calls are safe cache
    hits; the first new shape re-traces and reuses the stale tracer."""
    _, fn = _make(cached=True)
    assert _suppressed_correctly(fn(jnp.zeros((8, 16))))
    assert _suppressed_correctly(fn(jnp.zeros((8, 16))))
    with pytest.raises(jax.errors.UnexpectedTracerError):
        fn(jnp.zeros((16, 16)))


def test_rebuild_per_call_survives_all_shapes_with_correct_output():
    """The p9 fix: rebuilding the index tensor every call constant-folds
    under jit — every bucket compiles, outputs stay correct, and repeat
    calls do not retrace."""
    _, fn = _make(cached=False)
    _warmup(fn, 8)
    _warmup(fn, 16)
    assert _suppressed_correctly(fn(jnp.zeros((8, 16))))
    assert _suppressed_correctly(fn(jnp.zeros((16, 16))))
    assert _suppressed_correctly(fn(jnp.zeros((16, 16))))
