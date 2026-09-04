"""No module may READ a `.kernel` attribute off one of this repo's layer
wrappers, because those wrappers delete it.

MEASURED 2026-09-02 17:06Z: the native 12B (`gemma4_unified.py:406`) read
`embed_audio.embedding_projection.kernel.value.dtype`. `JaxEinsum.__init__`
does `self.weight = self.kernel` then `delattr(self, 'kernel')`, so the read
raised AttributeError inside the model-execute path and took down EngineCore
(EngineDeadError, HTTP 500, container restart). Text and vision had been
serving for 12 minutes; only a live AUDIO request reaches this line, so every
CPU test, the boot gate and the compile-time precompilation all passed.

Two guards, because either alone is escapable:
  1. the class contract (behavioral, real flax): `weight` exists, `kernel`
     does not, on every wrapper that aliases;
  2. no `.kernel` attribute read anywhere in the jax model/layer sources
     (AST, so the string literals in the HF weight-mapping tables -- which
     legitimately contain "...kernel" -- are not confused for attribute reads).
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
JAX_SRC = sorted((ROOT / "tpu_inference" / "models" / "jax").rglob("*.py")) + \
          sorted((ROOT / "tpu_inference" / "layers" / "jax").rglob("*.py"))


def _aliasing_classes():
    """Classes that do `self.weight = self.kernel` in their __init__."""
    out = []
    for f in JAX_SRC:
        tree = ast.parse(f.read_text())
        for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
            src = ast.unparse(cls)
            if "self.weight = self.kernel" in src:
                out.append((f.name, cls.name, "delattr(self, 'kernel')"
                            in src))
    return out


def test_the_aliasing_classes_are_known_and_delete_kernel():
    found = _aliasing_classes()
    assert found, "no class aliases weight = kernel any more -- this guard is stale, delete or update it"
    for fname, cls, deletes in found:
        assert deletes, (
            f"{fname}::{cls} aliases weight = kernel but does NOT delattr kernel. "
            "Mixed contracts across wrappers are exactly how gemma4_unified.py:406 "
            "was written against the wrong one; either delete there too or update this test."
        )


def test_the_alias_lines_this_replays_are_still_the_shipped_ones():
    """AUDIT 2026-09-03: this was `test_jax_einsum_really_has_weight_and_not_
    kernel`, whose docstring called it "the actual contract". It was not: it
    built a VANILLA upstream `nnx.Einsum` and then performed the alias and the
    delete IN THE TEST BODY, so it asserted that `delattr` deletes an
    attribute. Measured: removing `delattr(self, 'kernel')` from the repo's
    real JaxEinsum left it green (the AST test at the bottom of this file
    caught that). It exercised flax, not this repo.

    Two lines are replayed below, so pin that they are still the lines the
    wrapper runs -- and in that order, because the alias must precede the
    delete or `weight` is never bound. Then keep the upstream-premise check,
    which is the part that genuinely cannot be asserted from source.
    """
    pytest.importorskip("flax")
    jax = pytest.importorskip("jax")
    from flax import nnx
    src = (ROOT / "tpu_inference" / "layers" / "jax" / "linear.py").read_text()
    i = src.find("self.weight = self.kernel")
    j = src.find("delattr(self, 'kernel')")
    assert i != -1, "the wrapper no longer aliases self.kernel to self.weight"
    assert j != -1, "the wrapper no longer deletes self.kernel"
    assert i < j, "the alias must precede the delete, or `weight` is never bound"

    m = nnx.Einsum(einsum_str="bd,dh->bh",
                   kernel_shape=(4, 8),
                   rngs=nnx.Rngs(0))
    assert hasattr(
        m, "kernel"
    ), "upstream nnx.Einsum no longer names it kernel; this test's premise moved"
    # Our wrapper renames it. Reproduce the two lines the wrapper runs.
    m.weight = m.kernel
    delattr(m, "kernel")
    assert hasattr(m, "weight") and not hasattr(m, "kernel")
    # `.value` is deprecated in this flax; read the array the supported way.
    assert m.weight[...].dtype == jax.numpy.float32


def _kernel_attribute_reads(path):
    """`x.kernel` as an ATTRIBUTE (never a string literal), excluding the
    assignment/alias lines inside the wrappers themselves."""
    tree = ast.parse(path.read_text())
    assigned = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            for t in n.targets:
                if isinstance(t, ast.Attribute) and t.attr == "kernel":
                    assigned.add(id(t))
    # AUDIT 2026-09-03: the `self` exemption used to be unconditional, which
    # excused EVERY `self.kernel` read in every file under models/jax and
    # layers/jax -- not just the alias lines it was meant to cover. Measured:
    # adding `0 * self.kernel.value.mean()` to Gemma4UnifiedVisionEmbedder.
    # __call__ left the gate green; that is bug ti #40 exactly, spelled with
    # `self.`. Narrow the exemption to the __init__ of a class that actually
    # aliases, which is the only place the read is legitimate.
    aliasing = set()
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        if "self.weight = self.kernel" not in ast.unparse(cls):
            continue
        init = next(
            (f for f in cls.body
             if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
        if init is not None:
            aliasing |= {id(n) for n in ast.walk(init)}
    hits = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and n.attr == "kernel" and id(
                n) not in assigned:
            # the wrapper's own `self.weight = self.kernel` read is legitimate
            if (isinstance(n.value, ast.Name) and n.value.id == "self"
                    and id(n) in aliasing):
                continue
            hits.append(n.lineno)
    return hits


def test_no_module_reads_dot_kernel_off_a_wrapper():
    bad = {}
    for f in JAX_SRC:
        hits = _kernel_attribute_reads(f)
        if hits:
            bad[str(f.relative_to(ROOT))] = hits
    assert not bad, (
        "`.kernel` is read as an attribute here, but the wrappers delete it "
        f"(AttributeError at runtime, on whichever path first reaches it): {bad}"
    )
