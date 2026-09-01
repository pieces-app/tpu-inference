"""`n_shards` must be the real TP degree, not the placeholder's 1.

`QuantLinearConfig.__init__` (jax) calls `super().__init__()` WITHOUT
`weight_sharding`, so the base computes `n_shards` from the placeholder
`P(None, None)` and gets 1. The real spec is assigned a few lines later and
`n_shards` was never recomputed.

Why that matters: `reorder_concatenated_tensor_for_sharding` becomes the
IDENTITY for every config built this way -- which includes `Fp8OnlineConfig`.
The bf16 control does not come through here (`unquantized.py` reads the
param's `out_sharding` and passes `weight_sharding=` into the common config),
so the control gets the true TP degree while the quantized arm does not.

The arithmetic stays correct either way, because load and apply share the same
stale value -- nothing raises. But at TP>1 the quantized arm ends up with a
DIFFERENT gate_up column layout than the control it is compared against, so
every throughput and latency delta between them carries a layout change nobody
asked for.

At TP=1 the stale value is accidentally correct. That is exactly why the
single-chip arms never surfaced this, and why the 4- and 8-chip fp8
comparisons are where it would have bitten.

This is a SOURCE-STRUCTURE test: constructing the real config needs a JaxEinsum
layer and a live mesh, which a CPU-only jax install cannot provide. It asserts
the recompute exists and is ordered AFTER the assignment -- the ordering IS the
bug, so an unordered assertion would pass on the broken code.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
CFG = ROOT / "tpu_inference" / "layers" / "jax" / "quantization" / "configs.py"


def _init_body():
    tree = ast.parse(CFG.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "QuantLinearConfig":
            for f in node.body:
                if isinstance(f, ast.FunctionDef) and f.name == "__init__":
                    return f
    pytest.fail("QuantLinearConfig.__init__ not found")


def _assign_lines(fn, attr):
    """Line numbers where `self.<attr> = ...` is assigned."""
    out = []
    for node in ast.walk(fn):
        if isinstance(node, ast.Assign):
            for t in node.targets:
                if (isinstance(t, ast.Attribute) and t.attr == attr
                        and isinstance(t.value, ast.Name)
                        and t.value.id == "self"):
                    out.append(node.lineno)
    return out


def test_n_shards_is_recomputed_in_the_jax_subclass():
    fn = _init_body()
    assert _assign_lines(fn, "n_shards"), (
        "QuantLinearConfig.__init__ never recomputes n_shards. The base class "
        "computed it from the placeholder P(None, None) before weight_sharding "
        "was known, so it is 1 regardless of the real TP degree.")


def test_the_recompute_happens_AFTER_weight_sharding_is_assigned():
    """The ordering is the whole bug: recomputing before the real spec is
    assigned reproduces the stale 1 while looking like a fix."""
    fn = _init_body()
    ws = _assign_lines(fn, "weight_sharding")
    ns = _assign_lines(fn, "n_shards")
    assert ws, "weight_sharding is never assigned in the jax subclass"
    assert ns, "n_shards is never recomputed (see the test above)"
    assert max(ns) > max(ws), (
        f"n_shards is recomputed at line {max(ns)} but weight_sharding is "
        f"last assigned at line {max(ws)} -- recomputing before the real spec "
        f"is known just reproduces the placeholder's n_shards=1")


def test_the_recompute_is_derived_from_sharding_not_a_literal():
    """Guard against a 'fix' that hardcodes a TP degree.

    AST-based rather than a line window: the recompute was later restructured
    to route through a safe `_out_axis` local (weight_sharding can be an EMPTY
    PartitionSpec for batched einsums, so indexing [1] blindly raises), and a
    proximity check would have failed on a correct change.
    """
    import ast
    fn = _init_body()
    src = CFG.read_text()
    body = ast.get_source_segment(src, fn) or ""

    # the assigned value must come from the mesh helper, not a constant
    assigns = [n for n in ast.walk(fn) if isinstance(n, ast.Assign)
               and any(isinstance(t, ast.Attribute) and t.attr == "n_shards"
                       for t in n.targets)]
    assert assigns, "n_shards is never assigned"
    call_src = ast.get_source_segment(src, assigns[-1]) or ""
    assert "get_mesh_shape_product" in call_src, (
        f"n_shards must come from the active mesh, got: {call_src.strip()!r}")
    assert not any(isinstance(assigns[-1].value, c)
                   for c in (ast.Constant, )), (
        "n_shards must not be a hardcoded constant")
    # and the axis it asks about must trace to a sharding, somewhere in __init__
    assert ("weight_sharding" in body or "out_features_sharding" in body), (
        "the recompute must derive its axis from the layer's sharding")
