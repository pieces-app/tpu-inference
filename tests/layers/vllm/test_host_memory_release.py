"""Every `process_weights_after_loading` must release its host memory.

`delattr(layer, "weight")` frees NOTHING during an incremental load. These
methods inherit `maybe_process_weights`, so under VLLM_INCREMENTAL_FP8_LOADING
they run mid-load while the model's own
`params_dict = dict(self.named_parameters())` still holds a strong reference to
every original Parameter for the whole loop. That is why the offline and MoE
siblings resize the storage (`_free_torch_storage`) instead of relying on the
attribute going away, and why they end with `_release_host_memory()`
(gc.collect + jax.effects_barrier + malloc_trim) -- without the trim, glibc
keeps the arena and the pod's RSS never drops.

`VllmFp8OnlineLinearMethod` did neither. A 26B/31B bf16 checkpoint would
accumulate in host RAM for the entire load and the pod would be OOM-killed
mid-load or during first compile -- reading as "fp8 needs more host RAM"
rather than as a missing free.

Written as an invariant over ALL such methods rather than a check on the one
that was broken: the next sibling added should not be able to skip this
silently, which is precisely how this one did.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
FP8 = ROOT / "tpu_inference" / "layers" / "vllm" / "quantization" / "fp8.py"


def _methods():
    """(class name, FunctionDef) for every process_weights_after_loading.

    AUDIT 2026-09-03: this used to return `ast.get_source_segment(...)` TEXT
    and every assertion below was a substring match on it. Measured bypasses:
      * replacing the `_release_host_memory()` call with a COMMENT that
        mentions it -- passed;
      * inserting a bare `return` immediately before the call, making it dead
        code -- passed;
      * deleting `_free_torch_storage(p_weight)` from the online method (the
        exact bug this file was written for) while leaving
        `_free_torch_storage(p_bias)` -- passed.
    Returning the node instead lets the tests assert on CALLS, not on text.
    """
    src = FP8.read_text()
    tree = ast.parse(src)
    out = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for f in cls.body:
            if (isinstance(f, ast.FunctionDef)
                    and f.name == "process_weights_after_loading"):
                out.append((cls.name, f))
    return out


def _calls_named(fn, name):
    """Every ast.Call to a bare `name(...)` inside `fn`."""
    return [
        n for n in ast.walk(fn)
        if isinstance(n, ast.Call) and getattr(n.func, "id", None) == name
    ]


def test_there_are_methods_to_check():
    """A collector that silently finds nothing would make every test below
    vacuously green -- the same failure mode as a skip."""
    got = _methods()
    assert len(got) >= 3, (
        f"expected at least 3 process_weights_after_loading methods "
        f"(offline linear, online linear, MoE); found {[n for n, _ in got]}")


@pytest.mark.parametrize("name", [n for n, _ in _methods()])
def test_every_method_releases_host_memory(name):
    fn = dict(_methods())[name]
    stmts = [
        i for i, s in enumerate(fn.body)
        if isinstance(s, ast.Expr) and isinstance(s.value, ast.Call)
        and getattr(s.value.func, "id", None) == "_release_host_memory"
    ]
    assert stmts, (
        f"{name}.process_weights_after_loading has no _release_host_memory() "
        f"CALL statement (a comment naming it is not a call). Freeing tensor "
        f"storage is not enough -- without gc.collect + malloc_trim the pod's "
        f"RSS does not drop, and a 26B/31B load under the incremental loader "
        f"is OOM-killed.")
    returns = [i for i, s in enumerate(fn.body) if isinstance(s, ast.Return)]
    assert not any(r < stmts[-1] for r in returns), (
        f"{name}'s _release_host_memory() sits after an unconditional return "
        f"at the top level of the method: it is dead code.")


@pytest.mark.parametrize("name", [n for n, _ in _methods()])
def test_every_method_frees_the_param_it_replaces(name):
    fn = dict(_methods())[name]
    deleted = _calls_named(fn, "delattr")
    # AUDIT 2026-09-03: this was `pytest.skip(...)` when no delattr was found.
    # Measured: deleting BOTH the delattrs and the _free_torch_storage calls
    # from VllmFp8MoEMethod produced "24 passed, 1 skipped" -- strictly worse
    # code silently dropped off the gate. A method on this path that replaces
    # no loaded param does not belong on it, so say so instead of skipping.
    assert deleted, (
        f"{name}.process_weights_after_loading delattrs nothing -- it either "
        f"never replaces a loaded param (then it does not belong on this "
        f"path) or it stopped freeing one.")
    freed = [
        ast.unparse(c.args[0]) for c in _calls_named(fn, "_free_torch_storage")
        if c.args
    ]
    assert freed, (
        f"{name} delattrs {len(deleted)} param(s) without any "
        f"_free_torch_storage CALL. During an incremental load the model's "
        f"params_dict still holds a strong reference, so delattr frees "
        f"nothing and the bf16 checkpoint accumulates in host RAM.")
    # The WEIGHT specifically, not just the bias or a scale: freeing the small
    # tensors while leaving the checkpoint-sized one resident is the bug.
    assert any(
        "weight" in a.lower() or "w13" in a.lower() or "w2" in a.lower()
        for a in freed), (
            f"{name} frees {freed} -- none of which is the weight it "
            f"replaces. Freeing only the bias/scale leaves the "
            f"checkpoint-sized tensor resident for the whole load.")
