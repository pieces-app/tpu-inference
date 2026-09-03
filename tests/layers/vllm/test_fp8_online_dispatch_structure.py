"""Structural guard for the opt-in online-fp8 dispatch (issue #158).

AST-based (not substring) so a commented-out or renamed dispatch cannot
satisfy it -- the earlier grep version passed with the call left in a comment,
which is the vacuous-marker trap. Dependency-free: runs on any CPython and is
the arm fork_gate reverts for the negative control.

Pins: the LinearBase non-serialized arm has (a) an `if envs.VLLM_FP8_ONLINE_DENSE`
branch that RETURNS a VllmFp8OnlineLinearMethod(...) call, and (b) still a
`raise` as the fail-closed default, with the opt-in return before the raise.
The behavioral proof (dispatch under the flag / raise without it) runs in the
in-image gate.
"""
import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
FP8_PATH = (REPO_ROOT / "tpu_inference" / "layers" / "vllm" / "quantization" /
            "fp8.py")


def _get_quant_method_fn():
    tree = ast.parse(FP8_PATH.read_text())
    cfg = next(n for n in ast.walk(tree)
               if isinstance(n, ast.ClassDef) and n.name == "VllmFp8Config")
    return next(
        n for n in ast.walk(cfg)
        if isinstance(n, ast.FunctionDef) and n.name == "get_quant_method")


def _online_return(fn):
    """The `return VllmFp8OnlineLinearMethod(...)` node, if present as real code."""
    for node in ast.walk(fn):
        if (isinstance(node, ast.Return) and isinstance(node.value, ast.Call)
                and getattr(node.value.func, "id",
                            None) == "VllmFp8OnlineLinearMethod"):
            return node
    return None


def _flag_guarded(fn):
    """True iff the online return sits under `if envs.VLLM_FP8_ONLINE_DENSE`."""
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        t = node.test
        if (isinstance(t, ast.Attribute) and t.attr == "VLLM_FP8_ONLINE_DENSE"
                and any(_online_return_in(b) for b in node.body)):
            return True
    return False


def _online_return_in(stmt):
    return any(
        isinstance(n, ast.Return) and isinstance(n.value, ast.Call)
        and getattr(n.value.func, "id", None) == "VllmFp8OnlineLinearMethod"
        for n in ast.walk(stmt))


def _fail_closed_raise(fn):
    for node in ast.walk(fn):
        if (isinstance(node, ast.If) and isinstance(node.test, ast.UnaryOp)
                and isinstance(node.test.op, ast.Not)):
            raises = [n for n in ast.walk(node) if isinstance(n, ast.Raise)]
            if raises:
                return raises[0]
    return None


def test_online_dispatch_is_flag_guarded():
    fn = _get_quant_method_fn()
    assert _online_return(fn) is not None, (
        "no real `return VllmFp8OnlineLinearMethod(...)` in get_quant_method")
    assert _flag_guarded(fn), (
        "the online dispatch must sit under `if envs.VLLM_FP8_ONLINE_DENSE`")


def test_fail_closed_default_preserved_and_after_optin():
    fn = _get_quant_method_fn()
    raise_node = _fail_closed_raise(fn)
    assert raise_node is not None, "the fail-closed raise must remain"
    online = _online_return(fn)
    assert online.lineno < raise_node.lineno, (
        "the opt-in return must precede the fail-closed raise")
    # the class exists and inherits the offline apply path
    tree = ast.parse(FP8_PATH.read_text())
    cls = next((n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)
                and n.name == "VllmFp8OnlineLinearMethod"), None)
    assert cls is not None
    assert any(
        getattr(b, "id", None) == "VllmFp8LinearMethod"
        for b in cls.bases), "must inherit VllmFp8LinearMethod (apply)"
    assert not any(
        isinstance(n, ast.FunctionDef) and n.name == "apply"
        for n in cls.body), "must inherit apply, not override it"
