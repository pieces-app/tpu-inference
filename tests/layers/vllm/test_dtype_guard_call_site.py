"""assert_torchax_representable must be CALLED where the torchax online path
selects its dtype -- not merely defined and unit-tested.

Review 2026-09-02: the guard helper had tests, the call site had none;
deleting the one call in VllmFp8Config.get_quant_method left 88/88 green
while e4m3b11fnuz would again reach j2t_dtype and kill the engine at boot.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
FP8 = ROOT / "tpu_inference" / "layers" / "vllm" / "quantization" / "fp8.py"


def _enclosing_funcs(tree):
    out = {}
    for fn in ast.walk(tree):
        if isinstance(fn, ast.FunctionDef):
            for node in ast.walk(fn):
                out.setdefault(id(node), []).append(fn.name)
    return out


def test_get_quant_method_calls_the_guard_on_the_selected_dtype():
    tree = ast.parse(FP8.read_text())
    enclosing = _enclosing_funcs(tree)
    hits = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "assert_torchax_representable"):
            arg_is_dtype_call = (node.args
                                 and isinstance(node.args[0], ast.Call)
                                 and isinstance(node.args[0].func, ast.Name)
                                 and node.args[0].func.id
                                 == "online_quant_dtype")
            hits.append((enclosing.get(id(node), []), arg_is_dtype_call))
    assert hits, "assert_torchax_representable is never called in the vllm fp8 path"
    assert any("get_quant_method" in fns and ok for fns, ok in hits), (
        f"the guard is not called on online_quant_dtype() inside get_quant_method: {hits}"
    )
