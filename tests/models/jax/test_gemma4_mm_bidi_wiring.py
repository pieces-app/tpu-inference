"""Structural guards for issue #156: the flax_nnx Gemma-4 tower models must
FORWARD md.mm_bidi_ranges into attention(), gated to sliding-window layers,
and the wired file must actually be ON those models' execution path.

Why AST and why the reach test: the first fix for #156 patched an Attention
class no Gemma-4 model consumes -- its unit spy was green, its negative
control red, and real hardware showed byte-identical outputs because the
patched file never executed (verdict of the 2026-08-31 adversarial trace).
A wiring test is only meaningful alongside a proof that the wired code is
reachable from the target architectures.
"""
import ast
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[3]
GEMMA4 = ROOT / "tpu_inference" / "models" / "jax" / "gemma4.py"
IFACE = ROOT / "tpu_inference" / "layers" / "common" / "attention_interface.py"
LOADER = ROOT / "tpu_inference" / "models" / "common" / "model_loader.py"
GEMMA4_MM = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mm.py"


def _attention_calls(tree):
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "attention"):
            yield node


def test_gemma4_attention_call_forwards_mm_bidi_ranges():
    tree = ast.parse(GEMMA4.read_text())
    calls = [
        c for c in _attention_calls(tree)
        if any(k.arg == "update_kv_cache" for k in c.keywords)
    ]
    assert calls, "Gemma4Attention's attention(...) call not found"
    for call in calls:
        kw = {k.arg: k.value for k in call.keywords}
        assert "mm_bidi_ranges" in kw, (
            "attention(...) is called WITHOUT mm_bidi_ranges -- the runner "
            "builds ranges and this call site drops them (issue #156)")


def test_mm_bidi_ranges_is_gated_to_sliding_window_layers():
    tree = ast.parse(GEMMA4.read_text())
    gated = False
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        t = node.targets[0]
        if not (isinstance(t, ast.Name) and t.id == "mm_bidi_ranges"):
            continue
        v = node.value
        # (getattr(attention_metadata, "mm_bidi_ranges", None)
        #  if self.sliding_window is not None else None)
        if not isinstance(v, ast.IfExp):
            continue
        body_ok = (isinstance(v.body, ast.Call)
                   and isinstance(v.body.func, ast.Name)
                   and v.body.func.id == "getattr"
                   and len(v.body.args) == 3
                   and isinstance(v.body.args[1], ast.Constant)
                   and v.body.args[1].value == "mm_bidi_ranges")
        test_ok = (isinstance(v.test, ast.Compare)
                   and isinstance(v.test.left, ast.Attribute)
                   and v.test.left.attr == "sliding_window"
                   and len(v.test.ops) == 1
                   and isinstance(v.test.ops[0], ast.IsNot))
        orelse_ok = (isinstance(v.orelse, ast.Constant)
                     and v.orelse.value is None)
        if body_ok and test_ok and orelse_ok:
            gated = True
    assert gated, (
        "mm_bidi_ranges must be the sliding-window-gated getattr expression "
        "(HF composition AND(sliding_window, OR(causal, blockwise)); "
        "full-attention layers stay causal -- torchax reference "
        "layers/vllm/backends/flash_attn.py)")


def test_attention_interface_accepts_mm_bidi_ranges():
    tree = ast.parse(IFACE.read_text())
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "attention":
            names = [a.arg for a in node.args.args + node.args.kwonlyargs]
            assert "mm_bidi_ranges" in names, (
                "attention() lost its mm_bidi_ranges parameter -- the "
                "gemma4 wiring would TypeError at runtime")
            return
    raise AssertionError("attention() not found in attention_interface.py")


def test_wired_file_is_on_the_target_archs_execution_path():
    """REACH: the fix is worthless unless models/jax/gemma4.py is what the
    target architectures actually load. Asserts the flax_nnx registry maps
    Gemma4ForConditionalGeneration to the gemma4 jax module -- if this
    mapping moves, the wiring must be re-verified on the new path."""
    tree = ast.parse(LOADER.read_text())
    # The registry is populated by subscript assignment:
    #   _MODEL_REGISTRY["Gemma4ForConditionalGeneration"] = Gemma4ForConditionalGeneration
    # and the value name must be imported FROM the wired module
    # (tpu_inference.models.jax.gemma4) -- reach means "this arch loads the
    # file this test suite guards", not merely "the string appears".
    reg_value_name = None
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Assign) and len(node.targets) == 1):
            continue
        t = node.targets[0]
        if (isinstance(t, ast.Subscript)
                and isinstance(t.value, ast.Name)
                and t.value.id == "_MODEL_REGISTRY"
                and isinstance(t.slice, ast.Constant)
                and t.slice.value == "Gemma4ForConditionalGeneration"
                and isinstance(node.value, ast.Name)):
            reg_value_name = node.value.id
    assert reg_value_name, (
        "Gemma4ForConditionalGeneration is not registered in "
        "_MODEL_REGISTRY -- the wired gemma4.py may be dead code again "
        "(the exact failure mode of the first #156 fix)")
    imported_from_gemma4_mm = set()
    for node in ast.walk(tree):
        if (isinstance(node, ast.ImportFrom) and node.module
                and node.module.endswith("models.jax.gemma4_mm")):
            imported_from_gemma4_mm.update(a.name for a in node.names)
    assert reg_value_name in imported_from_gemma4_mm, (
        f"registry maps the arch to {reg_value_name!r}, which is not "
        "imported from tpu_inference.models.jax.gemma4_mm -- re-verify "
        "which module the mm arch actually loads")
    # Link 2: gemma4_mm composes its language model from the WIRED module.
    # (This test caught its own first bug here: the arch loads gemma4_mm,
    # and only composition makes gemma4.py's call site reachable.)
    mm_tree = ast.parse(GEMMA4_MM.read_text())
    mm_imports_gemma4 = set()
    for node in ast.walk(mm_tree):
        if (isinstance(node, ast.ImportFrom) and node.module
                and node.module.endswith("models.jax.gemma4")):
            mm_imports_gemma4.update(a.name for a in node.names)
    assert "Gemma4Model" in mm_imports_gemma4, (
        "gemma4_mm.py no longer imports Gemma4Model from models.jax.gemma4 "
        "-- the wired call site may be dead code for the mm arch")
    composed = any(
        isinstance(node, ast.Assign) and len(node.targets) == 1
        and isinstance(node.targets[0], ast.Attribute)
        and node.targets[0].attr == "language_model"
        and isinstance(node.value, ast.Call)
        and isinstance(node.value.func, ast.Name)
        and node.value.func.id == "Gemma4Model"
        for node in ast.walk(mm_tree))
    assert composed, (
        "gemma4_mm no longer builds language_model = Gemma4Model(...) -- "
        "the wired attention call site is not on the mm arch's path")
