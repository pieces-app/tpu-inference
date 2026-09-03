"""The host-quant hook must be WIRED, not merely present.

The leaf (online_host_quant) is executed by the CPU gate; the three call
sites that connect it to a live load are behind imports of vllm/torch and
cannot be executed here, so their shape is pinned structurally:

  1. assign_and_shard_param -- the one choke point every loader path (plain
     `load_nnx_param_from_reshaped_torch`, the merged-shard loaders, jax_dummy)
     goes through -- tries the host hook BEFORE placing bf16, and the bf16
     placement is its `else`;
  2. Fp8OnlineLinearMethod.create_weights_jax requests host quantization with
     the 2-D layout and spec the APPLY path assumes (kernel_shape,
     linear_config.weight_sharding), scale on the per-OUTPUT axis;
  3. process_weights_after_loading adopts the parked scale BEFORE any
     on-device quantize_tensor call -- the device requant is the fallback for
     a kernel that was already on the mesh, never the first choice.

Live check (no TPU here): the arm's log must show
"online quant: kernels quantized on the host before placement" and the
engagement marker; a boot that shows only the marker took the fallback.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
WU = ROOT / "tpu_inference" / "models" / "jax" / "utils" / "weight_utils.py"
FP8 = ROOT / "tpu_inference" / "layers" / "jax" / "quantization" / "fp8.py"
UNQ = ROOT / "tpu_inference" / "layers" / "jax" / "quantization" / "unquantized.py"
LEAF = ROOT / "tpu_inference" / "layers" / "common" / "quantization" / "online_host_quant.py"


def _code_only(text):
    return "\n".join(l.split("#")[0] for l in text.splitlines())


def _fn(path, name, cls=None):
    src = path.read_text()
    tree = ast.parse(src)
    scope = tree
    if cls is not None:
        scope = next(n for n in ast.walk(tree) if isinstance(n, ast.ClassDef) and n.name == cls)
    for n in (scope.body if cls else ast.walk(tree)):
        if isinstance(n, ast.FunctionDef) and n.name == name:
            return _code_only(ast.get_source_segment(src, n) or "")
    pytest.fail(f"{cls + '.' if cls else ''}{name} not found in {path.name}")


# ------------------------------------------------------- 1. the choke point
def test_assign_and_shard_param_tries_the_host_hook_before_placing_bf16():
    body = _fn(WU, "assign_and_shard_param")
    assert "place_host_quantized(" in body, "the loader never asks the hook"
    assert body.index("place_host_quantized(") < body.index("set_value(shard_put("), (
        "bf16 is placed before the hook runs: the whole point is that it never is")
    assert "if not placed:" in body, "the bf16 placement must be the else of the hook, never both"
    assert "put=lambda" in body and "shard_put(" in body, "placement must stay shard_put's"


def test_every_loader_path_ends_in_assign_and_shard_param():
    """One hook covers all paths only if all paths go through it."""
    assert "assign_and_shard_param(jax_param, jax_weight" in _fn(WU, "load_nnx_param_from_reshaped_torch")
    assert "assign_and_shard_param(param, fused" in _fn(UNQ, "_load_merged_tensor", "UnquantizedMergedLinearMethod")
    assert "assign_and_shard_param(param, dummy_weight" in _fn(WU, "load_weights", "JaxDummyModelLoader")


# ------------------------------------------------------- 2. the request
def test_create_weights_requests_host_quant_with_the_apply_paths_layout():
    body = _fn(FP8, "create_weights_jax", "Fp8OnlineLinearMethod")
    assert "request_host_quant(" in body
    call = body[body.index("HostQuantRequest("):]
    assert "kernel_shape=tuple(self.kernel_shape)" in call, "the placed kernel must be the 2-D [in, out] the apply path serves"
    assert "weight_spec=tuple(self.weight_sharding)" in call, "placement spec must be linear_config.weight_sharding (what sharded_quantized_matmul assumes)"
    assert "scale_spec=(self.weight_sharding[1]" in call, (
        "the per-OUTPUT scale follows the OUT axis; [0] is the Fp8Tensorwise wart of sharding it along the INPUT axis")
    assert "if not self.batch_features:" in body, "batched kernels keep the fail-closed refusal in process_weights"
    assert "layer" not in call, (
        "the request must not capture the layer: nnx.eval_shape re-merges the abstract model into NEW "
        "module objects, so a layer captured at construction is stale by load time")


def test_the_request_rides_on_param_metadata_not_on_the_layer():
    src = LEAF.read_text()
    assert 'HOST_QUANT_REQUEST = "_online_host_quant"' in src
    assert 'HOST_QUANT_SCALE = "_online_host_quant_scale"' in src
    fields = _fn(LEAF, "__init__", "HostQuantRequest") if False else None  # dataclass: no __init__ in source
    cls = next(n for n in ast.walk(ast.parse(src)) if isinstance(n, ast.ClassDef) and n.name == "HostQuantRequest")
    names = [n.target.id for n in cls.body if isinstance(n, ast.AnnAssign)]
    assert names == ["dtype", "kernel_shape", "weight_spec", "scale_spec", "axis"], names


# ------------------------------------------------------- 3. the adoption
def test_process_weights_adopts_the_parked_scale_before_any_device_requant():
    body = _fn(FP8, "process_weights_after_loading", "Fp8OnlineLinearMethod")
    assert "adopt_host_quant_scale(layer.weight)" in body
    adopt_at = body.index("adopt_host_quant_scale(")
    requant_at = body.index("quantize_tensor(")
    assert adopt_at < requant_at, "the on-device requant must be the FALLBACK, after adoption"
    branch = body[adopt_at:requant_at]
    assert "layer.weight_scale = nnx.Param(w_s)" in branch
    assert "layer.weight = nnx.Param(w_q)" in branch and "w_q = layer.weight[...]" in branch, (
        "the adopted kernel must be re-wrapped from the SAME placed buffer (no copy)")
    assert "return True" in branch
    # the guards the device path had stay in front of both branches
    assert body.index("_is_loaded") < adopt_at and body.index("batch_features") < adopt_at


def test_engagement_marker_is_emitted_on_both_paths():
    src = FP8.read_text()
    assert "_ONLINE_DENSE_MARKER = (" in src
    assert '"VLLM_FP8_ONLINE_DENSE=1: serving dense on-the-fly fp8 "' in src, "the lane tooling greps this exact line"
    body = _fn(FP8, "process_weights_after_loading", "Fp8OnlineLinearMethod")
    assert body.count("logger.info_once(_ONLINE_DENSE_MARKER)") == 2, "both the host path and the fallback must announce"
    assert "quantized on the host before" in body, "the live check needs a line that says WHICH path ran"


# ------------------------------------------------------- the leaf stays a leaf
def test_the_leaf_imports_only_jax_and_the_quantization_leaf():
    tree = ast.parse(LEAF.read_text())
    mods = set()
    for n in tree.body:
        if isinstance(n, ast.Import):
            mods.update(a.name for a in n.names)
        elif isinstance(n, ast.ImportFrom):
            mods.add(n.module)
    allowed = {"dataclasses", "typing", "jax", "jax.sharding", "tpu_inference.layers.common.quantization"}
    assert mods <= allowed, f"the CPU gate cannot execute a leaf that imports {sorted(mods - allowed)}"


def test_the_leaf_computes_under_the_arrays_own_mesh():
    """Inside the loader the active mesh is the TPU one; any op on the host
    array under it raises 'Received incompatible devices' (measured, jax
    0.11.1). The leaf must nest the array's own mesh around its compute."""
    body = _fn(LEAF, "place_host_quantized")
    assert "with jax.set_mesh(_mesh_of(weight)):" in body
    inner = body[body.index("with jax.set_mesh("):body.index("param.set_value(")]
    assert "reshape(" in inner and "_quantize(" in inner, "reshape AND quantize must both be inside the host mesh context"
    calls = {n.func.id for n in ast.walk(ast.parse(LEAF.read_text()))
             if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)}
    assert "cpu_mesh_context" not in calls, "the leaf must not borrow layers.common.utils (envs) for this"
