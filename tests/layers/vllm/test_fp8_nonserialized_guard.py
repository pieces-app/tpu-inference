"""Fail-closed guard for `--quantization fp8` on non-fp8-serialized checkpoints.

The trap (2026-08-27 scoping, gemma-4-12b-it bf16 on v6e-1): "fp8" is in
TpuPlatform supported_quantization, so a bf16 checkpoint plus
`--quantization fp8` starts up cleanly. Upstream vLLM at our pin routes
non-serialized checkpoints to Fp8PerTensorOnlineLinearMethod (real online
quant), but the fork's `VllmFp8Config.get_quant_method` overrides dispatch
and unconditionally returned the OFFLINE `VllmFp8LinearMethod`: its
`create_weights` registers `weight_scale = torch.empty(...)`, a bf16
checkpoint has no scale tensors to fill it, the loader casts bf16 -> e4m3
UNSCALED, and `process_weights_after_loading` ships the uninitialized scale
memory to the TPU. Deterministic garbage behind a healthy /health — the
int8-MoE-requant fleet failure mode re-armed on the dense path.

The fix is a load-time NotImplementedError in the LinearBase dispatch arm
(after the ignored-layers skip, before the offline method), naming the trap
and the supported alternatives. These tests pin:

  * non-serialized fp8 config -> raises at dispatch time with the expected
    actionable message;
  * fp8-serialized config -> still dispatches VllmFp8LinearMethod (zero
    behavior change for legit users);
  * ignored layers -> still fall through to the unquantized method before
    the guard, serialized or not.

The behavioral tests need the fork's vllm+jax+torchax stack (they skip
elsewhere); the AST/compile tests below them run on any CPython >= 3.10 and
gate the guard's presence, placement, and message with no dependencies.
"""

import ast
import py_compile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
FP8_PATH = REPO_ROOT / "tpu_inference" / "layers" / "vllm" / "quantization" / "fp8.py"

GUARD_PHRASES = (
    "not implemented on the TPU torchax path",
    "uninitialized",
    "fp8-serialized",
    "compressed-tensors",
)

# ---------------------------------------------------------------------------
# Behavioral tests: exercise the real dispatch. Require the fork's full
# stack (vllm, jax, torchax), as tests/layers/vllm/test_fp8.py does; skip
# cleanly where it is absent.
# ---------------------------------------------------------------------------

PREFIX = "model.layers.0.mlp.down_proj"


@pytest.fixture
def stack():
    """(fp8 module, vllm linear module); skips without the full stack so
    the dependency-free AST tests below still run."""
    fp8_mod = pytest.importorskip(
        "tpu_inference.layers.vllm.quantization.fp8",
        reason="requires the fork's vllm+jax+torchax stack (dev/CI container)",
    )
    vllm_linear = pytest.importorskip("vllm.model_executor.layers.linear")
    return fp8_mod, vllm_linear


def test_nonserialized_fp8_linear_raises_at_load_time(stack):
    """bf16 checkpoint + --quantization fp8 must die at dispatch, loudly."""
    fp8_mod, vllm_linear = stack
    config = fp8_mod.VllmFp8Config(is_checkpoint_fp8_serialized=False)
    layer = MagicMock(spec=vllm_linear.RowParallelLinear)

    with pytest.raises(NotImplementedError) as exc_info:
        config.get_quant_method(layer, prefix=PREFIX)

    message = str(exc_info.value)
    for phrase in GUARD_PHRASES:
        assert phrase in message, f"guard message must mention {phrase!r}"


def test_serialized_fp8_linear_still_dispatches_offline_method(stack):
    """fp8-serialized checkpoints keep the exact pre-guard dispatch."""
    fp8_mod, vllm_linear = stack
    config = fp8_mod.VllmFp8Config(is_checkpoint_fp8_serialized=True)
    layer = MagicMock(spec=vllm_linear.ColumnParallelLinear)

    # Constructor internals (linear config sharding, upstream Fp8LinearMethod
    # init) are covered by tests/layers/vllm/test_fp8.py; here we pin only
    # that dispatch still selects the offline method.
    with patch.object(fp8_mod.VllmFp8Config, "get_linear_config",
                      return_value=MagicMock()), \
         patch.object(fp8_mod.VllmFp8LinearMethod, "__init__",
                      return_value=None) as method_init:
        method = config.get_quant_method(layer, prefix=PREFIX)

    assert isinstance(method, fp8_mod.VllmFp8LinearMethod)
    method_init.assert_called_once()


@pytest.mark.parametrize("serialized", [False, True])
def test_ignored_layers_still_skip_before_the_guard(stack, serialized):
    """The ignored-layers escape hatch is unchanged, serialized or not."""
    fp8_mod, vllm_linear = stack
    config = fp8_mod.VllmFp8Config(
        is_checkpoint_fp8_serialized=serialized,
        ignored_layers=[PREFIX],
    )
    layer = MagicMock(spec=vllm_linear.RowParallelLinear)

    with patch.object(fp8_mod.VllmFp8Config, "get_linear_config",
                      return_value=MagicMock()), \
         patch.object(fp8_mod.VllmUnquantizedLinearMethod, "__init__",
                      return_value=None):
        method = config.get_quant_method(layer, prefix=PREFIX)

    assert isinstance(method, fp8_mod.VllmUnquantizedLinearMethod)


# ---------------------------------------------------------------------------
# Dependency-free AST + compile sanity: runs anywhere, keeps the guard from
# silently drifting or being reordered behind the dispatch it must precede.
# ---------------------------------------------------------------------------


def _linear_case_body() -> list[ast.stmt]:
    tree = ast.parse(FP8_PATH.read_text())
    cls = next(
        node for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "VllmFp8Config")
    fn = next(node for node in cls.body if isinstance(node, ast.FunctionDef)
              and node.name == "get_quant_method")
    match_stmt = next(node for node in ast.walk(fn)
                      if isinstance(node, ast.Match))
    for case in match_stmt.cases:
        pattern = case.pattern
        if not isinstance(pattern, ast.MatchClass):
            continue
        cls_node = pattern.cls
        name = (cls_node.attr if isinstance(cls_node, ast.Attribute) else
                getattr(cls_node, "id", None))
        if name == "LinearBase":
            return case.body
    raise AssertionError(
        "LinearBase dispatch arm not found in get_quant_method")


def _guard_raise(body: list[ast.stmt]) -> ast.Raise:
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if (isinstance(test, ast.UnaryOp) and isinstance(test.op, ast.Not)
                and isinstance(test.operand, ast.Attribute)
                and test.operand.attr == "is_checkpoint_fp8_serialized"):
            raises = [n for n in node.body if isinstance(n, ast.Raise)]
            assert raises, "the non-serialized branch must raise"
            return raises[0]
    raise AssertionError(
        "no `if not self.is_checkpoint_fp8_serialized: raise` guard found")


def test_guard_exists_and_precedes_the_offline_dispatch():
    body = _linear_case_body()
    guard = _guard_raise(body)

    dispatch_lines = [
        node.lineno
        for node in ast.walk(ast.Module(body=body, type_ignores=[]))
        if isinstance(node, ast.Call)
        and getattr(node.func, "id", None) == "VllmFp8LinearMethod"
    ]
    assert dispatch_lines, "offline dispatch must still exist for legit users"
    assert guard.lineno < min(dispatch_lines), (
        "the fail-closed raise must come before VllmFp8LinearMethod dispatch")


def test_guard_raises_notimplementederror_with_actionable_message():
    guard = _guard_raise(_linear_case_body())
    exc = guard.exc
    assert isinstance(exc, ast.Call)
    assert getattr(exc.func, "id", None) == "NotImplementedError"

    literals = " ".join(
        node.value for node in ast.walk(exc)
        if isinstance(node, ast.Constant) and isinstance(node.value, str))
    for phrase in GUARD_PHRASES:
        assert phrase in literals, f"guard message must mention {phrase!r}"


def test_fp8_module_byte_compiles():
    py_compile.compile(str(FP8_PATH), doraise=True)
