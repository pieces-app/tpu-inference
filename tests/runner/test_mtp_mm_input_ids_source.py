# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Source-level pins for the mm + spec-decode input_ids fix.

These tests parse the runner sources with `ast` and need neither jax, vllm
nor a TPU, so they are provable on any CPU. They pin the shape of the fix
that keeps the raw token ids alive across the multimodal path (the mm path
used to rebind `input_ids` to None in `_execute_model` and kill the engine
at `assert input_ids is not None`, tpu_runner.py:2116 in the deployed wheel,
measured 2026-08-27). The behavioral counterparts, which drive the real
`TPUModelRunner` on a CPU mesh, live in `test_mtp_mm_input_ids.py`.
"""

import ast
import pathlib

import pytest

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]
_RUNNER_PATH = _REPO_ROOT / "tpu_inference" / "runner" / "tpu_runner.py"
_SPEC_MANAGER_PATH = (_REPO_ROOT / "tpu_inference" / "runner" /
                      "speculative_decoding_manager.py")

_GUARD_MESSAGE_FRAGMENT = "multimodal path must not null"


def _load(path: pathlib.Path) -> ast.Module:
    return ast.parse(path.read_text(), filename=str(path))


def _find_method(tree: ast.Module, class_name: str,
                 method_name: str) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            for item in node.body:
                if isinstance(item,
                              ast.FunctionDef) and item.name == method_name:
                    return item
    raise AssertionError(f"{class_name}.{method_name} not found")


def _is_self_attr_call(node: ast.AST, attr: str) -> bool:
    return (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
            and node.func.attr == attr
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "self")


def _bare_asserts_on_input_ids(fn: ast.FunctionDef):
    """`assert input_ids is not None` statements inside `fn`."""
    found = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.Assert):
            continue
        test = node.test
        if (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
                and test.left.id == "input_ids"
                and any(isinstance(op, ast.IsNot) for op in test.ops)):
            found.append(node)
    return found


def _runtime_error_guards_on_input_ids(fn: ast.FunctionDef):
    """`if input_ids is None: raise RuntimeError("...must not null...")`."""
    found = []
    for node in ast.walk(fn):
        if not isinstance(node, ast.If):
            continue
        test = node.test
        if not (isinstance(test, ast.Compare)
                and isinstance(test.left, ast.Name)
                and test.left.id == "input_ids"
                and any(isinstance(op, ast.Is) for op in test.ops)):
            continue
        for stmt in node.body:
            if (isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call)
                    and isinstance(stmt.exc.func, ast.Name)
                    and stmt.exc.func.id == "RuntimeError"):
                message = "".join(
                    ast.literal_eval(arg) for arg in stmt.exc.args
                    if isinstance(arg, ast.Constant))
                if _GUARD_MESSAGE_FRAGMENT in message:
                    found.append(node)
    return found


@pytest.fixture(scope="module")
def execute_model() -> ast.FunctionDef:
    return _find_method(_load(_RUNNER_PATH), "TPUModelRunner",
                        "_execute_model")


def test_get_input_ids_embeds_does_not_clobber_input_ids(execute_model):
    """The mm embed lookup must bind a separate name (`model_input_ids`) so
    the raw token ids stay live for the spec-decode machinery."""
    assigns = [
        node for node in ast.walk(execute_model)
        if isinstance(node, ast.Assign)
        and _is_self_attr_call(node.value, "_get_input_ids_embeds")
    ]
    assert len(assigns) == 1, "expected one _get_input_ids_embeds call"
    (target, ) = assigns[0].targets
    assert isinstance(target, ast.Tuple)
    names = [elt.id for elt in target.elts]
    assert names == ["model_input_ids", "inputs_embeds"], names


def test_model_forward_receives_model_input_ids(execute_model):
    """The forward pass keeps its contract: it sees the (possibly None)
    model-side ids plus the embeds, never the preserved raw ids."""
    calls = [
        node for node in ast.walk(execute_model)
        if _is_self_attr_call(node, "model_fn")
    ]
    assert len(calls) == 1, "expected one self.model_fn call"
    args = calls[0].args
    assert isinstance(args[2], ast.Name) and args[2].id == "model_input_ids"
    assert isinstance(args[4], ast.Name) and args[4].id == "inputs_embeds"


def test_execute_model_state_carries_raw_input_ids(execute_model):
    """ExecuteModelState.input_ids (consumed by _sample_from_logits and the
    drafters) must be the raw ids from _prepare_inputs."""
    calls = [
        node for node in ast.walk(execute_model)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == "ExecuteModelState"
    ]
    assert len(calls) == 1
    kw = {k.arg: k.value for k in calls[0].keywords}
    assert isinstance(kw["input_ids"], ast.Name)
    assert kw["input_ids"].id == "input_ids"


def test_sample_from_logits_guard_is_diagnosed_runtime_error():
    """The spec-decode scoring branch must fail with an explicit diagnosis
    (survives `python -O`), not a bare `assert input_ids is not None`."""
    fn = _find_method(_load(_RUNNER_PATH), "TPUModelRunner",
                      "_sample_from_logits")
    assert _bare_asserts_on_input_ids(fn) == []
    assert len(_runtime_error_guards_on_input_ids(fn)) == 1


def test_spec_manager_guard_is_diagnosed_runtime_error():
    """Twin guard on the drafter dispatch path
    (SpeculativeDecodingManager.propose_draft_token_ids)."""
    fn = _find_method(_load(_SPEC_MANAGER_PATH), "SpeculativeDecodingManager",
                      "propose_draft_token_ids")
    assert _bare_asserts_on_input_ids(fn) == []
    assert len(_runtime_error_guards_on_input_ids(fn)) == 1
