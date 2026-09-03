# Copyright 2026 Google LLC
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
"""PIECES_MM_DEBUG defaults to False and parses like every other env_bool.

``tests/test_envs.py`` imports ``tpu_inference.envs`` through the package,
whose ``__init__`` needs vLLM; the CPU gate has none, so this loads
``envs.py`` by path (it imports only the stdlib).
"""

import ast
import importlib.util
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[1]
ENVS = ROOT / "tpu_inference" / "envs.py"


def _envs():
    spec = importlib.util.spec_from_file_location("_pieces_envs", ENVS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_default_is_false(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("PIECES_MM_DEBUG", raising=False)
    envs = _envs()
    assert envs.environment_variables["PIECES_MM_DEBUG"]() is False
    assert envs.PIECES_MM_DEBUG is False  # module __getattr__ path
    assert "PIECES_MM_DEBUG" in envs.__dir__()


@pytest.mark.parametrize("value, want", [("1", True), ("true", True),
                                         ("True", True), ("0", False),
                                         ("false", False), ("", False)])
def test_parses_like_env_bool(monkeypatch: pytest.MonkeyPatch, value, want):
    monkeypatch.setenv("PIECES_MM_DEBUG", value)
    assert _envs().PIECES_MM_DEBUG is want


def test_invalid_value_is_refused(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("PIECES_MM_DEBUG", "yes")
    with pytest.raises(ValueError, match="PIECES_MM_DEBUG"):
        _envs().PIECES_MM_DEBUG


def test_type_checking_stub_declares_it_false():
    tree = ast.parse(ENVS.read_text())
    block = next(
        n for n in tree.body
        if isinstance(n, ast.If) and ast.unparse(n.test) == "TYPE_CHECKING")
    stubs = {
        n.target.id: ast.unparse(n.value)
        for n in block.body if isinstance(n, ast.AnnAssign)
    }
    assert stubs.get("PIECES_MM_DEBUG") == "False"
