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
"""Seed rejection must surface as an actionable HTTP 400, never an empty 500.

The incident (2026-08-27, every TPU lane): ANY request carrying `seed` (text
or image) returned HTTP 500 with an EMPTY error body in <1s. Root cause chain
at our vLLM pin:

  1. `TpuPlatform.validate_request` raised a plain
     `ValueError("JAX does not support per-request seed.")`.
  2. vLLM's `AsyncLLM.generate()` re-raises only
     `vllm.exceptions.VLLMClientError` to the API layer; every other
     exception is wrapped as `raise EngineGenerateError() from e`
     (vllm/v1/engine/async_llm.py) -- an exception whose str() is "".
  3. The OpenAI frontend's `create_error_response`
     (vllm/entrypoints/serve/exception_handling/error_response.py) classifies
     `EngineGenerateError` into the terminal else-branch: InternalServerError,
     HTTP 500, message `str(exc)` == "".

The fix: raise `TpuRequestValidationError`, a subclass of BOTH
`vllm.exceptions.VLLMValidationError` (so vLLM maps it to a 400
BadRequestError carrying the message and `param="seed"`) and `ValueError`
(so callers written against the old contract keep working).

Layout follows tests/layers/vllm/test_fp8_nonserialized_guard.py: behavioral
tests need the fork's vllm+jax stack and skip cleanly elsewhere; the AST /
compile tests below them run on any CPython and also serve as the negative
control (`TPU_PLATFORM_SRC=<pre-fix file> pytest ...` must fail them).
"""

import ast
import os
import py_compile
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PLATFORM_PATH = (REPO_ROOT / "tpu_inference" / "platforms" /
                         "tpu_platform.py")

REQUIRED_MESSAGE_PHRASES = (
    "seed",
    "not supported",
    "temperature=0",
)

# ---------------------------------------------------------------------------
# Behavioral tests: real SamplingParams through the real platform hook.
# Require the fork's vllm+jax stack (skip cleanly where absent).
# ---------------------------------------------------------------------------


@pytest.fixture
def platform_mod():
    return pytest.importorskip(
        "tpu_inference.platforms.tpu_platform",
        reason="requires the fork's vllm+jax stack (dev/CI container)",
    )


@pytest.fixture
def seeded_params():
    sampling_params = pytest.importorskip("vllm.sampling_params")
    params = sampling_params.SamplingParams(seed=1234, temperature=0.7)
    assert params.sampling_type == sampling_params.SamplingType.RANDOM_SEED
    return params


def test_seeded_request_raises_client_error(platform_mod, seeded_params):
    """The rejection is a vLLM client error AND a ValueError, with a
    non-empty actionable message naming the offending parameter."""
    with pytest.raises(ValueError) as exc_info:
        platform_mod.TpuPlatform.validate_request(None, seeded_params)

    err = exc_info.value
    assert isinstance(err, platform_mod.TpuRequestValidationError)
    assert err.parameter == "seed"
    assert err.value == 1234

    message = str(err)
    assert message, "empty message == the original empty-500 fingerprint"
    for phrase in REQUIRED_MESSAGE_PHRASES:
        assert phrase in message, f"message must contain {phrase!r}"

    vllm_exceptions = pytest.importorskip("vllm.exceptions")
    assert isinstance(err, vllm_exceptions.VLLMClientError)
    assert isinstance(err, vllm_exceptions.VLLMValidationError)


def test_rejection_classifies_as_http_400(platform_mod, seeded_params):
    """End to end through vLLM's REAL error classifier: the raised exception
    must map to 400 BadRequestError with the message and param intact --
    the exact code path that rendered the empty 500 before the fix."""
    error_response_mod = pytest.importorskip(
        "vllm.entrypoints.serve.exception_handling.error_response",
        reason="needs vLLM entrypoints deps (fastapi) present",
    )

    with pytest.raises(ValueError) as exc_info:
        platform_mod.TpuPlatform.validate_request(None, seeded_params)

    response = error_response_mod.create_error_response(exc_info.value)
    assert response.error.code == 400
    assert response.error.type == "BadRequestError"
    assert response.error.param == "seed"
    assert "seed" in response.error.message
    assert response.error.message.strip(), "message must not be empty"


def test_unseeded_random_and_greedy_pass(platform_mod):
    sampling_params = pytest.importorskip("vllm.sampling_params")
    for params in (
            sampling_params.SamplingParams(temperature=0.7),  # RANDOM
            sampling_params.SamplingParams(temperature=0.0),  # GREEDY
    ):
        platform_mod.TpuPlatform.validate_request(None, params)  # no raise


# ---------------------------------------------------------------------------
# Dependency-free AST + compile tests. `TPU_PLATFORM_SRC` overrides the file
# under test so the suite doubles as its own negative control against the
# pre-fix source (where these must FAIL).
# ---------------------------------------------------------------------------


def _platform_path() -> Path:
    return Path(os.environ.get("TPU_PLATFORM_SRC", DEFAULT_PLATFORM_PATH))


def _platform_tree() -> ast.Module:
    return ast.parse(_platform_path().read_text())


def _find_validate_request(tree: ast.Module) -> ast.FunctionDef:
    for node in ast.walk(tree):
        if isinstance(
                node,
            (ast.FunctionDef,
             ast.AsyncFunctionDef)) and node.name == "validate_request":
            return node
    raise AssertionError("validate_request not found")


def test_validate_request_raises_the_client_error_type():
    """Every raise inside validate_request must construct
    TpuRequestValidationError -- never a bare ValueError (which vLLM's
    AsyncLLM boundary turns into an empty HTTP 500)."""
    fn = _find_validate_request(_platform_tree())
    raises = [node for node in ast.walk(fn) if isinstance(node, ast.Raise)]
    assert raises, "validate_request must reject seeded requests"
    for node in raises:
        call = node.exc
        assert isinstance(call, ast.Call), ast.dump(node)
        name = getattr(call.func, "id", getattr(call.func, "attr", None))
        assert name == "TpuRequestValidationError", (
            f"validate_request raises {name!r}; a non-client error surfaces "
            "as an empty HTTP 500 through AsyncLLM.generate()")
        keywords = {kw.arg for kw in call.keywords}
        assert "parameter" in keywords and "value" in keywords


def test_client_error_class_derives_from_vllm_validation_error():
    tree = _platform_tree()
    class_defs = [
        node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)
        and node.name == "TpuRequestValidationError"
    ]
    assert class_defs, "TpuRequestValidationError class missing"
    base_names = {
        getattr(base, "id", getattr(base, "attr", None))
        for node in class_defs
        for base in node.bases
    }
    assert "_VllmValidationError" in base_names, (
        "primary class must derive from vllm.exceptions.VLLMValidationError "
        "(aliased _VllmValidationError) so vLLM maps it to HTTP 400")
    assert "ValueError" in base_names, (
        "ValueError base keeps the old raises-ValueError contract")

    imports = [
        node for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
        and node.module == "vllm.exceptions"
    ]
    assert any(alias.name == "VLLMValidationError" for node in imports
               for alias in node.names)


def test_seed_message_is_actionable():
    source = _platform_path().read_text()
    assert "_SEED_UNSUPPORTED_MSG" in source
    for phrase in REQUIRED_MESSAGE_PHRASES:
        assert phrase in source, f"rejection message must contain {phrase!r}"


def test_platform_module_compiles(tmp_path):
    py_compile.compile(str(_platform_path()),
                       cfile=str(tmp_path / "tpu_platform.pyc"),
                       doraise=True)
