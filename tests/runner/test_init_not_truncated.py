"""No method body may be orphaned behind a `return`, and __init__ must survive.

MEASURED 2026-09-01: a helper method was inserted INTO THE MIDDLE of
`CompilationManager.__init__`. Python accepted it -- the file parsed, every
test stayed green, and the image built -- but `__init__` then ended three
statements in, and the JAX PERSISTENT COMPILE CACHE setup that followed became
unreachable code sitting after a `return` in the new method.

Consequence: no compile cache. Every boot pays a full cold compile, and the
bank-delta engagement proof the isolation harness depends on would read zero.
Three independent reviewers found it; nothing in the test suite did, because
"does this file parse" and "does this helper exist" were both still true.

Two guards, because either alone is escapable:
  * __init__ must still contain the thing that was orphaned, and
  * NO function in the file may have statements after a return at the same
    level -- the general shape of the mistake.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
CM = ROOT / "tpu_inference" / "runner" / "compilation_manager.py"


def _tree():
    return ast.parse(CM.read_text())


def test_compilation_manager_init_still_enables_the_compile_cache():
    cls = next((n for n in ast.walk(_tree())
                if isinstance(n, ast.ClassDef) and n.name == "CompilationManager"),
               None)
    assert cls is not None
    init = next((f for f in cls.body
                 if isinstance(f, ast.FunctionDef) and f.name == "__init__"), None)
    assert init is not None, "CompilationManager.__init__ not found"
    src = ast.get_source_segment(CM.read_text(), init) or ""
    assert "VLLM_DISABLE_COMPILE_CACHE" in src, (
        "__init__ no longer contains the compile-cache enablement. If a method "
        "was inserted mid-__init__, the rest became another method's dead code "
        "and the persistent XLA cache is never configured.")
    assert "jax_compilation_cache_dir" in src, (
        "__init__ must still set jax_compilation_cache_dir")


def test_no_function_has_statements_after_a_return():
    """The general form of the bug: unreachable trailing statements."""
    offenders = []
    for fn in [n for n in ast.walk(_tree())
               if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]:
        for i, stmt in enumerate(fn.body[:-1]):
            if isinstance(stmt, ast.Return):
                nxt = fn.body[i + 1]
                offenders.append(f"{fn.name}: statement at line {nxt.lineno} "
                                 f"is unreachable (return at {stmt.lineno})")
    assert not offenders, (
        "unreachable code after a return -- this is exactly how "
        "CompilationManager.__init__ was truncated:\n  " +
        "\n  ".join(offenders))
