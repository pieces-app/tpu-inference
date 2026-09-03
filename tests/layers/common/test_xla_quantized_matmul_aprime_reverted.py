"""Structural guard that the W4A16 stopgap a' stays reverted (issue #157).

a' (`bf62f6fce5a8`, "xla_quantized_matmul: dequantize in the activation
dtype") added a fast leg to the 2D-scale branch of
`xla_quantized_matmul`: when `w_scale.dtype == x.dtype` it dequantized the
int4 codes directly in bf16 instead of through the historical float32
expression, avoiding a 450 MiB f32 broadcast per forward and buying a
measured 2.06x decode speedup (26.59 vs 12.90 tok/s on gemma-4-12B W4A16).

a' shipped with four CPU tests asserting bit-exactness against the f32 leg,
and it was bit-identical BY CONSTRUCTION. On real v6e hardware it was NOT:
the `eval-12b-w4a16-refmatch` isolation arm (image the only variable)
measured 0/6 greedy outputs exact, diverging at chars 27-358
(BENCH_RESULTS.md "The a' verdict"). Per the pre-registered gate, no
W4A16-class route promotes candidate -> serving until that is resolved, so
a' is reverted to the unconditional f32 leg here.

WHAT THIS TEST IS AND IS NOT. It is a structural guard, dependency-free,
that a' does not silently return: it asserts the a'-only fast-leg condition
is absent and the unconditional f32 expression is present. It is
deliberately NOT the instrument of record for the a' *decision* -- a'
already passed four CPU bit-exactness tests while the hardware diverged, so
a CPU test cannot render that verdict. The verdict is the on-demand-TPU
cold greedy canary (harness fix `b-aprime-w4a16`); this file only keeps the
revert from regressing unnoticed.

Negative control: `git revert` of this revert (i.e. re-applying a')
reintroduces `w_scale.dtype == x.dtype`, and `test_aprime_fast_leg_absent`
goes red. Run: `pytest tests/layers/common/test_xla_quantized_matmul_aprime_reverted.py`
(CPython only; no jax/torchax needed).
"""

import py_compile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
LINEAR_PATH = REPO_ROOT / "tpu_inference" / "layers" / "common" / "linear.py"

# The a'-ONLY signature. NOT "Skipping activation quantization" -- that
# info_once predates a' and is present in both the a' and reverted forms,
# so it would be a false marker (verified 2026-08-31).
APRIME_SIGNATURES = (
    "w_scale.dtype == x.dtype",
    "w4tax",
    "dequantize in the activation dtype",
)

# The historical f32 leg the revert restores as the UNCONDITIONAL path.
F32_LEG = "astype(jnp.float32) *"


def _linear_src() -> str:
    assert LINEAR_PATH.is_file(), f"missing {LINEAR_PATH}"
    return LINEAR_PATH.read_text()


def test_linear_compiles():
    py_compile.compile(str(LINEAR_PATH), doraise=True)


def test_aprime_fast_leg_absent():
    src = _linear_src()
    hits = [s for s in APRIME_SIGNATURES if s in src]
    assert not hits, (
        f"a' fast-leg signature(s) present in linear.py: {hits}. a' is "
        f"supposed to be reverted (issue #157); it diverges on TPU. If this "
        f"is a deliberate re-land, the OD-TPU refmatch arm must be re-run "
        f"and BENCH_RESULTS updated before it lands.")


def test_f32_leg_restored_and_unconditional():
    src = _linear_src()
    assert F32_LEG in src, "the historical float32 dequant leg is missing"
    # The 2D-scale branch must not fork on the activation dtype any more:
    # exactly the `if w_scale.dtype == x.dtype:` split a' introduced.
    assert "if w_scale.dtype == x.dtype:" not in src, (
        "the a' activation-dtype fast-leg branch is back")
