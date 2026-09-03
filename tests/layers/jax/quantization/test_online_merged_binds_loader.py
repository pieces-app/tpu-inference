"""Fp8OnlineLinearMethod must bind _load_merged_tensor before delegating.

MEASURED on v6e 2026-09-01 23:07Z: all four flax dense-quant arms of the
26B-A4B (fp8, e4m3b11fnuz, int8, allint8) died identically ~70s into boot at
MODEL CONSTRUCTION:

    AttributeError: 'Fp8OnlineLinearMethod' object has no attribute
                    '_load_merged_tensor'

create_weights_jax delegates merged layers (gate_up_proj) to
UnquantizedMergedLinearMethod.create_weights_jax(self, ...), which builds
functools.partial(self._load_merged_tensor, ...). That name is a staticmethod
of the Merged class, not of the online one. Never reachable on the 12B, whose
torchax path has its own merged loader -- which is why #24's gate missed it.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
FP8 = ROOT / "tpu_inference" / "layers" / "jax" / "quantization" / "fp8.py"


def _create_weights_src():
    src = FP8.read_text()
    tree = ast.parse(src)
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef) and c.name == "Fp8OnlineLinearMethod":
            for f in c.body:
                if isinstance(
                        f, ast.FunctionDef) and f.name == "create_weights_jax":
                    return ast.get_source_segment(src, f)
    pytest.fail("Fp8OnlineLinearMethod.create_weights_jax not found")


def test_merged_branch_binds_the_loader_it_delegates_to():
    body = _create_weights_src()
    assert "UnquantizedMergedLinearMethod.create_weights_jax(" in body
    assert "self._load_merged_tensor = " in body, (
        "the merged branch delegates to a function that reads "
        "self._load_merged_tensor, but never binds it -- every gate_up_proj "
        "raises AttributeError at model construction")
    # and the bind must come BEFORE the delegation
    assert body.index("self._load_merged_tensor = ") < body.index(
        "UnquantizedMergedLinearMethod.create_weights_jax(")


def test_the_bound_loader_is_the_interleaving_one():
    """Binding the TENSORWISE merged loader would be wrong: it concatenates
    without reorder_concatenated_tensor_for_sharding, so at TP>1 the apply
    side de-interleaves columns that were never interleaved."""
    body = _create_weights_src()
    assert "UnquantizedMergedLinearMethod._load_merged_tensor" in body
    assert "Fp8TensorwiseMergedLinearMethod._load_merged" not in body
