"""Fp8TensorwiseMergedLinearMethod must store the fused kernel INTERLEAVED.

REVIEW-CONFIRMED (2026-09-01, 3/3). The apply path this class inherits
(common/quantization/fp8.py _apply_fused -> slice_sharded_tensor_for_concatenation)
de-interleaves the fused output as if the kernel were stored interleaved by TP
shard. The loader stored plain [gate | up]. Identity at n_shards=1 -- every
arm so far -- and silently permuted columns at TP>1 for compressed-tensors fp8
checkpoints. UnquantizedMergedLinearMethod already interleaves at load; this
makes the tensorwise path do the same, for the weight AND its 1-D per-channel
scale (same axis, same reorder).
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[4]
FP8 = ROOT / "tpu_inference" / "layers" / "jax" / "quantization" / "fp8.py"


def _cls_src(name):
    src = FP8.read_text(); tree = ast.parse(src)
    for c in ast.walk(tree):
        if isinstance(c, ast.ClassDef) and c.name == name:
            return ast.get_source_segment(src, c)
    pytest.fail(f"{name} not found")


def test_loader_reorders_before_assign():
    body = _cls_src("Fp8TensorwiseMergedLinearMethod")
    i = body.index("def _load_merged_shard"); j = body.index("def create_weights_jax")
    loader = body[i:j]
    assert "reorder_concatenated_tensor_for_sharding(" in loader, (
        "the merged kernel is stored plain [gate|up]; the apply side "
        "de-interleaves it -> wrong columns at TP>1")
    assert loader.index("reorder_concatenated_tensor_for_sharding(") < loader.index(
        "assign_and_shard_param("), "reorder must happen BEFORE the param is assigned"


def test_partials_thread_n_shards_and_output_sizes():
    body = _cls_src("Fp8TensorwiseMergedLinearMethod")
    i = body.index("def create_weights_jax"); cw = body[i:]
    n = cw.count("functools.partial(self._load_merged_shard")
    assert n >= 2, "weight AND weight_scale loaders expected"
    assert cw.count("n_shards=self.linear_config.n_shards") == n
    assert cw.count("output_sizes=self.linear_config.output_sizes") == n, (
        "every _load_merged_shard partial must carry n_shards + output_sizes, "
        "or the reorder silently runs as the identity")
