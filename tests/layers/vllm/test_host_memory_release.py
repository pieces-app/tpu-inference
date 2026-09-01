"""Every `process_weights_after_loading` must release its host memory.

`delattr(layer, "weight")` frees NOTHING during an incremental load. These
methods inherit `maybe_process_weights`, so under VLLM_INCREMENTAL_FP8_LOADING
they run mid-load while the model's own
`params_dict = dict(self.named_parameters())` still holds a strong reference to
every original Parameter for the whole loop. That is why the offline and MoE
siblings resize the storage (`_free_torch_storage`) instead of relying on the
attribute going away, and why they end with `_release_host_memory()`
(gc.collect + jax.effects_barrier + malloc_trim) -- without the trim, glibc
keeps the arena and the pod's RSS never drops.

`VllmFp8OnlineLinearMethod` did neither. A 26B/31B bf16 checkpoint would
accumulate in host RAM for the entire load and the pod would be OOM-killed
mid-load or during first compile -- reading as "fp8 needs more host RAM"
rather than as a missing free.

Written as an invariant over ALL such methods rather than a check on the one
that was broken: the next sibling added should not be able to skip this
silently, which is precisely how this one did.
"""
import ast
import pathlib

import pytest

ROOT = pathlib.Path(__file__).resolve().parents[3]
FP8 = ROOT / "tpu_inference" / "layers" / "vllm" / "quantization" / "fp8.py"


def _methods():
    src = FP8.read_text()
    tree = ast.parse(src)
    out = []
    for cls in [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]:
        for f in cls.body:
            if (isinstance(f, ast.FunctionDef)
                    and f.name == "process_weights_after_loading"):
                out.append((cls.name, ast.get_source_segment(src, f) or ""))
    return out


def test_there_are_methods_to_check():
    """A collector that silently finds nothing would make every test below
    vacuously green -- the same failure mode as a skip."""
    got = _methods()
    assert len(got) >= 3, (
        f"expected at least 3 process_weights_after_loading methods "
        f"(offline linear, online linear, MoE); found {[n for n, _ in got]}")


@pytest.mark.parametrize("name", [n for n, _ in _methods()])
def test_every_method_releases_host_memory(name):
    body = dict(_methods())[name]
    assert "_release_host_memory()" in body, (
        f"{name}.process_weights_after_loading never calls "
        f"_release_host_memory(). Freeing tensor storage is not enough -- "
        f"without gc.collect + malloc_trim the pod's RSS does not drop, and a "
        f"26B/31B load under the incremental loader is OOM-killed.")


@pytest.mark.parametrize("name", [n for n, _ in _methods()])
def test_every_method_frees_the_param_it_replaces(name):
    body = dict(_methods())[name]
    if "delattr" not in body:
        pytest.skip(f"{name} does not delattr a loaded param")
    assert "_free_torch_storage" in body, (
        f"{name} delattrs a param without _free_torch_storage. During an "
        f"incremental load the model's params_dict still holds a strong "
        f"reference, so delattr frees nothing and the bf16 checkpoint "
        f"accumulates in host RAM for the whole load.")
