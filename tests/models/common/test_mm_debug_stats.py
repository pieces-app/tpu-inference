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
"""PIECES_MM_DEBUG: the shared one-line stats helper, and the native
call-site pattern it is used with.

The helper (``models/common/mm_debug_stats.py``) is loaded by path: it
imports only jax and numpy, so the CPU gate can run this without vLLM.
``gemma4_mm.py`` / ``gemma4_unified.py`` cannot be imported here (they pull
vLLM), so their call sites are covered two ways:

1. behaviourally, through a stand-in that reproduces the exact call-site
   pattern (``debug_sink = [] if flag else None`` -> tower -> projector ->
   ``if debug_sink is not None: emit``), asserting that with the flag OFF
   the jaxpr text and the outputs are byte-identical to the pre-change
   body, and that with the flag ON exactly one ``debug_callback`` is added
   and the outputs still do not change;
2. structurally, by AST over the real files: every ``emit_mm_debug_stats``
   call sits under ``if envs.PIECES_MM_DEBUG`` or ``if debug_sink is not
   None``, and the tower's ``debug_sink`` is optional with a ``None``
   default.
"""

import ast
import importlib.util
import pathlib
import re

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = pytest.importorskip("jax.numpy")

ROOT = pathlib.Path(__file__).resolve().parents[3]
STATS_SRC = ROOT / "tpu_inference" / "models" / "common" / "mm_debug_stats.py"
GEMMA4_MM = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_mm.py"
GEMMA4_UNIFIED = ROOT / "tpu_inference" / "models" / "jax" / "gemma4_unified.py"


def _stats_module():
    spec = importlib.util.spec_from_file_location("_mm_debug_stats", STATS_SRC)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MM = _stats_module()

B, NP, PP, H, D = 2, 8, 6, 5, 4  # images, patches, pixels/patch, hidden, lm
_RNG = np.random.default_rng(0)
W_ENC = jnp.asarray(_RNG.standard_normal((PP, H)) * 0.3, dtype=jnp.float32)
W_PROJ = jnp.asarray(_RNG.standard_normal((H, D)) * 0.3, dtype=jnp.float32)


def _inputs(nan_at=None, pad_rows_of_image=None):
    pv = _RNG.standard_normal((B, NP, PP)).astype(np.float32)
    pp = _RNG.integers(0, 4, (B, NP, 2)).astype(np.int32)
    if pad_rows_of_image is not None:
        image, rows = pad_rows_of_image
        pp[image, rows] = -1
        pv[image, rows] = 1e6  # padding rows carry garbage on purpose
    if nan_at is not None:
        pv[nan_at] = np.nan
    return jnp.asarray(pv), jnp.asarray(pp)


# ---------------------------------------------------------------- stand-ins
def _fake_tower(pv, input_mask, pixel_position_ids, debug_sink=None):
    """Mirrors Gemma4VisionModel.__call__: layers -> (optional sink) -> pool."""
    hidden = jnp.tanh(pv @ W_ENC) * input_mask[..., None].astype(pv.dtype)
    if debug_sink is not None:
        debug_sink.append(hidden)
    pooled = hidden.reshape(B, NP // 2, 2, H).mean(axis=2)
    pooler_mask = input_mask.reshape(B, NP // 2, 2).all(axis=-1)
    return ((pooled, pooler_mask), )


def _fake_proj(x):
    return x @ W_PROJ


def _baseline(pv, pp):
    """The pre-change body: no sink, no callback."""
    input_mask = pp[..., 0] != -1
    vision_outputs = _fake_tower(pv, input_mask, pp)
    tower = vision_outputs[0][0]
    return _fake_proj(tower)


def _call_site(flag, log):
    """The exact shape of gemma4_mm.get_single_image_embedding after the
    change, with the flag as a Python constant (envs.PIECES_MM_DEBUG is one
    at trace time)."""

    def f(pv, pp):
        input_mask = pp[..., 0] != -1
        debug_sink = [] if flag else None
        vision_outputs = _fake_tower(pv, input_mask, pp, debug_sink=debug_sink)
        tower = vision_outputs[0][0]
        pooler_mask = vision_outputs[0][1]
        proj = _fake_proj(tower)
        if debug_sink is not None:
            MM.emit_mm_debug_stats(
                log,
                "native",
                tensors={
                    "pv": pv,
                    "enc": debug_sink[0] if debug_sink else None,
                    "tower": tower,
                    "proj": proj,
                },
                masks={
                    "pv": input_mask,
                    "enc": input_mask,
                    "tower": pooler_mask,
                    "proj": pooler_mask,
                },
                counts={"soft_tokens": pooler_mask},
                extra={
                    "site": "stand_in",
                    "n_images": int(pv.shape[0])
                },
            )
        return proj

    return f


def _fields(line):
    assert line.startswith(MM.LINE_PREFIX + " "), line
    out = {}
    for token in line[len(MM.LINE_PREFIX) + 1:].split(" "):
        key, value = token.split("=", 1)
        out[key] = value
    return out


# ------------------------------------------------------------ flag off
def test_flag_off_jaxpr_and_outputs_are_byte_identical_to_the_old_body():
    log = []
    pv, pp = _inputs()
    off = _call_site(False, log.append)
    assert jax.make_jaxpr(off)(
        pv, pp).pretty_print() == jax.make_jaxpr(_baseline)(pv,
                                                            pp).pretty_print()
    got = np.asarray(jax.jit(off)(pv, pp))
    want = np.asarray(jax.jit(_baseline)(pv, pp))
    assert np.array_equal(got, want)
    assert log == []


# ------------------------------------------------------------- flag on
def test_flag_on_adds_one_callback_and_leaves_the_outputs_unchanged():
    log = []
    pv, pp = _inputs()
    on = _call_site(True, log.append)
    text = jax.make_jaxpr(on)(pv, pp).pretty_print()
    # One debug_callback PRIMITIVE (its repr also names the function).
    assert len(re.findall(r"^\s*debug_callback\[", text, re.M)) == 1, text
    assert "debug_callback" not in jax.make_jaxpr(_baseline)(
        pv, pp).pretty_print()

    log.clear()
    got = jax.jit(on)(pv, pp)
    jax.block_until_ready(got)
    assert np.array_equal(np.asarray(got),
                          np.asarray(jax.jit(_baseline)(pv, pp)))
    assert len(log) == 1, log
    f = _fields(log[0])
    assert f["path"] == "native"
    assert f["site"] == "stand_in"
    assert f["n_images"] == str(B)
    for tensor in ("pv", "enc", "tower", "proj"):
        for key in MM.STAT_KEYS:
            assert f"{tensor}.{key}" in f, (tensor, key, log[0])
        assert f[f"{tensor}.nan"] == "0"
        assert f[f"{tensor}.inf"] == "0"
    assert f["pv.shape"] == f"({B},{NP},{PP})"
    assert f["enc.shape"] == f"({B},{NP},{H})"
    assert f["tower.shape"] == f"({B},{NP // 2},{H})"
    assert f["proj.shape"] == f"({B},{NP // 2},{D})"
    assert f["pv.dtype"] == "float32"
    # No padding in this input: every pooled row is a soft token.
    assert f["soft_tokens"] == str(B * (NP // 2))


def test_nan_in_the_input_is_counted_on_every_downstream_tensor():
    log = []
    pv, pp = _inputs(nan_at=(0, 3, 1))
    got = jax.jit(_call_site(True, log.append))(pv, pp)
    jax.block_until_ready(got)
    f = _fields(log[0])
    assert f["pv.nan"] == "1"
    assert int(f["enc.nan"]) > 0
    assert int(f["tower.nan"]) > 0
    assert int(f["proj.nan"]) > 0
    # The finite part is still measured: mean/max are numbers, not nan.
    assert f["pv.mean"] != "nan" and f["proj.maxabs"] != "nan"


def test_masks_drop_padding_rows_from_stats_and_from_soft_tokens():
    log = []
    pv, pp = _inputs(pad_rows_of_image=(1, slice(4, 8)))
    got = jax.jit(_call_site(True, log.append))(pv, pp)
    jax.block_until_ready(got)
    f = _fields(log[0])
    # Padding rows hold 1e6; with the row mask they must not reach pv.max.
    assert float(f["pv.max"]) < 100.0, f["pv.max"]
    assert f[
        "pv.shape"] == f"({B},{NP},{PP})", "shape reports the unmasked array"
    # Image 1 lost 4 of 8 patches = 2 of 4 pooled rows.
    assert f["soft_tokens"] == str(B * (NP // 2) - 2)


def test_masks_negative_control_without_a_mask_the_padding_leaks():
    pv, pp = _inputs(pad_rows_of_image=(1, slice(4, 8)))
    unmasked = MM.host_tensor_stats([np.asarray(pv)])
    masked = MM.host_tensor_stats([np.asarray(pv)],
                                  [np.asarray(pp[..., 0] != -1)])
    assert unmasked["max"] == pytest.approx(1e6)
    assert masked["max"] < 100.0
    assert masked["count"] == unmasked["count"] - 4 * PP


# ------------------------------------------------------------- eager
def test_eager_call_logs_immediately_with_the_same_keys():
    log = []
    pv, pp = _inputs()
    _call_site(True, log.append)(pv, pp)  # no jit: concrete arrays
    assert len(log) == 1
    f = _fields(log[0])
    assert f["soft_tokens"] == str(B * (NP // 2))
    assert f["tower.dtype"] == "float32"


def test_chunks_merge_into_one_entry_and_shapes_say_so():
    log = []
    a = jnp.asarray(_RNG.standard_normal((1, 3, 2)), dtype=jnp.float32)
    b = jnp.asarray(_RNG.standard_normal((1, 3, 2)), dtype=jnp.float32)
    c = jnp.asarray(_RNG.standard_normal((1, 5, 2)), dtype=jnp.float32)
    MM.emit_mm_debug_stats(log.append, "torchax", tensors={"pv": [a, b]})
    MM.emit_mm_debug_stats(log.append, "torchax", tensors={"pv": [a, c]})
    uniform, ragged = (_fields(line) for line in log)
    assert uniform["pv.shape"] == "2x(1,3,2)"
    assert ragged["pv.shape"] == "[(1,3,2),(1,5,2)]"
    want = MM.host_tensor_stats([
        np.concatenate([np.asarray(a).reshape(-1),
                        np.asarray(b).reshape(-1)])
    ])
    assert float(uniform["pv.mean"]) == pytest.approx(want["mean"], rel=1e-5)
    assert float(uniform["pv.std"]) == pytest.approx(want["std"], rel=1e-5)
    assert float(uniform["pv.maxabs"]) == pytest.approx(want["maxabs"],
                                                        rel=1e-5)


def test_none_and_empty_tensors_are_skipped_not_crashed():
    log = []
    x = jnp.ones((2, 2), dtype=jnp.float32)
    MM.emit_mm_debug_stats(log.append,
                           "torchax",
                           tensors={
                               "pv": x,
                               "enc": None,
                               "tower": [],
                               "proj": x
                           },
                           counts={"soft_tokens": 7})
    f = _fields(log[0])
    assert "pv.mean" in f and "proj.mean" in f
    assert not any(k.startswith("enc.") or k.startswith("tower.") for k in f)
    assert f["soft_tokens"] == "7"


def test_bfloat16_is_named_and_measured_in_float32():
    log = []
    x = jnp.asarray(_RNG.standard_normal((4, 8)), dtype=jnp.bfloat16)
    MM.emit_mm_debug_stats(log.append, "native", tensors={"pv": x})
    f = _fields(log[0])
    assert f["pv.dtype"] == "bfloat16"
    want = float(np.asarray(x).astype(np.float32).mean())
    assert float(f["pv.mean"]) == pytest.approx(want, rel=1e-4, abs=1e-6)


def test_call_counter_advances_per_line():
    log = []
    x = jnp.ones((1, 1), dtype=jnp.float32)
    MM.emit_mm_debug_stats(log.append, "native", tensors={"pv": x})
    MM.emit_mm_debug_stats(log.append, "native", tensors={"pv": x})
    first, second = (int(_fields(line)["call"]) for line in log)
    assert second == first + 1


# ------------------------------------------------------- real call sites
def _parents(tree):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    return parents


def _enclosing_if_tests(node, parents):
    tests = []
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If):
            tests.append(ast.unparse(node.test))
    return tests


_GUARDS = {"envs.PIECES_MM_DEBUG", "debug_sink is not None"}


@pytest.mark.parametrize("path, expected_sites", [
    (GEMMA4_MM, {"get_single_image_embedding", "encoder_cudagraph_forward"}),
    (GEMMA4_UNIFIED, {"get_single_image_embedding"}),
])
def test_every_native_emit_is_guarded_by_the_flag(path, expected_sites):
    tree = ast.parse(path.read_text())
    parents = _parents(tree)
    sites = set()
    for node in ast.walk(tree):
        if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "emit_mm_debug_stats"):
            continue
        guards = _enclosing_if_tests(node, parents)
        assert _GUARDS & set(guards), (
            f"{path.name}:{node.lineno} emit_mm_debug_stats is not under "
            f"`if envs.PIECES_MM_DEBUG` / `if debug_sink is not None`: {guards}"
        )
        site = next(kw for kw in node.keywords if kw.arg == "extra")
        sites.add(
            ast.literal_eval(
                next(v for k, v in zip(site.value.keys, site.value.values)
                     if ast.literal_eval(k) == "site")))
    assert sites == expected_sites


def test_gemma4_mm_sink_is_created_only_under_the_flag():
    src = GEMMA4_MM.read_text()
    assert src.count("debug_sink = [] if envs.PIECES_MM_DEBUG else None") == 2
    tree = ast.parse(src)
    tower = next(
        n for n in ast.walk(tree)
        if isinstance(n, ast.ClassDef) and n.name == "Gemma4VisionModel")
    call = next(n for n in tower.body
                if isinstance(n, ast.FunctionDef) and n.name == "__call__")
    args = call.args
    names = [a.arg for a in args.args]
    assert "debug_sink" in names
    default = args.defaults[names.index("debug_sink") -
                            (len(names) - len(args.defaults))]
    assert ast.unparse(default) == "None"
    parents = _parents(tree)
    appends = [
        n for n in ast.walk(call) if isinstance(n, ast.Call)
        and ast.unparse(n.func) == "debug_sink.append"
    ]
    assert len(appends) == 1
    assert "debug_sink is not None" in _enclosing_if_tests(appends[0], parents)


def test_the_helper_imports_only_jax_and_numpy():
    """The gate has no vLLM; a tpu_inference.logger import would sink it."""
    tree = ast.parse(STATS_SRC.read_text())
    roots = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots |= {a.name.split(".")[0] for a in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module:
            roots.add(node.module.split(".")[0])
    assert roots <= {
        "functools", "itertools", "math", "typing", "jax", "numpy"
    }, roots
    assert not re.search(r"^\s*from tpu_inference", STATS_SRC.read_text(),
                         re.M)
