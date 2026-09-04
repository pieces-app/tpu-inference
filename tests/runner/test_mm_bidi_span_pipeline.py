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
"""The image spans the attention kernel receives, one and two images.

WHAT THIS PINS
--------------
`mm_bidi_ranges` is the only thing that makes Gemma-4's image soft tokens
attend BIDIRECTIONALLY instead of causally.  It is a `(max_num_seqs, 2)`
int32 operand, `[start, end)` per request, built by the runner and read by
the RPA v3 kernel.  Until now every property of it -- the boundaries, the
DP row layout, what happens with two images, what happens when the feature
is off -- could only be checked by booting a chip.

`tpu_inference/runner/mm_bidi_ranges.py` holds that arithmetic as a pure
function so it can be checked here.  The tests below are the CPU half of
the "does the torchax path get a correct bidirectional mask" question:

  * ONE image  -> exactly `[first, last+1)`, the placeholder run and
    nothing else (a fencepost here reads 28 soft tokens as text, or lets a
    text token attend forward).
  * TWO images -> NO range, plus a per-request warning.  The operand holds
    one range, so a two-image prompt is causal-only; the failure is a
    quality loss with no error, so it must be loud.
  * `is_embed` holes -> one span per contiguous embedded stretch
    (`extract_embeds_range`), matching vLLM's GPU runner.  A prompt whose
    placeholder run is broken by BOI/EOI tokens is therefore multi-block,
    i.e. it falls back to causal -- stated here so it is a decision, not a
    surprise.
  * AUDIO features are skipped (audio is not blockwise-bidirectional).
  * The row index is the DP-offset persistent-batch row, the SAME row
    `seq_lens` uses; a row-layout slip silently applies one request's image
    range to another request.

BOTH PATHS READ THE SAME OPERAND.  The flax models take it at
`models/jax/gemma4.py` (`Gemma4Attention.__call__`, sliding layers only)
and the torchax models at `layers/vllm/backends/flash_attn.py`
(`PallasAttentionBackendImpl.forward`, same gate).  There is no separate
torchax construction to get wrong -- which is why the E4B image defect is
NOT a bidi defect: see `test_the_feature_can_be_globally_inert` and the
`_init_mm_bidi` config gate, which returns before building anything for a
text config that does not declare `use_bidirectional_attention="vision"`
(gemma-4 E4B/E2B).

NEGATIVE CONTROL: reverting `build_mm_bidi_ranges` to `end = last` (instead
of `last + 1`) turns the one-image test red; dropping the `len(spans) > 1`
branch so both images produce the first range turns the two-image tests
red; using `i` instead of `req_offset + i` turns the DP test red.
"""
import ast
import pathlib
import types

import numpy as np
import pytest

ROOT = pathlib.Path(__file__).resolve().parents[2]
SRC = ROOT / "tpu_inference" / "runner" / "mm_bidi_ranges.py"
RUNNER = ROOT / "tpu_inference" / "runner" / "tpu_runner.py"
FLAX_ATTN = ROOT / "tpu_inference" / "models" / "jax" / "gemma4.py"
TORCHAX_ATTN = (ROOT / "tpu_inference" / "layers" / "vllm" / "backends" /
                "flash_attn.py")


def _load():
    """Compile the source directly -- NOT `spec_from_file_location`.

    The bytecode cache is keyed on (mtime, size), and two edits of equal
    length inside one filesystem-timestamp tick hand back the STALE .pyc:
    while writing these tests a negative control read as green because the
    reverted file and the mutated one differed only in `==` vs `>=`.
    """
    module = types.ModuleType("_mm_bidi_ranges")
    module.__file__ = str(SRC)
    exec(compile(SRC.read_text(), str(SRC), "exec"), module.__dict__)
    return module


M = _load()


class _Placeholder:
    """A vLLM `PlaceholderRange` stand-in.

    `extract_embeds_range` returns INCLUSIVE `(first, last)` pairs, one per
    contiguous `is_embed` stretch -- the real contract, copied from
    vllm/multimodal/inputs.py.
    """

    def __init__(self, offset, length, is_embed=None):
        self.offset = offset
        self.length = length
        self._is_embed = is_embed

    def extract_embeds_range(self):
        if self._is_embed is None:
            return [(self.offset, self.offset + self.length - 1)]
        out, run_start = [], None
        for i, flag in enumerate(self._is_embed):
            if flag and run_start is None:
                run_start = i
            elif not flag and run_start is not None:
                out.append((self.offset + run_start, self.offset + i - 1))
                run_start = None
        if run_start is not None:
            out.append((self.offset + run_start,
                        self.offset + len(self._is_embed) - 1))
        return out


class _Feature:

    def __init__(self, position, modality="image"):
        self.mm_position = position
        self.modality = modality


class _Logger:

    def __init__(self):
        self.warnings = []
        self.infos = []

    def warning(self, fmt, *args):
        self.warnings.append(fmt % args)

    def info(self, fmt, *args):
        self.infos.append(fmt % args)


def _image(offset, length, is_embed=None):
    return _Feature(_Placeholder(offset, length, is_embed))


# --------------------------------------------------------------------- #
# One image: the span is the placeholder run, half-open, and nothing else
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("offset,length", [
    (14, 1092),
    (0, 1),
    (7, 1104),
    (2048, 1092),
])
def test_one_image_yields_exactly_the_placeholder_run(offset, length):
    log = _Logger()
    out = M.build_mm_bidi_ranges(8, [(0, "r0", [_image(offset, length)])],
                                 enabled=True,
                                 logger=log)
    assert out.shape == (8, 2)
    assert out.dtype == np.int32
    assert tuple(out[0]) == (offset, offset + length), (
        "the range is half-open [first, last+1); an off-by-one here either "
        "drops the last soft token out of the block or pulls the next text "
        "token into it")
    assert not log.warnings
    # Every other row is the "no block" sentinel.
    assert (out[1:] == 0).all()


def test_the_block_covers_every_soft_token_and_no_neighbour():
    """Character-level statement of the boundary, in token indices."""
    offset, length = 14, 1092
    out = M.build_mm_bidi_ranges(4, [(0, "r0", [_image(offset, length)])],
                                 enabled=True)
    start, end = int(out[0, 0]), int(out[0, 1])
    inside = set(range(start, end))
    assert inside == set(range(offset, offset + length))
    assert offset - 1 not in inside, "the token before the image is text"
    assert offset + length not in inside, "the token after the image is text"


# --------------------------------------------------------------------- #
# Two images: one operand row cannot describe two blocks
# --------------------------------------------------------------------- #


def test_two_images_get_no_range_rather_than_the_wrong_one():
    log = _Logger()
    feats = [_image(10, 1092), _image(1200, 1092)]
    out = M.build_mm_bidi_ranges(4, [(0, "r0", feats)],
                                 enabled=True,
                                 logger=log)
    assert tuple(out[0]) == (0, 0), (
        "a two-image request must fall back to causal, not silently apply "
        "the first image's range (which would make the second image's "
        "tokens attend into the first image's span)")
    assert len(log.warnings) == 1
    assert "2 image blocks" in log.warnings[0]
    assert "CAUSAL-ONLY" in log.warnings[0]


def test_the_two_image_warning_is_per_request_not_once_globally():
    log = _Logger()
    rows = [
        (0, "r0", [_image(10, 1092), _image(1200, 1092)]),
        (1, "r1", [_image(10, 1092)]),
        (2, "r2", [_image(10, 1092), _image(1200, 1092)]),
    ]
    out = M.build_mm_bidi_ranges(4, rows, enabled=True, logger=log)
    assert tuple(out[0]) == (0, 0)
    assert tuple(out[1]) == (10, 1102)
    assert tuple(out[2]) == (0, 0)
    assert len(log.warnings) == 2, (
        "a once-globally warning lets the second multi-image request hide "
        "behind the first")


def test_a_placeholder_run_broken_by_is_embed_holes_is_multi_block():
    """`is_embed` holes make ONE image several spans -> causal fallback."""
    log = _Logger()
    mask = [True] * 4 + [False] * 2 + [True] * 4
    spans = M.image_spans_for_request([_image(100, 10, is_embed=mask)])
    assert spans == [(100, 103), (106, 109)]
    out = M.build_mm_bidi_ranges(2, [(0, "r0", [_image(100, 10, mask)])],
                                 enabled=True,
                                 logger=log)
    assert tuple(out[0]) == (0, 0)
    assert len(log.warnings) == 1


# --------------------------------------------------------------------- #
# Modality, row layout, empties
# --------------------------------------------------------------------- #


def test_audio_features_are_not_blockwise_bidirectional():
    feats = [
        _Feature(_Placeholder(5, 300), modality="audio"),
        _image(400, 1092)
    ]
    out = M.build_mm_bidi_ranges(2, [(0, "r0", feats)], enabled=True)
    assert tuple(out[0]) == (400, 1492), (
        "an audio block counted as an image block would make this a "
        "two-block request and lose the image's range entirely")


def test_a_text_only_request_gets_the_no_block_sentinel():
    log = _Logger()
    out = M.build_mm_bidi_ranges(4, [(0, "r0", None), (1, "r1", [])],
                                 enabled=True,
                                 logger=log)
    assert (out == 0).all()
    assert not log.warnings and not log.infos


def test_rows_follow_the_dp_offset_persistent_batch_layout():
    """Row = dp_rank * max_num_reqs_per_dp_rank + i, same as seq_lens."""
    max_num_reqs, per_rank = 8, 4
    rows = []
    for dp_rank in range(2):
        for i in range(2):
            offset = 100 * (dp_rank + 1) + 10 * i
            rows.append((dp_rank * per_rank + i, f"r{dp_rank}{i}",
                         [_image(offset, 1092)]))
    out = M.build_mm_bidi_ranges(max_num_reqs, rows, enabled=True)
    assert tuple(out[0]) == (100, 1192)
    assert tuple(out[1]) == (110, 1202)
    assert tuple(out[4]) == (200, 1292), "dp_rank 1 starts at row 4"
    assert tuple(out[5]) == (210, 1302)
    for empty in (2, 3, 6, 7):
        assert tuple(out[empty]) == (0, 0)


# --------------------------------------------------------------------- #
# The feature can be globally inert -- and then it must SAY so
# --------------------------------------------------------------------- #


def test_the_feature_can_be_globally_inert():
    """`enabled=False` builds no operand at all.

    This is the E4B/E2B reality: `_init_mm_bidi` returns False for a text
    config that does not declare `use_bidirectional_attention="vision"`, so
    `AttentionMetadata.mm_bidi_ranges` is None and image tokens attend
    causally on BOTH paths. Any explanation of a torchax-vs-flax image
    difference that rests on the blockwise mask is therefore wrong for
    those checkpoints.
    """
    log = _Logger()
    out = M.build_mm_bidi_ranges(4, [(0, "r0", [_image(14, 1092)])],
                                 enabled=False,
                                 logger=log,
                                 debug=True)
    assert out is None
    assert len(log.infos) == 1
    assert "bidi=off" in log.infos[0]


def test_the_debug_line_names_the_blocks_and_what_the_kernel_carries():
    log = _Logger()
    M.build_mm_bidi_ranges(4, [(0, "r0", [_image(14, 1092)])],
                           enabled=True,
                           logger=log,
                           debug=True)
    assert len(log.infos) == 1
    line = log.infos[0]
    assert line.startswith(M.SPAN_DEBUG_PREFIX)
    assert "[14,1105]" in line, "the inclusive block, as the processor wrote it"
    assert "carried=[14,1106)" in line, "the half-open range the kernel reads"
    assert "req=r0" in line and "row=0" in line and "blocks=1" in line


def test_the_debug_line_distinguishes_a_causal_fallback_from_no_images():
    log = _Logger()
    M.build_mm_bidi_ranges(
        4, [(0, "r0", [_image(10, 1092), _image(1200, 1092)]),
            (1, "r1", None)],
        enabled=True,
        logger=log,
        debug=True)
    assert len(log.infos) == 1, "a text request has no spans to report"
    assert "causal-fallback" in log.infos[0]
    assert "blocks=2" in log.infos[0]


def test_debug_off_emits_nothing():
    log = _Logger()
    M.build_mm_bidi_ranges(4, [(0, "r0", [_image(14, 1092)])],
                           enabled=True,
                           logger=log)
    assert not log.infos


# --------------------------------------------------------------------- #
# Source-structure: the runner uses the helper, and BOTH paths read the
# operand under the SAME sliding-layer gate
# --------------------------------------------------------------------- #


def test_the_runner_builds_its_ranges_through_this_helper():
    src = RUNNER.read_text()
    assert "build_mm_bidi_ranges(" in src
    assert "from tpu_inference.runner.mm_bidi_ranges import" in src
    tree = ast.parse(src)
    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "build_mm_bidi_ranges"
    ]
    assert len(calls) == 1, (
        "one construction site; a second would be a second layout to keep "
        "in sync with seq_lens")
    kwargs = {k.arg for k in calls[0].keywords if k.arg}
    assert {"enabled", "logger", "debug"} <= kwargs


@pytest.mark.parametrize("path", [FLAX_ATTN, TORCHAX_ATTN])
def test_both_paths_read_mm_bidi_ranges_under_the_sliding_window_gate(path):
    """HF composition is AND(sliding_window, OR(causal, blockwise)).

    Full-attention layers must stay purely causal on both paths; a path that
    reads the operand unconditionally would make global layers bidirectional
    over the image.
    """
    # AUDIT 2026-09-03: this was two unrelated substrings anywhere in a
    # 1000+ line file plus a 300-CHARACTER PROXIMITY WINDOW around
    # `src.index(...)` -- which takes only the FIRST occurrence. It holds
    # today because each file has exactly one quoted "mm_bidi_ranges", but a
    # second read site added anywhere later is unchecked and the test stays
    # green. Walk every read instead.
    import ast as _ast
    src = path.read_text()
    tree = _ast.parse(src)
    parents = {}
    for node in _ast.walk(tree):
        for child in _ast.iter_child_nodes(node):
            parents[child] = node
    reads = [
        n for n in _ast.walk(tree)
        if isinstance(n, _ast.Call) and _ast.unparse(n.func) == "getattr"
        and len(n.args) > 1 and isinstance(n.args[1], _ast.Constant)
        and n.args[1].value == "mm_bidi_ranges"
    ]
    assert reads, f"{path.name} never reads mm_bidi_ranges via getattr"
    for read in reads:
        gated, node = False, read
        while node in parents:
            node = parents[node]
            if isinstance(
                    node,
                (_ast.IfExp, _ast.If)) and ("self.sliding_window is not None"
                                            in _ast.unparse(node.test)):
                gated = True
                break
        assert gated, (
            f"{path.name}: an mm_bidi_ranges read at line {read.lineno} is "
            f"not gated on the layer being a sliding-window layer -- a "
            f"full-attention layer would go bidirectional over the image")


def test_the_runner_walks_the_rows_even_when_the_feature_is_off():
    """The probe must be able to say `bidi=off`, not just stay silent."""
    src = RUNNER.read_text()
    assert "if self.mm_bidi_enabled or mm_bidi_debug:" in src, (
        "with `if self.mm_bidi_enabled:` alone, an inert mask and a "
        "text-only request look identical in the log")
