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
"""One log line of vision-encoder statistics per call (PIECES_MM_DEBUG=1).

Shared by the native flax path (``models/jax/gemma4_mm.py``,
``models/jax/gemma4_unified.py``) and the torchax path
(``models/vllm/experimental/mm_debug_patch.py``) so the two paths write the
SAME keys and a per-request diff between them is a text diff.

The statistics are computed on the host, in float32/float64 NumPy, from a
copy handed over by ``jax.debug.callback``. That has two consequences the
call sites rely on:

* nothing here touches device numerics or dtypes -- the arrays that flow on
  are the arrays the model produced;
* it works both on concrete arrays (the callback runs immediately) and on
  tracers inside ``jax.jit`` (the callback runs when the compiled program
  does), so one helper serves an eager torchax call and a jitted flax
  method alike.

Every call site guards the call with a trace-time Python ``if`` on the flag,
so with the flag off no ``debug_callback`` op is added to any jaxpr. This
module deliberately imports only jax and numpy: the CPU gate loads it by
path on a runner that has neither torch nor vLLM.

Line format (one line, ``key=value`` pairs, a tensor's keys share a prefix)::

    [mm-debug] path=native call=3 site=get_single_image_embedding n_images=1
      pv.shape=(1,10080,768) pv.dtype=bfloat16 pv.min=-1 pv.max=1 pv.mean=..
      pv.std=.. pv.maxabs=.. pv.nan=0 pv.inf=0
      enc.shape=.. enc.dtype=.. enc.min=.. .. enc.nan=0 enc.inf=0
      tower.shape=.. .. proj.shape=.. .. soft_tokens=280

``pv`` is pixel_values, ``enc`` the pre-pooler encoder output (tower variant
only), ``tower`` the vision-tower output as the projector consumes it, and
``proj`` the projector (embed_vision) output. Row masks restrict a tensor's
statistics to valid (non-padding) rows so both paths measure the same
elements; ``shape`` always reports the unmasked device shape.
"""

import functools
import itertools
import math
from typing import Any, Callable, Mapping, Sequence

import jax
import numpy as np

LINE_PREFIX = "[mm-debug]"
STAT_KEYS = ("shape", "dtype", "min", "max", "mean", "std", "maxabs", "nan",
             "inf")

_CALL_COUNTER = itertools.count(1)


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [v for v in value if v is not None]
    return [value]


def _shape_str(shapes: Sequence[tuple[int, ...]]) -> str:
    """``(1,2,3)`` for one array, ``2x(1,2,3)`` for uniform chunks, else a
    bracketed list -- never a space, so the line stays one token per field."""
    rendered = ["(" + ",".join(str(int(d)) for d in s) + ")" for s in shapes]
    if len(rendered) == 1:
        return rendered[0]
    if len(set(rendered)) == 1:
        return f"{len(rendered)}x{rendered[0]}"
    return "[" + ",".join(rendered) + "]"


def _dtype_str(dtype: Any) -> str:
    try:
        return np.dtype(dtype).name
    except TypeError:
        return str(dtype)


def _fmt(value: float) -> str:
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return f"{value:.6g}"


def host_tensor_stats(chunks: Sequence[np.ndarray],
                      masks: Sequence[np.ndarray] | None = None) -> dict:
    """Stats over the concatenation of ``chunks`` (host NumPy, float32 view).

    ``masks[i]``, when given, is a boolean row mask of shape
    ``chunks[i].shape[:-1]`` selecting the rows that count. NaN/Inf are
    counted over the selected elements; min/max/mean/std/maxabs are computed
    over the FINITE selected elements so a handful of NaNs still leaves the
    magnitude of the rest readable.
    """
    count = 0
    total = 0.0
    total_sq = 0.0
    lo = math.inf
    hi = -math.inf
    maxabs = 0.0
    nan = 0
    inf = 0
    for i, chunk in enumerate(chunks):
        x = np.asarray(chunk)
        if masks is not None and i < len(masks) and masks[i] is not None:
            x = x[np.asarray(masks[i], dtype=bool)]
        x = np.asarray(x, dtype=np.float32).reshape(-1)
        nan += int(np.isnan(x).sum())
        inf += int(np.isinf(x).sum())
        finite = x[np.isfinite(x)]
        if finite.size == 0:
            continue
        f64 = finite.astype(np.float64)
        count += int(finite.size)
        total += float(f64.sum())
        total_sq += float(np.square(f64).sum())
        lo = min(lo, float(finite.min()))
        hi = max(hi, float(finite.max()))
        maxabs = max(maxabs, float(np.abs(finite).max()))
    if count:
        mean = total / count
        std = math.sqrt(max(total_sq / count - mean * mean, 0.0))
    else:
        mean = std = lo = hi = math.nan
    return {
        "count": count,
        "min": lo,
        "max": hi,
        "mean": mean,
        "std": std,
        "maxabs": maxabs,
        "nan": nan,
        "inf": inf,
    }


def format_mm_debug_line(
    path: str,
    call: int,
    meta: Mapping[str, tuple[str, str]],
    stats: Mapping[str, Mapping[str, Any]],
    counts: Mapping[str, int],
    extra: Mapping[str, Any],
) -> str:
    fields = [f"path={path}", f"call={call}"]
    fields += [f"{k}={v}" for k, v in extra.items()]
    for name, (shape, dtype) in meta.items():
        st = stats[name]
        fields.append(f"{name}.shape={shape}")
        fields.append(f"{name}.dtype={dtype}")
        for key in ("min", "max", "mean", "std", "maxabs"):
            fields.append(f"{name}.{key}={_fmt(st[key])}")
        fields.append(f"{name}.nan={st['nan']}")
        fields.append(f"{name}.inf={st['inf']}")
    fields += [f"{k}={v}" for k, v in counts.items()]
    return LINE_PREFIX + " " + " ".join(fields)


def _host_emit(log: Callable[[str],
                             None], path: str, meta: Mapping[str, tuple[str,
                                                                        str]],
               static_counts: Mapping[str, int], extra: Mapping[str, Any],
               chunks: Mapping[str, list], mask_chunks: Mapping[str, list],
               count_arrays: Mapping[str, list]) -> None:
    stats = {
        name: host_tensor_stats(chunks[name], mask_chunks.get(name))
        for name in meta
    }
    counts = dict(static_counts)
    for name, arrays in count_arrays.items():
        counts[name] = int(
            sum(int(np.asarray(a, dtype=np.int64).sum()) for a in arrays))
    log(
        format_mm_debug_line(path, next(_CALL_COUNTER), meta, stats, counts,
                             extra))


def emit_mm_debug_stats(
    log: Callable[[str], None],
    path: str,
    *,
    tensors: Mapping[str, Any],
    masks: Mapping[str, Any] | None = None,
    counts: Mapping[str, Any] | None = None,
    extra: Mapping[str, Any] | None = None,
) -> None:
    """Log ONE ``[mm-debug]`` line for this encoder call.

    Args:
        log: sink for the finished line (a logger's ``info``).
        path: ``"native"`` or ``"torchax"``.
        tensors: ordered ``name -> array | list[array] | None``. A list is
            treated as chunks of one tensor (several encoder micro-batches,
            or one array per image) and reported as one entry. ``None`` and
            empty lists are skipped.
        masks: ``name -> bool array | list[bool array]`` row masks aligned
            with ``tensors[name]`` (shape ``array.shape[:-1]``).
        counts: ``name -> int | array | list[array]``; arrays are summed on
            the host (a valid-token mask gives the soft-token count).
        extra: static ``key -> value`` pairs written before the tensors.

    Call this ONLY under ``envs.PIECES_MM_DEBUG``: it adds a
    ``jax.debug.callback`` when traced, and a host round trip when not.
    """
    meta: dict[str, tuple[str, str]] = {}
    chunks: dict[str, list] = {}
    for name, value in tensors.items():
        arrays = _as_list(value)
        if not arrays:
            continue
        meta[name] = (_shape_str([tuple(a.shape) for a in arrays]),
                      _dtype_str(arrays[0].dtype))
        chunks[name] = arrays
    mask_chunks = {
        name: _as_list(value)
        for name, value in (masks or {}).items() if name in chunks
    }
    static_counts: dict[str, int] = {}
    count_arrays: dict[str, list] = {}
    for name, value in (counts or {}).items():
        if value is None:
            continue
        if isinstance(value, (int, np.integer)):
            static_counts[name] = int(value)
        else:
            count_arrays[name] = _as_list(value)
    host = functools.partial(_host_emit, log, path, meta, static_counts,
                             dict(extra or {}))
    jax.debug.callback(host, chunks, mask_chunks, count_arrays)
