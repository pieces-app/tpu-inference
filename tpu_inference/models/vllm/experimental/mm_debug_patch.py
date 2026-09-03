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
"""PIECES_MM_DEBUG=1 on the torchax path: per-call vision stats and the
boot-time census of what runs inside the vision tower.

Installed by ``model_patcher.apply_model_specific_patches`` ONLY under the
flag, after ``patch_mm_model`` has (optionally) wrapped
``model.vision_tower.encoder`` in a ``torchax.interop.JittableModule``. It
never touches the module tree -- ``torch.func.functional_call`` keys weights
by module path -- it only replaces ``forward`` on three instances and two
bound methods:

* ``vision_tower.encoder.forward``: records ``last_hidden_state`` (the
  pre-pooler encoder output, i.e. the output of the jitted region) and the
  ``attention_mask`` it was called with;
* ``vision_tower.pooler.forward``: records the ``(pooled_states,
  valid_mask)`` pair as ``tower`` -- see WHY THE POOLER below;
* ``embed_vision.forward``: records its output (the projector output), and
  its input as ``tower`` only when the tower exposes no pooler to hook;
* ``_process_image_input`` (the eager vLLM path -- what the
  ``eval-e4b-torchax`` lane runs) and ``encoder_cudagraph_forward`` (the
  MM-encoder JIT manager's traced path, hooked as defensive coverage):
  reset the recorder, run, then emit ONE line for the whole call.

Under ``PIECES_MM_DEBUG_LAYERS`` it additionally wraps every encoder layer
and every attention module inside the tower and emits a SECOND line,
``site=<site>:layers``, carrying ``A0..An`` (each layer's attention output)
and ``L0..Ln`` (each layer's output).  That is the instrument for the one
thing a CPU differential cannot settle: the 2026-09-03 CPU run showed the
torchax tower reproducing transformers' eager tower to bf16 noise at every
layer and every grid on the real E4B weights, so a divergence that only
appears on a chip has to be localised on the chip.  The flax tower emits
the same two names from ``models/jax/gemma4_mm.py``, so the two lanes'
lines can be diffed layer by layer.

WHY THE POOLER, and not the projector's input, is ``tower``
-----------------------------------------------------------
The two runtimes hand the projector different tensors for the same image,
and reading ``tower`` off the projector made that difference look like lost
embeddings. For a 3402x2158 screenshot at ``max_soft_tokens=1120`` the
pooler produces a ``(1, 1120, D)`` buffer of which 1092 rows are valid:

* the flax path (``models/jax/gemma4_mm.py``) keeps the padding, projects
  all 1120 rows and sorts the valid ones to the front, so it logged
  ``tower.shape=(1,1120,D)``;
* the vLLM path strips the padding first (``pooled_states[valid_mask]``,
  matching HF's own ``Gemma4VisionModel.forward``) and projects 1092 rows,
  so it logged ``tower.shape=(1,1092,D)``.

Both are correct and both feed the language model 1092 embeddings for the
1092 ``<image>`` placeholders the processor put in the prompt -- 1092 is the
answer, 1120 is the buffer. Recording the pooler's own output makes the two
lines report the same tensor at the same point, with the pooler's mask
restricting the statistics to the valid rows, so the padded-vs-packed
difference can no longer be misread as 28 missing image embeddings.
``proj`` still differs in shape by construction (the vLLM projector runs on
the packed rows); ``soft_tokens`` is the valid count on BOTH paths and is
the number that has to match the placeholders.

The recorder holds torchax tensors converted with the injected ``to_jax``
(``jax_view(t.detach())`` in the patcher); the stats themselves are computed
on the host by ``mm_debug_stats.emit_mm_debug_stats`` through
``jax.debug.callback``, which is why the same code serves the eager path
(concrete arrays, logged immediately) and the traced path (tracers, logged
when the compiled encoder runs).

This module imports neither torch nor vLLM so the CPU gate can drive it with
plain-Python stand-ins; ``JittableModule`` and ``torch.nn.Module`` both
resolve ``self.forward`` through the instance first, which is the property
the stand-ins mirror.
"""

from collections import Counter
from typing import Any, Callable, Sequence

import jax.numpy as jnp

POSITIONS_PAD_VALUE = -1
INSTALLED_ATTR = "_pieces_mm_debug_installed"


def _as_list(value: Any) -> list:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _field(obj: Any, name: str) -> Any:
    """``obj[name]`` (TensorSchema / dict) with an attribute fallback."""
    try:
        return obj[name]
    except (KeyError, TypeError, IndexError):
        return getattr(obj, name, None)


def _valid_rows(position_ids: Any) -> Any:
    """Rows whose position ids are not the padding value (bool, [..., N])."""
    return jnp.logical_not(
        jnp.all(position_ids == POSITIONS_PAD_VALUE, axis=-1))


class _Recorder:

    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.enc: list = []
        self.enc_mask: list = []
        # ``tower`` is the pooler's padded output when a pooler was hooked
        # (``tower_mask`` then carries its valid-row mask), and the
        # projector's input otherwise. ``tower_mask`` is empty in the
        # fallback: those rows are already stripped.
        self.tower: list = []
        self.tower_mask: list = []
        self.tower_from_pooler = False
        self.proj: list = []
        # PIECES_MM_DEBUG_LAYERS: index -> chunks, one entry per encoder
        # layer. Kept sparse (a dict) so a partial run still formats.
        self.attn: dict[int, list] = {}
        self.layer: dict[int, list] = {}


def _wrap_forward(module: Any, after: Callable[[tuple, dict, Any],
                                               None]) -> None:
    """Replace ``module.forward`` on the INSTANCE with a recording wrapper.

    The wrapped call is unchanged; ``after(args, kwargs, output)`` runs on
    its result. Works for ``torch.nn.Module`` (``__call__`` resolves
    ``self.forward`` per instance) and for ``JittableModule`` (whose
    ``forward`` is the eager entry into its jitted ``functional_call``).
    """
    orig = module.forward

    def forward(*args, **kwargs):
        out = orig(*args, **kwargs)
        after(args, kwargs, out)
        return out

    forward._mm_debug_wrapped = orig
    module.forward = forward


ENCODER_LAYER_SUFFIX = "EncoderLayer"
ATTENTION_SUFFIX = "Attention"


def _install_layer_hooks(tower: Any, rec: "_Recorder",
                         to_jax: Callable[[Any], Any]) -> list[str]:
    """Record each encoder layer's attention output and layer output.

    Ordering is the module tree's, which for ``nn.ModuleList`` is the layer
    order, so ``A3``/``L3`` are layer 3's. Both hooks run inside the jitted
    region when the encoder is a ``JittableModule``; the recorded values are
    then tracers, and ``emit_mm_debug_stats`` turns them into one host
    ``jax.debug.callback`` -- the same mechanism the existing fields use.
    """
    named = getattr(tower, "named_modules", None)
    if not callable(named):
        return []
    layers = [(n, m) for n, m in named()
              if type(m).__name__.endswith(ENCODER_LAYER_SUFFIX)]
    attns = [(n, m) for n, m in named()
             if type(m).__name__.endswith(ATTENTION_SUFFIX)]

    def _recorder(attribute: str, index: int):
        # The store is looked up on ``rec`` at CALL time, not captured here:
        # ``rec.reset()`` rebinds the dicts before every encoder call, and a
        # captured dict would collect into the previous call's store.

        def after(args: tuple, kwargs: dict, out: Any) -> None:
            value = out[0] if isinstance(out, (tuple, list)) and out else out
            value = getattr(value, "last_hidden_state", value)
            getattr(rec, attribute).setdefault(index, []).append(to_jax(value))

        return after

    installed: list[str] = []
    for index, (name, module) in enumerate(attns):
        if callable(getattr(module, "forward", None)):
            _wrap_forward(module, _recorder("attn", index))
            installed.append(f"attn[{index}]={name}")
    for index, (name, module) in enumerate(layers):
        if callable(getattr(module, "forward", None)):
            _wrap_forward(module, _recorder("layer", index))
            installed.append(f"layer[{index}]={name}")
    return installed


def install_mm_debug(
    vllm_model: Any,
    *,
    to_jax: Callable[[Any], Any],
    log: Callable[[str], None],
    emit: Callable[..., None] | None = None,
    path: str = "torchax",
    per_layer: bool = False,
) -> list[str]:
    """Install the recorder on ``vllm_model``; return what was hooked.

    Args:
        vllm_model: the vLLM model instance (Gemma-4 tower or Unified).
        to_jax: converts a tensor the model produced into a jax array
            without changing it (the patcher passes ``jax_view(t.detach())``).
        log: sink for the finished line.
        emit: ``mm_debug_stats.emit_mm_debug_stats`` (imported lazily so this
            module stays importable without the ``tpu_inference`` package).
        path: the ``path=`` field of the line.
        per_layer: also hook every encoder layer and attention module in the
            tower and emit the second, ``site=<site>:layers`` line
            (``PIECES_MM_DEBUG_LAYERS``).

    Returns the list of hook names installed; empty when the model has no
    vision-shaped members. Idempotent per instance.
    """
    if getattr(vllm_model, INSTALLED_ATTR, False):
        return []
    if emit is None:
        from tpu_inference.models.common.mm_debug_stats import \
            emit_mm_debug_stats
        emit = emit_mm_debug_stats

    rec = _Recorder()
    installed: list[str] = []

    tower = getattr(vllm_model, "vision_tower", None)
    encoder = getattr(tower, "encoder", None) if tower is not None else None
    if encoder is not None and callable(getattr(encoder, "forward", None)):

        def after_encoder(args: tuple, kwargs: dict, out: Any) -> None:
            hidden = getattr(out, "last_hidden_state", out)
            rec.enc.append(to_jax(hidden))
            mask = kwargs.get("attention_mask")
            if mask is None and len(args) > 1:
                mask = args[1]
            rec.enc_mask.append(None if mask is None else to_jax(mask))

        _wrap_forward(encoder, after_encoder)
        installed.append("vision_tower.encoder")

    pooler = getattr(tower, "pooler", None) if tower is not None else None
    if pooler is not None and callable(getattr(pooler, "forward", None)):

        def after_pooler(args: tuple, kwargs: dict, out: Any) -> None:
            # Gemma4VisionPooler.forward -> (pooled_states, valid_mask).
            if not isinstance(out, (tuple, list)) or len(out) != 2:
                return
            pooled, mask = out
            rec.tower.append(to_jax(pooled))
            rec.tower_mask.append(None if mask is None else to_jax(mask))
            rec.tower_from_pooler = True

        _wrap_forward(pooler, after_pooler)
        installed.append("vision_tower.pooler")

    embed_vision = getattr(vllm_model, "embed_vision", None)
    if embed_vision is not None and callable(
            getattr(embed_vision, "forward", None)):

        def after_embed_vision(args: tuple, kwargs: dict, out: Any) -> None:
            # Only the fallback: with a pooler hooked, ``tower`` is already
            # the padded pooled buffer and its mask, which is what the flax
            # path reports.
            if not rec.tower_from_pooler:
                inputs = kwargs.get("inputs_embeds")
                if inputs is None and args:
                    inputs = args[0]
                if inputs is not None:
                    rec.tower.append(to_jax(inputs))
            rec.proj.append(to_jax(out))

        _wrap_forward(embed_vision, after_embed_vision)
        installed.append("embed_vision")

    if per_layer and tower is not None:
        installed += _install_layer_hooks(tower, rec, to_jax)

    def _emit(site: str, pixel_values: Any, position_ids: Any,
              soft_tokens: int, n_images: int) -> None:
        pv = [to_jax(t) for t in _as_list(pixel_values)]
        pos = [to_jax(t) for t in _as_list(position_ids)]
        pv_mask = [_valid_rows(p)
                   for p in pos] if len(pos) == len(pv) else None
        enc_masks = rec.enc_mask if all(m is not None
                                        for m in rec.enc_mask) else None
        tower_masks = rec.tower_mask if (rec.tower_mask and all(
            m is not None for m in rec.tower_mask)) else None
        emit(
            log,
            path,
            tensors={
                "pv": pv,
                "enc": list(rec.enc),
                "tower": list(rec.tower),
                "proj": list(rec.proj),
            },
            masks={
                "pv": pv_mask,
                "enc": enc_masks,
                "tower": tower_masks,
            },
            counts={"soft_tokens": int(soft_tokens)},
            extra={
                "site": site,
                "n_images": int(n_images)
            },
        )
        if not rec.layer and not rec.attn:
            return
        per_layer_tensors: dict = {}
        for prefix, store in (("A", rec.attn), ("L", rec.layer)):
            for index in sorted(store):
                per_layer_tensors[f"{prefix}{index}"] = list(store[index])
        # The encoder's chunks are per encoder CALL and pv's are per image;
        # they line up only when the call carried one image, which is what
        # the debug lane runs. Drop the masks rather than mis-index when
        # they do not, so the line still prints (over all rows, padding
        # included -- which the shape field makes visible).
        per_layer_masks = None
        if pv_mask:
            per_layer_masks = {
                name: pv_mask
                for name, chunks in per_layer_tensors.items()
                if len(chunks) == len(pv_mask) and all(
                    tuple(c.shape[:-1]) == tuple(m.shape)
                    for c, m in zip(chunks, pv_mask))
            } or None
        emit(
            log,
            path,
            tensors=per_layer_tensors,
            masks=per_layer_masks,
            counts={},
            extra={
                "site": f"{site}:layers",
                "n_images": int(n_images)
            },
        )

    orig_process = getattr(vllm_model, "_process_image_input", None)
    if callable(orig_process):

        def _process_image_input(image_input: Any) -> Any:
            rec.reset()
            out = orig_process(image_input)
            outputs = _as_list(out)
            _emit(
                "_process_image_input",
                _field(image_input, "pixel_values"),
                _field(image_input, "pixel_position_ids"),
                soft_tokens=sum(int(e.shape[0]) for e in outputs),
                n_images=len(outputs),
            )
            return out

        _process_image_input._mm_debug_wrapped = orig_process
        vllm_model._process_image_input = _process_image_input
        installed.append("_process_image_input")

    orig_cudagraph = getattr(vllm_model, "encoder_cudagraph_forward", None)
    if callable(orig_cudagraph):

        def encoder_cudagraph_forward(inputs: Any, *args: Any,
                                      **kwargs: Any) -> Any:
            rec.reset()
            out = orig_cudagraph(inputs, *args, **kwargs)
            pixel_values = _field(inputs, "pixel_values")
            n_images = int(pixel_values.shape[0]) if hasattr(
                pixel_values, "shape") else 0
            _emit(
                "encoder_cudagraph_forward",
                pixel_values,
                _field(inputs, "pixel_position_ids"),
                soft_tokens=int(out.shape[0]),
                n_images=n_images,
            )
            return out

        encoder_cudagraph_forward._mm_debug_wrapped = orig_cudagraph
        vllm_model.encoder_cudagraph_forward = encoder_cudagraph_forward
        installed.append("encoder_cudagraph_forward")

    if installed:
        setattr(vllm_model, INSTALLED_ATTR, True)
    return installed


def describe_linears(root: Any) -> str:
    """Census of the linear implementations under ``root``, one string.

    Counts every module whose class name contains ``Linear`` (HF
    ``nn.Linear``, and the vLLM ``LinearBase`` subclasses that
    ``recursive_replace_linear`` puts in their place) or that carries a
    ``quant_method`` (the tpu_inference replacement lives there), keyed as
    ``Class[QuantMethodClass]``. ``type()`` is used on purpose:
    ``JittableModule`` lies about ``__class__``.
    """
    counts: Counter = Counter()
    named = getattr(root, "named_modules", None)
    members: Sequence[tuple[str,
                            Any]] = list(named()) if callable(named) else []
    for _, module in members:
        cls = type(module).__name__
        quant = getattr(module, "quant_method", None)
        if "Linear" in cls or quant is not None:
            key = f"{cls}[{type(quant).__name__ if quant is not None else '-'}]"
            counts[key] += 1
    encoder = getattr(root, "encoder", None)
    config = getattr(root, "config", None)
    attn = getattr(config, "_attn_implementation", None)
    linears = ", ".join(f"{k}={v}" for k, v in sorted(counts.items()))
    return (
        f"root={type(root).__name__} "
        f"encoder={type(encoder).__name__ if encoder is not None else '-'} "
        f"attn_impl={attn} linears={{{linears}}} "
        f"total={sum(counts.values())}")


def tower_census_lines(vllm_model: Any) -> list[str]:
    """Boot-time lines: one per vision-shaped member the model has."""
    lines = []
    for attr in ("vision_tower", "vision_embedder", "embed_vision"):
        member = getattr(vllm_model, attr, None)
        if member is None:
            continue
        lines.append(f"census {attr}: {describe_linears(member)}")
    return lines
