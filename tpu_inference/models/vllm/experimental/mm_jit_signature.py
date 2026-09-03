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
"""Derive the JIT static-argument split for a jitted multimodal submodule.

``patch_mm_model`` wraps each module named in ``JITTED_MM_MODULE_KEYS`` in a
``torchax.interop.JittableModule``. Some encoders take a grid/shape argument
that drives Python-level control flow and therefore has to be a ``jax.jit``
static argument; most do not.

The static split is a property of *that module's forward signature*, not of
the fleet-wide config, so it must be derived per module. A fixed mapping
copied from one architecture silently addresses a different parameter in
another: Qwen3-VL's ``static_argnums=(3,)`` is ``grid_thw``, but the same
index on Gemma-4's ``Gemma4VisionEncoder.forward`` is ``attention_mask``.

This module is deliberately stdlib-only so it can be exercised on a CPU
runner that has neither torch, torchax nor vLLM installed.
"""

import inspect
from typing import Any, Callable, Sequence

# ``torchax.interop.JittableModule`` jits
# ``functional_call(method_name, params, buffers, *args, **kwargs)`` with
# ``method_name`` bound by ``functools.partial``, then invokes it as
# ``jitted(self.params, self.buffers, *args, **kwargs)``. The module's own
# i-th positional forward parameter therefore lands at jit positional index
# ``i + 2``.
JITTABLE_MODULE_POSITIONAL_OFFSET = 2

# Forward-parameter names that must be jit-static when a jitted multimodal
# module declares them. Only grid metadata that is passed as a hashable
# Python value belongs here -- Qwen3-VL's ``grid_thw`` arrives as the
# ``vision_tower_jit.GridTHW`` tuple precisely so it can be static. Adding a
# name here makes it static for every module that declares it, so the list
# stays at the single name the previous hard-coded mapping carried.
MM_STATIC_ARGNAMES: tuple[str, ...] = ("grid_thw", )

_POSITIONAL_KINDS = (
    inspect.Parameter.POSITIONAL_ONLY,
    inspect.Parameter.POSITIONAL_OR_KEYWORD,
)


def compute_mm_jit_static_args(
    forward: Callable[..., Any],
    static_argnames: Sequence[str] = MM_STATIC_ARGNAMES,
) -> dict[str, tuple]:
    """Return the ``extra_jit_args`` for jitting ``forward``.

    Args:
        forward: The *bound* forward method of the module about to be wrapped
          in a ``JittableModule`` (bound, so ``self`` is not a parameter --
          which matches what ``JittableModule`` jits).
        static_argnames: Parameter names to mark static when ``forward``
          declares them.

    Returns:
        A dict suitable for ``JittableModule(extra_jit_args=...)`` holding
        ``static_argnums`` and ``static_argnames``, or an empty dict when
        ``forward`` declares none of ``static_argnames``. An empty dict is
        returned rather than empty tuples because ``jax.jit`` treats an
        omitted and an empty static spec identically, and the empty dict says
        plainly that this module has no static arguments.

    A parameter that can be passed positionally contributes both an argnum
    (its positional index plus ``JITTABLE_MODULE_POSITIONAL_OFFSET``) and its
    name; a keyword-only parameter contributes only its name. Nothing is
    emitted for a name the signature does not declare, so a static index can
    never land on an argument that happens to sit at that position.
    """
    try:
        parameters = inspect.signature(forward).parameters
    except (TypeError, ValueError):
        # Builtins and C-implemented callables have no introspectable
        # signature. No static args is the safe answer: every argument is
        # traced, which is correct for any signature, only less specialised.
        return {}

    wanted = set(static_argnames)
    argnums: list[int] = []
    argnames: list[str] = []
    positional_index = 0

    for name, parameter in parameters.items():
        if parameter.kind in _POSITIONAL_KINDS:
            if name in wanted:
                argnums.append(positional_index +
                               JITTABLE_MODULE_POSITIONAL_OFFSET)
                argnames.append(name)
            positional_index += 1
        elif parameter.kind is inspect.Parameter.KEYWORD_ONLY:
            if name in wanted:
                argnames.append(name)

    if not argnums and not argnames:
        return {}
    return {
        "static_argnums": tuple(argnums),
        "static_argnames": tuple(argnames),
    }
