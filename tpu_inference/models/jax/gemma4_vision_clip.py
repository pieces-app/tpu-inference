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
"""Activation clipping for the Gemma-4 vision tower, in pure JAX.

WHAT THIS IS
------------
``transformers``' ``Gemma4ClippableLinear`` (models/gemma4/modeling_gemma4.py)
is the reference for every Gemma-4 vision/audio projection::

    if use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, input_min, input_max)
    hidden_states = self.linear(hidden_states)
    if use_clipped_linears:
        hidden_states = torch.clamp(hidden_states, output_min, output_max)

The four bounds are per-projection scalars that ship IN THE CHECKPOINT.
``google/gemma-4-E4B-it`` sets ``vision_config.use_clipped_linears: true`` and
carries 448 finite BF16 clamp tensors for the vision tower -- 16 layers x
{q,k,v,o,gate,up,down}_proj x {input,output}x{min,max} -- with bounds as tight
as +-1.91 (``o_proj.input_max`` median 2.35).  They are not a debug artefact:
they are calibrated activation ranges, and dropping them changes the encoder's
output representation, not merely its numerics.

WHY IT LIVES IN ITS OWN MODULE
------------------------------
``models/jax/gemma4_mm.py`` imports torch, vllm and transformers, so the CPU
gate (jax[cpu] + flax + numpy, no vllm/torch) cannot import it.  This module
depends on jax alone, so the clipping arithmetic and the load-completeness
check are testable on the gate machine; ``gemma4_mm.py`` calls into here
rather than open-coding either.

MEASURED (2026-09-03, CPU differential, tests/models/jax/
test_gemma4_vision_clipping.py plus the torch half in the PR body):
with the clamps honoured the flax tower reproduces the transformers tower to
fp32 rounding (cos 0.999998 after 16 layers); with them dropped the two
diverge at layer 0 and end at cos 0.35 on the same weights and input.
"""

from typing import Any, Callable, Iterable, Sequence, Tuple

import jax
import jax.numpy as jnp

# The four checkpoint tensors that carry a projection's clamp, exactly as
# transformers registers them (``nn.Buffer``, persistent, scalar).
CLAMP_SUFFIXES: Tuple[str, ...] = (
    ".input_min",
    ".input_max",
    ".output_min",
    ".output_max",
)


def clamp_activation(x: jax.Array, lo: Any, hi: Any) -> jax.Array:
    """``torch.clamp(x, lo, hi)`` with the bounds read in ``x``'s dtype.

    transformers clamps a bf16 activation against bf16 buffers, so the bound
    is rounded to the activation dtype on both paths and the two agree
    exactly.  Casting here (rather than at load time) also keeps a clamp
    correct when the activation dtype and the checkpoint dtype differ.
    """
    return jnp.clip(x, jnp.asarray(lo, x.dtype), jnp.asarray(hi, x.dtype))


def neutral_clamps() -> Tuple[jax.Array, jax.Array]:
    """``(min, max)`` a clamp pair holds before the checkpoint fills it.

    transformers constructs the buffers at -/+inf and its ``_init_weights``
    resets them to -/+inf, which makes ``clamp`` a no-op.  Initialising to the
    same thing means an unloaded clamp degrades to the previous, unclipped
    behaviour instead of zeroing the activation -- and ``unloaded_clamps``
    below refuses to let that state reach a forward pass silently.
    """
    return (jnp.asarray(-jnp.inf,
                        jnp.float32), jnp.asarray(jnp.inf, jnp.float32))


def unloaded_clamps(
    named_params: Iterable[Tuple[str, Any]],
    is_loaded: Callable[[Any], bool],
) -> Sequence[str]:
    """Names of declared clamp params the checkpoint did not fill.

    A clamp left at its -/+inf init clips nothing, so a load that misses one
    is indistinguishable at runtime from having no clamp at all.  That is the
    exact failure this module exists to end, so the caller turns a non-empty
    result into a boot-time error rather than a wrong picture.
    """
    return [
        name for name, param in named_params
        if name.endswith(CLAMP_SUFFIXES) and not is_loaded(param)
    ]
