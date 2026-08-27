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
"""Regression test: resume of a KV-preempted multimodal request must not
kill the engine (observed live 2026-08-27, v6e-1 gemma-4-12B, vLLM d626108b,
93-97% KV usage: `AttributeError: 'NoneType' object has no attribute
'keys'/'items'` in vllm/multimodal/utils.py via execute_mm_encoder).

Mechanism (all on CPU, no TPU needed):

1. ADMISSION with a prefix-cache hit covering the image placeholder
   (duplicate image: an identical earlier request cached those blocks).
   vLLM #52041's `strip_covered_mm_data` nulls the payload out of the
   `NewRequestData` wire copy; the worker's CachedRequestState keeps that
   stripped copy for the request's whole lifetime (resume travels via
   CachedRequestData, which carries no mm payloads).
2. PREEMPTION under KV pressure: blocks freed, num_computed_tokens=0,
   per-request encoder cache freed (Scheduler._preempt_request).
3. RESUME after the covering blocks were evicted: the scheduler
   legitimately re-schedules the encoder input
   (_try_schedule_encoder_inputs), but the worker no longer has the
   pixels -> engine death in group_and_batch_mm_kwargs.

The fix is two-layered and both layers are covered here:
- `patch_vllm_mm_data_strip_for_preemption` (tpu_inference.core.sched.utils,
  applied by TpuPlatform.check_and_update_config) neutralizes the strip so
  the worker always holds the payload and can simply re-encode on resume;
- `MultiModalManager.execute_mm_encoder` grew a guard that turns any
  residual None payload into an actionable RuntimeError (or a clean skip
  when the encoder output is already cached) instead of vLLM's opaque
  NoneType death.
"""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from vllm.multimodal.inputs import (MultiModalBatchedField,
                                    MultiModalFeatureSpec,
                                    MultiModalFieldElem, MultiModalKwargsItem,
                                    PlaceholderRange)

import vllm.multimodal.utils as mm_utils
import vllm.v1.core.sched.output as sched_output
from tpu_inference.core.sched.utils import (
    _keep_mm_data, patch_vllm_mm_data_strip_for_preemption)
from tpu_inference.runner.multimodal_manager import MultiModalManager

REQ_ID = "chatcmpl-test-mm-preempt"
MM_HASH = "062d919f3fc8c1cc7cc21cbf49a6d894036372098a34ce496e868f19367c41c2"
# Live-crash geometry: ~3.5k-token shared system prefix, then the image.
IMG_OFFSET = 3520
IMG_LENGTH = 1024


@pytest.fixture(autouse=True)
def _restore_strip_binding():
    """The production patch rebinds vLLM module globals; keep each test
    hermetic so unrelated tests in the session see pristine vLLM."""
    saved = {
        mod: mod.strip_covered_mm_data
        for mod in (mm_utils, sched_output)
        if hasattr(mod, "strip_covered_mm_data")
    }
    try:
        yield
    finally:
        for mod, fn in saved.items():
            mod.strip_covered_mm_data = fn


def _image_feature() -> MultiModalFeatureSpec:
    elem = MultiModalFieldElem(
        data=torch.zeros(2, 3),
        field=MultiModalBatchedField(),
    )
    return MultiModalFeatureSpec(
        data=MultiModalKwargsItem({"pixel_values": elem}),
        modality="image",
        identifier=MM_HASH,
        mm_position=PlaceholderRange(offset=IMG_OFFSET, length=IMG_LENGTH),
    )


def _request(num_computed_tokens: int) -> SimpleNamespace:
    """Scheduler-side Request stand-in with the attributes
    NewRequestData.from_request reads."""
    return SimpleNamespace(
        request_id=REQ_ID,
        prompt_token_ids=list(range(IMG_OFFSET + IMG_LENGTH + 64)),
        mm_features=[_image_feature()],
        sampling_params=None,
        pooling_params=None,
        num_computed_tokens=num_computed_tokens,
        lora_request=None,
        prompt_embeds=None,
        prompt_is_token_ids=None,
    )


def _admit(num_computed_tokens: int):
    """First scheduling: the one and only time the worker receives
    mm_features (vllm/v1/core/sched/output.py NewRequestData.from_request)."""
    return sched_output.NewRequestData.from_request(
        _request(num_computed_tokens),
        block_ids=([1, 2, 3],),
    )


def _worker_after_admission(new_req_data):
    """Worker-side state after update_states consumed the NewRequestData —
    mirrors tpu_inference/runner/persistent_batch_manager.py (the
    CachedRequestState keeps new_req_data.mm_features verbatim; resume via
    CachedRequestData never refreshes it)."""

    def fake_embed(state_leaves, *, modality, **kwargs):
        del state_leaves, modality
        num_items = next(iter(kwargs.values())).shape[0]
        fake_embed.calls += 1
        return [np.zeros((IMG_LENGTH, 8)) for _ in range(num_items)]

    fake_embed.calls = 0
    runner = SimpleNamespace(
        requests={
            REQ_ID: SimpleNamespace(mm_features=new_req_data.mm_features)
        },
        encoder_cache={},
        state_leaves=None,
        embed_multimodal_fn=fake_embed,
    )
    return runner, MultiModalManager(runner)


def _resume_step():
    """Scheduler output for the resume step: the request came back through
    the waiting queue after preemption (encoder cache freed, covering
    blocks evicted) and _try_schedule_encoder_inputs re-scheduled input 0."""
    return SimpleNamespace(scheduled_encoder_inputs={REQ_ID: [0]})


COVERING = IMG_OFFSET + IMG_LENGTH + 32  # prefix-cache hit past the image
NOT_COVERING = IMG_OFFSET  # hit ends at the shared system prefix


def test_upstream_strip_nulls_payload_at_covered_admission():
    """Documents the vendored-vLLM half of the trap: admission with the
    image span prefix-cache-covered strips the payload from the wire copy
    (the scheduler-side Request keeps it — the worker is the one starved).
    If this fails after a vLLM bump, re-verify the whole preemption path:
    the neutralization patch may be obsolete (or the strip renamed)."""
    stripped = _admit(num_computed_tokens=COVERING)
    assert stripped.mm_features[0].data is None

    kept = _admit(num_computed_tokens=NOT_COVERING)
    assert kept.mm_features[0].data is not None


def test_upstream_none_payload_signature_in_group_and_batch():
    """Pins the exact production death: a None item reaching vLLM's
    group_and_batch_mm_kwargs raises AttributeError on .items()/.keys()
    (the two live tracebacks: single item -> _batch_mm_items .items();
    two items -> _can_batch_mm_items .keys())."""
    with pytest.raises(AttributeError, match="items"):
        list(mm_utils.group_and_batch_mm_kwargs([("image", None)]))
    with pytest.raises(AttributeError, match="keys"):
        list(
            mm_utils.group_and_batch_mm_kwargs([("image", None),
                                                ("image", None)]))


def test_preempt_resume_without_patch_is_caught_by_the_guard():
    """The full sequence on UNPATCHED vLLM: covered admission -> worker
    holds a stripped payload -> resume re-schedules the encoder. Without
    the fork's guard this is the engine-killing NoneType (previous test);
    with it, the worker raises an actionable error naming the mechanism."""
    runner, manager = _worker_after_admission(
        _admit(num_computed_tokens=COVERING))
    with pytest.raises(RuntimeError, match="resumed"):
        manager.execute_mm_encoder(_resume_step())
    assert runner.embed_multimodal_fn.calls == 0


def test_patched_strip_keeps_payload_and_resume_reencodes():
    """With the production patch active (as TpuPlatform.check_and_update_config
    applies it), the covered admission keeps the payload, and the resume
    step simply re-runs the encoder and repopulates the cache."""
    assert patch_vllm_mm_data_strip_for_preemption() is True
    assert sched_output.strip_covered_mm_data is _keep_mm_data

    new_req_data = _admit(num_computed_tokens=COVERING)
    assert new_req_data.mm_features[0].data is not None

    runner, manager = _worker_after_admission(new_req_data)
    manager.execute_mm_encoder(_resume_step())

    assert runner.embed_multimodal_fn.calls == 1
    assert MM_HASH in runner.encoder_cache
    assert runner.encoder_cache[MM_HASH].shape == (IMG_LENGTH, 8)


def test_guard_skips_reencode_when_output_already_cached():
    """A residual None payload with the encoder output already cached is
    not an error — nothing needs encoding."""
    runner, manager = _worker_after_admission(
        _admit(num_computed_tokens=COVERING))
    assert runner.requests[REQ_ID].mm_features[0].data is None
    cached = np.ones((IMG_LENGTH, 8))
    runner.encoder_cache[MM_HASH] = cached

    manager.execute_mm_encoder(_resume_step())

    assert runner.embed_multimodal_fn.calls == 0
    assert runner.encoder_cache[MM_HASH] is cached
