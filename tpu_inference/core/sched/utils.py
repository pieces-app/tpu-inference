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

DEFAULT_MAX_DECODE_STEPS = 10


def patch_vllm_scheduler_for_continue_decode():
    """Monkeypatches vLLM's Scheduler and AsyncScheduler for Continue Decode.

    In Continue Decode, the host schedules 1 step while the TPU runner executes
    up to max_decode_steps (N) decode iterations on-device in a single step.

    This function applies three patches:
    1. patched_init: Forces KVCacheManager to reserve enough KV cache blocks for
       all N tokens during schedule() (num_lookahead_tokens = max_decode_steps - 1).
    2. patched_update_base: Reconciles request.num_computed_tokens on host by
       adding the extra (N - 1) tokens generated on-device once model output returns.
    3. patched_async_update_request_with_output: Pre-compensates in-flight
       num_output_placeholders in AsyncScheduler by adding (N - 1) before
       subtracting N, preventing placeholder underflow in async mode.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    # Avoid patching multiple times
    if not getattr(Scheduler, "_continue_decode_patched", False):
        original_update_base = Scheduler._update_request_with_output

        def patched_update_base(scheduler_self,
                                request,
                                new_token_ids,
                                is_stale=False,
                                **kwargs):
            # Original update appends new_token_ids to request output and trims on stop token.
            res_token_ids, stopped = original_update_base(scheduler_self,
                                                          request,
                                                          new_token_ids,
                                                          is_stale=is_stale,
                                                          **kwargs)

            # schedule() only incremented num_computed_tokens by 1. Advance by the remaining
            # (N - 1) tokens generated on-device so host-side num_computed_tokens is accurate.
            # A stale delivery predates the preemption rollback of
            # num_computed_tokens and must not advance it (mirrors the
            # `if not output_is_stale` guards in vLLM's update_from_output).
            # AsyncScheduler's original calls super() WITHOUT forwarding
            # is_stale, so the async wrapper threads staleness through
            # _cd_stale_in_flight instead — check both.
            stale = is_stale or getattr(scheduler_self, "_cd_stale_in_flight",
                                        False)
            diff = len(res_token_ids) - 1
            if diff > 0 and not stale:
                request.num_computed_tokens += diff

            return res_token_ids, stopped

        Scheduler._update_request_with_output = patched_update_base

        original_init = Scheduler.__init__

        def patched_init(scheduler_self, vllm_config, *args, **kwargs):
            original_init(scheduler_self, vllm_config, *args, **kwargs)

            additional_config = getattr(vllm_config, "additional_config", {})
            max_decode_steps = additional_config.get("max_decode_steps",
                                                     DEFAULT_MAX_DECODE_STEPS)
            # Reserve max_decode_steps - 1 lookahead tokens so KVCacheManager allocates
            # sufficient blocks for up to max_decode_steps tokens before execution on TPU.
            scheduler_self.num_lookahead_tokens = max(
                scheduler_self.num_lookahead_tokens, max_decode_steps - 1)

        Scheduler.__init__ = patched_init

    from vllm.v1.core.sched.async_scheduler import AsyncScheduler

    if not getattr(AsyncScheduler, "_continue_decode_patched", False):
        original_async_update_req = AsyncScheduler._update_request_with_output

        def patched_async_update_request_with_output(scheduler_self,
                                                     request,
                                                     new_token_ids,
                                                     is_stale=False,
                                                     **kwargs):
            if len(new_token_ids) > 1 and not is_stale:
                # In AsyncScheduler, _update_after_schedule() added 1 in-flight placeholder token.
                # When N tokens return, original_async_update_req will subtract N from
                # num_output_placeholders. Pre-compensate by adding (N - 1) first so that
                # num_output_placeholders cleanly decrements by 1 without underflowing < 0.
                # Placeholders are zeroed at preemption, so a stale delivery must
                # not be pre-compensated (the original skips its decrement too).
                request.num_output_placeholders += (len(new_token_ids) - 1)
            # The original's super() call does not forward is_stale, so the
            # patched base can't see it as a parameter. Thread it through an
            # instance flag for the duration of this call.
            scheduler_self._cd_stale_in_flight = is_stale
            try:
                return original_async_update_req(scheduler_self,
                                                 request,
                                                 new_token_ids,
                                                 is_stale=is_stale,
                                                 **kwargs)
            finally:
                scheduler_self._cd_stale_in_flight = False

        AsyncScheduler._update_request_with_output = patched_async_update_request_with_output
        AsyncScheduler._continue_decode_patched = True

    Scheduler._continue_decode_patched = True


def _keep_mm_data(mm_features, num_computed_tokens, uses_mrope=False):
    """Preemption-safe replacement for vLLM's ``strip_covered_mm_data``:
    keep every multimodal payload. Signature-compatible with the original
    (vllm/multimodal/utils.py) so call sites need no change."""
    return mm_features


def patch_vllm_mm_data_strip_for_preemption() -> bool:
    """Neutralize vLLM's prefix-cache mm-payload strip; it kills the engine
    on resume of a KV-preempted multimodal request.

    vLLM #52041 ("Skip broadcasting mm tensor data to workers for
    prefix-cache-covered items") makes ``NewRequestData.from_request``
    (vllm/v1/core/sched/output.py) null out ``mm_feature.data`` for any
    multimodal item whose placeholder span is fully covered by
    ``num_computed_tokens`` at FIRST scheduling — on the assumption that "no
    encoder run can be scheduled for them". That assumption breaks under
    preemption:

      1. Request admitted with a prefix-cache hit covering its image span
         (duplicate image: an identical earlier request cached those
         blocks). The worker's CachedRequestState stores the STRIPPED
         mm_features (data=None) for the request's whole lifetime — resume
         goes through CachedRequestData, which carries no mm payloads.
      2. KV pressure preempts the request: blocks freed,
         num_computed_tokens=0, per-request encoder cache freed
         (Scheduler._preempt_request).
      3. On resume the covering blocks have been evicted, so the scheduler
         legitimately re-schedules the encoder input
         (_try_schedule_encoder_inputs) — but the worker no longer has the
         pixels: execute_mm_encoder hits ``None.keys()``/``None.items()``
         inside vllm.multimodal.utils and the EngineCore dies.

    Observed live 2026-08-27 on v6e-1 gemma-4-12B (p9 image, vLLM d626108b)
    at 93-97% KV usage: C=12 with duplicate images and C=16 with 32 jobs
    over 23 unique images both killed the engine this way.

    The strip is purely a serialization/memory optimization: on the TPU
    single-host deployment the executor is uniproc (objects pass by
    reference), so keeping the payload costs nothing; on multihost it
    restores the pre-#52041 broadcast, trading a little bandwidth for not
    dying. Rebinds the name in BOTH modules: the sched.output binding is the
    one ``from_request`` resolves at call time; the multimodal.utils
    binding covers any module imported later.

    Idempotent. Returns True if the patch is active, False if the function
    no longer exists in vLLM (upstream rename/removal — log and re-verify
    the preemption path against that vLLM before trusting C>12 multimodal).
    """
    import vllm.multimodal.utils as mm_utils
    import vllm.v1.core.sched.output as sched_output

    patched = False
    for mod in (sched_output, mm_utils):
        if hasattr(mod, "strip_covered_mm_data"):
            mod.strip_covered_mm_data = _keep_mm_data
            patched = True
    return patched
