# The CPU gate: what each production change is held up by

`.github/workflows/pieces-cpu-gate.yml` runs 37 test files on
`jax[cpu] + flax + numpy + pytest + requests` — no torch, no vllm, no
transformers. It is this fork's only automated evidence, so the question that
matters about it is not "does it pass" but **"which production reverts does it
turn red?"**

This file answers that per change, from a mutation campaign run on
2026-09-03 against `pieces/main` @ `0bb4bda3` (PRs #38–#62). Method: revert or
corrupt the covered lines in a scratch copy of the tree, run the whole gate,
record which tests fail, restore from a sha256-verified backup. 56 mutations.

Baseline after this PR: **316 passed, 7 skipped, 1 xfailed**.

A row here is a claim that can be re-checked in about a minute. If you change
one of these lines and the named test does not go red, the row is wrong and
the test needs fixing — that is the whole point of writing them down.

---

## The map

| # | Production change | Mutation applied | Goes red |
|---|---|---|---|
| **#38** | `envs.TPU_ONLINE_QUANT_ACT` reaches `sharded_quantized_matmul` | drop the guard | `test_w8a16_switch.py::test_env_actually_reaches_the_dense_sharded_matmul`, `::test_env_flips_the_sharded_default_without_touching_explicit_callers` |
| **#38** | …and the batched wrapper | drop `and envs.TPU_ONLINE_QUANT_ACT` | `test_w8a16_switch.py::test_env_flips_the_sharded_default_without_touching_explicit_callers` |
| **#38** | the switch is *honoured*, not just present | keep the guard text, hardcode `maybe_quantize_x = True` | `test_w8a16_switch.py::test_env_actually_reaches_the_dense_sharded_matmul` |
| **#38** | `TPU_ONLINE_QUANT_ACT` defaults to W8A8 | `default=True` → `default=False` | `test_w8a16_switch.py::test_the_env_defaults_to_w8a8_in_the_shipped_envs_module` |
| **#38** | `get_max_min` returns f32 for int targets | restore the bare Python int | `test_int8_quant_scale_precision.py::test_int8_scale_is_float32_for_bf16_activations`, `::test_int8_codes_track_the_f32_reference`, `::test_kernel_and_xla_paths_quantize_int8_alike` |
| **#38 / #43** | explicit `w_dense` widen (dense and batched) | remove the widen | **NONE** — see "Not covered, and why" |
| **#40** | audio projection reads `.weight`, not `.kernel` | restore `.kernel` | `test_no_kernel_attribute_reads.py::test_no_module_reads_dot_kernel_off_a_wrapper` |
| **#41 / #47** | audio hatch reads the vLLM `limit_per_prompt` accessor | restore `limits.get("audio", 1) != 0` | `test_audio_hatch_requires_api_limit.py::test_real_vllm_shape_count_zero_is_accepted` |
| **#45** | `quantize_array` rounds for integer targets | drop `jnp.round` | `test_int8_quant_scale_precision.py::test_quantize_array_int8_rounds_and_is_unbiased`, `::test_kernel_and_xla_paths_quantize_int8_alike` |
| **#45** | …and computes the scale in f32 | restore the bf16 scale | same two, plus `::test_quantize_array_fp8_scale_is_f32_and_residual_is_the_bf16_multiply` |
| **#45** | the exact pre-fix body (bf16 scale **and** truncation) | full revert | the three above |
| **#46** | audio mask fallback is `shape[:-1]` | restore `shape[:2]` | `test_audio_mask_fallback_rank.py::test_fallback_mask_uses_all_but_the_feature_dim`, `::test_the_rule_gives_the_gatherable_shape[shape1]` |
| **#48** | `take_draft_token_ids` uses proposal-time owners | restore `self.runner.input_batch.req_ids` | `test_draft_token_owners_survive_condense.py::test_take_returns_the_owners_the_drafts_were_proposed_for`, `::test_condense_between_propose_and_take_makes_the_recovery_token_illegal` |
| **#48** | `propose_draft_token_ids` snapshots the owners | drop the snapshot | the two above, plus `::test_take_without_a_batch_change_is_the_live_order` |
| **#48** | the reorder buffer is sized by owners | restore `len(draft_token_ids)` | `test_draft_token_owners_survive_condense.py::test_the_returned_rows_line_up_one_to_one_with_the_owners` |
| **#49** | `gmm_wrapper` forwards `maybe_quantize_lhs` | neuter the env consult | 6 in `test_moe_w8a16_switch.py`, incl. `::test_act0_expert_activations_reach_the_kernel_in_bf16` |
| **#49** | …and actually passes the kwarg | drop `maybe_quantize_lhs=` from the call | the same 6, incl. `::test_act1_default_quantizes_expert_activations` |
| **#49** | fused-RS kernel refuses W8A16 | drop the `NotImplementedError` | `test_moe_w8a16_switch.py::test_fused_rs_kernel_refuses_w8a16_before_importing_the_kernel`, `::test_env_reaches_the_moe_kernel_call_site` |
| **#51** | `create_weights_jax` parks a host-quant request | drop `request_host_quant` | `test_online_host_quant_wiring.py::test_create_weights_requests_host_quant_with_the_apply_paths_layout` |
| **#51** | `process_weights_after_loading` adopts the parked scale | `if w_s is not None:` → `if False:` | `test_online_host_quant_wiring.py::test_process_weights_adopts_the_parked_scale_before_any_device_requant` |
| **#51** | `assign_and_shard_param` calls the host hook | drop `place_host_quantized` | all 5 in `test_online_quant_load_peak.py` |
| **#54** | `JITTABLE_MODULE_POSITIONAL_OFFSET` | `2` → `3` | 4 in `test_mm_jit_static_args.py`, incl. `::test_static_index_tracks_the_position_in_the_signature` |
| **#55** | the mm-debug line carries every stat | drop `std` from the field list | 2 in `test_mm_debug_stats.py` + 3 in `test_mm_debug_patch.py` |
| **#55** | `PIECES_MM_DEBUG` defaults off | `default=False` → `True` | 2 in `test_pieces_mm_debug_env.py` |
| **#56** | the pooler restricts stats to valid rows | invert `_valid_rows` | `test_mm_debug_patch.py::test_one_line_per_process_image_input_with_every_key` |
| **#57** | `clamp_activation` clips | make it the identity | 4 in `test_gemma4_vision_clipping.py`, incl. `::test_the_checkpoints_own_bounds_reproduce_the_reference_and_bite` |
| **#57** | …reads the bound in the activation dtype | drop the dtype cast | `::test_clamp_activation_reads_the_bound_in_the_activation_dtype` |
| **#57** | `neutral_clamps` is -/+inf | return zeros | `::test_neutral_clamps_are_a_no_op` |
| **#57** | `unloaded_clamps` reports the gaps | always return `[]` | `::test_unloaded_clamps_names_exactly_the_unfilled_clamps`, `::test_verify_vision_clamps_loaded_refuses_a_partial_load` |
| **#57** | the load verifier refuses a partial load | `if missing:` → `if False:` | `::test_verify_vision_clamps_loaded_refuses_a_partial_load` |
| **#57** | `Gemma4VisionClippedEinsum.__call__` applies both clamps | 5 separate reverts (see below) | `::test_the_clipped_einsum_falls_back_to_the_plain_projection` |
| **#58** | fp32 score operands | drop `.float()` on both | `test_gemma4_vision_attention.py::test_both_score_operands_are_cast_to_fp32_before_the_matmul` |
| **#58** | the chunked mask is sliced only when it has real query rows | `!= 1` → `== 1` | `::test_the_chunked_mask_is_sliced_only_when_it_has_real_query_rows` |
| **#60** | the PLE guard does not require `input_ids` | restore the `and input_ids is not None` conjunction | `test_gemma4_ple_image_prefill.py::test_the_guard_does_not_require_input_ids` |
| **#60** | `mm_bidi_ranges` emits a half-open range | `last + 1` → `last` | 8 in `test_mm_bidi_span_pipeline.py` |
| **#60** | multi-block requests fall back to causal | take the first span instead | 4 in `test_mm_bidi_span_pipeline.py` |
| **#60** | audio features are skipped | stop skipping them | `::test_audio_features_are_not_blockwise_bidirectional` |
| **#61** | the embeds primer builds an ids operand | restore `input_ids=None` | `test_backbone_primer_padding_grid.py::test_the_embeds_primer_keeps_the_ids_operand_the_text_primer_uses` (+2) |
| **#61** | the flax PLE fallback is only for a caller with no ids | `if input_ids is None:` → `if True:` | `test_gemma4_ple_image_prefill.py::test_the_flax_fallback_this_mirrors_still_exists`, `test_mm_step_model_fn_operands.py::test_the_non_ple_variants_have_no_id_track_to_get_wrong` |
| **#62** | the mm-embeds primer walks the whole request-padding ladder | return `paddings[-1:]` unconditionally | 11 in `test_backbone_primer_padding_grid.py` |
| **#62** | `MM_EMBEDS_PRIMER_ALL_REQ_PADDINGS` defaults on | `default=True` → `False` | `::test_the_env_defaults_to_on_in_the_shipped_envs_module` |

The five separate `#57` `__call__` reverts, each caught by
`test_the_clipped_einsum_falls_back_to_the_plain_projection`: `if True: return`
(clipping fully off), `or True` in the guard, clamping the output with the
*input* bounds, deleting the input clamp, and `if True:` in `__init__`
(clamp params created when clipping is off). All five passed the gate before
this PR.

### Anti-vacuity mutations

Three mutations do not revert a feature; they check the gate cannot be fooled.

| Mutation | Goes red |
|---|---|
| a comment carrying the phrase an assertion greps for, over broken code (`leading = x.shape[:-2]`) | both params of `test_rank_generic_sharding.py::test_apply_jax_restores_one_axis_per_contracting_axis` |
| an unstubbed `import vllm.x` in `linear.py` — the leaf under test | 15 tests across 3 files (it used to **skip**, exit 0, gate green) |
| the switch text preserved while the semantics are inverted | `test_w8a16_switch.py::test_env_actually_reaches_the_dense_sharded_matmul` |

---

## Not covered, and why

**`#38` / `#43`, the explicit `w_dense = w_q.astype(x.dtype)` widen.** Removing
it from either the dense or the batched path leaves all 316 tests green. This
is not a hole in the tests — it is a change with no observable behaviour.
`lax.dot_general` accepts the mixed `bf16 x int8` pair and promotes to the
preferred `f32`, so the widen makes the convert an explicit op rather than an
implicit promotion. `test_w8a16_switch.py::test_both_w8a16_implementations_agree`
says this in its own docstring and pins the property that *is* load-bearing:
the dense and batched paths agree. **Left alone deliberately** — a test that
fails on this revert would have to assert the jaxpr's op list, which pins a
JAX implementation detail rather than a fork behaviour.

## TPU-only by nature

These are real and unverified on the gate, by construction rather than by
neglect:

- **Kernel behaviour.** Every numeric test here is dense XLA on CPU. The
  Pallas quantized-matmul kernel, the RPA v3 attention kernel and the fused
  reduce-scatter MoE kernel are never executed. `quantize_array`'s VMEM
  behaviour in particular is asserted only through its CPU arithmetic.
- **The compile cache and the primers.** `test_backbone_primer_padding_grid.py`
  runs the shipped primer loops against fakes and records `(num_tokens,
  num_reqs, ...)` tuples. It proves the *grid*, never that XLA compiles those
  programs or that the persistent cache is hit.
- **The load peak.** `test_online_quant_load_peak.py` is a real measurement,
  but of `jax.live_arrays()` on 3 forced CPU devices. HBM on a v6e is not what
  it measured.
- **`transformers` parity.** `test_gemma4_ple_reference_parity.py` (9 tests)
  and 2 tests in `test_gemma4_vision_clipping.py` `importorskip("torch")` and
  never run on the gate. The vision-clipping differential on the gate compares
  this repo's NumPy transcription against its own JAX transcription; the
  transformers tower is the reference only in the skipped half. A shared
  mis-transcription agrees with itself.
- **torchax / vLLM module behaviour.** `models/vllm/experimental/*` cannot be
  imported without torch, so those files are checked by AST and by executing
  pure-jax leaves. Where an assertion is structural, it is written to pin the
  expression, not a substring — see the anti-vacuity table above.

## The stale-`.pyc` hazard

The hazard PR #60 hit is real and the usual flags do **not** fix it.

Measured: corrupt a source file with a **same-length** edit and restore its
mtime, so `(mtime, size)` still matches the cached bytecode. The gate stays
green — the corrupted `clamp_activation` was not picked up. Adding
`python -B`, `--import-mode=importlib` and `-p no:cacheprovider` **does not
help**: `-B` only disables *writing*, `--import-mode=importlib` still uses the
cached-bytecode loader, and `no:cacheprovider` is pytest's own cache. Only
removing `__pycache__` catches it (4 tests then fail).

CI is safe because `actions/checkout@v4` starts from a tree with no
`__pycache__`. Locally it is not. Two files defend themselves properly, by
compiling the source directly instead of using `spec_from_file_location`:

```python
module = types.ModuleType(name)
exec(compile(path.read_text(), str(path), "exec"), module.__dict__)
```

`test_mm_bidi_span_pipeline.py` and `test_gemma4_ple_reference_parity.py`
already did this; `test_mm_debug_patch.py` was switched to it in this PR
(it is the largest file in the gate and the one whose negative controls matter
most). **Before trusting any local negative control on this suite, delete
`__pycache__` first.**
