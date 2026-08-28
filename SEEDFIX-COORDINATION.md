# fix/sampling-seed-500 — live coordination note (untracked; delete before PR)

Two agents are editing this worktree tonight. To avoid thrash, division as of 21:12 EDT:

- KEPT from the other writer: module-level `TpuRequestValidationError` (+ fallback),
  `_SEED_UNSUPPORTED_MSG`, the ordering-caveat docstring, the strengthened
  `test_validate_request_random_seed` assertions (typed error, parameter, non-empty msg).
- THIS session (seedfix task, time-boxed) is adding, on top:
  1. `validate_sampling_params` (single gate; `validate_request` delegates to it),
  2. `_install_early_seed_gate` — wraps vLLM `InputProcessor._validate_params` so the
     rejection lands BEFORE mm preprocessing (P0/P1 cache-drift poisoning, coordinator
     evidence from the 12B bf16 bench),
  3. REAL per-request seed support in the JAX sampler (per-row key from (seed, draw idx)):
     sampling.py / sampling_metadata.py / input_batch.py / compilation_manager.py + tests.
     Seed then REJECTED only under spec decode (`_per_request_seed_unsupported_reason`).
     NOTE: this flips the default in `test_validate_request_random_seed` — seeded
     requests are ACCEPTED unless the class attr reason is set; test updated accordingly.
- Please don't revert the default-accept change without checking with the coordinator;
  everything else in tpu_platform.py is yours to shape.
