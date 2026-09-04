# Gemma-4 on TPU v6e: bench findings and open items (2026-09-04)

Source of every number: `enrichment_evals_and_optimizations/testing_ledger.md` and the
`isolation/runs/<stamp>-<lane>/` directories named below (RUN.json, probe outputs, pod logs).
All rates are ok-requests per hour on 69 image+text JSON prompts (median 2,947 prompt tokens),
temperature 0, at C=8 / C=16, 16 seats, 16k context unless stated. "native" = flax_nnx,
"torchax" = MODEL_IMPL_TYPE=vllm. Issues are disabled on this fork, so these live here.

## Verified (no action)

| model | chips | best config measured | req/hr C=8 / C=16 | single-stream |
|---|---|---|---|---|
| 12B | 1 | native W8A16 + MTP | 1868 / 2452 | — |
| 12B | 4 | native W8A16 + MTP | 1875 / 2502 | 413 tok/s |
| 26B-A4B | 4 | native W8A16 dense + MoE int8, no MTP | 1851 / 2588 | 205 tok/s |
| 26B-A4B | 4 | same + MTP (latency switch) | 1536 / 2080 | 457 tok/s |
| E4B | 1 | native W8A8 + MTP | 2607 / 3499 | 406 tok/s |
| E2B | 1 | native W8A16, no MTP | 3380 / 5296 | 288 tok/s |
| E2B | 1 | native W8A16 + MTP (latency) | 2433 / 3234 | 512 tok/s |
| 31B | 4 | native int8 W8A8 | 1377 / 1794 | 124 tok/s |

Native vs torchax, image-matched, same pin: 12B 1.06x / 1.92x (1 chip), 1.05x / 1.09x (4, bf16),
tie (4, W8A16), 1.22x / 1.03x (8); E4B 1.24x / 1.45x; E2B 1.62x / 1.82x. Native never loses and is
the only path with MTP on the 12B. MTP + JSON: zero grammar rejections on every variant at 1/4/8 chips.

## Open items (model / engine), with evidence

1. **Online int8 activation dtype should be a per-model default.** The 12B is fastest with
   W8A16 (bf16 activations) at every chip count and W8A8 loses to bf16 on eight chips
   (1148 / 1932 vs 1469 / 2166; `20260904T154952Z-eval-12b-tp8-native-int8`); the E4B is
   8-11 % faster with W8A8 (`20260904T042733Z-eval-e4b-int8` vs `20260904T054235Z-eval-e4b-int8-w8a16`);
   the 26B needs W8A16 + MoE int8. Today the choice is a lane env (`TPU_ONLINE_QUANT_ACT`);
   a model-family default (or a startup pick from the projection shapes) would remove a foot-gun.
2. **torchax int8 (W8A8) 12B produces degenerate answers** where the native path does not:
   3 % at TP=8 (`20260904T161636Z-eval-12b-tp8-q-int8`, native twin 0 %). Worth a differential
   on the torchax online-quant path's output distribution before it is used for anything.
3. **E2B MTP drafter acceptance is 0.41 at k=4**, which costs 23-27 % of batch throughput
   against no-MTP (2234 / 3063 vs 3040 / 3968; `20260904T045624Z-eval-e2b-mtp`); k=2 accepts
   0.57 and halves the loss (`20260904T064303Z-eval-e2b-mtp-k2`). Either default the E2B to
   k=2 or retrain/replace the drafter; on the E4B the same drafter family accepts 0.63.
4. **26B + MTP on one chip fits only at <= 6144 context.** The drafter leaves 0.72 GiB of KV
   (1.53 GiB without it); one 8k sequence needs 0.86 GiB (`20260904T095322Z-eval-26b-1chip-mtp`,
   pod-previous.log). Document the lane default, or shrink the drafter's footprint.
5. **Under KV preemption MTP killed EngineCore** (`tpu_runner.py:2660`,
   `_prepare_async_token_substitution_indices`) — fixed in #64 and proven live
   (`20260904T123423Z-eval-26b-1chip-mtp`: 69/69 through 182 preemptions). Keep a
   preemption-heavy lane in the gate so this path stays exercised.
6. **MTP and int8 do not stack at batch on the MoE 26B (TP=4) or the E2B**; they do on the
   12B and E4B. On the 26B the verify step re-runs the full MoE for k+1 tokens per request;
   a MoE-aware verify (shared routing across the draft window) is the obvious lever.
7. **31B int8 W8A8 at TP=4 read 9 % / 19 % below the 09-02 pin** (1377 / 1794 vs 1521 / 2202;
   `20260904T102424Z-eval-31b-native-int8-tp4`). One run; a same-pin repeat is queued (Y11).
   If it reproduces, bisect between pins e2cb0302 and f7cb94c4 (#51 changed the quant path).
8. **Quantized 26B on one chip: C=16 throughput tracks how many JSON answers fail to stop**
   (capped at 4096 tokens): fp8 e4m3fn 6-9 capped, e4m3b11 8-13, int8 12-14, per 69, with
   0-1 % degenerate. A stop-token/quality check on quantized experts would settle whether
   this is quantization noise or a real stopping regression.
9. **Chip count is flat for batch throughput at 16 seats**: 12B one chip 1396 / 1975 (bf16
   E4B-class budget) ~ four chips 1399 / 2015 ~ eight chips 1469 / 2166; 26B four chips
   1375 / 2160 > eight chips 1215 / 2070. More chips buy single-stream latency (with MTP)
   and context headroom only. Serving should scale by replicas, not by TP, for these models.

## Harness lessons that affect anyone benchmarking this fork

- A cluster operation on the GKE cluster (node-pool create/delete) can close a live
  `kubectl exec` stream with exit 0 and time out API calls from other sessions; a benchmark
  harness must read its own terminal line, not the exit code.
- Under GCE_STOCKOUT a hunting node pool places only if left alone: one `--no-wait` create
  and polling placed 4-chip hosts in 5-187 min and an 8-chip host in 322 min; seven
  cancel-and-delete hunts placed nothing.
