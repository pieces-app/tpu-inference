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
   the 26B needs W8A16 + MoE int8. Direct eight-chip measurement (added 18:27Z):
   12B W8A16 at TP=8 is 1349 / 2363 against W8A8's 1148 / 1932 — **+18 % / +22 %** —
   and beats eight-chip bf16 by 9 % at C=16 (`20260904T172328Z-eval-12b-tp8-native-int8-w8a16`).
   Today the choice is a lane env (`TPU_ONLINE_QUANT_ACT`); a model-family default
   (or a startup pick from the projection shapes) would remove a foot-gun.
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
7. **CONFIRMED REGRESSION: 31B int8 W8A8 at TP=4 lost ~20 % of its C=16 throughput after
   pin `e2cb0302`.** Three samples, two pins: C=16 2202 (09-02, `e2cb0302`) → 1794 (pr57
   `f7cb94c4`, `20260904T102424Z`) → 1758 (pr64 `9142cdc6`, `20260904T174145Z`), with capped
   counts flat (0 / 2 / 2 of 69) and 0 % degenerate throughout; C=8 over the same three is
   1521 / 1377 / 1470, i.e. noise. A C=16-only, ~20 % loss that reproduces across two later
   pins wants a bisect between `e2cb0302` and `f7cb94c4`; #51 (host-side online quantization)
   rewrote the path this lane uses and is the prime suspect. The 31B has no W8A16 lane, so
   whether the winning dtype elsewhere would recover it is untested.
8. **Quantized 26B on one chip: C=16 throughput tracks how many JSON answers fail to stop**
   (capped at 4096 tokens): fp8 e4m3fn 6-9 capped, e4m3b11 8-13, int8 12-14, per 69, with
   0-1 % degenerate. A stop-token/quality check on quantized experts would settle whether
   this is quantization noise or a real stopping regression.
9. **Chip count is flat for batch throughput at 16 seats — until the model stops fitting
   comfortably.** 12B: one chip 1396 / 1975 ~ four 1399 / 2015 ~ eight 1469 / 2166; 26B-A4B:
   four 1375 / 2160 > eight 1215 / 2070. But the **31B dense gains from eight chips: 1347 / 1885
   vs 1067 / 1607 at TP=4, +26 % / +17 %** (`20260904T183043Z-eval-31b-tp8-native-bf16`), and at
   TP=8 its int8 is a dead heat with bf16 (1339 / 1875 vs 1347 / 1885), where at TP=4 int8 led by
   29 % at C=8. Reading: ~62 GB of bf16 weights leave the 31B bandwidth-bound on four chips, so
   the eighth chip pays for its collectives and simultaneously removes the reason to quantize.
   Guidance is therefore per-model, not global: scale by replicas for the 12B / 26B / E4B / E2B,
   and give the 31B eight chips in bf16.
10. **MTP is a loss on eight chips for both models measured there.** 12B production config
   (W8A16 + MTP): TP=4 1878 / 2532 at 394 tok/s vs TP=8 1236 / 1752 at 195 tok/s — −34 % / −31 %
   at batch AND half the single-stream speed (`20260904T174929Z-eval-12b-tp8-native-mtp-int8-w8a16`);
   26B: TP=4 1394 / 1941 at 483 tok/s vs TP=8 1052 / 1565 at 242 tok/s. The verify step's
   collectives scale with TP while the draft window does not, so speculative decoding should
   be gated off (or k reduced) above TP=4 unless a measurement says otherwise.
12. **Quantization stops paying at TP=8, and on the MoE it turns sharply negative.** 26B
   W8A16 dense + int8 experts: TP=4 1851 / 2588 vs TP=8 1116 / 2029 — **−40 % / −22 %**, the
   largest chip-count penalty measured, and at TP=8 it is 8 % BELOW plain eight-chip bf16
   (`20260904T193324Z-eval-26b-tp8-q-int8-w8a16`). The 31B shows the same shape more gently
   (int8 1339 / 1875 vs bf16 1347 / 1885 at TP=8, a dead heat, where int8 led by 29 % at
   TP=4). Expert-parallel collectives scale with TP while the per-expert work does not, so a
   quantized MoE spread over eight chips is nearly all overhead. Quantization should be
   chosen per (model, chip count), not per model.
11. **P0 on the 26B: ANSWERED, and the answer is parity — the MoE does serve under torchax.**
   `eval-26b-tp4-torchax` (a one-variable twin of `eval-26b-tp4-bf16`) ran clean on four chips:
   1453 / 2220 req/hr, 0 % degenerate, 151 tok/s (`20260904T200743Z-eval-26b-tp4-torchax`),
   against the native twin's 1375 / 2160 — torchax nominally +5.7 % / +2.8 %, which is inside
   this model's own pin-to-pin spread, so it is parity and a same-pin control is queued rather
   than a claimed reversal. Native still owns the 26B in production for two reasons that do not
   depend on this number: MTP runs only on native, and the best batch config (native W8A16 +
   MoE int8, 1851 / 2588) is 27 % above this cell. **The 31B on torchax remains unmeasured.**
   Worth noting for anyone tuning the fallback path: on the E4B the native margin WIDENS under
   quantization — 1.39× / 1.62× at W8A16 versus 1.24× / 1.45× in bf16
   (`20260904T195625Z-eval-e4b-torchax-int8-w8a16`) — so the paths diverge most where the model
   is least bandwidth-bound.

## Harness lessons that affect anyone benchmarking this fork

- A cluster operation on the GKE cluster (node-pool create/delete) can close a live
  `kubectl exec` stream with exit 0 and time out API calls from other sessions; a benchmark
  harness must read its own terminal line, not the exit code.
- Under GCE_STOCKOUT a hunting node pool places only if left alone: one `--no-wait` create
  and polling placed 4-chip hosts in 5-187 min and an 8-chip host in 322 min; seven
  cancel-and-delete hunts placed nothing.
