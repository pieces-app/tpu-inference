# Gemma-4 Unified (encoder-free) vision fidelity diagnostics

CPU harness that localized the TPU fine-text defect on
`google/gemma-4-12b-it` (`Gemma4UnifiedForConditionalGeneration`) — the
misread digits/IDs (PID `36793` -> `36732`, garbled paths/hashes) seen at a
proven-active 1120 soft-token budget while local llama.cpp reads the same
strings exactly.

## Architecture facts (why the old "SigLIP tower" framing was wrong)

The Unified variant has **no vision tower**. The processor resizes
(torchvision bicubic, `antialias=True`), rescales 1/255 (mean 0 / std 1),
patchifies to 16px patches, space-to-depth merges 3x3 into 48x48x3 = 6912-dim
"model patches" (max 1120), and the model projects them with a tiny embedder:
LN -> Dense(6912->3840) -> LN -> +factorized 2D posemb -> LN -> RMSNorm ->
Linear. All spatial integration happens in the LANGUAGE model, whose
sliding-attention layers attend **bidirectionally within each image block**
(`text_config.use_bidirectional_attention == "vision"`; HF mask:
`AND(sliding_window, OR(causal, blockwise))`). On TPU the model runs through
the vLLM `gemma4_unified` implementation under torchax (no JAX-native impl).

## Findings (gemma-4-12b-it, screenshot 3456x2160, max_soft_tokens=1120)

### Stage diff: embedder, f32 torch eager reference vs serving arms

| stage | torch eager bf16 (GPU analog) | torchax bf16 (TPU semantics) | torchax f32 (patched) |
|---|---|---|---|
| S0 pixels | rel 5.5e-5 | rel 5.5e-5 | 0 |
| S1 patch_ln1 | max 1.33 / rel 1.4e-3 | max 13.1 / rel 5.7e-2 | max 1.5e-3 / rel 5.6e-6 |
| S2 patch_dense | rel 3.2e-3 | rel 4.8e-2 | rel 5.0e-6 |
| S3 patch_ln2 | max 1.80 / rel 2.1e-3 | **max 438 / rel 3.6e-2, cos 0.984** | max 4.1e-2 / rel 6.6e-6 |
| S5 pos_norm | rel 4.1e-3 | rel 5.5e-2 | rel 9.5e-6 |
| S7 soft tokens | rel 3.1e-3, cos 1.0000 | rel 4.4e-2, cos 0.9944 | rel 6.9e-6, cos 1.0000 |

Per-token error at S3 is bimodal: p50 = 3.2 but **p95 = 438** — >=5% of the
image's soft tokens (near-flat patches: uniform background, faint fine
text; worst token mean 0.974 / std 0.0049) are effectively noise under
torchax bf16. Mechanism: torchax's `aten.native_layer_norm` computes
mean/var via `jnp.mean`/`jnp.var` in the INPUT dtype (bf16), unlike PyTorch
native eager which accumulates LayerNorm statistics in fp32; for patches
whose std ~ the bf16 quantization step, rstd is badly wrong and LayerNorm
amplifies by ~1/std. Fix: run the embedder in fp32
(`gemma4_unified_patcher.py`) — matches the f32 reference to ~1e-5.

### LM-level A/B (12B full model, CPU f32, greedy, real screenshot)

| arm | masks | soft tokens | PID read | path read |
|---|---|---|---|---|
| R | correct (bidi) | clean f32 | **36793** (exact) | exact |
| C | causal-only (TPU today) | clean f32 | **36732** (the TPU's exact misread) | garbled `data:/local/screen/...` (TPU's garble pattern) |
| N | correct (bidi) | torchax-bf16 noisy | **36793** (exact) | exact |
| NC | causal-only | noisy | 36732 + hallucinated all-zero hash | garbled |

**Verdict: the missing PrefixLM blockwise-bidirectional mask is the
dominant defect** — dropping it reproduces the TPU's character-exact
failure on CPU with everything else held fixed. tpu-inference's
`AttentionMetadata` had no `mm_prefix`/blockwise concept at all, so the
vLLM model's bidirectional ranges were silently discarded and image tokens
attended causally on all 48 layers. The torchax bf16 embedder corruption is
a real secondary fidelity bug (fixed by the fp32 patcher) but does not by
itself flip the fine-text reads.

Fixes in this branch:
1. RPA v3 kernel + AttentionMetadata + runner: optional per-request
   `mm_bidi_ranges` (PrefixLM blockwise) applied on sliding layers —
   `AND(sliding_window, OR(causal, blockwise))`, exactly HF's composition.
   Static-gated: non-bidi models compile byte-identical kernels.
2. `gemma4_unified_patcher.py`: fp32 vision embedder under torchax.

## Scripts

Set `GEMMA4_CKPT` (checkpoint dir with `model.safetensors`) and
`GEMMA4_IMAGE` (test PNG). Environments: torch + transformers>=5.10 +
torchax 0.0.13 + jax (CPU).

- `frontend_diff.py` — stage-by-stage embedder diff (eager f32 vs eager
  bf16 vs torchax bf16 vs torchax f32) + resize-method probes.
- `lm_mask_ab.py` — the 4-arm mask/noise A/B on the full 12B (needs
  ~64 GB RAM, ~15 min/arm on an M4 Max).
