# Patches to the vendored vLLM checkout (`/workspace/vllm`)

The `vllm/vllm-tpu` images build TWO editable installs (see
`docker/Dockerfile`):

- `/workspace/tpu_inference` — this repository;
- `/workspace/vllm` — a plain `git clone` of
  [vllm-project/vllm](https://github.com/vllm-project/vllm) checked out at
  the build-arg `VLLM_COMMIT_HASH`. The nightly image tag encodes both
  commits: `nightly-<date>-<tpu_inference_sha>-<vllm_sha>`.

Files under `/workspace/vllm` are therefore NOT in this repository's tree.
Fixes to them cannot land as normal tpu-inference commits — they live here
as (patched full file + unified diff) pairs, keyed by the vLLM commit they
were generated against, until they are upstreamed to vllm-project/vllm.

## d626108 — gemma4_mm.py: guard `torch.accelerator.get_memory_info()`

vLLM commit `d626108b1841888ec90aced33367149a6bbc7e4b` — the `/workspace/vllm`
pin of `vllm/vllm-tpu:nightly-20260822`…`nightly-20260824-b98d381-d626108`
(and of `gcr.io/global-cloud-runtime/vllm-tpu-patched:nightly-20260824-p1/p2`,
which are `FROM` that nightly and do not touch `/workspace/vllm`).

**Bug** (captured live 2026-08-24 on GKE TPU v6e, serving
`google/gemma-4-26B-A4B-it` bf16, `MODEL_IMPL_TYPE=vllm`): any request with
an image kills the engine —

```
File "/workspace/vllm/vllm/model_executor/models/gemma4_mm.py", line 1296, in _process_image_input
  free, total = torch.accelerator.get_memory_info()
RuntimeError: PyTorch is not linked with support for jax devices
```

→ `EngineDeadError`, HTTP 500, pod restart (~15–25 min XLA recompile).
(The line number varies with the vLLM commit: 1296 in the captured build,
1345 at `d626108`. A second identical call sits in `_process_video_input`.)

`Gemma4ForConditionalGeneration._process_image_input` /
`_process_video_input` read free/total accelerator memory to size
memory-safe vision-encoder chunks via `_encoder_chunk` (which caps the
`F.one_hot` position-embedding transient at `min(free/2, total/10)`).
`torch.accelerator.get_memory_info()` requires a PyTorch-native accelerator
backend; on vLLM's TPU platform the model executes through torchax on JAX,
so the call raises and the whole engine dies.

**Fix** (platform-generic, no JAX special-casing): a
`_accelerator_memory_info()` helper wraps the call and returns `(0, 0)`
("unknown") when the platform cannot answer. `_encoder_chunk` already
treats a non-positive budget as chunk size 1 — the minimal, always
memory-safe fallback — so unknown-memory platforms encode multimodal
inputs one item at a time instead of crashing. A `warning_once` records
that the heuristic is disabled.

| File | What |
|---|---|
| `d626108/gemma4_mm.py` | full patched file — byte-safe to overlay over `/workspace/vllm/vllm/model_executor/models/gemma4_mm.py` in any image with vLLM == `d626108` (e.g. as a ConfigMap `subPath` volumeMount, no image rebuild) |
| `d626108/gemma4_mm.py.diff` | the unified diff (3 hunks; applies cleanly to neighboring vLLM commits too) |
| `d626108/Dockerfile` | **image-proper form**: layered build `FROM vllm/vllm-tpu:nightly-20260824-b98d381-d626108` that copies the patched file over the vendored path and hard-verifies the base's `/workspace/vllm` HEAD == `d626108` (mismatch fails the build) |
| `d626108/cloudbuild.yaml` | Cloud Build recipe (linux/amd64) publishing `us-central1-docker.pkg.dev/global-cloud-runtime/llm-serving/vllm-tpu:nightly-20260824-pieces-gemma4vision-94fd1fe` |

### Image-proper build (reproducible from this directory alone)

```bash
cd patches/vllm/d626108
gcloud builds submit --project global-cloud-runtime --config cloudbuild.yaml .
```

This is deliberately a **layered build** from the pinned official nightly,
not a from-source rebuild: the official `docker/Dockerfile` build (vLLM TPU
compile + tpu-inference editable install) is multi-hour and would reproduce
bit-for-bit the layers the nightly already published. Layering keeps the
delta auditable — one changed source file, verified against the exact vLLM
pin at build time — and the `__pycache__` purge in the `RUN` step guarantees
the patched source is what actually imports. Tags are immutable per fork
commit: `<base-nightly-date>-pieces-<fix>-<short fork sha>`; any new commit
gets a new tag.

**Upstream destination:** vllm-project/vllm (`vllm/model_executor/models/gemma4_mm.py`).
The diff is self-contained and carries no TPU-specific code — it is
upstreamable as-is.

**Live verification (2026-08-24/25, GKE v6e-4 `ct6e-standard-4t`,
`google/gemma-4-26B-A4B-it` bf16, `MODEL_IMPL_TYPE=vllm`, TP=4,
`max_soft_tokens` 1120, temp 0 / seed 42, schema-constrained
visual-memory workload) — both proof points PASSED the guard:**

1. *Overlay smoke* (us-central1-b Spot, ConfigMap `subPath` over the
   in-image path on `…/vllm-tpu-patched:nightly-20260824-p2`): the
   previously-fatal image request **no longer raises**
   `PyTorch is not linked with support for jax devices` — it proceeded
   into the vision encoder and exposed a separate HBM-headroom OOM at
   `--gpu-memory-utilization 0.90` (6.08 GB encoder transient vs 2.76 GB
   free). Overlay byte-verified in-pod (vLLM HEAD `d626108`, pristine base
   sha `e005365b…`, mounted sha `c5cd4773…`).
2. *Image-proper* (europe-west4-a on-demand,
   `…/llm-serving/vllm-tpu:nightly-20260824-pieces-gemma4vision-94fd1fe`,
   no overlay, `--gpu-memory-utilization 0.58`): **vision PASS** —
   the screenshot+schema request completes (`finish_reason=stop`), first
   post-restart single 59.8 s, warm singles 10.97 s / 9.10 s with
   byte-identical 1,189-token outputs; valid JSON, `schema_version='vm1'`
   - instruction canary correct; quality harness: parse/schema 1.0,
   21/22 deterministic checks (sole miss: resolved-absolute-date, a
   model-quality item), ANLS 0.682, tier-2 macro-F1 0.754. Engine
   survived everything — zero restarts.

**Operational findings** (deploy-config, not code): 26B-A4B + vision on
v6e-4 needs `--gpu-memory-utilization 0.58` (0.90 and 0.75 both OOM the
~6.1 GB vision-encoder transient). And tpu-inference **#1531 remains** on
this vanilla-nightly base at C≥2 concurrent constrained requests — here it
failed CLOSED (`backend_xgrammar: Failed to advance FSM … grammar rejected
tokens` → one HTTP 500, engine alive) — that fix (#1563) is deliberately
NOT in this image; the lane-2 `p2` image carries it separately.

## d626108 — gemma4_mm.py fix 2: jit-safe suppress-tokens in `compute_logits` (the C≤8 ceiling)

Upstream `compute_logits` caches the suppressed-token index tensor
(`generation_config.suppress_tokens` → `int32[2]` for gemma-4) in **module
state**, keyed by device — a sound eager-CUDA optimization that is fatal on
vLLM's TPU backend, where the model body runs inside `jax.jit` via torchax:
the first trace stores a trace-bound tensor, and the poisoning happens
**during warmup itself**: the AOT `.lower()` at the second batch bucket
silently reuses the tracer cached by the first, producing a poisoned
lowering. The engine then dies with `UnexpectedTracerError` →
`EngineDeadError` on the FIRST batch that pads to that bucket — being
"inside the precompiled set" does not protect a shape. C≤8 stays safe only
because MIN_NUM_SEQS=8 pads every small batch to the one un-poisoned first
shape. (Mechanism reproduced on CPU jax 0.11.1; see
`tests/models/test_suppress_tokens_jit_safety.py`.)

Observed live (2026-08-26, v6e-1, gemma-4-12B): C=4 and C=8 clean, **C=12
dead in seconds**, identically for vision and audio requests (they share the
LM head). No config workaround exists — generation-config overrides do not
reach this code path. NOTE: this fix removes the kill; it does not by itself
lift the throughput ceiling — with `max-num-seqs=8` the runner's only batch
bucket is [8] and C=12 clients simply queue. Lifting the ceiling requires
this image PLUS `max-num-seqs>8` PLUS a C=12 validation run, which has not
yet been performed on p9.

Fix: rebuild the index tensor on every call. Under `jax.jit` the token list
is static, so the tensor constant-folds into the compiled graph (free after
compile); on eager CUDA it costs one async `int64[2]` H2D per step. The
image build now hard-fails if `_suppress_token_ids_cache` reappears in the
vendored file (upstream bump guard). Ships in `p9`
(`patches/image/p9/`), which also bakes pinned audio runtime deps
(librosa/soundfile/audioread, with a build-time assertion that they do not
rewrite the validated numpy/scipy) — their absence from the official image
was the observed blocker on the 12B audio path (audio revalidation on p9
still pending).
