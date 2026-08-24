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

**Live verification plan** (two proof points, GKE Spot v6e-4,
`ct6e-standard-4t`, us-central1-b):
1. *Overlay smoke* — the patched file ConfigMap-`subPath`-mounted over the
   in-image path on `gcr.io/global-cloud-runtime/vllm-tpu-patched:nightly-20260824-p2`;
2. *Image-proper* — redeploy on the Cloud-Built
   `…/llm-serving/vllm-tpu:nightly-20260824-pieces-gemma4vision-94fd1fe`
   (no overlay), then the full single-call scorecard + quality scoring +
   the #1531 C=2 constrained probe.

Results are recorded in a follow-up commit here and in
`global-cloud-runtime/deploy/gke-tpu/README.md` (isolation matrix).
