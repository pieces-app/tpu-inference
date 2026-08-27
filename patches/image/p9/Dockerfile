# p9: p8 + the C=12 concurrency-ceiling fix + audio runtime deps (branch
# fix/gemma4-c12-suppress-tokens):
#   1. gemma4_mm.py compute_logits — remove upstream's per-device
#      suppressed-token index cache. The cache stores a tensor created
#      DURING a jax.jit trace in module state; any batch shape outside the
#      precompiled set re-traces and reuses the stale trace-bound value,
#      killing the engine (the observed hard C<=8 ceiling: C=12 dead in
#      seconds, vision and audio alike — they share this LM head). Under
#      jit the rebuilt tensor constant-folds into the compiled graph, so
#      per-call recreation is free after compile.
#   2. vllm[audio] runtime deps (librosa/soundfile/audioread) baked in —
#      absent from the official vllm-tpu image; their absence was the ONLY
#      thing blocking the (validated-working) gemma-4-12B audio path.
#
# Layered FROM the p8 image, mirroring the p1/p2/p8 builds. p8's
# /workspace/vllm is the official nightly checkout at d626108 with the
# patched gemma4_mm.py already applied once (memory-info guard); this build
# overwrites it with the current pair (guard + C=12 fix) and hard-verifies
# both markers. Build:
#   cd patches/image/p9 && ./assemble.sh && \
#   gcloud builds submit --project global-cloud-runtime --config cloudbuild.yaml .
#
# Base pinned BY DIGEST (tags are mutable):
# sha256:7bf98db5... == vllm-tpu-patched:nightly-20260824-p8 as of 2026-08-27.
FROM gcr.io/global-cloud-runtime/vllm-tpu-patched@sha256:7bf98db5125e77e9d1e4a93e0bd14c43a121ebc752a24283c8d8e70f4bfce461

# The vLLM commit the patched file was generated against (and the base
# image's expected /workspace/vllm HEAD).
ARG VLLM_COMMIT=d626108b1841888ec90aced33367149a6bbc7e4b

COPY gemma4_mm.py /tmp/p9/gemma4_mm.py

RUN set -e; \
    cd /workspace/vllm; \
    HEAD="$(git rev-parse HEAD)"; \
    if [ "${HEAD}" != "${VLLM_COMMIT}" ]; then \
        echo "FATAL: base image vLLM pin ${HEAD} != patch target ${VLLM_COMMIT}"; \
        exit 1; \
    fi; \
    cp /tmp/p9/gemma4_mm.py vllm/model_executor/models/gemma4_mm.py; \
    rm -rf vllm/model_executor/models/__pycache__ /tmp/p9; \
    grep -q "_accelerator_memory_info" vllm/model_executor/models/gemma4_mm.py; \
    if grep -q "_suppress_token_ids_cache" vllm/model_executor/models/gemma4_mm.py; then \
        echo "FATAL: module-state suppress-token cache still present"; \
        exit 1; \
    fi; \
    python3 -c "import ast; ast.parse(open('vllm/model_executor/models/gemma4_mm.py').read())"; \
    # Pins match the DEPLOYED p9's resolved set (build log 2026-08-27 16:22Z,
    # image built from b8e79925 with the then-unpinned install) — the versions
    # the hardware validation sweep is exercising. Bump ONLY alongside fresh
    # validation evidence.
    python3 -c "import numpy, scipy; open('/tmp/numstack','w').write(numpy.__version__+' '+scipy.__version__)"; \
    pip install --no-cache-dir librosa==1.0.0 soundfile==0.14.0 audioread==3.1.0; \
    python3 -c "import numpy, scipy; pre=open('/tmp/numstack').read().split(); cur=[numpy.__version__, scipy.__version__]; assert cur==pre, f'FATAL: audio deps rewrote the validated numeric stack: {pre} -> {cur}'"; \
    python3 -c "import librosa, soundfile, audioread; print('audio deps OK')"; \
    echo "P9 PATCHES APPLIED AND VERIFIED"

# Provenance: the exact fork commit assemble.sh read the sources from.
COPY GIT_COMMIT /etc/pieces-tpu-inference-commit

LABEL org.opencontainers.image.source="https://github.com/pieces-app/tpu-inference" \
      org.opencontainers.image.description="p8 + gemma4_mm compute_logits jit-safe suppress-tokens (module-state tracer-cache fix; C>8 needs max-num-seqs>8 + validation) + pinned librosa/soundfile/audioread (fix/gemma4-c12-suppress-tokens)"
