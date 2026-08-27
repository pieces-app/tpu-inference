# pieces-app/tpu-inference — image-proper form of the gemma4_mm.py fix
# (branch fix/gemma4-vision-jax-memory-info).
#
# The official vllm/vllm-tpu images vendor vllm-project/vllm as an editable
# install in /workspace/vllm, cloned at the build arg VLLM_COMMIT_HASH (see
# docker/Dockerfile at the repo root); the nightly tag encodes both pins:
#   nightly-<date>-<tpu_inference_sha>-<vllm_sha>
#
# This build reproduces the official image with EXACTLY ONE delta: the
# patched vllm/model_executor/models/gemma4_mm.py in this directory
# (generated against the same vLLM pin, d626108) copied over the vendored
# path. It is a LAYERED build FROM the pinned official image rather than a
# from-source rebuild: same base layers, same wheels, same everything except
# the one patched source file — the from-source path (vLLM TPU compile +
# tpu-inference install) is multi-hour and reproduces bit-for-bit what the
# official nightly already published. The RUN step hard-verifies that the
# base image's /workspace/vllm checkout is at the exact commit this patch
# was generated against, so a base/patch mismatch fails the build instead
# of shipping skew.
#
# Build (Cloud Build, linux/amd64):
#   cd patches/vllm/d626108 && gcloud builds submit --config cloudbuild.yaml .
ARG BASE_IMAGE=vllm/vllm-tpu:nightly-20260824-b98d381-d626108
FROM ${BASE_IMAGE}

# The vLLM commit the patched file was generated against (and the base
# image's expected /workspace/vllm HEAD).
ARG VLLM_COMMIT=d626108b1841888ec90aced33367149a6bbc7e4b

COPY gemma4_mm.py /tmp/patched-gemma4_mm.py
RUN set -e; \
    cd /workspace/vllm; \
    HEAD="$(git rev-parse HEAD)"; \
    if [ "${HEAD}" != "${VLLM_COMMIT}" ]; then \
        echo "FATAL: base image vLLM pin ${HEAD} != patch target ${VLLM_COMMIT}"; \
        exit 1; \
    fi; \
    cp /tmp/patched-gemma4_mm.py vllm/model_executor/models/gemma4_mm.py; \
    rm -rf vllm/model_executor/models/__pycache__; \
    rm /tmp/patched-gemma4_mm.py; \
    grep -q "_accelerator_memory_info" vllm/model_executor/models/gemma4_mm.py; \
    if grep -q "_suppress_token_ids_cache" vllm/model_executor/models/gemma4_mm.py; then echo "FATAL: module-state suppress-token cache still present"; exit 1; fi; \
    python3 -c "import ast; ast.parse(open('vllm/model_executor/models/gemma4_mm.py').read())"; \
    echo "gemma4_mm.py patch applied and verified against vLLM ${VLLM_COMMIT}"

LABEL org.opencontainers.image.source="https://github.com/pieces-app/tpu-inference" \
      org.opencontainers.image.revision="94fd1fe19448401925109a5b77b657032a397a5c" \
      org.opencontainers.image.description="vllm/vllm-tpu nightly-20260824-b98d381-d626108 + gemma4_mm torch.accelerator.get_memory_info() guard + jit-safe compute_logits suppress-tokens (fix/gemma4-vision-jax-memory-info, fix/gemma4-c12-suppress-tokens)"
