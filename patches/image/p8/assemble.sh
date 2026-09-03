#!/usr/bin/env bash
# Copy the patched files from the repo into the build context.
#
# Refuses to run on a dirty tree: the Dockerfile's provenance LABEL claims a
# specific fork commit, and copying uncommitted edits would make that claim
# false (the p6 recipe had exactly this hole).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..

if ! git -C "$ROOT" diff --quiet HEAD -- tpu_inference; then
  echo "FATAL: tpu_inference/ has uncommitted changes; commit them first so" >&2
  echo "       the image's recorded commit actually describes its contents." >&2
  exit 1
fi
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

rm -rf tpu_inference
for f in \
  envs.py \
  kernels/ragged_paged_attention/v3/kernel.py \
  layers/common/attention_metadata.py \
  layers/common/attention_interface.py \
  layers/vllm/backends/flash_attn.py \
  models/common/mm_debug_stats.py \
  models/jax/gemma4_mm.py \
  models/jax/gemma4_unified.py \
  models/vllm/experimental/gemma4_unified_patcher.py \
  models/vllm/experimental/mm_debug_patch.py \
  models/vllm/experimental/mm_jit_signature.py \
  models/vllm/experimental/model_patcher.py \
  runner/tpu_runner.py \
  runner/compilation_manager.py \
  runner/multimodal_manager.py \
; do
  mkdir -p "tpu_inference/$(dirname "$f")"
  git -C "$ROOT" show "HEAD:tpu_inference/$f" > "tpu_inference/$f"
done
echo "$COMMIT" > GIT_COMMIT
echo "context assembled from ${COMMIT}"
