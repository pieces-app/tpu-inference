#!/usr/bin/env bash
# Copy the patched files from the repo working tree into the build context.
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..
rm -rf tpu_inference
for f in \
  kernels/ragged_paged_attention/v3/kernel.py \
  layers/common/attention_metadata.py \
  layers/common/attention_interface.py \
  layers/vllm/backends/flash_attn.py \
  models/vllm/experimental/gemma4_unified_patcher.py \
  models/vllm/experimental/model_patcher.py \
  runner/tpu_runner.py \
  runner/compilation_manager.py \
; do
  mkdir -p "tpu_inference/$(dirname "$f")"
  cp "$ROOT/tpu_inference/$f" "tpu_inference/$f"
done
echo "context assembled"
