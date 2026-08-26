# p6: p5 + Gemma-4 Unified vision fidelity fixes (branch
# fix/gemma4-unified-vision-fidelity):
#   1. mm_bidi_ranges — PrefixLM blockwise-bidirectional attention for
#      image soft-token blocks on sliding layers (RPA v3 kernel + metadata
#      + runner + flash_attn backend). Root cause of the 12B fine-text
#      misreads (PID 36793 -> 36732): image tokens attended causally.
#   2. fp32 Unified vision embedder under torchax (bf16 LayerNorm stats
#      corrupt near-flat patches; p95 per-token err 438 at LN2).
#
# Layered FROM the p5 image (= p4 tpu-inference fixes + vllm parser
# commits); copies whole files over the editable install, mirroring the
# p1/p2 build. Build:
#   cd patches/image/p6 && ./assemble.sh && \
#   gcloud builds submit --project global-cloud-runtime --config cloudbuild.yaml .
FROM gcr.io/global-cloud-runtime/vllm-tpu-patched:nightly-20260824-p5

COPY tpu_inference/ /tmp/p6/tpu_inference/

RUN set -e; \
    TI_DIR=/workspace/tpu_inference/tpu_inference; \
    test -f "$TI_DIR/kernels/ragged_paged_attention/v3/kernel.py" || { echo "tpu_inference not located"; exit 1; }; \
    for f in \
      kernels/ragged_paged_attention/v3/kernel.py \
      layers/common/attention_metadata.py \
      layers/common/attention_interface.py \
      layers/vllm/backends/flash_attn.py \
      models/vllm/experimental/gemma4_unified_patcher.py \
      models/vllm/experimental/model_patcher.py \
      runner/tpu_runner.py \
      runner/compilation_manager.py \
    ; do \
      cp "/tmp/p6/tpu_inference/$f" "$TI_DIR/$f"; \
    done; \
    rm -rf "$TI_DIR/kernels/ragged_paged_attention/v3/__pycache__" \
           "$TI_DIR/layers/common/__pycache__" \
           "$TI_DIR/layers/vllm/backends/__pycache__" \
           "$TI_DIR/models/vllm/experimental/__pycache__" \
           "$TI_DIR/runner/__pycache__" /tmp/p6; \
    grep -q "mm_bidi_ranges" "$TI_DIR/kernels/ragged_paged_attention/v3/kernel.py"; \
    grep -q "mm_bidi_enabled" "$TI_DIR/runner/tpu_runner.py"; \
    grep -q "maybe_apply_gemma4_unified_patches" "$TI_DIR/models/vllm/experimental/model_patcher.py"; \
    python3 -c "import ast,glob; [ast.parse(open(p).read()) for p in glob.glob('/workspace/tpu_inference/tpu_inference/**/*.py', recursive=True)]"; \
    python3 -c "import tpu_inference.kernels.ragged_paged_attention.v3.kernel; print('kernel module imports OK')"; \
    echo "P6 PATCHES APPLIED AND VERIFIED"

LABEL org.opencontainers.image.source="https://github.com/pieces-app/tpu-inference" \
      org.opencontainers.image.description="p5 + Gemma-4 Unified PrefixLM blockwise attention (mm_bidi_ranges) + fp32 unified vision embedder (fix/gemma4-unified-vision-fidelity)"
