# p8: p5 + Gemma-4 Unified vision fidelity fixes (branch
# fix/gemma4-unified-vision-fidelity):
#   1. mm_bidi_ranges — PrefixLM blockwise-bidirectional attention for
#      image soft-token blocks on sliding layers (RPA v3 kernel + metadata
#      + runner + flash_attn backend). Root cause of the 12B fine-text
#      misreads (PID 36793 -> 36732): image tokens attended causally.
#   2. fp32 LayerNorm STATISTICS in the Unified vision embedder under
#      torchax (bf16 mean/var corrupts near-flat patches; p95 per-token
#      err 438 -> 1.2 at LN2, soft-token cosine 0.9944 -> 1.000000).
#   3. The multimodal step carries input_ids ALONGSIDE inputs_embeds, so
#      Gemma-4 E2B/E4B compute the real per-token PLE id-track on an image
#      prefill instead of looking up embedding slot 0 for the whole prompt.
#      Both backbone primers build the matching operand; three greps below
#      verify the runner's return and the primers' arguments survived the
#      copy. models/jax/gemma4.py is deliberately NOT copied: its change is
#      comments only, and adding a new whole-file copy would widen the
#      image's surface for no behavioural gain.
#
# Layered FROM the p5 image (= p4 tpu-inference fixes + vllm parser
# commits); copies whole files over the editable install, mirroring the
# p1/p2 build. Build:
#   cd patches/image/p8 && ./assemble.sh && \
#   gcloud builds submit --project global-cloud-runtime --config cloudbuild.yaml .
#
# assemble.sh copies the eight patched files out of git HEAD and
# refuses to run on a dirty tree, stamping the exact commit into
# GIT_COMMIT so the image records what it was built from.
# Base pinned BY DIGEST: the p5 tag lives outside this repo and is mutable,
# so a tag-only FROM would let the base drift under a fixed recipe.
# sha256:926f2818... == vllm-tpu-patched:nightly-20260824-p5 as of 2026-08-26.
FROM gcr.io/global-cloud-runtime/vllm-tpu-patched@sha256:926f28185c2eb8ecce7e37003b9947f0416fd672a320d084a1a15ee1d2ae66de

COPY tpu_inference/ /tmp/p8/tpu_inference/

RUN set -e; \
    TI_DIR=/workspace/tpu_inference/tpu_inference; \
    test -f "$TI_DIR/kernels/ragged_paged_attention/v3/kernel.py" || { echo "tpu_inference not located"; exit 1; }; \
    for f in \
      envs.py \
      kernels/ragged_paged_attention/v3/kernel.py \
      layers/common/attention_metadata.py \
      layers/common/attention_interface.py \
      layers/vllm/backends/flash_attn.py \
      models/common/mm_debug_stats.py \
      models/jax/gemma4_mm.py \
      models/jax/gemma4_vision_clip.py \
      models/jax/gemma4_unified.py \
      models/vllm/experimental/gemma4_mm_patcher.py \
      models/vllm/experimental/gemma4_vision_attention.py \
      models/vllm/experimental/gemma4_unified_patcher.py \
      models/vllm/experimental/mm_debug_patch.py \
      models/vllm/experimental/mm_jit_signature.py \
      models/vllm/experimental/model_patcher.py \
      runner/mm_bidi_ranges.py \
      runner/tpu_runner.py \
      runner/compilation_manager.py \
      runner/multimodal_manager.py \
    ; do \
      cp "/tmp/p8/tpu_inference/$f" "$TI_DIR/$f"; \
    done; \
    rm -rf "$TI_DIR/__pycache__" \
           "$TI_DIR/kernels/ragged_paged_attention/v3/__pycache__" \
           "$TI_DIR/layers/common/__pycache__" \
           "$TI_DIR/layers/vllm/backends/__pycache__" \
           "$TI_DIR/models/common/__pycache__" \
           "$TI_DIR/models/jax/__pycache__" \
           "$TI_DIR/models/vllm/experimental/__pycache__" \
           "$TI_DIR/runner/__pycache__" /tmp/p8; \
    grep -q "mm_bidi_ranges" "$TI_DIR/kernels/ragged_paged_attention/v3/kernel.py"; \
    grep -q "mm_bidi_enabled" "$TI_DIR/runner/tpu_runner.py"; \
    test -f "$TI_DIR/runner/mm_bidi_ranges.py"; \
    grep -q "build_mm_bidi_ranges" "$TI_DIR/runner/tpu_runner.py"; \
    grep -q "if inputs_embeds is not None:" "$TI_DIR/models/vllm/experimental/gemma4_mm_patcher.py"; \
    grep -q "return input_ids, inputs_embeds" "$TI_DIR/runner/tpu_runner.py"; \
    ! grep -q "input_ids=None" "$TI_DIR/runner/compilation_manager.py"; \
    test "$(grep -c "input_ids=input_ids" "$TI_DIR/runner/compilation_manager.py")" = 2; \
    grep -q "maybe_apply_gemma4_unified_patches" "$TI_DIR/models/vllm/experimental/model_patcher.py"; \
    grep -q "compute_mm_jit_static_args" "$TI_DIR/models/vllm/experimental/model_patcher.py"; \
    test -f "$TI_DIR/models/vllm/experimental/mm_jit_signature.py"; \
    grep -q "PIECES_MM_DEBUG" "$TI_DIR/envs.py"; \
    grep -q "_install_torchax_mm_debug" "$TI_DIR/models/vllm/experimental/model_patcher.py"; \
    grep -q "emit_mm_debug_stats" "$TI_DIR/models/jax/gemma4_mm.py"; \
    grep -q "emit_mm_debug_stats" "$TI_DIR/models/jax/gemma4_unified.py"; \
    grep -q "debug_owners" "$TI_DIR/runner/multimodal_manager.py"; \
    test -f "$TI_DIR/models/common/mm_debug_stats.py"; \
    test -f "$TI_DIR/models/vllm/experimental/mm_debug_patch.py"; \
    grep -q "vision_tower.pooler" "$TI_DIR/models/vllm/experimental/mm_debug_patch.py"; \
    test -f "$TI_DIR/models/jax/gemma4_vision_clip.py"; \
    grep -q "Gemma4VisionClippedEinsum" "$TI_DIR/models/jax/gemma4_mm.py"; \
    test -f "$TI_DIR/models/vllm/experimental/gemma4_vision_attention.py"; \
    grep -q "fp32_logit_attention" "$TI_DIR/models/vllm/experimental/gemma4_vision_attention.py"; \
    grep -q "maybe_apply_gemma4_vision_attention_patch" "$TI_DIR/models/vllm/experimental/gemma4_mm_patcher.py"; \
    grep -q "PIECES_GEMMA4_VISION_ATTN_FP32" "$TI_DIR/envs.py"; \
    grep -q "_verify_vision_clamps_loaded" "$TI_DIR/models/jax/gemma4_mm.py"; \
    python3 -c "import ast,glob; [ast.parse(open(p).read()) for p in glob.glob('/workspace/tpu_inference/tpu_inference/**/*.py', recursive=True)]"; \
    python3 -c "import tpu_inference.kernels.ragged_paged_attention.v3.kernel; print('kernel module imports OK')"; \
    echo "P6 PATCHES APPLIED AND VERIFIED"

# Provenance: the exact fork commit assemble.sh read the sources from.
COPY GIT_COMMIT /etc/pieces-tpu-inference-commit

LABEL org.opencontainers.image.source="https://github.com/pieces-app/tpu-inference" \
      org.opencontainers.image.description="p5 + Gemma-4 Unified PrefixLM blockwise attention (mm_bidi_ranges) + fp32 unified vision embedder (fix/gemma4-unified-vision-fidelity)"
