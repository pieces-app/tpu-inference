#!/usr/bin/env bash
# Copy the patched vendored-vllm file from the repo into the build context.
#
# Refuses to run on a dirty tree: the Dockerfile's provenance LABEL claims a
# specific fork commit, and copying uncommitted edits would make that claim
# false (the p6 recipe had exactly this hole).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..

if ! git -C "$ROOT" diff --quiet HEAD -- patches/vllm patches/image/p9; then
  echo "FATAL: patches/vllm/ or patches/image/p9/ has uncommitted changes;" >&2
  echo "       commit them first so the image's recorded commit actually" >&2
  echo "       describes its contents (the p6 recipe had this hole)." >&2
  exit 1
fi
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

git -C "$ROOT" show "HEAD:patches/vllm/d626108/gemma4_mm.py" > gemma4_mm.py
echo "$COMMIT" > GIT_COMMIT
echo "context assembled from ${COMMIT}"
