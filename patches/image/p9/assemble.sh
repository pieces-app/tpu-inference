#!/usr/bin/env bash
# Copy the patched vendored-vllm file from the repo into the build context.
#
# Refuses to run on a dirty tree: the Dockerfile's provenance LABEL claims a
# specific fork commit, and copying uncommitted edits would make that claim
# false (the p6 recipe had exactly this hole).
set -euo pipefail
cd "$(dirname "$0")"
ROOT=../../..

if ! git -C "$ROOT" diff --quiet HEAD -- patches/vllm; then
  echo "FATAL: patches/vllm/ has uncommitted changes; commit them first so" >&2
  echo "       the image's recorded commit actually describes its contents." >&2
  exit 1
fi
COMMIT="$(git -C "$ROOT" rev-parse HEAD)"

git -C "$ROOT" show "HEAD:patches/vllm/d626108/gemma4_mm.py" > gemma4_mm.py
echo "$COMMIT" > GIT_COMMIT
echo "context assembled from ${COMMIT}"
