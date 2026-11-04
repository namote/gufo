#!/usr/bin/env bash
# Build tools/bench/exl3_wmma_threshold_bench.hip against a configured build
# tree (default build/gpu-test) whose qwen27b_exl3_to_gguf is built:
#   tools/bench/exl3_wmma_threshold_bench.sh [build-dir]
set -euo pipefail
cd "$(dirname "$0")/../.."
B=${1:-build/gpu-test}
HIPCC=${HIPCC:-/opt/rocm/llvm/bin/clang++}
"$HIPCC" -x hip -O3 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 \
  -DNDEBUG -I. -c tools/bench/exl3_wmma_threshold_bench.hip \
  -o "$B/exl3_wmma_threshold_bench.o"
LIBS=$(grep -A3 '^build tests/models/qwen27b/qwen27b_exl3_to_gguf:' \
  "$B/build.ninja" | sed -n 's/^  LINK_LIBRARIES = //p')
(cd "$B" && "$HIPCC" --hip-link --rtlib=compiler-rt -unwindlib=libgcc \
  --offload-arch=gfx1151 exl3_wmma_threshold_bench.o $LIBS \
  -o exl3_wmma_threshold_bench)
