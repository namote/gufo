#!/usr/bin/env bash
# Build tools/qwen38_flash_next/bench/exl3_prefill_shapes_bench.hip against a configured build
# tree (default build/f3) whose qwen38_flash_next_exl3_ops_test is built:
#   tools/qwen38_flash_next/bench/exl3_prefill_shapes_bench.sh [build-dir]
set -euo pipefail
cd "$(dirname "$0")/../../.."
B=${1:-build/f3}
HIPCC=/opt/rocm-7.2.4/lib/llvm/bin/clang++
"$HIPCC" -x hip -O3 -std=c++20 --offload-arch=gfx1151 -DENGINE_ENABLE_HIP=1 \
  -DNDEBUG -I. -c tools/qwen38_flash_next/bench/exl3_prefill_shapes_bench.hip \
  -o "$B/exl3_prefill_shapes_bench.o"
LIBS=$(grep -A3 '^build tests/models/qwen38_flash_next/qwen38_flash_next_exl3_ops_test:' \
  "$B/build.ninja" | sed -n 's/^  LINK_LIBRARIES = //p')
(cd "$B" && "$HIPCC" --hip-link --rtlib=compiler-rt -unwindlib=libgcc \
  --offload-arch=gfx1151 exl3_prefill_shapes_bench.o $LIBS \
  -o exl3_prefill_shapes_bench)
