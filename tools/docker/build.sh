#!/bin/bash
# CPU-only incremental build in the pinned ROCm 10.1 closure (no GPU devices, no network).
# Usage: tools/docker/build.sh [--build-dir DIR] [targets...]   (default targets: gufo)
#        tools/docker/build.sh [--build-dir DIR] --run CMD...   (run CMD in the same environment)
# DIR (default build/pinned-rocm101) is mounted at /build; objects live in DIR/obj.
# Configure DIR/obj once with:
#   tools/docker/build.sh --run sh -c "$(tools/docker/build.sh --run python3 \
#     tools/docker/pinned_rocm101.py configure-args <revision>)"
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR=$ROOT/build/pinned-rocm101
if [[ "${1:-}" == "--build-dir" ]]; then DIR=$2; shift 2; fi
if [[ "${1:-}" == "--run" ]]; then shift; CMD=("$@"); else
  T=("$@"); [[ ${#T[@]} -eq 0 ]] && T=(gufo)
  CMD=(cmake --build /build/obj --parallel 4 --target "${T[@]}"); fi
exec python3 -I "$ROOT/tools/docker/pinned_rocm101.py" run --build-dir "$DIR" --source "$ROOT" -- "${CMD[@]}"
