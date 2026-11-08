#!/bin/bash
# Run one command in a pinned ROCm 10.1 GPU container; record argv, timing, logs and exit code.
# Usage: tools/docker/gpu-run.sh [--build-dir DIR] [--mount SRC:DST[:ro]]... <outdir> <env|raw> CMD...
#   env: CMD runs with the pinned environment in /build/source (read-only repository).
#   raw: CMD[0] is the container entrypoint and the rest its arguments.
# Mount models, references and corpora explicitly, e.g.
#   --mount /home/m2on/projects/models/llm/flash-next-exl3-gufo-c03:/models/fnx:ro
# stdout.log, stderr.log, argv.txt, start, end and rc are written to <outdir>.
set -uo pipefail
ROOT=$(cd "$(dirname "$0")/../.." && pwd)
DIR=$ROOT/build/pinned-rocm101
M=()
while [[ $# -gt 0 ]]; do case "$1" in
  --build-dir) DIR=$2; shift 2 ;;
  --mount) M+=(--mount "$2"); shift 2 ;;
  *) break ;;
esac; done
[[ $# -ge 3 ]] || { sed -n '2,8p' "$0" >&2; exit 2; }
OUT=$1; MODE=$2; shift 2
case "$MODE" in env) RAW=() ;; raw) RAW=(--raw "$1"); shift ;; *) echo "mode must be env or raw" >&2; exit 2 ;; esac
mkdir -p "$OUT"
printf '%q ' "$MODE" "${RAW[@]}" -- "$@" > "$OUT/argv.txt"; echo >> "$OUT/argv.txt"
date +%s.%N > "$OUT/start"
python3 -I "$ROOT/tools/docker/pinned_rocm101.py" run --gpu --build-dir "$DIR" --source "$ROOT" --source-readonly \
  "${M[@]}" "${RAW[@]}" -- "$@" > "$OUT/stdout.log" 2> "$OUT/stderr.log"
rc=$?; echo $rc > "$OUT/rc"; date +%s.%N > "$OUT/end"; exit $rc
