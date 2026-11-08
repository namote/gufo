#!/bin/bash
# Fetch the pinned Kyojin v1.8.6 converter source into <runtime>/source and verify it.
# Usage: tools/exl3_quant/kyojin/fetch_kyojin.sh <runtime-dir> [--serve-patch]
# The archive is taken from the offline backup
#   /home/m2on/projects/_backup/third_party/kyojin-v1.8.6.tar.gz
# when its sha256 matches the pin (override with KYOJIN_BACKUP=<path|none>), else
# from https://github.com/Yamz-Labs/kyojin/archive/<rev>.tar.gz. The unpacked tree is
# checked against runtime/INPUTS.json, the source the c03 converter was built from.
# --serve-patch also applies patches/ (tools/qwen/serve.py --max-context; serving only).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[[ $# -ge 1 ]] || { sed -n '2,9p' "$0" >&2; exit 2; }
RT=$1; mkdir -p "$RT"
ARCHIVE=$RT/kyojin-v1.8.6.tar.gz
BACKUP=${KYOJIN_BACKUP:-/home/m2on/projects/_backup/third_party/kyojin-v1.8.6.tar.gz}
[[ -f $ARCHIVE ]] || python3 -I "$HERE/pin.py" fetch --archive "$ARCHIVE" --backup "$BACKUP"
python3 -I "$HERE/pin.py" unpack --archive "$ARCHIVE" --dest "$RT/source"
python3 -I "$HERE/pin.py" check --dest "$RT/source"
if [[ "${2:-}" == "--serve-patch" ]]; then python3 -I "$HERE/pin.py" patch --dest "$RT/source"; fi
