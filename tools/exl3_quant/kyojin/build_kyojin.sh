#!/bin/bash
# Build the Kyojin converter runtime (Python site + exllamav3_ext for gfx1151) in <runtime-dir>.
# Usage: tools/exl3_quant/kyojin/build_kyojin.sh <runtime-dir> [<wheel-dir>]
#   <wheel-dir> defaults to the verified offline copy
#   /home/m2on/projects/_backup/third_party/kyojin-wheels-rocm10.1.0a20260822
# Reproduces runtime/build.sh (the recorded c03 converter build) with the pin:
#   1. <runtime-dir>/source must come from fetch_kyojin.sh (verified tree).
#   2. Every wheel named in runtime/WHEELS.json must be in <wheel-dir> with the recorded
#      sha256 (ROCm 10.1.0a20260822 nightlies, https://rocm.nightlies.amd.com/v2/gfx1151/);
#      they are installed without dependencies or network into <runtime-dir>/site.
#      marisa_trie 1.3.1 was also present in the recorded site but is in neither WHEELS.json
#      nor the wheel cache; it is not installed here.
#   3. In the converter image (no GPU, no network) the extension is built in place:
#      PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=4 EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD.
# Image: ai-tools/gufo-eval:f3-native-b4ebe27c-rocm101 (KYOJIN_IMAGE to override).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
[[ $# -eq 1 || $# -eq 2 ]] || { sed -n '2,15p' "$0" >&2; exit 2; }
RT=$(cd "$1" && pwd)
WH=$(cd "${2:-/home/m2on/projects/_backup/third_party/kyojin-wheels-rocm10.1.0a20260822}" && pwd)
IMAGE=${KYOJIN_IMAGE:-sha256:79e3e0b7bc23ed230dc90a11e342aeae109086d145bb0b302363ac2069b43126}
python3 -I "$HERE/pin.py" check --dest "$RT/source"
NAMES=$(python3 -I -c 'import json,sys; print(" ".join(w["name"] for w in json.load(open(sys.argv[1]))))' "$HERE/runtime/WHEELS.json")
python3 -I - "$HERE/runtime/WHEELS.json" "$WH" <<'PY'
import hashlib, json, sys
from pathlib import Path
bad = []
for w in json.loads(Path(sys.argv[1]).read_text()):
    p = Path(sys.argv[2]) / w["name"]
    if not p.is_file():
        bad.append("missing " + w["name"]); continue
    h = hashlib.sha256()
    with p.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""): h.update(b)
    if h.hexdigest() != w["sha256"]: bad.append("sha256 differs " + w["name"])
print("\n".join(bad) or "wheels verified", file=sys.stderr if bad else sys.stdout)
sys.exit(1 if bad else 0)
PY
mkdir -p "$RT/site" "$RT/home"
docker run --rm --pull never --network none --user 1000:1000 --memory 20g --memory-swap 20g \
  --tmpfs /tmp:rw,size=2g --mount "type=bind,src=$RT,dst=/runtime" --mount "type=bind,src=$WH,dst=/wheels,readonly" \
  -e "WHEELS=$NAMES" --entrypoint /bin/bash "$IMAGE" -c '
set -euo pipefail
ROOT=/runtime
/opt/venv/bin/python -m pip install --no-deps --no-index --no-compile --target "$ROOT/site" $(for w in $WHEELS; do echo "/wheels/$w"; done)
export PYTHONPATH="$ROOT/site" PATH="$ROOT/site/bin:/opt/venv/bin:$PATH"
export EXL3_ROCM_SDK="$ROOT/site/_rocm_sdk_devel" EXL3_VENV=/opt/venv EXL3_ROOT="$ROOT/source"
export PYTORCH_ROCM_ARCH=gfx1151 MAX_JOBS=4 EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD HOME="$ROOT/home"
rocm-sdk init
source "$ROOT/source/tools/strix_halo/env.sh" build
python -c "import torch; print(torch.__version__, torch.version.hip, torch.__file__); assert \"rocm10.1.0a20260822\" in torch.__version__"
cd "$ROOT/source"
python setup.py build_ext --inplace
python -c "import torch,exllamav3_ext; print(exllamav3_ext.__file__)"'
