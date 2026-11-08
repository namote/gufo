#!/bin/bash
set -euo pipefail
ROOT=/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime
export PYTHONPATH="$ROOT/site"
export PATH="$ROOT/site/bin:/opt/venv/bin:$PATH"
export EXL3_ROCM_SDK="$ROOT/site/_rocm_sdk_devel"
export EXL3_VENV=/opt/venv
export EXL3_ROOT="$ROOT/source"
export PYTORCH_ROCM_ARCH=gfx1151
export MAX_JOBS=4
export EXL3_HIP_DEFINES=EXL3_HIP_STG_PAD
export HOME=/receipt/home
mkdir -p "$HOME"
rocm-sdk init
source "$ROOT/source/tools/strix_halo/env.sh" build
python -c 'import torch; print(torch.__version__, torch.version.hip, torch.__file__); assert "rocm10.1.0a20260822" in torch.__version__'
cd "$ROOT/source"
python setup.py build_ext --inplace
python -c 'import torch,exllamav3_ext; print(exllamav3_ext.__file__)'
