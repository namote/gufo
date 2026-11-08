"""Compile a ctypes HIP diagnostic library with the exact active runtime SDK; no GPU."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

root = Path('/output/capture-gram'); root.mkdir(exist_ok=True)
source = Path('/task/capture_gram.hip')
sdk = Path(os.environ['EXL3_ROCM_SDK'])
command = [os.environ['EXL3_HIPCC'], '-O3', '--offload-arch=gfx1151',
           '-std=c++20', '-shared', '-fPIC', '-Rpass-analysis=kernel-resource-usage',
           str(source), '-I'+str(sdk/'include'), '-L'+str(sdk/'lib'),
           '-Wl,-rpath,'+str(sdk/'lib'), '-lhipblaslt', '-o', str(root/'gram.so')]
started = time.monotonic()
with (root/'BUILD.log').open('w') as log:
    subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True, timeout=180)
root.joinpath('BUILD.json').write_text(json.dumps(dict(status='compiled',gpu_used=False,
    command=command, seconds=time.monotonic()-started,
    source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
    binary_sha256=hashlib.sha256((root/'gram.so').read_bytes()).hexdigest()), indent=2))
