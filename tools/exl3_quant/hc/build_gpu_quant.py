"""Build the offline feedback shared library in the CPU-only pinned toolchain."""
import json
import os
from pathlib import Path
import subprocess
from tools.docker.pinned_rocm101 import environment, GCC_PREFIX, GLIBC
from tools.exl3_quant.hc.quant import sha

if Path('/dev/kfd').exists(): raise RuntimeError('CPU build must not see GPU')
root=Path('/build'); root.mkdir(exist_ok=True)
for name in ('tmp','cache'): (root/name).mkdir(exist_ok=True)
env,_=environment()
source=Path('/build/source/tools/exl3_quant/hc/gptq_gpu.hip.cpp')
# Compile through the pinned HIP compiler, retaining separate FP64 multiply/add.
compiler=env.get('HIP_PATH','')+'/bin/hipcc'
if not Path(compiler).exists(): compiler='hipcc'
cmd=[compiler,'-shared','-fPIC','-O2','-ffp-contract=off','--offload-arch=gfx1151','-std=c++20',
     '--gcc-toolchain='+GCC_PREFIX,'-B'+GLIBC+'/lib',str(source),'-o','/build/libhc_feedback.so',
     '-L'+GLIBC+'/lib','-Wl,-rpath,'+GLIBC+'/lib']
subprocess.run(cmd,env=env,check=True)
(root/'BUILD.json').write_text(json.dumps({'source_sha256':sha(source),'library_sha256':sha(root/'libhc_feedback.so'),
    'command':cmd,'gpu_used':False,'fp_contract':'off'},indent=2)+'\n')
