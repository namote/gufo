"""CPU qualification of production adapter contracts, not a numerical GPU rerun."""
from pathlib import Path
from types import SimpleNamespace
import hashlib,json,torch
from exllamav3.modules.linear import flush_capture_H
import capture_lean_runtime as runtime
R=Path('/output/capture-lean');checks=[]
class Native:
    def __init__(self,path):self.closed=False;self.calls=[];self.cache=SimpleNamespace(stats={})
    def __call__(self,x):self.calls.append(x.dtype);return torch.bmm(x.float().transpose(1,2),x.float())
    def close(self):self.closed=True
contract=json.loads(Path('/task/capture-lt-contract.json').read_text())
original=runtime.PreparedLtGram;runtime.PreparedLtGram=Native
try:
    backend=runtime.QualifiedPrepared(contract,'/task/native/capture-gram.so')
    for half in (True,False):
        shape=next(s for s in contract['allowed_shapes'] if s[3]==half)
        # Meta tensor exercises dtype/shape dispatch without allocating its payload.
        x=torch.empty(shape[:3],dtype=torch.float16 if half else torch.float32,device='meta')
        assert backend(x).shape==(shape[0],shape[2],shape[2])
    assert backend.stats==dict(half_calls=1,fp32_calls=1)
    for x in (torch.empty(1,3,7),torch.empty(1,3,7,dtype=torch.float64)):
        try:backend(x)
        except AssertionError:pass
        else:raise AssertionError('unqualified shape/type accepted')
    backend.close();assert backend.native.closed;checks.append('qualified_shape_dtype_guards_preserved')
    cm=SimpleNamespace(_flush_capture_H=flush_capture_H)
    restore=runtime.install_gram(cm,'/task/capture-lt-contract.json','/task/native/capture-gram.so',contract['calibration_sha256'])
    assert cm._flush_capture_H is not flush_capture_H
    cm._flush_capture_H({});restore();assert cm._flush_capture_H is flush_capture_H
    checks.append('gram_install_restore')
    try:runtime.install_gram(cm,'/task/capture-lt-contract.json','/task/native/capture-gram.so','wrong')
    except AssertionError:pass
    else:raise AssertionError('wrong calibration accepted')
    checks.append('calibration_identity_guard')
finally:runtime.PreparedLtGram=original
# MoE production adapter must enter/exit the very same reviewed context manager.
seen=[]
class Scope:
    def __enter__(self):seen.append('enter')
    def __exit__(self,*args):seen.append('exit')
old=runtime.installed
try:
    runtime.installed=lambda contract:Scope()
    restore=runtime.install_moe('/task/capture-lean-contract.json');restore();assert seen==['enter','exit']
finally:runtime.installed=old
checks.append('moe_scope_lifetime')
for n in ('convert_main_lean_candidate.py','model_job_lean_candidate.py'):
    compile(Path('/task',n).read_text(),n,'exec')
caller=Path('/task/convert_main_lean_candidate.py').read_text()
assert caller.index('install_capture_fast(cm)')<caller.index('restore_lean=install_lean_moe(')<caller.index('cm.main(in_args,job)')
assert "finally:\n    restore_lean()\n    restore_owner()\n    restore_lt()" in caller
supervisor=Path('/task/model_job_lean_candidate.py').read_text()
assert supervisor.index('disk_receipt=prepare_disk(OUT,ROOT)')<supervisor.index('subprocess.Popen(')
checks.append('caller_order_and_disk_preflight')
files=['capture_lean_runtime.py','test_capture_lean_runtime.py','convert_main_lean_candidate.py','model_job_lean_candidate.py']
receipt=dict(status='pass',gpu_used=False,checks=checks,identity={n:hashlib.sha256(Path('/task',n).read_bytes()).hexdigest() for n in files})
(R/'RUNTIME-CPU.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt))
