"""CPU contracts for qualified dispatch and calibrated conversion integration."""
import ast,copy,hashlib,json,sys,threading,tempfile
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import patch
import torch
import capture_lt_runtime as runtime
checks=[]
class FakeNative:
    def __init__(self,path,mode):assert mode=='original_f16';self.cache=NS(stats={});self.calls=[]
    def __call__(self,xf,xs,finite):self.calls.append(xs.dtype);return xf
    def close(self):self.closed=True
with tempfile.TemporaryDirectory(dir='/output') as tmp:
    library=Path(tmp)/'stub.so';library.write_bytes(b'cpu-stub')
    c=dict(status='qualified_for_idx10_resume',native_sha256=hashlib.sha256(library.read_bytes()).hexdigest(),allowed_shapes=[[1,2,3,True],[1,2,3,False]],calibration_sha256='cal')
    with patch.object(runtime,'LtGram',FakeNative):
        g=runtime.QualifiedGram(c,library);xf=torch.ones(1,2,3);finite=torch.ones(1,2,dtype=torch.bool)
        for dt in (torch.float16,torch.float32,torch.bfloat16):g(xf,xf.to(dt),finite)
        assert g.stats==dict(half_calls=1,fp32_calls=2)
        assert g.native.calls==[torch.float16,torch.float32,torch.bfloat16]
        for xs in (torch.ones(2,2,3),torch.ones(1,2,3,dtype=torch.float64)):
            try:g(xf,xs,finite)
            except AssertionError:pass
            else:raise AssertionError('Unsupported input accepted')
        g.close();assert g.native.closed
        checks+=['preserve_original_dtype_route','reject_unqualified_shape_dtype','close_native']
        policy=Path(tmp)/'policy.json';policy.write_text(json.dumps(c));original=lambda p:None;cm=NS(_flush_capture_H=original)
        with patch.object(runtime,'make_flush',lambda old,gram:lambda params:gram):
            restore=runtime.install(cm,policy,library,'cal');assert cm._flush_capture_H is not original
            restore();assert cm._flush_capture_H is original
            try:runtime.install(cm,policy,library,'wrong')
            except AssertionError:pass
            else:raise AssertionError('calibration mismatch accepted')
        checks+=['flush_install_restore','calibration_binding']
p=Path(__file__).with_name('convert_main_owner_candidate.py');tree=ast.parse(p.read_text());node=next(n for n in tree.body if getattr(n,'name',None)=='checked_convert')
called=[]
def convert(self,hd,qa,*args,**kw):
    called.append(True);qa['g_scale']=1.;hd['q_fallback']=qa.pop('inject_fallback',False)
    self.inner=NS(suh=torch.ones(2),svh=torch.ones(2),bias=None)
    return .01
ns=dict(torch=torch,math=__import__('math'),audit_lock=threading.Lock(),cal={'total_tokens':4},original_convert=convert)
exec(compile(ast.Module(body=[node],type_ignores=[]),str(p),'exec'),ns)
def fresh(n=4):return dict(H=torch.eye(2),inf_nan=torch.zeros(2,dtype=torch.long),count_dev=torch.tensor(n),count=0)
self=NS(key='fixture');ns['checked_convert'](self,fresh(),{})
for hd,qa in [(fresh(3),{}),(fresh(),{'inject_fallback':True})]:
    try:ns['checked_convert'](self,hd,qa)
    except (AssertionError,RuntimeError):pass
    else:raise AssertionError('Incomplete data/fallback accepted')
assert len(called)==2
checks+=['complete_calibration_count','H_data_fallback_rejected']
r=dict(status='pass',gpu_used=False,checks=checks,identity={str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [Path(runtime.__file__),p,Path(__file__)]})
Path(sys.argv[1]).write_text(json.dumps(r,indent=2));print(json.dumps(r))
