"""CPU source-teacher orientation/precision and error-metric contracts."""
import ast,hashlib,json,sys
from pathlib import Path
from types import SimpleNamespace as NS
import torch
p=Path(__file__).with_name('capture_lt_quality_probe.py');tree=ast.parse(p.read_text())
ns={'torch':torch}
exec(compile(ast.Module(body=[n for n in tree.body if getattr(n,'name',None) in ('source_weight','error_metrics')],type_ignores=[]),str(p),'exec'),ns)
raw=torch.tensor([[1.,2.],[3.,4.],[1e-8,6.]],dtype=torch.bfloat16)
class Collection:
    def __init__(self,direct=True):self.direct=direct;self.calls=[]
    def has_tensor(self,key):return key==('test.weight' if self.direct else 'fused')
    def get_tensor(self,key,device,**kwargs):
        assert device=='cpu' and kwargs['allow_bf16'] is True and kwargs['no_defer'] is True
        self.calls.append((key,kwargs));return raw.clone()
def linear(direct=True):
    return NS(config=NS(stc=Collection(direct)),weight_scale=1,key='test',inner=NS(bias=None,get_weight_tensor=lambda:raw.half()),in_features=3,out_features=2,transposed_load=True,fkey='fused',fidx=0,transpose_fused_weights=True,frange=None,frange_dim=0,ftranspose_after_load=False)
checks=[]
for direct in (True,False):
    layer=linear(direct);w,route=ns['source_weight'](layer)
    assert w.dtype==torch.bfloat16 and torch.equal(w,raw) and not route['source_values_exact']
    checks.append('preserve_BF16_and_report_FP16_loading_'+str(direct))
bad=linear();bad.inner.get_weight_tensor=lambda:torch.zeros_like(raw,dtype=torch.float16)
try:ns['source_weight'](bad)
except AssertionError:pass
else:raise AssertionError('loader mismatch missed')
checks.append('loader_relation_checked')
y=torch.tensor([[2.,0.],[0.,2.]]);ref=torch.eye(2)
r=ns['error_metrics'](y,ref);assert r==dict(relative_error=1.,sampled_p99=1.)
checks.append('relative_error_and_p99_independent_handcase')
for y in [torch.ones(3,3),torch.full((2,2),float('nan'))]:
    try:ns['error_metrics'](y,ref)
    except AssertionError:pass
    else:raise AssertionError('invalid comparison accepted')
checks.append('shape_and_nonfinite_rejected')
r=dict(status='pass',gpu_used=False,checks=checks,probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(r,indent=2));print(json.dumps(r))
