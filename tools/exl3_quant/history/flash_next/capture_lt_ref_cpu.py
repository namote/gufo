"""CPU-only inspection of retained failed Lt loop; never recompute a baseline."""
import json
from pathlib import Path
import torch

torch.set_num_threads(2)
p=Path('/output/capture-lt-pipeline')
a=torch.load('/output/capture-pipeline/control.pt',map_location='cpu',weights_only=True)
b=torch.load(p/'warm-f32-f32/failed-numeric.pt',map_location='cpu',weights_only=True)
def eq(x,y):
    if isinstance(x,torch.Tensor):
        return x.shape==y.shape and x.dtype==y.dtype and torch.equal(x.contiguous().reshape(-1).view(torch.uint8),y.contiguous().reshape(-1).view(torch.uint8))
    if isinstance(x,dict):return x.keys()==y.keys() and all(eq(x[k],y[k]) for k in x)
    if isinstance(x,(list,tuple)):return type(x)==type(y) and len(x)==len(y) and all(eq(u,v) for u,v in zip(x,y))
    return x==y

def diff(x,y):
    d=y.double()-x.double()
    return dict(shape=list(x.shape),dtype=str(x.dtype),finite=bool(torch.isfinite(y).all()),exact=eq(x,y),changed_values=int(torch.count_nonzero(x!=y)),max_abs=float(d.abs().max()),relative_l2=float(d.norm()/x.double().norm().clamp_min(1e-30)))

r=dict(refs={k:diff(v,b['ref_states'][k]) for k,v in a['ref_states'].items()},other={k:eq(v,b[k]) for k,v in a.items() if k not in ('capture_H','ref_states')},H_count=len(b['capture_H']),H_metadata_exact=all(eq({k:v for k,v in x.items() if k!='H'},{k:v for k,v in b['capture_H'][name].items() if k!='H'}) for name,x in a['capture_H'].items()),H_first5={k:diff(v['H'],b['capture_H'][k]['H']) for k,v in list(a['capture_H'].items())[:5]})
(p/'CPU-FAILURE-ANALYSIS.json').write_text(json.dumps(r,indent=2))
print(json.dumps(r,indent=2))
