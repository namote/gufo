import json,torch
from pathlib import Path
R=Path('/output/capture-owner');a=torch.load(R/'long-first.pt',weights_only=True);b=torch.load(R/'fixed-long-first.pt',weights_only=True);c=torch.load(R/'short-first.pt',weights_only=True)
def same(x,y):return torch.equal(x.contiguous().reshape(-1).view(torch.uint8),y.contiguous().reshape(-1).view(torch.uint8))
def diff(x,y):
 d=x.double()-y.double();return dict(exact=same(x,y),relative_l2=float(d.norm()/y.double().norm().clamp_min(1e-30)),max_abs=float(d.abs().max()))
assert all(int(v['count_dev'])==3729 for v in b['H'].values())
refs={str(i):dict(candidate_vs_broken=same(a['refs'][i],b['refs'][i]),candidate_vs_valid=same(b['refs'][i],c['refs'][i])) for i in (0,4)}
valid={k:diff(v['H'],c['H'][k]['H']) for k,v in b['H'].items()};old={k:diff(v['H'],a['H'][k]['H']) for k,v in b['H'].items()}
xx=torch.cat([b['inputs'][i].reshape(-1,2560)[:,:16].double() for i in (0,4)]);oracle=xx.T@xx
r=dict(status='stored_owner_audit',refs=refs,all516_counts=3729,candidate_vs_valid_short_first_all_H_exact=all(v['exact'] for v in valid.values()),candidate_vs_valid=valid,changes_vs_incomplete_control={k:v for k,v in old.items() if not v['exact']},FP64_input_H=diff(b['H']['block.attn.input']['H'][:16,:16],oracle))
(R/'AUDIT.json').write_text(json.dumps(r,indent=2));print(json.dumps({k:v for k,v in r.items() if k!='candidate_vs_valid'},indent=2))
