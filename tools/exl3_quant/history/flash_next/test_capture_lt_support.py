"""CPU support-grid and integrity contracts; actual native support is GPU-only."""
import ast,collections,copy,json,hashlib,sys
from pathlib import Path
import torch
p=Path(__file__).with_name('capture_lt_support_probe.py');tree=ast.parse(p.read_text());ns=dict(collections=collections,Path=Path,json=json,ROOT=Path('/output'),torch=torch,copy=copy)
nodes=[];capture=False
for n in tree.body:
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='cal' for t in n.targets):capture=True
    if isinstance(n,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='native' for t in n.targets):break
    if capture:nodes.append(n)
exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
shapes=set(ns['shapes']);assert len(shapes)==len(ns['lengths'])*10
assert {(1,512,2560,True),(2,8192,2560,False),(64,512,640,True),(1,8192,640,False),(1,33,6144,True)}<=shapes
assert all(b*r*d>0 and r<=8192 for b,r,d,h in shapes)
nodes=[n for n in tree.body if getattr(n,'name',None) in ('exact','diff')];exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),ns)
a=torch.eye(2);assert ns['diff'](a,a)['exact'] and not ns['diff'](a,a+.01)['exact']
for b in (a+.01,a.double(),torch.ones(3,3)):
    try:ns['exact'](a,b)
    except AssertionError:pass
    else:raise AssertionError('exact integrity missed')
r=dict(status='pass',gpu_used=False,shape_count=len(shapes),lengths=len(ns['lengths']),patterns=ns['patterns'],checks=['qmap_group_and_tail_grid','original_dtype_routes','numeric_diff_recorded','integrity_exact'],probe_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(r,indent=2));print(json.dumps(r))
