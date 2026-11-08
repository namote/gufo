"""Preflight main Gram shapes and exercise attention against retained evidence."""
import collections,copy,ctypes as C,hashlib,json,os,time
from pathlib import Path
import torch
from exllamav3 import Config,Model
from exllamav3.conversion import convert_model as cm
from exllamav3.modules.quant import fp16,exl3
from exllamav3.modules.linear import flush_capture_H
from exllamav3.loader.safetensors_alt import safe_open
from capture_fast import install,_discarded_capture
from capture_lt import LtGram,make_flush
from disk_state import _read_row
ROOT=Path('/output');OUT=ROOT/'capture-lt-support';OLD=ROOT/'capture-speed'
assert not (OUT/'SUMMARY.json').exists()
assert os.environ['EXL3_HIP_F32OUT_VIA_F16']=='0' and not fp16._f32_via_f16 and not exl3._f32_via_f16
started=time.monotonic()
def write(n,r):
    p=OUT/(n+'.tmp');p.write_text(json.dumps(r,indent=2));p.replace(OUT/n)
def cpu(v):
    if isinstance(v,torch.Tensor):return v.detach().cpu().clone()
    if isinstance(v,dict):return {k:cpu(x) for k,x in v.items()}
    if isinstance(v,list):return [cpu(x) for x in v]
    return copy.deepcopy(v)
def exact(a,b):
    if isinstance(a,torch.Tensor):
        assert a.shape==b.shape and a.dtype==b.dtype
        assert torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,list):
        assert len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b

def diff(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all()
    d=b.double()-a.double()
    return dict(exact=torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8)),relative_l2=float(d.norm()/a.double().norm().clamp_min(1e-30)),max_abs=float(d.abs().max()))

cal=json.loads(Path('/task/c03-input-ids.json').read_text());lengths=sorted({v['tokens'] for v in cal['sequences']}|{33})
inv=json.loads((ROOT/'model-inventory.json').read_text());patterns=[]
for m in inv['modules']:
    if m['component']!='main':continue
    q={v['qmap']:v['in_features'] for v in inv['linears'] if v['component']=='main' and v['qmap'] and (v['key']==m['key'] or v['key'].startswith(m['key']+'.'))}
    pattern=sorted(collections.Counter(q.values()).items())
    assert pattern in ([],[(2560,1)],[(640,513),(2560,2),(6144,1)]),('New qmap grouping',m,pattern)
    if pattern and pattern not in patterns:patterns.append(pattern)
# 512 uniform routed experts + one shared expert; 2 same-dimension qmaps may
# be one group or split by original dtype. Validate actual observed groups below.
pairs=[(1,640),(64,640),(1,2560),(2,2560),(1,6144)]
shapes=[(b,r,d,half) for r in lengths for b,d in pairs for half in (False,True)]
native=LtGram(ROOT/'capture-gram/gram.so','original_f16');preflight=[]
for i,(b,r,d,half) in enumerate(shapes):
    assert time.monotonic()-started<240
    status=C.c_int();found=C.c_int();p=native.lib.gram_lt_create(b,r,d,int(half),C.byref(status),C.byref(found))
    selected=1 if half else 3
    item=dict(batch=b,rows=r,dim=d,half=half,selected=selected,status=status.value,found=found.value)
    preflight.append(item)
    if p:native.lib.gram_lt_destroy(p)
    if not p or status.value!=0 or found.value<=selected:
        write('SUPPORT.json',preflight);raise AssertionError(('Unsupported selected shape',item))
write('SUPPORT.json',preflight)
allowed=set(shapes);observed=set();real_call=native.__call__
def gram(xf,xs,finite):
    b,r,d=xs.shape;sig=(b,r,d,xs.dtype==torch.float16)
    assert sig in allowed,('Unqualified actual shape',sig)
    observed.add(sig)
    return real_call(xf,xs,finite)
flush=make_flush(flush_capture_H,gram)
config=Config.from_directory('/source');model=Model.from_config(config);module=model.modules[6]
assert module.key=='model.language_model.layers.3' and type(module.attn).__name__=='Attention'
with safe_open('/task/c03-input-ids.safetensors') as reader:ids={i:_read_row(reader,f'input_ids.{i}') for i in (4,364)}
inputs={}
for i in (4,364):
    with safe_open(str(OLD/f'attention-input-{i}.safetensors')) as reader:inputs[i]=_read_row(reader,'tensor')
base_H=torch.load(OLD/'baseline-6-H.pt',map_location='cpu',weights_only=True)
base_y=torch.load(OLD/'baseline-6-normal.pt',map_location='cpu',weights_only=True)
records=[]
with torch.inference_mode():
    config.stc.begin_deferred_load()
    try:module.load(torch.device('cuda:0'))
    finally:config.stc.end_deferred_load()
    config.stc.close();restore=install(cm,automatic=False)
    try:
        for repeat in range(2):
            H={};ys=[];prs=[]
            for i,length in [(4,33),(4,512),(364,8192)]:
                assert time.monotonic()-started<540
                x=inputs[i][:,:length].contiguous().cuda();before=cpu(x)
                params=dict(input_ids=ids[i][:,:length],attn_mode='flash_attn_nc',quant_preserve={},capture=H,activate_all_experts=True)
                with _discarded_capture(module,params):y=module.forward(module.prepare_for_device(x,params),params)
                flush(params);exact(before,cpu(x));assert torch.isfinite(y).all();del y
                exact({},cpu(params['quant_preserve']))
                normal=dict(input_ids=ids[i][:,:length],attn_mode='flash_attn_nc',quant_preserve={})
                y=module.forward(module.prepare_for_device(x,normal),normal)
                ys.append(cpu(y));prs.append(cpu(normal['quant_preserve']));exact(before,cpu(x));del x,y,params,normal
            saved=cpu(H);torch.save(dict(H=saved,outputs=ys,preserves=prs),OUT/f'attention-{repeat}.pt')
            assert saved.keys()==base_H.keys() and len(saved)==516
            differences={}
            for key,base in base_H.items():
                actual=saved[key]
                exact({k:v for k,v in base.items() if k!='H'},{k:v for k,v in actual.items() if k!='H'})
                assert int(actual['count_dev'])==8737 and int(actual['inf_nan'].sum())==0
                differences[key]=diff(base['H'],actual['H'])
            exact(base_y['preserves'],prs)
            assert len(base_y['outputs'])==len(ys)==3
            refs=[diff(a,b) for a,b in zip(base_y['outputs'],ys)]
            write(f'attention-{repeat}-DIFFERENCES.json',dict(H=differences,refs=refs))
            if repeat:
                first=torch.load(OUT/'attention-0.pt',map_location='cpu',weights_only=True)
                exact(first,dict(H=saved,outputs=ys,preserves=prs));del first
            records.append(dict(repeat=repeat,H_max_relative_l2=max(v['relative_l2'] for v in differences.values()),refs=refs,repeat_exact=bool(repeat)))
            del H,saved,ys,prs
    finally:restore();native.close();module.unload();config.stc.close()
write('SUMMARY.json',dict(status='main_shapes_attention_qualified',preflight_shape_count=len(shapes),calibration_sha256=cal['sha256'],patterns=patterns,
    allowed_shapes=shapes,observed_attention_shapes=sorted(observed),attention=records,cache=native.cache.stats,seconds=time.monotonic()-started,
    native_sha256=hashlib.sha256((ROOT/'capture-gram/gram.so').read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(Path('/task/capture_lt.py').read_bytes()).hexdigest(),
    limits='Heuristic creation for all calibrated lengths is support discovery, not execution/quality proof for every shape. Attention actual33/512/8192 and priorGDN evidence qualify bounded rollout; full-model accuracy/performance remains final.',production_changed=False))
