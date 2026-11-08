"""First-block c02 operator screening; final model quality remains unqualified."""
import copy
import json
import math
import os
from pathlib import Path
import time
import types
import torch
from safetensors.torch import load_file
from exllamav3 import Config,Model
from exllamav3.modules import Linear
from exllamav3.modules.linear import flush_capture_H
from exllamav3.conversion.convert_model import make_quant_args
from exllamav3.modules.quant import fp16 as fp16_impl, exl3 as exl3_impl

screen_out=Path(os.environ.get('GUFO_BANK_SCREEN_OUT','/output/bank-pilot'))
screen_out.mkdir(exist_ok=True)
assert fp16_impl._f32_via_f16==exl3_impl._f32_via_f16

torch.set_grad_enabled(False);torch.manual_seed(0)
config=Config.from_directory('/source');model=Model.from_config(config)
ids=load_file('/task/c02-input-ids.safetensors')['input_ids'][:4].long()
state=[]
for top in model.modules[:2]:
    top.load(torch.device('cuda:0'))
    if not state:state=[r.unsqueeze(0) for r in ids]
    for i in range(len(state)):
        p={'input_ids':ids[i:i+1]}
        state[i]=top.forward(top.prepare_for_device(state[i],p),p).cpu()
    top.unload()
block=model.modules[2]
assert block.key=='model.language_model.layers.0'
defer=block.can_defer_load()
if defer:config.stc.begin_deferred_load()
try:block.load(torch.device('cuda:0'))
finally:
    if defer:config.stc.end_deferred_load()
suffixes={'linear_attn.in_proj_qkv':(5,6),'linear_attn.in_proj_z':(5,6),
          'linear_attn.out_proj':(6,7),'mlp.shared_expert.gate_proj':(6,8),
          'mlp.shared_expert.up_proj':(6,8),'mlp.shared_expert.down_proj':(6,8),
          'mlp.experts.0.down_proj':(4,5)}
selected={m.key:m for m in block if isinstance(m,Linear) and any(m.key.endswith('.'+s) for s in suffixes)}
assert len(selected)==len(suffixes)
inputs={k:[] for k in selected}
for key,m in selected.items():
    original=m.capture_H
    def capture(self,x,params,_original=original,_key=key):
        inputs[_key].append(x.detach().reshape(-1,x.shape[-1]).cpu().clone())
        return _original(x,params)
    m.capture_H=types.MethodType(capture,m)
H={};start=time.monotonic()
for i in range(len(state)):
    p={'attn_mode':'flash_attn_nc','capture':H,'activate_all_experts':True,
       'input_ids':ids[i:i+1],'quant_preserve':{}}
    y=block.forward(block.prepare_for_device(state[i],p),p)
    assert torch.isfinite(y).all()
    flush_capture_H(p)
print('Captured first-block Hessians',len(H),'seconds',time.monotonic()-start,flush=True)
results=[]
for key,m in selected.items():
    bits=next(v for s,v in suffixes.items() if key.endswith('.'+s))
    original=m.inner
    w=original.get_weight_tensor().float().clone()
    x=torch.cat(inputs[key]).to(w.device).float()
    assert torch.isfinite(x).all() and torch.isfinite(w).all()
    assert torch.isfinite(H[m.qmap]['H']).all()
    expected=x@w
    assert torch.linalg.vector_norm(expected)>0
    for K in bits:
        qa=make_quant_args({'apply_out_scales':True,'codebook':'mul1','work_dir':str(screen_out)},2,K,[0])
        t=time.monotonic()
        try:
            proxy,wq=m.convert_exl3(copy.deepcopy(H[m.qmap]),qa,return_weight_q=True)
            for name in ('suh','svh','bias'):
                value=getattr(m.inner,name)
                assert value is None or torch.isfinite(value).all(),(key,name)
            actual=x@wq.to(w.device).float()
            rel=float(torch.linalg.vector_norm(actual-expected)/torch.linalg.vector_norm(expected))
            packed=m.inner.forward(x.to(inputs[key][0].dtype).unsqueeze(0).contiguous(),{}).reshape_as(expected).float()
            packed_rel=float(torch.linalg.vector_norm(packed-expected)/torch.linalg.vector_norm(expected))
            finite=bool(torch.isfinite(wq).all() and torch.isfinite(actual).all() and torch.isfinite(packed).all())
            row=dict(key=key,K=K,proxy_error=proxy,activation_relative_error=rel,finite=finite,
                     packed_activation_relative_error=packed_rel,q_fallback=qa.get('q_fallback'),seconds=time.monotonic()-t)
            results.append(row)
            (screen_out/'RESULTS.json').write_text(json.dumps(results,indent=2))
            print(json.dumps(row),flush=True)
            assert finite and math.isfinite(proxy) and rel<0.05 and packed_rel<0.05 and not qa.get('q_fallback'),row
        finally:
            m.inner=original
    del x,w,expected
block.unload()
(screen_out/'SUMMARY.json').write_text(json.dumps(dict(
    status='operator_screen_pass',calibration_rows=4,tokens=2048,results=len(results),
    fp32_output_shortcut=fp16_impl._f32_via_f16,
    peak_gpu_allocated_bytes=torch.cuda.max_memory_allocated(),
    scope='first block operator screening only; no full model noninferiority claim'),indent=2))
