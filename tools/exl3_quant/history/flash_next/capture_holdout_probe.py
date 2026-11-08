"""Evaluate retained full/quarter packed projections on fresh fixed inputs.

No calibration capture, quantization, or production checkpoint writes.
"""
import hashlib
import json
import math
from pathlib import Path
import time
import torch
from exllamav3 import Config,Model
from exllamav3.modules import Linear
from exllamav3.modules.quant import fp16,exl3
from exllamav3.modules.quant.exl3 import LinearEXL3
from exllamav3.conversion.ngram import prepare_ngram_table_for_conversion
from exllamav3.loader.safetensors_alt import safe_open
from disk_state import _read_row
from capture_inputs import capture_inputs,InputsCaptured

OUT=Path('/output/capture-quarter-holdout')
QUARTER=Path('/output/capture-profile-quarter/quarter')
FULL=Path('/output/work/qtensors/model.language_model.layers.0.safetensors')
assert not (OUT/'SUMMARY.json').exists() and not (OUT/'RESULTS.json').exists()
assert not fp16._f32_via_f16 and not exl3._f32_via_f16
mounts=[line.split() for line in Path('/proc/self/mountinfo').read_text().splitlines()]
for path in ('/output/work','/output/pack','/output/capture-profile-quarter'):
    assert any(p[4]==path and 'ro' in p[5].split(',') for p in mounts)
data=json.loads((OUT/'DATA.json').read_text())
assert data['status']=='prepared' and data['gpu_used'] is False
assert hashlib.sha256((OUT/'input-ids.safetensors').read_bytes()).hexdigest()==data['input_sha256']
assert hashlib.sha256((OUT/'TEXTS.json').read_bytes()).hexdigest()==data['texts_sha256']
assert json.loads((QUARTER/'SUMMARY.json').read_text())['status']=='quarter_projection_screen_complete'
def write(name,value):
    temp=OUT/(name+'.tmp');temp.write_text(json.dumps(value,ensure_ascii=False,indent=2));temp.replace(OUT/name)
def stamp(path):
    s=path.stat();return [s.st_size,s.st_mtime_ns,s.st_ino]
protected={str(p):stamp(p) for p in (FULL,Path('/output/work/ckpt/job.json'),Path('/output/pack/ngram_embedding.safetensors'))}
started=time.monotonic();config=Config.from_directory('/source');model=Model.from_config(config)
ngram=prepare_ngram_table_for_conversion(dict(out_dir='/output/pack',ngram_file='/output/pack/ngram_embedding.safetensors'),config,model)
assert ngram=='/output/pack/ngram_embedding.safetensors'
assert model.modules[0].__class__.__name__=='Embedding' and model.modules[1].__class__.__name__=='ExpandStreams'
block=model.modules[2];assert block.key=='model.language_model.layers.0'
results=[]
with torch.inference_mode():
    # Match the source embedding/HC path used to build the archived idx2 state.
    for module in model.modules[:2]:module.load(torch.device('cuda:0'))
    deferred=block.can_defer_load()
    if deferred:config.stc.begin_deferred_load()
    try:block.load(torch.device('cuda:0'))
    finally:
        if deferred:config.stc.end_deferred_load()
    config.stc.close()
    bits={'linear_attn.out_proj':6,'mlp.experts.0.down_proj':4}
    selected={m.key:m for m in block if isinstance(m,Linear) and any(m.key.endswith('.'+s) for s in bits)}
    assert len(selected)==2
    terminal=next(k for k in selected if k.endswith('.experts.0.down_proj'))
    packs={};rotations={};pack_hashes={}
    with safe_open(str(FULL)) as reader:
        for key,linear in selected.items():
            full={n.removeprefix(key+'.'):_read_row(reader,n) for n in reader.keys() if n.startswith(key+'.')}
            path=QUARTER/(key+'.pt');pack_hashes[str(path)]=hashlib.sha256(path.read_bytes()).hexdigest()
            raw=torch.load(path,map_location='cpu',weights_only=True)
            assert raw and all(n.startswith(key+'.') for n in raw)
            quarter={n.removeprefix(key+'.'):v for n,v in raw.items()}
            packs[key]={}
            for label,values in (('full',full),('quarter',quarter)):
                def get(name):
                    value=values.get(name)
                    return value if value is None or name in ('mul1','mcg') else value.to('cuda:0')
                inner=LinearEXL3(config,linear.in_features,linear.out_features,None,get('su'),get('sv'),get('suh'),get('svh'),get('trellis'),get('mcg'),get('mul1'),get('bias'),linear.out_dtype,key=key)
                assert inner.K==next(v for s,v in bits.items() if key.endswith('.'+s))
                for name in ('suh','svh','bias'):
                    v=getattr(inner,name);assert v is None or torch.isfinite(v).all()
                packs[key][label]=inner
            rotations[key]=all(torch.equal(torch.sign(getattr(packs[key]['full'],n)),torch.sign(getattr(packs[key]['quarter'],n))) for n in ('suh','svh'))
            assert rotations[key]
    with safe_open(str(OUT/'input-ids.safetensors')) as reader:
        for i,row in enumerate(data['rows']):
            assert time.monotonic()-started<500
            ids=_read_row(reader,f'input_ids.{i}');assert ids.shape==(1,row['tokens'])
            assert hashlib.sha256(ids.numpy().tobytes()).hexdigest()==row['token_sha256']
            ids=ids.long();x=ids.clone()
            for module in model.modules[:2]:
                params={'input_ids':ids}
                x=module.forward(module.prepare_for_device(x,params),params)
            assert x.shape==(1,row['tokens'],4,2560) and x.dtype==torch.float32
            params=dict(input_ids=ids,attn_mode='flash_attn_nc',capture={},activate_all_experts=True,quant_preserve={})
            inputs={}
            with capture_inputs(selected,terminal,inputs):
                try:block.forward(block.prepare_for_device(x,params),params)
                except InputsCaptured:pass
                else:raise AssertionError('Expected input observer boundary')
            assert params['capture']=={} and '_cap_pending' not in params
            for key,linear in selected.items():
                activation=inputs[key].reshape(-1,linear.in_features)
                positions=torch.linspace(0,len(activation)-1,min(256,len(activation)),device=activation.device).long()
                activation=activation[positions].unsqueeze(0).contiguous()
                reference=linear.inner.forward(activation,{}).float()
                assert torch.isfinite(reference).all()
                ref_norm=torch.linalg.vector_norm(reference);assert torch.isfinite(ref_norm) and ref_norm>0
                entry=dict(id=row['id'],domain=row['domain'],tokens=row['tokens'],evaluated_tokens=len(positions),key=key)
                for label,inner in packs[key].items():
                    y=inner.forward(activation,{}).float();assert torch.isfinite(y).all()
                    error=float(torch.linalg.vector_norm(y-reference)/ref_norm)
                    per_token=torch.linalg.vector_norm(y-reference,dim=-1)/torch.linalg.vector_norm(reference,dim=-1).clamp_min(1e-20)
                    assert math.isfinite(error) and torch.isfinite(per_token).all()
                    entry[label+'_relative_error']=error
                    entry[label+'_p99_token_error']=float(torch.quantile(per_token.flatten(),.99))
                entry['error_ratio']=entry['quarter_relative_error']/entry['full_relative_error']
                results.append(entry);write('RESULTS.json',results)
            write('PROGRESS.json',dict(rows=i+1,total=len(data['rows']),seconds=time.monotonic()-started))
    for module in model.modules[:3]:module.unload()
    summary=[]
    for key in selected:
        entries=[r for r in results if r['key']==key];mean=sum(r['error_ratio'] for r in entries)/len(entries);worst=max(r['error_ratio'] for r in entries)
        summary.append(dict(key=key,mean_document_error_ratio=mean,max_document_error_ratio=worst,screen_flag=mean>1.02 or worst>1.05))
    assert all(stamp(Path(p))==v for p,v in protected.items())
    assert all(hashlib.sha256(Path(p).read_bytes()).hexdigest()==v for p,v in pack_hashes.items())
    write('SUMMARY.json',dict(status='holdout_operator_screen_complete',results=summary,rotation_signs_same=rotations,
        elapsed_seconds=time.monotonic()-started,data_sha256=hashlib.sha256((OUT/'DATA.json').read_bytes()).hexdigest(),
        script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),helper_sha256=hashlib.sha256(Path('/task/capture_inputs.py').read_bytes()).hexdigest(),
        pack_hashes=pack_hashes,protected_stats=protected,production_changed=False,full_model_quality_validated=False,
        scope='Eight fixed fresh diagnostic documents, including synthetic Korean4K and local C++8K; original layer0 projection references, at most256 evaluation positions each. No H capture/requantization/full-model run. NgramK8 registered, PLE occurs after this layer so no lookup used here.'))
