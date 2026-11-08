"""Recover only lost first-five terminal-head comparisons; never capture/quantize.

Read-only idx52 and completed K8 weights. Directly persist five terminal logits
in private output for separately reviewed CPU checkpoint promotion. No promotion,
main restart, compilation, source mutation or asynchronous state queue.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time
from row_checkpoint import open_rows, tensor_header, sync_directory

LENGTHS = [3217, 7793, 4655, 3671, 512]
VOCAB = 248320


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''): h.update(b)
    return h.hexdigest()


def stat_id(p):
    s=Path(p).stat()
    return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]


def write_json(p,d):
    q=p.with_suffix('.pending')
    with q.open('x') as f: json.dump(d,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(q,p);sync_directory(p.parent)


def row_identity(source):
    paths,infos=open_rows(source)
    assert len(paths)==2309 and sum(s[1] for _,s in infos)==1999496
    assert all(d=='F16' and len(s)==3 and s[0]==1 and s[2]==2560 for d,s in infos)
    assert [s[1] for _,s in infos[:5]]==LENGTHS
    records=[]
    for p,info in zip(paths,infos,strict=True):
        s=p.stat();records.append([s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns,info])
    return paths,infos,dict(manifest_sha256=sha(Path(source).with_suffix('.rows.json')),
        rows_stat_sha256=hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest())


def check_output_header(path,index):
    h=tensor_header(path)
    assert set(h)=={'tensor'} and h['tensor']['dtype']=='F16'
    assert h['tensor']['shape']==[1,LENGTHS[index],VOCAB]
    return h


def validate_metrics(m):
    assert m['finite_raw'] is True and m['finite_quant'] is True
    assert all(math.isfinite(m[k]) for k in ('rfn','cos','sqnr'))
    assert m['rfn']>=0


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--approval',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    c=json.loads(a.approval.read_text())
    assert c['status']=='reviewed_head_forward_recovery'
    assert c['candidate_sha256']==sha(__file__)
    assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16')=='0'
    assert os.environ.get('EXL3_SKIP_SIDE_MODELS')=='1'
    work=Path('/output/work');ckpt=work/'ckpt';packed=work/'qtensors/lm_head.safetensors'
    assert json.loads((ckpt/'job.json').read_text())=={'next_module_idx':52,'q_strategy':None,'bad_rows':[]}
    paths,infos,state_id=row_identity(ckpt/'state.safetensors')
    assert state_id==c['state_identity']
    audit_path=Path('/output/head-recovery/AUDIT.json')
    assert str(audit_path) in c['files_sha256']
    audit=json.loads(audit_path.read_text())
    assert audit['status']=='head_pack_and_source52_audited'
    assert audit['head_pack_sha256']==c['packed_sha256']
    assert stat_id(packed)==audit['head_pack_stat'] and packed.stat().st_size==636201289
    assert audit['rows_identity']==state_id
    for raw,expected in c['files_sha256'].items(): assert sha(raw)==expected,raw
    assert bool(os.statvfs(work).f_flag & os.ST_RDONLY)
    a.out.mkdir(parents=True,exist_ok=True)
    binding={k:c[k] for k in ('candidate_sha256','state_identity','packed_sha256','files_sha256')}
    bindfile=a.out/'BINDING.json'
    if bindfile.exists(): assert json.loads(bindfile.read_text())==binding
    else: write_json(bindfile,binding)
    import torch
    from safetensors.torch import load_file,save_file
    from exllamav3.conversion import convert_model as cm
    from exllamav3.modules import Linear
    from exllamav3.modules.quant import fp16,exl3
    assert not fp16._f32_via_f16 and not exl3._f32_via_f16
    # Fail closed if any future edit accidentally enters H/quantization/main.
    def forbidden(*args,**kwargs): raise RuntimeError('Head recovery cannot capture or quantize')
    Linear.capture_H=forbidden;Linear.convert_exl3=forbidden
    cm.main=forbidden;cm.quantize_linears_parallel=forbidden
    args=json.loads((work/'args.json').read_text())
    assert args['head_bits']==8 and args['cal_rows']==2309 and args['cal_cols']==8192
    assert args['in_dir']=='/source' and args['codebook']=='mul1'
    assert args['apply_out_scales'] is True
    args.update(vision_bits=16,max_module=52,image_dump=False,verbose=False)
    config,model,mtp,vision,tokenizer,use_ref=cm.get_base_model(args)
    assert len(model.modules)==53 and model.modules[52].key=='lm_head' and vision is None
    strategy,bpw=cm.create_q_strategy_from_recipe(model,mtp,config,args['recipe_strategy'],8,args['mtp_bits'],vision_model=None,vision_bpw=16)
    args['final_bits']=round(bpw,2)
    # Second model metadata owns a distinct raw loader/cache; only head loads GPU weights.
    qconfig,qmodel,_,_,_,_=cm.get_base_model(dict(args))
    raw=model.modules[52];quant=qmodel.modules[52]
    assert raw.num_slices==quant.num_slices==1
    device=torch.device('cuda:0');raw.load(device);config.stc.close()
    tensors=load_file(str(packed));qconfig.stc.set_new_tensors(tensors)
    quant.load(device,source=tensors);qconfig.stc.set_new_tensors(None);del tensors
    ids=load_file(str(ckpt/'original_input_ids.safetensors'))
    calids=load_file('/task/c03-input-ids.safetensors')
    assert set(ids)=={f'tensor.{i}' for i in range(2309)}
    assert set(calids)=={f'input_ids.{i}' for i in range(2309)}
    for i,(_,shape) in enumerate(infos):
        assert tuple(ids[f'tensor.{i}'].shape)==(1,shape[1])
        assert torch.equal(ids[f'tensor.{i}'],calids[f'input_ids.{i}'])
    started=time.monotonic();results=[]
    try:
        with torch.inference_mode():
            for i in range(5):
                receipt=a.out/f'ROW-{i}.json';output=a.out/f'{i}.safetensors'
                if receipt.exists():
                    entry=json.loads(receipt.read_text());assert entry['binding_sha256']==sha(bindfile)
                    assert entry['index']==i and entry['output_sha256']==sha(output)
                    check_output_header(output,i);validate_metrics(entry);results.append(entry);continue
                assert not output.exists() and not output.with_suffix('.pending').exists(), 'Partial row requires audited recovery, never overwrite'
                t=time.monotonic();x=load_file(str(paths[i]))['tensor']
                params={'attn_mode':'flash_attn_nc','input_ids':ids[f'tensor.{i}'],'quant_preserve':{}}
                rawx=raw.prepare_for_device(x,params);ref=raw.forward(rawx,params)
                params={'attn_mode':'flash_attn_nc','input_ids':ids[f'tensor.{i}'],'quant_preserve':{}}
                quantx=quant.prepare_for_device(x,params);y=quant.forward(quantx,params)
                assert ref.dtype==y.dtype==torch.float16 and list(y.shape)==[1,LENGTHS[i],VOCAB]
                entry=dict(index=i,binding_sha256=sha(bindfile),finite_raw=bool(torch.isfinite(ref).all()),finite_quant=bool(torch.isfinite(y).all()))
                write_json(a.out/f'ROW-{i}-NUMERIC.json',entry)
                assert entry['finite_raw'] and entry['finite_quant'],entry
                # Use pinned converter's exact whole-row error implementation, not chunks.
                torch.cuda.synchronize();cpu=y.cpu().contiguous();refcpu=ref.cpu().contiguous()
                del ref,y,x,rawx,quantx
                # DiskRows originally compares CPU state and CPU reference. Preserve reductions.
                rfn,cos,sq=cm.get_state_error(cpu,refcpu);del refcpu
                entry.update(rfn=float(rfn),cos=float(cos),sqnr=float(sq))
                write_json(a.out/f'ROW-{i}-NUMERIC.json',entry);validate_metrics(entry)
                pending=output.with_suffix('.pending');save_file({'tensor':cpu},str(pending));del cpu
                with pending.open('rb') as f:os.fsync(f.fileno())
                check_output_header(pending,i);os.replace(pending,output);sync_directory(output.parent)
                entry.update(output_sha256=sha(output),output_bytes=output.stat().st_size,seconds=time.monotonic()-t)
                write_json(receipt,entry);results.append(entry);print('HEAD_RECOVERY_ROW',json.dumps(entry),flush=True)
    finally:
        raw.unload();quant.unload();config.stc.close();qconfig.stc.close()
    assert row_identity(ckpt/'state.safetensors')[2]==state_id
    assert stat_id(packed)==audit['head_pack_stat']
    assert json.loads((ckpt/'job.json').read_text())=={'next_module_idx':52,'q_strategy':None,'bad_rows':[]}
    write_json(a.out/'MAIN-COMPILE-ARGS.json',args)
    write_json(a.out/'SUMMARY.json',dict(status='head_forward_complete_pending_independent_review',
        source_index=52,proposed_next_index=53,head_capture_repeated=False,head_quantization_repeated=False,
        rows=2309,tokens=1999496,forward_rows=5,retained_input_rows=2304,state_identity=state_id,
        packed_sha256=c['packed_sha256'],binding_sha256=sha(bindfile),results=results,
        mean_rfn=sum(r['rfn'] for r in results)/5,mean_cos=sum(r['cos'] for r in results)/5,
        mean_sqnr=sum(r['sqnr'] for r in results)/5,seconds=time.monotonic()-started,
        compile_args_sha256=sha(a.out/'MAIN-COMPILE-ARGS.json'),checkpoint_promoted=False))

if __name__=='__main__':main()
