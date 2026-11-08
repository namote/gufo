"""Real-input Gufo norm/raw/packed FC comparison; no quantization, no qualification.

Raw stage precedes two-row input quantization; packed stage consumes its immutable
mtp.safetensors and reuses the same four real next-token positions. Caller owns GPU.
"""
import argparse,ctypes as C,json,os,time
from pathlib import Path
from mtp_calibration_candidate import digest,write_json
from mtp_calibration_gufo import SEMANTICS,rms_norm_f32,raw_fc_f32,packed_fc_f32,gufo_input_rotation

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['raw','packed'],required=True)
    ap.add_argument('--source-checkpoint',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--library',type=Path,required=True);ap.add_argument('--input-pack',type=Path)
    a=ap.parse_args();assert os.environ.get('GUFO_MTP_LEASE_VERIFIED')=='1'
    assert (a.stage=='packed')==(a.input_pack is not None)
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_file,save_file
    from exllamav3 import Config,Model
    from exllamav3.conversion import convert_model as cm
    from mtp_borrowed_rows import BorrowedRows as DiskRows
    out=a.output;out.mkdir();(out/'scratch').mkdir();start=time.monotonic()
    lib=C.CDLL(str(a.library));P=C.c_void_p;U=C.c_uint32;I=C.c_int;L=C.c_uint64
    lib.mtpg_error.restype=C.c_char_p;lib.mtpg_abi_version.restype=I
    assert lib.mtpg_abi_version()==1
    signatures={'norm':[P,P,P,U,U,C.c_float,P],'raw':[P,I,P,P,U,U,U,P],
                'rotate':[P,P,P,U,U,P],'packed':[P,P,P,P,P,U,U,U,I,I,P,L,P,L,P],
                'cpu_dequant':[P,U,U,I,I,P],'cpu_matvec':[P,P,P,P,P,U,U,U,I,I]}
    for name,signature in signatures.items():
        fn=getattr(lib,'mtpg_'+name);fn.argtypes=signature;fn.restype=I
    def call(name,*args):
        rc=getattr(lib,'mtpg_'+name)(*args)
        if rc:raise RuntimeError((name,rc,lib.mtpg_error().decode()))
    def compare(ref,y):
        assert torch.isfinite(ref).all() and torch.isfinite(y).all()
        d=y.double()-ref.double()
        return {'relative_l2':float(d.norm()/ref.double().norm().clamp_min(1e-30)),
                'max_abs':float(d.abs().max()),'reference_rms':float(ref.double().square().mean().sqrt()),
                'candidate_rms':float(y.double().square().mean().sqrt()),'exact':torch.equal(ref.view(torch.uint8),y.view(torch.uint8))}
    source=a.source_checkpoint.resolve();assert json.loads((source/'job.json').read_text())=={'next_module_idx':51,'q_strategy':None,'bad_rows':[]}
    protected={name:digest(source/name) for name in ('job.json','state.rows.json','original_input_ids.safetensors')}
    cal=json.loads(Path('/task/c03-input-ids.json').read_text())
    ids=load_file('/task/c03-input-ids.safetensors');assert digest('/task/c03-input-ids.safetensors')==cal['sha256']
    sourceids=cm.load_tensor('ckpt/original_input_ids.safetensors',{'work_dir':str(source.parent)})
    rows=DiskRows(source/'state.safetensors',out/'scratch');samples=[];token_ids=[];selection=[];source_values={}
    for length in (512,8192):
        index=next(i for i,s in enumerate(cal['sequences']) if s['tokens']==length)
        state=rows[index];source_values[index]=state.clone()
        assert state.dtype==torch.float32 and state.shape==(1,length,4,2560)
        assert torch.equal(sourceids[index],ids[f'input_ids.{index}'].long())
        for pos in (0,length-2):
            samples.append(state[0,pos]);token_ids.append(int(sourceids[index][0,pos+1]));selection.append([index,pos,pos+1])
    hidden=torch.stack(samples).flatten(-2).cuda()
    mapping=json.loads(Path('/source/model.safetensors.index.json').read_text())['weight_map']
    def source_tensor(key):
        with safe_open(str(Path('/source')/mapping[key]),framework='pt',device='cpu') as f:return f.get_tensor(key)
    key='model.language_model.embed_tokens.weight'
    with safe_open(str(Path('/source')/mapping[key]),framework='pt',device='cpu') as f:
        tensor=f.get_slice(key);embedding=torch.stack([tensor[t:t+1][0] for t in token_ids]).float().cuda()
    cfg=Config.from_directory('/source');model=Model.from_config(cfg,component='mtp');module=model.input_layer
    if a.input_pack:cfg.stc.set_new_tensors(load_file(str(a.input_pack)))
    module.load(torch.device('cuda:0'));stream=torch.cuda.current_stream().cuda_stream
    receipt={'status':'running','stage':a.stage,'semantics':SEMANTICS,'selected_positions':selection,
             'source_identity':protected,'native_sha256':digest(a.library),'code_sha256':digest(__file__),
             'input_pack_sha256':digest(a.input_pack) if a.input_pack else None,'comparisons':[],
             'screen_relative_l2_limits':{'norm':2e-6,'raw_fc':1e-5,'packed_fc':1e-4},
             'screen_note':'Bounds flag operator regressions only; reference_rms and max_abs aid near-zero interpretation; independent review required.'}
    write_json(out/'STATE.json',receipt)
    try:
        with torch.inference_mode():
            operands={}
            for kind,x,gamma in [('hidden',hidden,module.norm_hidden_w_raw),('embedding',embedding,module.pre_fc_norm_embedding.weight)]:
                source_gamma=source_tensor(f'mtp.pre_fc_norm_{kind}.weight').float().cuda()
                candidate=rms_norm_f32(x,gamma,module.rms_eps);native=torch.empty_like(candidate)
                native_gamma=(source_gamma+1).contiguous();call('norm',x.data_ptr(),native_gamma.data_ptr(),native.data_ptr(),x.shape[0],x.shape[1],module.rms_eps,stream)
                torch.cuda.synchronize();teacher=(x.double()/(x.double().square().mean(-1,keepdim=True)+module.rms_eps).sqrt())*(source_gamma.double()+1)
                entry={'kind':'norm_'+kind,'loaded_gamma_vs_source':compare(source_gamma,gamma.float()),
                       'native_vs_candidate':compare(native,candidate),'candidate_vs_fp64':compare(teacher,candidate.double()),'native_vs_fp64':compare(teacher,native.double())}
                receipt['comparisons'].append(entry);write_json(out/'RESULTS.json',receipt)
                save_file({'x':x.cpu(),'source_gamma':source_gamma.cpu(),'candidate':candidate.cpu(),'native':native.cpu()},str(out/f'{kind}-norm.safetensors'))
                assert entry['native_vs_candidate']['relative_l2']<2e-6,entry
                operands[kind]=candidate.reshape(-1,2560).contiguous()
            for kind,linear in [('hidden',module.fc_hidden),('embedding',module.fc_embedding)]:
                x=operands[kind];inner=linear.inner;n=linear.out_features;k=linear.in_features;native=torch.empty((len(x),n),device=x.device)
                if a.stage=='raw':
                    original=source_tensor(f'mtp.fc_{kind}.weight').contiguous().cuda();assert original.shape==(n,k)
                    loaded=inner.weight.T.contiguous();candidate=raw_fc_f32(x,inner.weight)
                    typ={torch.float32:0,torch.float16:1,torch.bfloat16:30}[original.dtype]
                    call('raw',original.data_ptr(),typ,x.data_ptr(),native.data_ptr(),len(x),k,n,stream);torch.cuda.synchronize()
                    teacher=x.double()@original.double().T
                    entry={'kind':'raw_fc_'+kind,'loaded_vs_original':compare(original.float(),loaded.float()),
                           'native_vs_candidate':compare(native,candidate),'candidate_vs_fp64':compare(teacher,candidate.double()),'native_vs_fp64':compare(teacher,native.double())}
                    threshold=1e-5
                else:
                    assert inner.suh is not None and inner.svh is not None and inner.K==8
                    assert inner.trellis.is_contiguous() and inner.trellis.element_size()==2
                    assert inner.trellis.numel()==(k//16)*(n//16)*(16*inner.K)
                    assert inner.suh.dtype==inner.svh.dtype==torch.float16 and inner.suh.shape==(k,) and inner.svh.shape==(n,)
                    assert inner.suh.is_contiguous() and inner.svh.is_contiguous()
                    assert inner.trellis.device==inner.suh.device==inner.svh.device==x.device
                    rotated_weight=inner.get_inner_weight_tensor();codebook=2 if inner.mul1 else (1 if inner.mcg else 0)
                    trellis=inner.trellis.cpu().contiguous();decoded=torch.empty((k,n),dtype=torch.float16)
                    call('cpu_dequant',trellis.data_ptr(),k,n,inner.K,codebook,decoded.data_ptr())
                    codec_exact=torch.equal(decoded.view(torch.uint8),rotated_weight.cpu().view(torch.uint8))
                    if not codec_exact:save_file({'native_decoded':decoded,'candidate_decoded':rotated_weight.cpu()},str(out/f'{kind}-codec-failure.safetensors'))
                    assert codec_exact
                    rotation=torch.empty_like(x,dtype=torch.float16)
                    call('rotate',x.data_ptr(),inner.suh.data_ptr(),rotation.data_ptr(),len(x),k,stream);torch.cuda.synchronize()
                    expected_rotation=gufo_input_rotation(x,inner.suh);rotation_exact=torch.equal(rotation.view(torch.uint8),expected_rotation.view(torch.uint8))
                    if not rotation_exact:save_file({'x':x.cpu(),'suh':inner.suh.cpu(),'native':rotation.cpu(),'candidate':expected_rotation.cpu()},str(out/f'{kind}-rotation-failure.safetensors'))
                    assert rotation_exact
                    workspace=torch.empty(64*k,dtype=torch.int16,device=x.device);counters=torch.zeros(n//128,dtype=torch.int32,device=x.device)
                    call('packed',inner.trellis.data_ptr(),inner.suh.data_ptr(),inner.svh.data_ptr(),x.data_ptr(),native.data_ptr(),len(x),k,n,inner.K,codebook,workspace.data_ptr(),workspace.numel(),counters.data_ptr(),counters.numel(),stream)
                    torch.cuda.synchronize();assert not torch.count_nonzero(counters)
                    candidate=packed_fc_f32(x,rotated_weight,inner.suh,inner.svh)
                    cpu_x=x[:2].cpu().contiguous();cpu_y=torch.empty((2,n));su=inner.suh.cpu().contiguous();sv=inner.svh.cpu().contiguous()
                    call('cpu_matvec',trellis.data_ptr(),su.data_ptr(),sv.data_ptr(),cpu_x.data_ptr(),cpu_y.data_ptr(),2,k,n,inner.K,codebook)
                    entry={'kind':'packed_fc_'+kind,'codec_exact':codec_exact,'rotation_exact':rotation_exact,
                           'native_vs_candidate':compare(native,candidate),'candidate_vs_cpu_oracle':compare(cpu_y,candidate[:2].cpu()),'native_vs_cpu_oracle':compare(cpu_y,native[:2].cpu())}
                    threshold=1e-4
                # Save bounded actual operands/outputs for retry of affected comparisons.
                save_file({'x':x.cpu(),'candidate':candidate.cpu(),'native':native.cpu()},str(out/f'{kind}.safetensors'))
                receipt['comparisons'].append(entry);write_json(out/'RESULTS.json',receipt)
                assert entry['native_vs_candidate']['relative_l2']<threshold,entry
            for index,before in source_values.items():assert torch.equal(before.view(torch.uint8),rows[index].view(torch.uint8))
            assert protected=={name:digest(source/name) for name in protected}
            receipt.update(status='completed_pending_independent_review',seconds=time.monotonic()-start,source_unchanged=True,
                           scope='Four actual next-token positions; two FCs. Numeric/codec screen, not full quality or MTP speed.')
            write_json(out/'SUMMARY.json',receipt)
    except BaseException as e:
        receipt.update(status='failed',error=repr(e));write_json(out/'FAILURE.json',receipt);raise
    finally:module.unload();rows.close();write_json(out/'STATE.json',receipt)

if __name__=='__main__':main()
