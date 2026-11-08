"""Offline final assembly candidate. No quantization or model forward is permitted.

Requires separately reviewed completed main/MTP artifacts; uses a new private
compile view and new output, leaving all calibration checkpoints/artifacts intact.
"""
import argparse,hashlib,json,math,os,struct,time
from pathlib import Path

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8<<20),b''):h.update(b)
    return h.hexdigest()
def write(path,value):
    path=Path(path);temp=path.with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def header(path):
    with Path(path).open('rb') as f:
        n=struct.unpack('<Q',f.read(8))[0];assert 0<n<32<<20
        h=json.loads(f.read(n))
    return n,h

def tensor_map(paths):
    result={}
    for path in paths:
        n,h=header(path)
        for key,info in h.items():
            if key=='__metadata__':continue
            assert key not in result,f'Duplicate tensor {key}'
            assert 0<=info['data_offsets'][0]<=info['data_offsets'][1]<=Path(path).stat().st_size-8-n
            result[key]=(Path(path),8+n,info)
    return result

def validate_mtp(mapping,recipe):
    trellis={k[:-8]:v for k,v in mapping.items() if k.endswith('.trellis')}
    assert set(trellis)==set(recipe),'MTP projection coverage mismatch'
    for key,bits in recipe.items():
        info=trellis[key][2]
        assert info['dtype'] in ('I16','U16') and info['shape'][-1]==16*bits,(key,'wrong bit rate')
        assert key+'.weight' not in mapping
        for scale in ('suh','svh'):
            assert mapping[key+'.'+scale][2]['dtype']=='F16',(key,'missing scales')
    assert all(k.startswith('mtp.') for k in mapping)

def vision_manifest(mapping):
    return {k:{'shape':v[2]['shape'],'dtype':v[2]['dtype'],'bytes':v[2]['data_offsets'][1]-v[2]['data_offsets'][0]}
            for k,v in mapping.items() if k.startswith('model.visual.')}

def verify_vision(source,output,*,count=333,parameters=448931056,bytes_expected=897862112):
    original=vision_manifest(source);actual=vision_manifest(output)
    assert original==actual,'Vision key/shape/dtype coverage changed'
    assert len(original)==count and all(v['dtype']=='BF16' for v in original.values())
    assert sum(math.prod(v['shape']) for v in original.values())==parameters
    assert sum(v['bytes'] for v in original.values())==bytes_expected
    compare_payloads(source,output,original)
    return {'tensors':count,'parameters':parameters,'bytes':bytes_expected,'dtype':'BF16','payload_exact':True}

def compare_payloads(source,output,keys):
    for key in keys:
        p,base,h=source[key];q,base2,h2=output[key]
        assert h['shape']==h2['shape'] and h['dtype']==h2['dtype'],f'Tensor type differs: {key}'
        remaining=h['data_offsets'][1]-h['data_offsets'][0]
        assert remaining==h2['data_offsets'][1]-h2['data_offsets'][0]
        with p.open('rb') as f,q.open('rb') as g:
            f.seek(base+h['data_offsets'][0]);g.seek(base2+h2['data_offsets'][0])
            while remaining:
                size=min(8<<20,remaining);a=f.read(size);b=g.read(size)
                assert len(a)==len(b)==size and a==b,f'Tensor payload differs: {key}'
                remaining-=size

def mixed_metadata(qcfg,recipe,calibration):
    qcfg.update(vision_bits=16,vision_storage='original_BF16_payload_exact',mtp_bits=4,
        mtp_bits_mode='mixed',mtp_bits_legacy_default_only=True,mtp_bits_by_projection=recipe,
        mtp_calibration=calibration)
    return qcfg

def stat_identity(path):
    s=Path(path).stat();return [s.st_dev,s.st_ino,s.st_size,s.st_mtime_ns]

def validate_calibration_identity(inputs,recipe_sha256,main_complete,cal):
    assert inputs['recipe_sha256']==recipe_sha256
    assert inputs['calibration_sha256']==main_complete['calibration_sha256']==cal['sha256']
    assert inputs['rows']==cal['rows']==2309
    assert inputs['source_total_tokens']==cal['total_tokens']==1999496
    assert inputs['total_tokens']==1997187
    assert main_complete['calibration_rows']==2309 and main_complete['calibration_tokens']==1999496

def validate_causal_mtp_receipts(receipts,hashes,pack_hashes,recipe,protected_rows_identity):
    inputs=receipts['INPUTS.json'];ic=receipts['INPUT-COMPLETE.json'];dc=receipts['DECODER-COMPLETE.json']
    ip=receipts['mtp.PACKED.json'];dp=receipts['mtp.layers.0.PACKED.json']
    ih=receipts['mtp.capture/MANIFEST.json'];dh=receipts['mtp.layers.0.capture/MANIFEST.json']
    fc_bits={k:recipe[k] for k in ('mtp.fc_hidden','mtp.fc_embedding')}
    assert ic['input_fc_pack_sha256']==ip['packed_sha256']==pack_hashes['mtp.safetensors']
    assert dp['packed_sha256']==pack_hashes['mtp.layers.0.safetensors']
    assert ip['inputs_sha256']==dp['inputs_sha256']==hashes['INPUTS.json']
    assert ip['strategy']==dp['strategy']==ih['source_strategy']==dh['source_strategy']==recipe
    assert ip['code_sha256']==dp['code_sha256']==inputs['code_sha256']
    assert ip['qualification_sha256']==dp['qualification_sha256']==inputs['qualification_sha256']
    assert ic['input_fc_bits']==inputs['input_fc_bits']==fc_bits
    assert ip['H']==ic['H'] and dp['H']==dc['H']
    assert ip['state_identity']['rows_stat_sha256']==protected_rows_identity['rows_stat_sha256']
    assert dp['state_identity']==ic['derived_state_identity']
    expected_fc={'input_fc_pack_sha256':ic['input_fc_pack_sha256'],
                 'derived_state_identity':ic['derived_state_identity'],'bits':fc_bits}
    assert dh['binding']['input_fc_identity']==expected_fc
    for packed,capture,index in ((ip,ih,0),(dp,dh,1)):
        binding=capture['binding']
        assert capture['status']=='complete_pre_quantization' and capture['format']=='gufo.mtp.capture.v1'
        assert capture['reference_indices']==[0,1,2,3,4]
        assert binding['state_identity']==packed['state_identity']
        assert binding['math_code']==packed['code_sha256']
        assert binding['index']==index and binding['rows']==2309 and binding['tokens']==1997187
        for key in ('source_identity','calibration_sha256','alignment','total_tokens','rows'):
            assert binding['context'][key]==inputs[key]
    assert inputs['alignment']=='gufo_mtp_f32_full_hc_shifted_v1'
    assert ih['binding']['input_fc_identity'] is None

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True)
    ap.add_argument('--main-work',type=Path,required=True);ap.add_argument('--mtp-work',type=Path,required=True)
    ap.add_argument('--work',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--source',type=Path,default=Path('/source'));ap.add_argument('--ngram',type=Path,required=True)
    ap.add_argument('--recipe',type=Path,required=True);a=ap.parse_args()
    assert os.environ.get('EXL3_SKIP_SIDE_MODELS')=='1'
    review=json.loads(a.review.read_text());assert review['status']=='approved_for_final_mtp_vision_compile'
    assert review['candidate_sha256']==sha(__file__) and review['recipe_sha256']==sha(a.recipe)
    assert not a.work.exists() and not a.output.exists(),'Fresh output required; recover partial compiles explicitly'
    assert a.work.resolve()!=a.output.resolve() and a.work.resolve() not in a.output.resolve().parents and a.output.resolve() not in a.work.resolve().parents
    for protected in (a.main_work,a.mtp_work,a.source):
        for destination in (a.work,a.output):assert protected.resolve()!=destination.resolve() and protected.resolve() not in destination.resolve().parents
    main_receipt=a.main_work.parent/'MAIN-CONVERSION.json';args_path=a.main_work.parent/'MAIN-COMPILE-ARGS.json'
    assert review['main_receipt_sha256']==sha(main_receipt) and review['main_args_sha256']==sha(args_path)
    complete=json.loads(main_receipt.read_text());assert complete['status']=='main_complete_side_models_skipped'
    assert complete['skip_side_models'] is True and complete['vision_bits']==16
    assert review['mtp_complete_sha256']==sha(a.mtp_work/'DECODER-COMPLETE.json')
    decoder=json.loads((a.mtp_work/'DECODER-COMPLETE.json').read_text())
    assert decoder['status']=='mtp_calibrated_pending_full_validation' and decoder['rows']==2309 and decoder['tokens']==1997187
    assert len(decoder['H'])==516 and all(v['count']==1997187 for v in decoder['H'].values())
    input_complete=json.loads((a.mtp_work/'INPUT-COMPLETE.json').read_text())
    assert input_complete['status']=='input_calibrated' and input_complete['rows']==2309 and input_complete['tokens']==1997187
    assert len(input_complete['H'])==2
    assert {v['first_key']:v['count'] for v in input_complete['H'].values()}=={'mtp.fc_hidden':7988748,'mtp.fc_embedding':1997187}
    # Parent must preserve original idx51 independently before advancing main.
    from mtp_calibration_candidate import rows_identity
    preserved=Path(review['protected_idx51']['checkpoint'])
    assert preserved.resolve()!=(a.main_work/'ckpt').resolve()
    assert json.loads((preserved/'job.json').read_text())['next_module_idx']==51
    assert rows_identity(preserved/'state.safetensors')==review['protected_idx51']['rows_identity']
    recipe=json.loads(a.recipe.read_text())['tensors'];assert len(recipe)==1546
    mtp_files=[a.mtp_work/'qtensors'/name for name in ('mtp.safetensors','mtp.layers.0.safetensors','mtp.hyper_connection_mixer.safetensors')]
    pack_hashes={p.name:sha(p) for p in mtp_files}
    assert pack_hashes==review['mtp_files_sha256']
    receipt_names=('INPUTS.json','INPUT-COMPLETE.json','DECODER-COMPLETE.json','mtp.PACKED.json','mtp.layers.0.PACKED.json','mtp.capture/MANIFEST.json','mtp.layers.0.capture/MANIFEST.json')
    receipt_hashes={name:sha(a.mtp_work/name) for name in receipt_names}
    assert receipt_hashes==review['mtp_receipts_sha256']
    receipts={name:json.loads((a.mtp_work/name).read_text()) for name in receipt_names}
    validate_causal_mtp_receipts(receipts,receipt_hashes,pack_hashes,recipe,review['protected_idx51']['rows_identity'])
    cal_path=Path('/task/c03-input-ids.json');cal=json.loads(cal_path.read_text())
    assert sha(cal_path.with_suffix('.safetensors'))==cal['sha256']
    validate_calibration_identity(receipts['INPUTS.json'],sha(a.recipe),complete,cal)
    cache_payload_stats={name:stat_identity(a.mtp_work/name/'numeric.safetensors') for name in ('mtp.capture','mtp.layers.0.capture')}
    assert cache_payload_stats==review['mtp_cache_payload_stats']
    mtp_map=tensor_map(mtp_files);validate_mtp(mtp_map,recipe)
    main_files=sorted((a.main_work/'qtensors').glob('*.safetensors'));assert main_files
    main_map=tensor_map(main_files);assert not any(k.startswith(('mtp.','model.visual.')) for k in main_map)
    identities={str(p):stat_identity(p) for p in main_files+mtp_files+[a.ngram]}
    assert identities==review['artifact_stats'],'Completed artifacts changed since review'
    source_index=json.loads((a.source/'model.safetensors.index.json').read_text())['weight_map']
    vision_source_files=sorted({a.source/f for k,f in source_index.items() if k.startswith('model.visual.')})
    source_vision=tensor_map(vision_source_files)
    assert len(vision_manifest(source_vision))==333
    a.work.mkdir();(a.work/'qtensors').mkdir();a.output.mkdir()
    for p in main_files+mtp_files:(a.work/'qtensors'/p.name).symlink_to(p.resolve())
    # Existing finalized ngram payload stays one inode; no quantization/copy fallback.
    try:os.link(a.ngram,a.output/'ngram_embedding.safetensors')
    except OSError as e:raise RuntimeError('Parent must pre-stage ngram hardlink inside the same writable mount; no cross-mount copy fallback') from e
    args=json.loads(args_path.read_text());args.update(work_dir=str(a.work),out_dir=str(a.output),in_dir=str(a.source),
        vision_bits=16,mtp_bits=4,ngram_file=str(a.output/'ngram_embedding.safetensors'))
    assert 'final_bits' in args
    from exllamav3.conversion import convert_model as cm
    import inspect
    runtime={'convert_model':sha(cm.__file__),'compile':sha(inspect.getsourcefile(cm.compile_model))}
    assert runtime==review['runtime_sha256']
    config,model,mtp,vision,tokenizer,_=cm.get_base_model(args)
    assert vision is None and mtp is not None
    assert json.loads((a.main_work/'ckpt/job.json').read_text())['next_module_idx']==len(model.modules)
    config.stc.add_tensor_files(args['ngram_file'],warn_if_override=False)
    started=time.monotonic();write(a.work/'STATE.json',{'status':'compiling','runtime_sha256':runtime})
    try:
        cm.compile_model(args,model,config,tokenizer,mtp,None)
        output_map=tensor_map(sorted(a.output.glob('model*.safetensors')))
        actual_mtp={k:v for k,v in output_map.items() if k.startswith('mtp.')}
        validate_mtp(actual_mtp,recipe)
        assert set(actual_mtp)==set(mtp_map),'Dropped or added MTP tensor'
        compare_payloads(mtp_map,actual_mtp,mtp_map)
        vision_check=verify_vision(source_vision,output_map)
        from ragged_calibration import sync_calibration_metadata
        sync_calibration_metadata(a.output,cal)
        mtpcal={'semantics':'gufo_mtp_f32_full_hc_shifted_v1','rows':2309,'source_tokens':1999496,
                'valid_pairs':1997187,'fc_hidden_H_count':7988748,'calibration_sha256':cal['sha256'],
                'completion_receipt_sha256':review['mtp_complete_sha256'],'quality_status':'pending_final_validation'}
        for name in ('config.json','quantization_config.json'):
            path=a.output/name;doc=json.loads(path.read_text())
            mixed_metadata(doc['quantization_config'] if name=='config.json' else doc,recipe,mtpcal);write(path,doc)
        assert identities=={p:stat_identity(p) for p in identities}
        assert cache_payload_stats=={name:stat_identity(a.mtp_work/name/'numeric.safetensors') for name in cache_payload_stats}
        assert receipt_hashes=={name:sha(a.mtp_work/name) for name in receipt_names}
        assert rows_identity(preserved/'state.safetensors')==review['protected_idx51']['rows_identity']
        write(a.output/'GUFO-COMPILE.json',{'status':'compiled_pending_final_quality_and_performance_validation',
              'vision':vision_check,'mtp_projection_count':len(recipe),'mtp_bits_mode':'mixed','mtp_payload_exact':True,
              'seconds':time.monotonic()-started,'review_sha256':sha(a.review),'source_artifacts_unchanged':True})
        write(a.work/'STATE.json',{'status':'completed_pending_final_validation'})
    except BaseException as e:
        write(a.work/'FAILURE.json',{'status':'failed','error':repr(e),'partial_pack_preserved':True});raise

if __name__=='__main__':main()
