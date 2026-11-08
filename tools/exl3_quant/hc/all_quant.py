"""Generate all200 HC candidates from bounded saved scalar inputs, GPU FP64."""
import argparse
import json
from pathlib import Path
import time
import shutil
import numpy as np
import torch
from safetensors.numpy import save_file, load_file
from . import quant as q
from .gpu_quant import feedback_library, gptq, measure, refine_up
from .weights import load_weight


def site_names():
    return [f'model.language_model.layers.{i}.{kind}_hyper_connection' for i in range(48) for kind in ('attn','mlp')]+[
        'model.language_model.hyper_connection_mixer','mtp.layers.0.attn_hyper_connection',
        'mtp.layers.0.mlp_hyper_connection','mtp.hyper_connection_mixer']


def save(path,value):
    tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(value,indent=2)+'\n');tmp.replace(path)


def inputs_for(capture, manifest, site, projection):
    names=site_names();parts={'calibration':[],'holdout':[]};records=[];cols=320 if projection=='up' else 10240
    for row in manifest['rows']:
        root=capture/str(row['window']);cover=json.loads((root/'COVERAGE.json').read_text())
        count=min(128,row['length']-33)
        if cover!={'phase':'scalar_decode_teacher_forced','first':32,'end':32+count,'rows_per_tensor':count,'sites':names}:
            raise ValueError('capture coverage mismatch')
        file=root/f'site-{site:03d}'/(projection+'.f32')
        if file.is_symlink() or file.stat().st_size!=count*cols*4:raise ValueError('invalid capture file')
        x=np.fromfile(file,dtype='<f4').reshape(count,cols)
        if not np.isfinite(x).all():raise ValueError('nonfinite input')
        parts[row['split']].append(x);records.append({'id':row['id'],'rows':count,'sha256':q.sha(file)})
    train,held=np.concatenate(parts['calibration']),np.concatenate(parts['holdout'])
    if (len(train),len(held))!=(1024,873):raise ValueError('unexpected calibration/holdout row counts')
    return train,held,records


def generate_all(library, capture, native, inputs, source_model, target_model, identities, pilot, previous_capture, output, sites="all", reuse=None):
    library,capture,native,inputs,source_model,target_model,identities,pilot,previous_capture,output=map(lambda p:Path(p).resolve(),
        (library,capture,native,inputs,source_model,target_model,identities,pilot,previous_capture,output))
    if any(output.is_relative_to(p) for p in (capture,native,inputs,source_model,target_model,pilot,previous_capture)):
        raise ValueError('output overlaps protected inputs')
    selected=list(range(100)) if sites=='all' else [int(x) for x in sites.split(',')]
    if not selected or len(set(selected))!=len(selected) or any(i<0 or i>=100 for i in selected):
        raise ValueError('invalid selected HC sites')
    run_identity={'selected_sites':selected,'protocol':'all100HC_scalar_prefix32_decode128_v1','inputs_sha256':q.sha(inputs/'INPUTS.json'),
        'identities_sha256':q.sha(identities),'library_sha256':q.sha(library),
        'generator_sha256':q.sha(__file__),'gpu_quant_sha256':q.sha(Path(__file__).with_name('gpu_quant.py')),
        'capture_coverage':{str(i):q.sha(capture/str(i)/'COVERAGE.json') for i in range(16)},
        'pilot_state_sha256':q.sha(pilot/'STATE.json'),
        'capture_validation_sha256':q.sha(capture.parent/'VALIDATION.json')}
    if output.exists() and any(output.iterdir()):
        if json.loads((output/'RUN.json').read_text())!=run_identity:raise ValueError('resume run identity changed')
    output.mkdir(parents=True,exist_ok=True)
    save(output/'RUN.json',run_identity)
    if reuse is not None:
        reuse=Path(reuse).resolve()
        prior=json.loads((reuse/'RUN.json').read_text())
        for field in ('protocol','inputs_sha256','identities_sha256','library_sha256','gpu_quant_sha256',
                      'capture_coverage','pilot_state_sha256','capture_validation_sha256'):
            if prior[field]!=run_identity[field]:raise ValueError('reused run differs: '+field)
        save(output/'REUSE.json',{'prior_run_sha256':q.sha(reuse/'RUN.json'),'prior':str(reuse)})
        for i in selected:
            for projection in ('down','up'):
                rel=Path(f'site-{i:03d}')/projection
                result_path=reuse/rel/'RESULT.json'
                if not result_path.exists() or (output/rel).exists():continue
                result=json.loads(result_path.read_text())
                for record in result['artifacts'].values():
                    f=reuse/record['path']
                    if not f.resolve().is_relative_to(reuse) or q.sha(f)!=record['sha256']:
                        raise ValueError('changed reused artifact')
                shutil.copytree(reuse/rel,output/rel)
                for record in result['artifacts'].values():
                    dest=output/record['path'];dest.parent.mkdir(parents=True,exist_ok=True)
                    shutil.copyfile(reuse/record['path'],dest)

    manifest=json.loads((inputs/'INPUTS.json').read_text());ids=json.loads(identities.read_text())
    if ids['input_manifest_sha256']!=q.sha(inputs/'INPUTS.json'):raise ValueError('changed inputs')
    validation=json.loads((capture.parent/'VALIDATION.json').read_text())
    if validation['status']!='passed' or validation['tensor_count']!=200:raise ValueError('capture not qualified')
    lib=feedback_library(library);started=time.monotonic()
    state={'status':'running','completed':[],'data_scope':'1024 calibration / 873 held-out rows per tensor; user approved bounded data',
           'torch':torch.__version__,'torch_hip':torch.version.hip,'library_sha256':q.sha(library),'gpu_used':True}
    save(output/'STATE.json',state)
    records=[]
    try:
        for site,name in enumerate(site_names()):
            if site not in selected:continue
            for p in ('down','up'):
                key=name+'.input_mix_weight_'+p+'.weight';label=f'site-{site:03d}/{p}'
                directory=output/f'site-{site:03d}'/p;directory.mkdir(parents=True,exist_ok=True)
                state['active']=label;save(output/'STATE.json',state)
                w,source=load_weight(source_model,key,'BF16');legacy,target=load_weight(target_model,key,'F16')
                if source!=ids['source'][key] or target!=ids['target'][key]:raise ValueError('source/target identity changed')
                baseline_path=native/f'site-{site:03d}'/(p+'.q8')
                legacy_pack=q.pack(*q.rtn(legacy))
                if baseline_path.read_bytes()!=legacy_pack.tobytes():raise ValueError('native Q8 byte parity failed: '+key)
                train,held,rows=inputs_for(capture,manifest,site,p)
                result_path=directory/'RESULT.json'
                if result_path.exists():
                    result=json.loads(result_path.read_text())
                    if (result['source_identity']!=source or result['target_identity']!=target
                            or result['tensor']!=key or result['site']!=site or result['projection']!=p
                            or set(result['artifacts'])!={'gptq','refined_up'}):raise ValueError('resume identity mismatch')
                    st=json.loads((directory/'STATS.json').read_text())
                    if (st['records']!=rows or st['sha256']!=q.sha(directory/'stats.npz')
                            or result['stats_sha256']!=st['sha256']):raise ValueError('resume input/H changed')
                    for v,r in result['artifacts'].items():
                        if q.sha(output/r['path'])!=r['sha256']:raise ValueError('changed completed artifact')
                    records.append(result);state['completed'].append(label);save(output/'STATE.json',state);continue
                step=time.monotonic()
                xt=torch.as_tensor(train.astype(np.float64),device='cuda')
                xh=torch.as_tensor(held.astype(np.float64),device='cuda')
                stats=directory/'stats.npz'
                reused=None
                if site==0:
                    for row in manifest['rows']:
                        n=min(128,row['length']-33)
                        old=b''.join((previous_capture/'decode/capture'/str(row['window'])/(p+'-'+str(j)+'.f32')).read_bytes()
                                     for j in range(32,32+n))
                        if old!=(capture/str(row['window'])/'site-000'/(p+'.f32')).read_bytes():
                            raise ValueError('site0 inputs differ from saved pilot; do not reuse')
                    reused=json.loads((pilot/('gptq-'+p)/'RESULT.json').read_text())
                    if reused['source_identity']!=source or reused['target_identity']!=target:raise ValueError('pilot identity changed')
                    oldstats=pilot/('stats-'+p)/'stats.npz'
                    if q.sha(oldstats)!=reused['receipt']['stats_sha256']:raise ValueError('pilot H changed')
                    if not stats.exists():
                        shutil.copyfile(oldstats,stats)
                        save(directory/'STATS.json',{'sha256':q.sha(stats),'records':rows,'calibration_rows':len(train),
                            'holdout_rows':len(held),'H':'reused_saved_CPU_FP64','source_identity':source,'target_identity':target})
                if stats.exists():
                    receipt=json.loads((directory/'STATS.json').read_text())
                    if receipt['records']!=rows or receipt['sha256']!=q.sha(stats):raise ValueError('invalid resumed H')
                    with np.load(stats,allow_pickle=False) as f:h=torch.tensor(f['H'],device='cuda')
                else:
                    h=xt.T@xt/len(train);h=(h+h.T)*.5
                    with (directory/'stats.tmp').open('wb') as f:np.savez(f,H=h.cpu().numpy())
                    (directory/'stats.tmp').replace(stats)
                    save(directory/'STATS.json',{'sha256':q.sha(stats),'records':rows,'calibration_rows':len(train),
                        'holdout_rows':len(held),'H':'full_FP64_sum_xxt_div_rows','source_identity':source,'target_identity':target})
                del xt
                q8=directory/'gptq.q8';checkpoint=directory/'GPTQ.json'
                if checkpoint.exists():
                    saved=json.loads(checkpoint.read_text())
                    if (saved['source_identity']!=source or saved['target_identity']!=target
                            or saved['stats_sha256']!=q.sha(stats) or saved['sha256']!=q.sha(q8)):
                        raise ValueError('changed saved GPTQ checkpoint')
                    packed=np.fromfile(q8,dtype=np.uint8).reshape(len(w),w.shape[1]//32,34)
                    detail=saved['detail']
                else:
                    if reused is not None:
                        oldpack=pilot/('gptq-'+p)/'candidate.safetensors'
                        if q.sha(oldpack)!=reused['candidate_sha256']:raise ValueError('pilot pack changed')
                        packed=load_file(str(oldpack))[key]
                        detail={'damping':reused['damping'],'seconds':0,'method':'reused_CPU_pilot','source_sha256':q.sha(oldpack)}
                    else:packed,detail=gptq(w,h,lib)
                    q8.write_bytes(packed.tobytes())
                    save(checkpoint,{'sha256':q.sha(q8),'stats_sha256':q.sha(stats),
                        'source_identity':source,'target_identity':target,'detail':detail})
                variants={'gptq':packed};refinement=None
                if p=='up' and reused is not None:
                    old=pilot/'refined-up/gptq_codes_scales.safetensors'
                    info=json.loads((pilot/'refined-up/RESULT.json').read_text())
                    if (q.sha(old)!=info['artifacts']['gptq_codes_scales'] or info['tensor']!=key
                            or info['source_identity']!=source or info['target_identity']!=target
                            or info['receipt']!=reused['receipt']
                            or info['baseline_result_sha256']!=q.sha(pilot/'gptq-up/RESULT.json')):
                        raise ValueError('pilot refined pack/baseline identity changed')
                    variants['refined_up']=load_file(str(old))[key];refinement={'reused_from':str(old),'seconds':0}
                elif p=='up':variants['refined_up'],refinement=refine_up(w,packed,h,detail['damping'])
                else:variants['refined_up']=packed
                metrics={'legacy_f16_rtn':measure(w,legacy_pack,h,xh)}
                for v,b in variants.items():metrics[v]=measure(w,b,h,xh)
                doc_metrics=[];offset=0
                for row in manifest['rows']:
                    if row['split']!='holdout':continue
                    n=min(128,row['length']-33);sub=xh[offset:offset+n];offset+=n
                    # Report document SSE; reuse the same source/reference, never fit holdout.
                    wt=torch.as_tensor(w.astype(np.float64),device='cuda');energy=(sub@wt.T).square().sum().item()
                    ds={v:(sub@(torch.as_tensor(q.unpack(b),device='cuda')-wt).T).square().sum().item()
                        for v,b in {'legacy_f16_rtn':legacy_pack,**variants}.items()}
                    doc_metrics.append({'id':row['id'],'rows':n,'reference_sse':energy,'output_sse':ds})
                artifacts={}
                for v,b in variants.items():
                    dest=output/v/f'site-{site:03d}'/(p+'.q8');dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(b.tobytes())
                    artifacts[v]={'path':str(dest.relative_to(output)),'sha256':q.sha(dest),'bytes':dest.stat().st_size}
                result={'tensor':key,'site':site,'projection':p,'source_identity':source,'target_identity':target,
                        'stats_sha256':q.sha(stats),'native_q8_sha256':q.sha(baseline_path),'native_byte_parity':True,
                        'gptq':detail,'refinement':refinement,'metrics':metrics,'documents':doc_metrics,'artifacts':artifacts,
                        'seconds':time.monotonic()-step,'status':'NOT_RUNTIME_QUALIFIED','input_space':'unfolded_projection_input'}
                save(result_path,result);records.append(result);state['completed'].append(label)
                state['elapsed_seconds']=time.monotonic()-started;save(output/'STATE.json',state)
                print(label,'seconds',round(result['seconds'],2),'remaining',2*len(selected)-len(state['completed']),flush=True)
                del h,xh;torch.cuda.empty_cache()
        if len(records)!=2*len(selected) or len({r['tensor'] for r in records})!=len(records):
            raise ValueError('incomplete selected HC pack')
        by_key={r['tensor']:r for r in records}
        for variant in ('gptq','refined_up'):
            tensors={};manifest_tensors={}
            for site,name in enumerate(site_names()):
                for projection in ('down','up'):
                    key=name+'.input_mix_weight_'+projection+'.weight'
                    rel=f'site-{site:03d}/{projection}.q8';dest=output/variant/rel
                    if key not in by_key:
                        dest.parent.mkdir(parents=True,exist_ok=True)
                        shutil.copyfile(native/rel,dest)
                    shape=ids['target'][key]['shape']
                    raw=np.fromfile(dest,dtype=np.uint8).reshape(shape[0],shape[1]//32,34)
                    tensors[key]=raw
                    manifest_tensors[key]={'site':site,'projection':projection,'calibrated':key in by_key,
                        'source_identity':ids['source'][key],'target_identity':ids['target'][key],
                        'path':rel,'sha256':q.sha(dest)}
            pack=output/(variant+'.safetensors')
            save_file(tensors,str(pack),metadata={'status':'NOT_RUNTIME_QUALIFIED',
                'coverage':f'{len(records)} calibrated; {200-len(records)} unchanged native Q8',
                'layout':'uint8[out,in/32,34]','calibration':'bounded_scalar_teacher_forced','variant':variant})
            save(output/variant/'PACK.json',{'schema':'gufo-hc-q8-complete-v1','variant':variant,'tensor_count':200,
                'calibrated_tensor_count':len(records),'selected_sites':selected,
                'status':'NOT_RUNTIME_QUALIFIED','target_index_sha256':ids['target_index_sha256'],
                'pack_sha256':q.sha(pack),'tensors':manifest_tensors})
        state['status']='completed_pending_native_validation';save(output/'STATE.json',state)
    except BaseException as e:
        state.update(status='failed',error=repr(e));save(output/'STATE.json',state);raise
    return state


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('library','capture','native','inputs','source-model','target-model','identities','pilot','previous-capture','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--sites',default='all')
    parser.add_argument('--reuse')
    print(json.dumps(generate_all(**vars(parser.parse_args())),indent=2))
