"""Bounded all-HC collection/generation in an owned, restoring GPU window."""
import argparse
import json
from pathlib import Path
import shutil
from .capture_window import REPO, save, sha
from .gpu_window import Window
from .weights import read_weight_payload
from tools.docker.pinned_rocm101 import docker_argv, PYTHON, GLIBC, ROCR_LIB


def names():
    return [f'model.language_model.layers.{i}.{k}_hyper_connection' for i in range(48) for k in ('attn','mlp')]+[
        'model.language_model.hyper_connection_mixer','mtp.layers.0.attn_hyper_connection',
        'mtp.layers.0.mlp_hyper_connection','mtp.hyper_connection_mixer']


def execute(collector, library_dir, inputs, source_model, target_model, pilot, previous_capture, output):
    collector,library_dir,inputs,source_model,target_model,pilot,previous_capture,output=map(lambda p:Path(p).resolve(),
        (collector,library_dir,inputs,source_model,target_model,pilot,previous_capture,output))
    if output.exists() or not output.is_relative_to(REPO/'build/exl3-hc'):
        raise ValueError('require fresh output under build/exl3-hc')
    if shutil.disk_usage(REPO).free<110*2**30:raise ValueError('need 110GiB including saved full H and reserve')
    binary=collector/'obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe'
    build=json.loads((collector/'BUILD.json').read_text());lib=library_dir/'libhc_feedback.so'
    lb=json.loads((library_dir/'BUILD.json').read_text())
    if sha(binary)!=build['binary_sha256'] or sha(collector/'SOURCE.json')!=build['source_manifest_sha256']:
        raise ValueError('collector identity changed')
    if sha(lib)!=lb['library_sha256']:raise ValueError('library identity changed')
    m=json.loads((inputs/'INPUTS.json').read_text());rows=m['rows']
    if (len(rows)!=16 or [r['window'] for r in rows]!=list(range(16)) or len({r['id'] for r in rows})!=16
            or [r['split'] for r in rows]!=['calibration']*8+['holdout']*8
            or sum(min(128,r['length']-33) for r in rows[:8])!=1024
            or sum(min(128,r['length']-33) for r in rows[8:])!=873
            or sha(inputs/'windows.json')!=m['windows_sha256'] or sha(inputs/'windows.bin')!=m['tokens_sha256']):
        raise ValueError('input manifest changed')
    ids={'source':{},'target':{},'input_manifest_sha256':sha(inputs/'INPUTS.json'),
         'target_index_sha256':sha(target_model/'model.safetensors.index.json')}
    for name in names():
        for p in ('down','up'):
            key=name+'.input_mix_weight_'+p+'.weight'
            for label,model,dtype in [('source',source_model,'BF16'),('target',target_model,'F16')]:
                _,identity=read_weight_payload(model,key,dtype)
                if identity['shape']!=([320,10240] if p=='down' else [10240,320]):raise ValueError('unexpected HC shape')
                ids[label][key]=identity
    output.mkdir(parents=True)
    save(output/'IDENTITIES.json',ids)
    frozen=output/'source'
    shutil.copytree(REPO/'tools/exl3_quant/hc',frozen/'tools/exl3_quant/hc',ignore=shutil.ignore_patterns('__pycache__'))
    save(output/'SOURCE.json',{str(p.relative_to(frozen)):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()})
    save(output/'BUILD.json',{'collector':build,'library':lb})
    mounts=((str(lib),'/libhc_feedback.so',True),(str(source_model),'/original',True),
            (str(target_model),'/model',True),(str(inputs),'/inputs',True),(str(pilot),'/pilot',True),
            (str(previous_capture),'/previous',True),(str(output/'IDENTITIES.json'),'/identities.json',True))
    root=output/'window'
    with Window(root,target_model) as window:
        def python_case(label,module,args,extra=(),timeout=600):
            argv=docker_argv(root/label,frozen,['PYTHONPATH=/build/source','OPENBLAS_NUM_THREADS=8',
                'OMP_NUM_THREADS=8',GLIBC+'/lib/ld-linux-x86-64.so.2','--library-path',
                GLIBC+'/lib:'+ROCR_LIB+':/opt/rocm/lib:/opt/rocm/lib64:/usr/lib/x86_64-linux-gnu',PYTHON,'-u','-m','tools.exl3_quant.hc.'+module,*args],gpu=True,
                source_readonly=True,raw_entrypoint='/usr/bin/env',mounts=mounts+extra)
            window.execute(label,argv,timeout)
        python_case('gpu-probe','gpu_probe',['--library','/libhc_feedback.so','--pilot','/pilot',
            '--source-model','/original','--target-model','/model','--output','/build/result'])
        proof=json.loads((root/'gpu-probe/result/RESULT.json').read_text())
        if proof['status']!='passed_pending_independent_review':raise ValueError('GPU numerical qualification failed')
        estimate=100*(proof['cases']['down']['gptq']['seconds']+proof['cases']['up']['gptq']['seconds']+
                      proof['cases']['up']['refinement']['meta']['seconds'])
        save(output/'ESTIMATE.json',{'projection_only_200_seconds':estimate,'excludes_capture_H_IO_metrics':True})
        # The user permits reducing the scope if full generation is too slow.
        # Preserve the completed numerical probe and restore service instead of starting hours of work.
        if estimate>1800:
            save(output/'RESULT.json',{'status':'full_generation_deferred_slow_probe','estimate_seconds':estimate,
                'completed_probe_projections':2,'reason':'user requested bounded cost; full200 needs faster implementation'})
            return
        def native(label,capture=False,positions=8,windows=1,import_dir=None,reference=None):
            command=['GUFO_HC_ALL_DIR='+('/build/capture' if capture else ''),
                     'GUFO_HC_Q8_EXPORT='+('/build/native' if capture and windows==16 else ''),
                     'GUFO_HC_Q8_IMPORT='+('/candidate' if import_dir else ''),
                     '/collector','/model','--windows','/inputs/windows.json','--path','decode','--out','/build/logits',
                     '--top-k','256','--prefix','32','--positions-limit',str(positions),
                     '--windows-limit',str(windows),'--batch','512','--label','hc-all-'+label,
                     '--toolchain','Nix-ROCm10.1-B10']
            extra=[(str(binary),'/collector',True)]
            if import_dir:extra.append((str(import_dir),'/candidate',True))
            if reference:
                extra.append((str(reference),'/reference',True));command+=['--reference','/reference/logits']
            argv=docker_argv(root/label,frozen,command,gpu=True,source_readonly=True,
                raw_entrypoint='/usr/bin/env',mounts=mounts+tuple(extra))
            window.execute(label,argv,600)
            meta=json.loads((root/label/'logits.json').read_text())
            if sha(root/label/'logits.bin')!=meta['bin_sha256'] or meta['records']!=sum(
                    min(positions,r['length']-33) for r in rows[:windows]):raise ValueError('invalid logits records')
            return meta
        control=native('control');captured=native('capture-control',True)
        for key in ('bin_sha256','full_logits_sha256','mtp_head_sha256'):
            if control[key]!=captured[key]:raise ValueError('capture changes '+key)
        full=native('capture',True,128,16)
        old=json.loads((previous_capture/'decode/logits.json').read_text())
        if full['full_logits_sha256']!=old['full_logits_sha256']:raise ValueError('main path differs from saved B10')
        for row in rows:
            count=min(128,row['length']-33);base=root/'capture/capture'/str(row['window'])
            cover=json.loads((base/'COVERAGE.json').read_text())
            if cover!={'phase':'scalar_decode_teacher_forced','first':32,'end':32+count,'rows_per_tensor':count,'sites':names()}:
                raise ValueError('coverage mismatch')
            for site in range(100):
                for p,cols in [('down',10240),('up',320)]:
                    if (base/f'site-{site:03d}'/(p+'.f32')).stat().st_size!=count*cols*4:
                        raise ValueError('invalid input bytes')
        validation={'status':'passed','tensor_count':200,'control_full_logits_sha256':control['full_logits_sha256'],
            'control_mtp_head_sha256':control['mtp_head_sha256'],'main_matches_previous':True,
            'full_logits_sha256':full['full_logits_sha256'],'mtp_head_sha256':full['mtp_head_sha256'],
            'collector_build_sha256':sha(collector/'BUILD.json'),'calibration_rows':1024,'holdout_rows':873}
        save(root/'capture/VALIDATION.json',validation)
        python_case('quant','all_quant',['--library','/libhc_feedback.so','--capture','/captured/capture',
            '--native','/captured/native','--inputs','/inputs','--source-model','/original','--target-model','/model',
            '--identities','/identities.json','--pilot','/pilot','--previous-capture','/previous','--output','/build/result'],
            ((str(root/'capture'),'/captured',True),),timeout=3600)
        results={}
        for variant in ('gptq','refined_up'):
            path=root/'quant/result'/variant;pack=json.loads((path/'PACK.json').read_text())
            if (pack['tensor_count']!=200 or set(pack['tensors'])!=set(ids['target'])
                    or pack['target_index_sha256']!=ids['target_index_sha256']):raise ValueError('incomplete pack')
            for key,t in pack['tensors'].items():
                expected=names()[t['site']]+'.input_mix_weight_'+t['projection']+'.weight'
                f=path/t['path'];shape=ids['target'][key]['shape']
                if (key!=expected or t['path']!=f"site-{t['site']:03d}/{t['projection']}.q8" or t['target_identity']!=ids['target'][key] or t['source_identity']!=ids['source'][key]
                        or not f.resolve().is_relative_to(path) or sha(f)!=t['sha256']
                        or f.stat().st_size!=shape[0]*(shape[1]//32)*34):raise ValueError('invalid import identity')
            results[variant]=native('native-'+variant,False,128,16,path,root/'capture')
        save(output/'RESULT.json',{'status':'all200_generated_native_cases_completed_pending_review',
            'variants':results,'scope':'bounded teacher-forced decode; not BF16 full-model KLD or recurrent MTP acceptance'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True)
    for name in ('collector','library-dir','inputs','source-model','target-model','pilot','previous-capture','output'):
        p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());args.pop('execute');execute(**args)
