"""Finish all200 from qualified saved captures, without repeating collection."""
import argparse
import json
from pathlib import Path
import shutil
from .capture_window import REPO,save,sha
from .gpu_window import Window
from .run_all import names
from tools.docker.pinned_rocm101 import docker_argv,PYTHON,GLIBC,ROCR_LIB


def finish(previous, collector, library_dir, inputs, source_model, target_model, pilot, previous_capture, output):
    previous,collector,library_dir,inputs,source_model,target_model,pilot,previous_capture,output=map(lambda p:Path(p).resolve(),
        (previous,collector,library_dir,inputs,source_model,target_model,pilot,previous_capture,output))
    if output.exists() or not output.is_relative_to(REPO/'build/exl3-hc'):raise ValueError('fresh output required')
    state=json.loads((previous/'window/STATE.json').read_text())
    if (not state['service_restored'] or state['cleanup_errors'] or state['kernel_final']['new_GPU_error']
            or any(state['cases'][k]['status']!='completed' for k in ('gpu-probe','control','capture-control','capture'))):
        raise ValueError('previous capture/cleanup not complete')
    capture=previous/'window/capture';v=json.loads((capture/'VALIDATION.json').read_text())
    if v['status']!='passed' or v['tensor_count']!=200:raise ValueError('invalid capture proof')
    for label in ('control','capture-control','capture'):
        d=previous/'window'/label;m=json.loads((d/'logits.json').read_text())
        if sha(d/'logits.bin')!=m['bin_sha256']:raise ValueError('changed logits')
    a=json.loads((previous/'window/control/logits.json').read_text())
    b=json.loads((previous/'window/capture-control/logits.json').read_text())
    if any(a[k]!=b[k] for k in ('bin_sha256','full_logits_sha256','mtp_head_sha256')):raise ValueError('control differs')
    binary=collector/'obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe'
    build=json.loads((collector/'BUILD.json').read_text());lb=json.loads((library_dir/'BUILD.json').read_text())
    lib=library_dir/'libhc_feedback.so'
    if (sha(binary)!=build['binary_sha256'] or sha(lib)!=lb['library_sha256']
            or sha(collector/'BUILD.json')!=v['collector_build_sha256']):raise ValueError('runtime identity changed')
    if shutil.disk_usage(REPO).free<100*2**30:raise ValueError('need 100GiB for retained full H and reserve')
    ids=json.loads((previous/'IDENTITIES.json').read_text())
    if sha(inputs/'INPUTS.json')!=ids['input_manifest_sha256']:raise ValueError('changed inputs')
    output.mkdir(parents=True);frozen=output/'source'
    shutil.copytree(REPO/'tools/exl3_quant/hc',frozen/'tools/exl3_quant/hc',ignore=shutil.ignore_patterns('__pycache__'))
    save(output/'SOURCE.json',{str(p.relative_to(frozen)):sha(p) for p in sorted(frozen.rglob('*')) if p.is_file()})
    save(output/'REUSE.json',{'previous':str(previous),'previous_state_sha256':sha(previous/'window/STATE.json'),
                            'capture_validation_sha256':sha(capture/'VALIDATION.json'),'all200':True})
    mounts=((str(lib),'/libhc_feedback.so',True),(str(source_model),'/original',True),
            (str(target_model),'/model',True),(str(inputs),'/inputs',True),(str(pilot),'/pilot',True),
            (str(previous_capture),'/previous',True),(str(previous/'IDENTITIES.json'),'/identities.json',True),
            (str(capture),'/captured',True),(str(previous/'window/quant/result'),'/prior-result',True))
    root=output/'window'
    with Window(root,target_model) as window:
        cmd=['PYTHONPATH=/build/source','OPENBLAS_NUM_THREADS=8','OMP_NUM_THREADS=8',
             GLIBC+'/lib/ld-linux-x86-64.so.2','--library-path',
             GLIBC+'/lib:'+ROCR_LIB+':/opt/rocm/lib:/opt/rocm/lib64:/usr/lib/x86_64-linux-gnu',
             PYTHON,'-u','-m','tools.exl3_quant.hc.all_quant','--library','/libhc_feedback.so',
             '--capture','/captured/capture','--native','/captured/native','--inputs','/inputs',
             '--source-model','/original','--target-model','/model','--identities','/identities.json',
             '--pilot','/pilot','--previous-capture','/previous','--output','/build/result','--reuse','/prior-result']
        argv=docker_argv(root/'quant',frozen,cmd,gpu=True,source_readonly=True,raw_entrypoint='/usr/bin/env',mounts=mounts)
        window.execute('quant',argv,3600)
        results={}
        for variant in ('gptq','refined_up'):
            path=root/'quant/result'/variant;pack=json.loads((path/'PACK.json').read_text())
            if (pack['tensor_count']!=200 or pack['calibrated_tensor_count']!=200 or set(pack['tensors'])!=set(ids['target'])
                    or pack['target_index_sha256']!=ids['target_index_sha256']):raise ValueError('incomplete pack')
            for key,t in pack['tensors'].items():
                if not t['calibrated']:raise ValueError('unprocessed tensor')
                expected=names()[t['site']]+'.input_mix_weight_'+t['projection']+'.weight'
                f=path/t['path'];shape=ids['target'][key]['shape']
                if (key!=expected or t['path']!=f"site-{t['site']:03d}/{t['projection']}.q8"
                        or t['target_identity']!=ids['target'][key] or t['source_identity']!=ids['source'][key]
                        or not f.resolve().is_relative_to(path) or sha(f)!=t['sha256']
                        or f.stat().st_size!=shape[0]*(shape[1]//32)*34):raise ValueError('invalid import identity')
            label='native-'+variant
            cmd=['GUFO_HC_ALL_DIR=','GUFO_HC_Q8_EXPORT=','GUFO_HC_Q8_IMPORT=/candidate','/collector','/model',
                 '--windows','/inputs/windows.json','--path','decode','--out','/build/logits','--top-k','256',
                 '--prefix','32','--positions-limit','128','--windows-limit','16','--batch','512',
                 '--label','hc-all-'+label,'--toolchain','Nix-ROCm10.1-B10','--reference','/captured/logits']
            argv=docker_argv(root/label,frozen,cmd,gpu=True,source_readonly=True,raw_entrypoint='/usr/bin/env',
                mounts=mounts+((str(binary),'/collector',True),(str(path),'/candidate',True)))
            window.execute(label,argv,600)
            meta=json.loads((root/label/'logits.json').read_text())
            if sha(root/label/'logits.bin')!=meta['bin_sha256'] or meta['records']!=1897:raise ValueError('invalid native records')
            results[variant]=meta
        save(output/'RESULT.json',{'status':'all200_generated_native_cases_completed_pending_review','variants':results,
             'scope':'bounded teacher-forced decode; not BF16 full-model KLD or recurrent MTP acceptance'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True)
    for name in ('previous','collector','library-dir','inputs','source-model','target-model','pilot','previous-capture','output'):
        p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());args.pop('execute');finish(**args)
