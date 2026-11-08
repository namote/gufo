"""Qualify actual HC loader against saved imports and saved BF16 logits."""
import argparse
import json
import os
from pathlib import Path
import shutil
from .capture_window import REPO,save,sha
from .gpu_window import Window
from .package import package
from tools.docker.pinned_rocm101 import docker_argv,PYTHON


def qualify(generated, captured, runtime, collector, inputs, corpus, reference, baseline, model, output):
    generated,captured,runtime,collector,inputs,corpus,reference,baseline,model,output=map(lambda p:Path(p).resolve(),
        (generated,captured,runtime,collector,inputs,corpus,reference,baseline,model,output))
    if output.exists() or not output.is_relative_to(REPO/'build/exl3-hc'):raise ValueError('fresh output required')
    state=json.loads((generated/'window/STATE.json').read_text())
    if not state['service_restored'] or state['cleanup_errors'] or state['kernel_final']['new_GPU_error']:
        raise ValueError('generation window not clean')
    if any(state['cases']['native-'+v]['status']!='completed' for v in ('gptq','refined_up')):
        raise ValueError('saved native cases missing')
    build=json.loads((runtime/'BUILD.json').read_text())
    for rel,digest in build['binaries'].items():
        if sha(runtime/rel)!=digest:raise ValueError('production binary changed')
    capture_build=json.loads((collector/'BUILD.json').read_text())
    probe=collector/'obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe'
    quality=runtime/'obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe'
    if sha(probe)!=capture_build['binary_sha256']:raise ValueError('collector binary changed')
    baseline_meta=json.loads(baseline.with_suffix('.json').read_text())
    if sha(baseline.with_suffix('.bin'))!=baseline_meta['bin_sha256']:
        raise ValueError('stored B10 baseline changed')
    output.mkdir(parents=True)
    views={}
    for variant in ('gptq','refined_up'):
        view=output/('model-'+variant);view.mkdir()
        for file in model.iterdir():
            if file.name in ('hc_q8','hc_q8.json'):raise ValueError('baseline already has sidecar')
            if file.is_file():os.link(file,view/file.name)
        package(generated/'window/quant/result',variant,model,view/'hc_q8')
        os.link(view/'hc_q8/manifest.json',view/'hc_q8.json');views[variant]=view
    save(output/'INPUTS.json',{'runtime_build_sha256':sha(runtime/'BUILD.json'),
         'collector_build_sha256':sha(collector/'BUILD.json'),'BF16_metadata_sha256':sha(reference/'bf16-prefill.json'),
         'captured_validation_sha256':sha(captured/'VALIDATION.json'),
         'baseline_metadata_sha256':sha(baseline.with_suffix('.json'))})
    root=output/'window';report={}
    with Window(root,model) as window:
        def case(label,command,view,extras=(),timeout=900):
            mounts=((str(view),'/model',True),(str(inputs),'/inputs',True),(str(corpus),'/corpus',True),
                    (str(reference),'/reference',True),(str(captured),'/captured',True),*extras)
            argv=docker_argv(root/label,runtime/'source',command,gpu=True,source_readonly=True,
                            raw_entrypoint='/usr/bin/env',mounts=mounts)
            window.execute(label,argv,timeout)
        for v,view in views.items():
            label='loader-'+v
            cmd=['GUFO_HC_ALL_DIR=','GUFO_HC_Q8_IMPORT=','GUFO_HC_Q8_EXPORT=','/probe','/model',
                 '--windows','/inputs/windows.json','--path','decode','--out','/build/logits','--top-k','256',
                 '--prefix','32','--positions-limit','128','--windows-limit','16','--batch','512',
                 '--label',label,'--toolchain','Nix-ROCm10.1-B10-HC','--reference','/captured/logits']
            case(label,cmd,view,((str(probe),'/probe',True),))
            actual=json.loads((root/label/'logits.json').read_text())
            expected=json.loads((generated/'window'/('native-'+v)/'logits.json').read_text())
            for key in ('records','bin_sha256','full_logits_sha256','mtp_head_sha256'):
                if actual[key]!=expected[key]:raise ValueError('actual loader differs from native import: '+v+' '+key)
            if sha(root/label/'logits.bin')!=actual['bin_sha256']:raise ValueError('loader file hash differs')
            report[v]={'loader_native_import_exact':True, 'manifest_sha256':sha(view/'hc_q8.json'),
                       'pack_sha256':sha(view/'hc_q8/model.safetensors')}
        for v,view in views.items():
            label='quality-'+v
            cmd=['/quality','/model','--windows','/corpus/windows.json','--path','prefill','--batch','512',
                 '--prefix','2048','--top-k','256','--reference','/reference/bf16-prefill',
                 '--out','/build/logits','--label','hc-'+v,'--toolchain','Nix-ROCm10.1-B10-HC']
            case(label,cmd,view,((str(quality),'/quality',True),))
            meta=json.loads((root/label/'logits.json').read_text())
            if meta['records']!=32760 or sha(root/label/'logits.bin')!=meta['bin_sha256']:
                raise ValueError('quality output mismatch')
            report[v]['quality_metadata']=meta
        score=root/'score'
        cmd=['OPENBLAS_NUM_THREADS=1',PYTHON,'/build/source/tools/qwen27b/rq_quality/metrics.py','score',
             '--reference','/reference/bf16-prefill','--out','/build/SCORE.json',
             '/gptq/logits','/refined/logits','/baseline/'+baseline.name]
        argv=docker_argv(score,REPO,cmd,source_readonly=True,raw_entrypoint='/usr/bin/env',mounts=(
             (str(reference),'/reference',True),(str(root/'quality-gptq'),'/gptq',True),
             (str(root/'quality-refined_up'),'/refined',True),(str(baseline.parent),'/baseline',True)))
        window.execute('score',argv,120)
        save(output/'RESULT.json',{'status':'loader_and_quality_scored_pending_review','variants':report,
                                  'score_sha256':sha(score/'SCORE.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True)
    for name in ('generated','captured','runtime','collector','inputs','corpus','reference','baseline','model','output'):
        p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());args.pop('execute');qualify(**args)
