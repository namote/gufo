"""Bounded scalar decode quality at saved BF16 histories; no teacher rerun.

Prefill can retain F16 HC-down, so qualify the changed Q8-down scalar route
separately. Collect the previously unavailable 1024-row legacy control once.
"""
import argparse
import json
from pathlib import Path
from .capture_window import REPO,save,sha
from .gpu_window import Window
from tools.docker.pinned_rocm101 import docker_argv,PYTHON


def run(runtime, quality, corpus, reference, model, output):
    runtime,quality,corpus,reference,model,output=map(lambda p:Path(p).resolve(),(runtime,quality,corpus,reference,model,output))
    if output.exists() or not output.is_relative_to(REPO/'build/exl3-hc'):raise ValueError('fresh output required')
    if any((model/n).exists() or (model/n).is_symlink() for n in ('hc_q8','hc_q8.json')):
        raise ValueError('legacy control already has a sidecar')
    state=json.loads((quality/'window/STATE.json').read_text())
    if state['cleanup_errors'] or not state['service_restored'] or state['kernel_final']['new_GPU_error']:
        raise ValueError('prior window not clean')
    build=json.loads((runtime/'BUILD.json').read_text())
    probe=runtime/'obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe'
    if sha(probe)!=build['binaries']['obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe']:
        raise ValueError('probe changed')
    qualified=json.loads((quality/'RESULT.json').read_text());views={'legacy':model}
    for v in ('gptq','refined_up'):
        view=quality/('model-'+v);record=qualified['variants'][v]
        if sha(view/'hc_q8.json')!=record['manifest_sha256'] or sha(view/'hc_q8/model.safetensors')!=record['pack_sha256']:
            raise ValueError('sidecar changed')
        views[v]=view
    output.mkdir(parents=True);root=output/'window';report={}
    save(output/'INPUTS.json',{'scope':'8 windows x 128 scalar positions following2048 prefix; saved BF16 teacher',
         'quality_result_sha256':sha(quality/'RESULT.json'),'runtime_build_sha256':sha(runtime/'BUILD.json'),
         'reference_metadata_sha256':sha(reference/'bf16-prefill.json')})
    with Window(root,model) as window:
        for v,view in views.items():
            cmd=['/probe','/model','--windows','/corpus/windows.json','--path','decode','--prefix','2048',
                 '--positions-limit','128','--windows-limit','8','--batch','512','--top-k','256',
                 '--reference','/reference/bf16-prefill','--prefill-reference','--out','/build/logits','--label','hc-decode-'+v,
                 '--toolchain','Nix-ROCm10.1-B10-HC']
            argv=docker_argv(root/v,runtime/'source',cmd,gpu=True,source_readonly=True,raw_entrypoint='/usr/bin/env',
                 mounts=((str(view),'/model',True),(str(probe),'/probe',True),(str(corpus),'/corpus',True),
                         (str(reference),'/reference',True)))
            window.execute(v,argv,900)
            meta=json.loads((root/v/'logits.json').read_text())
            if meta['records']!=1024 or sha(root/v/'logits.bin')!=meta['bin_sha256']:raise ValueError('decode output mismatch')
            report[v]=meta;save(output/'COLLECT.json',report)
        mounts=((str(reference),'/reference',True),)+tuple((str(root/v),'/'+v,True) for v in views)
        for label,tail in [('score',['score','/legacy/logits','/gptq/logits','/refined_up/logits']),
                           ('compare-gptq',['compare','/gptq/logits','/legacy/logits']),
                           ('compare-refined_up',['compare','/refined_up/logits','/legacy/logits'])]:
            cmd=['OPENBLAS_NUM_THREADS=1',PYTHON,'/build/source/tools/qwen27b/rq_quality/metrics.py',tail[0],
                 '--reference','/reference/bf16-prefill','--out','/build/RESULT.json',*tail[1:]]
            argv=docker_argv(root/label,REPO,cmd,source_readonly=True,raw_entrypoint='/usr/bin/env',mounts=mounts)
            window.execute(label,argv,120)
        save(output/'RESULT.json',{'status':'completed_pending_review','cases':report,
             'scope':'1024 scalar positions only; not full decode corpus or speculative distribution proof'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True)
    for name in ('runtime','quality','corpus','reference','model','output'):p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());args.pop('execute');run(**args)
