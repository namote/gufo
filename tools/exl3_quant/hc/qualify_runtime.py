"""One bounded final HC throughput/functional window; reuse saved B10 controls."""
import argparse
import json
from pathlib import Path
import re
from .capture_window import REPO,save,sha
from .gpu_window import Window
from tools.docker.pinned_rocm101 import docker_argv,PYTHON


def rates(text):
    return {test.strip():float(rate) for test,rate in re.findall(r'\|\s*((?:pp|tg)\d+(?: @ d\d+)?)\s*\|\s*([\d.]+)\s*±',text)}


def run(runtime, view, model, quality, decode, baseline, output):
    runtime,view,model,quality,decode,baseline,output=map(lambda p:Path(p).resolve(),(runtime,view,model,quality,decode,baseline,output))
    if output.exists() or not output.is_relative_to(REPO/'build/exl3-hc'):raise ValueError('fresh output required')
    state=json.loads((quality/'window/STATE.json').read_text())
    if state['cleanup_errors'] or not state['service_restored'] or state['kernel_final']['new_GPU_error']:
        raise ValueError('prior window not clean')
    ds=json.loads((decode/'window/STATE.json').read_text())
    if ds['cleanup_errors'] or not ds['service_restored'] or ds['kernel_final']['new_GPU_error']:
        raise ValueError('decode window not clean')
    if json.loads((decode/'INPUTS.json').read_text())['quality_result_sha256']!=sha(quality/'RESULT.json'):
        raise ValueError('decode proof belongs to another quality run')
    build=json.loads((runtime/'BUILD.json').read_text());binary=runtime/'obj/gufo'
    if sha(binary)!=build['binaries']['obj/gufo']:raise ValueError('binary changed')
    if not (view/'hc_q8.json').is_file():raise ValueError('qualified sidecar missing')
    qualified=json.loads((quality/'RESULT.json').read_text())
    manifest_sha=sha(view/'hc_q8.json');pack_sha=sha(view/'hc_q8/model.safetensors')
    if not any(v['manifest_sha256']==manifest_sha and v['pack_sha256']==pack_sha
               and v['loader_native_import_exact'] for v in qualified['variants'].values()):
        raise ValueError('selected sidecar not qualified')
    output.mkdir(parents=True)
    save(output/'INPUTS.json',{'runtime_build_sha256':sha(runtime/'BUILD.json'),'model_view':str(view),
         'sidecar_manifest_sha256':sha(view/'hc_q8.json'),'quality_result_sha256':sha(quality/'RESULT.json'),
         'baseline':str(baseline),'decode_result_sha256':sha(decode/'RESULT.json')})
    root=output/'window';report={'status':'running','cases':{}}
    common=['/gufo','bench','-m','/model','-r','2','--kv-cache-type','fp16','--temperature','0',
            '--top-p','1','--top-k','0','--min-p','0','--repeat-penalty','1','--seed','314159','--verbose']
    with Window(root,model) as window:
        for label,args in [('pp-off',['--speculative','off','-p','2048,8192','-d','0']),
                           ('tg-off',['--speculative','off','-n','128','-d','0,32768']),
                           ('tg-mtp',['--speculative','mtp','--mtp-model','/model','--draft-tokens','7','-n','128','-d','0,32768'])]:
            argv=docker_argv(root/label,runtime/'source',common+args,gpu=True,source_readonly=True,
                raw_entrypoint='/usr/bin/env',mounts=((str(view),'/model',True),(str(binary),'/gufo',True)))
            window.execute(label,argv,900)
            text=(root/label/'run.log').read_text();measured=rates(text)
            if len(measured)!=2:raise ValueError('missing timing rows')
            controls=[rates((baseline/(label+'-C'+run)/'stdout.log').read_text()) for run in ('r1','r2')]
            if any(set(c)!=set(measured) for c in controls):raise ValueError('control case mismatch')
            report['cases'][label]={'tokens_s':measured,'saved_controls':controls,
                'relative_to_control_mean':{k:v/(sum(c[k] for c in controls)/2) for k,v in measured.items()},
                'log_sha256':sha(root/label/'run.log')}
            save(output/'RESULT.json',report)
        def tokens(label):
            rows=re.findall(r'tg depth=(\d+).*?output_sha256=([0-9a-f]{64})',(root/label/'run.log').read_text())
            if len(rows)!=4:raise ValueError('missing emitted-token hashes')
            return rows
        if tokens('tg-off')!=tokens('tg-mtp'):raise ValueError('greedy MTP differs from target')
        draft=[int(n) for n in re.findall(r'\bdrafted=(\d+)',(root/'tg-mtp/run.log').read_text())]
        if len(draft)!=4 or not all(draft):raise ValueError('draft was inactive')
        accepted=[int(n) for n in re.findall(r'(?<!_)\baccepted=(\d+)',(root/'tg-mtp/run.log').read_text())]
        if len(accepted)!=4 or not all(accepted) or any(a>d for a,d in zip(accepted,draft)):
            raise ValueError('no valid accepted drafts')
        report['drafts']={'proposed':draft,'accepted':accepted,'acceptance':[a/d for a,d in zip(accepted,draft)]}
        lookup=[tuple(map(int,m)) for m in re.findall(r'lookup_proposed=(\d+) lookup_accepted=(\d+)',(root/'tg-mtp/run.log').read_text())]
        if len(lookup)!=4:raise ValueError('missing lookup counters')
        pure=[(d-l[0],a-l[1]) for d,a,l in zip(draft,accepted,lookup)]
        if any(d<=0 or a<=0 or a>d for d,a in pure):raise ValueError('no accepted neural MTP drafts')
        report['drafts'].update(lookup=lookup,neural_mtp=pure)
        report['greedy_mtp_exact']=True;save(output/'RESULT.json',report)
        cmd=[PYTHON,'/build/source/tests/functional/hc_sidecar.py','--binary','/gufo','--model','/model','--output','/build/result']
        argv=docker_argv(root/'smoke',REPO,cmd,gpu=True,source_readonly=True,raw_entrypoint='/usr/bin/env',
            mounts=((str(view),'/model',True),(str(binary),'/gufo',True)))
        window.execute('smoke',argv,900)
        smoke=json.loads((root/'smoke/result/RESULT.json').read_text())
        if smoke['status']!='passed_pending_review':raise ValueError('functional smoke failed')
        report.update(status='completed_pending_independent_review',smoke=smoke)
        save(output/'RESULT.json',report)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--execute',action='store_true',required=True)
    for name in ('runtime','view','model','quality','decode','baseline','output'):p.add_argument('--'+name,required=True)
    args=vars(p.parse_args());args.pop('execute');run(**args)
