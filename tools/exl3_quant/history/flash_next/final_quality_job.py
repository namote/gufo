"""Exclusive one-shot final quality collect / separately resumable CPU score.

No baseline generation, automatic retries, serving, model changes or deployment.
GPU collect requires independently pinned successful production smoke. Score uses
no GPU devices and cannot call collector. Parent schedules each phase explicitly.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
from run_stage import OUT,TASK,LOCK,available,save
from mtp_calibration_job import sha,held_flock,owned_path
from mtp_final_compile_job import stat_id
from mtp_calibration_boundary import notify
from owned_cleanup import cleanup_owned

IMAGE='sha256:e7a761a40187efb13515a0275362ab09d084a129f4544aa4300ec5cccb8c7468'
ELF='62f0c59a15f3aee3aeb5336ab8e85a15406629dc89bc1cdce95d9a95672a3341'
LD='/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib'
G=Path('/home/m2on/projects/gufo');ROOT=OUT/'final-validation';JOB=OUT/'model-job-c03/JOB.lock';GIB=2**30


def pack_stats(pack):return {str(p.relative_to(pack)):stat_id(p) for p in sorted(pack.rglob('*')) if p.is_file()}


def validate_smoke(c):
    p=Path(c['smoke_state']);assert sha(p)==c['smoke_state_sha256']
    state=json.loads(p.read_text())
    assert state['status']=='passed_pending_independent_review'
    assert state['pack_unchanged'] and state['http_greedy_equal'] and state['vision_and_cache_pass']
    assert state['token_hashes']['off']==state['token_hashes']['mtp'] and state['token_hashes']['off']
    smoke=json.loads(Path(c['smoke_contract']).read_text())
    assert sha(c['smoke_contract'])==state['contract_sha256']
    assert smoke['pack']==c['pack'] and smoke['pack_stats']==c['pack_stats']
    assert smoke['image']==IMAGE and smoke['elf_sha256']==ELF
    assert set(c['smoke_cases_sha256'])=={'http-off','http-mtp','tokens-off','tokens-mtp'}
    for name,digest in c['smoke_cases_sha256'].items():
        q=p.parent/name/'STATE.json';assert sha(q)==digest
        case=json.loads(q.read_text());assert case['status']=='passed'
        assert not case['cleanup_errors'] and not case['returned_KFD_contexts']


def check_contract(c):
    assert c['status']=='reviewed_final_quality_job'
    required={str(TASK/n) for n in ('final_quality_job.py','test_final_quality_job.py','final_quality.py',
        'test_final_quality.py','run_stage.py','mtp_calibration_job.py','mtp_stage_disk_budget.py',
        'mtp_final_compile_job.py','mtp_calibration_boundary.py','owned_cleanup.py',
        'conversion_admission.py','conversion-desktop-contract.json')}
    required|={str(TASK.parent/'gufo_upstream_port_20261023/window.py'),c['cpu'],c['stage_contract'],
               c['build_receipt'],c['assembly_review'],c['reference_verified'],c['smoke_contract']}
    assert required<=c['identity'].keys()
    for p,digest in c['identity'].items():assert sha(p)==digest,p
    cpu=json.loads(Path(c['cpu']).read_text());assert cpu['status']=='pass' and not cpu['gpu_used']
    for p,digest in cpu['identity'].items():assert sha(p)==digest,p
    stage=json.loads(Path(c['stage_contract']).read_text())
    assert stage['status']=='reviewed_final_quality_stage' and stage['script_sha256']==sha(TASK/'final_quality.py')
    assert stage['binary']=='/quality-bin/rq_logits' and stage['model']=='/models/fnx' and stage['out']=='/receipt/quality'
    assert stage['reference']=='/reference/bf16-prefill' and stage['windows']=='/corpus/windows.json'
    assert stage['build_receipt']=='/quality-meta/QUALITY-BUILD.json' and stage['ld_library_path']==LD
    build=json.loads(Path(c['build_receipt']).read_text());assert build['status']=='linked' and build['returncode']==0
    assert stage['binary_sha256']==build['binary_sha256']==sha(TASK/'final-quality-runtime/rq_logits')
    verified=json.loads(Path(c['reference_verified']).read_text())
    ref=Path(c['reference_dir'])/'bf16-prefill';meta=json.loads(ref.with_suffix('.json').read_text())
    assert verified['sha256']==meta['bin_sha256'] and verified['bytes']==ref.with_suffix('.bin').stat().st_size
    # Parent already checked payload hash; native collector retains its own format-integrity check.
    assert sha(ref.with_suffix('.json'))==stage['reference_metadata_sha256']
    pack=Path(c['pack']);assert pack==pack.resolve() and pack_stats(pack)==c['pack_stats']
    compiled=json.loads((pack/'GUFO-COMPILE.json').read_text())
    assert compiled['status']=='compiled_pending_final_quality_and_performance_validation'
    assert compiled['vision']['payload_exact'] and compiled['mtp_payload_exact'] and compiled['mtp_projection_count']==1546
    assert sha(pack/'GUFO-COMPILE.json')==stage['compile_receipt_sha256']
    assembly=json.loads(Path(c['assembly_review']).read_text())
    assert assembly['status']=='assembled_artifacts_reviewed_pending_final_validation'
    assert assembly['pack']==str(pack) and assembly['compile_receipt_sha256']==sha(pack/'GUFO-COMPILE.json')
    validate_smoke(c)
    assert subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()==IMAGE
    return pack,stage


def command(c,pack,out,name,label,phase,groups):
    assert phase in ('collect','score')
    cmd=['docker','run','--rm','--pull','never','--network','none','--name',name,'--label','gufo.quant.run='+label,
        '--read-only','--user','1000:1000','--memory','100g' if phase=='collect' else '8g',
        '--memory-swap','100g' if phase=='collect' else '8g','--tmpfs','/tmp:rw,size=1g,mode=1777',
        '-e','HOME=/tmp','-e','PYTHONDONTWRITEBYTECODE=1','-e','LD_LIBRARY_PATH='+LD]
    if phase=='collect':
        cmd+=['--device','/dev/kfd','--device','/dev/dri','--ipc','host']
        for group in groups:cmd+=['--group-add',str(group)]
    mounts=[(pack,'/models/fnx',True),(TASK,'/task',True),(G,'/src',True),
        (Path(c['stage_contract']).parent,'/quality-meta',True),(TASK/'final-quality-runtime','/quality-bin',True),
        (Path(c['reference_dir']),'/reference',True),(Path(c['corpus_dir']),'/corpus',True),(out,'/receipt',False)]
    for source,dest,readonly in mounts:cmd+=['--mount',f'type=bind,src={source},dst={dest}'+(',readonly' if readonly else '')]
    return cmd+['--entrypoint','/opt/venv/bin/python3',IMAGE,'-u','/task/final_quality.py',
        '--contract','/quality-meta/'+Path(c['stage_contract']).name,'--phase',phase]


def guard(deadline):
    assert time.monotonic()<deadline,'Quality phase deadline'
    assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'Latest user stop'
    mem=available();assert mem>=4*GIB,'Host memory reserve below4GiB'
    assert shutil.disk_usage(OUT).free>=100*GIB,'Disk reserve below100GiB'
    text=Path('/proc/locks').read_text()
    for p in (LOCK,JOB):assert held_flock(text,os.getpid(),p.stat())
    return mem


def validate_collection(out,contract_sha):
    collected=json.loads((out/'COLLECT-STATE.json').read_text())
    assert collected['status']=='collected_pending_cpu_scoring' and collected['contract_sha256']==contract_sha
    assert not collected['cleanup_errors'] and not collected['returned_KFD_contexts'] and collected['pack_unchanged']
    assert collected['result_sha256']==sha(out/'quality/COLLECTION.json')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True)
    ap.add_argument('--phase',choices=('collect','score'),required=True);a=ap.parse_args();c=json.loads(a.contract.read_text())
    review=json.loads(Path(c['review']).read_text())
    assert review['status']=='reviewed_final_quality_job' and review['contract_sha256']==sha(a.contract)
    out=owned_path(c['out'],ROOT);statepath=out/(a.phase.upper()+'-STATE.json');recorded=False
    state=dict(status='admitting',phase=a.phase,started=time.time(),contract_sha256=sha(a.contract),cleanup_errors=[],returned_KFD_contexts=[])
    ROOT.mkdir(exist_ok=True)
    try:
        with ExitStack() as stack:
            for path in (ROOT/'QUALITY-OWNER.lock',LOCK,JOB):
                f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            pack,stage=check_contract(c)
            if a.phase=='collect':assert not out.exists(),'One-shot collection; preserve previous output'
            else:
                validate_collection(out,sha(a.contract))
            assert not statepath.exists(),'One-shot phase receipt; explicit reviewed failure recovery required'
            assert available()>=28*GIB and shutil.disk_usage(OUT).free>=150*GIB
            from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
            from conversion_admission import wait_conversion_admission
            assert not npu();_,admission=wait_conversion_admission(host,quiet,TASK/'conversion-desktop-contract.json')
            assert not host()['KFD_contexts'] and not npu()
            out.mkdir(exist_ok=True);save(statepath,state);recorded=True
            seconds=1200 if a.phase=='collect' else 660;deadline=time.monotonic()+seconds
            child=None;name='gufo-final-quality-'+uuid.uuid4().hex[:12];label=uuid.uuid4().hex
            groups=sorted({Path('/dev/kfd').stat().st_gid,*[p.stat().st_gid for p in Path('/dev/dri').glob('renderD*')]})
            cmd=command(c,pack,out,name,label,a.phase,groups)
            state.update(status='running',command=cmd,name=name,label=label,admission=admission,deadline_seconds=seconds,
                         minimum_available_bytes=available(),memory_warning_observed=False)
            save(statepath,state)
            def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:
                with (out/(a.phase+'-supervisor.log')).open('x') as log:
                    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
                    while child.poll() is None:
                        mem=guard(deadline);assert not npu(),'NPU owner appeared'
                        state['minimum_available_bytes']=min(state['minimum_available_bytes'],mem)
                        state['memory_warning_observed']|=mem<8*GIB
                        state['updated']=time.time();save(statepath,state);time.sleep(1)
                    state['returncode']=child.returncode;assert child.returncode==0,f'Quality phase exit {child.returncode}'
                p=out/'quality'/('COLLECTION.json' if a.phase=='collect' else 'QUALITY.json');result=json.loads(p.read_text())
                expected='collected_pending_cpu_scoring' if a.phase=='collect' else 'scored_pending_independent_quality_review'
                assert result['status']==expected
                state.update(status=expected,result_sha256=sha(p))
            except BaseException as e:state.update(status='failed',error=repr(e));raise
            finally:
                if child is not None:
                    state.update(cleanup_owned(name,label,child,host))
                    if state['cleanup_errors'] or state['returned_KFD_contexts']:state['status']='cleanup_failed'
                try:check_contract(c);state['pack_unchanged']=True
                except BaseException as e:state.update(status='protection_failed',protection_error=repr(e))
                state['finished']=time.time();save(statepath,state)
            expected='collected_pending_cpu_scoring' if a.phase=='collect' else 'scored_pending_independent_quality_review'
            assert state['status']==expected,state
    finally:
        if recorded:
            try:save(out/(a.phase+'-FOLLOWUP.json'),notify('[Gufo 최종 품질 검증 '+a.phase+' 종료] '+str(statepath)+
                '와 quality/ 영수증을 독립 검토하라. 저장 BF16 baseline/성공 candidate 수집 재실행 금지. collect 성공이면 GPU cleanup 확인 후 CPU score만 진행한다. '
                '점수는 prefill32760행 한정이며 MTP/vision/속도와 구분한다. 모델 자동 서비스 실행 금지.'))
            except BaseException as e:save(out/(a.phase+'-FOLLOWUP.json'),{'error':repr(e)})

if __name__=='__main__':main()
