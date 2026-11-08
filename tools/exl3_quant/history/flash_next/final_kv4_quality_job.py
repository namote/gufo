"""Exclusive manual K4V4 collect, separately resumable CPU score and saved FP16 comparison.

No baseline generation, automatic retries, serving, model changes or deployment.
GPU collect reuses completed FP16 compatibility evidence; K4V4 is not prequalified. Score uses
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
from owned_cleanup import cleanup_owned

IMAGE='sha256:e7a761a40187efb13515a0275362ab09d084a129f4544aa4300ec5cccb8c7468'
REVISION='a81c0f93358f7b6817bded574d8d20490875ce88'
RUNTIME=TASK/'final-current-nix101/runtime'
LD='/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib'
G=Path('/home/m2on/projects/gufo');ROOT=OUT/'final-validation';JOB=OUT/'model-job-c03/JOB.lock';GIB=2**30


def pack_stats(pack):return {str(p.relative_to(pack)):stat_id(p) for p in sorted(pack.rglob('*')) if p.is_file()}


def validate_runtime(c):
    path=Path(c['runtime_manifest'])
    assert path==RUNTIME/'RUNTIME.json' and path==path.resolve()
    assert sha(path)==c['runtime_manifest_sha256']
    runtime=json.loads(path.read_text())
    assert runtime['status']=='cpu_checked_current_nix101_pending_gpu'
    assert runtime['source_revision']==REVISION and runtime['base_image']==IMAGE
    assert runtime['nix_volume']=='gufo-nix' and runtime['ld_library_path']==LD
    assert set(runtime['binaries'])=={'gufo','rq_logits'}
    for name,data in runtime['binaries'].items():
        binary=RUNTIME/name
        assert not binary.is_symlink() and binary.is_file()
        assert binary.stat().st_size==data['bytes'] and sha(binary)==data['sha256']
    for field,pathfield in (('build_receipt_sha256','build_receipt'),('cpu_receipt_sha256','cpu_build_receipt')):
        assert sha(Path(c[pathfield]))==runtime[field]
        receipt=json.loads(Path(c[pathfield]).read_text())
        assert receipt['status']=='passed_pending_review'
    return runtime


def validate_prior(c,runtime):
    # Completed FP16 qualification is compatibility evidence, not a K4V4 pass.
    p=Path(c['prior_qualification']);assert sha(p)==c['prior_qualification_sha256']
    prior=json.loads(p.read_text());assert prior['status']=='qualified_for_measured_scope'
    assert prior['pack']==c['pack'] and prior['runtime_manifest_sha256']==c['runtime_manifest_sha256']
    assert prior['actual_gufo_sha256']==runtime['binaries']['gufo']['sha256'] and prior['source_revision']==REVISION
    assert prior['compile_receipt_sha256']==sha(Path(c['pack'])/'GUFO-COMPILE.json')
    assert prior['quality']['KV']=='fp16'
    assert c['prior_scope']=='FP16 compatibility only; K4V4 not yet qualified'
    fp=Path(c['fp16_dir']);meta=fp/'c03-prefill.json'
    assert sha(meta)==c['fp16_metadata_sha256']
    old=json.loads(meta.read_text());assert old['kv_cache']=='fp16' and old['records']==32760
    assert old['bin_sha256']==c['fp16_bin_sha256']
    assert (fp/'c03-prefill.bin').stat().st_size==32760*3096
    assert stat_id(fp/'c03-prefill.bin')==c['fp16_bin_stat']
    evidence=fp/'COLLECTION.json';assert sha(evidence)==c['fp16_collection_sha256']
    collected=json.loads(evidence.read_text())
    assert collected['status']=='collected_pending_cpu_scoring'
    assert collected['metadata_sha256']==c['fp16_metadata_sha256']
    assert collected['runtime_manifest_sha256']==c['runtime_manifest_sha256']
    assert collected['binary_sha256']==runtime['binaries']['rq_logits']['sha256']
    assert collected['source_revision']==REVISION


def validate_control(c):
    assert c['control']==str(ROOT/'kv4/CONTROL.json')
    control=json.loads(Path(c['control']).read_text())
    assert control['enabled'] is True,'Latest user stop'
    assert control['scope']=='c03-kvarn-k4v4-validation'
    assert control['automatic_resume'] is False and control['callbacks'] is False


def check_contract(c):
    assert c['status']=='reviewed_final_kv4_quality_job'
    required={str(TASK/n) for n in ('final_kv4_quality_job.py','test_final_kv4_quality_job.py','final_kv4_quality.py',
        'test_final_kv4_quality.py','run_stage.py','mtp_calibration_job.py','mtp_stage_disk_budget.py',
        'mtp_final_compile_job.py','owned_cleanup.py',
        'conversion_admission.py','conversion-desktop-contract.json')}
    required|={str(TASK.parent/'gufo_upstream_port_20261023/window.py'),c['cpu'],c['stage_contract'],
               c['runtime_manifest'],c['build_receipt'],c['cpu_build_receipt'],c['assembly_review'],c['reference_verified'],c['prior_qualification'],str(Path(c['fp16_dir'])/'c03-prefill.json'),str(Path(c['fp16_dir'])/'COLLECTION.json')}
    assert required<=c['identity'].keys()
    for p,digest in c['identity'].items():assert sha(p)==digest,p
    cpu=json.loads(Path(c['cpu']).read_text());assert cpu['status']=='pass' and not cpu['gpu_used']
    for p,digest in cpu['identity'].items():assert sha(p)==digest,p
    stage=json.loads(Path(c['stage_contract']).read_text())
    assert stage['status']=='reviewed_final_kv4_quality_stage' and stage['script_sha256']==sha(TASK/'final_kv4_quality.py')
    assert stage['binary']=='/runtime/rq_logits' and stage['model']=='/models/fnx' and stage['out']=='/receipt/quality'
    assert stage['reference']=='/reference/bf16-prefill' and stage['windows']=='/corpus/windows.json'
    assert stage['runtime_manifest']=='/runtime/RUNTIME.json' and stage['ld_library_path']==LD
    runtime=validate_runtime(c)
    assert stage['binary_sha256']==runtime['binaries']['rq_logits']['sha256']
    assert stage['source_revision']==runtime['source_revision']
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
    validate_prior(c,runtime)
    validate_control(c)
    assert stage['fp16_metadata_sha256']==c['fp16_metadata_sha256'] and stage['fp16_bin_sha256']==c['fp16_bin_sha256']
    assert c['out']==str(ROOT/'quality-kv4-001')
    assert subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()==IMAGE
    return pack,stage


def command(c,pack,out,name,label,phase,groups):
    assert phase in ('collect','score','compare')
    cmd=['docker','run','--rm','--pull','never','--network','none','--name',name,'--label','gufo.quant.run='+label,
        '--read-only','--user','1000:1000','--memory','100g' if phase=='collect' else '8g',
        '--memory-swap','100g' if phase=='collect' else '8g','--tmpfs','/tmp:rw,size=1g,mode=1777',
        '-e','HOME=/tmp','-e','PYTHONDONTWRITEBYTECODE=1','-e','LD_LIBRARY_PATH='+LD]
    if phase=='collect':
        cmd+=['--device','/dev/kfd','--device','/dev/dri','--ipc','host']
        for group in groups:cmd+=['--group-add',str(group)]
    cmd+=['--mount','type=volume,src=gufo-nix,dst=/nix,readonly']
    mounts=[(pack,'/models/fnx',True),(TASK,'/task',True),(G,'/src',True),
        (Path(c['stage_contract']).parent,'/quality-meta',True),(RUNTIME,'/runtime',True),
        (Path(c['fp16_dir']),'/fp16',True),(Path(c['reference_dir']),'/reference',True),(Path(c['corpus_dir']),'/corpus',True),(out,'/receipt',False)]
    for source,dest,readonly in mounts:cmd+=['--mount',f'type=bind,src={source},dst={dest}'+(',readonly' if readonly else '')]
    return cmd+['--entrypoint','/opt/venv/bin/python3',IMAGE,'-u','/task/final_kv4_quality.py',
        '--contract','/quality-meta/'+Path(c['stage_contract']).name,'--phase',phase]


def guard(deadline,c):
    assert time.monotonic()<deadline,'Quality phase deadline'
    validate_control(c)
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


def validate_score(out,contract_sha):
    scored=json.loads((out/'SCORE-STATE.json').read_text())
    assert scored['status']=='scored_pending_independent_quality_review' and scored['contract_sha256']==contract_sha
    assert not scored['cleanup_errors'] and not scored['returned_KFD_contexts'] and scored['pack_unchanged']
    assert scored['result_sha256']==sha(out/'quality/QUALITY.json')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True)
    ap.add_argument('--phase',choices=('collect','score','compare'),required=True);a=ap.parse_args();c=json.loads(a.contract.read_text())
    review=json.loads(Path(c['review']).read_text())
    assert review['status']=='reviewed_final_kv4_quality_job' and review['contract_sha256']==sha(a.contract)
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
                if a.phase=='compare':validate_score(out,sha(a.contract))
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
                        mem=guard(deadline,c);assert not npu(),'NPU owner appeared'
                        state['minimum_available_bytes']=min(state['minimum_available_bytes'],mem)
                        state['memory_warning_observed']|=mem<8*GIB
                        state['updated']=time.time();save(statepath,state);time.sleep(1)
                    state['returncode']=child.returncode;assert child.returncode==0,f'Quality phase exit {child.returncode}'
                p=out/'quality'/({'collect':'COLLECTION.json','score':'QUALITY.json','compare':'COMPARISON.json'}[a.phase]);result=json.loads(p.read_text())
                expected={'collect':'collected_pending_cpu_scoring','score':'scored_pending_independent_quality_review','compare':'compared_pending_independent_quality_review'}[a.phase]
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
            expected={'collect':'collected_pending_cpu_scoring','score':'scored_pending_independent_quality_review','compare':'compared_pending_independent_quality_review'}[a.phase]
            assert state['status']==expected,state
    finally:
        pass  # Explicit manual phases only: no callback or watcher notification.

if __name__=='__main__':main()
