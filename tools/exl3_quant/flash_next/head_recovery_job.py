"""Fresh-lock head-forward recovery only; never promote idx52 or compile/serve.

The failed head's H and K8 pack are reused. A separately reviewed candidate
writes five terminal outputs privately; completion requests independent review.
CPU-only import. Original production files and both preserved checkpoints stay RO.
"""
import argparse
from contextlib import ExitStack
import fcntl,json,os
from pathlib import Path
import shutil,signal,subprocess,time,uuid
from run_stage import OUT,TASK,SOURCE,LOCK,available,save
from mtp_calibration_job import sha,held_flock,owned_path,build_command as base_command
from mtp_final_compile_job import output_path,stat_id,protected_stats
from owned_cleanup import cleanup_owned

GIB=2**30
ROOT=OUT/'head-recovery'
RUNS=ROOT/'runs'
DATA=ROOT/'data'
JOB=OUT/'model-job-c03/JOB.lock'
JOB52={'next_module_idx':52,'q_strategy':None,'bad_rows':[]}
OUTPUT_BYTES=9857310720


def disk_required():
    # Explicit tiny recovery budget plus the repo's 150GiB admission minimum.
    return max(150*GIB,100*GIB+OUTPUT_BYTES+8*GIB)


def build_command(prior,name,label,data):
    c={'script':'head_recovery.py','argv':['--approval','/mtp-run/APPROVAL.json','--out','/mtp-run/result']}
    result=base_command(prior,c,name,label,data)
    i=result.index('GUFO_MTP_LEASE_VERIFIED=1');del result[i-1:i+1]
    for key,value in [('EXL3_HIP_F32OUT_VIA_F16','0'),('EXL3_SKIP_SIDE_MODELS','1')]:
        old=[i for i,x in enumerate(result) if i and result[i-1]=='-e' and x.startswith(key+'=')]
        assert len(old)<=1
        if old:result[old[0]]=key+'='+value
        else:
            i=result.index('--workdir');result[i:i]=['-e',key+'='+value]
    return result


def required_pins(c):
    names=('head_recovery_job.py','test_head_recovery_job.py','head_recovery.py','test_head_recovery.py',
           'mtp_final_compile_job.py','mtp_calibration_job.py','mtp_stage_disk_budget.py',
           'run_stage.py','owned_cleanup.py','row_checkpoint.py',
           'conversion_admission.py','conversion-desktop-contract.json','c03-input-ids.json')
    paths={str(TASK/n) for n in names}
    paths.add(str(TASK.parent/'gufo_upstream_port_20261023/window.py'))
    paths|={str(OUT/n) for n in ('model-job-c03/FROZEN.json','source-inventory.json',
            'model-job-c03/module-060/STATE.json','model-job-c03/module-062/STATE.json',
            'model-job-c03/module-062/run.log','head-recovery/AUDIT.json','head-recovery/CANDIDATE-CPU.json')}
    paths|={str(output_path(c['approval'])),c['cpu'],c['candidate_review']}
    return paths


def check_pins(c):
    assert required_pins(c)<=c['identity'].keys(),'Unpinned recovery dependencies'
    for p,d in c['identity'].items():assert sha(p)==d,p
    frozen=OUT/'model-job-c03/FROZEN.json'
    for p,d in json.loads(frozen.read_text()).items():assert sha(p)==d,p
    inv=json.loads((OUT/'source-inventory.json').read_text())
    assert Path(inv['source']).resolve()==SOURCE.resolve()
    for e in inv['files']:
        s=(SOURCE/e['name']).stat();assert [s.st_size,s.st_mtime_ns]==[e['bytes'],e['mtime_ns']]


def check_source(approval_path):
    from head_recovery import row_identity
    a=json.loads(approval_path.read_text());audit=json.loads((ROOT/'AUDIT.json').read_text())
    assert a['status']=='reviewed_head_forward_recovery'
    assert a['candidate_sha256']==sha(TASK/'head_recovery.py')
    assert audit['status']=='head_pack_and_source52_audited'
    assert audit['frozen_sha256']==sha(OUT/'model-job-c03/FROZEN.json')
    assert a['packed_sha256']==audit['head_pack_sha256']
    pack=OUT/'work/qtensors/lm_head.safetensors'
    assert stat_id(pack)==audit['head_pack_stat']
    assert a['state_identity']==audit['rows_identity']
    for ckpt in [OUT/'work/ckpt',ROOT/'protected-idx52/ckpt']:
        assert json.loads((ckpt/'job.json').read_text())==JOB52
        assert row_identity(ckpt/'state.safetensors')[2]==audit['rows_identity']
    for name,digest in audit['source_metadata_sha256'].items():assert sha(OUT/'work/ckpt'/name)==digest
    for name in ('STATE.json','run.log'):
        k='module062_state_sha256' if name=='STATE.json' else 'module062_log_sha256'
        assert sha(OUT/'model-job-c03/module-062'/name)==audit[k]
    failed=json.loads((OUT/'model-job-c03/module-062/STATE.json').read_text())
    assert failed['status']=='failed' and not failed['cleanup_errors'] and not failed['returned_KFD_contexts']
    assert audit['first5_logit_bytes']==OUTPUT_BYTES
    assert a['files_sha256']['/output/head-recovery/AUDIT.json']==sha(ROOT/'AUDIT.json')
    assert a['files_sha256'], 'Candidate must pin its readonly dependencies'
    return audit


def guard():
    assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'Latest user stop'
    text=Path('/proc/locks').read_text()
    for p in (LOCK,JOB):assert held_flock(text,os.getpid(),p.stat()),'Fresh GPU/JOB ownership lost'
    assert json.loads((OUT/'work/ckpt/job.json').read_text())==JOB52,'Main checkpoint changed'


def check_completion(data,approval_path):
    summary=json.loads((data/'result/SUMMARY.json').read_text())
    assert summary['status']=='head_forward_complete_pending_independent_review'
    assert summary['source_index']==52 and summary['proposed_next_index']==53
    assert summary['forward_rows']==5 and summary['retained_input_rows']==2304
    assert summary['rows']==2309 and summary['tokens']==1999496
    assert summary['checkpoint_promoted'] is False
    assert summary['head_capture_repeated'] is False and summary['head_quantization_repeated'] is False
    approval=json.loads(approval_path.read_text())
    assert summary['packed_sha256']==approval['packed_sha256'] and summary['state_identity']==approval['state_identity']
    from head_recovery import check_output_header,validate_metrics
    assert len(summary['results'])==5
    for i,r in enumerate(summary['results']):
        assert r['index']==i;validate_metrics(r);check_output_header(data/f'result/{i}.safetensors',i)
        assert json.loads((data/f'result/ROW-{i}.json').read_text())==r
    return summary


def notify(receipt,state):
    control=json.loads((OUT/'WATCH-CONTROL.json').read_text())
    if not control['enabled']:return {'status':'disabled_by_user'}
    message=(f'[Gufo head 복구 검사 종료] {receipt}/STATE.json 및 run.log, '
             f'{state.get("data_dir","출력 생성 전")}/result/SUMMARY.json을 독립 검토하라. 상태 {state["status"]}. '
             'GPU cleanup/idx52·idx51·완료K8팩 보호와 5행 raw-vs-packed 수치/로그잇을 확인한 뒤 별도 CPU checkpoint 승격을 검토하라. '
             '이 실행기는 checkpoint/MAIN marker를 변경하지 않았다. H/양자화/전체 baseline 반복 및 자동 서비스 실행 금지.')
    r=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],'--message',message,'--resume'],
                     capture_output=True,text=True,timeout=45)
    return dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True);args=ap.parse_args()
    c=json.loads(args.contract.read_text())
    assert c['status']=='reviewed_for_head_recovery_run' and c['purpose']=='lost_terminal_refs_recovery_not_benchmark'
    review=json.loads(Path(c['review']).read_text())
    assert review['status']=='reviewed_for_head_recovery_run' and review['contract_sha256']==sha(args.contract)
    cpu=json.loads(Path(c['cpu']).read_text())
    assert cpu['status']=='pass' and cpu['gpu_used'] is False
    assert cpu['identity'][str(Path(__file__).resolve())]==sha(__file__)
    for p,d in cpu['identity'].items():assert sha(p)==d
    receipt=owned_path(c['receipt_dir'],RUNS);data=owned_path(c['data_dir'],DATA)
    assert not data.exists(),'One-shot private outputs; do not overwrite earlier recovery'
    approval=output_path(c['approval'])
    seconds=c.get('seconds',1200);assert isinstance(seconds,int) and 0<seconds<=1200
    RUNS.mkdir(parents=True,exist_ok=True);recorded=False
    state=dict(status='admitting',pid=os.getpid(),started=time.time(),contract_sha256=sha(args.contract),
               cleanup_errors=[],returned_KFD_contexts=[],container_started=False)
    try:
        with ExitStack() as stack:
            for p in (RUNS/'OWNER.lock',LOCK,JOB):
                f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not receipt.exists();receipt.mkdir(parents=True)
            save(receipt/'STATE.json',state);recorded=True
            child=None;before=None;name='gufo-head-recovery-'+uuid.uuid4().hex[:12];label=uuid.uuid4().hex
            def stop(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:
                guard();check_pins(c);check_source(approval)
                assert available()>=28*GIB and shutil.disk_usage(OUT).free>=disk_required()
                from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
                from conversion_admission import wait_conversion_admission
                assert not npu()
                _,admission=wait_conversion_admission(host,quiet,TASK/'conversion-desktop-contract.json')
                assert not host()['KFD_contexts'] and not npu();guard()
                roots=[OUT/'work',OUT/'pack',OUT/'mtp-calibration-work/full/work',OUT/'mtp-protected-idx51/ckpt',ROOT/'protected-idx52/ckpt',SOURCE]
                before=protected_stats(roots);save(receipt/'PROTECTED-BEFORE.json',before)
                data.mkdir(parents=True)
                for n in ('home','cache','triton-cache','scratch'):(data/n).mkdir()
                shutil.copyfile(approval,data/'APPROVAL.json');assert sha(data/'APPROVAL.json')==sha(approval)
                prior=json.loads((OUT/'model-job-c03/module-060/STATE.json').read_text())
                command=build_command(prior,name,label,data)
                state.update(status='running',command=command,name=name,label=label,admission=admission,
                             data_dir=str(data),disk_required_bytes=disk_required(),deadline_seconds=seconds,minimum_available_bytes=available())
                save(receipt/'STATE.json',state);started=time.monotonic()
                with (receipt/'run.log').open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);state['container_started']=True
                    while child.poll() is None:
                        guard();assert not npu(),'NPU owner appeared'
                        mem=available();state['minimum_available_bytes']=min(state['minimum_available_bytes'],mem)
                        assert mem>=8*GIB,'Memory reserve below 8GiB'
                        assert shutil.disk_usage(OUT).free>=100*GIB,'Disk reserve below 100GiB'
                        assert time.monotonic()-started<seconds,'Head recovery deadline expired'
                        state['updated']=time.time();save(receipt/'STATE.json',state);time.sleep(2)
                    state['returncode']=child.returncode;assert child.returncode==0,f'Head recovery exit {child.returncode}'
                check_completion(data,approval)
                state.update(status='completed_pending_independent_review',summary_sha256=sha(data/'result/SUMMARY.json'))
            except BaseException as error:
                state.update(status='failed',error=repr(error));raise
            finally:
                if child is not None:
                    state.update(cleanup_owned(name,label,child,host))
                    if state['cleanup_errors'] or state['returned_KFD_contexts']:state['status']='cleanup_failed'
                if before is not None:
                    try:
                        after=protected_stats(roots);save(receipt/'PROTECTED-AFTER.json',after)
                        assert before==after,'Readonly source metadata changed'
                        check_pins(c);check_source(approval);guard();state['protected_unchanged']=True
                    except BaseException as error:state.update(status='protection_failed',protection_error=repr(error))
                state['finished']=time.time();save(receipt/'STATE.json',state)
    finally:
        if recorded:
            try:save(receipt/'FOLLOWUP.json',notify(receipt,state))
            except BaseException as error:save(receipt/'FOLLOWUP.json',{'error':repr(error)})
    assert state['status']=='completed_pending_independent_review',state['status']


if __name__=='__main__':main()
