"""Wait for natural conversion boundary, run one bounded pair qualification.

Never stops a converter or changes frozen sources. Main's next nonblocking GPU
admission may exit while this task holds its lease; callback resumes only after
review. No automatic candidate deployment or failure retry.
"""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time,uuid
from pathlib import Path
from run_stage import OUT,TASK,LOCK,available,save
from owned_cleanup import cleanup_owned
from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
R=OUT/'capture-forward-pair'
def stopped(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,stopped)
signal.signal(signal.SIGALRM,stopped)
def enabled():assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'Latest user stop'
owner=(R/'OWNER.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'STATE.json').exists(),'One-shot; preserve any prior attempt'
review=json.loads((R/'READINESS-REVIEW.json').read_text());assert review['status']=='qualified_for_capture_screen'
cpu=json.loads((R/'CPU.json').read_text());assert cpu['status']=='pass' and not cpu['gpu_used']
identity={str(TASK/n):hashlib.sha256((TASK/n).read_bytes()).hexdigest() for n in ['capture_forward_pair.py','capture_forward_pair_probe.py','capture_forward_pair_job.py','test_capture_forward_pair.py','capture-forward-pair-contract.json','capture_forward_ops.py','capture_lt.py','capture_fast.py']}
assert review['identity']==identity
assert review['input_sha256']==hashlib.sha256((R/'INPUT.json').read_bytes()).hexdigest()
for n,h in cpu['identity'].items():assert hashlib.sha256((TASK/n).read_bytes()).hexdigest()==h
state=dict(status='waiting_natural_boundary',started=time.time(),pid=os.getpid(),identity=identity)
save(R/'STATE.json',state)
def waitlock(f,seconds):
    deadline=time.monotonic()+seconds
    while True:
        enabled()
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return
        except BlockingIOError:
            assert time.monotonic()<deadline,'Natural-boundary lease timeout'
            time.sleep(1)
try:
    with LOCK.open('a') as gpu,(OUT/'model-job-c03/JOB.lock').open('a') as job:
        enabled();signal.alarm(4*3600)
        try:fcntl.flock(gpu,fcntl.LOCK_EX)
        finally:signal.alarm(0)
        waitlock(job,180);enabled()
        latest=max((OUT/'model-job-c03').glob('module-[0-9]*'))
        prior=json.loads((latest/'STATE.json').read_text())
        if prior['status']=='failed':
            # Explicit completed-row disk recovery, original failure STATE retained.
            assert prior['error']=="AssertionError('Disk reserve below 100 GiB')"
            recovery=json.loads((latest/'DISK-RECOVERY.json').read_text())
            journal=json.loads((latest/'CPU-COMMIT.json').read_text())
            assert recovery['status']=='committed' and recovery['gpu_used'] is False
            assert recovery['source_failure_preserved'] and recovery['rows']==2309
            assert journal['phase']=='committed' and journal['new_index']==recovery['next_module_idx']
            current=json.loads((OUT/'work/ckpt/job.json').read_text())
            assert current['next_module_idx']==journal['new_index'] and current['bad_rows']==[]
            review_recovery=json.loads((OUT/'model-job-c03/repair-010-disk-checkpoint/REVIEW.json').read_text())
            assert review_recovery['status']=='qualified_disk_checkpoint_recovery'
            assert review_recovery['committed_idx']==current['next_module_idx']
        else:
            assert prior['status'] in ('passed','passed_after_cpu_checkpoint')
        assert not prior['cleanup_errors'] and not prior['returned_KFD_contexts']
        frozen=json.loads((OUT/'model-job-c03/FROZEN.json').read_text())
        for p,h in {**frozen,**identity}.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
        h=host();assert not h['KFD_contexts'] and quiet(h) and not npu()
        assert available()>=28*2**30 and shutil.disk_usage(OUT).free>=150*2**30
        protected={}
        for p in [OUT/'work/ckpt/job.json',OUT/'work/ckpt/state.rows.json',OUT/'pack/ngram_embedding.safetensors',OUT/'capture-lt-evaluate/1-original_f16/numeric.pt']:
            st=p.stat();protected[str(p)]=[st.st_size,st.st_mtime_ns,st.st_ino]
        name='gufo-forward-pair';label=uuid.uuid4().hex
        assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
        command=prior['command'].copy();command[command.index('--name')+1]=name;command[command.index('--label')+1]='gufo.quant.run='+label
        command[-1]='/task/capture_forward_pair_probe.py'
        mounts=[]
        for p in ['work','pack','capture-pipeline','capture-lt-evaluate','capture-forward','capture-gram']:
            mounts+=['--mount',f'type=bind,src={OUT/p},dst=/output/{p},readonly']
        ix=command.index('--workdir');command[ix:ix]=mounts
        state.update(status='running_capture_screen',admitted=time.time(),previous_module=str(latest),checkpoint=json.loads((OUT/'work/ckpt/job.json').read_text()),command=command,protected=protected);save(R/'STATE.json',state)
        with (R/'run.log').open('x') as log:
            child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
            try:
                while child.poll() is None:
                    enabled();assert available()>=8*2**30 and shutil.disk_usage(OUT).free>=100*2**30
                    assert time.time()-state['admitted']<600,'Capture qualification window expired'
                    state['updated']=time.time();save(R/'STATE.json',state);time.sleep(1)
                assert child.returncode==0,child.returncode
                for p,expected in protected.items():
                    st=Path(p).stat();assert [st.st_size,st.st_mtime_ns,st.st_ino]==expected,p
                state['status']='completed_pending_review'
            finally:
                state.update(cleanup_owned(name,label,child,host))
                if state['cleanup_errors'] or state['returned_KFD_contexts']:state['status']='cleanup_failed'
except BaseException as e:
    state.update(status='failed',error=repr(e));raise
finally:
    state['finished']=time.time();save(R/'STATE.json',state)
    control=json.loads((OUT/'WATCH-CONTROL.json').read_text())
    if control['enabled']:
        message=('[사용자가 승인한 전문가 gate/up Lt 실제 capture 검사 종료] /home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 /home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/capture-forward-pair/{STATE,SUMMARY,PROGRESS,FAILURE}.json 및 run.log를 확인해라. 기존 layer6 64행/50056토큰과 저장 FP16 Lt control을 재사용했고 baseline 재수집은 없다. 본체는 자연경계에서 중단됐을 수 있다. cleanup/실제checkpoint/독립리뷰 후 모든516 H 카운터·5정상참조·IDs/RNG·반복exact·down 이외H 불변을 확인하라. H 차이만으로 품질동등을 주장하지 말고 유망후보만 양자화/출력 검사로 이어가라. 전체속도 주장은 금지한다. 이전 생산433동결과800행10GiB설정은 유지한다. 실패자료/저장control을 재사용하고 해당 검사만 수정한다. 최신사용자지시 우선, 중복GPU/실행중코드수정/전체baseline반복 금지. 후보 자동적용 금지. 검토 뒤 기존 승인 범위에서 본체 재개를 처리하라.')
        try:
            sent=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],'--message',message,'--resume'],capture_output=True,text=True,timeout=45)
            followup=dict(exit=sent.returncode,stdout=sent.stdout,stderr=sent.stderr)
        except (OSError,subprocess.TimeoutExpired) as error:
            followup=dict(exit=None,error=repr(error))
        save(R/'FOLLOWUP.json',followup)
