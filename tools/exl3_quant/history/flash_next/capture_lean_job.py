"""Single operator screen inside the existing natural-boundary GPU/JOB lease."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time,uuid
from pathlib import Path
from run_stage import OUT,TASK,LOCK,available,save
from owned_cleanup import cleanup_owned
from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
R=OUT/'capture-lean';L=OUT/'disk-boundary'
def stopped(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,stopped)
owner=(R/'OWNER.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'SUMMARY.json').exists(),'Reuse completed operator baseline'
assert json.loads((R/'READINESS-REVIEW.json').read_text())['status']=='qualified_for_capture_screen'
cpu=json.loads((R/'CPU.json').read_text());assert cpu['status']=='pass'
for p,digest in cpu['identity'].items():
    assert hashlib.sha256((TASK/p).read_bytes()).hexdigest()==digest,p
identity={str(TASK/n):hashlib.sha256((TASK/n).read_bytes()).hexdigest() for n in ['capture_lean_gram.py','capture_lean_moe.py','capture-lean-contract.json','capture_lean_probe.py','capture_lean_job.py']}
assert json.loads((R/'READINESS-REVIEW.json').read_text())['identity']==identity
state=dict(status='waiting_for_existing_boundary',started=time.time(),pid=os.getpid(),identity=identity)
save(R/'STATE.json',state)
def check_lease(lease):
    assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'User stopped watch'
    assert json.loads((L/'BOUNDARY.json').read_text())['status']=='reserved'
    os.kill(lease['pid'],0)
    assert 'disk_boundary_reserve.py' in Path(f'/proc/{lease["pid"]}/cmdline').read_text()
    locks=Path('/proc/locks').read_text().splitlines()
    for p in [LOCK,OUT/'model-job-c03/JOB.lock']:
        st=p.stat();device=f'{os.major(st.st_dev):02x}:{os.minor(st.st_dev):02x}:{st.st_ino}'
        assert any(len(s:=line.split())>=6 and s[1:4]==['FLOCK','ADVISORY','WRITE'] and s[4]==str(lease['pid']) and s[5]==device for line in locks),'Lease lock ownership changed'
try:
    deadline=time.monotonic()+4*3600
    while True:
        lease=json.loads((L/'BOUNDARY.json').read_text())
        if lease['status']=='reserved':break
        assert lease['status']=='waiting' and time.monotonic()<deadline
        assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled']
        time.sleep(1)
    check_lease(lease)
    last=max((OUT/'model-job-c03').glob('module-[0-9]*'))
    prior=json.loads((last/'STATE.json').read_text())
    assert prior['status'] in ('passed','passed_after_cpu_checkpoint')
    assert not prior['cleanup_errors'] and not prior['returned_KFD_contexts']
    frozen=json.loads((OUT/'model-job-c03/FROZEN.json').read_text())
    for p,digest in frozen.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==digest,p
    h=host();assert not h['KFD_contexts'] and quiet(h) and not npu()
    assert available()>28*2**30 and shutil.disk_usage(OUT).free>150*2**30
    protected={}
    for p in [OUT/'work/ckpt/job.json',OUT/'work/ckpt/state.rows.json',OUT/'pack/ngram_embedding.safetensors',OUT/'model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors']:
        st=p.stat();protected[str(p)]=[st.st_size,st.st_mtime_ns,st.st_ino]
    save(R/'PROVENANCE.json',dict(frozen=frozen,protected_stats=protected,lease_pid=lease['pid']))
    name='gufo-capture-lean';label=uuid.uuid4().hex
    assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
    cmd=json.loads((OUT/'model-job-c03/module-017/STATE.json').read_text())['command'].copy()
    cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--label')+1]='gufo.quant.run='+label;cmd[-1]='/task/capture_lean_probe.py'
    ix=cmd.index('--workdir')
    mounts=[]
    for p in ['work','pack','model-job-c03/repair-002-checkpoint','capture-pipeline','capture-lt-evaluate','capture-gram','capture-forward']:
        mounts+=['--mount',f'type=bind,src={OUT/p},dst=/output/{p},readonly']
    cmd[ix:ix]=mounts
    state.update(status='running_capture_screen',admitted=time.time(),command=cmd,lease_pid=lease['pid']);save(R/'STATE.json',state)
    with (R/'run.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                check_lease(lease)
                assert available()>8*2**30 and shutil.disk_usage(OUT).free>100*2**30
                assert time.time()-state['admitted']<600,'Capture window expired'
                time.sleep(1)
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
        msg=('[사용자가 승인한 capture 복사/폐기출력 정리 검사 완료] /home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 /home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/capture-lean/{STATE,SUMMARY,PROGRESS,FAILURE}.json 및run.log를읽어라. 기존64행/50056토큰control 재사용, FP16 Gram중복FP32변환/마스킹 제거와 owning gather재사용/폐기down zero mul scatter제거 후보이다. 516H/카운터/5refs/IDs/RNG byteexact 및반복/cleanup을독립리뷰하라. 실패는보존numeric/작은검사만고치고 baseline전체반복금지. disk-boundary예약보유중이며 디스크사전정리도동일경계에서통합해야한다. 유망후보 최소통합/CPU/새동결후release와락반환확인/실제checkpoint재개. 25%미적용, 전체속도단정금지, 중복GPU/실행중코드수정금지.')

        try:
            r=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],'--message',msg,'--resume'],capture_output=True,text=True,timeout=45)
            save(R/'FOLLOWUP.json',dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
        except (OSError,subprocess.TimeoutExpired) as e:save(R/'FOLLOWUP.json',dict(error=repr(e)))
