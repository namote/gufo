"""Single operator screen inside the existing natural-boundary GPU/JOB lease."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,time,uuid
from pathlib import Path
from run_stage import OUT,TASK,LOCK,available,save
from owned_cleanup import cleanup_owned
from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
R=OUT/'capture-forward';L=OUT/'capture-io-large'
def stopped(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,stopped)
owner=(R/'OWNER.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (R/'SUMMARY.json').exists(),'Reuse completed operator baseline'
assert json.loads((R/'READINESS-REVIEW.json').read_text())['status']=='qualified_for_operator_screen'
cpu=json.loads((R/'CPU.json').read_text());assert cpu['status']=='pass'
for p,digest in cpu['identity'].items():
    assert p.startswith('/task/') and hashlib.sha256((TASK/p[6:]).read_bytes()).hexdigest()==digest,p
identity={str(TASK/n):hashlib.sha256((TASK/n).read_bytes()).hexdigest() for n in ['capture_forward.hip','capture_forward_ops.py','capture_forward_probe.py','capture_forward_job.py']}
state=dict(status='waiting_for_existing_boundary',started=time.time(),pid=os.getpid(),identity=identity)
save(R/'STATE.json',state)
def check_lease(lease):
    assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'User stopped watch'
    assert json.loads((L/'BOUNDARY.json').read_text())['status']=='reserved'
    os.kill(lease['pid'],0)
    assert 'state_large_boundary.py' in Path(f'/proc/{lease["pid"]}/cmdline').read_text()
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
    name='gufo-capture-forward';label=uuid.uuid4().hex
    assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
    cmd=json.loads((OUT/'model-job-c03/module-015/STATE.json').read_text())['command'].copy()
    cmd[cmd.index('--name')+1]=name;cmd[cmd.index('--label')+1]='gufo.quant.run='+label;cmd[-1]='/task/capture_forward_probe.py'
    ix=cmd.index('--workdir')
    mounts=[]
    for p in ['work','pack','model-job-c03/repair-002-checkpoint']:
        mounts+=['--mount',f'type=bind,src={OUT/p},dst=/output/{p},readonly']
    cmd[ix:ix]=mounts
    state.update(status='running_operator_screen',admitted=time.time(),command=cmd,lease_pid=lease['pid']);save(R/'STATE.json',state)
    with (R/'run.log').open('x') as log:
        child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
        try:
            while child.poll() is None:
                check_lease(lease)
                assert available()>8*2**30 and shutil.disk_usage(OUT).free>100*2**30
                assert time.time()-state['admitted']<600,'Operator window expired'
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
        msg=('[사용자가 승인한 forward Lt/WMMA 연산검사 종료] /home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 /home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/capture-forward/{STATE,SUMMARY,PROGRESS,ANALYTIC}.json 및run.log를읽어라. 현재모듈자연경계에서large-prefetch 예약락을유지중이다. 원본출력형식(expertFP16/denseFP32),FP64표본,exact/finite/반복성/ABBA를독립리뷰하고유망후보만실제capture/H/normalrefs 검증으로이어가라. 작은가중치hot-cache 연산속도를전체가속으로주장하지말라. 실패는저장입력/control재사용해영향검사만고쳐라. 다음층재개전800행10GiB설정도통합해야한다. 기존baseline전체반복/중복GPU/실행중코드수정금지. 최신사용자중지변경우선.')
        try:
            r=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],'--message',msg,'--resume'],capture_output=True,text=True,timeout=45)
            save(R/'FOLLOWUP.json',dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
        except (OSError,subprocess.TimeoutExpired) as e:save(R/'FOLLOWUP.json',dict(error=repr(e)))
