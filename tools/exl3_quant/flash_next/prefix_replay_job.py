"""Owned finite GPU window for retained-prefix replay, with completion handoff."""
import fcntl,hashlib,json,os,shutil,signal,subprocess,sys,time,uuid
from pathlib import Path
from run_stage import OUT,TASK,LOCK,save,available
from owned_cleanup import cleanup_owned
from host_guard import host,npu,quiet
mode=sys.argv[1];assert mode in ('pilot','full')
root=OUT/'prefix-recovery';base=root/'pilot' if mode=='pilot' else root
owner=(base/'OWNER.lock').open('a');fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
assert not (base/'STATE.json').exists(),'Inspect existing attempt instead of repeating'
state=dict(status='admission',mode=mode,started=time.time(),pid=os.getpid());save(base/'STATE.json',state)
def stopped(signum,frame):raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM,stopped)
try:
 with LOCK.open('a') as gpu,(OUT/'model-job-c03/JOB.lock').open('a') as job:
  fcntl.flock(gpu,fcntl.LOCK_EX|fcntl.LOCK_NB);fcntl.flock(job,fcntl.LOCK_EX|fcntl.LOCK_NB)
  assert not (OUT/'work/ckpt').exists(),'Main checkpoint must remain absent until explicit idx6 promotion'
  cpu=json.loads((root/'CPU.json').read_text());assert cpu['status']=='pass'
  for p,h in cpu['identity'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
  frozen=json.loads((OUT/'model-job-c03/FROZEN.json').read_text())
  for p,h in frozen.items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
  ng=json.loads((OUT/'ngram-full/FROZEN.json').read_text())
  for p,h in ng['code'].items():assert hashlib.sha256(Path(p).read_bytes()).hexdigest()==h,p
  native=Path(ng['native']).stat();assert [native.st_size,native.st_mtime_ns,native.st_ino]==ng['native_stat']
  inventory=json.loads((OUT/'source-inventory.json').read_text())
  for entry in inventory['files']:
   st=(Path(inventory['source'])/entry['name']).stat();assert (st.st_size,st.st_mtime_ns)==(entry['bytes'],entry['mtime_ns'])
  prior=json.loads((OUT/'model-job-c03/module-013/STATE.json').read_text());assert prior['status']=='failed' and not prior['cleanup_errors'] and not prior['returned_KFD_contexts']
  assert json.loads((OUT/'capture-owner/REVIEW.json').read_text())['status']=='owner_fix_qualified_by_saved_valid_control'
  if mode=='full':
   pilot=json.loads((root/'pilot/STATE.json').read_text());assert pilot['status']=='completed_pending_review' and not pilot['cleanup_errors'] and not pilot['returned_KFD_contexts']
   assert json.loads((root/'pilot/REVIEW.json').read_text())['status']=='qualified_for_full_prefix_replay'
  for _ in range(30):
   h=host();assert not h['KFD_contexts'],'Do not interrupt another GPU owner'
   if quiet(h):break
   time.sleep(2)
  assert quiet(h) and not npu();assert available()>=28*2**30 and shutil.disk_usage(OUT).free>=150*2**30
  name='gufo-prefix-replay-'+mode;label=uuid.uuid4().hex
  assert subprocess.run(['docker','inspect',name],capture_output=True).returncode!=0
  command=prior['command'].copy();command[command.index('--name')+1]=name;command[command.index('--label')+1]='gufo.quant.run='+label
  command[-1:] = ['/task/prefix_replay_run.py',mode]
  ix=command.index('--workdir')
  for relative in ('work','pack','model-job-c03/repair-002-checkpoint/checkpoint-idx2'):
   command[ix:ix]=['--mount',f'type=bind,src={OUT/relative},dst=/output/{relative},readonly']
  deadline=600 if mode=='pilot' else 7200
  state.update(status='running',command=command,identity=cpu['identity'],admitted=time.time(),deadline_seconds=deadline,minimum_available_bytes=available());save(base/'STATE.json',state)
  with (base/'run.log').open('x') as log:
   child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT)
   try:
    while child.poll() is None:
     mem=available();state['minimum_available_bytes']=min(state['minimum_available_bytes'],mem)
     assert mem>=8*2**30 and shutil.disk_usage(OUT).free>=100*2**30
     assert time.time()-state['admitted']<deadline,'Prefix replay window expired'
     state['updated']=time.time();save(base/'STATE.json',state);time.sleep(2)
    assert child.returncode==0,f'Prefix replay exit {child.returncode}'
    assert json.loads((base/'SUMMARY.json').read_text())['status']=='prefix_replay_complete'
    state['status']='completed_pending_review'
   finally:
    state.update(cleanup_owned(name,label,child,host))
    if state['cleanup_errors'] or state['returned_KFD_contexts']:state['status']='cleanup_failed'
    save(base/'STATE.json',state)
except BaseException as error:
 state.update(status='failed',error=repr(error));raise
finally:
 state['finished']=time.time();save(base/'STATE.json',state)
 if mode=='full':
  control=json.loads((OUT/'WATCH-CONTROL.json').read_text())
  if control['enabled']:
   msg=('[사용자가 승인한 layer3 복구용 prefix 재생 종료] /home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 /home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/prefix-recovery/{STATE,SUMMARY,INPUTS}.json 및 run.log를 읽어라. 사용자 승인으로 불필요 idx10/layer3~6 기존결과 삭제 완료. 정상0/PLE/1/2 packed만 재사용해 idx6복구했으며 새양자화없다. GPUcleanup·원본보호·2309행/1999496토큰/첫3행pilot exact·독립리뷰 후 capture_owner 최소통합/CPU검사/새동결과 복구idx6승격을 기록하고 layer3부터 본체를 재개하라. 실패면 남은 실제checkpoint와 완료행을 점검하여 그 지점부터 복구하며 prefix/기존baseline 전체반복을 금한다. 중복GPU/실행중코드수정/정상0~2 재양자화금지. 최신 사용자 중지변경 지시우선.')
   try:
    r=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],'--message',msg,'--resume'],capture_output=True,text=True,timeout=45);save(root/'FOLLOWUP.json',dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
   except (OSError,subprocess.TimeoutExpired) as error:save(root/'FOLLOWUP.json',dict(error=repr(error)))
