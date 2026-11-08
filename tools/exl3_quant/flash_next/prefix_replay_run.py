"""Recover idx6 from idx2 using packed modules2..5; no capture or quantization."""
import hashlib,json,os,sys,time
from pathlib import Path
import torch
from exllamav3.conversion import convert_model as cm
from exllamav3.loader.safetensors_alt import safe_open
from disk_state import install as disk_install,_read_row
from ragged_calibration import load_calibration
from prefix_replay import make_replay
from row_checkpoint import open_rows
from capture_progress import install as progress_install
ROOT=Path('/output');OUT=ROOT/'prefix-recovery';WORK=OUT/'work';TASK=Path('/task')
mode=sys.argv[1];assert mode in ('pilot','full')
base=OUT/'pilot' if mode=='pilot' else OUT
started=time.monotonic();cal=json.loads((TASK/'c03-input-ids.json').read_text());identity=json.loads((OUT/'INPUTS.json').read_text())
def write(n,v):(base/n).write_text(json.dumps(v,indent=2))
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while chunk:=f.read(4*2**20):h.update(chunk)
 return h.hexdigest()
for p,h in identity['files'].items():assert sha(Path(p))==h,p
for p,st in identity['protected_stats'].items():
 v=Path(p).stat();assert dict(bytes=v.st_size,mtime_ns=v.st_mtime_ns,inode=v.st_ino)==st,p
assert os.environ['EXL3_HIP_F32OUT_VIA_F16']=='0'
args=json.loads((ROOT/'work/args.json').read_text());args.update(work_dir=str(base/'work'),out_dir=str(ROOT/'pack'),ngram_file=str(ROOT/'pack/ngram_embedding.safetensors'),max_module=5,checkpoint_interval=0,cal_rows=cal['rows'])
args.pop('clear_files',None)
args['last_checkpoint_index']=-1
(base/'scratch').mkdir(exist_ok=True)
stores=disk_install(cm,base/'scratch');cm.get_file_calibration=load_calibration
job=json.loads((base/'work/ckpt/job.json').read_text());assert job['next_module_idx'] in range(2,6)
modules=[]
def load_packed(module):
 p=ROOT/'work/qtensors'/(module.key+'.safetensors');assert str(p) in identity['files']
 with safe_open(str(p)) as f:values={k:_read_row(f,k) for k in f.keys()}
 assert values
 modules.append(module.key);print('GUFO_PREFIX_PACKED',module.key,flush=True)
 return values
# Use first three actual rows in a separate pilot checkpoint, without changing
# its selected source tokens or the full-run calibration/input contract.
if mode=='pilot':args['cal_rows']=3
cm.check_bad_rows=lambda bad,*args,**kwargs: (_ for _ in ()).throw(AssertionError(('nonfinite rows',bad))) if bad else None
original_prepare=cm.prepare_state
def prepare(*a,**kw):
 state,ids=original_prepare(*a,**kw)
 assert len(state)==len(ids)==args['cal_rows']
 if mode=='full':assert sum(x.numel() for x in ids)==cal['total_tokens']
 return state,ids
cm.prepare_state=prepare
# Fail closed if the replay clone ever reaches an excluded conversion path.
def forbidden(*args,**kwargs):raise AssertionError('Capture/quantization/compile forbidden during prefix replay')
cm.quantize_linears_single=cm.quantize_linears_parallel=cm.compile_model=forbidden
cm._flush_capture_H=forbidden
def feedback(state,module,config,bpw,error,cos,sqnr,seconds):
 print('GUFO_PREFIX_MODULE',json.dumps(dict(module=module.key,rows=len(state),seconds=seconds,quality_measured=False)),flush=True)
cm.feedback_module=feedback
progress_install(cm)
run=make_replay(cm.main,load_packed)
try:
 try:run(args,job)
 except SystemExit as e:assert e.code in (None,0)
 else:raise AssertionError('Replay passed max_module boundary')
 final=json.loads((base/'work/ckpt/job.json').read_text());assert final==dict(next_module_idx=6,q_strategy=None,bad_rows=[])
 paths,infos=open_rows(base/'work/ckpt/state.safetensors');assert len(paths)==args['cal_rows']
 assert all(info==('F32',(1,cal['sequences'][i]['tokens'],4,2560)) for i,info in enumerate(infos))
 with safe_open(str(base/'work/ckpt/original_input_ids.safetensors')) as f, safe_open('/task/c03-input-ids.safetensors') as src:
  for i in range(args['cal_rows']):assert torch.equal(_read_row(f,f'tensor.{i}').long(),_read_row(src,f'input_ids.{i}').long())
 for p,h in identity['files'].items():assert sha(Path(p))==h,p
 if mode=='full':
  pilot_paths,_=open_rows(OUT/'pilot/work/ckpt/state.safetensors');assert len(pilot_paths)==3
  for actual,expected in zip(paths[:3],pilot_paths):
   with safe_open(str(actual)) as f,safe_open(str(expected)) as ref:
    a=_read_row(f,'tensor');b=_read_row(ref,'tensor');assert torch.equal(a.view(torch.uint8),b.view(torch.uint8)),'Full prefix differs from qualified pilot'
 for p,st in identity['protected_stats'].items():
  v=Path(p).stat();assert dict(bytes=v.st_size,mtime_ns=v.st_mtime_ns,inode=v.st_ino)==st,p
 write('SUMMARY.json',dict(status='prefix_replay_complete',mode=mode,modules=modules,full_first3_pilot_exact=(mode=='full'),rows=len(paths),next_module_idx=6,finite_checked_each_row=True,quantization_called=False,source_inputs_unchanged=True,seconds=time.monotonic()-started,quality_scores='Not measured: replay has no original reference outputs',peak_gpu_bytes=torch.cuda.max_memory_allocated()))
finally:
 for store in stores:store.close()
