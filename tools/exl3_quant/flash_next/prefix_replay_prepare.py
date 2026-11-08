"""CPU-only input staging; retained packed weights and idx2 remain immutable."""
import hashlib,json,os,shutil
from pathlib import Path
import torch
from exllamav3.loader.safetensors_alt import safe_open,save_file
from disk_state import _read_row
ROOT=Path('/output');OUT=ROOT/'prefix-recovery';OLD=ROOT/'model-job-c03/repair-002-checkpoint/checkpoint-idx2'
assert not (OUT/'INPUTS.json').exists()
files=[OLD/'job.json',OLD/'original_input_ids.safetensors',ROOT/'work/args.json',Path('/task/c03-input-ids.json'),Path('/task/b1-recipe.json')]+[ROOT/'work/qtensors'/(k+'.safetensors') for k in ('model.language_model.layers.0','model.language_model.layers.1.ple','model.language_model.layers.1','model.language_model.layers.2')]
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  while data:=f.read(4*2**20):h.update(data)
 return h.hexdigest()
stat=lambda p:dict(bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns,inode=p.stat().st_ino)
assert json.loads((OLD/'job.json').read_text())==dict(next_module_idx=2,q_strategy=None,bad_rows=[])
for base in (OUT,OUT/'pilot'):
 ckpt=base/'work/ckpt';ckpt.mkdir(parents=True)
 shutil.copy2(OLD/'job.json',ckpt/'job.json')
if True:
 ckpt=OUT/'work/ckpt'
 for n in ('state.safetensors','original_input_ids.safetensors'):os.link(OLD/n,ckpt/n)
with safe_open(str(OLD/'state.safetensors')) as f:rows={f'tensor.{i}':_read_row(f,f'tensor.{i}') for i in range(3)}
assert all(torch.isfinite(v).all() for v in rows.values())
save_file(rows,str(OUT/'pilot/work/ckpt/state.safetensors'));del rows
with safe_open(str(OLD/'original_input_ids.safetensors')) as f:ids={f'tensor.{i}':_read_row(f,f'tensor.{i}') for i in range(3)}
save_file(ids,str(OUT/'pilot/work/ckpt/original_input_ids.safetensors'))
manifest=dict(files={str(p):sha(p) for p in files},protected_stats={str(p):stat(p) for p in (OLD/'state.safetensors',OLD/'original_input_ids.safetensors',ROOT/'pack/ngram_embedding.safetensors')},source_idx=2,target_idx=6,prefix_modules=[2,3,4,5],calibration_rows=2309,total_tokens=1999496)
(OUT/'INPUTS.json').write_text(json.dumps(manifest,indent=2));print('prefix inputs ready')
