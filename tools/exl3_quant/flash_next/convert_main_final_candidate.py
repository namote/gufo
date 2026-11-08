"""Convert one resumable module boundary, then side models/compile at the final boundary."""
import json
import math
import os
os.environ['EXL3_SKIP_SIDE_MODELS']='1'  # Final assembly runs separately from completed artifacts.
from pathlib import Path
import subprocess
import sys
import threading
import torch
from ragged_calibration import load_calibration, sync_calibration_metadata
from exllamav3.conversion import convert_model as cm
from exllamav3.modules import Linear
from exllamav3.modules.quant import fp16 as fp16_impl, exl3 as exl3_impl
from disk_state_large import install as install_disk_state
from capture_fast import install as install_capture_fast, STATS as capture_stats
from capture_progress import install as install_capture_progress
from capture_static_runtime import install_gram as install_capture_lt, install_moe as install_lean_moe
from capture_owner import install as install_capture_owner
from exllamav3.modules.attn import Attention

assert os.environ.get('EXL3_HIP_F32OUT_VIA_F16')=='0'
assert not fp16_impl._f32_via_f16 and not exl3_impl._f32_via_f16

root=Path('/output');work=root/'work';work.mkdir(exist_ok=True)
scratch=Path(os.environ['GUFO_STATE_SCRATCH']);scratch.mkdir(exist_ok=True)
disk_states=install_disk_state(cm,scratch)
cal_path=Path('/task/c03-input-ids.safetensors')
cal=json.loads(cal_path.with_suffix('.json').read_text())
assert cal['format']=='gufo.ragged-calibration.v1'
cm.get_file_calibration=load_calibration
ng=json.loads((root/'ngram-full/RESULT.json').read_text())
assert ng['status']=='complete_operator_verified' and ng['K']==8
screen_dir=root/'bank-pilot-fp32'
if not (screen_dir/'SUMMARY.json').exists():
    assert not (work/'ckpt/job.json').exists(),'Missing screen for existing checkpoint'
    screen_dir.mkdir(exist_ok=True)
    with (screen_dir/'run.log').open('a') as screen_log:
        subprocess.run([sys.executable,'-u','/task/bank_pilot.py'],
                       env={**os.environ,'GUFO_BANK_SCREEN_OUT':str(screen_dir)},
                       stdout=screen_log,stderr=subprocess.STDOUT,check=True,timeout=900)
screen=json.loads((screen_dir/'SUMMARY.json').read_text())
assert screen['status']=='operator_screen_pass'
assert screen['fp32_output_shortcut'] is False
checkpoint=work/'ckpt/job.json'
next_idx=json.loads(checkpoint.read_text())['next_module_idx'] if checkpoint.exists() else 0
layout=json.loads((root/'model-inventory.json').read_text())['modules']
count=sum(m['component']=='main' for m in layout)
target=min(next_idx,count-1)
args=['-i','/source','-w','/output/work','-o','/output/pack',
      '--recipe','/task/b1-recipe.json','--head_bits','8','--mtp_bits','4','--vision_bits','16',
      '--ngram_bits','8','--ngram_file','/output/pack/ngram_embedding.safetensors',
      '--cal_data',str(cal_path),'--cal_rows',str(cal['rows']),'--cal_cols',str(cal['max_columns']),
      '--checkpoint_interval','0','--max_module',str(target),'--shard_size','4096','--devices','0']
if checkpoint.exists():args+=['--resume']
parsed=cm.parser.parse_args(args)
in_args,job,ok,error=cm.prepare(parsed)
assert ok,error

def strict_rows(bad_rows,num_rows,max_fraction=0):
    if bad_rows:
        raise RuntimeError(f'Nonfinite calibration rows: {sorted(bad_rows)}; refusing to exclude data')
cm.check_bad_rows=strict_rows
original_convert=Linear.convert_exl3
audit_lock=threading.Lock()

def checked_convert(self,H_data,quant_args,*args,**kwargs):
    # MTP/vision deliberately use meta (uncalibrated) H. Main calibrated layers must
    # retain every finite calibration sample and must never switch to uncalibrated quantization.
    calibrated=not H_data['H'].is_meta
    if calibrated:
        with audit_lock:
            if not H_data.get('_gufo_audited'):
                actual_count=int(H_data.get('count_dev',H_data['count']))
                assert actual_count==cal['total_tokens'],f'Incomplete calibration: {self.key}: actual={actual_count}, expected={cal["total_tokens"]}, owner={H_data.get("first_key")}'
                assert torch.isfinite(H_data['H']).all(),f'Nonfinite Hessian: {self.key}'
                assert int(H_data['inf_nan'].sum().item())==0,f'Nonfinite calibration activations: {self.key}'
                H_data['_gufo_audited']=True
    result=original_convert(self,H_data,quant_args,*args,**kwargs)
    if calibrated and (H_data.get('q_fallback') or quant_args.get('q_fallback')):
        raise RuntimeError(f'Uncalibrated fallback forbidden in main model: {self.key}')
    proxy=result[0] if isinstance(result,tuple) else result
    assert math.isfinite(float(proxy)) and float(proxy)>=0,f'Invalid quantization proxy: {self.key}'
    assert math.isfinite(float(quant_args['g_scale'])),f'Invalid quantization scale: {self.key}'
    for name in ('suh','svh','bias'):
        value=getattr(self.inner,name)
        if value is not None:
            assert torch.isfinite(value).all(),f'Nonfinite packed {name}: {self.key}'
    return result
Linear.convert_exl3=checked_convert
original_parallel=cm.quantize_linears_parallel
def synchronized_parallel(*args,**kwargs):
    # Captured Hessians are produced on the parent stream before private worker streams read them.
    torch.cuda.synchronize()
    return original_parallel(*args,**kwargs)
cm.quantize_linears_parallel=synchronized_parallel
restore_lt=install_capture_lt(cm,'/task/capture-lt-contract.json','/task/native/capture-gram.so',cal['sha256'])
restore_owner=install_capture_owner(Attention)
install_capture_fast(cm)
restore_lean=install_lean_moe('/task/capture-static-contract.json')
install_capture_progress(cm)
try:
    cm.main(in_args,job)
except SystemExit as e:
    assert e.code in (None,0),e.code
    assert target<count-1,'Unexpected early exit during final module/compile'
    assert json.loads(checkpoint.read_text())['next_module_idx']>target
else:
    assert target==count-1
    assert os.environ.get('EXL3_SKIP_SIDE_MODELS')=='1'
    assert json.loads(checkpoint.read_text())['next_module_idx']==count
    assert in_args['vision_bits']==16 and 'final_bits' in in_args
    (root/'MAIN-COMPILE-ARGS.json').write_text(json.dumps(in_args,indent=2))
    (root/'MAIN-CONVERSION.json').write_text(json.dumps(dict(
        status='main_complete_side_models_skipped',skip_side_models=True,vision_bits=16,
        calibration_rows=cal['rows'],calibration_tokens=cal['total_tokens'],
        calibration_sha256=cal['sha256'],main_modules=count,
        next_module_idx=count,fp32_output_shortcut=False),indent=2))

finally:
    restore_lean()
    restore_owner()
    restore_lt()
    print('GUFO_CAPTURE_STATS',json.dumps(capture_stats),flush=True)
    for disk_state in disk_states:
        disk_state.close()
print('MODULE_BOUNDARY_COMPLETE',target,flush=True)
