"""CPU-only, reviewed terminal-head checkpoint commit; no numerical replay.

Preserve original idx52 directory, replace first five rows with verified packed
logits and keep remaining2304 input rows exactly as original final-module path.
Durable journal supports restart between either rename and completion markers.
"""
import argparse
from contextlib import ExitStack
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import time
from head_recovery import sha, stat_id, row_identity, check_output_header, validate_metrics
from row_checkpoint import open_rows, tensor_header, sync_directory
from run_stage import OUT,TASK,LOCK,available


def durable(path,value):
    pending=path.with_suffix('.pending')
    with pending.open('w') as f:json.dump(value,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(pending,path);sync_directory(path.parent)


def identical_link(source,target):
    assert source.is_file() and not source.is_symlink()
    assert not target.is_symlink()
    if target.exists():assert stat_id(source)==stat_id(target),'Partial stage link changed'
    else:os.link(source,target,follow_symlinks=False)


def stage_checkpoint(stage,paths,infos,ids):
    assert stage==stage.resolve()
    stage.mkdir(exist_ok=True);rows=stage/'state.rows';assert not rows.is_symlink();rows.mkdir(exist_ok=True)
    assert {p.name for p in rows.iterdir()} <= {f'{i}.safetensors' for i in range(len(paths))}
    records=[]
    for i,(p,(dtype,shape)) in enumerate(zip(paths,infos,strict=True)):
        h=tensor_header(p);assert len(h)==1 and h['tensor']['dtype']==dtype and h['tensor']['shape']==list(shape)
        target=rows/f'{i}.safetensors';identical_link(p,target)
        records.append(dict(dtype=dtype,shape=list(shape),bytes=p.stat().st_size))
    sync_directory(rows)
    durable(stage/'state.rows.json',dict(format='gufo.calibration-rows.v1',rows=records))
    identical_link(ids,stage/'original_input_ids.safetensors')
    durable(stage/'job.json',dict(next_module_idx=53,q_strategy=None,bad_rows=[]))
    new,ni=open_rows(stage/'state.safetensors');assert ni==infos
    assert all(stat_id(a)==stat_id(b) for a,b in zip(paths,new,strict=True))
    sync_directory(stage)


def commit_directories(canonical,stage,archive,journal,expected,*,after_rename=lambda _:None):
    """Idempotent two-rename transaction; callback is solely a CPU crash-test seam."""
    if canonical.exists():
        job=json.loads((canonical/'job.json').read_text())
        if job['next_module_idx']==52:
            assert not archive.exists()
            canonical.rename(archive);sync_directory(canonical.parent);sync_directory(archive.parent)
            after_rename('old_archived')
        else:assert job==dict(next_module_idx=53,q_strategy=None,bad_rows=[])
    assert archive.exists() and json.loads((archive/'job.json').read_text())['next_module_idx']==52
    if not canonical.exists():
        assert stage.exists()
        durable(journal,{**expected,'phase':'old_archived'})
        stage.rename(canonical);sync_directory(canonical.parent)
        after_rename('new_promoted')
    assert not stage.exists()
    durable(journal,{**expected,'phase':'checkpoint_committed'})


def check_source_metadata(source,audit):
    expected=audit['source_metadata_sha256']
    assert set(expected)=={'job.json','state.rows.json','original_input_ids.safetensors'}
    for name,digest in expected.items():
        p=source/name;assert not p.is_symlink() and sha(p)==digest,'Source metadata changed: '+name


def validate_runner_state(run,summary_sha256):
    assert run['status']=='completed_pending_independent_review'
    assert not run['cleanup_errors'] and not run['returned_KFD_contexts']
    assert run['protected_unchanged'] is True
    assert run['summary_sha256']==summary_sha256


def validate(review,source,result):
    assert review['status']=='head_recovery_reviewed_for_idx53'
    assert review['commit_sha256']==sha(__file__)
    audit=json.loads((OUT/'head-recovery/AUDIT.json').read_text())
    assert sha(OUT/'head-recovery/AUDIT.json')==review['audit_sha256']
    check_source_metadata(source,audit)
    assert row_identity(source/'state.safetensors')[2]==audit['rows_identity']
    assert json.loads((source/'job.json').read_text())==dict(next_module_idx=52,q_strategy=None,bad_rows=[])
    assert stat_id(OUT/'work/qtensors/lm_head.safetensors')==audit['head_pack_stat']
    run=json.loads(Path(review['runner_state']).read_text())
    assert sha(Path(review['runner_state']))==review['runner_state_sha256']
    validate_runner_state(run,review['summary_sha256'])
    summary=json.loads((result/'SUMMARY.json').read_text())
    assert sha(result/'SUMMARY.json')==review['summary_sha256']
    assert summary['status']=='head_forward_complete_pending_independent_review'
    assert (summary['rows'],summary['tokens'],summary['forward_rows'],summary['retained_input_rows'])==(2309,1999496,5,2304)
    assert summary['head_capture_repeated'] is False and summary['head_quantization_repeated'] is False
    assert summary['checkpoint_promoted'] is False and summary['state_identity']==audit['rows_identity']
    assert summary['packed_sha256']==audit['head_pack_sha256']
    assert sha(result/'BINDING.json')==summary['binding_sha256']
    binding=json.loads((result/'BINDING.json').read_text())
    assert binding['candidate_sha256']==sha(TASK/'head_recovery.py')
    assert binding['state_identity']==audit['rows_identity'] and binding['packed_sha256']==audit['head_pack_sha256']
    assert binding['files_sha256']['/output/head-recovery/AUDIT.json']==review['audit_sha256']
    paths,infos=open_rows(source/'state.safetensors')
    assert len(summary['results'])==5
    for i in range(5):
        r=json.loads((result/f'ROW-{i}.json').read_text())
        assert r==summary['results'][i] and r['index']==i and r['binding_sha256']==summary['binding_sha256']
        validate_metrics(r);p=result/f'{i}.safetensors';h=check_output_header(p,i)
        assert sha(p)==r['output_sha256'] and p.stat().st_size==r['output_bytes']
        paths[i]=p;infos[i]=('F16',tuple(h['tensor']['shape']))
    args=json.loads((result/'MAIN-COMPILE-ARGS.json').read_text())
    assert sha(result/'MAIN-COMPILE-ARGS.json')==summary['compile_args_sha256']
    assert args['vision_bits']==16 and args['head_bits']==8 and args['apply_out_scales'] is True
    assert args['work_dir']=='/output/work' and args['in_dir']=='/source' and args['out_dir']=='/output/pack'
    assert args['max_module']==52 and type(args['final_bits']) in (int,float)
    assert math.isfinite(args['final_bits']) and args['final_bits']>0
    cal=json.loads((TASK/'c03-input-ids.json').read_text())
    assert cal['rows']==2309 and cal['total_tokens']==1999496
    return paths,infos,args,cal


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--review',type=Path,required=True);a=ap.parse_args()
    review=json.loads(a.review.read_text());base=OUT/'head-recovery';work=OUT/'work'
    result=Path(review['result']);assert result==result.resolve() and base in result.parents
    canonical=work/'ckpt';stage=work/'ckpt_head_recovery_new';archive=base/'previous-canonical-idx52';journal=base/'COMMIT-JOURNAL.json'
    with ExitStack() as locks:
        for p in (LOCK,OUT/'model-job-c03/JOB.lock'):
            f=locks.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled']
        from test.experiments.gufo_upstream_port_20261023.window import host,npu
        assert not host()['KFD_contexts'] and not npu();assert available()>=8*2**30
        import shutil
        assert shutil.disk_usage(OUT).free>=100*2**30
        expected=dict(review_sha256=sha(a.review),result=str(result),canonical=str(canonical),stage=str(stage),archive=str(archive))
        if journal.exists():
            old=json.loads(journal.read_text());assert all(old[k]==v for k,v in expected.items())
        else:
            assert canonical.exists() and not stage.exists() and not archive.exists()
            durable(journal,{**expected,'phase':'building'})
        source=archive if archive.exists() else canonical
        paths,infos,args,cal=validate(review,source,result)
        failed=OUT/'model-job-c03/module-062/STATE.json'
        assert sha(failed)==review['failed_module_state_sha256']
        if not archive.exists():
            stage_checkpoint(stage,paths,infos,source/'original_input_ids.safetensors')
            durable(journal,{**expected,'phase':'prepared'})
        commit_directories(canonical,stage,archive,journal,expected)
        current,ci=open_rows(canonical/'state.safetensors');assert ci==infos
        assert all(stat_id(p)==stat_id(q) for p,q in zip(paths,current,strict=True))
        assert sha(canonical/'original_input_ids.safetensors')==sha(archive/'original_input_ids.safetensors')
        durable(OUT/'MAIN-COMPILE-ARGS.json',args)
        marker=dict(status='main_complete_side_models_skipped',skip_side_models=True,vision_bits=16,
            calibration_rows=2309,calibration_tokens=1999496,calibration_sha256=cal['sha256'],
            main_modules=53,next_module_idx=53,fp32_output_shortcut=False,
            head_recovery_review_sha256=sha(a.review),head_recovery_summary_sha256=review['summary_sha256'],
            recovered_failed_module='module-062',terminal_rows='first5_packed_logits_remaining2304_prehead_inputs')
        durable(OUT/'MAIN-CONVERSION.json',marker)
        statefile=OUT/'model-job-c03/STATE.json';before=base/'SUPERVISOR-STATE-before-head-recovery.json'
        if not before.exists():
            with before.open('xb') as f:f.write(statefile.read_bytes());f.flush();os.fsync(f.fileno())
        state=json.loads(before.read_text());state.update(status='main_complete_pending_calibrated_mtp_vision_compile',
            recovery_record=str(journal),recovered_failed_module='module-062',failed_module_receipt_preserved=True)
        durable(statefile,state)
        assert sha(failed)==review['failed_module_state_sha256']
        durable(journal,{**expected,'phase':'complete','next_module_idx':53,'finished':time.time(),
            'main_receipt_sha256':sha(OUT/'MAIN-CONVERSION.json'),'main_args_sha256':sha(OUT/'MAIN-COMPILE-ARGS.json')})

if __name__=='__main__':main()
