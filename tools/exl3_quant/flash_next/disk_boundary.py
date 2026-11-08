"""CPU disk housekeeping at a quiescent, locked converter module boundary.

Never deletes model weights, diagnostic controls or an unresolved failed attempt.
Successful attempt-owned transient links can be retired only after a validated
new checkpoint. Keep metadata/logs and record every cleanup before removing it.
"""
import json,os,shutil,subprocess,time
from pathlib import Path
from row_checkpoint import open_rows,sync_directory,tensor_header
GiB=2**30
class DiskSpacePending(RuntimeError):pass


def checkpoint(out,calibration_path=None):
    root=out/'work/ckpt'
    assert root.resolve()==root and root.is_dir(), 'Refuse aliased checkpoint'
    for name in ('job.json','state.rows.json','state.rows','original_input_ids.safetensors'):
        assert not (root/name).is_symlink(), 'Refuse symlink checkpoint component'
    calibration_path=calibration_path or Path(__file__).parent/'c03-input-ids.json'
    cal=json.loads(Path(calibration_path).read_text())
    args=json.loads((out/'work/args.json').read_text())
    assert cal['rows']==args['cal_rows']==len(cal['sequences'])
    job=json.loads((root/'job.json').read_text())
    assert not job['bad_rows']
    paths,infos=open_rows(root/'state.safetensors')
    assert len(paths)==len(infos)==cal['rows']>0
    ids=tensor_header(root/'original_input_ids.safetensors')
    assert set(ids)=={f'tensor.{i}' for i in range(cal['rows'])}
    for i,((dtype,shape),sequence) in enumerate(zip(infos,cal['sequences'],strict=True)):
        assert shape[:2]==(1,sequence['tokens'])
        assert ids[f'tensor.{i}']['shape']==[1,sequence['tokens']]
        assert ids[f'tensor.{i}']['dtype'] in ('I32','I64')
    assert not (out/'work/ckpt_cpu_pending').exists(),'Unresolved checkpoint transaction'
    assert not (out/'work/ckpt_new').exists(),'Unresolved checkpoint save needs audit'
    for p in paths:assert not p.is_symlink()
    return job,paths,sum(p.stat().st_size for p in paths)


def completed_attempts(root,current_idx):
    """Only unambiguously completed attempts, never the current/failed writer."""
    for attempt in sorted(root.glob('module-[0-9][0-9][0-9]')):
        assert not attempt.is_symlink(), 'Refuse symlink attempt'
        journal=attempt/'CPU-COMMIT.json'
        # Keep unresolved commits blocking even when their transient dirs are gone.
        if journal.exists() and json.loads(journal.read_text()).get('phase')!='committed':
            raise RuntimeError('Unresolved CPU checkpoint journal: '+str(journal))
        state_path=attempt/'STATE.json'
        if not state_path.exists():continue
        state=json.loads(state_path.read_text())
        if state.get('cleanup_errors') or state.get('returned_KFD_contexts'):continue
        # Require explicit cleanup receipts; absent fields are not success.
        if state.get('cleanup_errors')!=[] or state.get('returned_KFD_contexts')!=[]:continue
        # A retired lineage can retain a later-index success log after rollback.
        # With no attempt-owned transient tree there is nothing to delete/audit.
        targets=[attempt/name for name in ('row-scratch','previous-checkpoint','partial-checkpoint')]
        if not any(p.exists() or p.is_symlink() for p in targets):continue
        if journal.exists():
            j=json.loads(journal.read_text())
            if j.get('phase')!='committed':raise RuntimeError('Unresolved CPU checkpoint journal: '+str(journal))
            done_idx=j['new_index']
            if done_idx>current_idx:raise RuntimeError('Checkpoint older than committed attempt')
            if state['status'] not in ('passed','passed_after_cpu_checkpoint','failed'):continue
            if state['status']=='failed':
                recovery=attempt/'DISK-RECOVERY.json'
                if not recovery.exists():continue
                r=json.loads(recovery.read_text())
                if r.get('status')!='committed' or r.get('next_module_idx')!=done_idx:continue
        elif state['status']=='passed':
            # The main supervisor appends this after a successful checkpoint.
            # MODULE_BOUNDARY_COMPLETE is emitted only after the converter's
            # persisted nextidx check; parse it without guessing from folder number.
            log=attempt/'run.log'
            if not log.exists():continue
            markers=[x.split() for x in log.read_text().splitlines() if x.startswith('MODULE_BOUNDARY_COMPLETE ')]
            if len(markers)!=1 or len(markers[0])!=2:continue
            done_idx=int(markers[0][1])+1
            if done_idx>current_idx:raise RuntimeError('Checkpoint behind completed attempt')
        else:continue
        yield attempt,done_idx


def tree_record(path):
    assert path.is_dir() and not path.is_symlink()
    files=[]
    for base,dirs,names in os.walk(path,followlinks=False):
        for name in dirs+names:assert not (Path(base)/name).is_symlink(),'Refuse symlink cleanup tree'
        for name in names:
            p=Path(base)/name;s=p.stat()
            files.append(dict(path=str(p),bytes=s.st_size,inode=s.st_ino,device=s.st_dev,links=s.st_nlink,mtime_ns=s.st_mtime_ns))
    return files


def budget(out,state_bytes):
    inventory=json.loads((out/'model-inventory.json').read_text())
    main=[m for m in inventory['modules'] if m['component']=='main']
    index=json.loads((out/'work/ckpt/job.json').read_text())['next_module_idx']
    last=max(m['index'] for m in main)
    assert 0<=index<=last
    # Dynamic state bytes + runtime reserve + conservative new packed/work margin.
    compile_bytes=0
    if index==last:
        packed=sum(p.stat().st_size for p in (out/'work/qtensors').glob('*.safetensors'))
        # Pinned source audit: head1.184GiB + side5.692GiB, twice with
        # 10% quant/format allowance + raw vision/aux/header slack <18GiB.
        # Round to20GiB; completed external ngram is already in pack, no copy.
        compile_bytes=packed+20*GiB
    return dict(state_bytes=state_bytes,runtime_reserve_bytes=100*GiB,
                packed_margin_bytes=16*GiB,compile_extra_bytes=compile_bytes,
                required_bytes=100*GiB+state_bytes+16*GiB+compile_bytes,next_module_idx=index)


def atomic(path,data):
    tmp=path.with_suffix('.pending')
    with tmp.open('w') as f:json.dump(data,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(tmp,path);sync_directory(path.parent)


def prepare(out,root,*,free=None,prune=None,calibration_path=None):
    out,root=Path(out),Path(root);assert root==out/'model-job-c03'
    assert root.resolve()==root and out.resolve()==out
    free=free or (lambda:shutil.disk_usage(out).free)
    if prune is None:
        def prune():
            try:
                p=subprocess.run(['docker','builder','prune','--force','--filter','until=24h'],capture_output=True,text=True,timeout=120)
                return dict(returncode=p.returncode,stdout=p.stdout,stderr=p.stderr)
            except (OSError,subprocess.TimeoutExpired) as error:
                return dict(returncode=None,error=repr(error))
    job,paths,state_bytes=checkpoint(out,calibration_path)
    live={(p.stat().st_dev,p.stat().st_ino) for p in paths}
    reports=root/'disk-boundaries';reports.mkdir(exist_ok=True)
    report=reports/f'{time.time_ns()}-idx{job["next_module_idx"]}.json'
    result=dict(status='checking',time=time.time(),checkpoint_idx=job['next_module_idx'],
                free_before_bytes=free(),deleted=[],budget=budget(out,state_bytes))
    atomic(report,result)
    for attempt,done_idx in completed_attempts(root,job['next_module_idx']):
        for name in ('row-scratch','previous-checkpoint','partial-checkpoint'):
            path=attempt/name
            assert not path.is_symlink(),'Refuse symlink cleanup tree'
            if not path.exists():continue
            files=tree_record(path)
            # A shared current inode is safe only if deleting these links leaves
            # at least one protected current checkpoint link outside this tree.
            counts={}
            for f in files:
                key=(f['device'],f['inode']);counts[key]=counts.get(key,0)+1
            for f in files:
                key=(f['device'],f['inode'])
                if key in live:assert f['links']>counts[key]
            entry=dict(path=str(path),completed_idx=done_idx,files=files,status='authorized')
            result['deleted'].append(entry);atomic(report,result)
            shutil.rmtree(path);entry['status']='removed';atomic(report,result)
    # Check the protected checkpoint again after cleanup.
    after,after_paths,after_bytes=checkpoint(out,calibration_path)
    assert after==job and after_bytes==state_bytes
    assert {(p.stat().st_dev,p.stat().st_ino) for p in after_paths}==live
    required=result['budget']['required_bytes']
    if free()<required:
        result['cache_prune']=prune();atomic(report,result)
    result['free_after_bytes']=free()
    result['status']='ready' if result['free_after_bytes']>=required else 'needs_space_before_module'
    atomic(report,result)
    if result['status']!='ready':
        raise DiskSpacePending(f'Disk space needed before module {job["next_module_idx"]}: free={result["free_after_bytes"]/GiB:.2f}GiB required={required/GiB:.2f}GiB; receipt={report}')
    return result
