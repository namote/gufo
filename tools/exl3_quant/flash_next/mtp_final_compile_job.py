"""Reviewed one-shot final assembly supervisor. CPU-only import; no automatic serving.

Uses fresh exclusive GPU/JOB locks after main completion, not the retired idx51
lease. All inputs are read-only container mounts. Only private output is writable.
"""
import argparse
from contextlib import ExitStack
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from run_stage import OUT, TASK, LOCK, SOURCE, RUNTIME, available, save
from owned_cleanup import cleanup_owned
from mtp_calibration_job import sha, held_flock, build_command as mtp_command, owned_path

GIB = 2**30
ROOT = OUT / 'mtp-final-assembly'
RUNS = ROOT / 'runs'
DATA = ROOT / 'data'
JOB = OUT / 'model-job-c03/JOB.lock'
RECEIPTS = ('INPUTS.json', 'INPUT-COMPLETE.json', 'DECODER-COMPLETE.json',
            'mtp.PACKED.json', 'mtp.layers.0.PACKED.json',
            'mtp.capture/MANIFEST.json', 'mtp.layers.0.capture/MANIFEST.json')


def output_path(raw):
    """Resolve explicit /output paths, rejecting traversal and host symlink aliases."""
    p = Path(raw)
    assert p.is_absolute() and p.parts[:2] == ('/', 'output'), raw
    q = OUT.joinpath(*p.parts[2:])
    assert q == q.resolve() and q != OUT and OUT in q.parents, raw
    return q


def stat_id(path):
    s = Path(path).stat()
    return [s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns]


def prelink_ngram(source, destination):
    """No copy fallback and no overwrite: both later links stay in one RW bind."""
    source, destination = Path(source), Path(destination)
    assert source.is_file() and not source.is_symlink()
    assert not destination.exists() and not destination.is_symlink()
    assert source.stat().st_dev == destination.parent.stat().st_dev, 'Ngram requires same filesystem'
    original = stat_id(source)
    os.link(source, destination)
    assert stat_id(destination) == original == stat_id(source)
    return original


def required_disk(main_files, mtp_files):
    # Existing ngram is hardlinked, not copied. Include raw vision, headers,
    # config/tokenizer files and final shard overhead conservatively.
    payload = sum(Path(p).stat().st_size for p in [*main_files, *mtp_files])
    return dict(packed_input_bytes=payload, vision_bytes=897862112,
                extra_margin_bytes=8*GIB, reserve_bytes=100*GIB,
                required_bytes=payload+897862112+108*GIB)


def build_command(prior, contract, name, label, data):
    argv = ['--review', contract['assembly_review'], '--main-work', contract['main_work'],
            '--mtp-work', contract['mtp_work'], '--work', '/mtp-run/compile-work',
            '--output', '/mtp-run/pack', '--source', '/source',
            '--ngram', '/mtp-run/ngram_embedding.safetensors', '--recipe', contract['recipe']]
    command = mtp_command(prior, dict(script='mtp_final_compile_candidate.py', argv=argv), name, label, data)
    # Final assembly owns both locks itself; no obsolete borrowed-lease assertion.
    ix = command.index('GUFO_MTP_LEASE_VERIFIED=1')
    del command[ix-1:ix+1]
    for i in reversed(range(1, len(command))):
        if command[i-1] == '-e' and command[i].startswith('EXL3_SKIP_SIDE_MODELS='):
            del command[i-1:i+1]
    ix = command.index('--workdir')
    command[ix:ix] = ['-e', 'EXL3_SKIP_SIDE_MODELS=1']
    return command


def required_pins(contract):
    names = ('mtp_final_compile_job.py', 'test_mtp_final_compile_job.py',
             'mtp_final_compile_candidate.py', 'test_mtp_final_compile_candidate.py',
             'mtp_calibration_job.py', 'mtp_stage_disk_budget.py', 'run_stage.py',
             'owned_cleanup.py', 'conversion_admission.py', 'conversion-desktop-contract.json',
             'row_checkpoint.py', 'mtp_calibration_candidate.py', 'mtp_borrowed_rows.py',
             'mtp_capture_identity.py', 'mtp_calibration_contract.py', 'mtp_calibration_gufo.py',
             'ragged_calibration.py', 'c03-input-ids.json')
    paths = {str(TASK / n) for n in names}
    paths.add(str(TASK.parent / 'gufo_upstream_port_20261023/window.py'))
    paths |= {str(OUT / n) for n in ('model-job-c03/FROZEN.json', 'source-inventory.json',
              'model-job-c03/module-060/STATE.json', 'mtp-final-compile-candidate/REVIEW.json')}
    paths |= {str(output_path(contract[n])) for n in ('assembly_review', 'recipe')}
    paths.add(contract['cpu'])
    return paths


def validate_pins(contract):
    assert required_pins(contract) <= contract['identity'].keys(), 'Missing mandatory source/evidence pins'
    for path, digest in contract['identity'].items():
        assert sha(path) == digest, 'Pinned identity changed: '+path
    for path, digest in json.loads((OUT / 'model-job-c03/FROZEN.json').read_text()).items():
        assert sha(path) == digest, path
    inv = json.loads((OUT / 'source-inventory.json').read_text())
    assert Path(inv['source']).resolve() == SOURCE.resolve()
    for item in inv['files']:
        s = (SOURCE / item['name']).stat()
        assert [s.st_size,s.st_mtime_ns] == [item['bytes'],item['mtime_ns']]


def validate_inputs(contract):
    main = output_path(contract['main_work']); mtp = output_path(contract['mtp_work'])
    assert main == OUT / 'work' and mtp == OUT / 'mtp-calibration-work/full/work'
    r = json.loads(output_path(contract['assembly_review']).read_text())
    assert r['status'] == 'approved_for_final_mtp_vision_compile'
    assert r['candidate_sha256'] == sha(TASK / 'mtp_final_compile_candidate.py')
    assert r['recipe_sha256'] == sha(output_path(contract['recipe']))
    job = json.loads((main / 'ckpt/job.json').read_text())
    assert job == {'next_module_idx':53,'q_strategy':None,'bad_rows':[]}, 'Main final checkpoint not complete'
    marker = main.parent / 'MAIN-CONVERSION.json'; args = main.parent / 'MAIN-COMPILE-ARGS.json'
    assert sha(marker) == r['main_receipt_sha256'] and sha(args) == r['main_args_sha256']
    assert json.loads(marker.read_text())['status'] == 'main_complete_side_models_skipped'
    preserved = output_path(r['protected_idx51']['checkpoint'])
    assert preserved != main / 'ckpt'
    assert json.loads((preserved / 'job.json').read_text()) == {'next_module_idx':51,'q_strategy':None,'bad_rows':[]}
    assert set(r['mtp_receipts_sha256']) == set(RECEIPTS)
    for n in RECEIPTS:
        assert sha(mtp / n) == r['mtp_receipts_sha256'][n], n
    complete = json.loads((mtp / 'DECODER-COMPLETE.json').read_text())
    assert complete['status'] == 'mtp_calibrated_pending_full_validation'
    assert sha(mtp / 'DECODER-COMPLETE.json') == r['mtp_complete_sha256']
    assert complete['rows'] == 2309 and complete['tokens'] == 1997187
    # Runtime compile code must match independent assembly review, separately from freeze.
    assert set(r['runtime_sha256']) == {'convert_model','compile'}
    for n,digest in r['runtime_sha256'].items():
        p=RUNTIME/'source/exllamav3/conversion'/(n+'.py')
        assert sha(p)==digest,n
    main_files = sorted((main/'qtensors').glob('*.safetensors'))
    mtp_files = [mtp/'qtensors'/n for n in ('mtp.safetensors','mtp.layers.0.safetensors','mtp.hyper_connection_mixer.safetensors')]
    assert main_files and all(p.is_file() for p in mtp_files)
    ngram = output_path(contract['ngram_source'])
    assert r['artifact_stats']['/mtp-run/ngram_embedding.safetensors'] == stat_id(ngram)
    return main, mtp, preserved, ngram, required_disk(main_files, mtp_files)


def protected_stats(roots):
    records = {}
    for root in roots:
        for p in [root, *sorted(root.rglob('*'))]:
            s = p.lstat()
            records[str(p)] = [s.st_dev,s.st_ino,s.st_mode,s.st_size,s.st_mtime_ns]
    return records


def guard():
    assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'], 'Latest user stop'
    lines = Path('/proc/locks').read_text()
    for p in (LOCK, JOB):
        assert held_flock(lines,os.getpid(),p.stat()), 'Exclusive whole-file lock lost: '+str(p)
    assert json.loads((OUT/'work/ckpt/job.json').read_text()) == {'next_module_idx':53,'q_strategy':None,'bad_rows':[]}


def notify(receipt, state):
    control=json.loads((OUT/'WATCH-CONTROL.json').read_text())
    if not control['enabled']: return {'status':'disabled_by_user'}
    message=(f'[Gufo 최종 조립 종료] {receipt}/STATE.json 및 {receipt}/run.log와 '
             f'{state.get("data_dir", "(출력 생성 전 실패)")}/pack/GUFO-COMPILE.json을 검토하라. '
             f'상태 {state["status"]}. 원본/idx51 보호와 GPU cleanup을 독립 확인한 뒤 승인된 마지막 정확성·성능 검증을 묶어서 진행하라. '
             '모델 서비스 자동 실행 금지, 새 양자화/기존 baseline 반복 금지. 실패면 보존된 부분 산출물부터 진단하라.')
    r=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],
                      '--message',message,'--resume'],capture_output=True,text=True,timeout=45)
    return dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True);a=ap.parse_args()
    c=json.loads(a.contract.read_text())
    assert c['status']=='reviewed_for_final_assembly' and c['purpose']=='assembly_not_benchmark'
    review=json.loads(Path(c['review']).read_text())
    assert review['status']=='reviewed_for_final_assembly' and review['contract_sha256']==sha(a.contract)
    cpu=json.loads(Path(c['cpu']).read_text())
    assert cpu['status']=='pass' and cpu['gpu_used'] is False
    assert cpu['identity'][str(Path(__file__).resolve())]==sha(__file__)
    for p,d in cpu['identity'].items(): assert sha(p)==d
    receipt=owned_path(c['receipt_dir'],RUNS);data=owned_path(c['data_dir'],DATA)
    assert not data.exists(), 'Private output is one-shot, preserve all prior attempts'
    seconds=c.get('seconds',1200);assert isinstance(seconds,int) and 0<seconds<=1200
    RUNS.mkdir(parents=True,exist_ok=True)
    state=dict(status='admitting',pid=os.getpid(),started=time.time(),contract_sha256=sha(a.contract),
               cleanup_errors=[],returned_KFD_contexts=[],container_started=False)
    recorded=False
    try:
        with ExitStack() as stack:
            for path in (RUNS/'OWNER.lock',LOCK,JOB):
                f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            assert not receipt.exists();receipt.mkdir(parents=True)
            save(receipt/'STATE.json',state);recorded=True
            child=None;before=None;name='gufo-final-'+uuid.uuid4().hex[:12];label=uuid.uuid4().hex
            def stop(signum,frame): raise KeyboardInterrupt(f'signal {signum}')
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            try:
                guard();validate_pins(c)
                main_work,mtp_work,preserved,ngram,budget=validate_inputs(c)
                assert available()>=28*GIB and shutil.disk_usage(OUT).free>=budget['required_bytes']
                from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
                from conversion_admission import wait_conversion_admission
                assert not npu()
                _,admission=wait_conversion_admission(host,quiet,TASK/'conversion-desktop-contract.json')
                assert not host()['KFD_contexts'] and not npu();guard()
                roots=[main_work,mtp_work,preserved,SOURCE]
                before=protected_stats(roots);save(receipt/'PROTECTED-BEFORE.json',before)
                data.mkdir(parents=True)
                for n in ('home','cache','triton-cache','scratch'): (data/n).mkdir()
                state['ngram_identity']=prelink_ngram(ngram,data/'ngram_embedding.safetensors')
                prior=json.loads((OUT/'model-job-c03/module-060/STATE.json').read_text())
                command=build_command(prior,c,name,label,data)
                state.update(status='running',command=command,label=label,name=name,admission=admission,
                             data_dir=str(data),disk_budget=budget,deadline_seconds=seconds,minimum_available_bytes=available())
                save(receipt/'STATE.json',state);started=time.monotonic()
                with (receipt/'run.log').open('x') as log:
                    child=subprocess.Popen(command,stdout=log,stderr=subprocess.STDOUT);state['container_started']=True
                    while child.poll() is None:
                        guard();assert not npu(),'NPU owner appeared'
                        mem=available();state['minimum_available_bytes']=min(state['minimum_available_bytes'],mem)
                        assert mem>=8*GIB,'Memory reserve below 8GiB'
                        assert shutil.disk_usage(OUT).free>=100*GIB,'Disk reserve below 100GiB'
                        assert time.monotonic()-started<seconds,'Final assembly deadline exceeded'
                        state['updated']=time.time();save(receipt/'STATE.json',state);time.sleep(2)
                    state['returncode']=child.returncode
                    assert child.returncode==0,f'Final assembly exit {child.returncode}'
                compiled=json.loads((data/'pack/GUFO-COMPILE.json').read_text())
                assert compiled['status']=='compiled_pending_final_quality_and_performance_validation'
                state.update(status='completed_pending_final_validation',compile_receipt_sha256=sha(data/'pack/GUFO-COMPILE.json'))
            except BaseException as error:
                state.update(status='failed',error=repr(error))
                raise
            finally:
                if child is not None:
                    state.update(cleanup_owned(name,label,child,host))
                    if state['cleanup_errors'] or state['returned_KFD_contexts']:state['status']='cleanup_failed'
                if before is not None:
                    try:
                        after=protected_stats(roots);save(receipt/'PROTECTED-AFTER.json',after)
                        assert before==after,'Protected source artifacts changed'
                        validate_pins(c);validate_inputs(c);guard()
                        if 'ngram_identity' in state: assert stat_id(ngram)==state['ngram_identity']
                        state['protected_unchanged']=True
                    except BaseException as error:state.update(status='protection_failed',protection_error=repr(error))
                state['finished']=time.time();save(receipt/'STATE.json',state)
    finally:
        # Callback only after relinquishing our own locks, and only after a recorded attempt.
        if recorded:
            try: save(receipt/'FOLLOWUP.json',notify(receipt,state))
            except BaseException as error:save(receipt/'FOLLOWUP.json',{'error':repr(error)})
    assert state['status']=='completed_pending_final_validation',state['status']


if __name__=='__main__':main()
