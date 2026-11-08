"""Explicit, reviewed MTP subjob inside the already reserved idx51 lease.

CPU-only import. No reservation release, main checkpoint edits, automatic phase
advance, production integration, or timing claims. The sole writable container
bind is the reviewed private data directory, exposed as /mtp-run.
"""
import argparse
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid

from run_stage import OUT, TASK, LOCK, SOURCE, IMAGE, available, save
from owned_cleanup import cleanup_owned
from mtp_stage_disk_budget import required_budget

GIB = 2**30
BOUNDARY = OUT / 'mtp-calibration-boundary'
RUNS = OUT / 'mtp-calibration-runs'
DATA = OUT / 'mtp-calibration-work'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def proc_identity(pid):
    root = Path('/proc') / str(pid)
    fields = (root / 'stat').read_text().rsplit(')', 1)[1].split()
    assert fields[0] not in ('Z', 'X'), 'Reservation process exited'
    return dict(pid=pid, starttime_ticks=int(fields[19]),
                boot_id=Path('/proc/sys/kernel/random/boot_id').read_text().strip(),
                cmdline_sha256=sha(root / 'cmdline'))


def held_flock(text, pid, st):
    """Require granted whole-file exclusive flock, not a queued waiter/POSIX lock."""
    device = (os.major(st.st_dev), os.minor(st.st_dev), st.st_ino)
    for line in text.splitlines():
        fields = line.split()
        if len(fields) != 8 or fields[1:4] != ['FLOCK', 'ADVISORY', 'WRITE']:
            continue
        major, minor, inode = fields[5].split(':')
        if (int(fields[4]) == pid and fields[6:] == ['0', 'EOF'] and
                (int(major, 16), int(minor, 16), int(inode)) == device):
            return True
    return False


def lease_guard(expected):
    assert json.loads((OUT / 'WATCH-CONTROL.json').read_text())['enabled'], 'Latest user stop'
    assert not (BOUNDARY / 'RELEASE.json').exists(), 'Boundary released'
    reservation = json.loads((BOUNDARY / 'BOUNDARY.json').read_text())
    assert reservation['status'] == 'reserved' and reservation['pid'] == expected['pid']
    assert proc_identity(expected['pid']) == expected, 'Reservation identity changed'
    locks = Path('/proc/locks').read_text()
    for path in (LOCK, OUT / 'model-job-c03/JOB.lock'):
        assert held_flock(locks, expected['pid'], path.stat()), f'Actual lease missing: {path}'
    assert json.loads((OUT / 'work/ckpt/job.json').read_text()) == reservation['checkpoint']
    assert reservation['checkpoint'] == {'next_module_idx': 51, 'q_strategy': None, 'bad_rows': []}
    for name, expected_stat in reservation['protected_stats'].items():
        s = (OUT / 'work/ckpt' / name).stat()
        assert [s.st_size, s.st_mtime_ns, s.st_ino] == expected_stat
    return reservation


def protected_stats():
    """Metadata only; no multi-gigabyte hashes of state, packed tensors or source."""
    records = {}
    for root in (OUT / 'work', OUT / 'pack', SOURCE):
        for path in [root, *sorted(root.rglob('*'))]:
            s = path.lstat()
            records[str(path)] = [s.st_dev, s.st_ino, s.st_mode, s.st_size, s.st_mtime_ns]
    return records


def check_frozen(identity):
    required = {str(TASK / name) for name in ('mtp_calibration_job.py', 'owned_cleanup.py',
                'conversion_admission.py', 'conversion-desktop-contract.json', 'mtp_calibration_boundary.py',
                'test_mtp_calibration_job.py','mtp_stage_disk_budget.py','test_mtp_stage_disk_budget.py')}
    required.add(str(TASK.parent / 'gufo_upstream_port_20261023/window.py'))
    required |= {str(OUT / 'model-job-c03/FROZEN.json'), str(OUT / 'source-inventory.json'),
                 str(OUT / 'model-job-c03/module-060/STATE.json'), str(OUT / 'ngram-full/FROZEN.json')}
    assert required <= identity.keys(), 'Unpinned runner/source/frozen evidence'
    for path, expected in identity.items():
        assert sha(path) == expected, path
    for path, expected in json.loads((OUT / 'model-job-c03/FROZEN.json').read_text()).items():
        assert sha(path) == expected, path
    native = json.loads((OUT / 'ngram-full/FROZEN.json').read_text())
    for path, expected in native['code'].items():
        assert sha(path) == expected, path
    s = Path(native['native']).stat()
    assert [s.st_size, s.st_mtime_ns, s.st_ino] == native['native_stat']
    inv = json.loads((OUT / 'source-inventory.json').read_text())
    assert Path(inv['source']).resolve() == SOURCE.resolve()
    for entry in inv['files']:
        s = (SOURCE / entry['name']).stat()
        assert (s.st_size, s.st_mtime_ns) == (entry['bytes'], entry['mtime_ns'])


def owned_path(raw, root):
    path = Path(raw)
    assert path.is_absolute() and path == path.resolve(), 'No symlink/relative output paths'
    assert path != root and root in path.parents, f'Output outside {root}'
    return path


def disk_required(phase, state_bytes, hrefs_bytes=0, margin_bytes=0):
    if phase != 'full':
        return 150 * GIB
    assert hrefs_bytes >= int(4.05 * GIB) and margin_bytes >= 16 * GIB
    return 100 * GIB + state_bytes + hrefs_bytes + margin_bytes


def build_command(prior, contract, name, label, data):
    assert prior['status'] == 'passed' and not prior['cleanup_errors'] and not prior['returned_KFD_contexts']
    command = prior['command'].copy()
    assert command[-2:] == ['-u', '/task/convert_main.py'] and IMAGE in command
    assert command[:3] == ['docker', 'run', '--rm']
    command[command.index('--name') + 1] = name
    command[command.index('--label') + 1] = 'gufo.quant.run=' + label
    output = f'type=bind,src={OUT},dst=/output'
    assert command.count(output) == 1
    command[command.index(output)] = output + ',readonly'
    for i, value in enumerate(command):
        if i and command[i-1] == '-e':
            key = value.split('=', 1)[0]
            if key in ('HOME', 'XDG_CACHE_HOME', 'TRITON_CACHE_DIR', 'GUFO_STATE_SCRATCH'):
                command[i] = key + '=' + {'HOME':'/mtp-run/home', 'XDG_CACHE_HOME':'/mtp-run/cache',
                    'TRITON_CACHE_DIR':'/mtp-run/triton-cache', 'GUFO_STATE_SCRATCH':'/mtp-run/scratch'}[key]
    ix = command.index('--workdir')
    command[ix:ix] = ['--mount', f'type=bind,src={data},dst=/mtp-run',
                      '-e', 'GUFO_MTP_LEASE_VERIFIED=1', '-e', 'PYTHONDONTWRITEBYTECODE=1']
    # Prevent inherited broad writable mounts from weakening source protection.
    mounts = [command[i+1] for i, x in enumerate(command) if x == '--mount']
    assert [m for m in mounts if not m.endswith(',readonly')] == [f'type=bind,src={data},dst=/mtp-run']
    assert f'type=bind,src={TASK},dst=/task,readonly' in mounts
    assert f'type=bind,src={SOURCE},dst=/source,readonly' in mounts
    command[-1:] = ['/task/' + contract['script'], *contract['argv']]
    return command


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--contract', type=Path, required=True)
    args = parser.parse_args()
    contract = json.loads(args.contract.read_text())
    assert contract['status'] == 'reviewed_for_mtp_run'
    phase = contract['phase']; assert phase in ('qualifier', 'probe', 'full')
    assert contract['purpose'] == 'numerical_calibration_not_benchmark'
    script = contract['script']
    allowed = {'qualifier': {'mtp_qualify_gram.py', 'mtp_qualify_native.py'},
               'probe': {'mtp_calibration_probe.py', 'mtp_calibration_staged_probe.py'},
               'full': {'mtp_calibration_candidate.py'}}
    assert script in allowed[phase], 'Unreviewed entrypoint for phase'
    assert str(TASK / script) in contract['identity']
    argv = contract['argv']; assert isinstance(argv, list) and all(isinstance(x, str) for x in argv)
    review = json.loads(Path(contract['review']).read_text())
    assert review['status'] == 'reviewed_for_mtp_run' and review['contract_sha256'] == sha(args.contract)
    cpu = json.loads(Path(contract['cpu']).read_text())
    assert cpu['status'] == 'pass' and cpu['gpu_used'] is False
    assert cpu['identity'][str(Path(__file__).resolve())] == sha(__file__)
    for path, expected in cpu['identity'].items():
        assert sha(path) == expected, 'CPU evidence source changed: ' + path
    if phase == 'full':
        qualification = argv[argv.index('--qualification') + 1]
        assert qualification.startswith('/output/') or qualification.startswith('/task/')
        qp = OUT / qualification[len('/output/'):] if qualification.startswith('/output/') else TASK / qualification[len('/task/'):]
        assert str(qp) in contract['identity']
        assert json.loads(qp.read_text())['status'] == 'qualified_for_full_mtp_calibration'
    receipt = owned_path(contract['receipt_dir'], RUNS)
    data = owned_path(contract['data_dir'], DATA)
    RUNS.mkdir(exist_ok=True)
    with (RUNS / 'OWNER.lock').open('a') as owner:
        fcntl.flock(owner, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert not receipt.exists(), 'One-shot receipt; preserve previous attempts'
        receipt.mkdir(parents=True)
        state = dict(status='admitting', started=time.time(), pid=os.getpid(), phase=phase,
                     contract_sha256=sha(args.contract), lease=contract['lease'])
        save(receipt / 'STATE.json', state)
        child = None; before = None; name = 'gufo-mtp-' + uuid.uuid4().hex[:12]; label = uuid.uuid4().hex
        def stopped(signum, frame): raise KeyboardInterrupt(f'signal {signum}')
        signal.signal(signal.SIGTERM, stopped); signal.signal(signal.SIGINT, stopped)
        try:
            reservation = lease_guard(contract['lease']); check_frozen(contract['identity'])
            from row_checkpoint import open_rows
            rows, infos = open_rows(OUT / 'work/ckpt/state.safetensors')
            assert len(rows) == 2309 and sum(s[1] for _, s in infos) == 1999496
            assert all(d == 'F32' and s[0] == 1 and s[2:] == (4, 2560) for d, s in infos)
            state_bytes = sum(p.stat().st_size for p in rows)
            assert state_bytes == reservation['state_bytes']
            budget_info = required_budget(contract,state_bytes,data,OUT,TASK)
            budget = budget_info['required_bytes']
            state['disk_budget'] = budget_info
            assert available() >= 28 * GIB and shutil.disk_usage(OUT).free >= budget
            from host_guard import host, npu, quiet
            from conversion_admission import wait_conversion_admission
            assert not npu()
            _, admission = wait_conversion_admission(host, quiet, TASK / 'conversion-desktop-contract.json')
            assert not npu(); lease_guard(contract['lease'])
            before = protected_stats(); save(receipt / 'PROTECTED-BEFORE.json', before)
            data.mkdir(parents=True, exist_ok=True)
            for dirname in ('home', 'cache', 'triton-cache', 'scratch'):
                (data / dirname).mkdir(exist_ok=True)
            prior = json.loads((OUT / 'model-job-c03/module-060/STATE.json').read_text())
            command = build_command(prior, contract, name, label, data)
            seconds = 7200 if phase == 'full' else 1200
            state.update(status='running', command=command, label=label, name=name, admitted=time.time(),
                         admission=admission, deadline_seconds=seconds, disk_required_bytes=budget,
                         minimum_available_bytes=available())
            save(receipt / 'STATE.json', state)
            started = time.monotonic()
            with (receipt / 'run.log').open('x') as logfile:
                child = subprocess.Popen(command, stdout=logfile, stderr=subprocess.STDOUT)
                while child.poll() is None:
                    lease_guard(contract['lease']); assert not npu(), 'NPU owner appeared'
                    mem = available(); state['minimum_available_bytes'] = min(state['minimum_available_bytes'], mem)
                    assert mem >= 8 * GIB, 'Host memory reserve below 8GiB'
                    assert shutil.disk_usage(OUT).free >= 100 * GIB, 'Disk reserve below 100GiB'
                    assert time.monotonic() - started < seconds, 'Bounded MTP window expired'
                    state['updated'] = time.time(); save(receipt / 'STATE.json', state); time.sleep(2)
                state['returncode'] = child.returncode
                assert child.returncode == 0, f'MTP exit {child.returncode}'
            state['status'] = 'completed_pending_review'
        except BaseException as error:
            state.update(status='failed', error=repr(error))
            raise
        finally:
            if child is not None:
                state.update(cleanup_owned(name, label, child, host))
                if state['cleanup_errors'] or state['returned_KFD_contexts']: state['status'] = 'cleanup_failed'
            if before is not None:
                try:
                    after = protected_stats(); save(receipt / 'PROTECTED-AFTER.json', after)
                    assert before == after, 'Protected main/source metadata changed'
                    check_frozen(contract['identity']); lease_guard(contract['lease'])
                    state['protected_unchanged'] = True
                except BaseException as error:
                    state.update(status='protection_failed', protection_error=repr(error))
            state['finished'] = time.time(); save(receipt / 'STATE.json', state)
        assert state['status'] == 'completed_pending_review', state['status']


if __name__ == '__main__':
    main()
