"""Bounded GPU probe within this task's separately held natural-boundary lease."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import time
import uuid
from run_stage import OUT, TASK, LOCK, available, save
from owned_cleanup import cleanup_owned
from test.experiments.gufo_upstream_port_20261023.window import host, npu, quiet

root = OUT / 'capture-speed'
def stopped(signum, frame):
    raise KeyboardInterrupt(f'signal {signum}')
signal.signal(signal.SIGTERM, stopped)
deadline = time.monotonic() + 7200
while True:
    lease = json.loads((root / 'BOUNDARY.json').read_text())
    if lease['status'] == 'reserved':
        break
    assert lease['status'] == 'waiting' and time.monotonic() < deadline
    time.sleep(2)
os.kill(lease['pid'], 0)
assert 'capture_boundary.py' in Path(f'/proc/{lease["pid"]}/cmdline').read_bytes().decode()
locks = Path('/proc/locks').read_text().splitlines()
for path in (LOCK, OUT / 'model-job-c03/JOB.lock'):
    inode = str(path.stat().st_ino)
    assert any(line.split()[4] == str(lease['pid']) and line.split()[5].split(':')[-1] == inode
               for line in locks if len(line.split()) >= 6), 'Lease does not own required lock'
last = max((OUT / 'model-job-c03').glob('module-[0-9]*'))
previous = json.loads((last / 'STATE.json').read_text())
assert previous['status'] in ('passed', 'passed_after_cpu_checkpoint'), previous
assert previous['returned_KFD_contexts'] == [] and not previous['cleanup_errors']
# Same frozen arithmetic/data/source admission as the existing converter.
frozen = json.loads((OUT / 'model-job-c03/FROZEN.json').read_text())
for path, digest in frozen.items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
ngram_frozen = json.loads((OUT / 'ngram-full/FROZEN.json').read_text())
for path, digest in ngram_frozen['code'].items():
    assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == digest, path
stat = Path(ngram_frozen['native']).stat()
assert [stat.st_size, stat.st_mtime_ns, stat.st_ino] == ngram_frozen['native_stat']
inventory = json.loads((OUT / 'source-inventory.json').read_text())
for entry in inventory['files']:
    stat = (Path(inventory['source']) / entry['name']).stat()
    assert (stat.st_size, stat.st_mtime_ns) == (entry['bytes'], entry['mtime_ns'])
inputs = [OUT / 'model-job-c03/repair-002-checkpoint/checkpoint-idx2/state.safetensors',
          root / 'attention-input-4.safetensors', root / 'attention-input-364.safetensors']
save(root / 'PROVENANCE.json', dict(frozen=frozen, ngram_frozen=ngram_frozen,
     source_inventory_sha256=hashlib.sha256((OUT / 'source-inventory.json').read_bytes()).hexdigest(),
     inputs={str(p): dict(bytes=p.stat().st_size, mtime_ns=p.stat().st_mtime_ns,
                         inode=p.stat().st_ino) for p in inputs}, checked=time.time()))
for _ in range(30):
    h = host()
    assert h['KFD_contexts'] == [], 'Another GPU owner; do not stop it'
    if quiet(h):
        break
    time.sleep(2)
assert quiet(h) and not npu()
assert available() >= 28*2**30 and shutil.disk_usage(OUT).free >= 150*2**30
assert not (root / 'SUMMARY.json').exists()
name, label = 'gufo-capture-probe', uuid.uuid4().hex
assert subprocess.run(['docker', 'inspect', name], capture_output=True).returncode != 0
command = json.loads((OUT / 'ngram-full/STATE.json').read_text())['command'].copy()
command[command.index('--name') + 1] = name
command[command.index('--label') + 1] = 'gufo.quant.run=' + label
i = command.index('EXL3_QT_KERNEL=opt'); del command[i-1:i+1]
i = command.index('--workdir')
command[i:i] = ['-e', 'EXL3_HIP_F32OUT_VIA_F16=0', '-e', 'EXL3_ENABLE_GROUPING=0',
                '-e', 'EXL3_QUANT_THREADS=2']
command[-1] = '/task/capture_speed_probe.py'
identity = {str(TASK / n): hashlib.sha256((TASK / n).read_bytes()).hexdigest()
            for n in ('capture_fast.py', 'capture_foreach.py', 'capture_speed_probe.py', 'capture_probe_job.py')}
state = dict(status='running', started=time.time(), command=command, identity=identity,
             minimum_available_bytes=available(), admission=h, boundary_pid=lease['pid'])
save(root / 'STATE.json', state)
with (root / 'run.log').open('x') as log:
    child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
    try:
        while child.poll() is None:
            os.kill(lease['pid'], 0)
            assert json.loads((root / 'BOUNDARY.json').read_text())['status'] == 'reserved'
            mem = available()
            state['minimum_available_bytes'] = min(state['minimum_available_bytes'], mem)
            assert mem >= 8*2**30 and shutil.disk_usage(OUT).free >= 100*2**30
            assert time.time() - state['started'] < 1200, 'Affected probe deadline'
            state['updated'] = time.time(); save(root / 'STATE.json', state)
            time.sleep(2)
        assert child.returncode == 0, f'Probe exit {child.returncode}'
        assert json.loads((root / 'SUMMARY.json').read_text())['status'] == 'capture_exact_pass'
        state['status'] = 'passed'
    except BaseException as error:
        state.update(status='failed', error=repr(error))
        raise
    finally:
        state.update(cleanup_owned(name, label, child, host))
        state['finished'] = time.time()
        if state['cleanup_errors'] or state['returned_KFD_contexts']:
            state['status'] = 'cleanup_failed'
        save(root / 'STATE.json', state)
        if state['status'] == 'cleanup_failed' and 'error' not in state:
            raise RuntimeError(f"Probe cleanup failed: {state.get('cleanup_errors')}")
