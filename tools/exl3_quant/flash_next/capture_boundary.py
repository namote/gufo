"""Wait for a natural GPU-owner boundary; never signal a running converter.

The conversion supervisor uses nonblocking reacquisition between modules. If it
wins the race this waiter simply waits for the next boundary. The root thread
owns this reservation and performs its bounded probe while the lock is held.
"""
import fcntl
import json
import os
import time
from pathlib import Path
from run_stage import LOCK, OUT, save

root = OUT / 'capture-speed'
root.mkdir(exist_ok=True)
assert not (root / 'BOUNDARY.json').exists(), 'Do not duplicate a boundary waiter'
save(root / 'BOUNDARY.json', dict(status='waiting', pid=os.getpid(), started=time.time()))
with LOCK.open('a') as lock:
    fcntl.flock(lock, fcntl.LOCK_EX)
    deadline = time.monotonic() + 180
    with (OUT / 'model-job-c03/JOB.lock').open('a') as job_lock:
        while True:
            try:
                fcntl.flock(job_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                assert time.monotonic() < deadline, 'Supervisor did not leave the module boundary'
                time.sleep(1)
        # Keep JOB.lock too, so an hourly manual restart cannot race the probe.
        save(root / 'BOUNDARY.json', dict(status='reserved', pid=os.getpid(), acquired=time.time(),
             checkpoint=json.loads((OUT / 'work/ckpt/job.json').read_text())))
        deadline = time.monotonic() + 3600
        while not (root / 'RELEASE.json').exists():
            assert time.monotonic() < deadline, 'Reservation expired; no converter automatically started'
            time.sleep(1)
        save(root / 'BOUNDARY.json', dict(status='released', pid=os.getpid(), finished=time.time()))
