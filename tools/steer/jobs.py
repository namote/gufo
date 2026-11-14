"""Keyed jobs persisted on disk, run by one worker per kind of work.

Each job lives in `WORK/jobs/KEY/`:

* `job.json`: key, operation, params, status (queued, running, done, failed,
  cancelled), progress {done, total, step, stage}, created_at, started_at,
  updated_at (Unix seconds), error, cancel_requested, last_seq and output
  (the operation's final summary, null until it finishes);
* `results.jsonl`: the results in order, each with `seq` (1, 2, ...) and
  `type`; a poll with `since=n` returns the results with seq > n;
* files the operation writes (profiles, generations, the SPSA log).

A restart reloads every job; one that was queued or running fails with
"interrupted by a restart" (its results so far are kept).

Operations that use the gufo server for long (fit, test, train, pairs,
pipeline) run one at a time in arrival order, as gufo itself serves one
collection job at a time; the quick ones (dataset, objective, profile, load)
run at once in their own thread. A pipeline runs its steps as child jobs
(`parent` is the pipeline's key) inside its own run.

Standard library only.
"""
import json
import os
from pathlib import Path
import secrets
import shutil
import threading
import time
import traceback

STATUSES = ('queued', 'running', 'done', 'failed', 'cancelled')
FINISHED = ('done', 'failed', 'cancelled')
HEAVY = ('fit', 'test', 'train', 'pairs', 'pipeline')
LIGHT = ('dataset', 'objective', 'profile', 'load')
OPERATIONS = HEAVY + LIGHT


class Cancelled(Exception):
    pass


def _write_json(path, value):
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=1) + '\n')
    os.replace(temporary, path)


class Job:
    """One job's state, its results and the handle an operation runs with."""

    def __init__(self, store, state, results=()):
        self.store = store
        self.state = state
        self.results = list(results)
        self.lock = threading.Lock()
        self.dir = store.root / state['key']
        self.cancel_event = threading.Event()
        if state.get('cancel_requested'):
            self.cancel_event.set()
        self.context = None  # set by the runner: gufo client, options
        self.ran = state['status'] in FINISHED  # loaded finished, or run_job has returned

    # Read-only views --------------------------------------------------------
    @property
    def key(self):
        return self.state['key']

    @property
    def params(self):
        return self.state['params']

    @property
    def work(self):
        return self.store.work

    def view(self, since=None):
        with self.lock:
            out = dict(self.state)
            if since is not None:
                out['results'] = [r for r in self.results if r['seq'] > since]
            return out

    # Used by operations -----------------------------------------------------
    def _save(self):
        self.state['updated_at'] = time.time()
        _write_json(self.dir / 'job.json', self.state)

    def emit(self, item):
        """Append one result; returns its seq."""
        with self.lock:
            seq = self.state['last_seq'] + 1
            item = dict(seq=seq, **item)
            with open(self.dir / 'results.jsonl', 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(item, ensure_ascii=False) + '\n')
            self.results.append(item)
            self.state['last_seq'] = seq
            self._save()
            return seq

    def progress(self, done=None, total=None, step=None, stage=None):
        with self.lock:
            p = self.state['progress']
            for key, value in (('done', done), ('total', total), ('step', step),
                               ('stage', stage)):
                if value is not None:
                    p[key] = value
            self._save()
        self.store.notify(self)

    def set_output(self, output):
        with self.lock:
            self.state['output'] = output
            self._save()

    def cancelled(self):
        return self.cancel_event.is_set()

    def check_cancel(self):
        if self.cancel_event.is_set():
            raise Cancelled()

    def wait(self, seconds):
        """Sleep, waking early on cancellation (then raise Cancelled)."""
        if self.cancel_event.wait(seconds):
            raise Cancelled()

    # Lifecycle (runner) -----------------------------------------------------
    def set_status(self, status, error=None):
        with self.lock:
            self.state['status'] = status
            if status == 'running':
                self.state['started_at'] = time.time()
            if error is not None:
                self.state['error'] = error
            self._save()
        self.store.notify(self)

    def request_cancel(self):
        with self.lock:
            if self.state['status'] in FINISHED:
                return False
            self.state['cancel_requested'] = True
            self.cancel_event.set()
            if self.state['status'] == 'queued':
                self.state['status'] = 'cancelled'
            self._save()
        self.store.notify(self)
        return True


class JobStore:
    def __init__(self, work, recover=True):
        """recover: fail the jobs a previous process left queued or running
        (the service does; the CLI, which may run beside a service on the
        same folder, does not touch other processes' jobs)."""
        self.work = Path(work)
        self.recover = recover
        self.root = self.work / 'jobs'
        self.root.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.lock = threading.Lock()
        self.changed = threading.Condition(self.lock)
        self._load()

    def _load(self):
        for path in sorted(self.root.iterdir()):
            state_path = path / 'job.json'
            if not state_path.exists():
                continue
            try:
                state = json.loads(state_path.read_text())
            except ValueError:
                continue
            results = []
            results_path = path / 'results.jsonl'
            if results_path.exists():
                for line in results_path.read_text(encoding='utf-8').splitlines():
                    try:
                        results.append(json.loads(line))
                    except ValueError:
                        break  # a torn last line from a crash
            job = Job(self, state, results)
            state['last_seq'] = results[-1]['seq'] if results else 0
            if self.recover and state['status'] in ('queued', 'running'):
                state['status'] = 'failed'
                state['error'] = 'interrupted by a restart'
                job._save()
            self.jobs[state['key']] = job

    def notify(self, job):
        with self.changed:
            self.changed.notify_all()

    def create(self, operation, params, parent=None):
        if operation not in OPERATIONS:
            raise ValueError(f'operation must be one of {OPERATIONS}')
        if not isinstance(params, dict):
            raise ValueError('params must be a JSON object')
        with self.lock:
            while True:
                key = secrets.token_hex(16)
                if key not in self.jobs:
                    break
            now = time.time()
            state = dict(key=key, operation=operation, params=params, status='queued',
                         progress=dict(done=0, total=0, step='queued', stage=None),
                         created_at=now, started_at=None, updated_at=now, error=None,
                         cancel_requested=False, last_seq=0, output=None, parent=parent)
            (self.root / key).mkdir()
            job = Job(self, state)
            job._save()
            self.jobs[key] = job
            return job

    def get(self, key):
        with self.lock:
            return self.jobs.get(key)

    def list(self):
        with self.lock:
            jobs = list(self.jobs.values())
        return sorted((j.view() for j in jobs), key=lambda s: s['created_at'], reverse=True)

    def delete(self, key):
        """Cancel an active job; forget a finished one and remove its folder.
        Returns the reply, or None for an unknown key."""
        job = self.get(key)
        if job is None:
            return None
        if job.request_cancel():
            return dict(job=key, status=job.state['status'], cancel_requested=True)
        if job.state.get('started_at') is None and job.state['status'] == 'cancelled' \
                and not job.ran:
            # Still in the runner's queue: forget it, keep the folder until popped.
            with self.lock:
                self.jobs.pop(key, None)
            return dict(job=key, deleted=True)
        with self.lock:
            self.jobs.pop(key, None)
        shutil.rmtree(job.dir, ignore_errors=True)
        return dict(job=key, deleted=True)


class Runner:
    """Runs jobs: heavy operations through one FIFO worker, light ones at once."""

    def __init__(self, store, context, operations):
        self.store, self.context, self.operations = store, context, operations
        self.queue = []
        self.cv = threading.Condition()
        self.stopping = False
        self.worker = threading.Thread(target=self._heavy_loop, name='steer-jobs', daemon=True)
        self.worker.start()

    def submit(self, operation, params):
        job = self.store.create(operation, params)
        if operation in HEAVY:
            with self.cv:
                self.queue.append(job)
                self.cv.notify_all()
        else:
            threading.Thread(target=self._run_light, args=(job,), daemon=True).start()
        return job

    def _run_light(self, job):
        try:
            self.run(job)
        except Exception:
            traceback.print_exc()

    def _heavy_loop(self):
        while True:
            with self.cv:
                while not self.queue and not self.stopping:
                    self.cv.wait()
                if self.stopping:
                    return
                job = self.queue.pop(0)
            try:
                self.run(job)
            except Exception:  # never let one job stop the worker
                traceback.print_exc()

    def stop(self, timeout=30.0):
        """Cancel every job and wait for the running one to clean up (cancel
        its gufo collection, stop its spsa.py child)."""
        with self.cv:
            self.stopping = True
            self.cv.notify_all()
        with self.store.lock:
            active = list(self.store.jobs.values())
        for job in active:
            job.request_cancel()
        self.worker.join(timeout)

    def run(self, job):
        run_job(job, self.context, self.operations)


def run_job(job, context, operations):
    """Run one job in the calling thread (the CLI's inline path and the
    runner's workers share it)."""
    try:
        _run_job(job, context, operations)
    finally:
        job.ran = True


def _run_job(job, context, operations):
    if job.cancelled():
        if job.store.get(job.key) is None:  # deleted while queued
            shutil.rmtree(job.dir, ignore_errors=True)
        elif job.state['status'] != 'cancelled':
            job.set_status('cancelled')
        return
    job.context = context
    job.set_status('running')
    job.progress(step='started')
    try:
        operations[job.state['operation']](job)
    except Cancelled:
        job.set_status('cancelled')
    except Exception as error:  # reported in the job, never raised to the server
        if job.cancelled():
            job.set_status('cancelled')
        else:
            detail = f'{type(error).__name__}: {error}'
            (job.dir / 'traceback.txt').write_text(traceback.format_exc())
            job.set_status('failed', error=detail)
    else:
        job.set_status('cancelled' if job.cancelled() else 'done')
