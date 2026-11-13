"""Train job: SPSA over per-layer strengths with the direction fixed.

Wraps tools/qwen38_flash_next/abliteration/spsa.py unchanged: it runs as a
child process (`spsa_child.py`, which adds the gufo API key to spsa.py's
requests from the environment, never from the command line) with
`--base-url`, `--log` and `--output` set by the tool; every other spsa.py
option is passed through from params `args` (a list of strings, for example
["--pairs", "p.jsonl", "--ordinary", "o.jsonl", "--init", "0", "--additive",
"--min-strength", "-64", "--max-strength", "64", "--profile", "recheck-add"]).

Progress: spsa.py appends one JSON line per step to its log; the job emits
each as a `step` result (step, J+ and J-, theta, and the eval every
--eval-every steps) and the best evaluated settings at the end (`best`).
Cancelling the job terminates the child; rerunning with the same args in a
new job starts over (spsa.py resumes only from its own log).
"""
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
RESERVED = ('--base-url', '--log', '--output')


def _steps_from(args):
    for i, a in enumerate(args):
        if a == '--steps' and i + 1 < len(args):
            return int(args[i + 1])
        if a.startswith('--steps='):
            return int(a.split('=', 1)[1])
    return 100  # spsa.py's default


def run_train_job(job):
    """Operation `train`. Params: args (spsa.py options), python (the
    interpreter; default this one)."""
    p = job.params
    unknown = set(p) - {'args', 'python'}
    if unknown:
        raise ValueError(f'unknown train parameters {sorted(unknown)}')
    args = p.get('args')
    if not (isinstance(args, list) and all(isinstance(a, str) for a in args)):
        raise ValueError('train needs args: a list of spsa.py options')
    for a in args:
        option = a.split('=', 1)[0]
        # argparse accepts unambiguous prefixes (--out for --output).
        if option.startswith('--') and len(option) > 2 and any(
                r.startswith(option) for r in RESERVED):
            raise ValueError(f'{option} is set by the tool')
        if option.startswith('--st') and '--steps'.startswith(option) and option != '--steps':
            raise ValueError('spell --steps in full')
    gufo = job.context.gufo
    if gufo is None:
        raise ValueError('train needs a gufo server (--gufo-url)')
    steps = _steps_from(args)
    log = job.dir / 'spsa.jsonl'
    best = job.dir / 'best.json'
    command = [p.get('python') or sys.executable, '-I', str(HERE / 'spsa_child.py'),
               '--base-url', gufo.base_url, '--log', str(log), '--output', str(best), *args]
    env = dict(os.environ)
    env.pop('STEER_GUFO_API_KEY', None)
    if gufo.api_key:
        env['STEER_GUFO_API_KEY'] = gufo.api_key
    job.emit(dict(type='train', args=args, steps=steps))
    job.progress(done=0, total=steps, stage='spsa', step='starting')
    output = open(job.dir / 'spsa.out', 'w')
    env['STEER_PARENT_PID'] = str(os.getpid())  # spsa_child.py dies with this process
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
    seen = 0
    try:
        while True:
            exited = child.poll() is not None
            seen = _emit_new(job, log, seen, steps)
            if exited:
                break
            if job.cancel_event.wait(job.context.poll_seconds):
                break
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        output.close()
    job.check_cancel()
    _emit_new(job, log, seen, steps)
    if child.returncode != 0:
        tail = (job.dir / 'spsa.out').read_text(errors='replace').strip().splitlines()[-5:]
        raise RuntimeError(f'spsa.py exited with {child.returncode}: ' + ' | '.join(tail))
    result = json.loads(best.read_text()) if best.exists() else None
    if result is not None:
        job.emit(dict(type='best', **result))
    job.set_output(dict(log=str(log), best=result, best_path=str(best) if result else None))


def _emit_new(job, log, seen, steps):
    """Emit log lines after the first `seen`; returns the new count. The
    header line is skipped; a torn last line waits for the next poll."""
    if not log.exists():
        return seen
    lines = log.read_text().split('\n')
    complete = lines[:-1]  # the text after the last newline may be torn
    for line in complete[seen:]:
        seen += 1
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('type') == 'header':
            continue
        job.emit(dict(type='step', **{k: v for k, v in record.items() if k != 'type'}))
        if record.get('step', -1) >= 0:
            job.progress(done=record['step'] + 1, total=steps,
                         step=f'step {record["step"] + 1}/{steps}')
    return seen

