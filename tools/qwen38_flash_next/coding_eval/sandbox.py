"""Problem sets, code extraction and isolated test runs for the R5 coding set.

Model-written code only ever runs through DockerRunner: no network, a
read-only root, a small noexec /tmp, no capabilities, an unprivileged user,
memory/CPU/PID limits and a wall-clock kill. The image is never pulled
(`--pull never`). LocalRunner exists for code we wrote ourselves (reference
solutions, unit tests) and refuses to start unless asked for explicitly.

Problem rows (problems.jsonl): id, split, lang, category, difficulty,
entry_point, signature, prompt, tests. Each test is the harness.py test object
plus "edge": an edge-case kind (empty, boundary, duplicate, negative, unicode,
large, type, invalid, ...) or null for an ordinary case.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import uuid

HERE = Path(__file__).resolve().parent
HARNESS = HERE / 'harness.py'
# Stored scores carry this; evaluate.py --use-stored rejects other versions.
HARNESS_VERSION = hashlib.sha256(HARNESS.read_bytes()).hexdigest()[:12]

# python:3.12-slim (3.12.14) already on the development host; referenced by
# digest so nothing is pulled or silently replaced.
DEFAULT_IMAGE = ('python@sha256:'
                 '78387bc3881b8273120a12ebe6c1ab22b018ccc2c9adf565ae1ac9b536e184ea')

SPLITS = ('direction', 'strength', 'eval')
PROBLEM_KEYS = ('id', 'split', 'lang', 'category', 'difficulty', 'entry_point',
                'signature', 'prompt', 'tests')


# Data -------------------------------------------------------------------------

def read_jsonl(path, torn_tail=False):
    """torn_tail: drop an unterminated last line (an interrupted append)."""
    rows = []
    with open(path, encoding='utf-8') as handle:
        lines = handle.readlines()
    for number, line in enumerate(lines, 1):
        if torn_tail and number == len(lines) and not line.endswith('\n'):
            break
        line = line.strip()
        if line:
            try:
                rows.append(json.loads(line))
            except ValueError as error:
                raise ValueError(f'{path}:{number}: {error}') from None
    return rows


def drop_torn_tail(path):
    """Cut an unterminated last line so the next append starts clean."""
    path = Path(path)
    if not path.exists():
        return
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        path.write_bytes(data[:data.rfind(b'\n') + 1])


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + '.partial')
    with open(partial, 'w', encoding='utf-8') as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + '\n')
    os.replace(partial, path)


def validate_problem(row):
    missing = [k for k in PROBLEM_KEYS if k not in row]
    if missing:
        raise ValueError(f'{row.get("id")}: missing {missing}')
    if row['split'] not in SPLITS:
        raise ValueError(f'{row["id"]}: split {row["split"]!r}')
    if not row['tests']:
        raise ValueError(f'{row["id"]}: no tests')
    for index, test in enumerate(row['tests']):
        outcomes = [k for k in ('expect', 'raises', 'check') if k in test]
        if 'args' not in test or len(outcomes) != 1:
            raise ValueError(f'{row["id"]} test {index}: needs args and one of '
                             'expect/raises/check')
        compile(test['args'], f'{row["id"]}[{index}].args', 'eval')
        if 'expect' in test:
            compile(test['expect'], f'{row["id"]}[{index}].expect', 'eval')
        if 'check' in test:
            compile(test['check'], f'{row["id"]}[{index}].check', 'eval')
    if f'def {row["entry_point"]}(' not in row['signature']:
        raise ValueError(f'{row["id"]}: signature does not define {row["entry_point"]}')


def load_problems(path, splits=None):
    rows = read_jsonl(path)
    seen = set()
    for row in rows:
        validate_problem(row)
        if row['id'] in seen:
            raise ValueError(f'duplicate problem id {row["id"]}')
        seen.add(row['id'])
    if splits:
        rows = [r for r in rows if r['split'] in splits]
    return rows


# Code extraction --------------------------------------------------------------

FENCE = re.compile(r'^[ \t]*(`{3,}|~{3,})[ \t]*([^\s`]*)[^\n]*$')
PYTHON_TAGS = {'', 'python', 'py', 'python3', 'py3'}


def code_blocks(text):
    """Fenced blocks as (language tag, body, closed); an unclosed final block
    (a truncated answer) is kept with closed=False."""
    blocks = []
    lines = text.splitlines()
    index = 0
    while index < len(lines):
        match = FENCE.match(lines[index])
        if not match:
            index += 1
            continue
        fence, tag = match.group(1), match.group(2).lower()
        body = []
        index += 1
        closed = False
        while index < len(lines):
            stripped = lines[index].strip()
            if stripped.startswith(fence[0] * len(fence)) and not stripped.strip(fence[0]):
                closed = True
                index += 1
                break
            body.append(lines[index])
            index += 1
        blocks.append((tag, '\n'.join(body) + '\n', closed))
    return blocks


def defines(code, name):
    return re.search(rf'^\s*(async\s+)?def\s+{re.escape(name)}\s*\(', code, re.M) is not None


def extract_code(text, entry_point):
    """The answer's solution code and how it was found.

    The last closed Python block that defines the entry point wins (answers
    that revise their code put the final version last); a truncated final
    block counts only when no closed block defines it. Otherwise every Python
    block joined; otherwise the whole text when it parses as Python.
    """
    blocks = [b for b in code_blocks(text) if b[0] in PYTHON_TAGS]
    defining = [b for b in blocks if defines(b[1], entry_point)]
    closed = [b for b in defining if b[2]]
    if closed:
        return closed[-1][1], 'block'
    if defining:
        return defining[-1][1], 'unclosed-block'
    if blocks:
        return '\n'.join(b[1] for b in blocks), 'joined-blocks'
    try:
        compile(text, '<answer>', 'exec')
    except (SyntaxError, ValueError):
        return None, 'none'
    return text, 'raw-text' if defines(text, entry_point) else 'none'


# Runners ----------------------------------------------------------------------

def job_for(problem, code, timeout=2.0, run_main=False, memory_mb=512):
    tests = [{k: v for k, v in t.items() if k != 'edge'} for t in problem['tests']]
    return dict(code=code, entry_point=problem['entry_point'], tests=tests,
                timeout=timeout, run_main=run_main, memory_mb=memory_mb,
                nonce=uuid.uuid4().hex)


def wall_limit(job):
    return 30.0 + (len(job['tests']) + 2) * (float(job['timeout']) + 9.0)


def parse_report(stdout, job):
    # Only the judge knows the nonce; anything else on stdout is ignored.
    report = None
    for line in stdout.splitlines():
        try:
            candidate = json.loads(line)
        except ValueError:
            continue
        if isinstance(candidate, dict) and candidate.get('nonce') == job['nonce']:
            report = candidate
    if report is None:
        return failed_report(job, 'crash', 'harness produced no report')
    if len(report.get('tests', ())) != len(job['tests']):
        return failed_report(job, 'crash', 'harness report has the wrong test count')
    return report


def failed_report(job, status, detail):
    return {'load': {'status': status, 'detail': detail}, 'main': None,
            'tests': [{'status': status, 'detail': detail} for _ in job['tests']]}


class DockerRunner:
    name = 'docker'

    def __init__(self, image=DEFAULT_IMAGE, memory='1g', cpus='1', docker='docker'):
        self.image, self.memory, self.cpus, self.docker = image, memory, cpus, docker

    def command(self, container):
        return [
            self.docker, 'run', '--rm', '-i', '--pull', 'never',
            '--name', container,
            '--init',
            '--network', 'none',
            '--read-only',
            '--tmpfs', '/tmp:rw,noexec,nosuid,nodev,size=16m',
            '--memory', self.memory, '--memory-swap', self.memory,
            '--cpus', self.cpus,
            '--pids-limit', '64',
            '--ulimit', 'nofile=256:256',
            '--cap-drop', 'ALL',
            '--security-opt', 'no-new-privileges',
            '--user', '65534:65534',
            '--workdir', '/tmp',
            '--env', 'PYTHONDONTWRITEBYTECODE=1',
            '--env', 'PYTHONHASHSEED=0',
            '--volume', f'{HARNESS}:/sandbox/harness.py:ro',
            self.image, 'python3', '-I', '/sandbox/harness.py',
        ]

    def check(self):
        """Fail early, with the reason, when the image is not available."""
        result = subprocess.run(
            [self.docker, 'image', 'inspect', '--format', '{{.Id}}', self.image],
            capture_output=True, text=True)
        if result.returncode != 0:
            raise SystemExit(
                f'sandbox image {self.image} is not on this host and is never pulled '
                f'automatically: {result.stderr.strip()}\nPick a local image with '
                'python3 (`docker images`) and pass --image.')

    def run(self, job):
        container = f'gufo-codeval-{uuid.uuid4().hex[:12]}'
        try:
            result = subprocess.run(self.command(container), input=json.dumps(job),
                                    capture_output=True, text=True,
                                    timeout=wall_limit(job))
        except subprocess.TimeoutExpired:
            subprocess.run([self.docker, 'kill', container], capture_output=True)
            return failed_report(job, 'timeout', 'sandbox wall-clock limit')
        if result.returncode != 0 and not result.stdout.strip():
            return failed_report(job, 'crash',
                                 f'docker exit {result.returncode}: {result.stderr.strip()[-300:]}')
        return parse_report(result.stdout, job)


class LocalRunner:
    """Unsandboxed: only for code written by us, never for model output."""
    name = 'local'

    def __init__(self, trusted=False, python=sys.executable):
        if not trusted:
            raise ValueError('LocalRunner runs code unsandboxed; pass trusted=True '
                             'only for reference code')
        self.python = python

    def check(self):
        pass

    def run(self, job):
        try:
            result = subprocess.run([self.python, '-I', str(HARNESS)], input=json.dumps(job),
                                    capture_output=True, text=True, timeout=wall_limit(job))
        except subprocess.TimeoutExpired:
            return failed_report(job, 'timeout', 'wall-clock limit')
        return parse_report(result.stdout, job)


def run_jobs(runner, jobs, workers=4):
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        return list(pool.map(runner.run, jobs))


# Scoring ----------------------------------------------------------------------

def score(problem, report, method='block'):
    """Summarise one answer's report against its problem's edge labels."""
    tests = report['tests']
    statuses = [t['status'] for t in tests]
    edge = [s for s, t in zip(statuses, problem['tests']) if t.get('edge')]
    plain = [s for s, t in zip(statuses, problem['tests']) if not t.get('edge')]
    failures = [dict(index=i, edge=t.get('edge'), status=s['status'], detail=s['detail'])
                for i, (s, t) in enumerate(zip(tests, problem['tests']))
                if s['status'] != 'pass']
    return dict(
        harness=HARNESS_VERSION,
        extract=method,
        load=report['load']['status'],
        load_detail=report['load'].get('detail', ''),
        main=None if report.get('main') is None else report['main']['status'],
        tests=len(statuses), passed=statuses.count('pass'),
        edge_tests=len(edge), edge_passed=edge.count('pass'),
        plain_tests=len(plain), plain_passed=plain.count('pass'),
        all_pass=bool(statuses) and all(s == 'pass' for s in statuses),
        edge_all_pass=bool(edge) and all(s == 'pass' for s in edge),
        plain_all_pass=bool(plain) and all(s == 'pass' for s in plain),
        failures=failures,
    )


def no_code_score(problem):
    edge = sum(1 for t in problem['tests'] if t.get('edge'))
    return dict(harness=HARNESS_VERSION, extract='none', load='none', load_detail='no code found', main=None,
                tests=len(problem['tests']), passed=0, edge_tests=edge, edge_passed=0,
                plain_tests=len(problem['tests']) - edge, plain_passed=0,
                all_pass=False, edge_all_pass=False, plain_all_pass=False,
                failures=[dict(index=i, edge=t.get('edge'), status='none', detail='no code')
                          for i, t in enumerate(problem['tests'])])


def score_answers(runner, problems, answers, timeout=2.0, workers=4):
    """answers: list of (problem id, answer text). Returns score dicts in order."""
    by_id = {p['id']: p for p in problems}
    jobs, slots, scores = [], [], [None] * len(answers)
    for index, (identity, text) in enumerate(answers):
        problem = by_id[identity]
        code, method = extract_code(text or '', problem['entry_point'])
        if code is None or method == 'none':
            scores[index] = no_code_score(problem)
            continue
        jobs.append(job_for(problem, code, timeout=timeout))
        slots.append((index, problem, method))
    for (index, problem, method), report in zip(slots, run_jobs(runner, jobs, workers)):
        scores[index] = score(problem, report, method)
    return scores


def make_runner(kind, image=DEFAULT_IMAGE, trusted=False):
    if kind == 'docker':
        runner = DockerRunner(image=image)
    elif kind == 'local':
        runner = LocalRunner(trusted=trusted)
    else:
        raise ValueError(kind)
    runner.check()
    return runner
