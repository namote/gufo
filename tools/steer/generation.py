"""Test job: a generation set under a list of configurations.

Sets (`set.kind`):

* `math`: JSONL {id, problem, answer, split?, level?} (build/b1/think/
  think_eval.py): the user turn is the problem plus `suffix` (default "Put
  the final answer in \\boxed{}."); correct when the last \\boxed{} of the
  answer, normalized as think_eval.py normalizes it, equals `answer`.
  Default sampling is the server's (the thinking preset); the seed is
  `seed_base` + the problem's index in the file after the `split` filter,
  as think_eval.py.
* `coding`: an R5 coding set folder (tools/qwen38_flash_next/coding_eval):
  the request is generate.py's (greedy, thinking off, 600 tokens, unless a
  configuration overrides them); answers are graded by its sandbox
  (`runner`: docker; never run model code unsandboxed).

A configuration is {name, abliteration, logit_bias_profiles,
logit_bias_overrides, reasoning_effort, sampling} where sampling is a JSON
object merged into the request (temperature, top_p, top_k, max_tokens, seed,
chat_template_kwargs, ...). `abliteration` defaults to {"enabled": false}
(dynamic detection off).

Results: one `row` per problem (config, id, correct, tokens, reasoning
tokens, hesitations, truncated, seconds, ...) followed by that
configuration's running `totals`. Configurations run in order; within one,
`concurrency` requests run at once (the server's session count). Full
answers go to `generations/CONFIG.jsonl` in the job folder.
"""
import json
from pathlib import Path
import queue as queue_module
import re
import statistics
import sys
import threading
import time

import data

HERE = Path(__file__).resolve().parent
CODING = HERE.parent / 'qwen38_flash_next' / 'coding_eval'

BOXED_SUFFIX = '\n\nPut the final answer in \\boxed{}.'
DEFAULT_MODEL = 'gufo-flash-next'
MAX_TOKENS = 16384
HES = re.compile(r'\b(wait|alternatively|hmm|let me (?:double[- ]check|re-?check|verify|'
                 r'check again|reconsider)|but wait|actually)\b', re.I)
CONFIG_KEYS = {'name', 'abliteration', 'logit_bias_profiles', 'logit_bias_overrides',
               'reasoning_effort', 'sampling'}


def boxed(text):
    i = text.rfind('\\boxed{')
    if i < 0:
        return None
    j, depth = i + 7, 1
    start = j
    while j < len(text) and depth:
        depth += {'{': 1, '}': -1}.get(text[j], 0)
        j += 1
    return text[start:j - 1].strip()


def norm(answer):
    if answer is None:
        return None
    answer = re.sub(r'\\(?:text|mathrm)\{[^}]*\}|\\[,!; ]|\$|\s|\^\\circ|\^\{\\circ\}|\\%', '',
                    answer).replace('{,}', '').replace(',', '')
    return re.sub(r'^[a-zA-Z]\w*=', '', answer)


def check_config(config):
    if not isinstance(config, dict):
        raise ValueError('a configuration is a JSON object')
    data.check_name(config.get('name'))
    unknown = set(config) - CONFIG_KEYS
    if unknown:
        raise ValueError(f'configuration {config["name"]}: unknown keys {sorted(unknown)}')
    if 'sampling' in config and not isinstance(config['sampling'], dict):
        raise ValueError(f'configuration {config["name"]}: sampling must be an object')
    return config


def config_fields(config):
    """The request fields a configuration sets."""
    out = {'abliteration': config.get('abliteration') or {'enabled': False}}
    for key in ('logit_bias_profiles', 'logit_bias_overrides', 'reasoning_effort'):
        if config.get(key) is not None:
            out[key] = config[key]
    out.update(config.get('sampling') or {})
    return out


def totals(rows):
    ok = [r for r in rows if not r.get('error')]
    tokens = [r['tokens'] for r in ok]
    return dict(n=len(rows), errors=len(rows) - len(ok),
                correct=sum(bool(r.get('correct')) for r in ok),
                accuracy=round(sum(bool(r.get('correct')) for r in ok) / len(rows), 6) if rows else None,
                tokens_mean=round(statistics.mean(tokens), 1) if tokens else None,
                tokens_median=statistics.median(tokens) if tokens else None,
                truncated=sum(bool(r.get('truncated')) for r in ok),
                hesitations_mean=round(statistics.mean(r['hesitations'] for r in ok), 2) if ok else None,
                seconds_mean=round(statistics.mean(r['seconds'] for r in ok), 1) if ok else None)


class MathSet:
    kind = 'math'

    def __init__(self, spec):
        self.path = spec['path']
        problems = data.read_jsonl(self.path)
        if spec.get('split'):  # think_eval.py seeds by the index within the split
            problems = [p for p in problems if p.get('split') == spec['split']]
        self.index = {p['id']: i for i, p in enumerate(problems)}
        self.problems = select(problems, spec)
        self.suffix = spec.get('suffix', BOXED_SUFFIX)
        self.max_tokens = int(spec.get('max_tokens', MAX_TOKENS))
        self.seed_base = int(spec.get('seed_base', 1000))
        for p in self.problems:
            if not isinstance(p.get('problem'), str) or 'answer' not in p:
                raise ValueError(f'math problem {p.get("id")} needs problem and answer')

    def body(self, problem, model, fields):
        body = {'model': model,
                'messages': [{'role': 'user', 'content': problem['problem'] + self.suffix}],
                'max_tokens': self.max_tokens, 'seed': self.seed_base + self.index[problem['id']],
                'cache_persist': False}
        body.update(fields)
        return body

    def grade(self, problem, content):
        got = norm(boxed(content))
        return dict(got=got, answer=str(problem['answer']).strip(),
                    correct=got == str(problem['answer']).strip())

    def close(self):
        pass


class CodingSet:
    kind = 'coding'

    def __init__(self, spec, runner_factory):
        if str(CODING) not in sys.path:
            sys.path.insert(0, str(CODING))
        import generate
        import sandbox
        self.generate, self.sandbox = generate, sandbox
        root = Path(spec['path'])
        problems = sandbox.load_problems(root / 'problems.jsonl', spec.get('splits'))
        self.problems = select(problems, spec)
        self.max_tokens = int(spec.get('max_tokens', 600))
        self.test_timeout = float(spec.get('test_timeout', 2.0))
        runner = spec.get('runner', 'docker')
        if runner != 'docker':
            raise ValueError('coding answers are graded only in the Docker sandbox')
        self.runner = runner_factory(spec)

    def body(self, problem, model, fields):
        fields = dict(fields)
        abliteration = fields.pop('abliteration')
        return self.generate.request_body(problem, model, self.max_tokens, abliteration, fields)

    def grade(self, problem, content):
        score = self.sandbox.score_answers(self.runner, [problem], [(problem['id'], content)],
                                           timeout=self.test_timeout, workers=1)[0]
        return dict(correct=bool(score['all_pass']), passed=score['passed'],
                    tests=score['tests'], extract=score['extract'])

    def close(self):
        pass


def select(problems, spec):
    if spec.get('split'):
        problems = [p for p in problems if p.get('split') == spec['split']]
    if spec.get('ids'):
        wanted = set(spec['ids'])
        problems = [p for p in problems if p['id'] in wanted]
    if spec.get('limit'):
        problems = problems[:int(spec['limit'])]
    if not problems:
        raise ValueError('the set selects no problem')
    return problems


def docker_runner(spec):
    import sandbox
    return sandbox.make_runner('docker', image=spec.get('image', sandbox.DEFAULT_IMAGE))


def run_one(job, problem_set, config, problem, model, request_timeout):
    gufo = job.context.gufo
    body = problem_set.body(problem, model, config_fields(config))
    start = time.monotonic()
    try:
        reply = gufo.chat(body, timeout=request_timeout)
    except Exception as error:  # reported per row; the set goes on
        return dict(id=problem['id'], error=f'{type(error).__name__}: {error}',
                    seconds=round(time.monotonic() - start, 1)), None
    seconds = round(time.monotonic() - start, 1)
    try:
        return _row(problem_set, problem, reply, seconds)
    except Exception as error:  # a malformed reply or a grader failure: one error row
        return dict(id=problem['id'], error=f'{type(error).__name__}: {error}',
                    seconds=seconds), None


def _row(problem_set, problem, reply, seconds):
    choice = reply['choices'][0]
    message = choice.get('message') or {}
    content = message.get('content') or ''
    reasoning = message.get('reasoning_content') or ''
    usage = reply.get('usage') or {}
    row = dict(id=problem['id'], split=problem.get('split'),
               tokens=usage.get('completion_tokens', 0),
               reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
               hesitations=len(HES.findall(reasoning)),
               truncated=choice.get('finish_reason') == 'length',
               finish_reason=choice.get('finish_reason'), seconds=seconds,
               think_chars=len(reasoning))
    row.update(problem_set.grade(problem, content))
    if reply.get('abliteration') is not None:
        row['abliteration'] = reply['abliteration']
    full = dict(row, content=content, reasoning=reasoning, usage=usage)
    return row, full


def _request(replies, *args):
    try:
        replies.put(run_one(*args))
    except BaseException as error:  # run_one reports errors; this is a last guard
        replies.put((dict(id=args[3]['id'], error=f'{type(error).__name__}: {error}',
                          seconds=0.0), None))


def run_test_job(job):
    """Operation `test`. Params: set {kind: math|coding, path, split, ids,
    limit, max_tokens, suffix, seed_base, splits, image}, configs [...],
    concurrency (2), model (gufo-flash-next), request_timeout (14400 s)."""
    p = job.params
    unknown = set(p) - {'set', 'configs', 'concurrency', 'model', 'request_timeout'}
    if unknown:
        raise ValueError(f'unknown test parameters {sorted(unknown)}')
    if job.context.gufo is None:
        raise ValueError('test needs a gufo server (--gufo-url)')
    spec = p.get('set') or {}
    configs = [check_config(c) for c in p.get('configs') or []]
    if not configs:
        raise ValueError('test needs at least one configuration')
    if len({c['name'] for c in configs}) != len(configs):
        raise ValueError('configuration names must be unique')
    if spec.get('kind') == 'math':
        problem_set = MathSet(spec)
    elif spec.get('kind') == 'coding':
        problem_set = CodingSet(spec, job.context.coding_runner or docker_runner)
    else:
        raise ValueError('set.kind must be math or coding')
    concurrency = max(1, int(p.get('concurrency', 2)))
    model = p.get('model', DEFAULT_MODEL)
    request_timeout = float(p.get('request_timeout', 14400))
    total = len(configs) * len(problem_set.problems)
    out_dir = job.dir / 'generations'
    out_dir.mkdir(exist_ok=True)
    finished = 0
    summary = {}
    write_lock = threading.Lock()
    for config in configs:
        job.check_cancel()
        name = config['name']
        rows = []
        job.emit(dict(type='config', config=name, fields=config_fields(config),
                      problems=len(problem_set.problems)))
        handle = open(out_dir / f'{name}.jsonl', 'a', encoding='utf-8')
        # Requests run on daemon threads, so a cancelled job (or a stopping
        # process) never waits for a generation still in flight.
        replies = queue_module.Queue()
        try:
            pending = 0
            todo = list(problem_set.problems)
            while todo or pending:
                while todo and pending < concurrency and not job.cancelled():
                    problem = todo.pop(0)
                    threading.Thread(target=_request, daemon=True, args=(
                        replies, job, problem_set, config, problem, model,
                        request_timeout)).start()
                    pending += 1
                if not pending:
                    break
                while True:
                    job.check_cancel()  # drops the in-flight requests' replies
                    try:
                        row, full = replies.get(timeout=0.5)
                        break
                    except queue_module.Empty:
                        pass
                pending -= 1
                row['config'] = name
                rows.append(row)
                if full is not None:
                    with write_lock:
                        handle.write(json.dumps(dict(full, config=name),
                                                ensure_ascii=False) + '\n')
                        handle.flush()
                job.emit(dict(type='row', **row))
                job.emit(dict(type='totals', config=name, **totals(rows)))
                finished += 1
                job.progress(done=finished, total=total, stage=name,
                             step=f'{name}: {len(rows)}/{len(problem_set.problems)}')
        finally:
            handle.close()
        summary[name] = totals(rows)
        job.check_cancel()
    problem_set.close()
    job.set_output(dict(totals=summary, generations=str(out_dir)))
