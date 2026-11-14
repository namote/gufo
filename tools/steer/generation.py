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
  configuration overrides them; THINK_XHIGH is the R5 think-xhigh sampling
  with thinking on); answers are graded by its sandbox (`runner`: docker;
  never run model code unsandboxed).
* `prompts`: JSONL {id, prompt|problem|text}, ungraded (the pairs
  generation job's prompts).

A configuration is {name, abliteration, logit_bias_profiles,
logit_bias_overrides, reasoning_effort, sampling, instruction} where sampling
is a JSON object merged into the request (temperature, top_p, top_k,
max_tokens, seed, chat_template_kwargs, ...). `abliteration` defaults to
{"enabled": false} (dynamic detection off). `instruction` is put before the
user turn, separated by a blank line (the pairs builder's exemplar text).

Resume (`resume`): finished rows of earlier generations are reused instead
of generated again, per (configuration, problem id):

* {"job": KEY} (or the bare key) or {"generations": DIR}: an earlier test
  job's `generations/CONFIG.jsonl`. A row is reused only when its
  `request_sha256` (the request body's hash) equals this job's request;
* {"config": NAME, "path": FILE}: an imported file (or a glob such as
  results/think-xhigh-*.jsonl, in name order) for configuration NAME:
  think_eval.py results (`build/b1/think/results/*.jsonl`), generate.py
  results (`build/b1/r5-coding/results/*.jsonl`) or test-job generations.
  Rows with `request_sha256` or generate.py's `settings` must match this
  job's request; rows with neither (think_eval.py) are trusted: the
  configuration must be the one they ran with (the math seeds already match
  think_eval.py's). `"trust": false` refuses such rows.

Math rows are graded again from their content; coding rows keep their stored
grade (generate.py's sandbox `score`, or a test row's `correct`). Reused rows
are copied into this job's generations (`resumed: true`) and emitted as rows,
so the totals and the generations folder cover the whole set.

Results: one `row` per problem (config, id, correct, tokens, reasoning
tokens, hesitations, truncated, seconds, ...) followed by that
configuration's running `totals`. Configurations run in order; within one,
`concurrency` requests run at once (the server's session count). Full
answers go to `generations/CONFIG.jsonl` in the job folder.
"""
import hashlib
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
               'reasoning_effort', 'sampling', 'instruction'}
# The R5 think-xhigh coding runs (build/b1/r5-coding/results/think-xhigh-*.jsonl).
THINK_XHIGH = {'reasoning_effort': 'xhigh', 'sampling': {
    'chat_template_kwargs': {'enable_thinking': True}, 'temperature': 1.0, 'top_p': 0.95,
    'seed': 7, 'max_tokens': 32768}}


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
    if config.get('instruction') is not None and not (
            isinstance(config['instruction'], str) and config['instruction'].strip()):
        raise ValueError(f'configuration {config["name"]}: instruction must be a non-empty string')
    return config


def with_instruction(text, config):
    return config['instruction'] + '\n\n' + text if config.get('instruction') else text


def request_sha(body):
    return hashlib.sha256(json.dumps(body, sort_keys=True, ensure_ascii=False,
                                     separators=(',', ':')).encode()).hexdigest()


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
    graded = any(r.get('correct') is not None for r in ok)
    correct = sum(bool(r.get('correct')) for r in ok) if graded else None
    return dict(n=len(rows), errors=len(rows) - len(ok), correct=correct,
                accuracy=round(correct / len(rows), 6) if rows and graded else None,
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

    def body(self, problem, model, fields, config=None):
        text = with_instruction(problem['problem'] + self.suffix, config or {})
        body = {'model': model,
                'messages': [{'role': 'user', 'content': text}],
                'max_tokens': self.max_tokens, 'seed': self.seed_base + self.index[problem['id']],
                'cache_persist': False}
        body.update(fields)
        return body

    def grade(self, problem, content, stored=None):
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

    def body(self, problem, model, fields, config=None):
        fields = dict(fields)
        abliteration = fields.pop('abliteration')
        problem = dict(problem, prompt=with_instruction(problem['prompt'], config or {}))
        return self.generate.request_body(problem, model, self.max_tokens, abliteration, fields)

    def grade(self, problem, content, stored=None):
        if stored is not None:  # an imported row: generate.py's score or a test row
            score = stored.get('score')
            if isinstance(score, dict) and 'all_pass' in score:
                return dict(correct=bool(score['all_pass']), passed=score.get('passed'),
                            tests=score.get('tests'), extract=score.get('extract'))
            if stored.get('correct') is not None:
                return {k: stored.get(k) for k in ('correct', 'passed', 'tests', 'extract')}
        score = self.sandbox.score_answers(self.runner, [problem], [(problem['id'], content)],
                                           timeout=self.test_timeout, workers=1)[0]
        return dict(correct=bool(score['all_pass']), passed=score['passed'],
                    tests=score['tests'], extract=score['extract'])

    def close(self):
        pass


class PromptSet:
    """Ungraded prompts: JSONL {id, prompt|problem|text}; `suffix` (default
    none) is appended, `max_tokens` (default 2048) and `seed_base` (default
    none: the server's sampling) as in math sets."""
    kind = 'prompts'

    def __init__(self, spec):
        self.path = spec['path']
        problems = data.read_jsonl(self.path)
        self.index = {p['id']: i for i, p in enumerate(problems)}
        self.problems = select(problems, spec)
        self.suffix = spec.get('suffix', '')
        self.max_tokens = int(spec.get('max_tokens', 2048))
        self.seed_base = spec.get('seed_base')
        for p in self.problems:
            data._prompt(p)

    def body(self, problem, model, fields, config=None):
        text = with_instruction(data._prompt(problem) + self.suffix, config or {})
        body = {'model': model, 'messages': [{'role': 'user', 'content': text}],
                'max_tokens': self.max_tokens, 'cache_persist': False}
        if self.seed_base is not None:
            body['seed'] = int(self.seed_base) + self.index[problem['id']]
        body.update(fields)
        return body

    def grade(self, problem, content, stored=None):
        return {}

    def close(self):
        pass


def make_set(spec, runner_factory):
    kind = spec.get('kind')
    if kind == 'math':
        return MathSet(spec)
    if kind == 'coding':
        return CodingSet(spec, runner_factory)
    if kind == 'prompts':
        return PromptSet(spec)
    raise ValueError('set.kind must be math, coding or prompts')


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
    body = problem_set.body(problem, model, config_fields(config), config)
    start = time.monotonic()
    try:
        reply = gufo.chat(body, timeout=request_timeout)
    except Exception as error:  # reported per row; the set goes on
        return dict(id=problem['id'], error=f'{type(error).__name__}: {error}',
                    seconds=round(time.monotonic() - start, 1)), None
    seconds = round(time.monotonic() - start, 1)
    try:
        return _row(problem_set, problem, reply, seconds, body)
    except Exception as error:  # a malformed reply or a grader failure: one error row
        return dict(id=problem['id'], error=f'{type(error).__name__}: {error}',
                    seconds=seconds), None


def _row(problem_set, problem, reply, seconds, body):
    choice = reply['choices'][0]
    message = choice.get('message') or {}
    row, full = make_row(problem_set, problem, message.get('content') or '',
                         message.get('reasoning_content') or '', reply.get('usage') or {},
                         choice.get('finish_reason'), seconds)
    if reply.get('abliteration') is not None:
        row['abliteration'] = full['abliteration'] = reply['abliteration']
    full.update(settings={k: v for k, v in body.items() if k != 'messages'},
                request_sha256=request_sha(body))
    return row, full


def make_row(problem_set, problem, content, reasoning, usage, finish_reason, seconds,
             stored=None):
    """(row, full generation row) of one answer; `stored` is an imported row
    whose grade a coding set keeps."""
    row = dict(id=problem['id'], split=problem.get('split'),
               tokens=usage.get('completion_tokens', 0),
               reasoning_tokens=(usage.get('completion_tokens_details') or {}).get('reasoning_tokens'),
               hesitations=len(HES.findall(reasoning)),
               truncated=finish_reason == 'length',
               finish_reason=finish_reason, seconds=seconds,
               think_chars=len(reasoning))
    row.update(problem_set.grade(problem, content, stored))
    full = dict(row, content=content, reasoning=reasoning, usage=usage)
    return row, full


def resume_sources(job, spec):
    """[(config or None, path, trusted, label)] from the `resume` parameter."""
    if spec is None:
        return []
    items = spec if isinstance(spec, list) else [spec]
    out = []
    for item in items:
        if isinstance(item, str):
            item = {'job': item}
        if not isinstance(item, dict):
            raise ValueError('resume takes a job key, {job}, {generations} or {config, path}')
        unknown = set(item) - {'job', 'generations', 'config', 'path', 'trust'}
        if unknown:
            raise ValueError(f'unknown resume keys {sorted(unknown)}')
        if item.get('job'):
            other = job.store.get(item['job'])
            if other is None:
                raise FileNotFoundError(f'resume: no job {item["job"]}')
            folder = other.dir / 'generations'
            if not folder.is_dir():  # it failed before generating anything
                continue
        elif item.get('generations'):
            folder = Path(item['generations'])
        else:
            folder = None
        if folder is not None:
            if not folder.is_dir():
                raise FileNotFoundError(f'resume: no generations folder {folder}')
            for path in sorted(folder.glob('*.jsonl')):
                out.append((path.stem, path, bool(item.get('trust', False)), str(path)))
            continue
        if not (item.get('config') and item.get('path')):
            raise ValueError('a resume import needs config and path')
        data.check_name(item['config'])
        paths = [Path(item['path'])]
        if any(c in item['path'] for c in '*?['):  # a glob: every match, in name order
            pattern = Path(item['path'])
            paths = sorted(pattern.parent.glob(pattern.name))  # may match nothing yet
        for path in paths:
            out.append((item['config'], path, bool(item.get('trust', True)), str(path)))
    return out


def _read_rows(path):
    rows = []
    for line in Path(path).read_text(encoding='utf-8').splitlines():
        if line.strip():
            try:
                rows.append(json.loads(line))
            except ValueError:
                break  # a torn last line from an interrupted run
    return rows


def reusable(sources, problem_set, config, model):
    """({id: full row}, stats) of the finished rows the sources hold for
    `config` that match this job's request."""
    found, stats = {}, dict(reused=0, mismatched=0, unfinished=0, sources=[])
    wanted = {p['id']: p for p in problem_set.problems}
    for name, path, trusted, label in sources:
        if name != config['name']:
            continue
        stats['sources'].append(label)
        for r in _read_rows(path):
            problem = wanted.get(r.get('id'))
            if problem is None or r['id'] in found:
                continue
            if r.get('error') or not r.get('finish_reason'):
                stats['unfinished'] += 1
                continue
            body = problem_set.body(problem, model, config_fields(config), config)
            if r.get('request_sha256'):
                ok = r['request_sha256'] == request_sha(body)
            elif isinstance(r.get('settings'), dict):
                ok = r['settings'] == {k: v for k, v in body.items() if k != 'messages'}
            else:
                ok = trusted
            if not ok:
                stats['mismatched'] += 1
                continue
            content = r.get('content') if r.get('content') is not None else (r.get('text') or '')
            seconds = r.get('seconds', r.get('elapsed_s', 0.0))
            row, full = make_row(problem_set, problem, content, r.get('reasoning') or '',
                                 r.get('usage') or {}, r.get('finish_reason'), seconds, r)
            full.update(settings={k: v for k, v in body.items() if k != 'messages'},
                        request_sha256=request_sha(body), resumed=True, resumed_from=label)
            row['resumed'] = True
            found[r['id']] = (row, full)
    stats['reused'] = len(found)
    return found, stats


def _request(replies, *args):
    try:
        replies.put(run_one(*args))
    except BaseException as error:  # run_one reports errors; this is a last guard
        replies.put((dict(id=args[3]['id'], error=f'{type(error).__name__}: {error}',
                          seconds=0.0), None))


def run_test_job(job):
    """Operation `test`. Params: set {kind: math|coding|prompts, path, split,
    ids, limit, max_tokens, suffix, seed_base, splits, image}, configs [...],
    concurrency (2), model (gufo-flash-next), request_timeout (14400 s),
    resume (see the module doc)."""
    p = job.params
    unknown = set(p) - {'set', 'configs', 'concurrency', 'model', 'request_timeout', 'resume'}
    if unknown:
        raise ValueError(f'unknown test parameters {sorted(unknown)}')
    spec = p.get('set') or {}
    configs = [check_config(c) for c in p.get('configs') or []]
    if not configs:
        raise ValueError('test needs at least one configuration')
    if len({c['name'] for c in configs}) != len(configs):
        raise ValueError('configuration names must be unique')
    problem_set = make_set(spec, job.context.coding_runner or docker_runner)
    summary, out_dir = generate(job, problem_set, configs, int(p.get('concurrency', 2)),
                                p.get('model', DEFAULT_MODEL),
                                float(p.get('request_timeout', 14400)),
                                resume_sources(job, p.get('resume')))
    job.set_output(dict(totals=summary, generations=str(out_dir),
                        configs=[c['name'] for c in configs], set_kind=problem_set.kind))


def generate(job, problem_set, configs, concurrency, model, request_timeout, sources=()):
    """Run every configuration over the set (reusing the sources' finished
    rows); returns ({config: totals}, generations folder)."""
    concurrency = max(1, concurrency)
    total = len(configs) * len(problem_set.problems)
    out_dir = job.dir / 'generations'
    out_dir.mkdir(exist_ok=True)
    finished = 0
    summary = {}
    write_lock = threading.Lock()
    for config in configs:
        job.check_cancel()
        name = config['name']
        prior, stats = reusable(sources, problem_set, config, model) if sources else ({}, None)
        if job.context.gufo is None and len(prior) < len(problem_set.problems):
            raise ValueError('test needs a gufo server (--gufo-url)')
        rows = []
        job.emit(dict(type='config', config=name, fields=config_fields(config),
                      instruction=config.get('instruction'),
                      problems=len(problem_set.problems), set_kind=problem_set.kind))
        if stats is not None:
            job.emit(dict(type='resume', config=name, **stats))
        handle = open(out_dir / f'{name}.jsonl', 'a', encoding='utf-8')

        def record(row, full):
            nonlocal finished
            row['config'] = name
            rows.append(row)
            if full is not None:
                with write_lock:
                    handle.write(json.dumps(dict(full, config=name), ensure_ascii=False) + '\n')
                    handle.flush()
            job.emit(dict(type='row', **row))
            job.emit(dict(type='totals', config=name, **totals(rows)))
            finished += 1
            job.progress(done=finished, total=total, stage=name,
                         step=f'{name}: {len(rows)}/{len(problem_set.problems)}')
        # Requests run on daemon threads, so a cancelled job (or a stopping
        # process) never waits for a generation still in flight.
        replies = queue_module.Queue()
        try:
            for problem in problem_set.problems:
                if problem['id'] in prior:
                    record(*prior[problem['id']])
            pending = 0
            todo = [q for q in problem_set.problems if q['id'] not in prior]
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
                record(row, full)
        finally:
            handle.close()
        job.check_cancel()
        _in_set_order(out_dir / f'{name}.jsonl', problem_set.problems)
        summary[name] = totals(rows)
    problem_set.close()
    return summary, out_dir


def _in_set_order(path, problems):
    """Rewrite a finished configuration's generations in the set's order (the
    requests finish in any order), so files built from them are
    deterministic."""
    order = {q['id']: i for i, q in enumerate(problems)}
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]
    lines.sort(key=lambda line: order.get(json.loads(line).get('id'), len(order)))
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(''.join(line + '\n' for line in lines), encoding='utf-8')
    temporary.replace(path)


PAIRS_PARAMS = {'name', 'prompts', 'set', 'instruction', 'abliteration', 'reasoning_effort',
                'sampling', 'mode', 'concurrency', 'model', 'request_timeout', 'resume',
                'max_chars', 'require_stop', 'replace', 'contrast', 'exclude_ids'}


def run_pairs_job(job):
    """Operation `pairs`: the model answers each prompt with and without
    `instruction` (configurations `with` and `without`, the same sampling),
    then the pairs builder makes the CAA dataset NAME from the `with`
    answers: exemplar = instruction + prompt, own = the bare prompt, the
    same continuation (mode answers: the answer; segments: every thought end
    of the reasoning). With `contrast` (default true, answers mode) it also
    writes NAME-answers: the bare prompt with the `with` answer (exemplar)
    against the `without` answer (own), the R5 own-answer contrast.

    Params: name, prompts (JSONL {id, prompt|problem|text}) or set (a test
    set spec, any kind), instruction, abliteration, reasoning_effort,
    sampling, mode (answers), concurrency (2), model, request_timeout,
    resume (as test), max_chars, require_stop (true), replace, contrast,
    exclude_ids."""
    p = dict(job.params)
    unknown = set(p) - PAIRS_PARAMS
    if unknown:
        raise ValueError(f'unknown pairs parameters {sorted(unknown)}')
    name = data.check_name(p.get('name'))
    contrast_name = data.check_name(f'{name}-answers')
    instruction = p.get('instruction')
    if not isinstance(instruction, str) or not instruction.strip():
        raise ValueError('pairs needs an instruction')
    mode = p.get('mode', 'answers')
    if mode not in ('answers', 'segments'):
        raise ValueError('mode must be answers or segments')
    replace = bool(p.get('replace', False))
    contrast = bool(p.get('contrast', True)) and mode == 'answers'
    for n in [name] + ([contrast_name] if contrast else []):
        if data.dataset_dir(job.work, n).exists() and not replace:
            raise FileExistsError(f'dataset {n} exists')
    spec = dict(p.get('set') or {'kind': 'prompts', 'path': p.get('prompts')})
    if not spec.get('path'):
        raise ValueError('pairs needs prompts (a JSONL path) or set')
    exclude = set(p.get('exclude_ids') or ())
    problem_set = make_set(spec, job.context.coding_runner or docker_runner)
    problem_set.problems = [q for q in problem_set.problems if q['id'] not in exclude]
    if not problem_set.problems:
        raise ValueError('no prompt left')
    shared = {k: p[k] for k in ('abliteration', 'reasoning_effort', 'sampling') if k in p}
    configs = [check_config(dict(shared, name='with', instruction=instruction)),
               check_config(dict(shared, name='without'))]
    summary, out_dir = generate(job, problem_set, configs, int(p.get('concurrency', 2)),
                                p.get('model', DEFAULT_MODEL),
                                float(p.get('request_timeout', 14400)),
                                resume_sources(job, p.get('resume')))
    job.progress(stage='dataset', step='building the pairs')
    # The builder reads the prompts as its problems; the suffix the set added is
    # part of the generated prompt, so it is the builder's suffix too.
    problems_path = job.dir / 'prompts.jsonl'
    problems_path.write_text(''.join(json.dumps(dict(id=q['id'], prompt=_set_text(problem_set, q)),
                                                ensure_ascii=False) + '\n'
                                     for q in problem_set.problems), encoding='utf-8')
    source = [dict(name=name, problems=str(problems_path), traces=str(out_dir / 'with.jsonl'))]
    kwargs = dict(mode=mode, require_stop=bool(p.get('require_stop', True)))
    if p.get('max_chars'):
        kwargs['max_chars'] = int(p['max_chars'])
    records, stats = data.build_pairs(source, instruction, **kwargs)
    if not records:
        raise ValueError('no pair kept (no finished answer)')
    written = [data.write_dataset(job.work, name, mode, records,
                                  dict(builder='pairs-generated', job=job.key,
                                       instruction=instruction, mode=mode),
                                  stats=dict(stats, generation=summary), replace=replace)]
    if contrast:
        records, cstats = contrast_records(name, problems_path, out_dir,
                                           bool(p.get('require_stop', True)))
        if records:
            written.append(data.write_dataset(
                job.work, contrast_name, 'answers', records,
                dict(builder='pairs-contrast', job=job.key, instruction=instruction),
                stats=cstats, replace=replace))
    for meta in written:
        job.emit(dict(type='dataset', **meta))
    job.set_output(dict(datasets=[m['name'] for m in written], totals=summary,
                        generations=str(out_dir)))


def _set_text(problem_set, problem):
    """The user text a set sends for a problem, without an instruction."""
    body = problem_set.body(problem, DEFAULT_MODEL, {'abliteration': {'enabled': False}})
    return body['messages'][-1]['content']


def contrast_records(name, problems_path, out_dir, require_stop=True):
    """Answers records: the bare prompt with the `with` answer (exemplar)
    against the `without` answer (own), per prompt both answered."""
    prompts = {q['id']: q['prompt'] for q in data.read_jsonl(problems_path)}

    def answers(config):
        out = {}
        for r in _read_rows(out_dir / f'{config}.jsonl'):
            if r.get('error') or not (r.get('content') or '').strip():
                continue
            if require_stop and r.get('finish_reason') != 'stop':
                continue
            out.setdefault(r['id'], r['content'])
        return out
    a, b = answers('with'), answers('without')
    records, stats = [], dict(pairs=0, skipped=0)
    for pid, text in prompts.items():
        if pid not in a or pid not in b:
            stats['skipped'] += 1
            continue
        key = f'{name}:{pid}'
        base = dict(pair=key, split='collection', text=text)
        records.append(dict(base, id=data.sha_text(key + ':cx'), group='exemplar', answer=a[pid]))
        records.append(dict(base, id=data.sha_text(key + ':co'), group='own', answer=b[pid]))
        stats['pairs'] += 1
    return records, stats
