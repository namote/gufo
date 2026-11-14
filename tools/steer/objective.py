"""Objective job: reasoning traces -> SPSA objective rows (pairs, keep, guard).

Wraps tools/qwen38_flash_next/abliteration/recheck_objective.py (imported,
not copied): math sets give re-check pairs, keep rows and guard rows through
its `build`, coding sets give guard rows from answers that pass every hidden
test through its `build_code_guard` (build/b1/r10/code_guard.py's rows).

A set is {name, kind: math|coding, problems, effort, splits} plus its traces,
one of:

* `traces`: a JSONL file (think_eval.py results, a test job's
  generations/CONFIG.jsonl, generate.py results);
* `generations` (a folder) or `job` (a test job key) with `config`: that
  configuration's generations file.

`problems` is the math JSONL or the coding set folder (or its
problems.jsonl). `effort` defaults to the rows' `settings.reasoning_effort`.
Coding `splits` defaults to ["direction"]; the eval split is refused, as AIME
2025 is (both are held out and must never feed an objective).

Scoring-cost caps (defaults sized for a 16384-token training context):
`max_prefix_tokens` 12288 (boundaries with a longer reasoning prefix are
dropped) and `max_guard_tokens` 12288 with `guard_cap` skip (guard rows with
a longer trace are left out; `cut` keeps them cut at a thought boundary).
`max_prefix_chars`/`max_guard_chars` cap in characters instead (0: off; a
token cap of 0 is off too). Tokens are estimated per trace from its
characters per generated token, so the job sizes the context with a 10%
margin: `context_tokens` is the longest row (prompt + reasoning + scored
tokens) x 1.1, rounded up to 1024.
"""
import json
import math
from pathlib import Path
import sys

import data

HERE = Path(__file__).resolve().parent
ABLITERATION = HERE.parent / 'qwen38_flash_next' / 'abliteration'
if str(ABLITERATION) not in sys.path:
    sys.path.insert(0, str(ABLITERATION))

import recheck_objective as ro  # noqa: E402

MAX_PREFIX_TOKENS = 12288
MAX_GUARD_TOKENS = 12288
MARGIN = 1.1
SET_KEYS = {'name', 'kind', 'problems', 'effort', 'splits', 'traces', 'generations', 'job',
            'config'}
PARAMS = {'sets', 'opener_tokens', 'guard_tokens', 'max_boundaries', 'redundant_after', 'seed',
          'keep_first_confirmations', 'examples', 'max_prefix_tokens', 'max_prefix_chars',
          'max_guard_tokens', 'max_guard_chars', 'guard_cap'}
EFFORTS = ('low', 'medium', 'xhigh')


def traces_path(job, spec):
    if spec.get('traces'):
        return Path(spec['traces'])
    if not spec.get('config'):
        raise ValueError(f'set {spec["name"]}: generations and job need config')
    data.check_name(spec['config'])
    if spec.get('job'):
        other = job.store.get(spec['job'])
        if other is None:
            raise FileNotFoundError(f'set {spec["name"]}: no job {spec["job"]}')
        folder = other.dir / 'generations'
    elif spec.get('generations'):
        folder = Path(spec['generations'])
    else:
        raise ValueError(f'set {spec["name"]}: give traces, or generations or job with config')
    return folder / f'{spec["config"]}.jsonl'


def load_set(job, spec):
    if not isinstance(spec, dict):
        raise ValueError('a set is a JSON object')
    unknown = set(spec) - SET_KEYS
    if unknown:
        raise ValueError(f'unknown set keys {sorted(unknown)}')
    data.check_name(spec.get('name'))
    kind = spec.get('kind', 'math')
    if kind not in ('math', 'coding'):
        raise ValueError(f'set {spec["name"]}: kind must be math or coding')
    problems_path = Path(spec.get('problems') or '')
    if problems_path.is_dir():
        problems_path = problems_path / 'problems.jsonl'
    problems = {p['id']: p for p in data.read_jsonl(problems_path)}
    records = [r for r in data.read_jsonl(traces_path(job, spec)) if 'id' in r]
    effort = spec.get('effort')
    if effort is None:
        seen = {(r.get('settings') or {}).get('reasoning_effort') for r in records
                if r['id'] in problems}
        if len(seen) != 1 or None in seen:
            raise ValueError(f'set {spec["name"]}: give effort (the rows ran at {sorted(map(str, seen))})')
        effort = seen.pop()
    if effort not in EFFORTS:
        raise ValueError(f'set {spec["name"]}: effort must be one of {EFFORTS}')
    return dict(name=spec['name'], kind=kind, problems=problems, records=records, effort=effort,
                splits=tuple(spec.get('splits') or ('direction',)))


def _int(p, key, default):
    value = p.get(key, default)
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f'{key} must be a non-negative integer')
    return value


def run_objective_job(job):
    """Operation `objective` (see the module doc)."""
    p = dict(job.params)
    unknown = set(p) - PARAMS
    if unknown:
        raise ValueError(f'unknown objective parameters {sorted(unknown)}')
    opener_tokens = _int(p, 'opener_tokens', 3)
    guard_tokens = _int(p, 'guard_tokens', 1024)
    if not 1 <= opener_tokens <= 1024 or not 1 <= guard_tokens <= 1024:
        raise ValueError('opener_tokens and guard_tokens must be 1..1024')
    caps = dict(max_prefix_tokens=_int(p, 'max_prefix_tokens', MAX_PREFIX_TOKENS),
                max_prefix_chars=_int(p, 'max_prefix_chars', 0),
                max_guard_tokens=_int(p, 'max_guard_tokens', MAX_GUARD_TOKENS),
                max_guard_chars=_int(p, 'max_guard_chars', 0),
                guard_cap=p.get('guard_cap', 'skip'))
    if caps['guard_cap'] not in ('skip', 'cut'):
        raise ValueError('guard_cap must be skip or cut')
    options = dict(opener_tokens=opener_tokens, guard_tokens=guard_tokens,
                   max_boundaries=_int(p, 'max_boundaries', 8),
                   redundant_after=_int(p, 'redundant_after', 1), seed=_int(p, 'seed', 0),
                   examples=_int(p, 'examples', 3),
                   keep_first=bool(p.get('keep_first_confirmations', False)))
    specs = p.get('sets')
    if not isinstance(specs, list) or not specs:
        raise ValueError('objective needs sets')
    sets = [load_set(job, s) for s in specs]
    if len({s['name'] for s in sets}) != len(sets):
        raise ValueError('set names must be unique')
    job.progress(done=0, total=len(sets), stage='objective', step='labelling')
    pairs, keep, guard, rates = [], [], [], {}
    for index, s in enumerate(sets):
        job.check_cancel()
        for r in s['records']:
            rates[f'{s["name"]}:{r["id"]}'] = ro.rate_of(r)
        try:
            if s['kind'] == 'math':
                a, b, c, report = ro.build(
                    [(s['name'], s['problems'], s['records'], s['effort'])], **options,
                    max_prefix_tokens=caps['max_prefix_tokens'],
                    max_prefix_chars=caps['max_prefix_chars'],
                    max_guard_tokens=caps['max_guard_tokens'],
                    max_guard_chars=caps['max_guard_chars'], guard_cap=caps['guard_cap'])
                rep = report[s['name']]
            else:
                a, b = [], []
                c, counts = ro.build_code_guard(
                    s['name'], s['problems'], s['records'], s['effort'], guard_tokens,
                    caps['max_guard_tokens'], caps['max_guard_chars'], caps['guard_cap'],
                    s['splits'])
                rep = dict(counts=counts, neutral_reasons={}, examples={},
                           cost=ro.cost(c, rates) if c else ro.cost([], {}))
        except SystemExit as error:  # the held-out refusals
            raise ValueError(str(error)) from None
        pairs += a
        keep += b
        guard += c
        job.emit(dict(type='objective_set', set=s['name'], kind=s['kind'], effort=s['effort'],
                      counts=rep['counts'], neutral_reasons=rep['neutral_reasons'],
                      examples=rep['examples'], cost=rep['cost']))
        job.progress(done=index + 1, total=len(sets), step=f'{s["name"]} labelled')
    if not pairs:
        raise ValueError('no pair row: every re-check was neutral, capped or without an alternative')
    if not keep and not guard:
        raise ValueError('no keep or guard row: spsa.py needs one of them')
    out = job.dir / 'rows'
    out.mkdir(exist_ok=True)
    paths = {}
    for name, rows in (('pairs', pairs), ('keep', keep), ('guard', guard)):
        path = out / f'{name}.jsonl'
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in rows))
        paths[name] = str(path) if rows else None

    def tokens(row):
        rate = rates.get(row['chain'], 2.0)
        return round((len(row.get('reasoning') or '') + len(row['prompt'])) / rate) + row['score_max_tokens']
    longest = max(tokens(r) for r in pairs + keep + guard)
    context = int(math.ceil(longest * MARGIN / 1024) * 1024)
    labels = dict(pairs={}, keep={})
    for kind, rows in (('pairs', pairs), ('keep', keep)):
        for r in rows:
            labels[kind][r['label']] = labels[kind].get(r['label'], 0) + 1
    counts = dict(pairs=len(pairs), keep=len(keep), guard=len(guard))
    summary = dict(pairs=paths['pairs'], keep=paths['keep'], guard=paths['guard'], counts=counts,
                   labels=labels, chains=len({r['chain'] for r in pairs + keep + guard}),
                   longest_row_tokens=longest, context_tokens=context, caps=caps,
                   effort=sorted({s['effort'] for s in sets}))
    job.emit(dict(type='objective_rows', **summary))
    job.set_output(summary)
    job.progress(stage='done', step='done')

