"""Datasets: folders of collector records (collect.cpp formats) and their builders.

A dataset is `WORK/datasets/NAME/` with

* `records.jsonl`: one record per line in a collector mode's input format
  (`collect`: id, split, group refusal|ordinary, text, thinking; `answers`:
  id, split, group exemplar|own, pair, text, answer; `segments`: id, split,
  group exemplar|own, pair, text, segments, select);
* `dataset.json`: `kind` (the mode), `groups` (record counts per group),
  `source` (how it was built), `created_at`, `records`, `data_sha256` (the
  records as compact JSON lines, as gufo hashes a collection's records),
  `control` (the name of its shuffled-label control dataset, or null) and
  `stats`.

Standard library only.
"""
import hashlib
import json
from pathlib import Path
import random
import re
import time

MODES = ('collect', 'answers', 'segments')
GROUPS = {'collect': ('refusal', 'ordinary'), 'answers': ('exemplar', 'own'),
          'segments': ('exemplar', 'own')}
NAME = re.compile(r'[A-Za-z0-9_.-]{1,64}')
RECORD_ID = re.compile(r'[0-9a-f\-plainthk]+')
# Opener of a re-check thought (build/b1/think/build_think_dirs.py).
RECHECK = (r'^\W*(wait|hmm|alternatively|actually|but wait|hold on|hmm+|double[- ]check|'
           r'let me (?:double[- ]check|re-?check|verify|check|reconsider|re-?examine|re-?read|'
           r're-?compute|re-?count|re-?verify|re-?think|make sure))\b')
# Under ~21k tokens at the >= 1.72 characters per token seen on AIME traces,
# so prompt + reasoning stay inside the segments mode's 24576-token context.
MAX_CHARS = 36000
MIN_EACH = 3


def sha_text(text):
    return hashlib.sha256(text.encode()).hexdigest()


def check_name(name):
    if not isinstance(name, str) or not NAME.fullmatch(name) or name in ('.', '..'):
        raise ValueError(f'invalid name {name!r}: 1-64 letters, digits, _, - or .')
    return name


def read_jsonl(path):
    rows = []
    with open(path, encoding='utf-8') as handle:
        for number, line in enumerate(handle, 1):
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except ValueError as error:
                    raise ValueError(f'{path}:{number}: {error}') from None
    return rows


def records_text(records):
    """The records as compact JSON lines, the bytes gufo hashes (data_sha256)."""
    return ''.join(json.dumps(r, ensure_ascii=False, separators=(',', ':')) + '\n'
                   for r in records)


def infer_mode(record):
    if 'segments' in record:
        return 'segments'
    if 'answer' in record:
        return 'answers'
    return 'collect'


def validate(mode, records):
    """The checks collect.cpp and gufo's job endpoint apply before running."""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if not records:
        raise ValueError('a dataset needs at least one record')
    ids, split = set(), None
    for index, r in enumerate(records):
        where = f'record {index}'
        identity = r.get('id')
        if not isinstance(identity, str) or not RECORD_ID.fullmatch(identity):
            raise ValueError(f'{where}: id must use 0-9 a-f - p l a i n t h k')
        if identity in ids:
            raise ValueError(f'{where}: duplicate id {identity}')
        ids.add(identity)
        if split is None:
            split = r.get('split')
        if not isinstance(r.get('split'), str) or r.get('split') != split:
            raise ValueError(f'{where}: every record needs the same split')
        if r.get('group') not in GROUPS[mode]:
            raise ValueError(f'{where}: group must be one of {GROUPS[mode]}')
        if not isinstance(r.get('text'), str) or not r['text']:
            raise ValueError(f'{where}: text must be a non-empty string')
        if mode == 'collect':
            if not isinstance(r.get('thinking'), bool):
                raise ValueError(f'{where}: collect records need thinking (a boolean)')
            continue
        if not isinstance(r.get('pair'), str) or not r['pair']:
            raise ValueError(f'{where}: pair must be a non-empty string')
        if mode == 'answers':
            if not isinstance(r.get('answer'), str) or not r['answer']:
                raise ValueError(f'{where}: answers records need an answer')
            continue
        segments, select = r.get('segments'), r.get('select')
        if not (isinstance(segments, list) and len(segments) >= 2 and
                all(isinstance(s, str) and s for s in segments)):
            raise ValueError(f'{where}: segments must be >= 2 non-empty strings')
        if not (isinstance(select, list) and select and
                all(isinstance(k, int) and not isinstance(k, bool) and
                    0 <= k < len(segments) - 1 for k in select)):
            raise ValueError(f'{where}: select must name segments followed by another')


def group_counts(records):
    counts = {}
    for r in records:
        counts[r['group']] = counts.get(r['group'], 0) + 1
    return counts


def datasets_root(work):
    return Path(work) / 'datasets'


def dataset_dir(work, name):
    return datasets_root(work) / check_name(name)


def write_dataset(work, name, mode, records, source, control=None, stats=None,
                  replace=False):
    """Write one dataset atomically (a temporary folder, then a rename)."""
    validate(mode, records)
    path = dataset_dir(work, name)
    if path.exists() and not replace:
        raise FileExistsError(f'dataset {name} exists')
    text = records_text(records)
    meta = dict(name=name, kind=mode, groups=group_counts(records), source=source,
                created_at=time.time(), records=len(records),
                data_sha256=sha_text(text), control=control, stats=stats or {})
    temporary = path.with_name(f'.{name}.tmp')
    if temporary.exists():
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
    temporary.mkdir(parents=True)
    (temporary / 'records.jsonl').write_text(text, encoding='utf-8')
    (temporary / 'dataset.json').write_text(json.dumps(meta, indent=1) + '\n')
    if path.exists():
        for child in path.iterdir():
            child.unlink()
        path.rmdir()
    temporary.rename(path)
    return meta


def load_dataset(work, name):
    path = dataset_dir(work, name)
    if not (path / 'dataset.json').exists():
        raise FileNotFoundError(f'no dataset {name}')
    meta = json.loads((path / 'dataset.json').read_text())
    return meta, read_jsonl(path / 'records.jsonl')


def list_datasets(work):
    root = datasets_root(work)
    if not root.exists():
        return []
    out = []
    for path in sorted(root.iterdir()):
        if path.is_dir() and (path / 'dataset.json').exists():
            out.append(json.loads((path / 'dataset.json').read_text()))
    return out


# Builders ---------------------------------------------------------------------

def thoughts(reasoning, max_chars=MAX_CHARS):
    """Reasoning cut to max_chars and split at blank lines; each thought keeps
    its trailing blank line, so the segments concatenate to the cut text."""
    text = reasoning[:max_chars]
    parts = text.split('\n\n')
    segs = [p + '\n\n' for p in parts[:-1]] + ([parts[-1]] if parts[-1] else [])
    return [s for s in segs if s.strip()]


def _prompt(problem):
    for key in ('problem', 'prompt', 'text'):
        if isinstance(problem.get(key), str) and problem[key]:
            return problem[key]
    raise ValueError(f'problem {problem.get("id")} has no problem, prompt or text')


def _load_source(source):
    """{name, problems, traces} -> (name, {id: problem}, [trace rows])."""
    for key in ('name', 'problems', 'traces'):
        if not source.get(key):
            raise ValueError(f'a source needs {key}')
    check_name(source['name'])
    problems = {p['id']: p for p in read_jsonl(source['problems'])}
    return source['name'], problems, read_jsonl(source['traces'])


def build_traces(sources, suffix='', max_chars=MAX_CHARS, min_each=MIN_EACH,
                 opener=RECHECK, seed=0, reasoning_field='reasoning',
                 exclude_ids=()):
    """Thought-boundary records (segments mode) from reasoning traces, as
    build_think_dirs.py's `recheck` and `recheck-shuf`.

    Per trace, the rows at the end of thought k (which predict thought k+1's
    first token): group exemplar = ends followed by a thought whose start
    matches `opener`; group own = the other ends. Same text and tokens, only
    `select` differs. Traces with fewer than `min_each` of either are
    skipped. The control permutes the labels within each kept trace (one
    random.Random(seed) over all traces in source order).

    Returns (records, control_records, stats)."""
    pattern = re.compile(opener, re.I)
    rng = random.Random(seed)
    exclude = set(exclude_ids)
    records, control = [], []
    stats = dict(traces=0, opener_rows=0, other_rows=0, skipped=0, max_chars=max_chars,
                 min_each=min_each)
    for source in sources:
        name, problems, traces = _load_source(source)
        for r in traces:
            if not r.get(reasoning_field):
                continue
            if r['id'] in exclude:
                continue
            if r['id'] not in problems:
                raise ValueError(f'{name}: trace {r["id"]} has no problem')
            segs = thoughts(r[reasoning_field], max_chars)
            if len(segs) < 2:
                stats['skipped'] += 1
                continue
            text = _prompt(problems[r['id']]) + suffix
            key = f'{name}:{r["id"]}'
            ks = range(len(segs) - 1)
            hit = [k for k in ks if pattern.match(segs[k + 1])]
            miss = [k for k in ks if not pattern.match(segs[k + 1])]
            if len(hit) < min_each or len(miss) < min_each:
                stats['skipped'] += 1
                continue
            stats['traces'] += 1
            stats['opener_rows'] += len(hit)
            stats['other_rows'] += len(miss)
            base = {'pair': key, 'segments': segs, 'split': 'collection'}
            records.append({**base, 'id': sha_text(key + ':rx'), 'group': 'exemplar',
                            'text': text, 'select': hit})
            records.append({**base, 'id': sha_text(key + ':ro'), 'group': 'own',
                            'text': text, 'select': miss})
            every = list(ks)
            rng.shuffle(every)
            sx, so = sorted(every[:len(hit)]), sorted(every[len(hit):])
            control.append({**base, 'id': sha_text(key + ':sx'), 'group': 'exemplar',
                            'text': text, 'select': sx})
            control.append({**base, 'id': sha_text(key + ':so'), 'group': 'own',
                            'text': text, 'select': so})
    return records, control, stats


def build_pairs(sources, instruction, mode='answers', suffix='', max_chars=MAX_CHARS,
                answer_field=None, require_stop=True, exclude_ids=()):
    """CAA pairs: the same continuation teacher-forced under instruction +
    prompt (exemplar) and under the bare prompt (own).

    mode `answers` (R5b, build/b1/r5b/build_dirs.py): the source's answer
    (`answer_field`, default `text`, then `content`); rows
    whose `finish_reason` is present and not `stop` are skipped unless
    require_stop is False. mode `segments` (build_think_dirs.py `depth`):
    every thought end of the source's reasoning (`answer_field`, default
    `reasoning`), cut to max_chars.

    Returns (records, stats)."""
    if mode not in ('answers', 'segments'):
        raise ValueError('pairs mode must be answers or segments')
    if not instruction:
        raise ValueError('an instruction is required')
    exclude = set(exclude_ids)
    records = []
    stats = dict(pairs=0, skipped=0)
    for source in sources:
        name, problems, rows = _load_source(source)
        for r in rows:
            if r['id'] in exclude:
                continue
            if r['id'] not in problems:
                raise ValueError(f'{name}: row {r["id"]} has no problem')
            text = _prompt(problems[r['id']]) + suffix
            key = f'{name}:{r["id"]}'
            if mode == 'answers':
                # generate.py rows hold the answer in text, think_eval and test-job rows
                # in content (their `answer` is the reference, never the default).
                fields = [answer_field] if answer_field else ['text', 'content']
                answer = next((r[f] for f in fields if isinstance(r.get(f), str)
                               and r[f].strip()), None)
                if answer is None or (require_stop and r.get('finish_reason', 'stop') != 'stop'):
                    stats['skipped'] += 1
                    continue
                base = {'pair': key, 'answer': answer, 'split': 'collection'}
                records.append({**base, 'id': sha_text(key + ':x'), 'group': 'exemplar',
                                'text': instruction + '\n\n' + text})
                records.append({**base, 'id': sha_text(key + ':o'), 'group': 'own', 'text': text})
            else:
                reasoning = r.get(answer_field or 'reasoning')
                segs = thoughts(reasoning or '', max_chars)
                if len(segs) < 2:
                    stats['skipped'] += 1
                    continue
                ks = list(range(len(segs) - 1))
                base = {'pair': key, 'segments': segs, 'split': 'collection'}
                records.append({**base, 'id': sha_text(key + ':dx'), 'group': 'exemplar',
                                'text': instruction + '\n\n' + text, 'select': ks})
                records.append({**base, 'id': sha_text(key + ':do'), 'group': 'own',
                                'text': text, 'select': ks})
            stats['pairs'] += 1
    return records, stats


def run_dataset_job(job):
    """Operation `dataset`: params {builder: traces|pairs|import, name, ...};
    see README.md. Results: one `dataset` item per written dataset."""
    p = dict(job.params)
    builder = p.pop('builder', None)
    name = check_name(p.pop('name', None))
    replace = bool(p.pop('replace', False))
    work = job.work
    written = []
    if builder == 'traces':
        control_name = p.pop('control_name', None) or f'{name}-shuf'
        check_name(control_name)
        records, control, stats = build_traces(**p)
        source = dict(builder='traces', **_jsonable(p))
        if not records:
            raise ValueError('no trace kept: every trace was skipped')
        for n in (name, control_name):
            if dataset_dir(work, n).exists() and not replace:
                raise FileExistsError(f'dataset {n} exists')
        written.append(write_dataset(work, control_name, 'segments', control,
                                     dict(source, control_of=name), stats=stats,
                                     replace=replace))
        written.append(write_dataset(work, name, 'segments', records, source,
                                     control=control_name, stats=stats, replace=replace))
    elif builder == 'pairs':
        records, stats = build_pairs(**p)
        if not records:
            raise ValueError('no pair kept')
        written.append(write_dataset(work, name, p.get('mode', 'answers'), records,
                                     dict(builder='pairs', **_jsonable(p)), stats=stats,
                                     replace=replace))
    elif builder == 'import':
        path = p.pop('path', None)
        if not path:
            raise ValueError('import needs path')
        control_path = p.pop('control', None)
        records = read_jsonl(path)
        mode = p.pop('mode', None) or infer_mode(records[0] if records else {})
        if p:
            raise ValueError(f'unknown import parameters {sorted(p)}')
        control_name = None
        validate(mode, records)
        if control_path:
            control_name = f'{name}-control'
            control_records = read_jsonl(control_path)
            validate(mode, control_records)
            for n in (name, control_name):
                if dataset_dir(work, n).exists() and not replace:
                    raise FileExistsError(f'dataset {n} exists')
            written.append(write_dataset(work, control_name, mode, control_records,
                                         dict(builder='import', path=str(control_path),
                                              control_of=name), replace=replace))
        written.append(write_dataset(work, name, mode, records,
                                     dict(builder='import', path=str(path)),
                                     control=control_name, replace=replace))
    else:
        raise ValueError('builder must be traces, pairs or import')
    for meta in written:
        job.emit(dict(type='dataset', **meta))
    job.set_output(dict(datasets=[m['name'] for m in written]))


def _jsonable(params):
    return {k: (list(v) if isinstance(v, (tuple, set)) else v) for k, v in params.items()}
