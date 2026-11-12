#!/usr/bin/env python3
"""Score and compare R5 coding answers across settings.

    python3 tools/qwen38_flash_next/coding_eval/evaluate.py \\
        --set build/b1/r5-coding \\
        base=build/b1/r5-coding/results/base.jsonl \\
        steer=build/b1/r5-coding/results/steer-a0.5.jsonl \\
        --splits eval --out build/b1/r5-coding/results/compare-eval

Each input is a generation JSONL (one answer per line: id and text; the
generate.py format, or references.jsonl) given as NAME=PATH or PATH (name =
file stem). Every answer's code block is extracted and the hidden tests run
again in the Docker sandbox (no network, read-only, time and memory limits);
--use-stored reuses the scores generate.py stored instead (inputs without
stored scores from the current harness.py are rescored). Every rate is over all selected problems;
a missing answer fails all of its tests.

Prints, per setting: problems answered, code found, truncated answers,
all-tests pass rate, test-level pass rate, the same two rates over the
edge-case tests only and over the ordinary tests only, and simple
thoroughness markers (raises, isinstance checks, asserts, stated complexity).
Against the first setting it prints problems fixed and broken. --out writes
<out>.json (every number and the per-problem results) and <out>.md (the
tables and the per-problem matrix).
"""
import argparse
import json
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox  # noqa: E402


def parse_input(spec):
    if '=' in spec and not Path(spec).exists():
        name, path = spec.split('=', 1)
        return name, Path(path)
    return Path(spec).stem, Path(spec)


def markers(text, code):
    code = code or ''
    return dict(
        raises=bool(re.search(r'^\s*raise\s+\w', code, re.M)),
        isinstance=('isinstance(' in code),
        asserts=bool(re.search(r'^\s*assert\b', code, re.M)),
        complexity=bool(re.search(r'O\(\s*[^)]*\)', text or '')),
    )


def collect(name, path, problems, runner, use_stored, timeout, workers):
    by_id = {p['id']: p for p in problems}
    rows = [r for r in sandbox.read_jsonl(path) if r['id'] in by_id]
    seen = {}
    for row in rows:  # one answer per problem; the last line wins
        seen[row['id']] = row
    rows = list(seen.values())
    if use_stored and all(r.get('score', {}).get('harness') == sandbox.HARNESS_VERSION
                          for r in rows):
        scores = [r['score'] for r in rows]
    elif runner is None:
        raise SystemExit(f'{path}: --use-stored needs a score from harness '
                         f'{sandbox.HARNESS_VERSION} on every line')
    else:
        scores = sandbox.score_answers(runner, problems, [(r['id'], r['text']) for r in rows],
                                       timeout=timeout, workers=workers)
    results = {}
    for row, result in zip(rows, scores):
        problem = by_id[row['id']]
        code, _ = sandbox.extract_code(row['text'] or '', problem['entry_point'])
        results[row['id']] = dict(
            result, finish_reason=row.get('finish_reason'), chars=len(row['text'] or ''),
            completion_tokens=(row.get('usage') or {}).get('completion_tokens'),
            markers=markers(row['text'], code))
    return name, results


def ratio(numerator, denominator):
    return None if denominator == 0 else numerator / denominator


def summarize(results, problems):
    """Every rate is over all the problems (and all their tests): a missing
    answer counts as failing every test."""
    ids = [p['id'] for p in problems]
    answered = [results[i] for i in ids if i in results]
    total = len(ids)

    def count(key):
        return sum(1 for r in answered if r[key])

    tests = sum(len(p['tests']) for p in problems)
    edge = sum(1 for p in problems for t in p['tests'] if t.get('edge'))
    with_edge = sum(1 for p in problems if any(t.get('edge') for t in p['tests']))
    chars = [r['chars'] for r in answered]
    return dict(
        problems=total, answered=len(answered),
        code=sum(1 for r in answered if r['extract'] != 'none'),
        truncated=sum(1 for r in answered if r['finish_reason'] == 'length'),
        all_pass=ratio(count('all_pass'), total),
        test_rate=ratio(sum(r['passed'] for r in answered), tests),
        edge_all_pass=ratio(count('edge_all_pass'), with_edge),
        edge_test_rate=ratio(sum(r['edge_passed'] for r in answered), edge),
        plain_all_pass=ratio(count('plain_all_pass'), total),
        plain_test_rate=ratio(sum(r['plain_passed'] for r in answered), tests - edge),
        mean_chars=ratio(sum(chars), len(chars)),
        raises=ratio(sum(r['markers']['raises'] for r in answered), len(answered)),
        isinstance=ratio(sum(r['markers']['isinstance'] for r in answered), len(answered)),
        asserts=ratio(sum(r['markers']['asserts'] for r in answered), len(answered)),
        complexity=ratio(sum(r['markers']['complexity'] for r in answered), len(answered)),
    )


def pct(value):
    return '-' if value is None else f'{100 * value:.1f}'


COLUMNS = (
    ('answered', 'answered', str), ('code', 'code', str), ('trunc', 'truncated', str),
    ('all-pass %', 'all_pass', pct), ('tests %', 'test_rate', pct),
    ('edge all-pass %', 'edge_all_pass', pct), ('edge tests %', 'edge_test_rate', pct),
    ('plain all-pass %', 'plain_all_pass', pct), ('plain tests %', 'plain_test_rate', pct),
    ('chars', 'mean_chars', lambda v: '-' if v is None else f'{v:.0f}'),
    ('raise %', 'raises', pct), ('isinstance %', 'isinstance', pct),
    ('assert %', 'asserts', pct), ('O() %', 'complexity', pct),
)


def table(headers, rows):
    lines = ['| ' + ' | '.join(headers) + ' |', '|' + '---|' * len(headers)]
    lines += ['| ' + ' | '.join(str(c) for c in row) + ' |' for row in rows]
    return '\n'.join(lines)


def comparison_tables(settings, problems, groups):
    """settings: list of (name, results). groups: list of (label, problem list)."""
    out = []
    for label, subset in groups:
        rows = []
        for name, results in settings:
            summary = summarize(results, subset)
            rows.append([name] + [fmt(summary[key]) for _, key, fmt in COLUMNS])
        out.append(f'### {label} ({len(subset)} problems)\n\n'
                   + table(['setting'] + [h for h, _, _ in COLUMNS], rows))
    return out


def paired(settings, problems):
    if len(settings) < 2:
        return None
    base_name, base = settings[0]
    rows = []
    for name, results in settings[1:]:
        both = [p['id'] for p in problems if p['id'] in base and p['id'] in results]
        fixed = [i for i in both if not base[i]['all_pass'] and results[i]['all_pass']]
        broken = [i for i in both if base[i]['all_pass'] and not results[i]['all_pass']]
        edge_delta = sum(results[i]['edge_passed'] - base[i]['edge_passed'] for i in both)
        rows.append([name, len(fixed), len(broken), f'{edge_delta:+d}',
                     ' '.join(fixed) or '-', ' '.join(broken) or '-'])
    return (f'### Paired against {base_name}\n\n'
            + table(['setting', 'fixed', 'broken', 'edge tests delta', 'fixed ids',
                     'broken ids'], rows))


def cell(result):
    if result is None:
        return '-'
    if result['all_pass']:
        return 'P'
    if result['extract'] == 'none':
        return 'none'
    if result['load'] != 'pass':
        return 'load'
    return f'{result["passed"]}/{result["tests"]}' + ('' if result['plain_all_pass'] else '*')


def matrix(settings, problems):
    rows = [[p['id'], p['split'], p['lang']]
            + [cell(results.get(p['id'])) for _, results in settings] for p in problems]
    return ('### Per problem\n\nP = all tests pass; n/m = tests passed, * = an ordinary '
            '(non-edge) test failed too; load = the code did not load; none = no code.\n\n'
            + table(['id', 'split', 'lang'] + [n for n, _ in settings], rows))


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('inputs', nargs='+', help='NAME=PATH or PATH of a generation JSONL')
    parser.add_argument('--set', type=Path, required=True)
    parser.add_argument('--splits', nargs='+', choices=sandbox.SPLITS,
                        help='default: every split present in the first input')
    parser.add_argument('--use-stored', action='store_true',
                        help='reuse stored scores instead of rerunning the sandbox')
    parser.add_argument('--image', default=sandbox.DEFAULT_IMAGE)
    parser.add_argument('--test-timeout', type=float, default=2.0)
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--out', type=Path, help='write <out>.json and <out>.md')
    args = parser.parse_args()
    problems = sandbox.load_problems(args.set / 'problems.jsonl', args.splits)
    inputs = [parse_input(spec) for spec in args.inputs]
    if args.splits is None:
        present = {r['id'] for r in sandbox.read_jsonl(inputs[0][1])}
        problems = [p for p in problems if p['id'] in present]
    runner = None
    if not args.use_stored or any(
            r.get('score', {}).get('harness') != sandbox.HARNESS_VERSION
            for _, path in inputs for r in sandbox.read_jsonl(path)):
        runner = sandbox.make_runner('docker', image=args.image)
    settings = [collect(name, path, problems, runner, args.use_stored, args.test_timeout,
                        args.jobs) for name, path in inputs]
    groups = [('all', problems)]
    for split in sandbox.SPLITS:
        subset = [p for p in problems if p['split'] == split]
        if subset and len(subset) != len(problems):
            groups.append((split, subset))
    for lang in ('ko', 'en'):
        subset = [p for p in problems if p['lang'] == lang]
        if subset and len(subset) != len(problems):
            groups.append((lang, subset))
    sections = comparison_tables(settings, problems, groups)
    pairs = paired(settings, problems)
    if pairs:
        sections.append(pairs)
    report = '\n\n'.join(sections)
    print(report)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.with_suffix('.md').write_text(
            report + '\n\n' + matrix(settings, problems) + '\n', encoding='utf-8')
        data = dict(
            inputs={name: str(path) for name, path in inputs},
            splits=args.splits, use_stored=args.use_stored,
            summaries={name: {label: summarize(results, subset) for label, subset in groups}
                       for name, results in settings},
            problems={name: results for name, results in settings})
        args.out.with_suffix('.json').write_text(
            json.dumps(data, ensure_ascii=False, indent=1) + '\n', encoding='utf-8')
        print(f'\nwrote {args.out.with_suffix(".md")} and {args.out.with_suffix(".json")}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
