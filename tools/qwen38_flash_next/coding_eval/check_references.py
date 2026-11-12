#!/usr/bin/env python3
"""Verify the R5 coding set on CPU: every canonical solution and every
reference answer must pass all hidden tests, and the solution's own
`__main__` self-checks must pass.

    python3 tools/qwen38_flash_next/coding_eval/check_references.py \\
        --set build/b1/r5-coding

Reads problems.jsonl, solutions.jsonl and references.jsonl from --set. The
code here is ours, but it still runs in the same Docker sandbox as model
output unless --runner local is given. Exit status 1 on any failure.
"""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
import sandbox  # noqa: E402


def check(problems, solutions, references, runner, timeout=2.0, workers=4):
    """Returns (rows, failures): one row per checked item."""
    by_id = {p['id']: p for p in problems}
    jobs, labels = [], []
    for row in solutions:
        problem = by_id[row['id']]
        jobs.append(sandbox.job_for(problem, row['code'], timeout=timeout, run_main=True))
        labels.append(('solution', problem, 'block'))
    problems_needing_reference = {p['id'] for p in problems if p['split'] != 'eval'}
    have_reference = set()
    failures = []
    for row in references:
        problem = by_id[row['id']]
        have_reference.add(row['id'])
        code, method = sandbox.extract_code(row['text'], problem['entry_point'])
        if code != row['code']:
            failures.append(f'{row["id"]} reference: extracted code differs from its '
                            f'solution ({method})')
            continue
        jobs.append(sandbox.job_for(problem, code, timeout=timeout))
        labels.append(('reference', problem, method))
    for missing in sorted(problems_needing_reference - have_reference):
        failures.append(f'{missing}: no reference answer')
    for missing in sorted(set(by_id) - {r['id'] for r in solutions}):
        failures.append(f'{missing}: no solution')
    rows = []
    for (kind, problem, method), report in zip(labels, sandbox.run_jobs(runner, jobs, workers)):
        result = sandbox.score(problem, report, method)
        rows.append(dict(kind=kind, id=problem['id'], **result))
        if not result['all_pass']:
            details = '; '.join(f'#{f["index"]} {f["status"]} {f["detail"]}'
                                for f in result['failures'][:3])
            failures.append(f'{problem["id"]} {kind}: {result["passed"]}/{result["tests"]} '
                            f'(load {result["load"]}: {result["load_detail"]}) {details}')
        if kind == 'solution' and result['main'] != 'pass':
            failures.append(f'{problem["id"]} solution: __main__ self-check {result["main"]}')
    return rows, failures


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--set', type=Path, required=True, help='directory of the set')
    parser.add_argument('--runner', choices=('docker', 'local'), default='docker')
    parser.add_argument('--image', default=sandbox.DEFAULT_IMAGE)
    parser.add_argument('--timeout', type=float, default=2.0, help='seconds per test')
    parser.add_argument('--jobs', type=int, default=4)
    parser.add_argument('--ids', nargs='*', help='only these problem ids')
    args = parser.parse_args()
    problems = sandbox.load_problems(args.set / 'problems.jsonl')
    solutions = sandbox.read_jsonl(args.set / 'solutions.jsonl')
    references = sandbox.read_jsonl(args.set / 'references.jsonl')
    if args.ids:
        keep = set(args.ids)
        problems = [p for p in problems if p['id'] in keep]
        solutions = [s for s in solutions if s['id'] in keep]
        references = [r for r in references if r['id'] in keep]
    runner = sandbox.make_runner(args.runner, image=args.image, trusted=True)
    rows, failures = check(problems, solutions, references, runner, args.timeout, args.jobs)
    tests = sum(r['tests'] for r in rows if r['kind'] == 'solution')
    edge = sum(r['edge_tests'] for r in rows if r['kind'] == 'solution')
    for kind in ('solution', 'reference'):
        selected = [r for r in rows if r['kind'] == kind]
        passed = sum(r['all_pass'] for r in selected)
        print(f'{kind:9}: {passed}/{len(selected)} pass all hidden tests')
    print(f'hidden tests: {tests} ({edge} edge-case) over {len(problems)} problems; '
          f'runner {runner.name}')
    for failure in failures:
        print('FAIL', failure)
    return 1 if failures else 0


if __name__ == '__main__':
    sys.exit(main())
