"""CPU contracts of the R5 coding evaluation tools: code extraction, the
test harness (run unsandboxed on our own snippets), the sandbox command line,
scoring, the generation request and resume, and the comparison tables.
No Docker, GPU or model server is used."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import sandbox  # noqa: E402


def load(name):
    spec = importlib.util.spec_from_file_location(name, HERE / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


generate = load('generate')
evaluate = load('evaluate')

PROBLEM = dict(
    id='list-01', split='eval', lang='en', category='list', difficulty='easy',
    entry_point='head', signature='def head(items: list, n: int) -> list:',
    prompt='Return the first n items.',
    tests=[
        dict(args='([1, 2, 3], 2)', expect='[1, 2]', edge=None),
        dict(args='([], 2)', expect='[]', edge='empty'),
        dict(args='([1], -1)', raises='ValueError', edge='negative'),
        dict(args='([3, 1], 1)', check='result == [3] and args[0] == [3, 1]',
             no_mutate=True, edge='mutation'),
    ])
GOOD = ('def head(items, n):\n    if n < 0:\n        raise ValueError(n)\n'
        '    return items[:n]\n')
SLOPPY = 'def head(items, n):\n    return items[:n]\n'


class ExtractTest(unittest.TestCase):
    def test_last_defining_block_wins(self):
        text = ('Intro\n```python\ndef head(a, n):\n    return 1\n```\nBetter:\n'
                '```py\ndef head(a, n):\n    return 2\n```\nUsage:\n```python\n'
                'print(head([1], 1))\n```\n')
        code, method = sandbox.extract_code(text, 'head')
        self.assertEqual(code, 'def head(a, n):\n    return 2\n')
        self.assertEqual(method, 'block')

    def test_closed_block_beats_truncated_restatement(self):
        text = ('```python\ndef head(a, n):\n    return a[:n]\n```\nOr:\n'
                '```python\ndef head(a, n):\n    if')
        self.assertEqual(sandbox.extract_code(text, 'head'),
                         ('def head(a, n):\n    return a[:n]\n', 'block'))

    def test_unclosed_block_is_kept(self):
        code, method = sandbox.extract_code('x\n```python\ndef head(a, n):\n    return a', 'head')
        self.assertEqual(method, 'unclosed-block')
        self.assertIn('return a', code)

    def test_other_languages_and_tilde_fences(self):
        text = '```bash\ndef head(\n```\n~~~python\ndef head(a, n):\n    pass\n~~~\n'
        code, method = sandbox.extract_code(text, 'head')
        self.assertEqual((code, method), ('def head(a, n):\n    pass\n', 'block'))

    def test_nested_shorter_fence_does_not_close(self):
        text = '````python\ndef head(a, n):\n    s = """\n```\n"""\n    return s\n````\n'
        code, _ = sandbox.extract_code(text, 'head')
        self.assertIn('return s', code)

    def test_raw_text_and_nothing(self):
        self.assertEqual(sandbox.extract_code('def head(a, n):\n    return a\n', 'head')[1],
                         'raw-text')
        self.assertEqual(sandbox.extract_code('I cannot help.', 'head'), (None, 'none'))

    def test_joined_blocks_without_entry_point(self):
        code, method = sandbox.extract_code('```python\nx = 1\n```\n```\ny = 2\n```', 'head')
        self.assertEqual(method, 'joined-blocks')
        self.assertEqual(code, 'x = 1\n\ny = 2\n')


class HarnessTest(unittest.TestCase):
    runner = sandbox.LocalRunner(trusted=True)

    def run_code(self, code, problem=PROBLEM, **job):
        report = self.runner.run(sandbox.job_for(problem, code, **job))
        return sandbox.score(problem, report)

    def test_local_runner_requires_trust(self):
        with self.assertRaises(ValueError):
            sandbox.LocalRunner()

    def test_good_and_sloppy(self):
        good = self.run_code(GOOD)
        self.assertTrue(good['all_pass'])
        self.assertEqual((good['edge_tests'], good['edge_passed']), (3, 3))
        sloppy = self.run_code(SLOPPY)
        self.assertFalse(sloppy['all_pass'])
        self.assertTrue(sloppy['plain_all_pass'])
        self.assertEqual(sloppy['failures'][0]['edge'], 'negative')

    def test_mutation_is_caught(self):
        code = 'def head(items, n):\n    out = items[:n]\n    items.clear()\n    return out\n'
        result = self.run_code(code)
        self.assertIn('mutated', result['failures'][-1]['detail'])

    def test_permissive_eq_does_not_pass(self):
        code = ('class Any(list):\n    def __eq__(self, other):\n        return True\n'
                'def head(items, n):\n    return Any()\n')
        result = self.run_code(code)
        self.assertEqual(result['passed'], 0)
        self.assertIn('solution.Any object', result['failures'][0]['detail'])
        self.assertIn('solution.Any object', result['failures'][3]['detail'])

    def test_forged_child_output_does_not_pass(self):
        # Writing a fake verdict to every open descriptor, then exiting, must
        # not pass: the judge decides from the decoded value, not the child.
        code = ('import os\nfor fd in os.listdir("/proc/self/fd"):\n    try:\n'
                '        os.write(int(fd), b\'{"status": "pass", "detail": "",'
                ' "kind": "ok"}\')\n    except OSError:\n        pass\n'
                'os._exit(0)\n'
                'def head(items, n):\n    return None\n')
        result = self.run_code(code)
        self.assertEqual(result['passed'], 0)

    def one(self, code, **test):
        problem = dict(PROBLEM, tests=[dict(test, edge=None)])
        return self.run_code(code, problem=problem)

    def test_big_integers(self):
        result = self.one('import math\ndef head(n):\n    return math.factorial(n)\n',
                          args='(3000,)', expect='math.factorial(3000)')
        self.assertTrue(result['all_pass'], result['failures'])

    def test_multi_base_exception_matches_only_its_first_builtin(self):
        code = ('class E(ValueError, TypeError, KeyError):\n    pass\n'
                'def head(x):\n    raise E()\n')
        self.assertTrue(self.one(code, args='(1,)', raises='ValueError')['all_pass'])
        self.assertFalse(self.one(code, args='(1,)', raises='TypeError')['all_pass'])
        subclass = 'class E(LookupError):\n    pass\ndef head(x):\n    raise E()\n'
        self.assertTrue(self.one(subclass, args='(1,)', raises='LookupError')['all_pass'])

    def test_check_sees_arguments_after_the_call(self):
        result = self.one('def head(a):\n    a.sort()\n', args='([3, 1, 2],)',
                          check='result is None and args[0] == [1, 2, 3]')
        self.assertTrue(result['all_pass'], result['failures'])

    def test_scramble_catches_aliasing(self):
        shared = 'def head(d):\n    return {"x": d["x"]}\n'
        copied = 'import copy\ndef head(d):\n    return {"x": copy.deepcopy(d["x"])}\n'
        test = dict(args='({"x": [[1]]},)', expect='{"x": [[1]]}', scramble_args=True)
        self.assertFalse(self.one(shared, **test)['all_pass'])
        self.assertTrue(self.one(copied, **test)['all_pass'])

    def test_kind_and_approx(self):
        problem = dict(PROBLEM, tests=[
            dict(args='(0.1, 0.2)', expect='0.3', approx=True, edge=None),
            dict(args='(1, 2)', expect='[1, 2]', edge=None)])
        result = self.run_code('def head(a, b):\n    return (a, b) if a == 1 else a + b\n',
                               problem=problem)
        self.assertEqual([f['index'] for f in result['failures']], [1])

    def test_timeout_crash_and_output_noise(self):
        problem = dict(PROBLEM, tests=[dict(args='([1], 1)', expect='[1]', edge=None)])
        slow = self.run_code('def head(a, n):\n    while True:\n        pass\n',
                             problem=problem, timeout=0.5)
        self.assertEqual(slow['failures'][0]['status'], 'timeout')
        dead = self.run_code('import os\ndef head(a, n):\n    os._exit(3)\n', problem=problem)
        self.assertEqual(dead['failures'][0]['status'], 'crash')
        noisy = self.run_code('print("{")\ndef head(a, n):\n    print("}")\n    return a[:n]\n',
                              problem=problem)
        self.assertTrue(noisy['all_pass'])

    def test_load_errors_and_main(self):
        broken = self.run_code('def head(a, n)\n    return a\n')
        self.assertEqual(broken['load'], 'error')
        self.assertEqual(broken['passed'], 0)
        main = self.run_code(GOOD + 'if __name__ == "__main__":\n    assert head([1], 0) == [1]\n',
                             run_main=True)
        self.assertEqual(main['main'], 'error')
        self.assertTrue(main['all_pass'])


class DockerCommandTest(unittest.TestCase):
    def test_isolation_flags(self):
        command = sandbox.DockerRunner().command('gufo-codeval-x')
        joined = ' '.join(command)
        for flag in ('--init', '--network none', '--read-only', '--pull never', '--cap-drop ALL',
                     '--security-opt no-new-privileges', '--user 65534:65534',
                     '--pids-limit 64', '--memory 1g', '--memory-swap 1g'):
            self.assertIn(flag, joined)
        self.assertIn(f'{sandbox.HARNESS}:/sandbox/harness.py:ro', command)
        self.assertEqual(command[-3:], ['python3', '-I', '/sandbox/harness.py'])

    def test_bad_reports_fail_every_test(self):
        job = sandbox.job_for(PROBLEM, GOOD)
        report = sandbox.parse_report('not json', job)
        self.assertEqual({t['status'] for t in report['tests']}, {'crash'})
        forged = json.dumps(dict(nonce='guess', load={'status': 'pass'}, main=None,
                                 tests=[{'status': 'pass', 'detail': ''}] * 4))
        report = sandbox.parse_report(forged + '\n', job)
        self.assertEqual({t['status'] for t in report['tests']}, {'crash'})

    def test_torn_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'r.jsonl'
            path.write_text('{"id": "a"}\n{"id": "b", "te')
            self.assertEqual(sandbox.read_jsonl(path, torn_tail=True), [{'id': 'a'}])
            with self.assertRaises(ValueError):
                sandbox.read_jsonl(path)
            sandbox.drop_torn_tail(path)
            self.assertEqual(path.read_text(), '{"id": "a"}\n')


class GenerateTest(unittest.TestCase):
    def test_request_body(self):
        body = generate.request_body(PROBLEM, 'gufo-flash-next', 600, {'enabled': False},
                                     {'steering': {'profile': 'x'}})
        self.assertEqual(body['temperature'], 0)
        self.assertEqual(body['max_tokens'], 600)
        self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
        self.assertEqual(body['abliteration'], {'enabled': False})
        self.assertEqual(body['steering'], {'profile': 'x'})
        self.assertEqual(body['messages'], [{'role': 'user', 'content': PROBLEM['prompt']}])

    def test_system_message_is_sent_and_recorded(self):
        body = generate.request_body(PROBLEM, 'm', 600, {'enabled': False}, system='Be brief.')
        self.assertEqual(body['messages'], [{'role': 'system', 'content': 'Be brief.'},
                                            {'role': 'user', 'content': PROBLEM['prompt']}])
        response = {'choices': [{'message': {'content': 'x'}, 'finish_reason': 'stop'}]}
        record = generate.answer_record(PROBLEM, 'c', body, response, 1.0)
        self.assertEqual(record['settings']['system'], 'Be brief.')
        plain = generate.answer_record(
            PROBLEM, 'c', generate.request_body(PROBLEM, 'm', 600, {}), response, 1.0)
        self.assertNotIn('system', plain['settings'])

    def test_resume_and_score_keep_other_lines(self):
        calls = []

        def fake_post(server, body, timeout):
            calls.append(body['messages'][0]['content'])
            return {'choices': [{'message': {'content': f'```python\n{GOOD}```'},
                                 'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 9}}

        other = dict(PROBLEM, id='list-02')
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / 'base.jsonl'
            out.write_text(json.dumps(dict(id='list-02', text='old')) + '\n')
            generate.generate([PROBLEM, other], out, 'base', 'http://x', 'm', 600,
                              {'enabled': False}, {}, 1, post=fake_post, log=lambda _: None)
            self.assertEqual(len(calls), 1)
            generate.score_file(out, [PROBLEM], sandbox.LocalRunner(trusted=True), 2.0, 1,
                                log=lambda _: None)
            rows = sandbox.read_jsonl(out)
        self.assertEqual([r['id'] for r in rows], ['list-02', 'list-01'])
        self.assertNotIn('score', rows[0])
        self.assertTrue(rows[1]['score']['all_pass'])
        self.assertEqual(rows[1]['settings']['abliteration'], {'enabled': False})


class EvaluateTest(unittest.TestCase):
    def test_tables_and_pairs(self):
        problems = [PROBLEM, dict(PROBLEM, id='list-02', lang='ko')]
        runner = sandbox.LocalRunner(trusted=True)
        with tempfile.TemporaryDirectory() as directory:
            paths = {}
            for name, answers in (('base', (SLOPPY, GOOD)), ('steer', (GOOD, GOOD))):
                paths[name] = Path(directory) / f'{name}.jsonl'
                sandbox.write_jsonl(paths[name], [
                    dict(id=p['id'], text=f'O(n).\n```python\n{code}```', finish_reason='stop')
                    for p, code in zip(problems, answers)])
            settings = [evaluate.collect(name, paths[name], problems, runner, False, 2.0, 1)
                        for name in ('base', 'steer')]
        base = evaluate.summarize(settings[0][1], problems)
        self.assertEqual(base['all_pass'], 0.5)
        self.assertEqual(base['edge_test_rate'], 5 / 6)
        self.assertEqual(base['plain_all_pass'], 1.0)
        self.assertEqual(base['complexity'], 1.0)
        pairs = evaluate.paired(settings, problems)
        self.assertIn('| steer | 1 | 0 | +1 | list-01 | - |', pairs)
        self.assertIn('| list-01 | eval | en | 3/4 | P |', evaluate.matrix(settings, problems))


if __name__ == '__main__':
    unittest.main()
