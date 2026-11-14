"""CPU contracts of the C+ operations (resume and import, objective, train
with the fast recipe, profile, load, pairs, pipeline, read-only resources)
against the fake gufo server of test_steer.py. Standard library only."""
import json
from pathlib import Path
import sys
import threading
import time
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import generation  # noqa: E402
import jobs  # noqa: E402
import ops  # noqa: E402
import pipeline  # noqa: E402
from test_steer import GufoTest, ServiceTest, write_jsonl  # noqa: E402

LAYERS = 4


def recheck_reasoning(n):
    """A trace with a first confirmation (keep) and a redundant re-check
    followed by a forward thought (a pair) of the answer n."""
    return (f'So x = {n}.\n\nWait, let me verify. So x = {n}.\n\nNext we go on.\n\n'
            f'Hmm, check again. So x = {n}.\n\nThen done.')


def math_answer(request):
    import re
    prompt = request['messages'][-1]['content']
    m = re.search(r'What is (\d+)\+(\d+)', prompt)
    n = int(m.group(1)) + int(m.group(2)) if m else 0
    reasoning = recheck_reasoning(n)
    return {'choices': [{'message': {'content': f'So \\boxed{{{n}}}.', 'reasoning_content': reasoning},
                         'finish_reason': 'stop'}],
            'usage': {'completion_tokens': len(reasoning) // 2,
                      'completion_tokens_details': {'reasoning_tokens': 30}}}


class CplusTest(GufoTest):
    props = dict(total_slots=24, default_generation_settings=dict(n_ctx=16384))

    def setUp(self):
        super().setUp()
        original = self.fake.route
        test = self

        def route(method, path, body):
            if path == '/props':
                return 200, dict(test.props)
            return original(method, path, body)
        self.fake.route = route
        self.fake.answer = math_answer

    def math_set(self, n=4, name='math.jsonl'):
        rows = [dict(id=f'q{i}', problem=f'What is {i}+{i}?', answer=str(2 * i), split='direction')
                for i in range(1, n + 1)]
        return write_jsonl(self.root / name, rows)

    def base_profile(self):
        path = self.root / 'base.json'
        path.write_text(json.dumps(dict(schema='gufo-abliteration-v2', layers=LAYERS,
                                        direction_layer=2, attention=[0.0] * LAYERS,
                                        mlp=[0.0] * LAYERS, residual=[1.0] * LAYERS,
                                        direction_kind='projected', direction_count=1,
                                        preserve_norm=False)))
        return str(path)


class ResumeTest(CplusTest):
    def test_resume_from_job_and_mismatch(self):
        path = self.math_set()
        config = dict(name='x', reasoning_effort='xhigh', sampling={'max_tokens': 32768})
        store, first = self.run_job('test', dict(set=dict(kind='math', path=path),
                                                 configs=[config]))
        self.assertEqual(first.state['status'], 'done', first.state['error'])
        sent = len(self.fake.chats)
        self.assertEqual(sent, 4)
        lines = (first.dir / 'generations' / 'x.jsonl').read_text().splitlines()
        self.assertEqual([json.loads(x)['id'] for x in lines], ['q1', 'q2', 'q3', 'q4'])
        full = json.loads(lines[0])
        self.assertEqual(full['settings']['seed'], 1000)
        self.assertEqual(len(full['request_sha256']), 64)
        # Same configuration: nothing generated, every row reused.
        _, second = self.run_job('test', dict(set=dict(kind='math', path=path), configs=[config],
                                              resume=first.key))
        self.assertEqual(second.state['status'], 'done', second.state['error'])
        self.assertEqual(len(self.fake.chats), sent)
        resume = [r for r in second.results if r['type'] == 'resume'][0]
        self.assertEqual((resume['reused'], resume['mismatched']), (4, 0))
        rows = [r for r in second.results if r['type'] == 'row']
        self.assertTrue(all(r['resumed'] for r in rows))
        self.assertEqual(second.state['output']['totals']['x']['n'], 4)
        lines = (second.dir / 'generations' / 'x.jsonl').read_text().splitlines()
        self.assertEqual(len(lines), 4)
        # A different request: nothing reused.
        changed = dict(config, sampling={'max_tokens': 1000})
        _, third = self.run_job('test', dict(set=dict(kind='math', path=path), configs=[changed],
                                             resume=[{'job': first.key}]))
        resume = [r for r in third.results if r['type'] == 'resume'][0]
        self.assertEqual((resume['reused'], resume['mismatched']), (0, 4))
        self.assertEqual(len(self.fake.chats), sent + 4)

    def test_import_think_eval_rows(self):
        path = self.math_set()
        # think_eval.py rows: no settings, an error row, a stale grade.
        rows = [dict(id='q1', answer='2', got='9', correct=False, finish_reason='stop',
                     usage={'completion_tokens': 10}, reasoning='Wait, ok.', content='\\boxed{2}',
                     elapsed_s=3.0),
                dict(id='q2', error='timeout'),
                dict(id='zz', answer='1', content='x', finish_reason='stop')]
        imported = write_jsonl(self.root / 'think.jsonl', rows)
        _, job = self.run_job('test', dict(set=dict(kind='math', path=path),
                                           configs=[dict(name='xhigh', reasoning_effort='xhigh')],
                                           resume=[dict(config='xhigh', path=imported)]))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        resume = [r for r in job.results if r['type'] == 'resume'][0]
        self.assertEqual((resume['reused'], resume['unfinished']), (1, 1))
        rows = {r['id']: r for r in job.results if r['type'] == 'row'}
        self.assertTrue(rows['q1']['resumed'])
        self.assertTrue(rows['q1']['correct'])  # graded again from the content
        self.assertEqual(rows['q1']['seconds'], 3.0)
        self.assertEqual(len(self.fake.chats), 3)
        # trust false: rows without a request identity are not reused.
        _, job = self.run_job('test', dict(set=dict(kind='math', path=path),
                                           configs=[dict(name='xhigh', reasoning_effort='xhigh')],
                                           resume=[dict(config='xhigh', path=imported, trust=False)]))
        self.assertEqual([r for r in job.results if r['type'] == 'resume'][0]['mismatched'], 1)

    def test_coding_thinking_and_generate_import(self):
        problem = dict(id='list-01', split='direction', lang='en', category='list',
                       difficulty='easy', entry_point='head',
                       signature='def head(items: list, n: int) -> list:',
                       prompt='Return the first n items.',
                       tests=[dict(args='([1, 2, 3], 2)', expect='[1, 2]', edge=None)])
        other = dict(problem, id='list-02')
        folder = self.root / 'coding'
        folder.mkdir()
        write_jsonl(folder / 'problems.jsonl', [problem, other])
        config = dict(name='xhigh', **generation.THINK_XHIGH)
        problem_set = generation.CodingSet(dict(path=str(folder)), lambda spec: None)
        body = problem_set.body(problem, 'gufo-flash-next', generation.config_fields(config))
        self.assertEqual((body['chat_template_kwargs'], body['temperature'], body['top_p'],
                          body['seed'], body['max_tokens'], body['reasoning_effort']),
                         ({'enable_thinking': True}, 1.0, 0.95, 7, 32768, 'xhigh'))
        settings = {k: v for k, v in body.items() if k != 'messages'}
        imported = write_jsonl(self.root / 'think-xhigh-list-01.jsonl', [dict(
            id='list-01', text='```python\ndef head(i, n):\n    return i[:n]\n```',
            reasoning='Think.', finish_reason='stop', usage={'completion_tokens': 9},
            elapsed_s=1.0, settings=settings, score=dict(all_pass=True, passed=1, tests=1,
                                                         extract='block'))])

        class NoSandbox:
            def run(self, *args, **kwargs):
                raise AssertionError('a stored grade must not be graded again')
        self.context.coding_runner = lambda spec: NoSandbox()
        _, job = self.run_job('test', dict(set=dict(kind='coding', path=str(folder),
                                                    ids=['list-01']),
                                           configs=[config],
                                           resume=[dict(config='xhigh',
                                                        path=str(self.root / 'think-xhigh-*.jsonl'))]))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        row = [r for r in job.results if r['type'] == 'row'][0]
        self.assertEqual((row['correct'], row['passed'], row['resumed']), (True, 1, True))
        self.assertEqual(self.fake.chats, [])


class ObjectiveTest(CplusTest):
    def traces(self, ids, name='t.jsonl', n=None):
        rows = []
        for i in ids:
            reasoning = recheck_reasoning(2 * i)
            rows.append(dict(id=f'q{i}', answer=str(2 * i), got=str(2 * i), correct=True,
                             finish_reason='stop', reasoning=reasoning, content=f'\\boxed{{{2 * i}}}',
                             usage={'completion_tokens': len(reasoning) // 2}))
        return write_jsonl(self.root / name, rows)

    def test_rows_caps_and_held_out(self):
        problems = self.math_set()
        traces = self.traces(range(1, 5))
        _, job = self.run_job('objective', dict(sets=[dict(name='m', problems=problems,
                                                           traces=traces, effort='xhigh')],
                                                keep_first_confirmations=True))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        out = job.state['output']
        self.assertEqual(out['counts'], dict(pairs=4, keep=4, guard=4))
        self.assertEqual(out['labels'], dict(pairs={'redundant': 4}, keep={'first-confirmation': 4}))
        self.assertEqual(out['caps']['max_prefix_tokens'], 12288)
        self.assertEqual(out['context_tokens'] % 1024, 0)
        self.assertGreaterEqual(out['context_tokens'], out['longest_row_tokens'])
        pair = json.loads(Path(out['pairs']).read_text().splitlines()[0])
        self.assertEqual((pair['score_reasoning'], pair['reasoning_effort'], pair['chain']),
                         (True, 'xhigh', 'm:q1'))
        report = [r for r in job.results if r['type'] == 'objective_set'][0]
        self.assertEqual(report['counts']['redundant'], 4)
        self.assertIn('redundant', report['examples'])
        # A prefix cap below the pair boundary (about 30 tokens in) drops the pairs.
        _, job = self.run_job('objective', dict(sets=[dict(name='m', problems=problems,
                                                           traces=traces, effort='xhigh')],
                                                keep_first_confirmations=True,
                                                max_prefix_chars=40))
        self.assertIn('no pair row', job.state['error'])
        # Guard reasoning cap: skip or cut at a thought boundary.
        for mode, kept in (('skip', 0), ('cut', 4)):
            _, job = self.run_job('objective', dict(sets=[dict(name='m', problems=problems,
                                                               traces=traces, effort='xhigh')],
                                                    keep_first_confirmations=True,
                                                    max_guard_chars=40, guard_cap=mode))
            self.assertEqual(job.state['output']['counts']['guard'], kept, mode)
            if kept:
                guard = json.loads(Path(job.state['output']['guard']).read_text().splitlines()[0])
                self.assertLessEqual(len(guard['reasoning']), 40)
                self.assertTrue(guard['reasoning'].endswith('\n\n'))
        # AIME 2025 is refused.
        held = write_jsonl(self.root / 'a25.jsonl', [dict(id='aime2025-01', problem='What is 1+1?',
                                                          answer='2', level=2025)])
        held_traces = write_jsonl(self.root / 'a25t.jsonl', [dict(
            id='aime2025-01', answer='2', correct=True, finish_reason='stop',
            reasoning=recheck_reasoning(2), content='\\boxed{2}')])
        _, job = self.run_job('objective', dict(sets=[dict(name='a', problems=held,
                                                           traces=held_traces, effort='xhigh')]))
        self.assertEqual(job.state['status'], 'failed')
        self.assertIn('held-out', job.state['error'])

    def test_coding_guard_rows(self):
        problems = self.math_set()
        traces = self.traces(range(1, 3))
        folder = self.root / 'coding'
        folder.mkdir()
        base = dict(lang='en', category='list', difficulty='easy', entry_point='f',
                    signature='def f():', tests=[])
        write_jsonl(folder / 'problems.jsonl', [
            dict(base, id='c1', split='direction', prompt='P1'),
            dict(base, id='c2', split='direction', prompt='P2'),
            dict(base, id='c3', split='strength', prompt='P3'),
            dict(base, id='c4', split='eval', prompt='P4')])
        code = write_jsonl(self.root / 'code.jsonl', [
            dict(id='c1', content='A1', reasoning='R1', correct=True, finish_reason='stop',
                 settings={'reasoning_effort': 'xhigh'}),
            dict(id='c2', content='A2', reasoning='R2', correct=False, finish_reason='stop',
                 settings={'reasoning_effort': 'xhigh'}),
            dict(id='c3', content='A3', reasoning='R3', correct=True, finish_reason='stop',
                 settings={'reasoning_effort': 'xhigh'})])
        sets = [dict(name='m', problems=problems, traces=traces, effort='xhigh'),
                dict(name='code', kind='coding', problems=str(folder), traces=code)]
        _, job = self.run_job('objective', dict(sets=sets))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        guard = [json.loads(x) for x in Path(job.state['output']['guard']).read_text().splitlines()]
        code_rows = [g for g in guard if g['chain'].startswith('code:')]
        self.assertEqual([(g['chain'], g['prompt'], g['answer']) for g in code_rows],
                         [('code:c1', 'P1', 'A1')])
        report = [r for r in job.results if r['type'] == 'objective_set'][1]
        self.assertEqual(report['counts']['other_split'], 1)
        held = write_jsonl(self.root / 'code-eval.jsonl', [
            dict(id='c4', content='A4', reasoning='R4', correct=True, finish_reason='stop',
                 settings={'reasoning_effort': 'xhigh'})])
        sets[1]['traces'] = held
        _, job = self.run_job('objective', dict(sets=sets))
        self.assertIn('eval split', job.state['error'])


class TrainProfileTest(CplusTest):
    def objective(self):
        problems = self.math_set()
        traces = ObjectiveTest.traces(self, range(1, 5))
        _, job = self.run_job('objective', dict(sets=[dict(name='m', problems=problems,
                                                           traces=traces, effort='xhigh')],
                                                keep_first_confirmations=True))
        return job

    def train_params(self, objective, **extra):
        return dict(objective=objective.key, base_profile=self.base_profile(),
                    options=dict(layers=LAYERS, direction_layer=2, init=0.5, block_size=2,
                                 steps=2, eval_every=1, max_strength=2), **extra)

    def test_fast_defaults_strengths_and_profile(self):
        objective = self.objective()
        self.props = dict(total_slots=2, default_generation_settings=dict(n_ctx=1048576))
        _, job = self.run_job('train', self.train_params(objective, profile_name='rc-r10'))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        train = [r for r in job.results if r['type'] == 'train'][0]
        self.assertEqual(train['recipe']['applied'], dict(concurrency=4, minibatch_chains=12))
        self.assertEqual(train['args'][train['args'].index('--concurrency') + 1], '4')
        self.assertIn('--pairs', train['args'])
        self.assertIn('--ordinary', train['args'])
        warnings = {r['code'] for r in job.results if r['type'] == 'warning'}
        self.assertEqual(warnings, {'few_sessions', 'long_context'})
        steps = [r for r in job.results if r['type'] == 'step']
        self.assertEqual([s['step'] for s in steps], [-1, 0, 1])
        self.assertEqual(len(steps[1]['theta']), 6)  # 3 kinds x 2 blocks
        self.assertEqual(len(steps[1]['strengths']['residual']), LAYERS)
        self.assertIsNotNone(steps[1]['j_plus'])
        self.assertIsNotNone(steps[-1]['eval_j'])
        profile = json.loads(Path(job.state['output']['profile_path']).read_text())
        best = job.state['output']['best']['settings']
        self.assertEqual(profile['residual'], best['residual'])
        self.assertEqual(profile['direction_layer'], 2)
        # The profile operation from the train job, uploaded.
        _, built = self.run_job('profile', dict(train=job.key, name='rc-up', upload=True))
        self.assertEqual(built.state['status'], 'done', built.state['error'])
        uploaded = json.loads(self.fake.profiles[('abliteration', 'rc-up')])
        self.assertEqual(uploaded['residual'], best['residual'])
        # The upload path builds from a train job too.
        raw = ops.profile_bytes(jobs.JobStore(self.work), 'abliteration', 'x',
                                dict(job=job.key, profile='nothing-written'))
        self.assertEqual(json.loads(raw)['residual'], best['residual'])
        # load: unload first (a missing one is fine), then upload.
        _, load = self.run_job('load', dict(name='rc-load', job=built.key, profile='rc-up',
                                            unload=['gone']))
        self.assertEqual(load.state['status'], 'done', load.state['error'])
        self.assertIn(('abliteration', 'rc-load'), self.fake.profiles)
        # Resume copies the log and the earlier recipe values (here on a server
        # with other sessions): spsa.py's header matches and it continues.
        self.props = dict(total_slots=40, default_generation_settings=dict(n_ctx=16384))
        _, again = self.run_job('train', self.train_params(objective, resume=job.key))
        self.assertEqual(again.state['status'], 'done', again.state['error'])
        train = [r for r in again.results if r['type'] == 'train'][0]
        self.assertEqual(train['recipe']['applied'], dict(concurrency=4, minibatch_chains=12))
        with self.assertRaisesRegex(ValueError, 'in full'):
            import training
            training.check_args(['--conc', '4'])
        # spsa.py's own short names are full options, not abbreviations (the R10 pipeline sets a and c)
        training.check_args(['--a', '0.2', '--c', '0.15', '--A', '10'])
        self.assertEqual(training.options_args(dict(a=0.2, c=0.15)), ['--a', '0.2', '--c', '0.15'])

    def test_short_context_fails_before_scoring(self):
        objective = self.objective()
        self.props = dict(total_slots=24, default_generation_settings=dict(n_ctx=16))
        chats = len(self.fake.chats)
        _, job = self.run_job('train', self.train_params(objective))
        self.assertEqual(job.state['status'], 'failed')
        self.assertIn('shorter than the longest row', job.state['error'])
        self.assertEqual(len(self.fake.chats), chats)
        _, job = self.run_job('train', self.train_params(objective, server_check='off',
                                                         fast_defaults=False))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        train = [r for r in job.results if r['type'] == 'train'][0]
        self.assertEqual(train['recipe']['applied'], {})


class PairsTest(CplusTest):
    def test_pairs_generation(self):
        prompts = write_jsonl(self.root / 'prompts.jsonl', [
            dict(id='a1', prompt='What is 1+2?'), dict(id='a2', text='What is 2+2?')])
        _, job = self.run_job('pairs', dict(name='careful', prompts=prompts,
                                            instruction='Be careful.',
                                            sampling={'temperature': 0}))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        self.assertEqual(job.state['output']['datasets'], ['careful', 'careful-answers'])
        texts = sorted(b['messages'][0]['content'] for b in self.fake.chats)
        self.assertEqual(texts, ['Be careful.\n\nWhat is 1+2?', 'Be careful.\n\nWhat is 2+2?',
                                 'What is 1+2?', 'What is 2+2?'])
        import data
        meta, records = data.load_dataset(self.work, 'careful')
        self.assertEqual(meta['groups'], {'exemplar': 2, 'own': 2})
        self.assertEqual(records[0]['text'], 'Be careful.\n\nWhat is 1+2?')
        self.assertEqual(records[1]['text'], 'What is 1+2?')
        self.assertEqual(records[0]['answer'], records[1]['answer'])
        meta, records = data.load_dataset(self.work, 'careful-answers')
        self.assertEqual(meta['kind'], 'answers')
        self.assertEqual(records[0]['text'], records[1]['text'])


class PipelineTest(CplusTest):
    def steps(self):
        path = self.math_set()
        return [
            dict(name='traces', operation='test', await_server={'min_sessions': 4},
                 params=dict(set=dict(kind='math', path=path),
                             configs=[dict(name='xhigh', reasoning_effort='xhigh')])),
            dict(name='objective', operation='objective', params=dict(
                sets=[dict(name='m', problems=path, job={'from': 'traces', 'key': 'job'},
                           config='xhigh')], keep_first_confirmations=True)),
            dict(name='train', operation='train', await_server={
                'min_context': {'from': 'objective', 'key': 'context_tokens'}},
                params=dict(objective={'from': 1, 'key': 'job'}, base_profile=self.base_profile(),
                            options=dict(layers=LAYERS, direction_layer=2, init=0.5,
                                         steps=1, eval_every=1))),
            dict(name='profile', operation='profile', params=dict(
                train={'from': 'train', 'key': 'job'}, name='rc-r10', upload=True)),
            dict(name='eval', operation='test', params=dict(
                set=dict(kind='math', path=self.math_set(2, 'aime25.jsonl')),
                configs=[dict(name='on', abliteration={'enabled': True, 'profile': 'rc-r10'})])),
        ]

    def test_pipeline_runs_and_resumes(self):
        store, job = self.run_job('pipeline', dict(steps=self.steps()))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        ends = [r for r in job.results if r['type'] == 'pipeline_step' and r['status'] == 'done']
        self.assertEqual([r['name'] for r in ends],
                         ['traces', 'objective', 'train', 'profile', 'eval'])
        children = [j for j in store.list() if j.get('parent') == job.key]
        self.assertEqual(len(children), 5)
        self.assertIn(('abliteration', 'rc-r10'), self.fake.profiles)
        objective_params = ends[1]['params']
        self.assertEqual(objective_params['sets'][0]['job'], ends[0]['job'])
        self.assertTrue([r for r in job.results if r['type'] == 'server' and r['matched']])
        # Resume: every finished step with the same params is reused.
        chats = len(self.fake.chats)
        _, again = self.run_job('pipeline', dict(steps=self.steps(), resume=job.key))
        self.assertEqual(again.state['status'], 'done', again.state['error'])
        steps = [r for r in again.results if r['type'] == 'pipeline_step']
        self.assertTrue(all(r.get('reused') for r in steps), steps)
        self.assertEqual(len(self.fake.chats), chats)  # nothing ran again

    def test_resume_after_a_failed_step(self):
        steps = self.steps()[:2]
        good = steps[0]['params']['set']['path']
        steps[0]['params']['set']['path'] = str(self.root / 'missing.jsonl')
        _, job = self.run_job('pipeline', dict(steps=steps))
        self.assertEqual(job.state['status'], 'failed')
        self.assertIn('step 0 (traces', job.state['error'])
        steps[0]['params']['set']['path'] = good
        steps[0]['params']['resume'] = [dict(config='xhigh', path=str(self.root / 'none-*.jsonl'))]
        _, again = self.run_job('pipeline', dict(steps=steps, resume=job.key))
        self.assertEqual(again.state['status'], 'done', again.state['error'])
        first = [r for r in again.results if r['type'] == 'pipeline_step'][0]
        old = [r for r in job.results if r['type'] == 'pipeline_step'][0]['job']
        self.assertEqual(first['resumed_from'], old)
        child = jobs.JobStore(self.work).get(first['job'])
        self.assertEqual(child.params['resume'][-1], {'job': old})  # added to its own list
        self.assertEqual(len(child.params['resume']), 2)

    def test_rerun_step_reruns_later_steps(self):
        steps = self.steps()
        _, job = self.run_job('pipeline', dict(steps=steps))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        steps[2]['params']['options']['seed'] = 5  # train changes: profile and eval run again
        chats = len(self.fake.chats)
        _, again = self.run_job('pipeline', dict(steps=steps, resume=job.key))
        self.assertEqual(again.state['status'], 'done', again.state['error'])
        ends = [r for r in again.results if r['type'] == 'pipeline_step' and r['status'] == 'done']
        self.assertEqual([bool(r.get('reused')) for r in ends], [True, True, False, False, False])
        train = jobs.JobStore(self.work).get(ends[2]['job'])
        self.assertNotIn('resume', train.params)  # other settings: spsa.py starts anew
        evaluation = jobs.JobStore(self.work).get(ends[4]['job'])
        self.assertNotIn('resume', evaluation.params)  # after a rerun: fresh, not reused
        self.assertIsNone(ends[4].get('resumed_from'))
        self.assertGreaterEqual(len([c for c in self.fake.chats[chats:]
                                     if not c.get('score_assistant')]), 2)

    def test_held_out_and_bad_refs(self):
        steps = self.steps()
        bad = [steps[4], dict(steps[1], params=dict(steps[1]['params'], sets=[dict(
            steps[1]['params']['sets'][0], job={'from': 0, 'key': 'job'})]))]
        with self.assertRaisesRegex(ValueError, 'held-out'):
            pipeline.check_steps(bad)
        with self.assertRaisesRegex(ValueError, 'not an earlier step'):
            pipeline.check_steps([steps[1]])
        with self.assertRaisesRegex(ValueError, 'AIME 2025 is held out'):
            pipeline.check_steps([dict(name='o', operation='objective', params=dict(
                sets=[dict(name='x', problems='build/b1/r10/aime25.jsonl', traces='t')]))])
        with self.assertRaisesRegex(ValueError, 'held-out set'):
            pipeline.check_steps([dict(name='p', operation='pairs', params=dict(
                name='x', instruction='i', set=dict(kind='coding', path='c', split='eval')))])
        coding_all = dict(name='c', operation='test', params=dict(set=dict(kind='coding', path='x')))
        self.assertTrue(pipeline.held_out(coding_all))
        r10 = json.loads((HERE / 'examples' / 'r10.json').read_text())
        names = pipeline.check_steps(r10['steps'])
        self.assertEqual(names[-2:], ['eval-aime25', 'eval-code'])
        self.assertTrue(pipeline.held_out(r10['steps'][-1]))
        self.assertTrue(pipeline.held_out(r10['steps'][-2]))

    def test_cancel_cancels_the_child(self):
        gate = threading.Event()

        def slow(request):
            gate.wait(5)
            return math_answer(request)
        self.fake.answer = slow
        store = jobs.JobStore(self.work)
        job = store.create('pipeline', dict(steps=self.steps()[:1]))
        thread = threading.Thread(target=jobs.run_job, args=(job, self.context, ops.OPERATIONS))
        thread.start()
        for _ in range(200):
            children = [j for j in store.list() if j.get('parent') == job.key]
            if children and children[0]['status'] == 'running':
                break
            time.sleep(0.02)
        job.request_cancel()
        thread.join(10)
        gate.set()
        self.assertEqual(job.state['status'], 'cancelled')
        child = store.get(children[0]['key'])
        self.assertEqual(child.state['status'], 'cancelled')


class ResourcesTest(ServiceTest):
    def test_read_only_resources(self):
        self.fake.answer = math_answer
        path = write_jsonl(self.root / 'm.jsonl', [
            dict(id=f'q{i}', problem=f'What is {i}+{i}?', answer=str(2 * i)) for i in range(1, 4)])
        status, reply = self.call('POST', '/jobs', dict(operation='test', params=dict(
            set=dict(kind='math', path=path), configs=[dict(name='a')])))
        test = self.wait_done(reply['job'])
        self.assertEqual(test['status'], 'done', test['error'])
        status, listing = self.call('GET', f'/jobs/{reply["job"]}/generations')
        self.assertEqual(listing['data'][0]['config'], 'a')
        self.assertEqual(listing['data'][0]['rows'], 3)
        status, page = self.call('GET', f'/jobs/{reply["job"]}/generations/a?offset=1&limit=1&fields=id,correct')
        self.assertEqual((page['object'], page['total'], page['offset'], page['limit']),
                         ('page', 3, 1, 1))
        self.assertEqual(set(page['data'][0]), {'id', 'correct'})
        self.assertEqual(self.call('GET', f'/jobs/{reply["job"]}/generations/..')[0], 404)
        self.assertEqual(self.call('GET', f'/jobs/{reply["job"]}/generations/a?limit=-1')[0], 400)
        # Objective rows from those generations.
        status, reply2 = self.call('POST', '/jobs', dict(operation='objective', params=dict(
            sets=[dict(name='m', problems=path, job=reply['job'], config='a', effort='low')],
            keep_first_confirmations=True)))
        objective = self.wait_done(reply2['job'])
        self.assertEqual(objective['status'], 'done', objective['error'])
        status, rows = self.call('GET', f'/jobs/{reply2["job"]}/rows')
        self.assertEqual({r['name'] for r in rows['data']}, {'pairs', 'keep', 'guard'})
        status, page = self.call('GET', f'/jobs/{reply2["job"]}/rows/pairs')
        self.assertEqual(page['total'], 3)
        # Dataset records, paged.
        problems = write_jsonl(self.root / 'p.jsonl', [dict(id='p1', prompt='x')])
        traces = write_jsonl(self.root / 't.jsonl', [dict(id='p1', reasoning='\n\n'.join(
            ['A.', 'Wait, b.', 'C.', 'Hmm, d.', 'E.', 'Wait, f.', 'G.', 'H.']))])
        status, reply3 = self.call('POST', '/jobs', dict(operation='dataset', params=dict(
            builder='traces', name='rc', sources=[dict(name='s', problems=problems, traces=traces)])))
        self.wait_done(reply3['job'])
        status, page = self.call('GET', '/datasets/rc/records?limit=1')
        self.assertEqual((page['total'], len(page['data'])), (2, 1))
        self.assertEqual(self.call('GET', '/datasets/nope/records')[0], 404)
        # Profiles in the work folder, one job's profile, gufo /props.
        folder = Path(self.store.get(reply['job']).dir) / 'profiles'
        folder.mkdir()
        (folder / 'p1.json').write_text(json.dumps(dict(layers=4, direction_layer=2)))
        status, listing = self.call('GET', '/profiles')
        self.assertEqual([(p['name'], p['job'], p['layers']) for p in listing['data']],
                         [('p1', reply['job'], 4)])
        status, profile = self.call('GET', f'/jobs/{reply["job"]}/profiles/p1')
        self.assertEqual(profile['direction_layer'], 2)
        original = self.fake.route
        self.fake.route = lambda m, p, b: (200, {'total_slots': 2}) if p == '/props' else original(m, p, b)
        self.assertEqual(self.call('GET', '/gufo/props'), (200, {'total_slots': 2}))
        status, listing = self.call('GET', f'/jobs?parent={reply["job"]}')
        self.assertEqual(listing['data'], [])


if __name__ == '__main__':
    unittest.main()
