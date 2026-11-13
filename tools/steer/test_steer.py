"""CPU contracts of the steering tool against a fake gufo server (no GPU, no
live server). Fit tests need NumPy and skip without it; run them in the
pinned container (README.md)."""
from array import array
import hashlib
import http.server
import json
from pathlib import Path
import random
import re
import shutil
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import client  # noqa: E402
import cli  # noqa: E402
import data  # noqa: E402
import jobs  # noqa: E402
import ops  # noqa: E402
import service  # noqa: E402

try:
    import numpy  # noqa: F401
    HAVE_NUMPY = True
except ImportError:
    HAVE_NUMPY = False

KEY = 'secret-token'
LAYERS, HIDDEN = 4, 8
PLANTED = 2  # the layer whose difference carries the planted direction


def f32(values):
    a = array('f', values)
    if sys.byteorder == 'big':
        a.byteswap()
    return a.tobytes()


class FakeGufo:
    """Collection jobs (writing a synthetic collection), runtime profile
    uploads, Chat Completions (math answers and score_assistant scores)."""

    def __init__(self, collect_dir):
        self.collect_dir = Path(collect_dir)
        self.jobs, self.profiles, self.chats = {}, {}, []
        self.auth_failures = 0
        self.lock = threading.Lock()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *args):
                pass

            def reply(self, status, value):
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def handle_any(self, method):
                if self.headers.get('Authorization') != f'Bearer {KEY}':
                    fake.auth_failures += 1
                    return self.reply(401, {'error': {'message': 'key', 'code': 'unauthorized'}})
                length = int(self.headers.get('Content-Length') or 0)
                body = self.rfile.read(length) if length else b''
                status, value = fake.route(method, self.path, body)
                self.reply(status, value)

            def do_GET(self):
                self.handle_any('GET')

            def do_POST(self):
                self.handle_any('POST')

            def do_DELETE(self):
                self.handle_any('DELETE')

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()

    def route(self, method, path, body):
        m = re.fullmatch(r'/v1/(abliteration|logit-bias)/profiles/([\w.-]+)', path)
        if m:
            kind, name = m.groups()
            if method == 'POST':
                self.profiles[(kind, name)] = body
                return 200, {'profile': name, 'identity': hashlib.sha256(body).hexdigest()}
            if (kind, name) not in self.profiles:
                return 404, {'error': {'message': 'no', 'code': 'unknown_abliteration_profile'}}
            del self.profiles[(kind, name)]
            return 200, {'profile': name, 'deleted': True}
        if path == '/v1/abliteration/jobs' and method == 'POST':
            request = json.loads(body)
            key = hashlib.sha256(body + str(len(self.jobs)).encode()).hexdigest()[:32]
            self.jobs[key] = dict(request=request, polls=0, cancelled=False)
            return 200, {'job': key, 'status': 'queued'}
        m = re.fullmatch(r'/v1/abliteration/jobs/(\w+)(?:\?since=(\d+))?', path)
        if m:
            job = self.jobs.get(m.group(1))
            if job is None:
                return 404, {'error': {'message': 'no', 'code': 'unknown_job'}}
            if method == 'DELETE':
                job['cancelled'] = True
                return 200, {'job': m.group(1), 'deleted': True}
            return 200, self.poll(m.group(1), job, int(m.group(2) or 0))
        if path == '/v1/chat/completions':
            request = json.loads(body)
            with self.lock:
                self.chats.append(request)
            if request.get('score_assistant'):
                return 200, self.score(request)
            return 200, self.answer(request)
        return 404, {'error': {'message': 'path', 'code': 'not_found'}}

    def poll(self, key, job, since):
        records = job['request']['records']
        job['polls'] += 1
        if job['polls'] == 1:
            return {'job': key, 'status': 'running', 'progress': {
                'done': 0, 'total': len(records), 'step': 'record 1'}, 'results': [],
                'last_seq': 0, 'numerics_deviations': []}
        self.write_collection(job['request'])
        results = [dict(seq=i + 1, id=r['id']) for i, r in enumerate(records)]
        return {'job': key, 'status': 'done', 'progress': {
            'done': len(records), 'total': len(records), 'step': 'done'},
            'results': [r for r in results if r['seq'] > since], 'last_seq': len(results),
            'numerics_deviations': []}

    def write_collection(self, request):
        """Own rows: random topic; exemplar rows: the topic plus a fixed
        direction at PLANTED (and a weaker one elsewhere) for a real
        dataset; labels carry nothing for a `-shuf` control."""
        out = self.collect_dir / request['name']
        if (out / 'manifest.json').exists():
            return
        out.mkdir(parents=True)
        control = request['name'].startswith('c-')
        rng = random.Random(request['name'])
        topics = {}
        for r in request['records']:
            topic = topics.setdefault(r['pair'], [[rng.gauss(0, 3) for _ in range(HIDDEN)]
                                                  for _ in range(LAYERS)])
            rows = []
            for layer in range(LAYERS):
                row = [t + rng.gauss(0, 0.5) for t in topic[layer]]
                if r['group'] == 'exemplar' and not control:
                    row[1] += 4.0 if layer == PLANTED else 0.3
                rows.extend(row)
            raw = f32(rows)
            (out / f'{r["id"]}.f32').write_bytes(raw)
            (out / f'{r["id"]}.json').write_text(json.dumps(dict(
                id=r['id'], activation_sha256=hashlib.sha256(raw).hexdigest())))
        readout = ('selected-segment-end-rows-mean-post-layer-hc-mean-fp32-v1'
                   if request['mode'] == 'segments' else
                   'answer-tokens-mean-post-layer-hc-mean-fp32-v1')
        (out / 'manifest.json').write_text(json.dumps(dict(
            schema='gufo-abliteration-capture-v1', model_identity='fake-model',
            readout=readout, hidden=HIDDEN, layers=LAYERS, split='collection',
            mode=request['mode'],
            records=[dict(id=r['id'], group=r['group'], pair=r['pair'])
                     for r in request['records']])))

    def answer(self, request):
        prompt = request['messages'][-1]['content']
        number = re.search(r'What is (\d+)\+(\d+)', prompt)
        right = str(int(number.group(1)) + int(number.group(2))) if number else '0'
        steered = (request.get('abliteration') or {}).get('enabled')
        got = right if steered or int(right) % 2 == 0 else str(int(right) + 1)
        reasoning = 'Let me add.\n\nWait, check.\n\nHmm, fine.' if not steered else 'Add.'
        return {'choices': [{'message': {'content': f'So \\boxed{{{got}}}.',
                                         'reasoning_content': reasoning},
                             'finish_reason': 'stop'}],
                'usage': {'completion_tokens': 50 if steered else 120,
                          'completion_tokens_details': {'reasoning_tokens': 30}}}

    def score(self, request):
        ab = request.get('abliteration') or {}
        residual = ab.get('residual') or [0.0] * LAYERS
        strength = sum(residual) / len(residual) if isinstance(residual, list) else residual
        answer = request['messages'][-1]['content']
        good = 'candid' in answer
        lp = -5.0 + (strength if good else -strength) - 0.1 * strength * strength
        return {'score': {'logprob_sum': lp * 4, 'token_count': 4}}


def write_jsonl(path, rows):
    Path(path).write_text(''.join(json.dumps(r) + '\n' for r in rows))
    return str(path)


def traces_fixture(root):
    problems = [dict(id=f'p{i}', problem=f'What is {i}+{i + 1}?', answer=str(2 * i + 1))
                for i in range(6)]
    thoughts = ['Start here.', 'Compute it.', 'Wait, recheck.', 'Go on.', 'Hmm, again.',
                'Next step.', 'Alternatively, try.', 'More work.', 'Let me verify this.', 'Done.']
    traces = [dict(id=f'p{i}', reasoning='\n\n'.join(thoughts)) for i in range(5)]
    traces.append(dict(id='p5', reasoning='Only one thought.'))
    return (write_jsonl(root / 'problems.jsonl', problems),
            write_jsonl(root / 'traces.jsonl', traces))


class Tmp(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)
        self.work = self.root / 'work'

    def tearDown(self):
        self._tmp.cleanup()


class BuildersTest(Tmp):
    def test_thoughts_split_and_cap(self):
        self.assertEqual(data.thoughts('a\n\nb\n\n\n\nc'), ['a\n\n', 'b\n\n', 'c'])
        segs = data.thoughts('x' * 50 + '\n\n' + 'y' * 50, max_chars=60)
        self.assertEqual(''.join(segs), ('x' * 50 + '\n\n' + 'y' * 50)[:60])

    def test_trace_labels_and_shuffled_control(self):
        problems, traces = traces_fixture(self.root)
        source = [dict(name='s', problems=problems, traces=traces)]
        records, control, stats = data.build_traces(source, suffix='!')
        self.assertEqual(stats['traces'], 5)
        self.assertEqual(stats['skipped'], 1)
        ex = [r for r in records if r['group'] == 'exemplar']
        own = [r for r in records if r['group'] == 'own']
        # ends 1, 3, 5, 7 precede Wait/Hmm/Alternatively/Let me verify
        self.assertEqual(ex[0]['select'], [1, 3, 5, 7])
        self.assertEqual(own[0]['select'], [0, 2, 4, 6, 8])
        self.assertTrue(ex[0]['text'].endswith('?!'))
        self.assertEqual(ex[0]['segments'], own[0]['segments'])
        data.validate('segments', records)
        data.validate('segments', control)
        for real, shuf in zip(records, control):
            self.assertEqual(len(real['select']), len(shuf['select']))
            self.assertEqual(real['pair'], shuf['pair'])
            self.assertNotEqual(real['id'], shuf['id'])
        cx = [r for r in control if r['group'] == 'exemplar']
        co = [r for r in control if r['group'] == 'own']
        for a, b in zip(cx, co):
            self.assertEqual(sorted(a['select'] + b['select']), list(range(9)))
        again = data.build_traces(source, suffix='!')
        self.assertEqual(again[1], control)  # seeded
        self.assertNotEqual([r['select'] for r in cx], [r['select'] for r in ex])

    def test_min_each_and_cap(self):
        problems, traces = traces_fixture(self.root)
        source = [dict(name='s', problems=problems, traces=traces)]
        records, _, stats = data.build_traces(source, min_each=5)
        self.assertEqual(records, [])
        # A cap after the third thought leaves too few labels.
        records, _, _ = data.build_traces(source, max_chars=40)
        self.assertEqual(records, [])

    def test_pairs(self):
        problems, traces = traces_fixture(self.root)
        answers = write_jsonl(self.root / 'answers.jsonl', [
            dict(id='p0', text='ans0', finish_reason='stop'),
            dict(id='p1', text='ans1', finish_reason='length'),
            dict(id='p2', text='  ', finish_reason='stop'),
            dict(id='p3', answer='7', content='model says 7', finish_reason='stop')])
        records, stats = data.build_pairs([dict(name='a', problems=problems, traces=answers)],
                                          'Be careful.')
        self.assertEqual(stats, dict(pairs=2, skipped=2))
        self.assertEqual(records[2]['answer'], 'model says 7')  # never the reference
        records = records[:2]
        self.assertEqual([r['group'] for r in records], ['exemplar', 'own'])
        self.assertEqual(records[0]['text'], 'Be careful.\n\nWhat is 0+1?')
        self.assertEqual(records[1]['text'], 'What is 0+1?')
        self.assertEqual(records[0]['answer'], records[1]['answer'])
        data.validate('answers', records)
        seg, _ = data.build_pairs([dict(name='a', problems=problems, traces=traces)], 'Low.',
                                  mode='segments')
        self.assertEqual(seg[0]['select'], list(range(9)))
        self.assertEqual(len(seg), 10)

    def test_validate_rejects(self):
        good = dict(id='ab', split='c', group='own', pair='p', text='t', answer='a')
        data.validate('answers', [good])
        for bad in (dict(good, id='XY'), dict(good, group='refusal'), dict(good, answer='')):
            with self.assertRaises(ValueError):
                data.validate('answers', [bad])
        with self.assertRaises(ValueError):
            data.validate('answers', [good, good])

    def test_import_and_dataset_job(self):
        path = write_jsonl(self.root / 'in.jsonl', [
            dict(id='a1', split='c', group='refusal', text='x', thinking=False),
            dict(id='a2', split='c', group='ordinary', text='y', thinking=False)])
        store = jobs.JobStore(self.work)
        job = store.create('dataset', dict(builder='import', name='imp', path=path))
        jobs.run_job(job, ops.Context(), ops.OPERATIONS)
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        meta, records = data.load_dataset(self.work, 'imp')
        self.assertEqual(meta['kind'], 'collect')
        self.assertEqual(meta['groups'], {'refusal': 1, 'ordinary': 1})
        self.assertEqual(meta['data_sha256'],
                         hashlib.sha256(data.records_text(records).encode()).hexdigest())
        job = store.create('dataset', dict(builder='import', name='imp', path=path))
        jobs.run_job(job, ops.Context(), ops.OPERATIONS)
        self.assertEqual(job.state['status'], 'failed')
        self.assertIn('exists', job.state['error'])


class JobStoreTest(Tmp):
    def test_lifecycle_paging_and_restart(self):
        store = jobs.JobStore(self.work)
        job = store.create('test', {'x': 1})
        for i in range(5):
            self.assertEqual(job.emit(dict(type='row', i=i)), i + 1)
        self.assertEqual([r['i'] for r in job.view(3)['results']], [3, 4])
        self.assertEqual(job.view(5)['results'], [])
        job.set_status('running')
        # A new process: the running job failed, its results are kept.
        again = jobs.JobStore(self.work)
        state = again.get(job.key).view(0)
        self.assertEqual(state['status'], 'failed')
        self.assertIn('restart', state['error'])
        self.assertEqual(len(state['results']), 5)
        self.assertEqual(state['last_seq'], 5)
        # The CLI's view leaves other processes' jobs alone.
        other = again.create('fit', {})
        other.set_status('running')
        jobs.JobStore(self.work, recover=False)
        self.assertEqual(json.loads((other.dir / 'job.json').read_text())['status'], 'running')

    def test_cancel_then_delete(self):
        store = jobs.JobStore(self.work)
        job = store.create('test', {})
        self.assertEqual(store.delete(job.key)['cancel_requested'], True)
        self.assertEqual(job.state['status'], 'cancelled')
        # Never run (still queued): forgotten now, its folder removed when popped.
        self.assertEqual(store.delete(job.key), dict(job=job.key, deleted=True))
        self.assertIsNone(store.delete(job.key))
        jobs.run_job(job, ops.Context(), ops.OPERATIONS)
        self.assertFalse(job.dir.exists())
        done = store.create('test', {})
        jobs.run_job(done, ops.Context(), {'test': lambda j: None})
        self.assertEqual(store.delete(done.key), dict(job=done.key, deleted=True))
        self.assertFalse(done.dir.exists())

    def test_failure_is_reported(self):
        store = jobs.JobStore(self.work)
        job = store.create('test', {})

        def boom(j):
            j.emit(dict(type='row'))
            raise ValueError('bad input')
        jobs.run_job(job, ops.Context(), {'test': boom})
        self.assertEqual(job.state['status'], 'failed')
        self.assertEqual(job.state['error'], 'ValueError: bad input')
        self.assertTrue((job.dir / 'traceback.txt').exists())


class GufoTest(Tmp):
    """Base with a fake gufo server."""

    def setUp(self):
        super().setUp()
        self.collect_dir = self.root / 'collect'
        self.collect_dir.mkdir()
        self.fake = FakeGufo(self.collect_dir)
        self.gufo = client.Gufo(self.fake.url, KEY, timeout=30)
        self.context = ops.Context(gufo=self.gufo, collections_dir=str(self.collect_dir),
                                   poll_seconds=0.01)

    def tearDown(self):
        self.fake.close()
        super().tearDown()

    def run_job(self, operation, params):
        store = jobs.JobStore(self.work)
        job = store.create(operation, params)
        jobs.run_job(job, self.context, ops.OPERATIONS)
        return store, job


class ServiceTest(GufoTest):
    def setUp(self):
        super().setUp()
        self.store = jobs.JobStore(self.work)
        self.runner = jobs.Runner(self.store, self.context, ops.OPERATIONS)
        self.server = service.make_server(service.Api(self.store, self.runner, self.context),
                                          port=0)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.runner.stop()
        self.server.shutdown()
        self.server.server_close()
        super().tearDown()

    def call(self, method, path, body=None):
        request = urllib.request.Request(self.url + path, method=method,
                                         data=None if body is None else json.dumps(body).encode())
        try:
            with urllib.request.urlopen(request, timeout=10) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as error:
            with error:
                return error.code, json.loads(error.read())

    def wait_done(self, key):
        for _ in range(500):
            status, state = self.call('GET', f'/jobs/{key}')
            if state['status'] in jobs.FINISHED:
                return state
            time.sleep(0.02)
        self.fail('job did not finish')

    def test_job_api(self):
        problems, traces = traces_fixture(self.root)
        status, reply = self.call('POST', '/jobs', dict(operation='dataset', params=dict(
            builder='traces', name='rc', sources=[dict(name='s', problems=problems,
                                                       traces=traces)])))
        self.assertEqual(status, 200)
        self.assertEqual(set(reply), {'job', 'status'})
        state = self.wait_done(reply['job'])
        self.assertEqual(state['status'], 'done', state['error'])
        self.assertEqual(len(state['results']), 2)
        self.assertEqual(state['last_seq'], 2)
        _, later = self.call('GET', f'/jobs/{reply["job"]}?since=1')
        self.assertEqual([r['seq'] for r in later['results']], [2])
        _, listing = self.call('GET', '/jobs')
        self.assertEqual([j['key'] for j in listing['data']], [reply['job']])
        self.assertNotIn('results', listing['data'][0])
        _, datasets = self.call('GET', '/datasets')
        self.assertEqual(sorted(d['name'] for d in datasets['data']), ['rc', 'rc-shuf'])
        _, meta = self.call('GET', '/datasets/rc')
        self.assertEqual(meta['control'], 'rc-shuf')
        self.assertEqual(self.call('GET', f'/jobs/{reply["job"]}?since=-1')[0], 400)
        self.assertEqual(self.call('GET', '/jobs/nope')[0], 404)
        self.assertEqual(self.call('POST', '/jobs', dict(operation='x'))[0], 400)
        self.assertEqual(self.call('DELETE', f'/jobs/{reply["job"]}'),
                         (200, dict(job=reply['job'], deleted=True)))
        self.assertEqual(self.call('GET', f'/jobs/{reply["job"]}')[0], 404)

    def test_heavy_jobs_queue_and_cancel(self):
        gate = threading.Event()

        def slow(job):
            job.emit(dict(type='row'))
            while not gate.is_set():
                job.wait(0.01)
        self.runner.operations = dict(ops.OPERATIONS, test=slow)
        first = self.runner.submit('test', {})
        second = self.runner.submit('test', {})
        for _ in range(200):
            if first.state['status'] == 'running':
                break
            time.sleep(0.01)
        self.assertEqual(second.state['status'], 'queued')
        status, reply = self.call('DELETE', f'/jobs/{second.key}')
        self.assertEqual(reply['status'], 'cancelled')
        self.call('DELETE', f'/jobs/{first.key}')
        state = self.wait_done(first.key)
        self.assertEqual(state['status'], 'cancelled')
        self.assertEqual(state['last_seq'], 1)

    def test_profile_load_and_delete(self):
        path = self.root / 'wait.json'
        path.write_text('{"format":"gufo-logit-bias-v1","name":"wait"}')
        status, reply = self.call('POST', '/profiles/logit-bias/wait', dict(file=str(path)))
        self.assertEqual(status, 200)
        self.assertEqual(self.fake.profiles[('logit-bias', 'wait')], path.read_bytes())
        status, reply = self.call('DELETE', '/profiles/logit-bias/wait')
        self.assertEqual(reply['deleted'], True)
        status, reply = self.call('DELETE', '/profiles/logit-bias/wait')
        self.assertEqual((status, reply['error']['code']), (404, 'unknown_abliteration_profile'))
        self.assertEqual(self.call('POST', '/profiles/abliteration/a%2Fb', {})[0], 400)
        self.assertEqual(self.fake.auth_failures, 0)


class TestJobTest(GufoTest):
    def math_set(self):
        rows = [dict(id=f'q{i}', problem=f'What is {i}+{i}?', answer=str(2 * i),
                     split='eval' if i % 2 else 'dev') for i in range(1, 5)]
        rows.append(dict(id='q5', problem='What is 2+3?', answer='5', split='eval'))
        return write_jsonl(self.root / 'math.jsonl', rows)

    def test_math_configs(self):
        path = self.math_set()
        configs = [dict(name='base'),
                   dict(name='steer', abliteration={'enabled': True, 'profile': 'rc-p45'},
                        logit_bias_profiles=['wait'], logit_bias_overrides={'wait': -2},
                        reasoning_effort='low', sampling={'max_tokens': 999, 'temperature': 0.5})]
        _, job = self.run_job('test', dict(set=dict(kind='math', path=path, split='eval'),
                                           configs=configs, concurrency=2))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        rows = [r for r in job.results if r['type'] == 'row']
        self.assertEqual(len(rows), 6)
        base = {r['id']: r for r in rows if r['config'] == 'base'}
        # 1+1=2 and 3+3=6 are even: right; 2+3=5 odd: the fake answers 6.
        self.assertEqual({k: v['correct'] for k, v in base.items()},
                         {'q1': True, 'q3': True, 'q5': False})
        self.assertEqual(base['q5']['got'], '6')
        self.assertEqual(base['q1']['hesitations'], 2)
        self.assertEqual(base['q1']['tokens'], 120)
        self.assertFalse(base['q1']['truncated'])
        last = [r for r in job.results if r['type'] == 'totals' and r['config'] == 'steer'][-1]
        self.assertEqual((last['n'], last['correct'], last['tokens_mean']), (3, 3, 50))
        self.assertEqual(job.state['output']['totals']['base']['correct'], 2)
        bodies = {(b['messages'][0]['content'], json.dumps(b.get('abliteration'))): b
                  for b in self.fake.chats}
        steer = [b for b in self.fake.chats if b['abliteration'].get('enabled')]
        plain = [b for b in self.fake.chats if not b['abliteration'].get('enabled')]
        self.assertEqual(len(bodies), 6)
        self.assertEqual(plain[0]['abliteration'], {'enabled': False})
        self.assertEqual(plain[0]['max_tokens'], 16384)
        self.assertNotIn('reasoning_effort', plain[0])
        b = steer[0]
        self.assertEqual((b['logit_bias_profiles'], b['logit_bias_overrides'],
                          b['reasoning_effort'], b['max_tokens'], b['temperature']),
                         (['wait'], {'wait': -2}, 'low', 999, 0.5))
        self.assertTrue(b['messages'][0]['content'].endswith('\\boxed{}.'))
        # think_eval.py's seed: 1000 + the problem's index within the split.
        seeds = {b['messages'][0]['content'].split('?')[0]: b['seed'] for b in plain}
        self.assertEqual(seeds['What is 1+1'], 1000)
        self.assertEqual(seeds['What is 2+3'], 1002)
        lines = (job.dir / 'generations' / 'steer.jsonl').read_text().splitlines()
        self.assertEqual(json.loads(lines[0])['reasoning'], 'Add.')

    def test_bad_reply_is_an_error_row(self):
        path = self.math_set()
        self.fake.answer = lambda request: {'unexpected': True}
        _, job = self.run_job('test', dict(set=dict(kind='math', path=path, limit=2),
                                           configs=[dict(name='a')]))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        rows = [r for r in job.results if r['type'] == 'row']
        self.assertTrue(all('KeyError' in r['error'] for r in rows))
        self.assertEqual(job.state['output']['totals']['a']['errors'], 2)

    def test_bad_configs(self):
        path = self.math_set()
        _, job = self.run_job('test', dict(set=dict(kind='math', path=path),
                                           configs=[dict(name='a', bogus=1)]))
        self.assertEqual(job.state['status'], 'failed')
        _, job = self.run_job('test', dict(set=dict(kind='math', path=path),
                                           configs=[dict(name='a'), dict(name='a')]))
        self.assertIn('unique', job.state['error'])

    def test_coding_set(self):
        sys.path.insert(0, str(HERE.parent / 'qwen38_flash_next' / 'coding_eval'))
        import sandbox
        problem = dict(
            id='list-01', split='eval', lang='en', category='list', difficulty='easy',
            entry_point='head', signature='def head(items: list, n: int) -> list:',
            prompt='Return the first n items.',
            tests=[dict(args='([1, 2, 3], 2)', expect='[1, 2]', edge=None)])
        folder = self.root / 'coding'
        folder.mkdir()
        write_jsonl(folder / 'problems.jsonl', [problem])

        def answer(request):
            code = 'def head(items, n):\n    return items[:n]\n'
            return {'choices': [{'message': {'content': f'```python\n{code}```'},
                                 'finish_reason': 'stop'}],
                    'usage': {'completion_tokens': 20}}
        self.fake.answer = answer
        # Our own fixture code: the unsandboxed runner is fine here only.
        self.context.coding_runner = lambda spec: sandbox.LocalRunner(trusted=True)
        _, job = self.run_job('test', dict(set=dict(kind='coding', path=str(folder)),
                                           configs=[dict(name='base')]))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        row = [r for r in job.results if r['type'] == 'row'][0]
        self.assertEqual((row['correct'], row['passed'], row['tests']), (True, 1, 1))
        body = self.fake.chats[0]
        self.assertEqual((body['temperature'], body['max_tokens']), (0, 600))
        self.assertEqual(body['chat_template_kwargs'], {'enable_thinking': False})
        _, job = self.run_job('test', dict(set=dict(kind='coding', path=str(folder),
                                                    runner='local'),
                                           configs=[dict(name='base')]))
        self.assertIn('Docker sandbox', job.state['error'])


class TrainJobTest(GufoTest):
    def test_spsa_steps_and_best(self):
        pairs = write_jsonl(self.root / 'pairs.jsonl', [
            dict(id=f'p{i}', prompt='q', candid='candid', propaganda='spin') for i in range(2)])
        ordinary = write_jsonl(self.root / 'ord.jsonl', [dict(id='o', prompt='q', answer='fine')])
        args = ['--pairs', pairs, '--ordinary', ordinary, '--layers', str(LAYERS), '--init', '0',
                '--additive', '--min-strength', '-2', '--max-strength', '2', '--steps', '3',
                '--eval-every', '1', '--profile', 'rc-add', '--concurrency', '2']
        _, job = self.run_job('train', dict(args=args))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        steps = [r for r in job.results if r['type'] == 'step']
        self.assertEqual([s['step'] for s in steps], [-1, 0, 1, 2])
        self.assertIn('eval', steps[-1])
        best = [r for r in job.results if r['type'] == 'best'][0]
        self.assertEqual(best['settings']['profile'], 'rc-add')
        self.assertEqual(job.state['progress']['done'], 3)
        self.assertEqual(self.fake.auth_failures, 0)  # the key reached spsa.py's requests
        for bad in (['--log', 'x'], ['--out=x'], ['--base', 'u']):
            _, job = self.run_job('train', dict(args=bad))
            self.assertIn('set by the tool', job.state['error'])


class CollectOnlyTest(GufoTest):
    def test_collect_only_needs_no_numpy(self):
        problems, traces = traces_fixture(self.root)
        self.run_job('dataset', dict(builder='traces', name='rc', sources=[
            dict(name='s', problems=problems, traces=traces)]))
        _, job = self.run_job('fit', dict(dataset='rc', collect_only=True))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        out = job.state['output']
        self.assertTrue((Path(out['collection']) / 'manifest.json').exists())
        self.assertTrue((Path(out['control_collection']) / 'manifest.json').exists())
        self.assertEqual(len(self.fake.jobs), 2)
        records = data.load_dataset(self.work, 'rc')[1]
        sent = list(self.fake.jobs.values())[0]['request']
        self.assertEqual(sent['records'], records)
        self.assertEqual(sent['name'], 'rc-' + data.sha_text(data.records_text(records))[:12])


@unittest.skipUnless(HAVE_NUMPY, 'fit needs NumPy (pinned container)')
class FitJobTest(GufoTest):
    def test_fit_with_control(self):
        problems, traces = traces_fixture(self.root)
        store = jobs.JobStore(self.work)
        job = store.create('dataset', dict(builder='traces', name='rc', control_name='c-rc',
                                           sources=[dict(name='s', problems=problems,
                                                         traces=traces)]))
        jobs.run_job(job, self.context, ops.OPERATIONS)
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        _, job = self.run_job('fit', dict(dataset='rc', top_n=2, profile_layers=[PLANTED]))
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        collects = [r for r in job.results if r['type'] == 'collect']
        self.assertEqual({c['dataset'] for c in collects}, {'real', 'control'})
        self.assertEqual(collects[-1]['status'], 'done')
        sent = [j['request'] for j in self.fake.jobs.values()]
        self.assertEqual([(s['mode'], s['resume']) for s in sent], [('segments', True)] * 2)
        layers = [r for r in job.results if r['type'] == 'layer' and r['dataset'] == 'real']
        self.assertEqual([r['layer'] for r in layers], list(range(LAYERS)))
        self.assertEqual(layers[PLANTED]['separated'], 5)
        top = [r for r in job.results if r['type'] == 'top']
        self.assertEqual(top[0]['layers'][0], PLANTED)
        control = [r for r in job.results if r['type'] == 'layer' and r['dataset'] == 'control']
        self.assertLess(control[PLANTED]['snr'] or 0, layers[PLANTED]['snr'])
        cosine = [r for r in job.results if r['type'] == 'cosine'][0]
        self.assertEqual(len(cosine['matrix']), LAYERS)
        self.assertAlmostEqual(cosine['matrix'][1][1], 1.0, places=3)
        profiles = {r['name']: r for r in job.results if r['type'] == 'profile'}
        self.assertEqual(set(profiles), {'rc-add', f'rc-p{PLANTED}', f'rc-s{PLANTED}'})
        import prepare
        pol = json.loads(Path(profiles[f'rc-p{PLANTED}']['path']).read_text())
        self.assertEqual((pol['direction_layer'], pol['residual'], pol['attention'], pol['mlp']),
                         (PLANTED, [1.0] * LAYERS, [0.0] * LAYERS, [0.0] * LAYERS))
        self.assertEqual(pol['direction_kind'], 'projected')
        self.assertNotIn('mode', pol)
        one = json.loads(Path(profiles[f'rc-s{PLANTED}']['path']).read_text())
        self.assertEqual(one['residual'], [1.0 if i == PLANTED else 0.0 for i in range(LAYERS)])
        add = json.loads(Path(profiles['rc-add']['path']).read_text())
        self.assertEqual(add['mode'], 'add')
        prepare.make_profile(add, PLANTED, [0.0] * LAYERS, [0.0] * LAYERS)  # loader checks
        # Upload a fitted profile through the load path.
        raw = ops.profile_bytes(jobs.JobStore(self.work), 'abliteration', 'rc-add',
                                dict(job=job.key))
        self.gufo.upload_profile('abliteration', 'rc-add', raw)
        self.assertEqual(self.fake.profiles[('abliteration', 'rc-add')], raw)
        # A second fit resumes the same collections (same names, resume).
        _, again = self.run_job('fit', dict(dataset='rc', control=False))
        self.assertEqual(again.state['status'], 'done', again.state['error'])
        names = [j['request']['name'] for j in self.fake.jobs.values()]
        self.assertEqual(names[0], names[2])
        # A gufo restart forgets the job: the fit resumes it by name.
        polls = []
        original = self.fake.poll

        def forgetful(key, job, since):
            polls.append(key)
            if len(polls) == 1:
                self.fake.jobs.pop(key)
                return original(key, job, since)
            return original(key, job, since)
        self.fake.poll = forgetful
        self.fake.jobs.clear()
        shutil.rmtree(self.collect_dir / names[0])
        _, third = self.run_job('fit', dict(dataset='rc', control=False))
        self.assertEqual(third.state['status'], 'done', third.state['error'])
        self.assertEqual(len(self.fake.jobs), 1)  # the forgotten one and its restart

    def test_fit_existing_collection(self):
        records = [dict(id=f'{i:02x}{g}', split='c', group=('exemplar' if g == 'a' else 'own'),
                        pair=f'q{i}', text='t', answer='a') for i in range(6) for g in 'ab']
        self.fake.write_collection(dict(name='plain', mode='answers', records=records))
        context = ops.Context()  # no gufo server at all
        store = jobs.JobStore(self.work)
        job = store.create('fit', dict(collection=str(self.collect_dir / 'plain'),
                                       profile_layers=[1]))
        jobs.run_job(job, context, ops.OPERATIONS)
        self.assertEqual(job.state['status'], 'done', job.state['error'])
        self.assertEqual(job.state['output']['top'][0], PLANTED)
        self.assertEqual(job.state['output']['profiles'], ['plain-add', 'plain-p1', 'plain-s1'])


class CliTest(GufoTest):
    def test_params(self):
        ap = cli.parser()
        args = ap.parse_args(['test', '--set-kind', 'math', '--set', 'm.jsonl', '--config',
                              'base={}', '--config', 'low={"reasoning_effort":"low"}',
                              '--concurrency', '4'])
        op, p = cli.params_for(args)
        self.assertEqual(op, 'test')
        self.assertEqual(p['configs'], [{'name': 'base'},
                                        {'reasoning_effort': 'low', 'name': 'low'}])
        self.assertEqual(p['set'], {'kind': 'math', 'path': 'm.jsonl'})
        args = ap.parse_args(['train', '--', '--pairs', 'p', '--additive'])
        self.assertEqual(cli.params_for(args), ('train', {'args': ['--pairs', 'p', '--additive']}))

    def test_inline_dataset(self):
        problems, traces = traces_fixture(self.root)
        out = []

        class Sink:
            def write(self, text):
                out.append(text)

            def flush(self):
                pass
        store = jobs.JobStore(self.work, recover=False)
        args = cli.parser().parse_args(['dataset', 'traces', '--name', 'rc', '--source',
                                        f's={problems},{traces}'])
        rc = cli.run_inline(store, self.context, *cli.params_for(args), out=Sink())
        self.assertEqual(rc, 0)
        self.assertIn('"name": "rc"', ''.join(out))


if __name__ == '__main__':
    unittest.main()
