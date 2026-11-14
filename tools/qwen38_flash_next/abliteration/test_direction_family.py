"""CPU tests of the R11 direction family: the basis (direction_basis.py, NumPy;
skipped without it), the family and its profiles (direction_family.py), and
spsa.py --basis against a fake gufo server that holds uploaded profiles and
scores by their direction rows (standard library)."""
import http.server
import importlib.util
import json
import math
from pathlib import Path
import random
import sys
import tempfile
import threading
import unittest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import direction_family as df  # noqa: E402
import prepare  # noqa: E402

spec = importlib.util.spec_from_file_location('spsa', HERE / 'spsa.py')
spsa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(spsa)

try:
    import numpy as np
    import direction_basis
except ImportError:
    np = None

HIDDEN, LAYERS, LAYER = 8, 4, 2


def unit(i, n=HIDDEN):
    return [1.0 if j == i else 0.0 for j in range(n)]


def base_profile():
    """A rank-1 projected suppression profile; its row at LAYER is e0."""
    rows = [unit((layer + 3) % HIDDEN) for layer in range(LAYERS)]
    rows[LAYER] = unit(0)
    return dict(schema=prepare.SCHEMA, model_identity='fake', hidden=HIDDEN, layers=LAYERS,
                placement='writer-before-hc', directions=rows, direction_layer=LAYER,
                attention=[0.0] * LAYERS, mlp=[0.0] * LAYERS, residual=[1.0] * LAYERS, rank=1,
                projected_directions=[df.encode_row(r) for r in rows], direction_kind='projected',
                direction_count=1, preserve_norm=False)


def basis_file(path, k=2):
    path.write_text(json.dumps(dict(schema=df.SCHEMA, hidden=HIDDEN, d0=unit(0),
                                    basis=[unit(i + 1) for i in range(k)])))
    return str(path)


class FamilyTest(unittest.TestCase):
    def test_angle_cap_holds(self):
        basis = dict(schema=df.SCHEMA, hidden=HIDDEN, d0=unit(0), basis=[unit(1), unit(2), unit(3)])
        rng = random.Random(1)
        for _ in range(200):
            alpha = [rng.uniform(-2, 2) for _ in range(3)]
            capped = df.cap(alpha, 20.0)
            self.assertLessEqual(df.angle(basis, capped), 20.0 + 1e-9)
            self.assertAlmostEqual(df.angle(basis, capped),
                                   math.degrees(math.atan(df.norm(capped))), places=9)
            if df.norm(alpha) <= math.tan(math.radians(20)):
                self.assertEqual(capped, alpha)  # inside the cap: unchanged
            else:
                self.assertAlmostEqual(df.angle(basis, capped), 20.0, places=9)
        with self.assertRaises(ValueError):
            df.max_norm(90)

    def test_profile_round_trip_and_alpha_zero(self):
        base = base_profile()
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        basis = df.load_basis(basis_file(Path(tmp.name) / 'b.json'))
        df.check_base(base, basis, LAYER, 'projected')
        same = df.profile_with_direction(base, basis, [0.0, 0.0], LAYER, 'projected')
        self.assertEqual(same['projected_directions'], base['projected_directions'])
        self.assertEqual(same['directions'], base['directions'])
        alpha = [0.2, -0.1]
        tuned = df.profile_with_direction(base, basis, alpha, LAYER, 'projected')
        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / 'p.json'
            path.write_bytes(df.profile_bytes(tuned))
            loaded = prepare.load_base(profile=path)
        row = prepare.decode_rows(loaded['projected_directions'][LAYER], 1, HIDDEN, 'row')[0]
        want = df.direction(basis, alpha)
        self.assertTrue(all(abs(a - b) < 1e-7 for a, b in zip(row, want)))
        self.assertEqual(list(row), [df.float32(x) for x in want])  # exactly the float32 values
        self.assertEqual(loaded['directions'][LAYER], [df.float32_text(x) for x in row])
        prepare.check_orthonormal([row], 'row')  # a unit row, as the loader checks
        # The loader's own validation (make_profile) accepts it.
        prepare.make_profile(loaded, LAYER, loaded['attention'], loaded['mlp'], loaded['residual'])
        # Other layers are untouched.
        for layer in range(LAYERS):
            if layer != LAYER:
                self.assertEqual(loaded['projected_directions'][layer], base['projected_directions'][layer])
        # A basis built for another row is refused; so is a rank-2 base.
        with self.assertRaisesRegex(ValueError, 'not the base profile'):
            df.check_base(base, basis, 1, 'projected')
        with self.assertRaisesRegex(ValueError, 'rank-1'):
            df.check_base(dict(base, rank=2), basis, LAYER, 'projected')

    @unittest.skipIf(np is None, 'NumPy (pinned container)')
    def test_encoding_matches_prepare(self):
        rng = random.Random(4)
        values = [rng.gauss(0, 1) for _ in range(64)]
        self.assertEqual(df.encode_row(values), prepare.encode_rows(np.array(values)[None]))
        self.assertEqual(df.encode_row(values), prepare.encode_rows(values))


@unittest.skipIf(np is None, 'NumPy (pinned container)')
class BasisTest(unittest.TestCase):
    def synthetic(self, flip_layer=None):
        """D [pairs][layers][hidden]: per layer a mean along d0 plus a small
        per-layer tilt, and per-pair variation along two planted axes
        (orthogonal to d0) with standard deviations 2 and 1, plus noise. The
        own mean is along the last axis (removed by the projected kind)."""
        rng = np.random.default_rng(0)
        hidden, layers, pairs = 16, 6, 400
        d0 = np.zeros(hidden); d0[0] = 1
        a1 = np.zeros(hidden); a1[1] = 1
        a2 = np.zeros(hidden); a2[2] = 1
        own = np.zeros((layers, hidden)); own[:, -1] = 5.0
        D = np.zeros((pairs, layers, hidden))
        for layer in range(layers):
            scale = 1.0 + layer
            mean = scale * d0 + 0.05 * scale * rng.normal(size=hidden) * (np.arange(hidden) > 4)
            sign = -1.0 if layer == flip_layer else 1.0
            D[:, layer] = sign * scale * (mean / scale + np.outer(rng.normal(0, 2, pairs), a1)
                                          + np.outer(rng.normal(0, 1, pairs), a2)
                                          + 0.05 * rng.normal(size=(pairs, hidden)))
            D[:, layer, -1] += 3.0  # along the own mean: the projected kind drops it
        return D, own, d0, a1, a2

    def test_residual_pca_and_sign_alignment(self):
        for flip in (None, 1):
            D, own, d0, a1, a2 = self.synthetic(flip)
            layers = [0, 1, 2, 3]
            X, signs, units = direction_basis.layer_rows(D, own, layers, ref_layer=2)
            self.assertEqual(signs, [1.0, -1.0 if flip == 1 else 1.0, 1.0, 1.0])
            self.assertTrue(all(u @ d0 > 0.9 for u in units))  # every layer aligned to the reference
            self.assertTrue(np.allclose(X[:, -1], 0, atol=1e-9))  # projected: no own-mean component
            u, variance, ratio, total = direction_basis.build(X, d0, 3)
            self.assertTrue(np.allclose(u @ u.T, np.eye(3), atol=1e-10))  # orthonormal
            self.assertTrue(np.allclose(u @ d0, 0, atol=1e-12))           # orthogonal to d0
            self.assertGreater(abs(u[0] @ a1), 0.99)  # the larger planted axis first
            self.assertGreater(abs(u[1] @ a2), 0.99)
            self.assertAlmostEqual(variance[0], 4.0, delta=0.5)  # sd 2 (rows are scaled by 1/|m_l|)
            self.assertAlmostEqual(variance[1], 1.0, delta=0.2)
            self.assertGreater(ratio[0] + ratio[1], 0.9)
            self.assertTrue(all(v[np.argmax(np.abs(v))] > 0 for v in u))  # sign convention
            # Against plain NumPy PCA of the residuals.
            R = X - np.outer(X @ d0, d0)
            R = R - R.mean(axis=0)
            w, v = np.linalg.eigh(R.T @ R / (len(R) - 1))
            self.assertAlmostEqual(variance[0], w[-1], places=8)
            self.assertGreater(abs(v[:, -1] @ u[0]), 1 - 1e-8)


class FakeProfiles:
    """/props with the loaded abliteration profiles, profile uploads and
    deletes (refusing a ninth profile), and scores from the named profile's
    row at LAYER: a candid answer gains with the mean residual strength times
    the row's alignment with TARGET; an ordinary answer loses with strength.
    Every request is logged in order."""

    TARGET = None

    def __init__(self, loaded):
        self.profiles = {name: base_profile() for name in loaded}
        self.log, self.max_loaded = [], len(self.profiles)
        self.lock = threading.Lock()
        fake = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def reply(self, status, value):
                payload = json.dumps(value).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def do_GET(self):
                self.reply(200, {'abliteration': {'profiles': [{'profile': n} for n in sorted(fake.profiles)]}})

            def do_DELETE(self):
                name = self.path.rsplit('/', 1)[1]
                with fake.lock:
                    fake.log.append(('DELETE', name))
                    if name not in fake.profiles:
                        return self.reply(404, {'error': {'code': 'unknown_abliteration_profile'}})
                    del fake.profiles[name]
                self.reply(200, {'profile': name, 'deleted': True})

            def do_POST(self):
                body = self.rfile.read(int(self.headers['Content-Length']))
                if self.path.startswith('/v1/abliteration/profiles/'):
                    name = self.path.rsplit('/', 1)[1]
                    with fake.lock:
                        fake.log.append(('POST', name))
                        if name not in fake.profiles and len(fake.profiles) >= 8:
                            return self.reply(400, {'error': {'code': 'invalid_abliteration'}})
                        fake.profiles[name] = json.loads(body)
                        fake.max_loaded = max(fake.max_loaded, len(fake.profiles))
                    return self.reply(200, {'profile': name})
                request = json.loads(body)
                ab = request.get('abliteration')
                with fake.lock:
                    fake.log.append(('CHAT', ab.get('profile') if ab else None))
                    profile = fake.profiles.get(ab.get('profile')) if ab else None
                if ab and profile is None:
                    return self.reply(400, {'error': {'code': 'unknown_abliteration_profile'}})
                strength = sum(ab['residual']) / LAYERS if ab else 0.0
                align = 0.0
                if ab:
                    row = prepare.decode_rows(profile['projected_directions'][LAYER], 1, HIDDEN, 'r')[0]
                    align = sum(a * b for a, b in zip(row, FakeProfiles.TARGET))
                answer = request['messages'][-1]['content']
                if answer.startswith('candid'):
                    value = -2.0 + strength * align
                elif answer.startswith('prop'):
                    value = -1.0
                else:
                    value = -1.0 - 0.1 * strength
                self.reply(200, {'score': {'token_count': 1, 'logprob_sum': value}})

        self.server = http.server.ThreadingHTTPServer(('127.0.0.1', 0), Handler)
        self.url = f'http://127.0.0.1:{self.server.server_address[1]}'
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class SpsaDirectionTest(unittest.TestCase):
    def setUp(self):
        # The best direction is 15 degrees from d0 towards u1.
        t = math.radians(15)
        FakeProfiles.TARGET = [math.cos(t), math.sin(t)] + [0.0] * (HIDDEN - 2)
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.dir = Path(tmp.name)
        d = self.dir
        (d / 'pairs.jsonl').write_text(json.dumps(
            {'id': 'p', 'prompt': 'q', 'candid': 'candid a', 'propaganda': 'prop a'}) + '\n')
        (d / 'ord.jsonl').write_text(json.dumps({'id': 'o', 'prompt': 'q', 'answer': 'plain'}) + '\n')
        (d / 'base.json').write_text(json.dumps(base_profile()))
        self.basis = basis_file(d / 'basis.json')

    def args(self, url, log, *extra):
        d = self.dir
        return ['--base-url', url, '--pairs', str(d / 'pairs.jsonl'), '--ordinary', str(d / 'ord.jsonl'),
                '--layers', str(LAYERS), '--direction-layer', str(LAYER), '--direction-kind', 'projected',
                '--init', '{"attention": [0, 0, 0, 0], "mlp": [0, 0, 0, 0], "residual": [1, 1, 1, 1]}',
                '--learn', 'residual', '--log', str(d / log), '--concurrency', '2', '--seed', '2',
                '--a', '0.05', '--c', '0.05', *extra]

    def direction_args(self, url, log, *extra):
        return self.args(url, log, '--basis', self.basis, '--base-profile', str(self.dir / 'base.json'),
                         '--max-angle', '20', '--c-dir', '0.04', '--a-dir', '0.08', *extra)

    def test_uploads_order_cap_and_alpha_zero(self):
        fake = FakeProfiles(['base', 'b1', 'b2', 'b3', 'b4'])  # 5 loaded + 3 names = the cap
        try:
            # Fixed direction (the loaded base) against alpha = 0: the same scores.
            spsa.main(self.args(fake.url, 'fixed.jsonl', '--profile', 'base', '--steps', '0',
                                '--eval-every', '1'))
            fixed = json.loads((self.dir / 'fixed.jsonl').read_text().splitlines()[1])['eval']
            fake.log.clear()
            out = self.dir / 'best.json'
            ready = self.dir / 'best-profile.json'
            self.assertEqual(spsa.main(self.direction_args(
                fake.url, 'dir.jsonl', '--steps', '40', '--eval-every', '10', '--output', str(out),
                '--profile-output', str(ready))), 0)
            lines = [json.loads(l) for l in (self.dir / 'dir.jsonl').read_text().splitlines()]
            self.assertEqual(lines[1]['alpha'], [0.0, 0.0])
            self.assertEqual(lines[1]['angle'], 0.0)
            self.assertEqual(lines[1]['eval'], fixed)  # alpha = 0 scores exactly as the fixed direction
            self.assertEqual(lines[0]['basis_sha256'], df.file_sha256(self.basis))
            # Order: each name is uploaded before the requests scoring it, the
            # cap is never passed, and every upload is deleted at the end.
            self.assertLessEqual(fake.max_loaded, 8)
            current = {}
            for i, (method, name) in enumerate(fake.log):
                if method == 'POST':
                    current[name] = i
                elif method == 'CHAT' and name is not None:
                    self.assertIn(name, current)
                    self.assertTrue(name.startswith('r11-'))
            self.assertEqual(fake.log[-3:], [('DELETE', 'r11-p'), ('DELETE', 'r11-m'), ('DELETE', 'r11-e')])
            self.assertEqual(sorted(fake.profiles), ['b1', 'b2', 'b3', 'b4', 'base'])
            # A step uploads theta+ and theta- and then scores them in one burst.
            first_step = fake.log.index(('POST', 'r11-p'), fake.log.index(('POST', 'r11-e')))
            self.assertEqual(fake.log[first_step + 1], ('POST', 'r11-m'))
            steps = [r for r in lines[1:] if r['step'] >= 0]
            self.assertTrue(all(r['angle'] <= 20.0 + 1e-6 for r in steps))
            # The direction moves towards the target (u1, 15 degrees).
            self.assertGreater(steps[-1]['alpha'][0], 0.1)
            self.assertGreater(steps[-1]['angle'], 5.0)
            best = json.loads(out.read_text())
            self.assertEqual(len(best['alpha']), 2)
            self.assertNotIn('profile', best['settings'])
            profile = json.loads(ready.read_text())
            row = prepare.decode_rows(profile['projected_directions'][LAYER], 1, HIDDEN, 'r')[0]
            want = df.direction(df.load_basis(self.basis), best['alpha'])
            self.assertTrue(all(abs(a - b) < 1e-6 for a, b in zip(row, want)))
            self.assertEqual(profile['residual'], best['settings']['residual'])
            self.assertEqual(profile['direction_layer'], LAYER)
            # Resume continues the log (same header) and the alpha it reached.
            fake.log.clear()
            self.assertEqual(spsa.main(self.direction_args(
                fake.url, 'dir.jsonl', '--steps', '42', '--eval-every', '10')), 0)
            again = [json.loads(l) for l in (self.dir / 'dir.jsonl').read_text().splitlines()]
            self.assertEqual([r['step'] for r in again[1:]], list(range(-1, 42)))
            # A different direction option is another run.
            with self.assertRaises(SystemExit):
                spsa.main(self.direction_args(fake.url, 'dir.jsonl', '--steps', '43', '--c-dir', '0.03'))
        finally:
            fake.close()

    def test_cap_refusal_two_names_and_bad_options(self):
        fake = FakeProfiles(['base', 'b1', 'b2', 'b3', 'b4', 'b5'])  # 6 loaded + 3 names > 8
        try:
            with self.assertRaisesRegex(SystemExit, 'cap'):
                spsa.main(self.direction_args(fake.url, 'cap.jsonl', '--steps', '1'))
            self.assertFalse([e for e in fake.log if e[0] == 'POST'])  # refused before any upload
            # Two names: evals reuse the first; 6 + 2 fits.
            self.assertEqual(spsa.main(self.direction_args(
                fake.url, 'two.jsonl', '--steps', '3', '--eval-every', '1',
                '--dir-names', 'r11-a,r11-b')), 0)
            self.assertLessEqual(fake.max_loaded, 8)
            names = {n for m, n in fake.log if m == 'POST'}
            self.assertEqual(names, {'r11-a', 'r11-b'})
            self.assertNotIn('r11-a', fake.profiles)
            for bad in (['--profile', 'base'], ['--additive'], ['--dir-names', 'one'],
                        ['--max-angle', '95']):
                with self.assertRaises(SystemExit):
                    spsa.main(self.direction_args(fake.url, 'bad.jsonl', '--steps', '1', *bad))
            with self.assertRaises(SystemExit):  # direction options need --basis
                spsa.main(self.args(fake.url, 'bad2.jsonl', '--steps', '1', '--base-profile', 'x'))
        finally:
            fake.close()

    def test_fixed_direction_header_unchanged(self):
        """Without --basis the log header has none of the direction options,
        so fixed-direction logs written before them still resume."""
        fake = FakeProfiles(['base'])
        try:
            spsa.main(self.args(fake.url, 'h.jsonl', '--profile', 'base', '--steps', '1'))
            header = json.loads((self.dir / 'h.jsonl').read_text().splitlines()[0])
            self.assertFalse(set(spsa.DIRECTION_ARGS) & set(header['args']))
            self.assertNotIn('profile_output', header['args'])
            self.assertNotIn('basis_sha256', header)
            self.assertNotIn('alpha', json.loads((self.dir / 'h.jsonl').read_text().splitlines()[1]))
        finally:
            fake.close()


if __name__ == '__main__':
    unittest.main()
