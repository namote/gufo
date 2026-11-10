"""CPU contracts for saved collection integrity, train/evaluation separation,
projected and subspace directions and profile members."""
import contextlib
import importlib.util
import io
import json
import math
import sys
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location('prepare', Path(__file__).with_name('prepare.py'))
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparationTest(unittest.TestCase):
    FLASH_NEXT = 'last-prompt-token-post-layer-hc-mean-fp32-v1'
    QWEN27B = 'last-prompt-token-post-layer-fp32-v1'

    def fixture(self, root, split='collection', readout=FLASH_NEXT, hidden=8,
                layers=4, **extra):
        source = root / 'collection'
        source.mkdir()
        rows = []
        for group, sign in [('refusal', 1), ('ordinary', -1)]:
            for i in range(2):
                identity = f'{group}{i}'
                rows.append(dict(id=identity, group=group))
                raw = np.full((layers, hidden), sign * (i + 1), dtype='<f4').tobytes()
                (source / (identity + '.f32')).write_bytes(raw)
                prepare.save(source / (identity + '.json'), dict(activation_sha256=prepare.sha(raw)))
        prepare.save(source / 'manifest.json', dict(split=split, hidden=hidden, layers=layers,
                     records=rows, model_identity='model-fixture', readout=readout,
                     **extra))
        return source

    def test_unit_directions_and_separate_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            prepare.fit(source, root / 'candidates')
            candidates = list((root / 'candidates').glob('layer*.json'))
            self.assertTrue(candidates)
            for path in candidates:
                candidate = json.loads(path.read_text())
                self.assertEqual(candidate['schema'], 'gufo-abliteration-v2')
                self.assertEqual(len(candidate['directions']), 4)
                for row in candidate['directions']:
                    self.assertAlmostEqual(np.linalg.norm(row), 1., places=6)
                self.assertIn(candidate['direction_layer'], range(4))
                self.assertEqual(candidate['model_identity'], 'model-fixture')
                self.assertEqual(candidate['status'], 'candidate-not-qualified')
                self.assertEqual(len(candidate['attention']), 4)
                self.assertEqual(candidate['placement'], 'writer-before-hc')
            self.assertFalse((root / 'default.json').exists())
            summary = json.loads((root / 'candidates' / 'SUMMARY.json').read_text())
            self.assertEqual(summary['placement'], 'writer-before-hc')

    def test_qwen27b_capture_geometry_and_placement(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root, readout=self.QWEN27B, hidden=6, layers=9,
                                  placement='writer-before-residual')
            prepare.fit(source, root / 'candidates')
            candidates = list((root / 'candidates').glob('layer*.json'))
            self.assertTrue(candidates)
            for path in candidates:
                candidate = json.loads(path.read_text())
                self.assertEqual(candidate['placement'], 'writer-before-residual')
                self.assertEqual((candidate['hidden'], candidate['layers']), (6, 9))
                self.assertEqual(len(candidate['directions']), 9)
                self.assertEqual(len(candidate['directions'][0]), 6)
                self.assertEqual(len(candidate['mlp']), 9)
                self.assertAlmostEqual(np.linalg.norm(candidate['directions'][8]), 1.,
                                       places=6)

    def test_placement_must_match_readout(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root, readout=self.QWEN27B, placement='writer-before-hc')
            with self.assertRaisesRegex(ValueError, 'placement'):
                prepare.fit(source, root / 'candidates')

    def test_evaluation_output_cannot_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root, readout='first-next-token-logits-fp32-v1')
            with self.assertRaisesRegex(ValueError, 'readout'):
                prepare.fit(source, root / 'candidates')

    def test_heldout_cannot_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaisesRegex(ValueError, 'never tuning/evaluation'):
                prepare.fit(self.fixture(root, 'evaluation'), root / 'candidates')

    def test_corrupt_saved_activation_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = self.fixture(root)
            (source / 'refusal0.f32').write_bytes(b'corrupt')
            with self.assertRaisesRegex(ValueError, 'integrity'):
                prepare.fit(source, root / 'candidates')


class ProfileTest(unittest.TestCase):
    def test_curve_formula(self):
        # max_w at the peak, linear to min_w at distance width, 0 beyond.
        weights = prepare.curve_weights(10, max_w=2, peak=4, min_w=0.5, width=3)
        expected = [0, 0.5, 1.0, 1.5, 2, 1.5, 1.0, 0.5, 0, 0]
        for got, want in zip(weights, expected):
            self.assertAlmostEqual(got, want)
        fractional = prepare.curve_weights(3, max_w=1, peak=0.5, min_w=0, width=1)
        self.assertEqual([round(w, 9) for w in fractional], [0.5, 0.5, 0])
        with self.assertRaises(ValueError):
            prepare.curve_weights(3, max_w=5, peak=1, min_w=0, width=1)
        with self.assertRaises(ValueError):
            prepare.curve_weights(3, max_w=1, peak=1, min_w=0, width=0)

    def test_parse_weights(self):
        self.assertEqual(prepare.parse_weights('2', 3), [2, 2, 2])
        self.assertEqual(prepare.parse_weights('0,1,2', 3), [0, 1, 2])
        self.assertEqual(prepare.parse_weights('max_w=2,peak=1,min_w=1,width=1', 3),
                         [1, 2, 1])
        for bad in ('0,1', '-1', '4.5', 'max_w=2,peak=1', 'nan'):
            with self.assertRaises(ValueError):
                prepare.parse_weights(bad, 3)

    def base(self, hidden=3, layers=2):
        rows = [[0.6, 0.8, 0.0], [0.0, 0.0, 1.0]][:layers]
        return dict(schema='gufo-abliteration-v2', model_identity='m', hidden=hidden,
                    layers=layers, placement='writer-before-hc', directions=rows)

    def test_make_profile_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'base.json'
            prepare.save_profile(source, prepare.make_profile(
                self.base(), 0, [1, 1], [0, 0]))
            output = Path(tmp) / 'chosen.json'
            prepare.save_profile(output, prepare.make_profile(
                prepare.load_base(profile=source), 1,
                prepare.parse_weights('2', 2), prepare.parse_weights('0.5', 2)))
            profile = json.loads(output.read_text())
            self.assertEqual(profile['schema'], 'gufo-abliteration-v2')
            self.assertEqual(profile['direction_layer'], 1)
            self.assertEqual(profile['attention'], [2, 2])
            self.assertEqual(profile['mlp'], [0.5, 0.5])
            self.assertEqual(profile['directions'], self.base()['directions'])
            # Compact: no indentation inside the vectors.
            self.assertNotIn('\n  ', output.read_text())
        with self.assertRaises(ValueError):
            prepare.make_profile(self.base(), 2, [1, 1], [1, 1])
        broken = self.base()
        broken['directions'] = [[1, 1, 0], [0, 0, 1]]
        with self.assertRaises(ValueError):
            prepare.make_profile(broken, 0, [1, 1], [1, 1])

    def test_make_profile_residual(self):
        # Absent unless given: the runtime reads a missing member as zeros.
        plain = prepare.make_profile(self.base(), 0, [1, 1], [0, 0])
        self.assertNotIn('residual', plain)
        curve = prepare.parse_weights('max_w=1,peak=1,min_w=0.5,width=2', 2)
        self.assertEqual(curve, [0.75, 1.0])
        profile = prepare.make_profile(self.base(), 1, [0, 0], [0, 0], residual=curve)
        self.assertEqual(profile['residual'], [0.75, 1.0])
        for bad in ([1], [1, 4.5], [1, -1], [1, float('nan')]):
            with self.assertRaises(ValueError):
                prepare.make_profile(self.base(), 0, [0, 0], [0, 0], residual=bad)

    def test_make_profile_cli_residual(self):
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp) / 'base.json'
            prepare.save_profile(source, prepare.make_profile(
                self.base(), 0, [1, 1], [0, 0]))
            def make(output, *flags):
                argv = sys.argv
                sys.argv = ['prepare.py', 'make-profile', '--profile', str(source),
                            '--direction-layer', '1', *flags, '--output', str(output)]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        prepare.main()
                finally:
                    sys.argv = argv
                return json.loads(output.read_text())

            profile = make(Path(tmp) / 'residual.json', '--attention', '0',
                           '--residual', 'max_w=1,peak=1,min_w=0.5,width=2')
            self.assertEqual(profile['attention'], [0, 0])
            # Omitted strengths keep the base profile's defaults.
            self.assertEqual(profile['mlp'], [0, 0])
            self.assertEqual(profile['residual'], [0.75, 1.0])
            self.assertEqual(profile['direction_layer'], 1)
            # Re-defaulting from that profile keeps its residual curve.
            source = Path(tmp) / 'residual.json'
            again = make(Path(tmp) / 'again.json', '--mlp', '0.5')
            self.assertEqual(again['attention'], [0, 0])
            self.assertEqual(again['mlp'], [0.5, 0.5])
            self.assertEqual(again['residual'], [0.75, 1.0])
            # A base without residual keeps it absent (read as 0).
            source = Path(tmp) / 'base.json'
            plain = make(Path(tmp) / 'plain.json')
            self.assertEqual(plain['attention'], [1, 1])
            self.assertNotIn('residual', plain)

    def test_fit_residual_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare.fit(PreparationTest().fixture(root), root / 'candidates')
            residual = sorted((root / 'candidates').glob('layer*-residual1.json'))
            self.assertTrue(residual)
            for path in residual:
                candidate = json.loads(path.read_text())
                self.assertEqual(candidate['residual'], [1.0] * 4)
                self.assertEqual(candidate['attention'], [0.0] * 4)
                self.assertEqual(candidate['mlp'], [0.0] * 4)
            # Writer candidates keep their D14 shape: no residual member.
            for path in (root / 'candidates').glob('layer*-strength*.json'):
                self.assertNotIn('residual', json.loads(path.read_text()))

    def test_make_profile_from_fit(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = PreparationTest().fixture(root)
            prepare.fit(source, root / 'candidates')
            base = prepare.load_base(fit_dir=root / 'candidates')
            profile = prepare.make_profile(base, 3, prepare.parse_weights('1', 4),
                                           prepare.parse_weights('0', 4))
            self.assertEqual((profile['hidden'], profile['layers']), (8, 4))
            self.assertAlmostEqual(np.linalg.norm(profile['directions'][3]), 1., places=6)

    def test_float32_text_round_trips(self):
        for x in np.random.default_rng(0).standard_normal(1000).astype(np.float32):
            self.assertEqual(np.float32(prepare.float32_text(float(x))), x)



def planted_collection(root, hidden=16, layers=3, refusal=12, ordinary=10, seed=7):
    """A collection whose refusal prompts deviate from the ordinary mean along
    three planted orthogonal directions (u0 for all, u1/u2 per prompt), with
    a large shared ordinary mean that overlaps u0."""
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.standard_normal((hidden, hidden)))
    u0, u1, u2, shared = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    source = root / 'collection'
    source.mkdir()
    rows = []
    for group, count in (('refusal', refusal), ('ordinary', ordinary)):
        for i in range(count):
            values = np.empty((layers, hidden))
            for layer in range(layers):
                # The ordinary mean has a component along u0, so the mean
                # difference and its projected form differ.
                x = 6 * shared + 2 * u0 + 0.01 * rng.standard_normal(hidden)
                if group == 'refusal':
                    x = (x + (3 + layer) * u0 + rng.choice([-2.0, 2.0]) * u1 +
                         rng.choice([-1.0, 1.0]) * u2)
                values[layer] = x
            identity = f'{group}{i}'
            raw = values.astype('<f4').tobytes()
            (source / (identity + '.f32')).write_bytes(raw)
            prepare.save(source / (identity + '.json'),
                         dict(activation_sha256=prepare.sha(raw)))
            rows.append(dict(id=identity, group=group))
    prepare.save(source / 'manifest.json', dict(
        split='collection', hidden=hidden, layers=layers, records=rows,
        model_identity='model-fixture',
        readout='last-prompt-token-post-layer-fp32-v1'))
    return source, (u0, u1, u2)


def ordinary_unit(statistics):
    mean = statistics['ordinary']
    return mean / np.linalg.norm(mean, axis=1, keepdims=True)


class SubspaceTest(unittest.TestCase):
    def assert_orthonormal(self, basis):
        for block in basis:
            np.testing.assert_allclose(block @ block.T, np.eye(len(block)), atol=1e-9)

    def test_projected_direction_is_orthogonal_to_the_ordinary_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = planted_collection(root)
            prepare.fit(source, root / 'fit', rank=1)
            stats = np.load(root / 'fit' / 'statistics.npz')
            h = ordinary_unit(stats)
            r = stats['refusal'] - stats['ordinary']
            projected = stats['basis_projected'][:, 0]
            for layer in range(3):
                self.assertAlmostEqual(projected[layer] @ h[layer], 0, places=9)
                self.assertAlmostEqual(np.linalg.norm(projected[layer]), 1, places=9)
                expected = r[layer] - (r[layer] @ h[layer]) * h[layer]
                expected /= np.linalg.norm(expected)
                np.testing.assert_allclose(projected[layer], expected, atol=1e-9)
                # It is another direction than the mean difference here.
                self.assertLess(projected[layer] @ stats['basis_mean_diff'][layer, 0],
                                0.9999)
            summary = json.loads((root / 'fit' / 'SUMMARY.json').read_text())
            self.assertEqual(summary['direction_kinds'], ['mean_diff', 'projected'])
            self.assertEqual(summary['rank'], 1)
            candidate = json.loads(next((root / 'fit').glob('layer*-residual1.json'))
                                   .read_text())
            # New fits default to the projected kind, one direction.
            self.assertEqual(candidate['direction_kind'], 'projected')
            self.assertEqual(candidate['direction_count'], 1)
            self.assertIs(candidate['preserve_norm'], False)
            self.assertEqual(candidate['rank'], 1)
            self.assertNotIn('subspace_directions', candidate)
            rows = prepare.decode_rows(candidate['projected_directions'][2], 1, 16, 'p')
            np.testing.assert_allclose(rows[0], projected[2], atol=1e-6)

    def test_pca_subspace(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, (u0, u1, u2) = planted_collection(root)
            prepare.fit(source, root / 'fit', rank=4, method='pca')
            stats = np.load(root / 'fit' / 'statistics.npz')
            mean_diff, projected = stats['basis_mean_diff'], stats['basis_projected']
            self.assertEqual(mean_diff.shape, (3, 4, 16))
            self.assertEqual(projected.shape, (3, 4, 16))
            self.assert_orthonormal(mean_diff)
            self.assert_orthonormal(projected)
            r = stats['refusal'] - stats['ordinary']
            h = ordinary_unit(stats)
            for layer in range(3):
                # Row 0 is the kind's single direction.
                np.testing.assert_allclose(mean_diff[layer, 0],
                                           r[layer] / np.linalg.norm(r[layer]), atol=1e-9)
                np.testing.assert_allclose(projected[layer] @ h[layer], 0, atol=1e-9)
                # The planted per-prompt directions lie in the first three rows.
                for u in (u1, u2):
                    self.assertGreater(np.linalg.norm(mean_diff[layer, :3] @ u), 0.99)
            profile = json.loads((root / 'fit' / 'layer1-residual1.json').read_text())
            self.assertEqual((profile['rank'], profile['subspace_method']), (4, 'pca'))
            extra = prepare.decode_rows(profile['subspace_directions'][1], 3, 16, 's')
            np.testing.assert_allclose(np.array(extra), mean_diff[1, 1:], atol=1e-6)
            self.assertEqual(len(profile['projected_directions']), 3)

    def test_som_subspace_is_deterministic(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = planted_collection(root)
            prepare.fit(source, root / 'a', rank=3, method='som', seed=5)
            prepare.fit(source, root / 'b', rank=3, method='som', seed=5)
            a = np.load(root / 'a' / 'statistics.npz')
            b = np.load(root / 'b' / 'statistics.npz')
            for key in ('basis_mean_diff', 'basis_projected'):
                np.testing.assert_array_equal(a[key], b[key])
                self.assertEqual(a[key].shape[1], 3)
                self.assert_orthonormal(a[key])
            self.assertEqual((root / 'a' / 'layer1-residual1.json').read_bytes(),
                             (root / 'b' / 'layer1-residual1.json').read_bytes())
            h = ordinary_unit(a)
            r = a['refusal'] - a['ordinary']
            for layer in range(3):
                np.testing.assert_allclose(a['basis_mean_diff'][layer, 0],
                                           r[layer] / np.linalg.norm(r[layer]), atol=1e-9)
                np.testing.assert_allclose(a['basis_projected'][layer] @ h[layer], 0,
                                           atol=1e-9)
            summary = json.loads((root / 'a' / 'SUMMARY.json').read_text())
            self.assertEqual((summary['subspace_method'], summary['seed']), ('som', 5))

    def test_som_prototypes_and_alignment_order(self):
        rng = np.random.default_rng(0)
        reference = np.array([1.0, 0, 0, 0])
        deviations = np.concatenate([
            reference + 0.01 * rng.standard_normal((6, 4)),
            np.array([0, 1.0, 0, 0]) + 0.01 * rng.standard_normal((6, 4))])
        first = prepare.som_candidates(deviations, reference, 2, 3)
        self.assertEqual(len(first), 4)  # a 2 x 2 grid
        np.testing.assert_array_equal(
            np.array(first), np.array(prepare.som_candidates(deviations, reference, 2, 3)))
        cosines = [v @ reference / np.linalg.norm(v) for v in first]
        self.assertEqual(cosines, sorted(cosines, reverse=True))

    def test_orthonormalize(self):
        first = np.array([3.0, 4.0, 0, 0])
        exclude = np.array([0, 0, 0, 1.0])
        candidates = [np.array([3.0, 4.0, 0, 1]), np.array([1.0, 0, 0, 0]),
                      np.array([0, 0, 2.0, 0]), np.array([0, 0, 0, 5.0])]
        basis = prepare.orthonormalize(first, candidates, 4, exclude=exclude)
        # The parallel candidate and the excluded direction add nothing.
        self.assertEqual(len(basis), 3)
        np.testing.assert_allclose(basis @ basis.T, np.eye(3), atol=1e-12)
        np.testing.assert_allclose(basis @ exclude, 0, atol=1e-12)
        np.testing.assert_allclose(basis[0], first / 5)
        self.assertEqual(len(prepare.orthonormalize(first, candidates, 2)), 2)

    def test_dependent_data_reduces_rank(self):
        # The constant fixture spans one direction, along the ordinary mean:
        # rank 1 and no projected kind; profiles keep their D16 shape.
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            prepare.fit(PreparationTest().fixture(root), root / 'fit', rank=4)
            summary = json.loads((root / 'fit' / 'SUMMARY.json').read_text())
            self.assertEqual((summary['rank'], summary['requested_rank']), (1, 4))
            self.assertEqual(summary['direction_kinds'], ['mean_diff'])
            candidate = json.loads(next((root / 'fit').glob('layer*.json')).read_text())
            self.assertEqual(candidate['direction_kind'], 'mean_diff')
            for key in ('subspace_directions', 'projected_directions', 'subspace_method'):
                self.assertNotIn(key, candidate)

    def test_fit_arguments(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = planted_collection(root)
            for kwargs in (dict(rank=0), dict(rank=9), dict(method='kmeans')):
                with self.assertRaises(ValueError):
                    prepare.fit(source, root / 'fit', **kwargs)

    def test_rows_round_trip(self):
        values = np.random.default_rng(1).uniform(-1, 1, (2, 5)).astype(np.float32)
        text = prepare.encode_rows(values)
        rows = prepare.decode_rows(text, 2, 5, 'x')
        np.testing.assert_array_equal(np.array(rows, dtype=np.float32), values)
        for bad in (text[:-4], text + 'AAAA', '*' + text[1:], 42):
            with self.assertRaises(ValueError):
                prepare.decode_rows(bad, 2, 5, 'x')

    def test_make_profile_subspace_options_and_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source, _ = planted_collection(root)
            prepare.fit(source, root / 'fit', rank=3)

            def make(output, *flags, base=('--fit', str(root / 'fit'))):
                argv = sys.argv
                sys.argv = ['prepare.py', 'make-profile', *base,
                            '--direction-layer', '2', *flags, '--output', str(output)]
                try:
                    with contextlib.redirect_stdout(io.StringIO()):
                        prepare.main()
                finally:
                    sys.argv = argv
                return json.loads(output.read_text())

            chosen = make(root / 'chosen.json', '--residual', '1', '--direction-kind',
                          'mean_diff', '--directions', '2', '--preserve-norm', 'on')
            self.assertEqual((chosen['direction_kind'], chosen['direction_count'],
                              chosen['preserve_norm'], chosen['rank']),
                             ('mean_diff', 2, True, 3))
            # Re-defaulting keeps the stored rows byte for byte and the options
            # not given.
            again = make(root / 'again.json', '--directions', '3',
                         base=('--profile', str(root / 'chosen.json')))
            for key in ('subspace_directions', 'projected_directions', 'directions'):
                self.assertEqual(again[key], chosen[key])
            self.assertEqual((again['direction_kind'], again['direction_count'],
                              again['preserve_norm']), ('mean_diff', 3, True))
            # A fit's own defaults.
            plain = make(root / 'plain.json')
            self.assertEqual((plain['direction_kind'], plain['direction_count'],
                              plain['preserve_norm']), ('projected', 1, False))
            with self.assertRaises(ValueError):
                make(root / 'bad.json', '--directions', '4')

    def test_make_profile_validates_subspace(self):
        base = ProfileTest().base()
        e1 = [[-0.8, 0.6, 0.0], [1.0, 0.0, 0.0]]
        good = dict(base, rank=2, subspace_directions=[prepare.encode_rows(r) for r in e1],
                    projected_directions=[prepare.encode_rows([[0, 0, 1], [1, 0, 0]]),
                                          prepare.encode_rows([[0, 1, 0], [1, 0, 0]])])
        profile = prepare.make_profile(good, 0, [0, 0], [0, 0], direction_kind='projected',
                                       direction_count=2, preserve_norm=True)
        self.assertEqual((profile['rank'], profile['direction_kind'],
                          profile['direction_count'], profile['preserve_norm']),
                         (2, 'projected', 2, True))
        # Legacy bases stay legacy: no new members appear.
        legacy = prepare.make_profile(base, 0, [0, 0], [0, 0])
        for key in prepare.SUBSPACE_MEMBERS:
            self.assertNotIn(key, legacy)
        bad = [
            dict(good, rank=9), dict(good, rank=0), dict(good, rank=True),
            dict(base, rank=2),
            dict(base, subspace_directions=good['subspace_directions']),
            dict(good, subspace_directions=[prepare.encode_rows([0.6, 0.8, 0.0]),
                                            prepare.encode_rows([1.0, 0, 0])]),
            dict(good, subspace_directions=good['subspace_directions'][:1]),
            dict(good, projected_directions=[prepare.encode_rows([[0, 0, 1], [0, 0, 1]])] * 2),
            dict(good, subspace_method='kmeans'),
        ]
        for value in bad:
            with self.assertRaises(ValueError):
                prepare.make_profile(value, 0, [0, 0], [0, 0])
        for options in (dict(direction_kind='projected'), dict(direction_count=2),
                        dict(preserve_norm='yes'), dict(direction_kind='pca')):
            with self.assertRaises(ValueError):
                prepare.make_profile(base, 0, [0, 0], [0, 0], **options)
        with self.assertRaises(ValueError):
            prepare.make_profile(good, 0, [0, 0], [0, 0], direction_count=3)
        with self.assertRaises(ValueError):
            prepare.make_profile(base, 0, [0, 0], [0, 0], rank=2)

    def test_rank8_27b_profile_fits_the_runtime_cap(self):
        layers, hidden, rank = 64, 5120, prepare.MAX_RANK
        # Base64 of rank - 1 mean-difference and rank projected rows per
        # layer, plus the JSON-number directions (about 14 bytes per value).
        encoded = layers * math.ceil((2 * rank - 1) * hidden * 4 / 3) * 4
        self.assertLess(encoded + layers * hidden * 14, 48 << 20)

if __name__ == '__main__':
    unittest.main()
