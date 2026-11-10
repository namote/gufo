"""CPU contracts for saved collection integrity and train/evaluation separation."""
import importlib.util
import json
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


if __name__ == '__main__':
    unittest.main()
