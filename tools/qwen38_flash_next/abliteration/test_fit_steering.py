"""CPU contracts of fit_steering.py: paired directions, kinds and strengths."""
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
spec = importlib.util.spec_from_file_location(
    'fit_steering', Path(__file__).with_name('fit_steering.py'))
fit_steering = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit_steering)

HIDDEN, LAYERS = 4, 2


def write(root, items):
    """items: (id, group, pair, [layers][hidden])."""
    root.mkdir(parents=True)
    records = []
    for identity, group, pair, values in items:
        raw = np.asarray(values, dtype='<f4').tobytes()
        (root / f'{identity}.f32').write_bytes(raw)
        (root / f'{identity}.json').write_text(json.dumps(dict(
            id=identity, activation_sha256=hashlib.sha256(raw).hexdigest())))
        records.append(dict(id=identity, group=group, pair=pair))
    (root / 'manifest.json').write_text(json.dumps(dict(
        readout=fit_steering.READOUT, hidden=HIDDEN, layers=LAYERS,
        model_identity='m', split='collection', records=records)))
    return root


def collection(root, shift):
    rng = np.random.default_rng(0)
    items = []
    for q in range(5):
        topic = rng.normal(0, 3, (LAYERS, HIDDEN)) + np.array([0, 0, 5, 0])
        own = topic + rng.normal(0, 0.01, (LAYERS, HIDDEN))
        items.append((f'{q:02x}a', 'own', f'q{q}', own))
        items.append((f'{q:02x}b', 'exemplar', f'q{q}', own + shift))
    items.append(('ffa', 'own', 'lonely', np.zeros((LAYERS, HIDDEN))))
    return write(root, items)


class FitSteeringTest(unittest.TestCase):
    def test_paired_mean_difference(self):
        with tempfile.TemporaryDirectory() as tmp:
            shift = np.array([[2.0, 0, 0, 0], [0, -1.0, 0, 0]])
            profile = fit_steering.fit(collection(Path(tmp) / 'c', shift),
                                       alpha=0.5)
            self.assertEqual(profile['mode'], 'add')
            self.assertEqual(profile['steering_fit']['pairs'], 5)  # lonely skipped
            d = np.array(profile['directions'])
            self.assertAlmostEqual(abs(d[0, 0]), 1.0, places=2)
            self.assertAlmostEqual(abs(d[1, 1]), 1.0, places=2)
            # Signs: the direction points from own to exemplar.
            self.assertGreater(d[0, 0], 0)
            self.assertLess(d[1, 1], 0)
            np.testing.assert_allclose(profile['magnitudes'], [2.0, 1.0], rtol=1e-3)
            np.testing.assert_allclose(profile['residual'], [1.0, 0.5], rtol=1e-3)
            self.assertEqual(profile['attention'], [0.0, 0.0])
            self.assertEqual(profile['direction_count'], 1)
            self.assertFalse(profile['preserve_norm'])

    def test_projected_removes_the_shared_mean(self):
        with tempfile.TemporaryDirectory() as tmp:
            # The shift has a component along the own mean (axis 2).
            shift = np.array([[1.0, 0, 1.0, 0], [1.0, 0, 1.0, 0]])
            profile = fit_steering.fit(collection(Path(tmp) / 'c', shift),
                                       kind='projected', alpha=1.0)
            self.assertEqual(profile['direction_kind'], 'projected')
            self.assertIn('projected_directions', profile)
            # |d| without the shared axis is about 1.
            self.assertLess(profile['magnitudes'][0], 1.2)

    def test_clamps_strengths_and_refuses_too_few_pairs(self):
        with tempfile.TemporaryDirectory() as tmp:
            shift = np.full((LAYERS, HIDDEN), 100.0)
            profile = fit_steering.fit(collection(Path(tmp) / 'c', shift), alpha=1)
            self.assertEqual(max(profile['residual']), 64.0)
            one = write(Path(tmp) / 'one', [
                ('aa', 'own', 'q', np.zeros((LAYERS, HIDDEN))),
                ('ab', 'exemplar', 'q', np.ones((LAYERS, HIDDEN)))])
            with self.assertRaises(ValueError):
                fit_steering.fit(one)


if __name__ == '__main__':
    unittest.main()
