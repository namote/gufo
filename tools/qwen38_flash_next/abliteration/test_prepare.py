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
    def fixture(self, root, split='collection'):
        source = root / 'collection'
        source.mkdir()
        rows = []
        for group, sign in [('refusal', 1), ('ordinary', -1)]:
            for i in range(2):
                identity = f'{group}{i}'
                rows.append(dict(id=identity, group=group))
                raw = np.full((4, 8), sign * (i + 1), dtype='<f4').tobytes()
                (source / (identity + '.f32')).write_bytes(raw)
                prepare.save(source / (identity + '.json'), dict(activation_sha256=prepare.sha(raw)))
        prepare.save(source / 'manifest.json', dict(split=split, hidden=8, layers=4,
                     records=rows, model_identity='model-fixture'))
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
                self.assertAlmostEqual(np.linalg.norm(candidate['direction']), 1., places=6)
                self.assertEqual(candidate['model_identity'], 'model-fixture')
                self.assertEqual(candidate['status'], 'candidate-not-qualified')
                self.assertEqual(len(candidate['attention']), 4)
            self.assertFalse((root / 'default.json').exists())

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


if __name__ == '__main__':
    unittest.main()
