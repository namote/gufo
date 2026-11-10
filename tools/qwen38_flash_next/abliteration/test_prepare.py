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
                self.assertAlmostEqual(np.linalg.norm(candidate['direction']), 1., places=6)
                self.assertEqual(candidate['model_identity'], 'model-fixture')
                self.assertEqual(candidate['status'], 'candidate-not-qualified')
                self.assertEqual(len(candidate['attention']), 4)
                self.assertEqual(candidate['placement'], 'writer-before-hc')
            self.assertFalse((root / 'default.json').exists())

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
                self.assertEqual(len(candidate['direction']), 6)
                self.assertEqual(len(candidate['mlp']), 9)
                self.assertAlmostEqual(np.linalg.norm(candidate['direction']), 1., places=6)

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


if __name__ == '__main__':
    unittest.main()
