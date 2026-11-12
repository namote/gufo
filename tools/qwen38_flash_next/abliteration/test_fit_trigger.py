"""CPU contracts of fit_trigger.py: topic-grouped folds, the per-set
false-positive threshold and the profile copy."""
import base64
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

spec = importlib.util.spec_from_file_location(
    'fit_trigger', Path(__file__).with_name('fit_trigger.py'))
fit_trigger = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit_trigger)

READOUT = 'last-prompt-token-post-layer-hc-mean-fp32-v1'


def write_collection(root, rows, hidden=4, layers=3, model='model-fixture'):
    """rows: (id, group, vector at every layer, topic or None, thinking)."""
    root.mkdir(parents=True)
    records = []
    for identity, group, vector, topic, thinking in rows:
        record = dict(id=identity, group=group, split='collection',
                      thinking=thinking)
        if topic is not None:
            record['topic'] = topic
        records.append(record)
        raw = np.tile(np.asarray(vector, dtype='<f4'), (layers, 1)).tobytes()
        (root / (identity + '.f32')).write_bytes(raw)
        (root / (identity + '.json')).write_text(json.dumps(
            dict(id=identity, activation_sha256=fit_trigger.sha(raw))))
    (root / 'manifest.json').write_text(json.dumps(dict(
        split='collection', readout=READOUT, hidden=hidden, layers=layers,
        model_identity=model, records=records)))
    return root


def separable(root, topics=8, per_topic=3, noise=0.05, seed=1, **kwargs):
    rng = np.random.default_rng(seed)
    rows = []
    for group, sign in (('refusal', 1.0), ('ordinary', -1.0)):
        for t in range(topics):
            for i in range(per_topic):
                vector = np.array([sign, 0.2 * t, 0, 0]) + rng.normal(0, noise, 4)
                rows.append((f'{group}{t}x{i}-plain', group, vector,
                             f'{group}{t}', i % 2 == 1))
    return write_collection(root, rows, **kwargs)


class FitTriggerTest(unittest.TestCase):
    def test_topics_never_split_across_folds(self):
        keys = np.array(['a', 'a', 'b', 'b', 'c', 'c', 'd', 'e', 'f'])
        fold = fit_trigger.assign_folds(keys, {'a', 'b', 'c'}, 3, 0)
        for key in set(keys.tolist()):
            self.assertEqual(len(set(fold[keys == key].tolist())), 1)
        # Each fold holds one target topic.
        self.assertEqual(sorted(fold[[0, 2, 4]].tolist()), [0, 1, 2])

    def test_default_topic_pairs_thinking_modes(self):
        self.assertEqual(fit_trigger.topic_of({'id': 'abc-think'}, {}), 'abc')
        self.assertEqual(fit_trigger.topic_of({'id': 'abc-plain'}, {}), 'abc')
        self.assertEqual(fit_trigger.topic_of({'id': 'abc-plain', 'topic': 't'}, {}), 't')
        self.assertEqual(fit_trigger.topic_of({'id': 'abc-plain'}, {'abc-plain': 'm'}), 'm')

    def test_threshold_bounds_every_set(self):
        a = np.arange(100.0)          # 1% of 100: one value may exceed
        b = np.arange(10.0) + 500     # 1% of 10: none may exceed
        t = fit_trigger.threshold_for([a], 0.01)
        self.assertEqual(t, 98.0)
        self.assertLessEqual(np.mean(a > t), 0.01)
        t = fit_trigger.threshold_for([a, b], 0.01)
        self.assertEqual(t, 509.0)

    def test_auc_counts_ties_half(self):
        self.assertEqual(fit_trigger.auc([1, 2], [0, 0]), 1.0)
        self.assertEqual(fit_trigger.auc([1], [1]), 0.5)

    def test_fit_separable_collection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = separable(root / 'c')
            harmless = write_collection(root / 'h', [
                (f'h{i}-plain', 'ordinary', [-1, 0, 0.1 * i, 0], None, False)
                for i in range(12)])
            final, report, manifest = fit_trigger.fit(
                collection, 'refusal', 1, [str(harmless)], folds=4)
            self.assertAlmostEqual(float(np.linalg.norm(final)), 1.0, places=5)
            self.assertGreater(final[0], 0.99)
            self.assertEqual(report['auc_ordinary'], 1.0)
            self.assertEqual(report['recall'], 1.0)
            self.assertEqual(report['false_positive_all'], 0.0)
            self.assertEqual(report['counts']['extra_negatives'], 12)
            self.assertEqual(report['counts']['target_topics'], 8)

    def test_rejects_topic_with_both_groups(self):
        with tempfile.TemporaryDirectory() as tmp:
            rows = [(f'r{i}', 'refusal', [1, 0, 0, 0], 'same', False)
                    for i in range(3)]
            rows += [(f'o{i}', 'ordinary', [-1, 0, 0, 0], 'same', False)
                     for i in range(3)]
            collection = write_collection(Path(tmp) / 'c', rows)
            with self.assertRaises(ValueError):
                fit_trigger.fit(collection, 'refusal', 0, folds=2)

    def test_profile_copy_keeps_source(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            collection = separable(root / 'c')
            final, report, manifest = fit_trigger.fit(
                collection, 'refusal', 2, folds=4)
            source = root / 'profile.json'
            text = json.dumps(dict(schema='gufo-abliteration-v2',
                                   model_identity='model-fixture', hidden=4,
                                   layers=3, placement='writer-before-hc'))
            source.write_text(text)
            output = root / 'out' / 'profile.json'
            fit_trigger.write_profile(source, output, final, report, manifest)
            self.assertEqual(source.read_text(), text)
            trigger = json.loads(output.read_text())['trigger']
            self.assertEqual(trigger['layer'], 2)
            self.assertEqual(trigger['readout'], READOUT)
            decoded = np.frombuffer(base64.b64decode(trigger['direction']), '<f4')
            np.testing.assert_array_equal(decoded, final)
            self.assertAlmostEqual(trigger['threshold'], report['threshold'],
                                   places=5)
            with self.assertRaises(SystemExit):
                fit_trigger.write_profile(source, output, final, report, manifest)
            other = root / 'other.json'
            other.write_text(text.replace('model-fixture', 'another'))
            with self.assertRaises(SystemExit):
                fit_trigger.write_profile(other, root / 'x.json', final, report,
                                          manifest)


if __name__ == '__main__':
    unittest.main()
