"""CPU-only checks for saved activation coverage and phase separation."""
import json
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np

from .prepare_stats import read_windows


class CaptureRows(unittest.TestCase):
    def fixture(self, root):
        directory = root / 'decode/capture/0'; directory.mkdir(parents=True)
        chunks = []
        for projection, cols in (('down', 10240), ('up', 320)):
            for pos, rows, phase in ((0, 32, 'prefill'), (32, 1, 'decode'), (33, 1, 'decode')):
                name = f'{projection}-{pos}.f32'
                np.full((rows, cols), pos + 1, dtype='<f4').tofile(directory / name)
                chunks.append(dict(projection=projection, position=pos, rows=rows, cols=cols, phase=phase, file=name))
        (directory / 'chunks.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in chunks))
        return directory, chunks, {'rows': [{'window': 0, 'id': 'doc', 'length': 35}]}

    def test_decode_excludes_prefix_and_rejects_missing_duplicate_rows(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp); directory, chunks, manifest = self.fixture(root)
            values, records = read_windows(root, manifest, 'decode', 'up')
            self.assertEqual(values['doc'].shape, (2, 320))
            np.testing.assert_array_equal(values['doc'][:, 0], [33, 34])
            self.assertEqual(sum(r['rows'] for r in records), 34)
            for broken in (chunks[:-1], chunks + [chunks[-1]]):
                (directory / 'chunks.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in broken))
                with self.assertRaises(ValueError): read_windows(root, manifest, 'decode', 'up')

    def test_payload_finite_size_and_path_guards(self):
        with tempfile.TemporaryDirectory(dir=os.environ['TMPDIR']) as tmp:
            root = Path(tmp); directory, chunks, manifest = self.fixture(root)
            p = directory / 'up-33.f32'; original = p.read_bytes()
            for payload in (b'bad', np.full((1, 320), np.nan, dtype='<f4').tobytes()):
                p.write_bytes(payload)
                with self.assertRaises(ValueError): read_windows(root, manifest, 'decode', 'up')
            p.write_bytes(original)
            chunks[-1]['file'] = '../up-33.f32'
            (directory / 'chunks.jsonl').write_text(''.join(json.dumps(c) + '\n' for c in chunks))
            with self.assertRaises(ValueError): read_windows(root, manifest, 'decode', 'up')


if __name__ == '__main__':
    unittest.main()
