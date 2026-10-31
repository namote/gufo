"""Header-only EXL3 artifact identity and incomplete-download checks."""
import hashlib
import importlib.util
import json
from pathlib import Path
import struct
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[3]
spec = importlib.util.spec_from_file_location(
    "fn_metadata", ROOT / "tools/exl3_quant/fn_metadata.py")
metadata = importlib.util.module_from_spec(spec)
spec.loader.exec_module(metadata)


class HeadersTest(unittest.TestCase):
    def test_identity_and_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.safetensors"
            header = json.dumps({"x": {"dtype": "F16", "shape": [128],
                                      "data_offsets": [0, 256]}}).encode()
            prefix = struct.pack("<Q", len(header))
            path.write_bytes(prefix + header + bytes(256))
            expected = hashlib.sha256(
                path.name.encode() + b"\0" + str(path.stat().st_size).encode()
                + b"\0" + prefix + header).hexdigest()
            self.assertEqual(metadata.headers_digest([path]), expected)
            # Identity deliberately reads headers, never the full payload.
            path.write_bytes(prefix + header + b"X" * 256)
            self.assertEqual(metadata.headers_digest([path]), expected)
            path.write_bytes(prefix + header + bytes(255))
            with self.assertRaisesRegex(ValueError, "truncated tensor"):
                metadata.headers_digest([path])

    def test_missing_or_truncated_header(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "missing.safetensors"
            with self.assertRaises(FileNotFoundError):
                metadata.headers_digest([path])
            path.write_bytes(struct.pack("<Q", 100) + b"{}")
            with self.assertRaisesRegex(ValueError, "truncated header"):
                metadata.headers_digest([path])
            path.write_bytes(struct.pack("<Q", 1 << 30))
            with self.assertRaisesRegex(ValueError, "invalid header size"):
                metadata.headers_digest([path])

    def test_supplementary_is_part_of_identity(self):
        with tempfile.TemporaryDirectory() as directory:
            paths = [Path(directory) / name for name in ("a.safetensors", "patch.safetensors")]
            for path in paths:
                path.write_bytes(struct.pack("<Q", 2) + b"{}")
            self.assertNotEqual(metadata.headers_digest(paths[:1]), metadata.headers_digest(paths))
            self.assertEqual(metadata.headers_digest(paths), metadata.headers_digest(paths[::-1]))


if __name__ == "__main__":
    unittest.main()
