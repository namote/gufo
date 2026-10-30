#!/usr/bin/env python3
"""Unit test of c02_to_caldata on a synthetic corpus (standard library only)."""
from __future__ import annotations

import hashlib
import json
import struct
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import c02_to_caldata as cd  # noqa: E402


def make_corpus(d: Path, chunk: int = 4) -> None:
    spec = [("a", "", 9), ("b", "", 5), ("a", "x", 6)]
    order = [(s, p, i) for s, p, n in spec for i in range(n)]
    order.reverse()
    chunks, tokens = [], []
    for s, p, i in order:
        base = {"a": 1000, "b": 2000}[s] + (500 if p else 0) + i * chunk
        chunks.append({"id": f"{s}-{p}{i}", "stream": s, "offset": len(tokens), "length": chunk})
        tokens.extend(range(base, base + chunk))
    payload = struct.pack(f"<{len(tokens)}i", *tokens)
    index = json.dumps({"chunk_tokens": chunk, "chunks": chunks,
                        "tokens_sha256": hashlib.sha256(payload).hexdigest()}).encode()
    (d / "chunks.bin").write_bytes(payload)
    (d / "chunks.json").write_bytes(index)
    (d / "MANIFEST.json").write_text(json.dumps({
        "corpus_version": "test", "tokenizer": {"sha256": "-"},
        "outputs": {"chunks.bin": hashlib.sha256(payload).hexdigest(),
                    "chunks.json": hashlib.sha256(index).hexdigest()}}))


class Test(unittest.TestCase):
    def test_rows(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            make_corpus(d)
            rows, meta = cd.build(d, rows=4, cols=8, seed=1)
            # segment rows: a/base 4 (9 chunks), b/base 2, a/ext 3 -> 9 rows, 4 allocated
            self.assertEqual(meta["segment_rows_available"], {"a/base": 4, "a/ext": 3, "b/base": 2})
            self.assertEqual(sum(meta["allocation"].values()), 4)
            for r, info in zip(rows, meta["row_list"]):
                # every row is two consecutive chunks of one segment, in stream order
                self.assertEqual(r, list(range(r[0], r[0] + 8)))
                s, g = info["segment"].split("/")
                first = int(info["chunks"][0].rsplit("-", 1)[1].lstrip("x"))
                self.assertEqual(first % 2, 0)
                self.assertEqual(r[0], {"a": 1000, "b": 2000}[s] + (500 if g == "ext" else 0) + first * 4)
            again, _ = cd.build(d, rows=4, cols=8, seed=1)
            self.assertEqual(rows, again)
            out = d / "o.safetensors"
            blob = cd.write_safetensors(out, rows)
            self.assertEqual(struct.unpack("<Q", blob[:8])[0] % 8, 0)
            self.assertEqual(cd.read_safetensors_ids(out.read_bytes()), rows)

    def test_allocate(self):
        self.assertEqual(cd.allocate({"x": 145, "y": 345, "z": 23}, 100), {"x": 28, "y": 67, "z": 5})

    def test_bad_sha(self):
        with tempfile.TemporaryDirectory() as t:
            d = Path(t)
            make_corpus(d)
            (d / "chunks.bin").write_bytes(b"\0" * 8)
            with self.assertRaises(ValueError):
                cd.build(d, rows=1, cols=8, seed=1)


if __name__ == "__main__":
    unittest.main()
