#!/usr/bin/env python3
"""Unit test of the Kyojin pin (kyojin/pin.py) on synthetic archives (standard library only)."""
from __future__ import annotations

import hashlib
import io
import json
import sys
import tarfile
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "kyojin"))

import pin  # noqa: E402

PREFIX = "kyojin-abc"


def make_archive(path: Path, files: dict[str, bytes], links: dict[str, str] = {}) -> str:
    with tarfile.open(path, "w:gz") as tar:
        d = tarfile.TarInfo(PREFIX)
        d.type = tarfile.DIRTYPE
        tar.addfile(d)
        for name, data in files.items():
            info = tarfile.TarInfo(f"{PREFIX}/{name}")
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
        for name, target in links.items():
            info = tarfile.TarInfo(f"{PREFIX}/{name}")
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tar.addfile(info)
    return hashlib.sha256(path.read_bytes()).hexdigest()


class Test(unittest.TestCase):
    def test_recorded_pin(self):
        p = pin.pin()
        self.assertEqual(p["revision"], "fbde8a3670c80727a202da3bce581da2de7c88da")
        self.assertEqual(p["sha256"], "dae498bd6bb2767dd06e46a94fa1dafa9b33be9362d0c8bbce0ae1c2b99f7194")
        self.assertEqual(p["url"], "https://github.com/Yamz-Labs/kyojin/archive/"
                                   "fbde8a3670c80727a202da3bce581da2de7c88da.tar.gz")
        self.assertEqual(p["prefix"], "kyojin-fbde8a3670c80727a202da3bce581da2de7c88da")
        inputs = json.loads(pin.INPUTS_JSON.read_text())
        self.assertEqual(inputs["source_revision"], p["revision"])
        self.assertIn("source/LICENSE", inputs["files"])

    def test_fetch_backup_unpack_check(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            files = {"LICENSE": b"MIT", "a/b.py": b"x = 1\n", "a/c.py": b"y\n"}
            digest = make_archive(t / "backup.tar.gz", files, {"a/link.py": "b.py"})
            origin = pin.fetch(t / "out.tar.gz", digest, "http://unused.invalid/", t / "backup.tar.gz")
            self.assertEqual(origin, str(t / "backup.tar.gz"))
            with self.assertRaises(FileExistsError):
                pin.fetch(t / "out.tar.gz", digest, "http://unused.invalid/", t / "backup.tar.gz")
            self.assertEqual(pin.unpack(t / "out.tar.gz", t / "src", PREFIX), 3)
            self.assertEqual((t / "src/a/link.py").read_bytes(), b"x = 1\n")
            record = {"source/" + k: hashlib.sha256(v).hexdigest() for k, v in files.items()}
            record["source/a/link.py"] = record["source/a/b.py"]
            record["source/exllamav3_ext.cpython-312-x86_64-linux-gnu.so"] = "0" * 64
            record["site/x/METADATA"] = "0" * 64
            (t / "inputs.json").write_text(json.dumps({"files": record}))
            self.assertEqual(pin.check(t / "src", t / "inputs.json"), [])
            (t / "src/a/c.py").write_bytes(b"changed\n")
            (t / "src/LICENSE").unlink()
            self.assertEqual(pin.check(t / "src", t / "inputs.json"), ["missing LICENSE", "differs a/c.py"])
            with self.assertRaises(FileExistsError):
                pin.unpack(t / "out.tar.gz", t / "src", PREFIX)

    def test_bad_backup_falls_back_and_mismatch_fails(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            good = make_archive(t / "good.tar.gz", {"f": b"1"})
            make_archive(t / "bad.tar.gz", {"f": b"2"})
            with self.assertRaises(ValueError):
                pin.fetch(t / "out.tar.gz", good, (t / "bad.tar.gz").as_uri(), t / "bad.tar.gz")
            self.assertFalse((t / "out.tar.gz").exists())
            self.assertFalse((t / "out.tar.gz.part").exists())
            self.assertEqual(pin.fetch(t / "out.tar.gz", good, (t / "good.tar.gz").as_uri(), t / "bad.tar.gz"),
                             (t / "good.tar.gz").as_uri())

    def test_unsafe_members_rejected(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            for files, links in (({"../evil": b"x"}, {}), ({}, {"a/l": "../../etc/passwd"}),
                                 ({}, {"l": "/etc/passwd"})):
                make_archive(t / "x.tar.gz", files, links)
                with tarfile.open(t / "x.tar.gz") as tar, self.assertRaises(ValueError):
                    list(pin.safe_members(tar, PREFIX))
            make_archive(t / "y.tar.gz", {"f": b"1"})
            with tarfile.open(t / "y.tar.gz") as tar, self.assertRaises(ValueError):
                list(pin.safe_members(tar, "other-prefix"))

    def test_chained_symlinks_cannot_escape(self):
        cases = {
            "parent link": ({"d/l/evil": b"x"}, {"d": ".", "d/l": ".."}),
            "chain": ({"x/f": b"1"}, {"x/b": "..", "a": "x/b/.."}),
            "later link": ({"x/f": b"1"}, {"x/b": "..", "a": "y/..", "y": "x/b"}),
        }
        for name, (files, links) in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as t:
                t = Path(t)
                make_archive(t / "a.tar.gz", files, links)
                with self.assertRaises((ValueError, OSError)):
                    pin.unpack(t / "a.tar.gz", t / "dest", PREFIX)
                self.assertEqual(sorted(p.name for p in t.iterdir()), ["a.tar.gz", "dest"])

    def test_failed_download_leaves_no_partial_file(self):
        with tempfile.TemporaryDirectory() as t:
            t = Path(t)
            with self.assertRaises(OSError):
                pin.fetch(t / "out.tar.gz", "0" * 64, (t / "absent.tar.gz").as_uri(), None)
            self.assertEqual(list(t.iterdir()), [])

    def test_serve_patch_targets_only_serving_files(self):
        text = (pin.PATCHES / "qwen-serve-max-context.patch").read_text()
        targets = sorted(line[6:] for line in text.splitlines() if line.startswith("+++ b/"))
        self.assertEqual(targets, ["tools/qwen/serve.py", "tools/qwen/test_serve.py"])
        self.assertIn("--max-context", text)


if __name__ == "__main__":
    unittest.main()
