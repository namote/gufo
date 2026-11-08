#!/usr/bin/env python3
"""The retained c03 inputs (recipe, calibration token ids) and the driver path config (standard library only)."""
from __future__ import annotations

import collections
import hashlib
import json
import os
import struct
import sys
import unittest
from pathlib import Path
from unittest import mock

FN = Path(__file__).resolve().parent / "flash_next"
sys.path.insert(0, str(FN))

RECIPE_SHA = "83bf62ada9550d0e50d9a913aecc6e4e2224eb567ecda72e08af301131b369c7"
C03_SHA = "36c8eddb18c51c3b9788d73195aa72527d9b9d48f818b79a6f0ab251acfebf3d"
C03_JSON_SHA = "d4fc88ae217bc91c540f669999caa85db2e00593a2d3579cb5884ee18aeba1e5"
C02_SHA = "b1d6e42852f6d821d3fa3df906594e88a69855de9288de83abd9b8c9a86805d5"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def header(data: bytes) -> tuple[dict, int]:
    n = struct.unpack("<Q", data[:8])[0]
    return json.loads(data[8:8 + n]), 8 + n


class Inputs(unittest.TestCase):
    def test_recipe(self):
        path = FN / "b1-recipe.json"
        self.assertEqual(sha256(path), RECIPE_SHA)
        recipe = json.loads(path.read_text())
        self.assertEqual(recipe["head_bits"], 8)
        self.assertAlmostEqual(recipe["target_bpw"], 4.03597648, places=7)
        counts = collections.Counter(recipe["tensors"].values())
        self.assertEqual(dict(counts), {4: 73728, 5: 72, 6: 204, 8: 36})

    def test_c03_calibration(self):
        st, js = FN / "c03-input-ids.safetensors", FN / "c03-input-ids.json"
        self.assertEqual(sha256(js), C03_JSON_SHA)
        data = st.read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), C03_SHA)
        manifest = json.loads(js.read_text())
        self.assertEqual(manifest["format"], "gufo.ragged-calibration.v1")
        self.assertEqual(manifest["sha256"], C03_SHA)
        self.assertEqual(manifest["c02_sha256"], C02_SHA)
        self.assertEqual(sha256(FN / "build_long_calibration.py"), manifest["builder_sha256"])
        seqs = manifest["sequences"]
        self.assertEqual(len(seqs), manifest["rows"])
        self.assertEqual(len(seqs), 2309)
        self.assertEqual(sum(s["tokens"] for s in seqs), manifest["total_tokens"])
        self.assertEqual(manifest["total_tokens"], 1999496)
        self.assertLessEqual(manifest["total_tokens"], manifest["budget"])
        self.assertLessEqual(max(s["tokens"] for s in seqs), manifest["max_columns"])
        self.assertEqual(sum(manifest["source_tokens"].values()), manifest["total_tokens"])
        head, base = header(data)
        tensors = {k: v for k, v in head.items() if k != "__metadata__"}
        self.assertEqual(set(tensors), {f"input_ids.{i}" for i in range(len(seqs))})
        for i, s in enumerate(seqs):
            t = tensors[f"input_ids.{i}"]
            self.assertEqual((t["dtype"], t["shape"]), ("I32", [1, s["tokens"]]))
            a, b = t["data_offsets"]
            self.assertEqual(hashlib.sha256(data[base + a:base + b]).hexdigest(), s["token_sha256"], i)

    def test_c02_calibration(self):
        data = (FN / "c02-input-ids.safetensors").read_bytes()
        self.assertEqual(hashlib.sha256(data).hexdigest(), C02_SHA)
        head, _ = header(data)
        self.assertEqual(head["input_ids"]["dtype"], "I32")
        self.assertEqual(head["input_ids"]["shape"], [3906, 512])


# Pinned files that later gufo commits edited (paths and host guard); their
# receipt bytes are in the first import commit.
EDITED = {"model_job.py", "mtp_calibration_job.py", "ngram_job.py", "prefix_replay_job.py",
          "run_stage.py", "test_disk_boundary.py"}
NOT_TRACKED = {"native/capture-gram.so"}  # built by build_capture_gram.py


class Pinned(unittest.TestCase):
    def test_receipt_pinned_files_are_production(self):
        files = json.loads((FN / "PINNED-FILES.json").read_text())["files"]
        self.assertGreaterEqual(len(files), 80)
        for rel, entry in files.items():
            if rel in NOT_TRACKED:
                continue
            path = FN / rel
            self.assertTrue(path.is_file(), rel)
            if rel not in EDITED:
                self.assertEqual(sha256(path), entry["sha256"], rel)

    def test_edited_set_is_current(self):
        files = json.loads((FN / "PINNED-FILES.json").read_text())["files"]
        for rel in EDITED:
            self.assertNotEqual(sha256(FN / rel), files[rel]["sha256"], rel)


class PathConfig(unittest.TestCase):
    def fresh(self, env: dict):
        for name in ("paths", "host_guard"):
            sys.modules.pop(name, None)
        with mock.patch.dict(os.environ, env, clear=False):
            for k in ("GUFO_QUANT_OUT", "GUFO_KYOJIN_RUNTIME", "GUFO_FN_SOURCE", "GUFO_OPS"):
                if k not in env:
                    os.environ.pop(k, None)
            import paths
            import host_guard
        return paths, host_guard

    def test_defaults_stay_in_repository(self):
        paths, guard = self.fresh({})
        repo = Path(__file__).resolve().parents[2]
        self.assertEqual(paths.REPO, repo)
        self.assertEqual(paths.TASK, FN)
        self.assertEqual(paths.OUT, repo / "build/flash-next-quant-c03")
        self.assertEqual(paths.RUNTIME, repo / "build/kyojin-runtime")
        self.assertEqual(guard.LOCK, Path.home() / ".local/share/gufo/ops/OWNER.lock")
        self.assertNotIn("ops", sys.modules)  # imported only when a driver checks the host

    def test_environment_overrides(self):
        paths, guard = self.fresh({"GUFO_QUANT_OUT": "/o", "GUFO_KYOJIN_RUNTIME": "/r",
                                   "GUFO_FN_SOURCE": "/s", "GUFO_OPS": "/x/ops"})
        self.assertEqual((paths.OUT, paths.RUNTIME, paths.SOURCE), (Path("/o"), Path("/r"), Path("/s")))
        self.assertEqual(guard.LOCK, Path("/x/ops/OWNER.lock"))

    def test_ai_tools_imports_only_in_verbatim_probes(self):
        # Probe drivers that the receipts pin (directly or through a pinned test) stay
        # byte-identical, so they keep their ai.tools imports; no driver of the
        # production stages imports ai.tools.
        verbatim = {"capture_probe_job.py", "capture_lean_job.py", "capture_static_job.py",
                    "state_io_job.py", "model_job_lean_candidate.py", "model_job_static_candidate.py"}
        pinned = json.loads((FN / "PINNED-FILES.json").read_text())["files"]
        found = {p.name for p in FN.glob("*.py")
                 if "from test.experiments" in p.read_text() or "import test.experiments" in p.read_text()}
        self.assertEqual(found, verbatim)
        for name in verbatim & pinned.keys():
            self.assertEqual(sha256(FN / name), pinned[name]["sha256"], name)

    def tearDown(self):
        for name in ("paths", "host_guard"):
            sys.modules.pop(name, None)


if __name__ == "__main__":
    unittest.main()
