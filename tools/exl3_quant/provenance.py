#!/usr/bin/env python3
"""Write PROVENANCE.json for an XQ EXL3 pack (XQ, 2026-10-30).

  python3 -I provenance.py --pack <pack dir> --source <HF BF16 dir> --cal <cal .safetensors> \
      --summary <run_convert summary> ... --tools-commit <gufo commit> [--image-provenance /opt/provenance]

Records sha256 and size of every source and output file, the calibration file and its manifest
(<cal>.json, copied in), the image's wheel list and pip freeze, every converter run (argv, env,
versions, wall time, peaks, copy_work) from the summaries, and the seed rule (ExLlamaV3 seeds each
tensor with its module index; no other randomness). Standard library only. Hashing reads every
file once (the BF16 source is 52 GB), so this runs inside a window.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree(directory: Path, skip: set[str] = frozenset()) -> list[dict]:
    return [{"path": str(p.relative_to(directory)), "size": p.stat().st_size, "sha256": sha256(p)}
            for p in sorted(directory.rglob("*")) if p.is_file() and p.name not in skip]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pack", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True)
    ap.add_argument("--cal", type=Path, required=True)
    ap.add_argument("--summary", type=Path, action="append", default=[])
    ap.add_argument("--tools-commit", required=True)
    ap.add_argument("--image-provenance", type=Path, default=Path("/opt/provenance"))
    ap.add_argument("--note", default="")
    a = ap.parse_args()
    t0 = time.monotonic()
    cal_manifest = a.cal.with_suffix(".json")
    if cal_manifest.exists():
        shutil.copyfile(cal_manifest, a.pack / "CALIBRATION.json")
    image = {}
    for name in ("wheels.sha256", "pip-freeze.txt"):
        p = a.image_provenance / name
        if p.exists():
            image[name] = p.read_text()
    runs = []
    for s in a.summary:
        x = json.loads(s.read_text())
        runs.append({k: x.get(k) for k in ("argv", "env", "versions", "sha256", "copy_work", "returncode", "wall_s",
                                           "peaks", "cgroup_memory_peak")} | {"summary": str(s)})
    rec = {
        "schema": "gufo.xq-exl3-provenance.v1",
        "note": a.note,
        "tool": {"converter": "exllamav3.conversion.convert_model (upstream v1.6.5 cdc785db release wheel)",
                 "gufo_tools_commit": a.tools_commit},
        "seed": "ExLlamaV3 quant_args seed = module index (convert_model.py make_quant_args); calibration rows fixed by the cal file",
        "source": {"dir": str(a.source), "files": tree(a.source)},
        "calibration": {"file": str(a.cal), "size": a.cal.stat().st_size, "sha256": sha256(a.cal)},
        "runs": runs,
        "total_wall_s": round(sum(r["wall_s"] or 0 for r in runs), 1),
        "max_tree_rss": max((r["peaks"] or {}).get("tree_rss", 0) for r in runs) if runs else None,
        "max_gtt": max((r["peaks"] or {}).get("gtt", 0) for r in runs) if runs else None,
        "image_provenance": image,
        "output": {"dir": str(a.pack), "files": tree(a.pack, {"PROVENANCE.json"})},
    }
    rec["hash_seconds"] = round(time.monotonic() - t0, 1)
    (a.pack / "PROVENANCE.json").write_text(json.dumps(rec, indent=1) + "\n")
    print(json.dumps({"files_source": len(rec["source"]["files"]), "files_output": len(rec["output"]["files"]),
                      "total_wall_s": rec["total_wall_s"], "hash_seconds": rec["hash_seconds"]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
