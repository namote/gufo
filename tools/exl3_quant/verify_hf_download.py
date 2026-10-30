#!/usr/bin/env python3
"""Verify a pinned Hugging Face download against the repository tree (XQ, 2026-10-31).

  nice -n 19 ionice -c3 python3 -I verify_hf_download.py <tree.json> <dir> --log <log>

<tree.json> is the HF API listing `api/models/<repo>/tree/<commit>?recursive=true`. LFS files are
hashed with sha256 and compared with their `lfs.oid`; small (non-LFS) files are compared by size and
by their git blob sha1 (`oid`, sha1 of "blob <size>\\0" + content). Files already verified in the log
(same size and an OK line) are skipped, so an interrupted run resumes. Exit 0 only when every file
passes. Standard library only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def digest(path: Path, algo: str, prefix: bytes = b"") -> str:
    h = hashlib.new(algo)
    h.update(prefix)
    with path.open("rb") as f:
        for block in iter(lambda: f.read(16 << 20), b""):
            h.update(block)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("tree", type=Path)
    ap.add_argument("dir", type=Path)
    ap.add_argument("--log", type=Path, required=True)
    a = ap.parse_args()
    files = [x for x in json.loads(a.tree.read_text()) if x["type"] == "file"]
    done = set()
    if a.log.exists():
        for line in a.log.read_text().splitlines():
            parts = line.split()
            if len(parts) >= 3 and parts[2] == "OK":
                done.add(parts[3])
    failed = 0
    with a.log.open("a") as log:
        for f in sorted(files, key=lambda x: x["size"]):
            p = a.dir / f["path"]
            if f["path"] in done and p.exists() and p.stat().st_size == f["size"]:
                continue
            t0 = time.time()
            if not p.exists() or p.stat().st_size != f["size"]:
                ok, got, want = False, f"size {p.stat().st_size if p.exists() else None}", f"size {f['size']}"
            elif f.get("lfs"):
                got, want = digest(p, "sha256"), f["lfs"]["oid"]
                ok = got == want
            else:
                got, want = digest(p, "sha1", f"blob {f['size']}\0".encode()), f["oid"]
                ok = got == want
            failed += not ok
            print(f"{time.strftime('%F %T')} {'OK' if ok else 'FAIL'} {f['path']} {f['size']} "
                  f"{'sha256' if f.get('lfs') else 'git-sha1'} {got} {'' if ok else 'want ' + want} {time.time() - t0:.0f}s",
                  file=log, flush=True)
        print(f"{time.strftime('%F %T')} SUMMARY files={len(files)} failed={failed}", file=log, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
