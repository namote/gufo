#!/usr/bin/env python3
"""Fetch, verify and unpack the pinned Kyojin (ExLlamaV3 fork) converter source.

Kyojin is an external, MIT-licensed build-time dependency (see
THIRD_PARTY_NOTICES.md); gufo never vendors it. The pin is
runtime/SOURCE.json: Yamz-Labs/kyojin v1.8.6, commit fbde8a36..., GitHub
commit-archive sha256 dae498bd...7194. (The `refs/tags/v1.8.6` archive has a
different top-level directory and therefore a different sha256; only the
commit archive is pinned.)

  pin.py fetch   --archive OUT.tar.gz        backup copy, else GitHub; verify sha256
  pin.py unpack  --archive A --dest DIR      safe extract to DIR (top directory stripped)
  pin.py check   --dest DIR                  compare DIR with runtime/INPUTS.json
  pin.py patch   --dest DIR                  apply patches/*.patch (serving-only, optional)

`check` proves the tree is the one the c03 converter was built from: every
`source/` entry of INPUTS.json (recorded before the build) must match by
sha256, symlinks compared through their targets; the built
exllamav3_ext*.so entry is skipped. Standard library only; run with python3 -I.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import shutil
import subprocess
import sys
import tarfile
import urllib.request

HERE = Path(__file__).resolve().parent
SOURCE_JSON = HERE / "runtime" / "SOURCE.json"
INPUTS_JSON = HERE / "runtime" / "INPUTS.json"
PATCHES = HERE / "patches"
DEFAULT_BACKUP = Path("/home/m2on/projects/_backup/third_party/kyojin-v1.8.6.tar.gz")


def pin(source_json: Path = SOURCE_JSON) -> dict:
    data = json.loads(source_json.read_text())
    origin = data["origin"].rstrip("/")
    return {"origin": origin, "revision": data["revision"], "tag": data["tag"],
            "sha256": data["archive_sha256"], "url": f"{origin}/archive/{data['revision']}.tar.gz",
            "prefix": origin.rsplit("/", 1)[1] + "-" + data["revision"]}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def fetch(out: Path, expected: str, url: str, backup: Path | None) -> str:
    """Copy the backup archive, or download url, into out; return where it came from."""
    if out.exists():
        raise FileExistsError(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_name(out.name + ".part")
    origin = url
    try:
        if backup is not None and backup.is_file() and sha256_file(backup) == expected:
            shutil.copyfile(backup, tmp)
            origin = str(backup)
        else:
            with urllib.request.urlopen(url, timeout=120) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)
        got = sha256_file(tmp)
        if got != expected:
            raise ValueError(f"sha256 mismatch for {origin}: {got} != {expected}")
        tmp.replace(out)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    return origin


def safe_members(tar: tarfile.TarFile, prefix: str):
    """Yield members under prefix/ with the prefix stripped; reject anything escaping."""
    for m in tar.getmembers():
        parts = PurePosixPath(m.name).parts
        if not parts or parts[0] != prefix:
            raise ValueError(f"unexpected archive member {m.name!r}")
        if len(parts) == 1:
            continue
        rel = PurePosixPath(*parts[1:])
        if rel.is_absolute() or ".." in rel.parts:
            raise ValueError(f"unsafe archive member {m.name!r}")
        if not (m.isfile() or m.isdir() or m.issym()):
            raise ValueError(f"unsupported archive member type {m.name!r}")
        if m.issym():
            target = PurePosixPath(m.linkname)
            if target.is_absolute() or not _inside(rel, target):
                raise ValueError(f"symlink escapes the tree: {m.name!r} -> {m.linkname!r}")
        yield m, rel


def _inside(rel: PurePosixPath, target: PurePosixPath) -> bool:
    depth = len(rel.parts) - 1
    for part in target.parts:
        depth += -1 if part == ".." else 0 if part == "." else 1
        if depth < 0:
            return False
    return True


def _resolved_parent(dest: Path, rel: PurePosixPath) -> PurePosixPath:
    """rel's parent after resolving symlinks already on disk; reject one outside dest or a taken path."""
    root = os.path.realpath(dest)
    parent = os.path.realpath(dest / rel.parent)
    if os.path.commonpath([root, parent]) != root:
        raise ValueError(f"archive member escapes the tree through a symlink: {rel}")
    if os.path.lexists(dest / rel) and not (dest / rel).is_dir():
        raise ValueError(f"duplicate archive member: {rel}")
    return PurePosixPath(os.path.relpath(parent, root))


def unpack(archive: Path, dest: Path, prefix: str) -> int:
    """Extract directories and files first, then symlinks, each checked against the tree on disk."""
    if dest.exists() and any(dest.iterdir()):
        raise FileExistsError(f"{dest} is not empty")
    dest.mkdir(parents=True, exist_ok=True)
    count, links = 0, []
    with tarfile.open(archive) as tar:
        for m, rel in safe_members(tar, prefix):
            path = dest / rel
            if m.issym():
                links.append((m.linkname, rel))
                continue
            _resolved_parent(dest, rel)
            if m.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                with tar.extractfile(m) as src, path.open("xb") as f:
                    shutil.copyfileobj(src, f)
                os.chmod(path, 0o755 if m.mode & 0o111 else 0o644)
                count += 1
    for target, rel in links:
        parent = _resolved_parent(dest, rel)
        if not _inside(PurePosixPath(*parent.parts, rel.name) if parent.parts != (".",) else PurePosixPath(rel.name),
                       PurePosixPath(target)):
            raise ValueError(f"symlink escapes the tree: {rel} -> {target}")
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        os.symlink(target, dest / rel)
    # A chain of individually contained links can still resolve outside; check the final tree.
    root = os.path.realpath(dest)
    for _, rel in links:
        if os.path.commonpath([root, os.path.realpath(dest / rel)]) != root:
            raise ValueError(f"symlink resolves outside the tree: {rel}")
    return count


def check(dest: Path, inputs_json: Path = INPUTS_JSON) -> list[str]:
    """Return the mismatches between dest and the recorded source/ hashes."""
    files = json.loads(inputs_json.read_text())["files"]
    problems = []
    for key, digest in sorted(files.items()):
        if not key.startswith("source/"):
            continue
        rel = key[len("source/"):]
        if "/" not in rel and rel.startswith("exllamav3_ext") and rel.endswith(".so"):
            continue  # the built extension, not a source input
        path = dest / rel
        if not path.is_file():
            problems.append(f"missing {rel}")
        elif sha256_file(path) != digest:
            problems.append(f"differs {rel}")
    return problems


def apply_patches(dest: Path, patches: Path = PATCHES) -> list[str]:
    applied = []
    for p in sorted(patches.glob("*.patch")):
        subprocess.run(["patch", "-p1", "--forward", "--batch", "-i", str(p)], cwd=dest, check=True)
        applied.append(p.name)
    return applied


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="action", required=True)
    f = sub.add_parser("fetch")
    f.add_argument("--archive", type=Path, required=True)
    f.add_argument("--backup", type=Path, default=DEFAULT_BACKUP,
                   help=f"offline copy tried first (default {DEFAULT_BACKUP}); 'none' to skip")
    u = sub.add_parser("unpack")
    u.add_argument("--archive", type=Path, required=True)
    u.add_argument("--dest", type=Path, required=True)
    c = sub.add_parser("check")
    c.add_argument("--dest", type=Path, required=True)
    p = sub.add_parser("patch")
    p.add_argument("--dest", type=Path, required=True)
    sub.add_parser("show")
    a = ap.parse_args(argv)
    pinned = pin()
    if a.action == "show":
        print(json.dumps(pinned, indent=2))
    elif a.action == "fetch":
        backup = None if str(a.backup) == "none" else a.backup
        origin = fetch(a.archive, pinned["sha256"], pinned["url"], backup)
        print(f"verified {a.archive} sha256 {pinned['sha256']} from {origin}")
    elif a.action == "unpack":
        if sha256_file(a.archive) != pinned["sha256"]:
            raise SystemExit(f"{a.archive}: sha256 is not the pinned {pinned['sha256']}")
        print(f"unpacked {unpack(a.archive, a.dest, pinned['prefix'])} files into {a.dest}")
    elif a.action == "check":
        problems = check(a.dest)
        for line in problems:
            print(line, file=sys.stderr)
        if problems:
            return 1
        print(f"{a.dest}: matches runtime/INPUTS.json (Kyojin {pinned['tag']} {pinned['revision']})")
    else:
        print("applied", *apply_patches(a.dest))
    return 0


if __name__ == "__main__":
    sys.exit(main())
