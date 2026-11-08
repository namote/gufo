#!/usr/bin/env python3
"""Install the Flash-Next serving supervisor into ~/.local/share/gufo (standard library only).

The live supervisor must not depend on the working tree while it is edited, so
this copies one committed revision of the files in FILES out of git (`git show
REV:path`, default HEAD) instead of linking them. It

  1. backs up the files it will replace to BACKUP_ROOT/gufo-serving-<stamp>-<sha>/
     and appends a line for that folder to BACKUP_ROOT/README.md (nothing is
     backed up or written when no file changes),
  2. writes each file atomically (temporary file in the destination, then rename),
  3. byte-compiles every installed .py file in memory, and
  4. records the revision, the time and each file's sha256 in DEST/INSTALLED.json.

Nothing else in DEST is touched: not ops/OWNER.lock (the GPU owner lock a running
supervisor holds), not ops/DESKTOP-*.json (boot-scoped admission manifests),
server-state.json, profile.json, logs/ or caches. It holds DEST/gufo-control.lock,
the lock `llm start` and `llm stop` take, while it writes.

Install with the server stopped. A running supervisor keeps the Python code it
loaded, but starts the root-run readers (ops/root_read.py, ops/fence_read.py)
from disk for each reading, so it would mix old and new code; a reader whose
output changed could then fail its health record at `llm stop`. The installer
refuses while a supervisor runs unless --allow-running is given, which is safe
only when the readers' behaviour is unchanged.

  install.py [--rev REV] [--dest DIR] [--backup-root DIR] [--dry-run] [--allow-running]
  install.py --check [--dest DIR]      compare DEST with its INSTALLED.json
  install.py --from-dir DIR ...        install a directory's files (tests only)
"""
from __future__ import annotations

import argparse
import datetime
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

HERE = Path(__file__).resolve().parent
REPO_PATH = 'tools/serving/flash_next'
FILES = ('gufo-server.py', 'ops/__init__.py', 'ops/admission.py', 'ops/desktop.py', 'ops/fence.py',
         'ops/fence_read.py', 'ops/host_audit.py', 'ops/journal.py', 'ops/legacy_root_read.py',
         'ops/npu_audit.py', 'ops/root_read.py', 'ops/window.py')
DEST = Path.home() / '.local/share/gufo'
BACKUP_ROOT = Path.home() / 'projects/_backup'
RECORD = 'INSTALLED.json'


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def git(*args: str) -> bytes:
    return subprocess.run(['git', '-C', str(HERE), *args], check=True, capture_output=True).stdout


def read_revision(rev: str) -> tuple[str, dict[str, bytes]]:
    """(full commit sha, {file: bytes}) of FILES at a committed revision."""
    commit = git('rev-parse', '--verify', rev + '^{commit}').decode().strip()
    return commit, {name: git('show', f'{commit}:{REPO_PATH}/{name}') for name in FILES}


def read_directory(source: Path) -> tuple[str, dict[str, bytes]]:
    return 'directory:' + str(source), {name: (source / name).read_bytes() for name in FILES}


def compile_check(files: dict[str, bytes]) -> None:
    for name, data in files.items():
        compile(data, name, 'exec')  # raises SyntaxError before anything is written


def backup(dest: Path, backup_root: Path, revision: str, stamp: str, changed: list[str]) -> Path | None:
    present = [name for name in changed if (dest / name).exists()]
    if not present:
        return None
    tag = 'dir' if revision.startswith('directory:') else revision[:8]
    folder = backup_root / f'gufo-serving-{stamp}-{tag}'
    suffix = 1
    while folder.exists():
        suffix += 1
        folder = backup_root / f'gufo-serving-{stamp}-{tag}-{suffix}'
    for name in present:
        target = folder / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes((dest / name).read_bytes())
    previous = dest / RECORD
    if previous.exists():
        (folder / RECORD).write_bytes(previous.read_bytes())
    readme = backup_root / 'README.md'
    line = (f'\n- `{folder.name}/`: serving supervisor files that '
            f'`tools/serving/flash_next/install.py` replaced with {revision}. Restore by copying '
            f'the folder\'s files back into `~/.local/share/gufo/` (then restart the server).\n')
    with readme.open('a') as out:
        out.write(line)
    return folder


def write_atomic(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + path.name + '.install-tmp')
    tmp.write_bytes(data)
    os.chmod(tmp, 0o664)
    os.replace(tmp, path)


def supervisor_running(dest: Path) -> bool:
    """Whether server-state.json names a live `gufo-server.py _supervise` process."""
    try:
        pid = json.loads((dest / 'server-state.json').read_text())['pid']
        command = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
    except (OSError, ValueError, KeyError, TypeError):
        return False
    return b'_supervise' in command and any(part.endswith(b'gufo-server.py') for part in command)


def install(files: dict[str, bytes], revision: str, dest: Path, backup_root: Path, dry_run: bool) -> dict:
    compile_check(files)
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    changed = [name for name in FILES if not (dest / name).exists() or (dest / name).read_bytes() != files[name]]
    record = {'revision': revision, 'installed_utc': stamp, 'source': REPO_PATH,
              'files': {name: sha256(files[name]) for name in FILES}, 'changed': changed}
    if dry_run or not changed:
        return record
    folder = backup(dest, backup_root, revision, stamp, changed)
    record['backup'] = None if folder is None else str(folder)
    for name in changed:
        write_atomic(dest / name, files[name])
    write_atomic(dest / RECORD, (json.dumps(record, indent=2) + '\n').encode())
    return record


def check(dest: Path) -> int:
    if not (dest / RECORD).exists():
        print(f'no {RECORD} in {dest}: nothing was installed there by this tool')
        return 2
    record = json.loads((dest / RECORD).read_text())
    bad = [name for name, digest in record['files'].items()
           if not (dest / name).exists() or sha256((dest / name).read_bytes()) != digest]
    print(json.dumps({'revision': record['revision'], 'installed_utc': record['installed_utc'], 'modified': bad},
                     indent=2))
    return 1 if bad else 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--rev', default='HEAD', help='committed revision to install (default HEAD)')
    ap.add_argument('--from-dir', type=Path, help='install FILES from this directory instead of git (tests)')
    ap.add_argument('--dest', type=Path, default=DEST)
    ap.add_argument('--backup-root', type=Path, default=BACKUP_ROOT)
    ap.add_argument('--dry-run', action='store_true', help='print what would change, write nothing')
    ap.add_argument('--check', action='store_true', help='verify DEST against its INSTALLED.json')
    ap.add_argument('--allow-running', action='store_true',
                    help='install while a supervisor runs (only when the root-run readers are unchanged)')
    a = ap.parse_args(argv)
    if a.check:
        return check(a.dest)
    revision, files = read_directory(a.from_dir) if a.from_dir else read_revision(a.rev)
    a.dest.mkdir(parents=True, exist_ok=True)
    with (a.dest / 'gufo-control.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not a.dry_run and not a.allow_running and supervisor_running(a.dest):
            print('a gufo supervisor is running; stop it first (llm stop) or pass --allow-running', file=sys.stderr)
            return 3
        print(json.dumps(install(files, revision, a.dest, a.backup_root, a.dry_run), indent=2))
    return 0


if __name__ == '__main__':
    sys.exit(main())
