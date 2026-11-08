"""Read-only protected idx51 rows with a private writable output overlay.

Source rows are borrowed only while the parent owns the reserved GPU/JOB lease.
No source hardlinks or payload copies are made. Complete output checkpoints may
link only rows already advanced into the same writable work mount. Existing
serial/async/large implementations are not patched.
"""
import hashlib
import json
import os
from pathlib import Path
import tempfile

from disk_state import DiskRows as SerialRows
from disk_state_async import DiskRows as AsyncRows
from row_checkpoint import open_rows

PROTECTED_CHECKPOINT = Path('/output/work/ckpt')
EXPECTED_ROWS = 2309
EXPECTED_TOKENS = 1999496
EXPECTED_STREAMS = 4
EXPECTED_HIDDEN = 2560


def _readonly(path):
    return bool(os.statvfs(path).f_flag & os.ST_RDONLY)


def _stat(path):
    s = path.stat()
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns


def _digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class BorrowedRows(SerialRows):
    def __init__(self, source, scratch_parent):
        source = Path(source).absolute()
        scratch_parent = Path(scratch_parent).resolve(strict=True)
        assert os.environ.get('GUFO_MTP_LEASE_VERIFIED') == '1'
        assert source == PROTECTED_CHECKPOINT / 'state.safetensors'
        assert source.parent.resolve() == PROTECTED_CHECKPOINT
        assert _readonly(source.parent), 'Borrowed source must be mounted read-only'
        assert not source.exists(), 'Borrowed mode requires row checkpoint only'
        assert not source.with_suffix('.rows.pending').exists()
        job = source.parent / 'job.json'
        assert json.loads(job.read_text()) == {'next_module_idx':51, 'q_strategy':None, 'bad_rows':[]}
        paths, infos = open_rows(source)
        assert len(paths) == EXPECTED_ROWS
        assert all(d == 'F32' and len(s) == 4 and s[0] == 1 and s[1] >= 2 and
                   s[2:] == (EXPECTED_STREAMS, EXPECTED_HIDDEN) for d,s in infos)
        assert sum(s[1] for _,s in infos) == EXPECTED_TOKENS
        self.borrowed_source = source
        self.borrowed_paths = tuple(paths)
        self.borrowed_stats = tuple(_stat(p) for p in paths)
        self.borrowed_meta = {p:_digest(p) for p in (job,source.with_suffix('.rows.json'))}
        self.writable_root = scratch_parent.parent
        assert self.writable_root != source.parent and source.parent not in self.writable_root.parents
        self.scratch = tempfile.TemporaryDirectory(prefix='mtp-borrowed-', dir=scratch_parent)
        self.reader = None
        self.base = source.with_suffix('.rows')
        self.infos = list(infos)
        self.count = len(paths)
        self.changed = set()
        self.index = self.tensor = None
        self.dirty = False

    def check_source(self):
        assert os.environ.get('GUFO_MTP_LEASE_VERIFIED') == '1'
        assert _readonly(self.borrowed_source.parent)
        assert all(_digest(p) == h for p,h in self.borrowed_meta.items()), 'Borrowed source metadata changed'
        assert tuple(_stat(p) for p in self.borrowed_paths) == self.borrowed_stats, 'Borrowed source rows changed'

    def _check_destination(self, destination):
        self.check_source()
        destination = Path(destination)
        assert destination.is_absolute() and destination == destination.resolve()
        assert self.writable_root in destination.parents, 'Checkpoint outside writable work root'
        self._flush()
        assert self.changed == set(range(self.count)), 'Cannot checkpoint borrowed/unadvanced rows'

    def save(self, destination):
        raise RuntimeError('Aggregate export/copy is forbidden for borrowed source')

    def save_checkpoint(self, destination):
        self._check_destination(destination)
        SerialRows.save_checkpoint(self, destination)

    def close(self):
        try:
            self.check_source()
        finally:
            super().close()


class BorrowedLargeRows(AsyncRows, BorrowedRows):
    """AsyncRows.super() resolves to guarded BorrowedRows, skipping hardlinks."""
    def __init__(self, source, scratch_parent, *, read_bytes=10*2**30,
                 write_bytes=2*2**30, ahead=800):
        super().__init__(source, scratch_parent, read_bytes=read_bytes,
                         write_bytes=write_bytes, ahead=ahead)

    def save_checkpoint(self, destination):
        self._check_destination(destination)
        AsyncRows.save_checkpoint(self, destination)
