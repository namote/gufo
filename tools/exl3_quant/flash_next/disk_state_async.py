"""Bounded CPU prefetch/write overlap; GPU work and row order stay on caller."""
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import os
import time
from disk_state import DiskRows as SerialRows, _read_row
from exllamav3.loader.safetensors_alt import safe_open, save_file, _TORCH_DTYPE_TO_ST
from row_checkpoint import link_rows


class DiskRows(SerialRows):
    def __init__(self, source, scratch_parent, *, read_bytes=2*2**30,
                 write_bytes=2*2**30, ahead=8):
        super().__init__(source, scratch_parent)
        self.source = Path(source)
        if self.reader is not None:
            self.source = Path(self.scratch.name)/'aggregate.safetensors'
            os.link(source, self.source)  # Survive upstream checkpoint directory rotation.
        self.read_limit, self.write_limit, self.ahead = read_bytes, write_bytes, ahead
        assert min(read_bytes, write_bytes, ahead) > 0
        self.read_pool = ThreadPoolExecutor(1, thread_name_prefix='state-read')
        self.write_pool = ThreadPoolExecutor(1, thread_name_prefix='state-write')
        self.prefetched = {}
        self.writes = deque()
        self.pending = {}
        self.write_bytes = 0
        self.stats = dict(prefetched_rows=0, read_wait_seconds=0., write_wait_seconds=0.,
                          peak_prefetch_bytes=0, peak_write_bytes=0, written_rows=0)
        self.closed = False

    def _bytes(self, index):
        dtype, shape = self.infos[index]
        sizes = {'BOOL':1,'U8':1,'I8':1,'I16':2,'I32':4,'I64':8,
                 'F16':2,'BF16':2,'F32':4,'F64':8}
        n = sizes[dtype]
        for d in shape:
            n *= d
        return n

    @staticmethod
    def _read(path, key, dependency):
        if dependency is not None:
            dependency.result()
        # Never share seek offsets with the main reader or another task.
        with safe_open(str(path)) as reader:
            return _read_row(reader, key)

    def _read_args(self, index):
        if index in self.changed:
            return self._path(index), 'tensor', self.pending.get(index)
        if self.reader is None:
            return self.base/f'{index}.safetensors', 'tensor', None
        return self.source, f'tensor.{index}', None

    def _discard(self, index):
        entry = self.prefetched.pop(index, None)
        if entry is not None:
            future, _ = entry
            if not future.cancel():
                future.result()  # Join abandoned reads; do not hide worker failures.

    def _prime(self, index):
        wanted = []
        size = 0
        for i in range(index+1, min(self.count, index+1+self.ahead)):
            n = self._bytes(i)
            if size+n > self.read_limit:
                break
            wanted.append(i)
            size += n
        for i in list(self.prefetched):
            if i not in wanted:
                self._discard(i)
        for i in wanted:
            if i not in self.prefetched:
                self.prefetched[i] = (self.read_pool.submit(self._read, *self._read_args(i)), self._bytes(i))
        self.stats['peak_prefetch_bytes'] = max(self.stats['peak_prefetch_bytes'], size)

    def _reap_writes(self, all_done=False, room=0):
        while self.writes:
            index, future, size = self.writes[0]
            if not all_done and self.write_bytes+room <= self.write_limit and not future.done():
                break
            start = time.monotonic()
            future.result()
            self.stats['write_wait_seconds'] += time.monotonic()-start
            self.writes.popleft()
            self.write_bytes -= size
            if self.pending.get(index) is future:
                del self.pending[index]

    @staticmethod
    def _write(path, snapshot):
        temporary = path.with_suffix('.pending')
        save_file({'tensor': snapshot}, str(temporary))
        os.replace(temporary, path)

    def _flush(self):
        if not self.dirty:
            self._reap_writes()
            return
        value = self.tensor.detach().cpu().contiguous()
        size = value.numel()*value.element_size()
        if size > self.write_limit:
            raise ValueError('Single row exceeds asynchronous write budget')
        self._reap_writes(room=size)
        # Caller may mutate its CPU tensor after assignment: writer owns a snapshot.
        snapshot = value.clone()
        future = self.write_pool.submit(self._write, self._path(self.index), snapshot)
        self.writes.append((self.index, future, size))
        self.pending[self.index] = future
        self.write_bytes += size
        self.stats['peak_write_bytes'] = max(self.stats['peak_write_bytes'], self.write_bytes)
        self.stats['written_rows'] += 1
        self.infos[self.index] = (_TORCH_DTYPE_TO_ST[value.dtype], tuple(value.shape))
        self.changed.add(self.index)
        self.tensor = value
        self.dirty = False

    def __getitem__(self, index):
        self._check(index)
        if index != self.index:
            self._flush()
            entry = self.prefetched.pop(index, None)
            start = time.monotonic()
            if entry is None:
                value = self._read(*self._read_args(index))
            else:
                value = entry[0].result()
                self.stats['prefetched_rows'] += 1
            self.stats['read_wait_seconds'] += time.monotonic()-start
            self.tensor, self.index = value, index
            self._prime(index)
        return self.tensor

    def __setitem__(self, index, value):
        self._check(index)
        if index != self.index:
            self._flush()
        # A queued read of this row predates this write; join and invalidate it.
        self._discard(index)
        self.index, self.tensor, self.dirty = index, value, True
        if value.device.type == 'cpu':
            self._flush()

    def save_checkpoint(self, destination):
        self._flush()
        paths = []
        for i in range(self.count):
            if i in self.changed:
                path = self._path(i)
            elif self.reader is None:
                path = self.base/f'{i}.safetensors'
            else:
                self[i] = self[i]
                path = self._path(i)
            paths.append(path)
        self._reap_writes(all_done=True)
        for i in list(self.prefetched):
            self._discard(i)
        # Persisted checkpoint sees only complete files, then existing fsync/link rules.
        link_rows(paths, destination, self.infos)

    def close(self):
        if self.closed:
            return
        self.closed = True
        try:
            self._flush()
            self._reap_writes(all_done=True)
            for i in list(self.prefetched):
                self._discard(i)
        finally:
            self.read_pool.shutdown(wait=True, cancel_futures=True)
            self.write_pool.shutdown(wait=True, cancel_futures=True)
            super().close()
        print('GUFO_STATE_IO', self.stats, flush=True)


def install(cm, scratch_parent):
    old_load, old_save = cm.load_tensor, cm.save_tensor
    stores = []
    def load(filename, args):
        if filename != 'ckpt/state.safetensors':
            return old_load(filename, args)
        store = DiskRows(Path(args['work_dir'])/filename, scratch_parent)
        if len(store) != args['cal_rows']:
            store.close()
            raise ValueError('Checkpoint/calibration row count mismatch')
        stores.append(store)
        print(f'GUFO_STATE_IO_ACTIVE rows={len(store)} read=2GiB write=2GiB ahead=8 CPU_workers=2', flush=True)
        return store
    def save(value, filename, args):
        if isinstance(value, DiskRows):
            value.save_checkpoint(Path(args['work_dir'])/filename)
        else:
            old_save(value, filename, args)
    cm.load_tensor, cm.save_tensor = load, save
    return stores
