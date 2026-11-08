"""Bounded resident calibration state; immutable input and streamed checkpoints.

Single row worker only. Tensor values/dtypes are unchanged. The converter's
temporary GPU assignment stays in a one-row cache until its CPU result arrives.
"""
import json
import os
from pathlib import Path
import struct
import tempfile
import torch
from row_checkpoint import link_rows, open_rows

from exllamav3.loader.safetensors_alt import (
    safe_open, save_file, _tensor_bytes_view, _stream_write, _TORCH_DTYPE_TO_ST,
    _ST_DTYPE_TO_TORCH, _stream_readinto,
)


def _read_row(reader, key):
    # Flatten before byte access: the pinned reader's BF16 numpy view otherwise
    # has multiple dimensions and len(view) is the first dimension, not bytes.
    info = reader._infos[key]
    count = 1
    for dim in info.shape:
        count *= dim
    value = torch.empty(count, dtype=_ST_DTYPE_TO_TORCH[info.dtype_tag])
    reader._f.seek(reader._data_start + info.begin)
    _stream_readinto(reader._f, _tensor_bytes_view(value), info.end-info.begin)
    return value.view(info.shape)


class DiskRows:
    def __init__(self, source, scratch_parent):
        self.scratch = tempfile.TemporaryDirectory(prefix='state-rows-', dir=scratch_parent)
        self.reader = None
        self.base = Path(self.scratch.name)/'base'
        if Path(source).with_suffix('.rows.json').exists():
            if Path(source).exists():
                raise ValueError('Ambiguous checkpoint formats')
            paths, self.infos = open_rows(source)
            self.count = len(paths)
            self.base.mkdir()
            # Private hardlinks survive upstream checkpoint directory rotation/deletion.
            for i, path in enumerate(paths):
                os.link(path,self.base/f'{i}.safetensors')
        else:
            self.reader = safe_open(str(source)).__enter__()
            self.count = len(self.reader.keys())
            if set(self.reader.keys()) != {f'tensor.{i}' for i in range(self.count)}:
                self.reader.__exit__(None, None, None)
                raise ValueError('Expected contiguous checkpoint row keys')
            self.infos = [(self.reader._infos[f'tensor.{i}'].dtype_tag,
                           self.reader._infos[f'tensor.{i}'].shape) for i in range(self.count)]
        self.changed = set()
        self.index = None
        self.tensor = None
        self.dirty = False

    def __len__(self):
        return self.count

    def _check(self, index):
        if not isinstance(index, int) or not 0 <= index < self.count:
            raise IndexError(index)

    def _path(self, index):
        return Path(self.scratch.name) / f'{index}.safetensors'

    def _flush(self):
        if not self.dirty:
            return
        value = self.tensor.detach().cpu().contiguous()
        path = self._path(self.index)
        pending = path.with_suffix('.pending')
        save_file({'tensor': value}, str(pending))
        os.replace(pending, path)
        self.infos[self.index] = (_TORCH_DTYPE_TO_ST[value.dtype], tuple(value.shape))
        self.changed.add(self.index)
        self.tensor = value
        self.dirty = False

    def __getitem__(self, index):
        self._check(index)
        if index != self.index:
            self._flush()
            self.tensor = None
            if index in self.changed:
                with safe_open(str(self._path(index))) as reader:
                    value = _read_row(reader, 'tensor')
            elif self.reader is None:
                with safe_open(str(self.base/f'{index}.safetensors')) as reader:
                    value = _read_row(reader, 'tensor')
            else:
                value = _read_row(self.reader, f'tensor.{index}')
            self.tensor, self.index = value, index
        return self.tensor

    def __setitem__(self, index, value):
        self._check(index)
        if index != self.index:
            self._flush()
        self.index, self.tensor, self.dirty = index, value, True
        if value.device.type == 'cpu':
            self._flush()

    def save(self, destination):
        self._flush()
        header, offset = {}, 0
        sizes = {'BOOL':1, 'U8':1, 'I8':1, 'I16':2, 'I32':4, 'I64':8,
                 'F16':2, 'BF16':2, 'F32':4, 'F64':8}
        for i, (dtype, shape) in enumerate(self.infos):
            size = sizes[dtype]
            for dim in shape:
                size *= dim
            header[f'tensor.{i}'] = dict(dtype=dtype, shape=list(shape),
                                        data_offsets=[offset, offset + size])
            offset += size
        raw = json.dumps(header, separators=(',', ':')).encode()
        with open(destination, 'wb', buffering=0) as stream:
            _stream_write(stream, memoryview(struct.pack('<Q', len(raw))))
            _stream_write(stream, memoryview(raw))
            for i in range(self.count):
                _stream_write(stream, _tensor_bytes_view(self[i].contiguous().view(-1)))
            os.fsync(stream.fileno())

    def close(self):
        self.tensor = None
        if self.reader is not None:
            self.reader.__exit__(None, None, None)
        self.scratch.cleanup()

    def save_checkpoint(self, destination):
        self._flush()
        paths = []
        for i in range(self.count):
            if i in self.changed:
                path = self._path(i)
            elif self.reader is None:
                path = self.base/f'{i}.safetensors'
            else:
                # Aggregate-format migration only; normal advance already spills every row.
                self[i] = self[i]
                path = self._path(i)
            paths.append(path)
        link_rows(paths,destination,self.infos)


def install(cm, scratch_parent):
    """Only replace state I/O; small original token IDs keep upstream handling."""
    old_load, old_save = cm.load_tensor, cm.save_tensor
    stores = []

    def load(filename, args):
        if filename != 'ckpt/state.safetensors':
            return old_load(filename, args)
        store = DiskRows(Path(args['work_dir']) / filename, scratch_parent)
        if len(store) != args['cal_rows']:
            store.close()
            raise ValueError('Checkpoint/calibration row count mismatch')
        stores.append(store)
        print(f' -- Disk-backed calibration state: {len(store)} rows, one resident row', flush=True)
        return store

    def save(value, filename, args):
        if isinstance(value, DiskRows):
            value.save_checkpoint(Path(args['work_dir']) / filename)
        else:
            old_save(value, filename, args)

    cm.load_tensor, cm.save_tensor = load, save
    return stores
