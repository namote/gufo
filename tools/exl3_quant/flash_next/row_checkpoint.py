"""Immutable per-row checkpoints: hardlinks, validated metadata, atomic manifest."""
import json
import os
from pathlib import Path
import struct


def tensor_header(path):
    path = Path(path)
    with path.open('rb') as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise ValueError(f'Truncated header: {path}')
        length = struct.unpack('<Q', raw)[0]
        if not 0 < length <= 16*2**20:
            raise ValueError(f'Invalid header length: {path}')
        header = json.loads(stream.read(length))
    entries = {k:v for k,v in header.items() if k != '__metadata__'}
    sizes = {'BOOL':1,'U8':1,'I8':1,'I16':2,'I32':4,'I64':8,
             'F16':2,'BF16':2,'F32':4,'F64':8}
    end = 0
    for info in sorted(entries.values(), key=lambda v:v['data_offsets'][0]):
        size = sizes[info['dtype']]
        for dim in info['shape']:
            if type(dim) is not int or dim < 0:
                raise ValueError('Invalid shape')
            size *= dim
        if info['data_offsets'] != [end,end+size]:
            raise ValueError('Invalid offsets')
        end += size
    if not entries or path.stat().st_size != 8+length+end:
        raise ValueError(f'Incomplete payload: {path}')
    return entries


def sync_directory(path):
    fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def link_rows(paths, destination, infos):
    """No tensor payload copies. Existing destination is an error, never overwritten."""
    destination = Path(destination)
    rows_dir = destination.with_suffix('.rows')
    manifest_path = destination.with_suffix('.rows.json')
    if destination.exists() or rows_dir.exists() or manifest_path.exists():
        raise FileExistsError(destination)
    if len(paths) != len(infos) or not paths:
        raise ValueError('Row count mismatch')
    rows_dir.mkdir()
    records = []
    for i, (source, (dtype, shape)) in enumerate(zip(paths,infos,strict=True)):
        source = Path(source)
        header = tensor_header(source)
        info = header.get('tensor')
        if len(header) != 1 or info is None or info['dtype'] != dtype or info['shape'] != list(shape):
            raise ValueError(f'Row metadata mismatch: {i}')
        target = rows_dir/f'{i}.safetensors'
        os.link(source,target,follow_symlinks=False)
        with target.open('rb') as stream:
            os.fsync(stream.fileno())
        records.append(dict(dtype=dtype,shape=list(shape),bytes=target.stat().st_size))
    sync_directory(rows_dir)
    manifest = dict(format='gufo.calibration-rows.v1',rows=records)
    temporary = manifest_path.with_suffix('.pending')
    with temporary.open('x') as stream:
        json.dump(manifest,stream,separators=(',',':'))
        stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,manifest_path)
    sync_directory(destination.parent)


def open_rows(source):
    source = Path(source)
    manifest = json.loads(source.with_suffix('.rows.json').read_text())
    if manifest['format'] != 'gufo.calibration-rows.v1' or not manifest['rows']:
        raise ValueError('Invalid row checkpoint manifest')
    paths, infos = [], []
    for i, record in enumerate(manifest['rows']):
        path = source.with_suffix('.rows')/f'{i}.safetensors'
        header = tensor_header(path)
        if len(header) != 1 or header.get('tensor',{}).get('dtype') != record['dtype'] or \
                header['tensor']['shape'] != record['shape'] or path.stat().st_size != record['bytes']:
            raise ValueError(f'Invalid checkpoint row: {i}')
        paths.append(path);infos.append((record['dtype'],tuple(record['shape'])))
    return paths, infos
