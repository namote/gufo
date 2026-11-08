"""Read only a selected tensor; BF16 bits expand exactly to FP32 (no F16 hop)."""
import hashlib
import json
from pathlib import Path
import struct


def read_header(path):
    with Path(path).open('rb') as f:
        prefix = f.read(8)
        if len(prefix) != 8:
            raise ValueError('truncated safetensors prefix')
        size = struct.unpack('<Q', prefix)[0]
        if not 2 <= size <= 64 * 1024**2:
            raise ValueError('invalid header length')
        raw = f.read(size)
        if len(raw) != size:
            raise ValueError('truncated safetensors header')
    return json.loads(raw), 8 + size


def read_weight_payload(model, tensor, expected_dtype):
    """Standard-library payload identity, also used before GPU collection."""
    model = Path(model).resolve()
    raw_index = (model / 'model.safetensors.index.json').read_bytes()
    shard = (model / json.loads(raw_index)['weight_map'][tensor]).resolve()
    if not shard.is_relative_to(model):
        raise ValueError('shard escaped model directory')
    header, start = read_header(shard)
    entry = header[tensor]
    shape, offsets = entry['shape'], entry['data_offsets']
    if entry['dtype'] != expected_dtype or expected_dtype not in ('BF16', 'F16'):
        raise ValueError('wrong source dtype: expected ' + expected_dtype)
    if len(shape) != 2 or any(type(v) is not int or v <= 0 for v in shape) or shape[1] % 32:
        raise ValueError('invalid HC matrix shape')
    if (len(offsets) != 2 or any(type(v) is not int for v in offsets)
            or offsets[0] < 0 or offsets[1] - offsets[0] != shape[0] * shape[1] * 2
            or start + offsets[1] > shard.stat().st_size):
        raise ValueError('invalid tensor offsets')
    with shard.open('rb') as f:
        f.seek(start + offsets[0])
        payload = f.read(offsets[1] - offsets[0])
    if len(payload) != offsets[1] - offsets[0]:
        raise ValueError('truncated tensor payload')
    return payload, {'dtype': expected_dtype, 'model_index_sha256': hashlib.sha256(raw_index).hexdigest(),
                     'weight_sha256': hashlib.sha256(payload).hexdigest(), 'shape': shape}


def load_weight(model, tensor, expected_dtype):
    import numpy as np

    payload, identity = read_weight_payload(model, tensor, expected_dtype)
    shape = identity['shape']
    if expected_dtype == 'BF16':
        bits = np.frombuffer(payload, dtype='<u2').astype('<u4') << 16
        w = bits.view('<f4').reshape(shape).astype(np.float32)
    else:
        w = np.frombuffer(payload, dtype='<f2').reshape(shape).astype(np.float16)
    if not np.isfinite(w).all():
        raise ValueError('nonfinite source weights')
    return w, identity
