"""Read model headers only; do not load tensor payloads or initialize a GPU."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def inventory(model, expected_dtype='BF16'):
    model = Path(model).resolve()
    index = model / 'model.safetensors.index.json'
    raw = index.read_bytes()
    mapping = json.loads(raw)['weight_map']
    headers, tensors = {}, []
    for key, shard in sorted(mapping.items()):
        if not key.endswith(('.input_mix_weight_down.weight', '.input_mix_weight_up.weight')):
            continue
        path = (model / shard).resolve()
        if not path.is_relative_to(model):
            raise ValueError('shard escapes model')
        if shard not in headers:
            with path.open('rb') as f:
                n = struct.unpack('<Q', f.read(8))[0]
                if n > 64 * 1024**2:
                    raise ValueError('oversized header')
                b = f.read(n)
                headers[shard] = json.loads(b)
        t = headers[shard][key]
        if t['dtype'] != expected_dtype or len(t['shape']) != 2 or t['shape'][1] % 32:
            raise ValueError('unsupported HC tensor: ' + key)
        m, k = t['shape']
        tensors.append({'key': key, 'shard': shard, 'shape': t['shape'], 'dtype': t['dtype'],
                        'data_offsets': t['data_offsets'], 'candidate_bytes': m * (k // 32) * 34})
    return {'status': 'headers_only_gpu_capture_pending', 'model': str(model), 'expected_dtype': expected_dtype,
            'index_sha256': hashlib.sha256(raw).hexdigest(), 'tensors': tensors,
            'tensor_count': len(tensors), 'candidate_bytes': sum(t['candidate_bytes'] for t in tensors),
            'shards': {s: {'size': (model / s).stat().st_size, 'mtime_ns': (model / s).stat().st_mtime_ns}
                       for s in headers}}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--dtype', choices=('BF16', 'F16'), default='BF16')
    args = p.parse_args(argv)
    result = inventory(args.model, args.dtype)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open('x') as f:
        json.dump(result, f, indent=2)
        f.write('\n')
    print(json.dumps({k: v for k, v in result.items() if k not in ('tensors', 'shards')}))


if __name__ == '__main__':
    main()
