"""Extract only the 200 original HC matrices and audit F16 value preservation.

CPU only. The output is a separate checkpoint used by frozen comparison
binaries; it never modifies the deployed model or its calibrated sidecar.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import struct

from .weights import read_weight_payload


def audit(source, target, identities, output):
    import numpy as np

    source, target, identities, output = map(Path, (source, target, identities, output))
    saved = json.loads(identities.read_text())
    assert len(saved['source']) == len(saved['target']) == 200
    assert set(saved['source']) == set(saved['target'])
    output.mkdir(parents=True, exist_ok=False)
    entries, offset = {}, 0
    for name, identity in sorted(saved['source'].items()):
        assert identity['dtype'] == 'BF16'
        length = int(np.prod(identity['shape'])) * 2
        entries[name] = {'dtype': 'BF16', 'shape': identity['shape'], 'data_offsets': [offset, offset + length]}
        offset += length
    header = json.dumps(entries, separators=(',', ':')).encode()
    header += b' ' * (-len(header) % 8)
    results = {}
    with (output / 'model.safetensors').open('xb') as f:
        f.write(struct.pack('<Q', len(header))); f.write(header)
        for name in entries:
            original, oi = read_weight_payload(source, name, 'BF16')
            packed, pi = read_weight_payload(target, name, 'F16')
            assert oi == saved['source'][name], name
            assert pi == saved['target'][name], name
            b = (np.frombuffer(original, dtype='<u2').astype('<u4') << 16).view('<f4')
            h = np.frombuffer(packed, dtype='<f2').astype('<f4')
            assert np.isfinite(b).all() and np.isfinite(h).all(), name
            results[name] = {'source': oi, 'target': pi, 'elements': len(b),
                             'mismatches': int(np.count_nonzero(b != h)),
                             'max_abs': float(np.max(np.abs(b - h)))}
            f.write(original)
    shutil.copyfile(target / 'config.json', output / 'config.json')
    (output / 'model.safetensors.index.json').write_text(json.dumps({
        'metadata': {'total_size': offset},
        'weight_map': {n: 'model.safetensors' for n in entries}}, indent=2) + '\n')
    digest = hashlib.sha256()
    with (output / 'model.safetensors').open('rb') as f:
        for chunk in iter(lambda: f.read(8 * 1024**2), b''): digest.update(chunk)
    report = {'scope': '200 HC down/up matrices only, main 194 + MTP 6',
              'status': 'completed', 'payload_bytes': offset,
              'mismatches': sum(r['mismatches'] for r in results.values()),
              'elements': sum(r['elements'] for r in results.values()),
              'max_abs': max(r['max_abs'] for r in results.values()),
              'pack_sha256': digest.hexdigest(), 'tensors': results}
    (output / 'AUDIT.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: v for k, v in report.items() if k != 'tensors'}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('source', 'target', 'identities', 'output'): p.add_argument('--' + name, required=True)
    audit(**vars(p.parse_args()))
