"""Prepare a bounded fixed HC pilot from existing c03 and separate holdout IDs.

No tokenization, model execution or corpus regeneration. This small screen
does not replace the approved full calibration dataset.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

from .weights import read_header


SOURCES = ('smith', 'open', 'ko_wiki', 'ko_chat', 'c02:multi', 'c02:code', 'c02:en_tool', 'c02:math')


def sha(data):
    return hashlib.sha256(data).hexdigest()


def token_row(raw, header, start, index, length, digest):
    entry = header[f'input_ids.{index}']
    a, b = entry['data_offsets']
    if entry['dtype'] != 'I32' or entry['shape'] != [1, length] or b - a != length * 4:
        raise ValueError('unexpected input row layout')
    if a < 0 or start + b > len(raw):
        raise ValueError('input row outside file')
    data = raw[start + a:start + b]
    if sha(data) != digest:
        raise ValueError('changed input row')
    if any(not 0 <= x[0] < 248320 for x in struct.iter_unpack('<i', data)):
        raise ValueError('token outside vocabulary')
    return data


def prepare(calibration, holdout, output):
    calibration, holdout, output = map(Path, (calibration, holdout, output))
    if output.exists():
        raise ValueError('refuse existing pilot input directory')
    cm_path = calibration.with_suffix('.json')
    cm = json.loads(cm_path.read_text()); cr = calibration.read_bytes()
    if sha(cr) != cm['sha256']:
        raise ValueError('calibration file identity mismatch')
    ch, cb = read_header(calibration)
    hm_path = holdout / 'DATA.json'
    hm = json.loads(hm_path.read_text()); hr = (holdout / 'input-ids.safetensors').read_bytes()
    texts = (holdout / 'TEXTS.json').read_bytes()
    if sha(hr) != hm['input_sha256'] or sha(texts) != hm['texts_sha256']:
        raise ValueError('holdout file identity mismatch')
    hh, hb = read_header(holdout / 'input-ids.safetensors')
    selected = []
    for source in SOURCES:
        index, row = next((i, r) for i, r in enumerate(cm['sequences']) if r['source'] == source and r['tokens'] >= 65)
        data = token_row(cr, ch, cb, index, row['tokens'], row['token_sha256'])
        selected.append((f'c03:{index}:{source}:{row["id"]}', source, 'calibration', data[:513 * 4], row))
    for index, row in enumerate(hm['rows']):
        data = token_row(hr, hh, hb, index, row['tokens'], row['token_sha256'])
        # Reuse the prior lexical contamination screen, but rerun it on the
        # exact bytes consumed here. It is not semantic decontamination.
        for off in range(0, len(data) - 64 * 4 + 1, 32 * 4):
            if data[off:off + 64 * 4] in cr:
                raise ValueError('holdout sampled 64-token span occurs in calibration')
        selected.append(('holdout:' + row['id'], row['domain'], 'holdout', data[:513 * 4], row))
    if len({r[0] for r in selected}) != len(selected) or not len(hm['rows']) == 8:
        raise ValueError('unexpected or duplicate pilot documents')
    windows, rows, payload = [], [], bytearray()
    for i, (name, category, split, data, original) in enumerate(selected):
        if len(data) < 65 * 4:
            raise ValueError('pilot decode prefix needs at least 65 tokens')
        windows.append({'id': name, 'category': category, 'offset': len(payload) // 4, 'length': len(data) // 4})
        rows.append({'window': i, 'id': name, 'split': split, 'category': category,
                     'length': len(data) // 4, 'token_sha256': sha(data),
                     'original_tokens': original['tokens'], 'original_token_sha256': original['token_sha256']})
        payload.extend(data)
    output.mkdir(parents=True, exist_ok=False)
    (output / 'windows.bin').write_bytes(payload)
    corpus = {'schema': 'gufo.rq-corpus-windows.v1', 'tokens_file': 'windows.bin',
              'tokens_sha256': sha(payload), 'windows': windows}
    (output / 'windows.json').write_text(json.dumps(corpus, indent=2) + '\n')
    manifest = {'status': 'pilot_inputs_prepared_not_captured', 'gpu_used': False,
                'scope': 'first HC only; at most 513 tokens/document; not full calibration or long-context quality',
                'calibration_sha256': sha(cr), 'calibration_manifest_sha256': sha(cm_path.read_bytes()),
                'holdout_sha256': sha(hr), 'holdout_manifest_sha256': sha(hm_path.read_bytes()),
                'holdout_texts_sha256': sha(texts), 'windows_sha256': sha((output / 'windows.json').read_bytes()),
                'tokens_sha256': sha(payload), 'rows': rows,
                'builder_sha256': sha(Path(__file__).read_bytes()),
                'overlap_check': 'sampled contiguous 64-token spans, stride32; not semantic decontamination'}
    (output / 'INPUTS.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'output': str(output), 'windows': len(windows), 'tokens': len(payload) // 4, 'gpu_used': False}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--calibration', required=True)
    p.add_argument('--holdout', required=True)
    p.add_argument('--output', required=True)
    print(json.dumps(prepare(**vars(p.parse_args()))))
