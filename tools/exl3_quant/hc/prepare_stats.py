"""Validate saved first-HC captures and form phase-specific full-H statistics.

CPU only. Decode statistics exclude the prefill prefix; no phases are silently
mixed. H/held-out inputs are reused by all quantization candidates.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np

from .quant import sha
from .weights import load_weight


def read_windows(capture, manifest, phase, projection):
    root = Path(capture) / phase / 'capture'
    columns = {'down': 10240, 'up': 320}[projection]
    expected_dirs = {str(r['window']) for r in manifest['rows']}
    if {p.name for p in root.iterdir()} != expected_dirs:
        raise ValueError('missing or unexpected capture window')
    data, records = {}, []
    for doc in manifest['rows']:
        directory = root / str(doc['window'])
        chunks = [json.loads(line) for line in (directory / 'chunks.jsonl').read_text().splitlines()]
        expected_end = min(512, doc['length'] - 1) if phase == 'prefill' else 32 + min(128, doc['length'] - 33)
        used = set(); parts = []; pointer = 0
        for c in sorted((c for c in chunks if c['projection'] == projection), key=lambda c: c['position']):
            pos, rows = c['position'], c['rows']
            if type(pos) is not int or type(rows) is not int or pos != pointer or not 1 <= rows <= 512:
                raise ValueError('duplicate/missing/out-of-order captured rows')
            expected_phase = 'decode' if phase == 'decode' and pos >= 32 else 'prefill'
            if c['cols'] != columns or c['phase'] != expected_phase or pos + rows > expected_end:
                raise ValueError('captured shape/phase mismatch')
            if phase == 'decode' and ((pos < 32 and pos + rows > 32) or (pos >= 32 and rows != 1)):
                raise ValueError('capture crosses phase boundary or wrong decode width')
            filename = f'{projection}-{pos}.f32'
            if c['file'] != filename or filename in used:
                raise ValueError('unsafe/duplicate capture filename')
            path = directory / filename
            if path.is_symlink() or path.stat().st_size != rows * columns * 4:
                raise ValueError('invalid capture payload size')
            values = np.fromfile(path, dtype='<f4').reshape(rows, columns)
            if not np.isfinite(values).all():
                raise ValueError('nonfinite captured input')
            used.add(filename); pointer += rows
            records.append({'window': doc['window'], 'file': filename, 'phase': expected_phase,
                            'rows': rows, 'sha256': sha(path)})
            if expected_phase == phase:
                parts.append(values)
        if pointer != expected_end or not parts:
            raise ValueError('incomplete capture coverage')
        # Both projections must describe identical row boundaries.
        signatures = {p: sorted((c['position'], c['rows'], c['phase']) for c in chunks if c['projection'] == p)
                      for p in ('down', 'up')}
        if signatures['down'] != signatures['up'] or any(c['projection'] not in signatures for c in chunks):
            raise ValueError('down/up row coverage mismatch')
        expected_files = {'chunks.jsonl'} | {c['file'] for c in chunks}
        if {p.name for p in directory.iterdir()} != expected_files:
            raise ValueError('unexpected capture files')
        data[doc['id']] = np.concatenate(parts).astype(np.float64)
    return data, records


def prepare(capture, inputs, source_model, target_model, phase, projection, output):
    capture, inputs, source_model, target_model, output = map(Path, (capture, inputs, source_model, target_model, output))
    if output.exists() or any(output.resolve().is_relative_to(p.resolve()) for p in (source_model, target_model, capture, inputs)):
        raise ValueError('require a new separate stats output')
    state = json.loads((capture / 'STATE.json').read_text())
    if (state['status'] != 'captured_pending_independent_review' or state['cleanup_errors']
            or not state['service_restored'] or state.get('kernel_final', {}).get('new_GPU_error', True)):
        raise ValueError('capture window did not complete cleanly')
    for a, b in (('control', 'capture-control'), ('decode-control', 'decode-capture-control')):
        if any(state['cases'][a][k] != state['cases'][b][k] for k in ('logits_sha256', 'full_logits_sha256')):
            raise ValueError('capture control mismatch')
    manifest = json.loads((inputs / 'INPUTS.json').read_text())
    if sha(inputs / 'INPUTS.json') != state['inputs_sha256']:
        raise ValueError('capture input identity mismatch')
    key = f'model.language_model.layers.0.attn_hyper_connection.input_mix_weight_{projection}.weight'
    w, source_identity = load_weight(source_model, key, 'BF16')
    _, target_identity = load_weight(target_model, key, 'F16')
    if target_identity['model_index_sha256'] != state['model_index_sha256']:
        raise ValueError('captured target pack changed')
    if target_identity != state['target_hc_identities'][projection]:
        raise ValueError('target HC weights changed after capture')
    start = time.monotonic()
    values, records = read_windows(capture, manifest, phase, projection)
    train_docs = [d['id'] for d in manifest['rows'] if d['split'] == 'calibration']
    hold_docs = [d['id'] for d in manifest['rows'] if d['split'] == 'holdout']
    if not train_docs or not hold_docs or set(train_docs) & set(hold_docs):
        raise ValueError('invalid document split')
    train = np.concatenate([values[d] for d in train_docs])
    hold = np.concatenate([values[d] for d in hold_docs])
    if train.shape[1] != w.shape[1]:
        raise ValueError('weight/input dimension mismatch')
    h = train.T @ train / len(train)
    if not np.isfinite(h).all():
        raise ValueError('nonfinite input Gram')
    down_key = 'model.language_model.layers.0.attn_hyper_connection.input_mix_weight_down.weight'
    _, down_identity = load_weight(target_model, down_key, 'F16')
    if down_identity != state['target_hc_identities']['down']:
        raise ValueError('upstream down weights changed after capture')
    upstream = {'target_down': down_identity, 'runtime_sha256': state['collector']['binary_sha256'],
                'phase': phase, 'weight_state': 'B10 original target F16 source and native runtime Q8 copies; no calibrated sidecar'}
    receipt = {'schema': 'gufo-hc-input-gram-v2', 'tensor': key,
               'source_identity': source_identity, 'target_identity': target_identity,
               'input_space': 'unfolded_projection_input', 'H_normalization': 'sum_xxt_div_rows',
               'synthetic': False, 'calibration_rows': len(train), 'holdout_rows': len(hold),
               'calibration_document_ids': train_docs, 'holdout_document_ids': hold_docs,
               'capture_source_sha256': state['collector']['source_manifest_sha256'],
               'capture_runtime_sha256': state['collector']['binary_sha256'],
               'upstream_weight_state_sha256': hashlib.sha256(json.dumps(upstream, sort_keys=True).encode()).hexdigest(),
               'upstream_weight_state': upstream, 'phase': phase,
               'scope': 'first-attention-HC pilot; not full calibration or whole-model quality',
               'input_capture_records': records, 'holdout_documents': [
                   {'id': d, 'rows': len(values[d])} for d in hold_docs],
               'underdetermined_by_row_count': len(train) < w.shape[1],
               'prepare_seconds': time.monotonic() - start, 'builder_sha256': sha(__file__)}
    output.mkdir(parents=True)
    np.savez(output / 'stats.npz', H=h, X_holdout=hold)
    receipt['stats_sha256'] = sha(output / 'stats.npz')
    (output / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    return {k: receipt[k] for k in ('tensor', 'calibration_rows', 'holdout_rows', 'phase', 'underdetermined_by_row_count', 'prepare_seconds')}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('capture', 'inputs', 'source-model', 'target-model', 'output'):
        parser.add_argument('--' + name, required=True)
    parser.add_argument('--phase', choices=('prefill', 'decode'), required=True)
    parser.add_argument('--projection', choices=('down', 'up'), required=True)
    print(json.dumps(prepare(**vars(parser.parse_args())), indent=2))
