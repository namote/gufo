"""Strict CPU reader for independent, unpadded calibration sequences."""
import hashlib
import json
from collections import Counter
from pathlib import Path

import torch
from safetensors.torch import load_file


def sync_calibration_metadata(pack, manifest):
    """Correct both upstream metadata files after successful pack compilation."""
    pack = Path(pack)
    config_path = pack / 'config.json'
    quant_path = pack / 'quantization_config.json'
    config = json.loads(config_path.read_text())
    quant = json.loads(quant_path.read_text())
    calibration = dict(
        format=manifest['format'], variable_length=True, rows=manifest['rows'],
        max_columns=manifest['max_columns'], total_tokens=manifest['total_tokens'],
        sha256=manifest['sha256'], source_tokens=manifest['source_tokens'],
        tokenizer_sha256=manifest['tokenizer_sha256'],
        template_sha256=manifest['template_sha256'],
        length_counts=dict(Counter(str(s['tokens']) for s in manifest['sequences'])))
    config['quantization_config']['calibration'] = calibration
    quant['calibration'] = calibration
    # Leave tensor_storage and every other quantization field untouched.
    for path, value in [(config_path, config), (quant_path, quant)]:
        temporary = path.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(value, indent=2))
        temporary.replace(path)
    if (json.loads(config_path.read_text())['quantization_config']['calibration'] !=
            json.loads(quant_path.read_text())['calibration']):
        raise ValueError('Final calibration metadata mismatch')


def load_calibration(args, tokenizer):
    path = Path(args['cal_data'])
    manifest = json.loads(path.with_suffix('.json').read_text())
    if manifest['format'] != 'gufo.ragged-calibration.v1' or not manifest['variable_length']:
        raise ValueError('Unsupported calibration format')
    source = Path(args['in_dir'])
    for name, key in [('tokenizer.json', 'tokenizer_sha256'), ('chat_template.jinja', 'template_sha256')]:
        if hashlib.sha256((source / name).read_bytes()).hexdigest() != manifest[key]:
            raise ValueError('Calibration tokenizer/template mismatch')
    if hashlib.sha256(path.read_bytes()).hexdigest() != manifest['sha256']:
        raise ValueError('Calibration hash mismatch')
    tensors = load_file(str(path))
    count = manifest['rows']
    if count <= 0 or len(manifest['sequences']) != count:
        raise ValueError('Calibration manifest row count mismatch')
    if set(tensors) != {f'input_ids.{i}' for i in range(count)}:
        raise ValueError('Noncontiguous or extra calibration keys')
    rows = []
    for i, info in enumerate(manifest['sequences']):
        t = tensors[f'input_ids.{i}']
        if t.dtype not in (torch.int32, torch.int64) or t.shape != (1, info['tokens']):
            raise ValueError(f'Invalid calibration shape/dtype: {i}')
        if not 1 <= t.shape[1] <= 8192 or t.min() < 0 or t.max() >= tokenizer.actual_vocab_size:
            raise ValueError(f'Invalid calibration length/token: {i}')
        rows.append(t.to(torch.long).contiguous())
    lengths = [t.shape[1] for t in rows]
    if (len(rows) != count or count != args['cal_rows'] or
            max(lengths) != args['cal_cols'] or sum(lengths) != manifest['total_tokens']):
        raise ValueError('Calibration accounting mismatch')
    return rows
