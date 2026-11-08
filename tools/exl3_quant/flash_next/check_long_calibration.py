"""CPU input contract checks, including rejection of corrupt calibration inputs."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace

import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from ragged_calibration import load_calibration, sync_calibration_metadata

root = Path('/research/c03')
path = root / 'c03-input-ids.safetensors'
manifest = json.loads(path.with_suffix('.json').read_text())
tokenizer = SimpleNamespace(actual_vocab_size=Tokenizer.from_file('/source/tokenizer.json').get_vocab_size())
args = dict(cal_data=str(path), cal_rows=manifest['rows'], cal_cols=manifest['max_columns'], in_dir='/source')
rows = load_calibration(args, tokenizer)
original = load_file('/task/c02-input-ids.safetensors')['input_ids']
assert hashlib.sha256(original.numpy().tobytes()).hexdigest() == '40d5d6535aa455a42514a0f31ccf40ba3863ad51a864968cd289ad4dc2b4bcc4'
assert manifest['tokenizer_sha256'] == '0997f410c57a1f4e53b09e4be8f4a172d90edd9564368fb0847030937229b9f3'
for row, info in zip(rows, manifest['sequences'], strict=True):
    if info['source'].startswith('c02:'):
        assert torch.equal(row[0], original[info['old_row']].long())
    if info['source'] in ('smith', 'open'):
        assert info['tool_groups'] >= 2
        assert info['work_body_tokens'] >= 2048
        assert info['work_body_tokens'] * 3 >= info['tokens']
assert len({r.shape[1] for r in rows}) > 3
assert 0 <= manifest['budget'] - sum(r.numel() for r in rows) < 512
assert sum(manifest['source_tokens'].values()) == manifest['total_tokens']

checks = []
with tempfile.TemporaryDirectory() as directory:
    pack = Path(directory)
    original_config = dict(model_type='fixture', quantization_config=dict(
        bits=4.03597648180518, calibration=dict(rows=2309, cols=8192)))
    original_quant = dict(original_config['quantization_config'],
                         tensor_storage={'fixture.weight': {'stored_tensors': ['fixture.trellis']}})
    (pack / 'config.json').write_text(json.dumps(original_config))
    (pack / 'quantization_config.json').write_text(json.dumps(original_quant))
    sync_calibration_metadata(pack, manifest)
    config = json.loads((pack / 'config.json').read_text())
    quant = json.loads((pack / 'quantization_config.json').read_text())
    cal = config['quantization_config'].pop('calibration')
    assert quant.pop('calibration') == cal
    del original_config['quantization_config']['calibration']
    del original_quant['calibration']
    assert config == original_config and quant == original_quant
    assert cal['total_tokens'] == sum(int(k) * n for k, n in cal['length_counts'].items())
    assert 'cols' not in cal and cal['sha256'] == manifest['sha256']
    checks.append('both_final_configs_match_and_preserve_tensor_storage')
    p = Path(directory) / 'candidate.safetensors'
    # Same per-row tensor representation used by upstream state checkpoints.
    cp = Path(directory) / 'checkpoint.safetensors'
    save_file({f'tensor.{i}': row for i, row in enumerate(rows)}, str(cp))
    reread = load_file(str(cp))
    assert all(torch.equal(row, reread[f'tensor.{i}']) for i, row in enumerate(rows))
    checks.append('variable_length_checkpoint_roundtrip')
    tensors = load_file(str(path))
    for case in ['negative_id', 'out_of_vocab', 'floating_dtype', 'missing_key',
                 'wrong_length', 'wrong_total', 'wrong_hash', 'wrong_tokenizer']:
        data = dict(tensors)
        meta = copy.deepcopy(manifest)
        if case == 'negative_id':
            data['input_ids.0'] = data['input_ids.0'].clone()
            data['input_ids.0'][0, 0] = -1
        elif case == 'out_of_vocab':
            data['input_ids.0'] = data['input_ids.0'].clone()
            data['input_ids.0'][0, 0] = tokenizer.actual_vocab_size
        elif case == 'floating_dtype':
            data['input_ids.0'] = data['input_ids.0'].float()
        elif case == 'missing_key':
            del data['input_ids.1']
        elif case == 'wrong_length':
            meta['sequences'][0]['tokens'] -= 1
        elif case == 'wrong_total':
            meta['total_tokens'] += 1
        elif case == 'wrong_tokenizer':
            meta['tokenizer_sha256'] = '0' * 64
        save_file(data, str(p))
        meta['sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
        if case == 'wrong_hash':
            meta['sha256'] = '0' * 64
        p.with_suffix('.json').write_text(json.dumps(meta))
        try:
            load_calibration({**args, 'cal_data': str(p)}, tokenizer)
        except ValueError:
            checks.append(case)
        else:
            raise AssertionError('Corrupt candidate accepted: ' + case)

result = dict(status='cpu_input_contract_pass', rows=len(rows),
              total_tokens=manifest['total_tokens'], sha256=manifest['sha256'],
              checks=checks, gpu_executed=False,
              limitation='No long-row model/Hessian memory or numerical qualification yet')
(root / 'CPU-CHECK.json').write_text(json.dumps(result, indent=2))
print(json.dumps(result), flush=True)
