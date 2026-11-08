"""CPU-only research sample audit. Does not create calibration input or run model code."""
import collections
import json
from pathlib import Path
import statistics
import pyarrow.parquet as pq
from tokenizers import Tokenizer

out = Path('/research')
tok = Tokenizer.from_file('/tokenizer.json')

def content(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list):
        return '\n'.join(content(v) for v in value)
    if isinstance(value, dict):
        if value.get('type') == 'text':
            return value.get('text', '')
        return json.dumps(value, ensure_ascii=False)
    return str(value)

def audit(messages, tools=None):
    if isinstance(messages, str):
        messages = json.loads(messages)
    if not isinstance(messages, list):
        raise ValueError('messages are not a list')
    # Content-only token lengths; no duplicate English translations, evaluation patches,
    # metadata, or target chat-template framing. Tool definitions/results are retained.
    parts = [content(tools)] if tools else []
    roles = collections.Counter()
    for m in messages:
        roles[m.get('role', m.get('from', 'unknown'))] += 1
        for k in ('content', 'value', 'reasoning_content', 'tool_calls'):
            if m.get(k):
                parts.append(content(m[k]))
    text = '\n'.join(parts)
    tokens = len(tok.encode(text, add_special_tokens=False).ids)
    return dict(tokens=tokens, messages=len(messages), roles=dict(roles),
                hangul_chars=sum('\uac00' <= c <= '\ud7a3' for c in text))

report = dict(scope='nonrandom bounded research samples, not full-corpus quality or length distribution',
              token_count='Flash-Next tokenizer; content plus tools, without chat-template overhead',
              sources={})
for f in sorted(out.glob('*--rows.json')):
    data = json.loads(f.read_text())
    records = []
    for wrapped in data['rows']:
        assert not wrapped.get('truncated_cells'), 'Reject truncated preview'
        row = wrapped['row']
        result = audit(row['messages'], row.get('tools'))
        result.update(row_idx=wrapped['row_idx'], resolved=row.get('resolved'),
                      id=row.get('trajectory_id', row.get('traj_id', row.get('custom_id', row.get('id')))))
        records.append(result)
    report['sources'][f.stem] = dict(records=records, sample='first eight API rows')

local = [
    ('local-ko-hermes', '/sources/hermes-fc-ko/train-00000-of-00001.parquet', 'ko_conversations', 12000),
    ('local-en-hermes', '/sources/hermes-reasoning-tool-use/train-00000-of-00001.parquet', 'conversations', 2000),
    ('local-ko-wiki', '/sources/wiki-ko/train-00002-of-00003.parquet', 'text', 3000),
]
for name, filename, column, limit in local:
    records = []
    for batch in pq.ParquetFile(filename).iter_batches(batch_size=128, columns=[column]):
        for row in batch.to_pylist():
            value = row[column]
            if column == 'text':
                value = [{'role': 'document', 'content': value}]
            result = audit(value)
            result['row_idx'] = len(records)
            records.append(result)
            if len(records) >= limit:
                break
        if len(records) >= limit:
            break
    report['sources'][name] = dict(records=records, sample=f'first up to {limit} local rows')

for name, data in report['sources'].items():
    vals = [r['tokens'] for r in data['records']]
    data['summary'] = dict(n=len(vals), minimum=min(vals), median=statistics.median(vals), maximum=max(vals),
                           at_least_2k=sum(v>=2048 for v in vals), at_least_4k=sum(v>=4096 for v in vals),
                           at_least_8k=sum(v>=8192 for v in vals), at_least_16k=sum(v>=16384 for v in vals))
    print(name, json.dumps(data['summary']), flush=True)
(out/'CONTENT-LENGTH-AUDIT.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
