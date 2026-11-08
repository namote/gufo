"""CPU-only c03 builder. No dataset instructions are executed; no GPU/model imports."""
import collections
import hashlib
import json
from pathlib import Path
import random

import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file, save_file
from tokenizers import Tokenizer
from transformers.utils.chat_template_utils import _compile_jinja_template
from tools.rotquant.calibration import hermes_messages

OUT = Path('/research/c03')
OUT.mkdir(exist_ok=True)
SEED = 20261105
BUDGET = 1999872
tok = Tokenizer.from_file('/source/tokenizer.json')
template_text = Path('/source/chat_template.jinja').read_text()
template = _compile_jinja_template(template_text)
old_manifest = json.loads(Path('/old/MANIFEST.json').read_text())
used_old = {s: {str(d['id']) for d in v['documents']}
            for s, v in old_manifest['streams'].items() if 'documents' in v}
selected, rejected = [], collections.Counter()
seen_tasks, seen_text, repos = set(), set(), collections.Counter()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        while b := f.read(2**20):
            h.update(b)
    return h.hexdigest()


def text_content(value):
    if value is None:
        return ''
    if isinstance(value, str):
        return value
    if isinstance(value, list) and all(v.get('type') == 'text' for v in value):
        return '\n'.join(v['text'] for v in value)
    raise ValueError('Non-text content')


def normalize(raw):
    raw = json.loads(raw) if isinstance(raw, str) else raw
    result, pending = [], []
    for m in raw:
        role = m['role']
        if role not in ('system', 'user', 'assistant', 'tool'):
            raise ValueError('Unknown role')
        n = dict(role=role, content=text_content(m.get('content')))
        if m.get('reasoning_content'):
            n['reasoning_content'] = m['reasoning_content']
        if role == 'tool':
            if not pending:
                raise ValueError('Orphan tool result')
            ids = m.get('tool_call_ids') or ([m['tool_call_id']] if m.get('tool_call_id') else [pending[0]])
            for call_id in ids:
                if call_id not in pending:
                    raise ValueError('Mismatched tool result')
                pending.remove(call_id)
        elif pending:
            raise ValueError('Missing tool results')
        calls = []
        for c in m.get('tool_calls') or []:
            f = dict(c['function'])
            if isinstance(f['arguments'], str):
                f['arguments'] = json.loads(f['arguments'])
            if not isinstance(f['arguments'], dict) or not f.get('name'):
                raise ValueError('Invalid tool arguments')
            if c['id'] in pending:
                raise ValueError('Duplicate tool ID')
            pending.append(c['id'])
            calls.append(dict(type='function', function=f))
        if calls:
            n['tool_calls'] = calls
        n['_complete'] = not pending
        result.append(n)
    # A final submit action without a tool result is permitted only in the
    # unselected suffix. Prefix selection below always ends with no pending calls.
    return result


def encode_chat(ms, tools):
    return tok.encode(template.render(messages=ms, tools=tools,
                      add_generation_prompt=False, enable_thinking=True,
                      reasoning_effort='medium', preserve_thinking=True),
                      add_special_tokens=False).ids


def check_hermes_links(ms):
    # This source has no tool-call IDs. Preserve order and require one result per
    # call before another turn. A final assistant function-call target is valid
    # without a result; it is a completed training answer, not an agent rollout.
    pending = 0
    for m in ms:
        if m['role'] == 'tool':
            if pending == 0:
                raise ValueError('Orphan Hermes result')
            pending -= 1
        else:
            if pending:
                raise ValueError('Missing intermediate Hermes result')
            pending = len(m.get('tool_calls', []))
    return pending


def prefix(ms, tools, cap):
    # Preserve consecutive messages from the start. Never cut a tool call/result pair.
    best = None
    for end, m in enumerate(ms, 1):
        if not m.get('_complete', True) or m['role'] not in ('assistant', 'tool'):
            continue
        ids = encode_chat(ms[:end], tools)
        if len(ids) > cap:
            break
        best = (ids, end)
    return best


def add(ids, source, identity, **extra):
    # Full candidate token hash detects duplicates across sources.
    tensor = torch.tensor([ids], dtype=torch.int32)
    digest = hashlib.sha256(tensor.numpy().tobytes()).hexdigest()
    if digest in seen_text:
        rejected[source + ':duplicate'] += 1
        return False
    seen_text.add(digest)
    selected.append((tensor, dict(source=source, id=str(identity), tokens=len(ids),
                                 token_sha256=digest, **extra)))
    return True


provenance = {}
for source, target in [('smith', 262144), ('open', 327680)]:
    p = Path('/research/pinned') / (source + '.parquet')
    meta = json.loads(p.with_suffix('.json').read_text())
    assert sha(p) == meta['sha256']
    provenance[source] = meta
    rows = pq.read_table(p).to_pylist()
    random.Random(f'{SEED}:{source}').shuffle(rows)
    used = 0
    for row in rows:
        if used >= target - 2048:
            break
        if row['resolved'] not in (True, 1):
            rejected[source + ':unresolved'] += 1
            continue
        task = row['instance_id']
        repo = row.get('repo') or task.split('.')[0]
        if task in seen_tasks or repos[repo] >= 4:
            rejected[source + ':task_or_repo_cap'] += 1
            continue
        try:
            ms = normalize(row['messages'])
            tools = [json.loads(t) if isinstance(t, str) else t for t in row.get('tools', [])]
            original_tool_count = len(tools)
            used_functions = {c['function']['name'] for m in ms for c in m.get('tool_calls', [])}
            # Keep exact definitions for tools used by this trajectory. Remove
            # unused tool boilerplate rather than spending most of8K on it.
            tools = [t for t in tools if t.get('function', t)['name'] in used_functions]
            cap = min(8192 if source == 'open' or len(selected) % 2 else 4096, target - used)
            item = prefix(ms, tools or None, cap)
            if item is None or len(item[0]) < 2048:
                rejected[source + ':short_complete_prefix'] += 1
                continue
            ids, end = item
            first_assistant = next(i for i, m in enumerate(ms) if m['role'] == 'assistant')
            header_tokens = len(encode_chat(ms[:first_assistant], tools or None))
            body_tokens = len(ids) - header_tokens
            if body_tokens < 2048 or body_tokens * 3 < len(ids):
                rejected[source + ':insufficient_work_body'] += 1
                continue
            tool_groups = sum(bool(m.get('tool_calls')) for m in ms[:end])
            if tool_groups < 2:
                rejected[source + ':fewer_than_two_tool_groups'] += 1
                continue
        except (ValueError, TypeError, KeyError) as e:
            rejected[source + ':' + type(e).__name__] += 1
            continue
        if add(ids, source, task, repo=repo, resolved=True, messages_used=end,
               messages_available=len(ms), selection='complete_tool_group_prefix', cap=cap,
               tool_groups=tool_groups, original_tool_count=original_tool_count,
               header_tokens=header_tokens, work_body_tokens=body_tokens,
               retained_tool_count=len(tools), tool_definitions='exact_used_in_full_trajectory',
               missing_tool_ids='ordered_results_assumed' if source == 'open' else False):
            used += len(ids)
            seen_tasks.add(task)
            repos[repo] += 1
    assert used >= target - 8192, (source, used, target)
    print(source, used, 'tokens', flush=True)

# New Korean material excludes all documents used by c02, including unselected chunks.
for source, file, target in [
    ('ko_chat', '/sources/hermes-fc-ko/train-00000-of-00001.parquet', 131072),
    ('ko_wiki', '/sources/wiki-ko/train-00002-of-00003.parquet', 196608),
]:
    provenance[source] = dict(path=file, sha256=sha(file))
    columns = ['id', 'ko_conversations'] if source == 'ko_chat' else ['id', 'text']
    rows = pq.read_table(file, columns=columns).to_pylist()
    random.Random(f'{SEED}:{source}').shuffle(rows)
    used = 0
    for row in rows:
        if used >= target - 8192:
            break
        if str(row['id']) in used_old[source]:
            continue
        try:
            if source == 'ko_chat':
                ms, tools = hermes_messages(row['ko_conversations'])
                terminal_calls = check_hermes_links(ms)
                ids = encode_chat(ms, tools)
                if not 2048 <= len(ids) <= min(8192, target - used):
                    continue
                extra = dict(selection='whole_conversation',
                             terminal_target_calls_without_result=terminal_calls,
                             tool_link_check='ordered_result_count_no_ids')
            else:
                ids = tok.encode(row['text'], add_special_tokens=False).ids
                if len(ids) < 4096:
                    continue
                ids = ids[:min(8192, target - used)]
                extra = dict(selection='continuous_document_prefix')
            if add(ids, source, row['id'], **extra):
                used += len(ids)
        except (ValueError, SyntaxError, TypeError, KeyError):
            rejected[source + ':invalid_chat'] += 1
    assert used >= target - 8192, (source, used, target)
    print(source, used, 'tokens', flush=True)

# Retain the original short mixture, sampled proportionally by source stream.
old = load_file('/task/c02-input-ids.safetensors')['input_ids']
chunks = json.loads(Path('/old/chunks.json').read_text())['chunks']
assert old.shape == (3906, 512) and len(chunks) == 3906
remaining = (BUDGET - sum(x.numel() for x, _ in selected)) // 512
groups = collections.defaultdict(list)
for i, c in enumerate(chunks):
    groups[c['stream']].append(i)
counts = {s: int(len(v) * remaining / len(chunks)) for s, v in groups.items()}
for s in sorted(groups, key=lambda s: -(len(groups[s]) * remaining / len(chunks) - counts[s]))[:remaining - sum(counts.values())]:
    counts[s] += 1
for stream, indexes in sorted(groups.items()):
    random.Random(f'{SEED}:old:{stream}').shuffle(indexes)
    for i in indexes[:counts[stream]]:
        assert add(old[i].tolist(), 'c02:' + stream, chunks[i]['id'], old_row=i,
                   selection='unchanged_512_token_row')

random.Random(SEED).shuffle(selected)
# The first reference-error rows cover domains/lengths, rather than all being512.
front = []
for domain in ['smith', 'open', 'ko_wiki', 'ko_chat', 'c02:multi']:
    i = next(i for i, (_, info) in enumerate(selected) if info['source'] == domain)
    front.append(selected.pop(i))
selected = front + selected
tensors = {f'input_ids.{i}': t for i, (t, _) in enumerate(selected)}
path = OUT / 'c03-input-ids.safetensors'
assert not path.exists(), 'Do not overwrite a frozen candidate'
save_file(tensors, str(path))
infos = [info for _, info in selected]
source_tokens = collections.Counter()
for info in infos:
    source_tokens[info['source']] += info['tokens']
manifest = dict(format='gufo.ragged-calibration.v1', variable_length=True, seed=SEED,
                rows=len(infos), max_columns=max(i['tokens'] for i in infos),
                total_tokens=sum(i['tokens'] for i in infos), budget=BUDGET,
                sha256=sha(path), tokenizer_sha256=sha('/source/tokenizer.json'),
                template_sha256=sha('/source/chat_template.jinja'),
                template_options=dict(enable_thinking=True, reasoning_effort='medium', preserve_thinking=True),
                source_tokens=dict(source_tokens), source_provenance=provenance,
                c02_sha256=sha('/task/c02-input-ids.safetensors'),
                builder_sha256=sha(__file__), hermes_parser_sha256=sha('/gufo/tools/rotquant/calibration.py'),
                rejection_counts=dict(rejected), sequences=infos,
                qualification='CPU input candidate only; long-row GPU memory/correctness pending',
                evaluation_exclusion='No benchmark files used. Future holdout must exclude these task/document IDs; semantic overlap with all public benchmarks is unverified.')
path.with_suffix('.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
print(json.dumps({k: manifest[k] for k in ['rows', 'max_columns', 'total_tokens', 'source_tokens']}), flush=True)
