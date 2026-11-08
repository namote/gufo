"""Select whole c03 rows for a separate, unapplied quarter-token candidate."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import struct


def length_bin(tokens):
    for limit in (512, 2048, 4096, 8192):
        if 0 < tokens <= limit:
            return str(limit)
    raise ValueError('Unsupported calibration length')


def choose_group(items, seed):
    """Nearest nonempty whole-row subset to one quarter of this stratum's tokens."""
    ordered = sorted(items, key=lambda pair: hashlib.sha256(
        f'{seed}|{pair[0]}|{pair[1]["id"]}|{pair[1]["token_sha256"]}'.encode()).digest())
    weights = [row['tokens'] for _, row in ordered]
    total = sum(weights)
    if len(set(weights)) == 1:
        count = max(1, (len(weights)+2)//4)
        return [index for index, _ in ordered[:count]]
    reachable = 1
    history = []
    for weight in weights:
        history.append(reachable)
        reachable |= reachable << weight
    # Keep at least one sample even when a rare stratum has a single long row.
    reachable &= ~1
    floor = total//4
    lower = (reachable & ((1 << (floor+1))-1)).bit_length()-1
    ceil = (total+3)//4
    tail = reachable >> ceil
    upper = ceil + ((tail & -tail).bit_length()-1) if tail else -1
    best = min((x for x in (lower, upper) if x > 0), key=lambda x: (abs(4*x-total), x))
    selected = []
    for j in range(len(weights)-1, -1, -1):
        if not (history[j] >> best) & 1:
            selected.append(ordered[j][0])
            best -= weights[j]
    assert best == 0 and selected
    return selected


def select(manifest, seed):
    groups = defaultdict(list)
    for index, row in enumerate(manifest['sequences']):
        groups[(row['source'], length_bin(row['tokens']))].append((index, row))
    selected = sorted(index for group in sorted(groups)
                      for index in choose_group(groups[group], seed))
    assert len(selected) == len(set(selected))
    chosen = set(selected)
    total = sum(manifest['sequences'][i]['tokens'] for i in selected)
    strata = []
    sources = defaultdict(lambda: [0, 0])
    for (source, bucket), items in sorted(groups.items()):
        before = sum(row['tokens'] for _, row in items)
        after = sum(row['tokens'] for i, row in items if i in chosen)
        count = sum(i in chosen for i, _ in items)
        assert count >= 1
        strata.append(dict(source=source, length_upper_bound=int(bucket),
                           original_rows=len(items), selected_rows=count,
                           original_tokens=before, selected_tokens=after,
                           ideal_quarter_tokens=before/4, token_deviation=after-before/4))
        sources[source][0] += before
        sources[source][1] += after
    original_total = sum(row['tokens'] for row in manifest['sequences'])
    assert original_total == manifest['total_tokens']
    return dict(format='gufo.capture-subset-candidate.v1', status='selected_not_applied',
                seed=seed, selected_indices=selected, original_rows=len(manifest['sequences']),
                selected_rows=len(selected), original_tokens=original_total,
                selected_tokens=total, selected_fraction=total/original_total,
                ideal_tokens=original_total/4, token_deviation=total-original_total/4,
                strata=strata, source_balance={source: dict(
                    original_tokens=a, selected_tokens=b,
                    original_share=a/original_total, selected_share=b/total,
                    share_difference_pp=100*(b/total-a/original_total))
                    for source, (a, b) in sorted(sources.items())})


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--seed', type=int, default=20261105)
    args = parser.parse_args()
    metadata_path = args.source.with_suffix('.json')
    metadata_bytes = metadata_path.read_bytes()
    manifest = json.loads(metadata_bytes)
    raw = args.source.read_bytes()
    assert hashlib.sha256(raw).hexdigest() == manifest['sha256']
    result = select(manifest, args.seed)
    header_length = struct.unpack('<Q', raw[:8])[0]
    header = json.loads(raw[8:8+header_length])
    rows = []
    for index in result['selected_indices']:
        row = manifest['sequences'][index]
        info = header[f'input_ids.{index}']
        assert info['dtype'] == 'I32' and info['shape'] == [1, row['tokens']]
        begin, end = info['data_offsets']
        assert end-begin == row['tokens']*4
        payload = raw[8+header_length+begin:8+header_length+end]
        assert hashlib.sha256(payload).hexdigest() == row['token_sha256']
        rows.append(dict(original_index=index, source=row['source'], id=row['id'],
                         tokens=row['tokens'], token_sha256=row['token_sha256']))
    result.update(source=str(args.source.resolve()), source_sha256=manifest['sha256'],
                  metadata_sha256=hashlib.sha256(metadata_bytes).hexdigest(),
                  selector_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  rows=rows, production_changed=False, quality_validated=False,
                  limitations=[
                      'Source labels and length bins do not prove semantic/topic coverage.',
                      'Rare strata retain at least one whole row; exact 25% per stratum is not always possible.',
                      'Unselected c03 rows are not an independent holdout for the full-calibration baseline.',
                      'Do not replace input_ids on a full-state checkpoint; explicit index mapping and application history are required.'])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x') as stream:
        json.dump(result, stream, ensure_ascii=False, indent=2)
    print(json.dumps({k: result[k] for k in ('status','selected_rows','selected_tokens',
          'selected_fraction','token_deviation')}, ensure_ascii=False))


if __name__ == '__main__':
    main()
