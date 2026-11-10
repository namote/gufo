"""Prepare model-bound runtime projection profiles; never start a server/GPU job.

Import pinned datasets with source notices, split before tuning, and fit candidate
unit directions from Gufo's native per-prompt activation collector. Candidates
are not installed automatically: held-out behavior and quality must be checked.
"""
import argparse
import hashlib
import json
from pathlib import Path
import tarfile

SCHEMA = 'gufo-abliteration-v1'


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def import_data(archive, destination):
    """Read exactly the pinned prompt files, never execute/extract archive paths."""
    archive, destination = Path(archive), Path(destination)
    if destination.exists():
        raise ValueError('use a new dataset destination; existing results are preserved')
    names = ('harmful_collect.jsonl', 'harmless_collect.jsonl',
             'harmful_heldout.jsonl', 'harmless_heldout.jsonl', 'SOURCES.md')
    contents = {}
    with tarfile.open(archive) as source:
        for name in names:
            matches = [m for m in source.getmembers()
                       if m.name.endswith('/tools/ablit/prompts/' + name)]
            if len(matches) != 1 or not matches[0].isfile() or matches[0].size > 4 << 20:
                raise ValueError('missing or invalid prompt archive member: ' + name)
            contents[name] = source.extractfile(matches[0]).read()
    # Deduplicate by text across groups/splits, reserving heldout first.
    seen, records = set(), []
    for name in sorted(names, key=lambda n: ('heldout' not in n, n)):
        if not name.endswith('.jsonl'):
            continue
        group = 'refusal' if name.startswith('harmful_') else 'ordinary'
        for row in map(json.loads, contents[name].splitlines()):
            text = row['text'].strip()
            identity = sha(text.encode())
            if identity in seen:
                continue
            seen.add(identity)
            # Heldout is never used for collection or candidate selection.
            split = 'evaluation' if 'heldout' in name else (
                'tuning' if int(identity[:8], 16) % 5 == 0 else 'collection')
            for thinking in (False, True):
                records.append(dict(id=identity + ('-think' if thinking else '-plain'),
                                    text=text, group=group, split=split,
                                    thinking=thinking, source=row['source']))
    destination.mkdir(parents=True)
    for name, content in contents.items():
        (destination / name).write_bytes(content)
    for split in ('collection', 'tuning', 'evaluation'):
        path = destination / (split + '.jsonl')
        path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n'
                                for r in records if r['split'] == split))
    manifest = dict(schema='gufo-abliteration-data-v1', archive=str(archive.resolve()),
                    archive_sha256=sha(archive.read_bytes()),
                    sources={n: sha(v) for n, v in contents.items()},
                    counts={s: sum(r['split'] == s for r in records)
                            for s in ('collection', 'tuning', 'evaluation')},
                    licenses='See SOURCES.md; mixed licenses, including CC BY-NC 4.0. '
                             'Data is not relicensed under Gufo code license.')
    save(destination / 'MANIFEST.json', manifest)
    return manifest


def fit(collection, output, strengths=(0.5, 1.0)):
    import numpy as np
    collection, output = Path(collection), Path(output)
    manifest = json.loads((collection / 'manifest.json').read_text())
    if manifest['split'] != 'collection':
        raise ValueError('fit only collection split, never tuning/evaluation')
    hidden, layers = manifest['hidden'], manifest['layers']
    sums = {g: np.zeros((layers, hidden), dtype=np.float64)
            for g in ('refusal', 'ordinary')}
    counts = dict.fromkeys(sums, 0)
    for row in manifest['records']:
        receipt = json.loads((collection / (row['id'] + '.json')).read_text())
        raw = (collection / (row['id'] + '.f32')).read_bytes()
        if sha(raw) != receipt['activation_sha256']:
            raise ValueError('activation integrity mismatch')
        values = np.frombuffer(raw, dtype='<f4').reshape(layers, hidden)
        if not np.isfinite(values).all():
            raise ValueError('nonfinite activation')
        sums[row['group']] += values
        counts[row['group']] += 1
    if min(counts.values()) < 2:
        raise ValueError('both prompt classes need at least two samples')
    means = {g: sums[g] / counts[g] for g in sums}
    directions = means['refusal'] - means['ordinary']
    norms = np.linalg.norm(directions, axis=1)
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'statistics.npz', **means, directions=directions, norms=norms)
    # A bounded candidate screen. Selection must use tuning, final check evaluation.
    selected = sorted(set(int(round((layers - 1) * f)) for f in (.4, .6, .8)))
    for layer in selected:
        if not np.isfinite(norms[layer]) or norms[layer] < 1e-12:
            continue
        direction = (directions[layer] / norms[layer]).astype(np.float32)
        for strength in strengths:
            if not 0 <= strength <= 4:
                raise ValueError('strength must be in [0, 4]')
            # Restrict to a neighborhood of the candidate layer initially.
            weights = [float(strength if abs(i - layer) <= 3 else 0)
                       for i in range(layers)]
            save(output / f'layer{layer}-strength{strength:g}.json', dict(
                schema=SCHEMA, model_identity=manifest['model_identity'],
                hidden=hidden, layers=layers, placement='writer-before-hc',
                direction=direction.tolist(), attention=weights, mlp=weights,
                status='candidate-not-qualified', collection_sha256=sha(
                    (collection / 'manifest.json').read_bytes()),
                source_layer=layer, strength=strength))
    save(output / 'SUMMARY.json', dict(counts=counts, candidate_layers=selected,
                                     status='requires-tuning-and-heldout-validation'))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    imp = sub.add_parser('import-data')
    imp.add_argument('--archive', required=True)
    imp.add_argument('--output', required=True)
    fitting = sub.add_parser('fit')
    fitting.add_argument('--collection', required=True)
    fitting.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'import-data':
        print(json.dumps(import_data(args.archive, args.output), indent=2))
    else:
        fit(args.collection, args.output)


if __name__ == '__main__':
    main()
