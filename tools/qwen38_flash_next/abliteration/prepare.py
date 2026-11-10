"""Prepare model-bound runtime projection profiles; never start a server/GPU job.

Import pinned datasets with source notices, split before tuning, and fit
candidate unit directions from Gufo's native per-prompt activation collector.
`fit` writes gufo-abliteration-v2 candidates: every layer's direction plus
default settings. `make-profile` writes one v2 profile with chosen defaults.
Candidates are not installed automatically: held-out behavior and quality must
be checked.

Strength curves (`make-profile --attention/--mlp`, search.py) follow Heretic:
for layer i, distance d = |i - peak|,

    weight(i) = max_w + (min_w - max_w) * d / width   if d <= width
    weight(i) = 0                                      if d >  width

so the weight is max_w at the peak and falls linearly to min_w at distance
`width`, and is zero beyond. `width` > 0; `peak` may be fractional. Weights
must lie in [0, 4].
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import tarfile

SCHEMA = 'gufo-abliteration-v2'
MAX_WEIGHT = 4.0
# Each model's collector readout names the placement its runtime applies:
# Flash-Next before HC injection, Qwen3.8-27B before the plain residual add.
PLACEMENTS = {
    'last-prompt-token-post-layer-hc-mean-fp32-v1': 'writer-before-hc',
    'last-prompt-token-post-layer-fp32-v1': 'writer-before-residual',
}


def placement(manifest):
    """The capture's placement; a recorded one must agree with its readout."""
    expected = PLACEMENTS.get(manifest.get('readout'))
    if expected is None:
        raise ValueError('unknown capture readout; fit only collector captures')
    if manifest.get('placement', expected) != expected:
        raise ValueError('capture placement does not match its readout')
    return expected


def sha(data):
    return hashlib.sha256(data).hexdigest()


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')
    temporary.replace(path)


def float32_text(x):
    """A float that round-trips the float32 nearest x in at most 9 digits."""
    return float('%.9g' % x)


def save_profile(path, profile):
    """Compact JSON: a 27B v2 profile (64 x 5120 directions) is about 4.5 MB,
    under the runtime's 16 MiB cap; indentation would nearly double it."""
    profile = dict(profile)
    profile['directions'] = [[float32_text(x) for x in row]
                             for row in profile['directions']]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(profile, ensure_ascii=False,
                                    separators=(',', ':')) + '\n')
    temporary.replace(path)


def curve_weights(layers, max_w, peak, min_w, width):
    """Heretic's per-layer strength curve (see the module docstring)."""
    if not width > 0:
        raise ValueError('curve width must be positive')
    weights = []
    for i in range(layers):
        distance = abs(i - peak)
        weight = (max_w + (min_w - max_w) * distance / width
                  if distance <= width else 0.0)
        if distance <= width:  # rounding never leaves the endpoint range
            weight = min(max(weight, min(min_w, max_w)), max(min_w, max_w))
        weights.append(float(weight))
    return check_weights(weights, layers)


def check_weights(weights, layers):
    if len(weights) != layers:
        raise ValueError(f'need {layers} weights, got {len(weights)}')
    for w in weights:
        if not (math.isfinite(w) and 0 <= w <= MAX_WEIGHT):
            raise ValueError('weights must be finite and in [0, 4]')
    return [float(w) for w in weights]


def parse_weights(spec, layers):
    """`2.0` (every layer), `w0,w1,...` (one per layer), or a curve
    `max_w=2,peak=36,min_w=0.5,width=10`."""
    spec = str(spec).strip()
    if '=' in spec:
        params = {}
        for part in spec.split(','):
            key, _, value = part.partition('=')
            params[key.strip()] = float(value)
        if set(params) != {'max_w', 'peak', 'min_w', 'width'}:
            raise ValueError('a curve needs exactly max_w, peak, min_w, width')
        return curve_weights(layers, **params)
    values = [float(v) for v in spec.split(',')]
    return check_weights(values * layers if len(values) == 1 else values, layers)


def make_profile(base, direction_layer, attention, mlp, **extra):
    """A v2 profile from `base` (model_identity, hidden, layers, placement,
    directions) with the given defaults; validated like the runtime loader."""
    hidden, layers = base['hidden'], base['layers']
    directions = base['directions']
    if len(directions) != layers:
        raise ValueError('need one direction per layer')
    for row in directions:
        norm = math.sqrt(sum(float(x) ** 2 for x in row))
        if len(row) != hidden or not math.isfinite(norm) or abs(norm - 1) > 1e-4:
            raise ValueError('every direction must be a unit vector of hidden floats')
    if not (isinstance(direction_layer, int) and 0 <= direction_layer < layers):
        raise ValueError(f'direction_layer must be in [0, {layers})')
    profile = dict(schema=SCHEMA, model_identity=base['model_identity'],
                   hidden=hidden, layers=layers, placement=base['placement'],
                   directions=directions, direction_layer=direction_layer,
                   attention=check_weights(attention, layers),
                   mlp=check_weights(mlp, layers))
    profile.update(extra)
    return profile


def load_base(fit_dir=None, profile=None):
    """Directions and binding from a `fit` output or an existing v2 profile."""
    if profile is not None:
        base = json.loads(Path(profile).read_text())
        if base.get('schema') != SCHEMA:
            raise ValueError('a base profile must be gufo-abliteration-v2')
        return base
    import numpy as np
    fit_dir = Path(fit_dir)
    summary = json.loads((fit_dir / 'SUMMARY.json').read_text())
    for key in ('model_identity', 'hidden', 'layers', 'placement'):
        if key not in summary:
            raise ValueError('fit output predates v2; rerun fit')
    directions = np.load(fit_dir / 'statistics.npz')['directions']
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-12).any():
        raise ValueError('a layer has no usable direction')
    return dict(summary, directions=(directions / norms).astype(np.float32).tolist())


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
    # Geometry and placement come from the capture, never from one model.
    hidden, layers = manifest['hidden'], manifest['layers']
    site = placement(manifest)
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
    unusable = [i for i, n in enumerate(norms) if not (np.isfinite(n) and n >= 1e-12)]
    if unusable:
        # A v2 profile needs every layer's direction.
        raise ValueError(f'layers {unusable} have no usable refusal direction')
    output.mkdir(parents=True, exist_ok=False)
    np.savez(output / 'statistics.npz', **means, directions=directions, norms=norms)
    collection_sha = sha((collection / 'manifest.json').read_bytes())
    binding = dict(model_identity=manifest['model_identity'], hidden=hidden,
                   layers=layers, placement=site)
    base = dict(binding, directions=(directions / norms[:, None]).astype(
        np.float32).tolist())
    # A bounded candidate screen. Selection must use tuning, final check
    # evaluation. Every candidate carries all directions, so a server can try
    # another direction layer per request.
    selected = sorted(set(int(round((layers - 1) * f)) for f in (.4, .6, .8)))
    for layer in selected:
        for strength in strengths:
            if not 0 <= strength <= MAX_WEIGHT:
                raise ValueError('strength must be in [0, 4]')
            # Restrict to a neighborhood of the candidate layer initially.
            weights = [float(strength if abs(i - layer) <= 3 else 0)
                       for i in range(layers)]
            save_profile(output / f'layer{layer}-strength{strength:g}.json',
                         make_profile(base, layer, weights, weights,
                                      status='candidate-not-qualified',
                                      collection_sha256=collection_sha,
                                      strength=strength))
    save(output / 'SUMMARY.json', dict(binding, counts=counts,
                                     candidate_layers=selected,
                                     collection_sha256=collection_sha,
                                     status='requires-tuning-and-heldout-validation'))


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    imp = sub.add_parser('import-data')
    imp.add_argument('--archive', required=True)
    imp.add_argument('--output', required=True)
    fitting = sub.add_parser('fit')
    fitting.add_argument('--collection', required=True)
    fitting.add_argument('--output', required=True)
    make = sub.add_parser('make-profile', help='write one v2 profile with chosen defaults')
    source = make.add_mutually_exclusive_group(required=True)
    source.add_argument('--fit', help='a fit output directory (statistics.npz)')
    source.add_argument('--profile', help='an existing v2 profile to re-default')
    make.add_argument('--direction-layer', type=int, required=True)
    make.add_argument('--attention', required=True,
                      help='2.0 | w0,...,wN-1 | max_w=..,peak=..,min_w=..,width=..')
    make.add_argument('--mlp', required=True, help='as --attention')
    make.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'import-data':
        print(json.dumps(import_data(args.archive, args.output), indent=2))
    elif args.command == 'fit':
        fit(args.collection, args.output)
    else:
        base = load_base(args.fit, args.profile)
        layers = base['layers']
        output = Path(args.output)
        if output.exists():
            raise SystemExit('use a new output path; existing profiles are preserved')
        save_profile(output, make_profile(
            base, args.direction_layer, parse_weights(args.attention, layers),
            parse_weights(args.mlp, layers), status='candidate-not-qualified'))
        print(output)


if __name__ == '__main__':
    main()
