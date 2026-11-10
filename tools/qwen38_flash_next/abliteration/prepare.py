"""Prepare model-bound runtime projection profiles; never start a server/GPU job.

Import pinned datasets with source notices, split before tuning, and fit
candidate unit directions from Gufo's native per-prompt activation collector.
`fit` writes gufo-abliteration-v2 candidates: every layer's direction plus
default settings (writer candidates per layer and strength, and one
residual-stream candidate per layer: residual 1 on every layer, writers 0).
`make-profile` writes one v2 profile with chosen defaults. Candidates are not
installed automatically: held-out behavior and quality must be checked.

Writer strengths (`attention`, `mlp`) project a layer's attention/GDN and
MLP outputs before they reach the residual; `residual` strengths project the
post-layer residual itself, where the collector measured the directions, so
strength 1 leaves no component along the direction there.

Strength curves (`make-profile --attention/--mlp/--residual`, search.py)
follow Heretic:
for layer i, distance d = |i - peak|,

    weight(i) = max_w + (min_w - max_w) * d / width   if d <= width
    weight(i) = 0                                      if d >  width

so the weight is max_w at the peak and falls linearly to min_w at distance
`width`, and is zero beyond. `width` > 0; `peak` may be fractional. Weights
must lie in [0, 4].

Direction kinds, subspaces and norm preservation (all optional v2 members;
a profile without them is rank 1, mean difference, no norm preservation):

* `mean_diff`: r = mean(refusal) - mean(ordinary), normalized (`directions`).
* `projected` (projected abliteration, arXiv 2512.13655; Heretic's default):
  r minus its component along the normalized ordinary mean h,
  p = r - (r . h) h, normalized (`projected_directions`, row 0). The ordinary
  mean is what every prompt shares, so removing p instead of r leaves that
  shared part of the activations alone; `fit` makes it the default.
* rank k (concept cones, Wollschlaeger et al., ICML 2025; multi-directional
  refusal via a self-organizing map, arXiv 2603.22061): per layer and kind,
  k orthonormal rows, the kind's single direction first, from the per-prompt
  deviations x_i - mean(ordinary) of the refusal prompts. `--method pca`
  takes their top right singular vectors; `--method som` trains a small
  batch self-organizing map on them (seeded, deterministic) and takes the
  prototypes most aligned with r. Gram-Schmidt (FP64, twice) orthonormalizes
  the candidates after the first row; for `projected` every row is also
  orthogonal to h. Rows are stored as base64 little-endian float32
  (`subspace_directions[l]`: rows 1..k-1 of mean_diff; `projected_directions
  [l]`: rows 0..k-1 of projected), so a rank-8 27B profile stays near 31 MB.
* `preserve_norm`: the runtime rescales each projected row back to its norm
  before the projection (the activation-space analogue of norm-preserving
  biprojected abliteration, which edits weights instead).
"""
import argparse
import base64
import hashlib
import json
import math
import operator
from pathlib import Path
import sys
import tarfile

SCHEMA = 'gufo-abliteration-v2'
MAX_WEIGHT = 4.0
# The runtime's subspace cap (kAbliterationMaxDirections).
MAX_RANK = 8
KINDS = ('mean_diff', 'projected')
METHODS = ('pca', 'som')
# Profile members of the direction kinds, subspace and norm options.
SUBSPACE_MEMBERS = ('rank', 'subspace_directions', 'projected_directions',
                    'subspace_method', 'direction_kind', 'direction_count',
                    'preserve_norm')
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


def encode_rows(rows):
    """Base64 of little-endian float32 values (a flat or nested sequence)."""
    import numpy as np
    return base64.b64encode(np.asarray(rows, dtype='<f4').tobytes()).decode()


def decode_rows(text, rows, hidden, name):
    """`rows` lists of `hidden` floats from canonical base64 float32."""
    from array import array
    if not isinstance(text, str):
        raise ValueError(f'{name} must be a base64 string')
    try:
        raw = base64.b64decode(text, validate=True)
    except ValueError:
        raise ValueError(f'{name} is not base64') from None
    if len(raw) != rows * hidden * 4 or base64.b64encode(raw).decode() != text:
        raise ValueError(f'{name} must hold {rows} x {hidden} float32 values')
    values = array('f')
    values.frombytes(raw)
    if sys.byteorder == 'big':
        values.byteswap()
    return [values[i * hidden:(i + 1) * hidden] for i in range(rows)]


def check_orthonormal(rows, name):
    """Unit rows, pairwise orthogonal, as the runtime loader checks (NumPy
    when available; search.py stays standard library)."""
    try:
        import numpy as np
    except ImportError:
        np = None
    if np is not None:
        block = np.array([list(row) for row in rows], dtype=np.float64)
        if not np.isfinite(block).all() or (np.abs(block) > 1).any():
            raise ValueError(f'{name} must be finite and within [-1, 1]')
        gram = block @ block.T
        if (np.abs(np.diag(gram) - 1) > 1e-4).any():
            raise ValueError(f'{name} rows must be unit vectors')
        if (np.abs(gram - np.diag(np.diag(gram))) > 1e-4).any():
            raise ValueError(f'{name} rows must be orthonormal')
        return
    for a, row in enumerate(rows):
        if not all(math.isfinite(x) and abs(x) <= 1 for x in row):
            raise ValueError(f'{name} must be finite and within [-1, 1]')
        if abs(sum(map(operator.mul, row, row)) - 1) > 1e-4:
            raise ValueError(f'{name} rows must be unit vectors')
        for other in rows[:a]:
            if abs(sum(map(operator.mul, row, other))) > 1e-4:
                raise ValueError(f'{name} rows must be orthonormal')


def subspace_members(base, direction_kind=None, direction_count=None,
                     preserve_norm=None):
    """The validated subspace members of a profile from `base`, with the
    given option defaults (None keeps the base value; absent stays absent)."""
    hidden, layers = base['hidden'], base['layers']
    members = {}
    rank = base.get('rank', 1)
    if not (isinstance(rank, int) and not isinstance(rank, bool) and
            1 <= rank <= MAX_RANK):
        raise ValueError(f'rank must be an integer in [1, {MAX_RANK}]')
    if 'rank' in base:
        members['rank'] = rank
    subspace = base.get('subspace_directions')
    if (rank > 1) != (subspace is not None):
        raise ValueError('subspace_directions needs rank > 1 and vice versa')
    if subspace is not None:
        if len(subspace) != layers:
            raise ValueError('need one subspace_directions string per layer')
        for layer, text in enumerate(subspace):
            name = f'subspace_directions[{layer}]'
            check_orthonormal([base['directions'][layer]] +
                              decode_rows(text, rank - 1, hidden, name), name)
        members['subspace_directions'] = list(subspace)
    kinds = ['mean_diff']
    projected = base.get('projected_directions')
    if projected is not None:
        if len(projected) != layers:
            raise ValueError('need one projected_directions string per layer')
        for layer, text in enumerate(projected):
            name = f'projected_directions[{layer}]'
            check_orthonormal(decode_rows(text, rank, hidden, name), name)
        members['projected_directions'] = list(projected)
        kinds.append('projected')
    method = base.get('subspace_method')
    if method is not None:
        if method not in METHODS:
            raise ValueError(f'subspace_method must be one of {METHODS}')
        members['subspace_method'] = method
    kind = direction_kind if direction_kind is not None else base.get('direction_kind')
    if kind is not None:
        if kind not in kinds:
            raise ValueError(f'direction_kind must be one of {kinds}')
        members['direction_kind'] = kind
    count = direction_count if direction_count is not None else base.get('direction_count')
    if count is not None:
        if not (isinstance(count, int) and not isinstance(count, bool) and
                1 <= count <= rank):
            raise ValueError(f'direction_count must be an integer in [1, {rank}]')
        members['direction_count'] = count
    preserve = preserve_norm if preserve_norm is not None else base.get('preserve_norm')
    if preserve is not None:
        if not isinstance(preserve, bool):
            raise ValueError('preserve_norm must be a boolean')
        members['preserve_norm'] = preserve
    return members


def make_profile(base, direction_layer, attention, mlp, residual=None,
                 direction_kind=None, direction_count=None, preserve_norm=None,
                 **extra):
    """A v2 profile from `base` (model_identity, hidden, layers, placement,
    directions, and any subspace members) with the given defaults; validated
    like the runtime loader. `residual` (post-layer residual strengths) is
    optional: None omits the member, which the runtime reads as all zero
    (writer-only, as before). `direction_kind`, `direction_count` and
    `preserve_norm` None keep the base value (absent: mean_diff, 1, false)."""
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
    if residual is not None:
        profile['residual'] = check_weights(residual, layers)
    profile.update(subspace_members(base, direction_kind, direction_count,
                                    preserve_norm))
    for key in SUBSPACE_MEMBERS:
        if key in extra:
            raise ValueError(f'pass {key} through the base or the options')
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
    statistics = np.load(fit_dir / 'statistics.npz')
    directions = statistics['directions']
    norms = np.linalg.norm(directions, axis=1, keepdims=True)
    if not np.isfinite(norms).all() or (norms < 1e-12).any():
        raise ValueError('a layer has no usable direction')
    base = {k: v for k, v in summary.items() if k not in SUBSPACE_MEMBERS}
    base['directions'] = (directions / norms).astype(np.float32).tolist()
    # Fits before direction kinds and subspaces have neither basis.
    if 'basis_mean_diff' in statistics:
        base.update(subspace_base(statistics['basis_mean_diff'],
                                  statistics.get('basis_projected'),
                                  summary.get('subspace_method')))
        base['directions'] = statistics['basis_mean_diff'][:, 0].astype(
            np.float32).tolist()
    return base


def subspace_base(mean_diff, projected, method):
    """Profile members of fitted bases ([layers][rank][hidden] each; projected
    may be None). The fit default kind is projected when it exists."""
    rank = int(mean_diff.shape[1])
    members = dict(rank=rank, direction_count=1, preserve_norm=False,
                   direction_kind='projected' if projected is not None else 'mean_diff')
    if rank > 1:
        members['subspace_directions'] = [encode_rows(block[1:]) for block in mean_diff]
        members['subspace_method'] = method
    if projected is not None:
        members['projected_directions'] = [encode_rows(block) for block in projected]
    return members


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


def orthonormalize(first, candidates, rank, exclude=None, tolerance=1e-6):
    """Up to `rank` orthonormal rows: `first` (normalized) then the
    candidates in order, each made orthogonal to the rows before it and to
    the unit `exclude` vector (not returned) by Gram-Schmidt in FP64 (twice,
    for stability). A candidate whose remainder is below `tolerance` of its
    norm is dependent and skipped."""
    import numpy as np
    basis = [] if exclude is None else [np.asarray(exclude, dtype=np.float64)]
    fixed = len(basis)
    for index, vector in enumerate([first, *candidates]):
        vector = np.asarray(vector, dtype=np.float64)
        scale = np.linalg.norm(vector)
        if not (np.isfinite(scale) and scale > 0):
            if index == 0:
                raise ValueError('the first direction is zero')
            continue
        remainder = vector / scale
        for _ in range(2):
            for row in basis:
                remainder = remainder - (row @ remainder) * row
        norm = np.linalg.norm(remainder)
        if norm < tolerance:
            if index == 0:
                raise ValueError('the first direction lies in the excluded span')
            continue
        basis.append(remainder / norm)
        if len(basis) - fixed == rank:
            break
    return np.array(basis[fixed:])


def pca_candidates(deviations, reference):
    """Right singular vectors of the per-prompt deviations, by singular value,
    each signed to point along `reference`. Vectors of numerically zero
    singular values span nothing in the data and are dropped."""
    import numpy as np
    _, singular, vt = np.linalg.svd(deviations, full_matrices=False)
    kept = singular > 1e-6 * singular[0] if singular[0] > 0 else singular > 0
    return [v if v @ reference >= 0 else -v for v in vt[kept]]


def som_candidates(deviations, reference, rank, seed, epochs=40):
    """Prototypes of a batch self-organizing map over the per-prompt
    deviations, most aligned with `reference` (cosine) first. The g x g grid
    has g = ceil(sqrt(2 * rank)) (at least 2), at most one prototype per
    prompt; initialization draws prompts with a seeded generator and the batch
    updates are deterministic."""
    import numpy as np
    count = len(deviations)
    side = max(2, math.ceil(math.sqrt(2 * rank)))
    units = min(side * side, count)
    grid = np.array([(i // side, i % side) for i in range(units)], dtype=np.float64)
    rng = np.random.default_rng(seed)
    prototypes = deviations[np.sort(rng.choice(count, units, replace=False))].copy()
    squared = (deviations ** 2).sum(axis=1)
    grid_distance = ((grid[:, None, :] - grid[None, :, :]) ** 2).sum(axis=2)
    start, end = max(side / 2, 1.0), 0.5
    for epoch in range(epochs):
        sigma = start * (end / start) ** (epoch / max(epochs - 1, 1))
        distance = (squared[:, None] - 2 * deviations @ prototypes.T +
                    (prototypes ** 2).sum(axis=1)[None, :])
        winners = np.argmin(distance, axis=1)  # first minimum: deterministic
        influence = np.exp(-grid_distance[winners] / (2 * sigma * sigma))
        weight = influence.sum(axis=0)
        used = weight > 1e-12
        prototypes[used] = (influence.T @ deviations)[used] / weight[used, None]
    norms = np.linalg.norm(prototypes, axis=1)
    cosine = (prototypes @ reference) / np.maximum(norms * np.linalg.norm(reference), 1e-30)
    order = sorted(range(units), key=lambda i: (-cosine[i], i))
    return [prototypes[i] for i in order]


def fit_bases(refusal, ordinary_mean, rank, method, seed):
    """Per-layer orthonormal bases [layers][rank][hidden] of the mean
    difference and, when every layer allows one, of the projected kind
    (None otherwise), plus the rank reached by every layer and kind (at most
    `rank`; dependent data yields fewer rows). `refusal` is
    [prompts][layers][hidden]."""
    import numpy as np
    layers = ordinary_mean.shape[0]
    mean_diff, projected = [], []
    projected_ok = True
    for layer in range(layers):
        deviations = refusal[:, layer, :].astype(np.float64) - ordinary_mean[layer]
        r = deviations.mean(axis=0)
        if method == 'pca':
            candidates = pca_candidates(deviations, r)
        else:
            candidates = som_candidates(deviations, r, rank, [seed, layer])
        mean_diff.append(orthonormalize(r, candidates, rank))
        h = ordinary_mean[layer]
        h_norm = np.linalg.norm(h)
        if projected_ok and h_norm > 0:
            h = h / h_norm
            p = r - (r @ h) * h
            # A refusal direction along the ordinary mean has no projected kind.
            if np.linalg.norm(p) > 1e-6 * np.linalg.norm(r):
                projected.append(orthonormalize(p, candidates, rank, exclude=h))
            else:
                projected_ok = False
        else:
            projected_ok = False
    reached = min(len(b) for b in mean_diff + (projected if projected_ok else []))
    mean_diff = np.array([b[:reached] for b in mean_diff])
    projected = np.array([b[:reached] for b in projected]) if projected_ok else None
    return mean_diff, projected, reached


def fit(collection, output, strengths=(0.5, 1.0), method='pca', rank=4, seed=0):
    import numpy as np
    collection, output = Path(collection), Path(output)
    if method not in METHODS:
        raise ValueError(f'method must be one of {METHODS}')
    if not (isinstance(rank, int) and 1 <= rank <= MAX_RANK):
        raise ValueError(f'rank must be an integer in [1, {MAX_RANK}]')
    manifest = json.loads((collection / 'manifest.json').read_text())
    if manifest['split'] != 'collection':
        raise ValueError('fit only collection split, never tuning/evaluation')
    # Geometry and placement come from the capture, never from one model.
    hidden, layers = manifest['hidden'], manifest['layers']
    site = placement(manifest)
    sums = {g: np.zeros((layers, hidden), dtype=np.float64)
            for g in ('refusal', 'ordinary')}
    counts = dict.fromkeys(sums, 0)
    # Per-prompt refusal activations, for the subspace (27B: about 0.8 GB).
    refusal = []
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
        if row['group'] == 'refusal':
            refusal.append(values)
    if min(counts.values()) < 2:
        raise ValueError('both prompt classes need at least two samples')
    means = {g: sums[g] / counts[g] for g in sums}
    directions = means['refusal'] - means['ordinary']
    norms = np.linalg.norm(directions, axis=1)
    unusable = [i for i, n in enumerate(norms) if not (np.isfinite(n) and n >= 1e-12)]
    if unusable:
        # A v2 profile needs every layer's direction.
        raise ValueError(f'layers {unusable} have no usable refusal direction')
    basis_mean_diff, basis_projected, reached = fit_bases(
        np.stack(refusal), means['ordinary'], rank, method, seed)
    output.mkdir(parents=True, exist_ok=False)
    arrays = dict(means, directions=directions, norms=norms,
                  basis_mean_diff=basis_mean_diff)
    if basis_projected is not None:
        arrays['basis_projected'] = basis_projected
    np.savez(output / 'statistics.npz', **arrays)
    collection_sha = sha((collection / 'manifest.json').read_bytes())
    binding = dict(model_identity=manifest['model_identity'], hidden=hidden,
                   layers=layers, placement=site)
    base = dict(binding, directions=basis_mean_diff[:, 0].astype(np.float32).tolist(),
                **subspace_base(basis_mean_diff, basis_projected, method))
    # A bounded candidate screen. Selection must use tuning, final check
    # evaluation. Every candidate carries all directions, both kinds and the
    # subspace, so a server can try another direction layer, kind or
    # direction count per request. Defaults: projected (when available), one
    # direction, no norm preservation.
    selected = sorted(set(int(round((layers - 1) * f)) for f in (.4, .6, .8)))
    for strength in strengths:
        if not 0 <= strength <= MAX_WEIGHT:
            raise ValueError('strength must be in [0, 4]')
    for layer in selected:
        for strength in strengths:
            # Restrict to a neighborhood of the candidate layer initially.
            weights = [float(strength if abs(i - layer) <= 3 else 0)
                       for i in range(layers)]
            save_profile(output / f'layer{layer}-strength{strength:g}.json',
                         make_profile(base, layer, weights, weights,
                                      status='candidate-not-qualified',
                                      collection_sha256=collection_sha,
                                      strength=strength))
        # Residual-stream directional ablation: the layer's direction removed
        # from the post-layer residual at every layer (strength 1 leaves no
        # component where the collector measured it), writers off.
        zeros = [0.0] * layers
        save_profile(output / f'layer{layer}-residual1.json',
                     make_profile(base, layer, zeros, zeros,
                                  residual=[1.0] * layers,
                                  status='candidate-not-qualified',
                                  collection_sha256=collection_sha,
                                  strength=1.0))
    save(output / 'SUMMARY.json', dict(
        binding, counts=counts, candidate_layers=selected,
        collection_sha256=collection_sha, subspace_method=method,
        requested_rank=rank, rank=reached, seed=seed,
        direction_kinds=['mean_diff'] + (['projected'] if basis_projected is not None else []),
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
    fitting.add_argument('--method', choices=METHODS, default='pca',
                         help='subspace directions: top principal directions '
                              '(pca) or self-organizing-map prototypes (som)')
    fitting.add_argument('--rank', type=int, default=4,
                         help=f'directions per layer and kind, 1-{MAX_RANK} '
                              '(fewer when the data spans fewer)')
    fitting.add_argument('--seed', type=int, default=0, help='SOM initialization')
    make = sub.add_parser('make-profile', help='write one v2 profile with chosen defaults')
    source = make.add_mutually_exclusive_group(required=True)
    source.add_argument('--fit', help='a fit output directory (statistics.npz)')
    source.add_argument('--profile', help='an existing v2 profile to re-default')
    make.add_argument('--direction-layer', type=int, required=True)
    make.add_argument('--attention',
                      help='2.0 | w0,...,wN-1 | max_w=..,peak=..,min_w=..,width=.. '
                           '(default: the --profile value, else 0)')
    make.add_argument('--mlp', help='as --attention')
    make.add_argument('--residual',
                      help='post-layer residual strengths, as --attention '
                           '(default: the --profile value, else omitted = 0)')
    make.add_argument('--direction-kind', choices=KINDS,
                      help='default direction kind (default: the base value)')
    make.add_argument('--directions', type=int,
                      help='default number of subspace directions (default: '
                           'the base value)')
    make.add_argument('--preserve-norm', choices=('on', 'off'),
                      help='default norm preservation (default: the base value)')
    make.add_argument('--output', required=True)
    args = parser.parse_args()
    if args.command == 'import-data':
        print(json.dumps(import_data(args.archive, args.output), indent=2))
    elif args.command == 'fit':
        fit(args.collection, args.output, method=args.method, rank=args.rank,
            seed=args.seed)
    else:
        base = load_base(args.fit, args.profile)
        layers = base['layers']
        output = Path(args.output)
        if output.exists():
            raise SystemExit('use a new output path; existing profiles are preserved')
        # An omitted strength keeps the base profile's default (zero, or no
        # residual member, for a fit output), so re-defaulting one member
        # never silently clears the others.
        def strength(spec, name, fallback):
            if spec is not None:
                return parse_weights(spec, layers)
            return base.get(name, fallback)

        save_profile(output, make_profile(
            base, args.direction_layer,
            strength(args.attention, 'attention', [0.0] * layers),
            strength(args.mlp, 'mlp', [0.0] * layers),
            residual=strength(args.residual, 'residual', None),
            direction_kind=args.direction_kind, direction_count=args.directions,
            preserve_norm=(None if args.preserve_norm is None
                           else args.preserve_norm == 'on'),
            status='candidate-not-qualified'))
        print(output)


if __name__ == '__main__':
    main()
