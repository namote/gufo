"""Fit an additive (steering, R5) profile from answer-token activations.

CPU only (NumPy); never starts a server or a GPU job.

Input: a collector `answers` collection (collect.cpp; readout
`answer-tokens-mean-post-layer-hc-mean-fp32-v1`). Each record is one
teacher-forced answer, saved as the per-layer mean over its answer tokens of
the post-layer residual averaged over the HC streams, with its `group`
("exemplar" or "own") and `pair` (the question both answers share).

Direction per layer, paired by question so topic differences cancel:

    d_L = mean over pairs of (mean(exemplar answers) - mean(own answers))

`--kind projected` also removes, per layer, the component along the
normalized mean of the own answers (what every answer shares), as prepare.py
does for refusal directions; `--rank k` adds k-1 further orthonormal rows
per layer and kind from the per-pair differences (PCA, or a SOM with
`--method som`). The runtime adds only the first row of the chosen kind:
`y += alpha_L * d_L` at layer L's sites.

Directions are stored as unit vectors (the runtime contract); `magnitudes`
records |d_L| per layer. Default strengths: residual alpha_L =
`--alpha` * |d_L| (so `--alpha 1` adds the full measured mean shift at every
layer, 0 leaves the profile neutral until a request or SPSA sets strengths),
attention and MLP 0. Strengths are clamped to the runtime bound [-64, 64].

The output is a gufo-abliteration-v2 profile with `"mode": "add"`.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

import prepare

READOUT = 'answer-tokens-mean-post-layer-hc-mean-fp32-v1'
MAX_ADD = 64.0


def sha(data):
    return hashlib.sha256(data).hexdigest()


def load(collection):
    """(manifest, {pair: {group: [rows [layers][hidden]]}})."""
    collection = Path(collection)
    manifest = json.loads((collection / 'manifest.json').read_text())
    if manifest.get('readout') != READOUT:
        raise ValueError('not an answer-token collection (collect.cpp answers)')
    hidden, layers = manifest['hidden'], manifest['layers']
    pairs = {}
    for record in manifest['records']:
        receipt = json.loads((collection / (record['id'] + '.json')).read_text())
        raw = (collection / (record['id'] + '.f32')).read_bytes()
        if sha(raw) != receipt['activation_sha256']:
            raise ValueError(f'{record["id"]}: activation integrity mismatch')
        values = np.frombuffer(raw, dtype='<f4')
        if values.size != layers * hidden or not np.isfinite(values).all():
            raise ValueError(f'{record["id"]}: bad activation')
        group = record['group']
        if group not in ('exemplar', 'own'):
            raise ValueError(f'{record["id"]}: group must be exemplar or own')
        pairs.setdefault(record['pair'], {}).setdefault(group, []).append(
            values.reshape(layers, hidden).astype(np.float64))
    return manifest, pairs


def paired_differences(pairs):
    """[pairs][layers][hidden] differences and the own-answer mean; questions
    lacking either group are skipped."""
    diffs, own = [], []
    for name in sorted(pairs):
        groups = pairs[name]
        if 'exemplar' not in groups or 'own' not in groups:
            continue
        exemplar = np.mean(groups['exemplar'], axis=0)
        mine = np.mean(groups['own'], axis=0)
        diffs.append(exemplar - mine)
        own.append(mine)
    if len(diffs) < 2:
        raise ValueError('need at least two questions with both groups')
    return np.stack(diffs), np.mean(own, axis=0)


def fit(collection, kind='mean_diff', rank=1, method='pca', seed=0, alpha=0.0):
    manifest, pairs = load(collection)
    diffs, own_mean = paired_differences(pairs)
    # fit_bases takes per-sample activations and the reference mean; the
    # paired differences on top of the own mean give mean d_L exactly.
    mean_diff, projected, reached = prepare.fit_bases(
        own_mean[None] + diffs, own_mean, rank, method, seed)
    if kind == 'projected' and projected is None:
        raise ValueError('a layer has no projected direction')
    d = diffs.mean(axis=0)
    if kind == 'projected':
        h = own_mean / np.linalg.norm(own_mean, axis=1, keepdims=True)
        d = d - (d * h).sum(axis=1, keepdims=True) * h
    magnitudes = np.linalg.norm(d, axis=1)
    if not np.isfinite(magnitudes).all() or (magnitudes < 1e-12).any():
        raise ValueError('a layer has no usable difference')
    binding = dict(model_identity=manifest['model_identity'],
                   hidden=manifest['hidden'], layers=manifest['layers'],
                   placement=prepare.PLACEMENTS[
                       'last-prompt-token-post-layer-hc-mean-fp32-v1'])
    base = dict(binding,
                directions=mean_diff[:, 0].astype(np.float32).tolist(),
                **prepare.subspace_base(mean_diff, projected, method))
    layers = manifest['layers']
    zeros = [0.0] * layers
    profile = prepare.make_profile(base, 0, zeros, zeros, direction_kind=kind,
                                   direction_count=1, preserve_norm=False)
    profile['mode'] = 'add'
    profile['residual'] = [prepare.float32_text(float(np.clip(
        alpha * m, -MAX_ADD, MAX_ADD))) for m in magnitudes]
    profile['magnitudes'] = [prepare.float32_text(float(m)) for m in magnitudes]
    profile['steering_fit'] = dict(
        pairs=int(diffs.shape[0]), kind=kind, rank=int(reached), method=method,
        seed=seed, alpha=alpha,
        collection_sha256=sha((Path(collection) / 'manifest.json').read_bytes()))
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--collection', required=True)
    parser.add_argument('--kind', choices=prepare.KINDS, default='mean_diff')
    parser.add_argument('--rank', type=int, default=1)
    parser.add_argument('--method', choices=prepare.METHODS, default='pca')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--alpha', type=float, default=0.0,
                        help='default residual strength per layer as a multiple '
                             'of |d_L| (0: neutral)')
    parser.add_argument('--output', required=True)
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise SystemExit('use a new output path; existing profiles are preserved')
    if not np.isfinite(args.alpha):
        raise SystemExit('--alpha must be finite')
    profile = fit(args.collection, args.kind, args.rank, args.method, args.seed,
                  args.alpha)
    prepare.save_profile(output, profile)
    print(output)


if __name__ == '__main__':
    main()
