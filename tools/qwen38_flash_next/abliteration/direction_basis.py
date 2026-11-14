"""Build a direction basis (gufo-direction-basis-v1) for joint direction and
strength SPSA (R11; spsa.py --basis). CPU only (NumPy); never starts a
server or a GPU job.

Input: the collector `answers` or `segments` collection the base direction
was fitted from (fit_steering.load: per-record [layers][hidden] FP32), and
the base profile (for example recheck-avgx: the L40-L47 projected rows of a
fit, averaged and sign-aligned to L45).

Per pair i and layer l in `--layers`, the paired difference D_il (exemplar
mean - own mean, fit_steering.paired_differences) becomes the row the fit
used: for `--kind projected`, D_il minus its component along the normalized
own mean h_l. Each layer's rows are scaled by 1 / |m_l| (m_l their mean, so
every layer weighs as one unit row, as the averaged direction does) and
sign-aligned as build/b1/r10/newavg.py aligns layer rows: flipped when m_l
points against the fit's unit row at `--ref-layer`. These pooled rows X
(pairs x layers of them) minus their component along d0 (the base profile's
row at `--direction-layer`) are the residuals R; their principal axes
(centred PCA by default; `--no-center` takes the uncentred singular vectors)
are u_1..u_k, orthonormalized against d0 in FP64 (Gram-Schmidt, twice),
each signed so its largest-magnitude entry is positive.

Output JSON: d0 (the base row's float32 values), `basis` (k rows), the
variance of R along each u_k (`explained_variance`, per pooled row) and its
share of R's total (`explained_ratio`), the spread of X along d0, the
cosine between d0 and the direction recomputed from the collection
(`d0_check`), the collection, its manifest SHA-256, the layers and the base
profile.

    direction_basis.py --collection DIR --base-profile recheck-avgx.json \
        --layers 40-47 --ref-layer 45 --direction-layer 45 -k 6 --output basis.json
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import direction_family  # noqa: E402

SCHEMA = direction_family.SCHEMA


def layer_rows(D, own_mean, layers, ref_layer, kind='projected'):
    """(X [pairs * len(layers)][hidden], per-layer signs, per-layer unit
    means) of the paired differences D [pairs][all layers][hidden] (see the
    module doc)."""
    rows, signs, units = [], [], {}
    for layer in sorted(set(layers) | {ref_layer}):
        P = D[:, layer, :].astype(np.float64)
        if kind == 'projected':
            h = own_mean[layer] / np.linalg.norm(own_mean[layer])
            P = P - np.outer(P @ h, h)
        m = P.mean(axis=0)
        units[layer] = (m / np.linalg.norm(m), np.linalg.norm(m), P)
    ref = units[ref_layer][0]
    for layer in layers:
        unit, size, P = units[layer]
        sign = -1.0 if unit @ ref < 0 else 1.0
        signs.append(sign)
        rows.append(sign * P / size)
    return np.concatenate(rows), signs, [units[layer][0] * s for layer, s in zip(layers, signs)]


def build(X, d0, k, center=True):
    """(u [k][hidden] orthonormal and orthogonal to d0, explained variance,
    explained ratio, total residual variance) of the pooled rows X."""
    d0 = np.asarray(d0, dtype=np.float64)
    d0 = d0 / np.linalg.norm(d0)
    R = X - np.outer(X @ d0, d0)
    if center:
        R = R - R.mean(axis=0)
    n = len(R)
    _, singular, vt = np.linalg.svd(R, full_matrices=False)
    total = float((singular ** 2).sum())
    span = min(len(vt), n - 1 if center else n)
    if k > span:
        raise ValueError(f'{k} components asked for, the data spans at most {span}')
    basis = []
    for v in vt[:k]:
        for _ in range(2):
            v = v - (v @ d0) * d0
            for u in basis:
                v = v - (v @ u) * u
        v = v / np.linalg.norm(v)
        if v[np.argmax(np.abs(v))] < 0:
            v = -v
        basis.append(v)
    scale = max(n - 1, 1)
    variance = singular[:k] ** 2 / scale
    ratio = singular[:k] ** 2 / total if total > 0 else np.zeros(k)
    return np.array(basis), variance, ratio, total / scale


def parse_layers(spec):
    out = []
    for part in str(spec).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            out.extend(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def make(collection, base_profile, layers, ref_layer, direction_layer, k, kind='projected',
         center=True):
    import fit_steering
    manifest, pairs = fit_steering.load(collection)
    D, own_mean, skipped = fit_steering.paired_differences(pairs)
    base = json.loads(Path(base_profile).read_text())
    d0 = direction_family.active_row(base, direction_layer, kind)
    X, signs, units = layer_rows(D, own_mean, layers, ref_layer, kind)
    basis, variance, ratio, total = build(X, d0, k, center)
    d0v = np.asarray(d0, dtype=np.float64)
    d0v /= np.linalg.norm(d0v)
    average = np.mean(units, axis=0)
    average /= np.linalg.norm(average)
    along = X @ d0v
    residual_norms = np.linalg.norm(X - np.outer(along, d0v), axis=1)
    return dict(
        schema=SCHEMA, hidden=int(manifest['hidden']), kind=kind,
        direction_layer=direction_layer, layers=list(layers), ref_layer=ref_layer,
        center=center, pairs=int(D.shape[0]), skipped_questions=skipped, rows=int(len(X)),
        collection=str(collection),
        collection_manifest_sha256=hashlib.sha256(
            (Path(collection) / 'manifest.json').read_bytes()).hexdigest(),
        base_profile=str(base_profile),
        base_profile_sha256=direction_family.file_sha256(base_profile),
        layer_signs=signs,
        d0_check=dict(cos_d0_recomputed_average=round(float(average @ d0v), 6),
                      cos_d0_layer_means=[round(float(u @ d0v), 4) for u in units]),
        along_d0=dict(mean=round(float(along.mean()), 6), std=round(float(along.std(ddof=1)), 6),
                      residual_norm_mean=round(float(residual_norms.mean()), 6)),
        residual_variance_total=round(total, 8),
        explained_variance=[round(float(v), 8) for v in variance],
        explained_ratio=[round(float(r), 6) for r in ratio],
        d0=[float(x) for x in d0],
        basis=[[float(x) for x in u] for u in basis])


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    ap.add_argument('--collection', required=True)
    ap.add_argument('--base-profile', required=True)
    ap.add_argument('--layers', default='40-47', help='layers whose rows form the family (a-b,c)')
    ap.add_argument('--ref-layer', type=int, default=45, help='the sign-alignment layer')
    ap.add_argument('--direction-layer', type=int, default=45,
                    help="the base profile's direction layer (d0 is its row there)")
    ap.add_argument('-k', type=int, default=6, help='basis vectors')
    ap.add_argument('--kind', choices=('projected', 'mean_diff'), default='projected')
    ap.add_argument('--no-center', action='store_true', help='uncentred singular vectors')
    ap.add_argument('--output', required=True)
    args = ap.parse_args(argv)
    if args.k < 1:
        ap.error('-k must be at least 1')
    out = make(args.collection, args.base_profile, parse_layers(args.layers), args.ref_layer,
               args.direction_layer, args.k, args.kind, not args.no_center)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out) + '\n')
    print(json.dumps({k: v for k, v in out.items() if k not in ('d0', 'basis')}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
