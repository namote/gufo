"""Fit job: dataset -> gufo collection job (polled) -> per-layer analysis and
default profiles.

Needs NumPy (and the abliteration tools `prepare.py` and `fit_steering.py`,
imported from tools/qwen38_flash_next/abliteration). The host has no NumPy:
run the tool with the pinned container's `/opt/venv/bin/python3 -s` or a
local venv (README.md).

Analysis, per layer L, over the per-pair differences D_i = mean(exemplar
rows) - mean(own rows) of the n pairs (build/b1/think/analyze.py):

* `acc`: held-out separation. Leave-one-out direction v = sum_{j != i} D_j;
  pair i is separated when D_i . v > 0. Reported as the separated count and
  the fraction.
* `snr`: mean / sample standard deviation of the held-out projections
  D_i . v / |v|.
* `norm`: |mean D| (|d|).

Cross-layer cosines: cos(mean D_L, mean D_M) for every pair of layers, and
for the top layers the held-out separation of layer M's differences along
layer L's leave-one-out direction (build/b1/think/xlayer.py).

The control dataset (shuffled labels) gets the same table; real directions
should separate held-out pairs far above it.

Default profiles (validated as the runtime loader validates, through
prepare.make_profile and fit_steering.fit):

* `NAME-add`: additive (`mode: add`), fit_steering's default strengths at
  `alpha` (the residual after layer L carries about alpha times the measured
  shift); a request may negate them (`residual` per layer) to push against
  the direction;
* `NAME-pL`: the pol-avg suppression recipe from layer L: projected mean
  difference of layer L, `residual` 1.0 at every layer, attention and mlp 0
  (build/b1/think/mksup.py);
* `NAME-sL`: suppression from one layer only: the same direction, `residual`
  1.0 at layer L only.
"""
import json
from pathlib import Path
import sys
import time

HERE = Path(__file__).resolve().parent
ABLITERATION = HERE.parent / 'qwen38_flash_next' / 'abliteration'
if str(ABLITERATION) not in sys.path:
    sys.path.insert(0, str(ABLITERATION))

import client  # noqa: E402
import data  # noqa: E402

TOP_N = 4
POLL_SECONDS = 2.0


def _np():
    try:
        import numpy as np
    except ImportError:
        raise RuntimeError('fit needs NumPy: run the tool with the pinned container\'s '
                           '/opt/venv/bin/python3 -s or a venv (see README.md)') from None
    return np


def _modules():
    _np()
    import fit_steering
    import prepare
    return fit_steering, prepare


def load_differences(collection):
    """(manifest, D [pairs][layers][hidden] float64, skipped pairs) of a
    collector answers or segments collection (integrity-checked)."""
    fit_steering, _ = _modules()
    manifest, pairs = fit_steering.load(collection)
    diffs, _, skipped = fit_steering.paired_differences(pairs)
    return manifest, diffs, skipped


def layer_row(D, layer):
    """Held-out separation, SNR and |d| of one layer (see the docstring)."""
    np = _np()
    n = len(D)
    S = D[:, layer].sum(0)
    v = S[None] - D[:, layer]
    norms = np.linalg.norm(v, axis=1)
    norms[norms == 0] = 1.0
    pr = (v * D[:, layer]).sum(1) / norms
    sd = pr.std(ddof=1) if n > 1 else 0.0
    return dict(layer=layer, separated=int((pr > 0).sum()), pairs=n,
                acc=round(float((pr > 0).mean()), 6),
                snr=round(float(pr.mean() / sd), 6) if sd > 0 else None,
                norm=round(float(np.linalg.norm(S) / n), 6))


def cosine_matrix(D):
    np = _np()
    mean = D.mean(0)
    norms = np.linalg.norm(mean, axis=1)
    norms[norms == 0] = 1.0
    unit = mean / norms[:, None]
    return np.round(unit @ unit.T, 4).tolist()


def transfer(D, sources):
    """{source: [held-out separated count of layer M along source's LOO
    direction, for every M]}."""
    np = _np()
    S = D.sum(0)
    out = {}
    for s in sources:
        v = S[s][None] - D[:, s]
        norms = np.linalg.norm(v, axis=1)
        norms[norms == 0] = 1.0
        out[str(s)] = [int(((v * D[:, t]).sum(1) / norms > 0).sum()) for t in range(D.shape[1])]
    return out


def rank_layers(rows, top_n):
    """Layers by held-out SNR (accuracy saturates at 100%), then accuracy."""
    order = sorted(rows, key=lambda r: (-(r['snr'] if r['snr'] is not None else -1e30),
                                        -r['acc'], r['layer']))
    return [r['layer'] for r in order[:top_n]]


def analyze(job, D, label, top_n):
    """Emit the per-layer rows (one result each, as they are computed), the
    cosine table and the top layers of one collection. Returns the rows."""
    rows = []
    layers = D.shape[1]
    for layer in range(layers):
        job.check_cancel()
        row = layer_row(D, layer)
        rows.append(row)
        job.emit(dict(type='layer', dataset=label, **row))
        job.progress(done=layer + 1, total=layers, step=f'{label}: layer {layer + 1}/{layers}')
    top = rank_layers(rows, top_n)
    job.emit(dict(type='cosine', dataset=label, layers=layers, matrix=cosine_matrix(D)))
    job.emit(dict(type='transfer', dataset=label, sources=top, separated=transfer(D, top),
                  pairs=len(D)))
    job.emit(dict(type='top', dataset=label, layers=top))
    return rows, top


def make_profiles(collection, prefix, layers, alpha=1.0, additive_kind='mean_diff'):
    """{name: profile} for the defaults (see the docstring)."""
    _, prepare = _modules()
    import fit_steering
    profiles = {}
    additive = fit_steering.fit(collection, kind=additive_kind, alpha=alpha)
    profiles[f'{prefix}-add'] = additive
    n = additive['layers']
    zeros = [0.0] * n
    kind = 'projected' if 'projected_directions' in additive else 'mean_diff'
    for layer in layers:
        profiles[f'{prefix}-p{layer}'] = prepare.make_profile(
            additive, int(layer), zeros, zeros, residual=[1.0] * n, direction_kind=kind,
            direction_count=1, preserve_norm=False)
        one = [0.0] * n
        one[int(layer)] = 1.0
        profiles[f'{prefix}-s{layer}'] = prepare.make_profile(
            additive, int(layer), zeros, zeros, residual=one, direction_kind=kind,
            direction_count=1, preserve_norm=False)
    for name in profiles:
        data.check_name(name)
    return profiles


def collect(job, dataset, records, mode, collections_dir, label, name=None):
    """Run (or resume) a gufo collection job for `records` and poll it to the
    end. Returns the local collection path."""
    gufo = job.context.gufo
    if gufo is None:
        raise ValueError('collecting needs a gufo server (--gufo-url)')
    if not collections_dir:
        raise ValueError('collecting needs collections_dir: the local path of the '
                         "server's --collect-dir")
    text = data.records_text(records)
    name = data.check_name(name or f'{dataset[:51]}-{data.sha_text(text)[:12]}')
    key = _start(gufo, mode, name, records)
    job.emit(dict(type='collect', dataset=label, gufo_job=key, collection=name,
                  status='queued', done=0, total=len(records)))
    since, last_status = 0, None
    outage = None  # when gufo stopped answering
    try:
        while True:
            job.wait(job.context.poll_seconds)
            try:
                if key is None:  # gufo restarted and forgot the job: resume it by name
                    key = _start(gufo, mode, name, records)
                    since, last_status = 0, None
                state = gufo.collect_job(key, since)
                outage = None
            except client.GufoError as error:
                if error.code == 'unknown_job':
                    key = None
                    continue
                if error.status < 500 and error.code != 'server_stopping':
                    raise
                outage = _outage(outage)
                continue
            except (OSError, ValueError):  # restarting: refused, reset or truncated
                outage = _outage(outage)
                continue
            since = state.get('last_seq', since)
            progress = state.get('progress') or {}
            done, total = progress.get('done', 0), progress.get('total', len(records))
            job.progress(done=done, total=total, stage=f'collect {label}',
                         step=progress.get('step') or state['status'])
            if state.get('results') or state['status'] != last_status:
                job.emit(dict(type='collect', dataset=label, gufo_job=key, collection=name,
                              status=state['status'], done=done, total=total,
                              numerics_deviations=state.get('numerics_deviations') or []))
                last_status = state['status']
            if state['status'] == 'done':
                break
            if state['status'] in ('failed', 'cancelled'):
                raise RuntimeError(f'gufo collection {name} {state["status"]}: {state.get("error")}')
    except BaseException:
        try:
            if key is not None:
                gufo.cancel_collect(key)
        except Exception:
            pass
        raise
    path = Path(collections_dir) / name
    if not (path / 'manifest.json').exists():
        raise FileNotFoundError(f'{path}/manifest.json: collections_dir must be the local '
                                "path of the server's --collect-dir")
    return path


OUTAGE_SECONDS = 900.0


def _start(gufo, mode, name, records):
    try:
        return gufo.start_collect(mode, name, records, resume=True)
    except client.GufoError as error:
        if error.code == 'collection_differs':
            raise RuntimeError(f'gufo collection {name} exists from another server build or '
                               'settings; set collection_name to collect anew') from None
        raise


def _outage(since):
    """Tolerate gufo being down (a restart reloads the model) for up to
    OUTAGE_SECONDS; returns when it started."""
    now = time.monotonic()
    if since is not None and now - since > OUTAGE_SECONDS:
        raise RuntimeError(f'gufo did not answer for {OUTAGE_SECONDS:.0f} s')
    return now if since is None else since


def check_prefix(prefix):
    """Profile names PREFIX-add, -pL and -sL must stay valid names (checked
    before a long collection, not after it)."""
    data.check_name(prefix)
    data.check_name(f'{prefix}-p999')
    return prefix


def run_fit_job(job):
    """Operation `fit`. Params: dataset (name) or collection (a local
    collection path) and optional control_collection; collect_only (stop
    after the collections: the host without NumPy collects, a NumPy
    interpreter fits the paths it reports); collections_dir,
    control (default true), top_n (4), profile_layers (default: the top 2),
    alpha (1.0), additive_kind (mean_diff), prefix (default the dataset or
    collection name)."""
    p = dict(job.params)
    known = {'dataset', 'collection', 'control_collection', 'collections_dir', 'control',
             'top_n', 'profile_layers', 'alpha', 'additive_kind', 'prefix', 'collection_name',
             'collect_only'}
    unknown = set(p) - known
    if unknown:
        raise ValueError(f'unknown fit parameters {sorted(unknown)}')
    for key in ('control', 'collect_only'):
        if key in p and not isinstance(p[key], bool):
            raise ValueError(f'{key} must be a boolean')
    if 'profile_layers' in p and not (
            isinstance(p['profile_layers'], list) and
            all(isinstance(x, int) and not isinstance(x, bool) and x >= 0
                for x in p['profile_layers'])):
        raise ValueError('profile_layers must be a list of layer indexes')
    collect_only = bool(p.get('collect_only', False))
    if not collect_only:
        _np()  # fail before a long collection, not after it
    top_n = int(p.get('top_n', TOP_N))
    use_control = p.get('control', True)
    collections_dir = p.get('collections_dir') or job.context.collections_dir
    if p.get('dataset'):
        meta, records = data.load_dataset(job.work, p['dataset'])
        if meta['kind'] not in ('answers', 'segments'):
            raise ValueError('fit needs an answers or segments dataset (paired groups)')
        prefix = check_prefix(p.get('prefix') or meta['name'])
        job.progress(stage='collect', step='starting the collection')
        collection = collect(job, meta['name'], records, meta['kind'], collections_dir, 'real',
                             p.get('collection_name'))
        control = None
        if use_control and meta.get('control'):
            cmeta, crecords = data.load_dataset(job.work, meta['control'])
            control = collect(job, cmeta['name'], crecords, cmeta['kind'], collections_dir,
                              'control')
        if collect_only:
            output = dict(collection=str(collection),
                          control_collection=str(control) if control else None)
            job.emit(dict(type='collected', **output))
            job.set_output(output)
            job.progress(stage='done', step='collected')
            return
    elif p.get('collection'):
        if collect_only:
            raise ValueError('collect_only needs a dataset')
        collection = Path(p['collection'])
        prefix = check_prefix(p.get('prefix') or collection.name)
        control = Path(p['control_collection']) if use_control and p.get('control_collection') else None
    else:
        raise ValueError('fit needs dataset or collection')
    data.check_name(prefix)
    job.progress(stage='fit', step='loading activations')
    manifest, D, skipped = load_differences(collection)
    job.emit(dict(type='collection', dataset='real', path=str(collection), pairs=len(D),
                  skipped=skipped, layers=int(manifest['layers']),
                  hidden=int(manifest['hidden']), readout=manifest['readout']))
    rows, top = analyze(job, D, 'real', top_n)
    summary = dict(collection=str(collection), pairs=len(D), top=top, layers=rows)
    if control is not None:
        _, C, cskipped = load_differences(control)
        job.emit(dict(type='collection', dataset='control', path=str(control), pairs=len(C),
                      skipped=cskipped))
        crow, ctop = analyze(job, C, 'control', top_n)
        summary.update(control=dict(collection=str(control), pairs=len(C), top=ctop,
                                    layers=crow))
        np = _np()
        cos = []
        for layer in range(D.shape[1]):
            a, b = D[:, layer].mean(0), C[:, layer].mean(0)
            denom = np.linalg.norm(a) * np.linalg.norm(b)
            cos.append(round(float(a @ b / denom), 4) if denom > 0 else None)
        job.emit(dict(type='control_cosine', cosines=cos))
    del D
    job.progress(stage='profiles', step='building default profiles')
    layers = p.get('profile_layers') or top[:2]
    layers = [int(x) for x in layers]
    profiles = make_profiles(collection, prefix, layers, float(p.get('alpha', 1.0)),
                             p.get('additive_kind', 'mean_diff'))
    _, prepare = _modules()
    out_dir = job.dir / 'profiles'
    written = []
    for name, profile in profiles.items():
        path = out_dir / f'{name}.json'
        prepare.save_profile(path, profile)
        item = dict(type='profile', name=name, path=str(path),
                    mode=profile.get('mode', 'suppress'),
                    direction_layer=profile.get('direction_layer'),
                    direction_kind=profile.get('direction_kind'))
        if profile.get('mode') == 'add':
            item['residual'] = profile['residual']
        job.emit(item)
        written.append(name)
    summary['profiles'] = written
    summary['profile_dir'] = str(out_dir)
    job.set_output(summary)
    job.progress(stage='done', step='done')
