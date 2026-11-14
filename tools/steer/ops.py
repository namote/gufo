"""The operations table, the job context and profile loading (shared by the
CLI and the HTTP service)."""
from dataclasses import dataclass
import json
from pathlib import Path

import data
import jobs


@dataclass
class Context:
    gufo: object = None              # client.Gufo, or None without --gufo-url
    collections_dir: str = None      # local path of gufo's --collect-dir
    poll_seconds: float = 2.0
    coding_runner: object = None     # tests inject a sandbox stand-in


def _fit(job):
    import fitting
    fitting.run_fit_job(job)


def _test(job):
    import generation
    generation.run_test_job(job)


def _train(job):
    import training
    training.run_train_job(job)


def _objective(job):
    import objective
    objective.run_objective_job(job)


def _pairs(job):
    import generation
    generation.run_pairs_job(job)


def _pipeline(job):
    import pipeline
    pipeline.run_pipeline_job(job)


def _profile(job):
    """Operation `profile`: the trained profile from a train job (or given
    `settings`): `base_profile` (a file or {job, profile}) with the best
    strengths as its defaults (and, for a train job with a `basis`, its tuned
    direction), written to the job's profiles/NAME.json;
    `upload` (false) also sends it to gufo (`kind` abliteration)."""
    import training
    p = job.params
    unknown = set(p) - {'train', 'settings', 'base_profile', 'name', 'upload'}
    if unknown:
        raise ValueError(f'unknown profile parameters {sorted(unknown)}')
    name = data.check_name(p.get('name'))
    settings = p.get('settings')
    train = None
    if p.get('train'):
        train = job.store.get(p['train'])
        if train is None or train.state['operation'] != 'train':
            raise ValueError(f'{p["train"]} is not a train job')
        best = (train.state.get('output') or {}).get('best')
        if not best:
            raise ValueError(f'train job {train.key} has no best settings')
        settings = best['settings']
    if not isinstance(settings, dict):
        raise ValueError('profile needs train (a train job key) or settings')
    base = p.get('base_profile') or (train.params.get('base_profile') if train else None)
    if not base:
        raise ValueError('profile needs base_profile')
    # A direction run's best also carries its tuned direction (alpha).
    best = (train.state['output']['best'] if train else None) or dict(settings=settings)
    item = training.write_profile(job, name, training.best_profile(
        training.load_base(job.store, base), best, train.params.get('basis') if train else None))
    output = dict(name=name, path=item['path'], uploaded=False)
    if p.get('upload'):
        reply = _upload(job, 'abliteration', name, Path(item['path']).read_bytes())
        output.update(uploaded=True, reply=reply)
    job.set_output(output)


def _upload(job, kind, name, raw):
    gufo = job.context.gufo
    if gufo is None:
        raise ValueError('uploading needs a gufo server (--gufo-url)')
    reply = gufo.upload_profile(kind, name, raw)
    job.emit(dict(type='upload', kind=kind, name=name, reply=reply))
    return reply


def _load(job):
    """Operation `load`: upload a profile (`file`, `job` + `profile`, or the
    `profile` object, as POST /profiles/...) as `name`; `unload` lists
    runtime profiles to delete first (gufo holds at most 8)."""
    import client
    p = dict(job.params)
    unknown = set(p) - {'kind', 'name', 'file', 'job', 'profile', 'base', 'unload'}
    if unknown:
        raise ValueError(f'unknown load parameters {sorted(unknown)}')
    kind = p.pop('kind', 'abliteration')
    name = data.check_name(p.pop('name', None))
    gufo = job.context.gufo
    if gufo is None:
        raise ValueError('load needs a gufo server (--gufo-url)')
    for old in p.pop('unload', None) or []:
        try:
            reply = gufo.delete_profile(kind, data.check_name(old))
        except client.GufoError as error:
            if error.status != 404:
                raise
            reply = dict(profile=old, deleted=False, missing=True)
        job.emit(dict(type='unload', kind=kind, name=old, reply=reply))
    reply = _upload(job, kind, name, profile_bytes(job.store, kind, name, p))
    job.set_output(dict(kind=kind, name=name, reply=reply))


OPERATIONS = {'fit': _fit, 'test': _test, 'train': _train, 'dataset': data.run_dataset_job,
              'objective': _objective, 'pairs': _pairs, 'pipeline': _pipeline,
              'profile': _profile, 'load': _load}
assert set(OPERATIONS) == set(jobs.OPERATIONS)


def profile_bytes(store, kind, name, body):
    """The bytes to upload: body {file: PATH} | {job: KEY, profile: NAME}
    (a profile a fit, train or profile job wrote; a train job without one
    builds it from its best settings and `base`, or its own base_profile) |
    {profile: {...}} (the JSON itself)."""
    if kind not in ('abliteration', 'logit-bias'):
        raise ValueError('kind must be abliteration or logit-bias')
    data.check_name(name)
    if isinstance(body.get('profile'), dict):
        return json.dumps(body['profile'], ensure_ascii=False, separators=(',', ':')).encode()
    if body.get('file'):
        if not isinstance(body['file'], str):
            raise ValueError('file is a path string')
        return Path(body['file']).read_bytes()
    if body.get('job'):
        if not isinstance(body['job'], str):
            raise ValueError('job is a key string')
        job = store.get(body['job'])
        if job is None:
            raise FileNotFoundError(f'no job {body["job"]}')
        profile = data.check_name(body.get('profile') or name)
        path = job.dir / 'profiles' / f'{profile}.json'
        if path.exists():
            return path.read_bytes()
        if job.state['operation'] == 'train':
            import training
            best = (job.state.get('output') or {}).get('best')
            base = body.get('base') or job.params.get('base_profile')
            if not best or not base:
                raise ValueError(f'train job {job.key} needs best settings and a base profile '
                                 '(body "base": a file or {job, profile})')
            profile = training.best_profile(training.load_base(store, base), best,
                                            job.params.get('basis'))
            return json.dumps(profile, separators=(',', ':')).encode()
        raise FileNotFoundError(f'job {job.key} wrote no profile {profile}')
    raise ValueError('give file, job (and profile) or the profile object')
