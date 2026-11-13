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


OPERATIONS = {'fit': _fit, 'test': _test, 'train': _train, 'dataset': data.run_dataset_job}
assert set(OPERATIONS) == set(jobs.OPERATIONS)


def profile_bytes(store, kind, name, body):
    """The bytes to upload: body {file: PATH} | {job: KEY, profile: NAME}
    (a profile a fit job wrote) | {profile: {...}} (the JSON itself)."""
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
        if not path.exists():
            raise FileNotFoundError(f'job {job.key} wrote no profile {profile}')
        return path.read_bytes()
    raise ValueError('give file, job (and profile) or the profile object')
