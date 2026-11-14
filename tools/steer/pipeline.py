"""Pipeline job: an ordered list of steps, each run as a child job.

Params:

* `steps`: [{name, operation, params}]; `name` defaults to the operation
  and index (`2-train`). Any value in a step's params may be a reference
  {"from": STEP, "key": KEY} (STEP an earlier step's index or name; KEY a
  dotted path into that step's job output, or `job` for its key and `dir`
  for its folder; "default" is used when the output lacks KEY).
* `resume`: an earlier pipeline's key. A step whose earlier run finished
  with the same resolved params is reused (not run again). An earlier test,
  pairs or train step that did not finish is resumed: its key is passed as
  the new step's `resume`, so finished generations and SPSA steps are kept.

A step may also carry:

* `await_server` {min_sessions, max_sessions, min_context, max_context,
  timeout_s}: before the step starts, poll gufo's /props (every 10 s; a
  server that does not answer is waiting too) until it matches. The tool
  never starts or stops servers: this is where the operator switches between
  the generation server and the training server (README.md, "Training
  server"). Without timeout_s it waits until cancelled.
* `gufo_url`: run this step against another gufo server (same API key).

Children are ordinary jobs with `parent` set to the pipeline's key; the
pipeline runs them one after another in its own run (it holds the gufo
worker). Progress shows the current step; `pipeline_step` results report
each step's start and end. Cancelling the pipeline cancels the running
child; a failed or cancelled child stops the pipeline.

Every step after one that runs again runs again too, from scratch (its
inputs may have changed, e.g. a profile uploaded under the same name). Held-out sets: a reference to a test step on AIME 2025 (a set path naming
aime25/aime2025) or on the coding eval split (or a coding set that includes
it) is refused, as is an objective, dataset or pairs step naming an AIME 2025
file. Evaluation steps may run on them; nothing may read their outputs.
"""
import copy
import dataclasses
import re
import threading
import time

import client
import jobs

NO_NESTING = ('pipeline',)
HELD_OUT = re.compile(r'aime[-_]?(?:20)?25', re.I)
RESUMABLE = ('test', 'pairs', 'train')
STEP_KEYS = {'name', 'operation', 'params', 'await_server', 'gufo_url'}
AWAIT_KEYS = {'min_sessions', 'max_sessions', 'min_context', 'max_context', 'timeout_s'}
AWAIT_POLL = 10.0


def server_matches(props, wait):
    sessions = props.get('total_slots')
    n_ctx = (props.get('default_generation_settings') or {}).get('n_ctx')
    for key, value, low in (('min_sessions', sessions, True), ('max_sessions', sessions, False),
                            ('min_context', n_ctx, True), ('max_context', n_ctx, False)):
        if wait.get(key) is None:
            continue
        if value is None or (value < wait[key] if low else value > wait[key]):
            return False
    return True


def await_server(job, gufo, wait, name):
    """Poll /props until the server matches `wait` (see the module doc)."""
    if gufo is None:
        raise ValueError(f'step {name}: await_server needs a gufo server')
    start = time.monotonic()
    while True:
        try:
            props = gufo.request('GET', '/props', timeout=10)
            if server_matches(props, wait):
                job.emit(dict(type='server', step=name, matched=True,
                              sessions=props.get('total_slots'),
                              n_ctx=(props.get('default_generation_settings') or {}).get('n_ctx')))
                return
            seen = (f'{props.get("total_slots")} sessions, context '
                    f'{(props.get("default_generation_settings") or {}).get("n_ctx")}')
        except (client.GufoError, OSError, ValueError) as error:
            seen = f'no answer ({type(error).__name__})'
        job.progress(step=f'{name}: waiting for a server with {wait} (now {seen})')
        if wait.get('timeout_s') is not None and time.monotonic() - start > wait['timeout_s']:
            raise RuntimeError(f'step {name}: no server matched {wait} (last: {seen})')
        job.wait(AWAIT_POLL)


def is_ref(value):
    return isinstance(value, dict) and 'from' in value and 'key' in value and \
        set(value) <= {'from', 'key', 'default'}


def refs(value):
    if is_ref(value):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from refs(v)
    elif isinstance(value, list):
        for v in value:
            yield from refs(v)


def strings(value):
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for v in value.values():
            yield from strings(v)
    elif isinstance(value, list):
        for v in value:
            yield from strings(v)


def held_out(step):
    """Whether a test step runs a held-out set."""
    if step['operation'] != 'test':
        return False
    spec = (step.get('params') or {}).get('set') or {}
    if isinstance(spec.get('path'), str) and HELD_OUT.search(spec['path']):
        return True
    if spec.get('kind') == 'coding':
        splits = spec.get('splits')
        return spec.get('split') == 'eval' or (
            not spec.get('split') and (not splits or 'eval' in splits))
    return False


def check_steps(steps):
    """Names, operations, references (earlier steps only) and the held-out
    rule, before anything runs. Returns the step names."""
    if not isinstance(steps, list) or not steps:
        raise ValueError('pipeline needs steps')
    names = []
    for i, step in enumerate(steps):
        if not isinstance(step, dict) or set(step) - STEP_KEYS:
            raise ValueError(f'step {i}: keys are {sorted(STEP_KEYS)}')
        wait = step.get('await_server')
        if wait is not None and (not isinstance(wait, dict) or set(wait) - AWAIT_KEYS):
            raise ValueError(f'step {i}: await_server keys are {sorted(AWAIT_KEYS)}')
        op = step.get('operation')
        if op not in jobs.OPERATIONS or op in NO_NESTING:
            raise ValueError(f'step {i}: operation must be one of '
                             f'{[o for o in jobs.OPERATIONS if o not in NO_NESTING]}')
        if not isinstance(step.get('params', {}), dict):
            raise ValueError(f'step {i}: params must be an object')
        name = step.get('name') or f'{i}-{op}'
        if not isinstance(name, str) or name in names:
            raise ValueError(f'step {i}: duplicate or invalid name {name!r}')
        names.append(name)
        for ref in refs([step.get('params', {}), step.get('await_server')]):
            j = index_of(ref['from'], names[:-1])
            if j is None:
                raise ValueError(f'step {i} ({name}): {ref["from"]!r} is not an earlier step')
            if held_out(steps[j]):
                raise ValueError(f'step {i} ({name}) reads step {j} ({names[j]}), a held-out '
                                 'evaluation (AIME 2025 or the coding eval split)')
        if op == 'pairs' and held_out(dict(operation='test', params=step.get('params', {}))):
            raise ValueError(f'step {i} ({name}) generates from a held-out set')
        if op in ('objective', 'dataset', 'pairs'):
            for text in strings(step.get('params', {})):
                if HELD_OUT.search(text):
                    raise ValueError(f'step {i} ({name}) names {text!r}: AIME 2025 is held out')
    return names


def index_of(ref, names):
    if isinstance(ref, int) and not isinstance(ref, bool):
        return ref if 0 <= ref < len(names) else None
    return names.index(ref) if ref in names else None


def lookup(output, key):
    value = output
    for part in key.split('.'):
        if isinstance(value, dict) and part in value:
            value = value[part]
        elif isinstance(value, list) and part.isdigit() and int(part) < len(value):
            value = value[int(part)]
        else:
            raise KeyError(key)
    return value


def resolve(value, outputs, names):
    if is_ref(value):
        output = outputs[index_of(value['from'], names)]
        try:
            return copy.deepcopy(lookup(output, value['key']))
        except KeyError:
            if 'default' in value:
                return copy.deepcopy(value['default'])
            raise ValueError(f'step {value["from"]!r} output has no {value["key"]!r}') from None
    if isinstance(value, dict):
        return {k: resolve(v, outputs, names) for k, v in value.items()}
    if isinstance(value, list):
        return [resolve(v, outputs, names) for v in value]
    return value


def earlier_steps(store, key):
    """{index: last pipeline_step result} of an earlier pipeline."""
    old = store.get(key)
    if old is None or old.state['operation'] != 'pipeline':
        raise ValueError(f'resume: {key} is not a pipeline job')
    out = {}
    for r in old.results:
        if r.get('type') == 'pipeline_step':
            out[r['index']] = r
    return out


def run_pipeline_job(job):
    """Operation `pipeline` (see the module doc)."""
    from ops import OPERATIONS
    p = job.params
    unknown = set(p) - {'steps', 'resume', 'description'}  # description: the pipeline file's note, kept with the job
    if unknown:
        raise ValueError(f'unknown pipeline parameters {sorted(unknown)}')
    steps = p.get('steps')
    names = check_steps(steps)
    previous = earlier_steps(job.store, p['resume']) if p.get('resume') else {}
    outputs, summary = [], []
    total = len(steps)
    reusing = True  # once a step runs again, later steps run again too
    for i, step in enumerate(steps):
        job.check_cancel()
        op, name = step['operation'], names[i]
        params = resolve(step.get('params') or {}, outputs, names)
        old = previous.get(i)
        old_job = job.store.get(old['job']) if old else None
        if (reusing and old and old_job is not None and old.get('status') == 'done'
                and old.get('operation') == op and old.get('params') == params
                and old_job.state['status'] == 'done'):
            output = dict(old_job.state.get('output') or {}, job=old_job.key, dir=str(old_job.dir))
            outputs.append(output)
            job.emit(dict(type='pipeline_step', index=i, name=name, operation=op,
                          job=old_job.key, status='done', reused=True, params=params,
                          output=old_job.state.get('output')))
            summary.append(dict(name=name, operation=op, job=old_job.key, status='done',
                                reused=True))
            job.progress(done=i + 1, total=total, stage=name, step=f'{i + 1}/{total} {name}: reused')
            continue
        first_rerun, reusing = reusing, False
        run_params = dict(params)
        resumed_from = None
        # Only the first step that runs again continues its earlier child: a
        # later step's inputs (a profile uploaded under the same name) may
        # have changed while its requests look the same.
        if first_rerun and old_job is not None and op in RESUMABLE and old.get('operation') == op and (
                op != 'train' or old.get('params') == params):
            # test and pairs reuse only rows whose request matches; spsa.py
            # resumes only the same settings, so train only for equal params.
            resumed_from = old_job.key
            if op == 'train':
                run_params.setdefault('resume', old_job.key)
            else:
                earlier = run_params.get('resume')
                earlier = [] if earlier is None else (earlier if isinstance(earlier, list)
                                                      else [earlier])
                run_params['resume'] = earlier + [{'job': old_job.key}]
        context = job.context
        if step.get('gufo_url'):
            api_key = context.gufo.api_key if context.gufo else None
            context = dataclasses.replace(context, gufo=client.Gufo(step['gufo_url'], api_key))
        if step.get('await_server'):
            await_server(job, context.gufo, resolve(step['await_server'], outputs, names), name)
        child = job.store.create(op, run_params, parent=job.key)
        job.emit(dict(type='pipeline_step', index=i, name=name, operation=op, job=child.key,
                      status='running', params=params,
                      resumed_from=resumed_from))
        job.progress(done=i, total=total, stage=name, step=f'{i + 1}/{total} {name}')
        thread = threading.Thread(target=jobs.run_job, args=(child, context, OPERATIONS),
                                  daemon=True, name=f'pipeline-{name}')
        thread.start()
        last = None
        while thread.is_alive():
            if job.cancel_event.wait(0.5):
                child.request_cancel()
                thread.join()
                job.emit(dict(type='pipeline_step', index=i, name=name, operation=op,
                              job=child.key, status='cancelled', params=params))
                raise jobs.Cancelled()
            now = child.state['progress'].get('step')
            if now != last:
                last = now
                job.progress(step=f'{i + 1}/{total} {name}: {now}')
        state = child.view()
        job.emit(dict(type='pipeline_step', index=i, name=name, operation=op, job=child.key,
                      status=state['status'], params=params, output=state.get('output'),
                      error=state.get('error')))
        summary.append(dict(name=name, operation=op, job=child.key, status=state['status']))
        if state['status'] != 'done':
            job.set_output(dict(steps=summary))
            raise RuntimeError(f'step {i} ({name}, job {child.key}) {state["status"]}'
                               + (f': {state["error"]}' if state.get('error') else ''))
        outputs.append(dict(state.get('output') or {}, job=child.key, dir=str(child.dir)))
        job.progress(done=i + 1, total=total, step=f'{i + 1}/{total} {name}: done')
    job.set_output(dict(steps=summary))
    job.progress(stage='done', step='done')
