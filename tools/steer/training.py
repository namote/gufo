"""Train job: SPSA over per-layer strengths with the direction fixed.

Wraps tools/qwen38_flash_next/abliteration/spsa.py unchanged: it runs as a
child process (`spsa_child.py`, which adds the gufo API key to spsa.py's
requests from the environment, never from the command line) with
`--base-url`, `--log` and `--output` set by the tool. Every other spsa.py
option comes from params `args` (a list of strings) and `options` (an object
{option: value}: `minibatch_chains: 24` is `--minibatch-chains 24`, true is a
bare flag, false or null leaves it out, lists and objects are sent as JSON).

Rows: `objective` (an objective job key) supplies `--pairs`, `--ordinary`
(its keep rows) and `--guard`; or give `pairs`, `ordinary`, `guard` paths.

Fast recipe (README.md, "Training server"; `fast_defaults`, default true):
spsa.py already scores theta+ and theta- in one burst and runs the chains of
a minibatch in parallel; the job fills in what is missing from the server's
`/props`: `--concurrency` 2 x sessions, and `--minibatch-chains` max(12,
sessions) when the rows carry chains and no minibatch option is given. It
warns (`warning` results) when the server is far from the recipe (fewer than
8 sessions, or a context above 4x the objective's longest row or above
65536), and fails before scoring when the context is shorter than that row
(`server_check`: "warn" (default) or "off").

Progress: spsa.py appends one JSON line per step to its log; the job emits
each as a `step` result (step, J+ and J-, theta, the per-layer strengths, and
the eval every --eval-every steps) and the best evaluated settings at the end
(`best`). With `base_profile` (a profile file, or {job, profile}) and
`profile_name`, the job also writes the trained profile (the base with the
best strengths) to `profiles/NAME.json` and emits a `profile` result.
`resume` (an earlier train job's key) copies its spsa.py log, so spsa.py
continues from its last step when the settings and rows are the same.
Cancelling the job terminates the child.

Direction (R11): `basis` (a direction_basis.py file) also learns the
direction in that subspace (spsa.py `--basis`; its options `max_angle`,
`c_dir`, `a_dir`, `dir_names`, `init_alpha` go in `options`). It needs
`base_profile`, which the job passes as `--base-profile`; the job sets
`--profile-output` (`best-profile.json`). Step results then carry `alpha`
and `angle`, `best` carries `alpha`, `angle` and the basis, and the trained
profile (here and in the profile operation) holds the tuned direction.
"""
import json
import math
import re
import os
from pathlib import Path
import shutil
import subprocess
import sys

import data

HERE = Path(__file__).resolve().parent
ABLITERATION = HERE.parent / 'qwen38_flash_next' / 'abliteration'
if str(ABLITERATION) not in sys.path:
    sys.path.insert(0, str(ABLITERATION))

import direction_family  # noqa: E402
import spsa  # noqa: E402

RESERVED = ('--base-url', '--log', '--output', '--profile-output')
ROW_OPTIONS = ('--pairs', '--ordinary', '--guard')
PARAMS = {'args', 'options', 'python', 'objective', 'pairs', 'ordinary', 'guard',
          'base_profile', 'profile_name', 'resume', 'fast_defaults', 'server_check', 'basis'}
MIN_SESSIONS = 8
MAX_CONTEXT = 65536
KINDS = ('attention', 'mlp', 'residual')


def option_value(args, name, default=None):
    """The value of `--name V` or `--name=V` (the last one), else default."""
    value = default
    for i, a in enumerate(args):
        if a == name and i + 1 < len(args):
            value = args[i + 1]
        elif a.startswith(name + '='):
            value = a.split('=', 1)[1]
    return value


def has_option(args, name):
    return any(a == name or a.startswith(name + '=') for a in args)


def options_args(options):
    if options is None:
        return []
    if not isinstance(options, dict):
        raise ValueError('options is an object {option: value}')
    out = []
    for key, value in options.items():
        flag = '--' + str(key).lstrip('-').replace('_', '-')
        if value is None or value is False:
            continue
        if value is True:
            out.append(flag)
        elif isinstance(value, (list, dict)):
            out += [flag, json.dumps(value)]
        else:
            out += [flag, str(value)]
    return out


INSPECTED = ('--concurrency', '--minibatch-chains', '--minibatch', '--steps', '--layers',
             '--block-size', '--pairs', '--ordinary', '--guard', '--basis')


# spsa.py's own option names (`--a`, `--c` and `--A` are full names, not prefixes of `--concurrency` or `--answer-tokens`).
SPSA_OPTIONS = frozenset(re.findall(r"add_argument\('(--[A-Za-z0-9-]+)'", Path(spsa.__file__).read_text()))


def check_args(args):
    for a in args:
        option = a.split('=', 1)[0]
        if option in SPSA_OPTIONS and option not in RESERVED:
            continue
        if option.startswith('--') and option not in INSPECTED and any(
                full.startswith(option) for full in INSPECTED):
            raise ValueError(f'spell {option} in full (the tool reads it)')
        # argparse accepts unambiguous prefixes (--out for --output).
        if option.startswith('--') and len(option) > 2 and any(
                r.startswith(option) for r in RESERVED):
            raise ValueError(f'{option} is set by the tool')


def strengths(theta, layers, block):
    """Per-layer {attention, mlp, residual} from a log theta (block parameters)."""
    nb = -(-layers // block)
    if len(theta) != 3 * nb:
        return None
    flat = spsa.expand(theta, layers, block)
    return {k: [round(x, 5) for x in flat[i * layers:(i + 1) * layers]] for i, k in enumerate(KINDS)}


def trained_profile(base, settings):
    """The base profile with the trained strengths (and the direction
    settings it was trained with) as its defaults."""
    profile = dict(base)
    layers = profile.get('layers')
    for kind in KINDS:
        if kind in settings:
            value = settings[kind]
            if not isinstance(layers, int):
                raise ValueError('the base profile has no layers count')
            value = list(value) if isinstance(value, list) else [float(value)] * layers
            if len(value) != layers or not all(isinstance(x, (int, float)) and math.isfinite(x)
                                               for x in value):
                raise ValueError(f'{kind}: {layers} finite strengths expected')
            profile[kind] = [float(x) for x in value]
    if profile.get('mode') != 'add':
        for setting, field in (('direction_layer', 'direction_layer'),
                               ('direction_kind', 'direction_kind'),
                               ('directions', 'direction_count'),
                               ('preserve_norm', 'preserve_norm')):
            if setting in settings:
                profile[field] = settings[setting]
    return profile


def base_path(store, base):
    """The file of a base profile: a file path or {job, profile} (a profile
    a job wrote)."""
    if isinstance(base, str):
        return Path(base)
    if isinstance(base, dict) and base.get('job'):
        other = store.get(base['job'])
        if other is None:
            raise FileNotFoundError(f'no job {base["job"]}')
        name = data.check_name(base.get('profile') or '')
        return other.dir / 'profiles' / f'{name}.json'
    raise ValueError('base_profile is a path or {job, profile}')


def load_base(store, base):
    """A base profile: a file path or {job, profile} (a profile a job wrote)."""
    return json.loads(base_path(store, base).read_text())


def best_profile(base, best, basis=None):
    """The trained profile of a train job's `best`: `base` with the best
    strengths and, for a direction run (`alpha` in best), the tuned direction
    from the basis (`basis`, default the one best names)."""
    settings = best['settings']
    if best.get('alpha') is not None:
        path = basis or best.get('basis')
        if not path:
            raise ValueError('the best settings carry alpha but no basis')
        if best.get('basis_sha256') and direction_family.file_sha256(path) != best['basis_sha256']:
            raise ValueError(f'{path} is not the basis the run used')
        loaded = direction_family.load_basis(path)
        layer, kind = settings['direction_layer'], settings.get('direction_kind', 'mean_diff')
        direction_family.check_base(base, loaded, layer, kind)
        base = direction_family.profile_with_direction(base, loaded, best['alpha'], layer, kind)
    return trained_profile(base, settings)


def write_profile(job, name, profile):
    data.check_name(name)
    out = job.dir / 'profiles'
    out.mkdir(exist_ok=True)
    path = out / f'{name}.json'
    path.write_text(json.dumps(profile, separators=(',', ':')) + '\n')
    item = dict(type='profile', name=name, path=str(path), mode=profile.get('mode', 'suppress'),
                direction_layer=profile.get('direction_layer'),
                direction_kind=profile.get('direction_kind'),
                **{k: profile.get(k) for k in KINDS})
    job.emit(item)
    return item


def server_recipe(job, gufo, args, rows_have_chains, context_needed, fast, check, earlier=None):
    """Fill in the fast-recipe defaults from /props and report or refuse a
    server far from it. `earlier` is a resumed job's applied defaults, kept
    so spsa.py's log header still matches. Returns (extra args, recipe)."""
    recipe = dict(sessions=None, n_ctx=None, context_needed=context_needed, applied={})
    if earlier is not None:
        extra = []
        for key, value in earlier.items():
            flag = '--' + key.replace('_', '-')
            if not has_option(args, flag):
                extra += [flag, str(value)]
                recipe['applied'][key] = value
        args = args + extra
    else:
        extra = []
    try:
        props = gufo.props()
    except Exception as error:  # an older server or a fake: no checks, no defaults
        job.emit(dict(type='warning', code='no_props',
                      message=f'/props failed ({type(error).__name__}); no server check'))
        return extra, recipe
    sessions = props.get('total_slots')
    n_ctx = (props.get('default_generation_settings') or {}).get('n_ctx')
    recipe.update(sessions=sessions, n_ctx=n_ctx)
    if check != 'off':
        if context_needed and n_ctx and n_ctx < context_needed:
            raise ValueError(f'the server context {n_ctx} is shorter than the longest row '
                             f'(about {context_needed} tokens): restart it with --context '
                             f'{context_needed}, or cap the objective shorter')
        if sessions is not None and sessions < MIN_SESSIONS:
            job.emit(dict(type='warning', code='few_sessions',
                          message=f'{sessions} sessions: the fast recipe runs 24-48 '
                                  '(README.md, "Training server")'))
        if n_ctx and (n_ctx > MAX_CONTEXT or (context_needed and n_ctx > 4 * context_needed)):
            job.emit(dict(type='warning', code='long_context',
                          message=f'context {n_ctx} is far above the longest row '
                                  f'({context_needed or "unknown"}): a short native-RoPE '
                                  'context fits more sessions'))
    if fast and sessions:
        if not has_option(args, '--concurrency'):
            extra += ['--concurrency', str(2 * sessions)]
            recipe['applied']['concurrency'] = 2 * sessions
        if rows_have_chains and not (has_option(args, '--minibatch-chains')
                                     or has_option(args, '--minibatch')):
            chains = max(12, sessions)
            extra += ['--minibatch-chains', str(chains)]
            recipe['applied']['minibatch_chains'] = chains
    return extra, recipe


def _rows_have_chains(paths):
    for path in paths:
        with open(path, encoding='utf-8') as handle:
            for line in handle:
                if line.strip():
                    return 'chain' in json.loads(line)
    return False


def run_train_job(job):
    """Operation `train` (see the module doc)."""
    p = job.params
    unknown = set(p) - PARAMS
    if unknown:
        raise ValueError(f'unknown train parameters {sorted(unknown)}')
    args = p.get('args') or []
    if not (isinstance(args, list) and all(isinstance(a, str) for a in args)):
        raise ValueError('args is a list of spsa.py options')
    args = list(args) + options_args(p.get('options'))
    check_args(args)
    context_needed = None
    rows = {}
    if p.get('objective'):
        other = job.store.get(p['objective'])
        if other is None or not other.state.get('output'):
            raise ValueError(f'objective {p["objective"]} is not a finished objective job')
        out = other.state['output']
        rows = dict(pairs=out.get('pairs'), ordinary=out.get('keep'), guard=out.get('guard'))
        context_needed = out.get('context_tokens')
    for name in ('pairs', 'ordinary', 'guard'):
        if p.get(name):
            rows[name] = p[name]
    for name, path in rows.items():
        if not path:
            continue
        if has_option(args, f'--{name}'):
            raise ValueError(f'--{name} is given twice (objective or {name}, and args)')
        args += [f'--{name}', str(path)]
    if not args:
        raise ValueError('train needs args or options (spsa.py options) and its rows')
    profile_name = p.get('profile_name')
    base = load_base(job.store, p['base_profile']) if p.get('base_profile') else None
    if profile_name:
        data.check_name(profile_name)
        if base is None:
            raise ValueError('profile_name needs base_profile')
    basis = p.get('basis')
    if basis is None and has_option(args, '--basis'):
        raise ValueError('give the direction basis as the basis param, not --basis '
                         '(the job then sets --base-profile and --profile-output)')
    if basis is not None:
        if not isinstance(basis, str) or not basis:
            raise ValueError('basis is a path')
        if has_option(args, '--basis'):
            raise ValueError('--basis is given twice (basis and args)')
        if base is None:
            raise ValueError('basis needs base_profile (the profile the direction family edits)')
        direction_family.load_basis(basis)  # fail before scoring
        args += ['--basis', basis]
        if not has_option(args, '--base-profile'):
            args += ['--base-profile', str(base_path(job.store, p['base_profile']))]
    gufo = job.context.gufo
    if gufo is None:
        raise ValueError('train needs a gufo server (--gufo-url)')
    check = p.get('server_check', 'warn')
    if check not in ('warn', 'off'):
        raise ValueError('server_check is warn or off')
    row_paths = [option_value(args, o) for o in ROW_OPTIONS if option_value(args, o)]
    earlier = job.store.get(p['resume']) if p.get('resume') else None
    if p.get('resume') and earlier is None:
        raise FileNotFoundError(f'resume: no job {p["resume"]}')
    applied = None
    if earlier is not None:
        applied = next((r['recipe']['applied'] for r in earlier.results
                        if r.get('type') == 'train'), None)
    extra, recipe = server_recipe(job, gufo, args, _rows_have_chains(row_paths), context_needed,
                                  p.get('fast_defaults', True), check, applied)
    args += extra
    steps = int(option_value(args, '--steps', 100))
    layers = int(option_value(args, '--layers', 48))
    block = int(option_value(args, '--block-size', 1))
    log = job.dir / 'spsa.jsonl'
    best = job.dir / 'best.json'
    tool_set = ['--base-url', gufo.base_url, '--log', str(log), '--output', str(best)]
    if basis is not None:
        tool_set += ['--profile-output', str(job.dir / 'best-profile.json')]
    if earlier is not None:
        for name in ('spsa.jsonl', 'spsa.jsonl.off.json'):
            if (earlier.dir / name).exists():
                shutil.copyfile(earlier.dir / name, job.dir / name)
    command = [p.get('python') or sys.executable, '-I', str(HERE / 'spsa_child.py'),
               *tool_set, *args]
    env = dict(os.environ)
    env.pop('STEER_GUFO_API_KEY', None)
    if gufo.api_key:
        env['STEER_GUFO_API_KEY'] = gufo.api_key
    job.emit(dict(type='train', args=args, steps=steps, layers=layers, block_size=block,
                  recipe=recipe))
    job.progress(done=0, total=steps, stage='spsa', step='starting')
    output = open(job.dir / 'spsa.out', 'w')
    env['STEER_PARENT_PID'] = str(os.getpid())  # spsa_child.py dies with this process
    child = subprocess.Popen(command, stdout=output, stderr=subprocess.STDOUT, env=env)
    seen = 0
    shape = (layers, block)
    try:
        while True:
            exited = child.poll() is not None
            seen = _emit_new(job, log, seen, steps, shape)
            if exited:
                break
            if job.cancel_event.wait(job.context.poll_seconds):
                break
    finally:
        if child.poll() is None:
            child.terminate()
            try:
                child.wait(10)
            except subprocess.TimeoutExpired:
                child.kill()
                child.wait()
        output.close()
    job.check_cancel()
    _emit_new(job, log, seen, steps, shape)
    if child.returncode != 0:
        tail = (job.dir / 'spsa.out').read_text(errors='replace').strip().splitlines()[-5:]
        raise RuntimeError(f'spsa.py exited with {child.returncode}: ' + ' | '.join(tail))
    result = json.loads(best.read_text()) if best.exists() else None
    if result is not None:
        job.emit(dict(type='best', **result))
    summary = dict(log=str(log), best=result, best_path=str(best) if result else None,
                   recipe=recipe)
    if profile_name and result is not None:
        item = write_profile(job, profile_name, best_profile(base, result, basis))
        summary.update(profile=profile_name, profile_path=item['path'])
    job.set_output(summary)


def _emit_new(job, log, seen, steps, shape):
    """Emit log lines after the first `seen`; returns the new count. The
    header line is skipped; a torn last line waits for the next poll."""
    if not log.exists():
        return seen
    lines = log.read_text().split('\n')
    complete = lines[:-1]  # the text after the last newline may be torn
    for line in complete[seen:]:
        seen += 1
        if not line.strip():
            continue
        record = json.loads(line)
        if record.get('type') == 'header':
            continue
        item = {k: v for k, v in record.items() if k != 'type'}
        item.update(j_plus=(record.get('plus') or {}).get('j'),
                    j_minus=(record.get('minus') or {}).get('j'),
                    eval_j=(record.get('eval') or {}).get('j'),
                    strengths=strengths(record.get('theta') or [], *shape))
        job.emit(dict(type='step', **item))
        if record.get('step', -1) >= 0:
            job.progress(done=record['step'] + 1, total=steps,
                         step=f'step {record["step"] + 1}/{steps}')
    return seen
