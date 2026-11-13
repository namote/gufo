"""Command line: every operation runs as a job in the work folder, through the
same code as the HTTP service, and prints its results as they arrive.

    python3 -I tools/steer [--work DIR] [--gufo-url URL] [--collections-dir DIR] COMMAND ...

The gufo API key, when the server has one, is read from the environment
variable GUFO_API_KEY (never from the command line). See README.md.
"""
import argparse
import json
import os
from pathlib import Path
import sys
import threading

import client
import data
import jobs
import ops

DEFAULT_WORK = Path(os.environ.get('XDG_STATE_HOME') or Path.home() / '.local' / 'state') / 'gufo-steer'


def _source(spec):
    """NAME=PROBLEMS,ROWS -> {name, problems, traces}."""
    name, sep, rest = spec.partition('=')
    problems, comma, rows = rest.partition(',')
    if not (sep and comma and name and problems and rows):
        raise argparse.ArgumentTypeError('a source is NAME=PROBLEMS.jsonl,ROWS.jsonl')
    return dict(name=name, problems=problems, traces=rows)


def _ids(path):
    if not path:
        return []
    text = Path(path).read_text()
    try:
        value = json.loads(text)
        return [str(x) for x in value]
    except ValueError:
        return text.split()


def _layers(text):
    return [int(x) for x in text.split(',') if x.strip()]


def _config(text):
    name, sep, body = text.partition('=')
    if not sep:
        raise argparse.ArgumentTypeError('a configuration is NAME=JSON or NAME=@FILE')
    if body.startswith('@'):
        body = Path(body[1:]).read_text()
    value = json.loads(body) if body.strip() else {}
    if not isinstance(value, dict):
        raise argparse.ArgumentTypeError('a configuration body is a JSON object')
    return dict(value, name=name)


def parser():
    ap = argparse.ArgumentParser(prog='steer', description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--work', type=Path, default=DEFAULT_WORK,
                    help=f'datasets and jobs (default {DEFAULT_WORK})')
    ap.add_argument('--gufo-url', help='the gufo server, e.g. http://127.0.0.1:18741')
    ap.add_argument('--collections-dir',
                    help="local path of the gufo server's --collect-dir (fit)")
    ap.add_argument('--poll-seconds', type=float, default=2.0)
    ap.add_argument('--json', action='store_true', help='print results as JSON lines')
    sub = ap.add_subparsers(dest='command', required=True)

    ds = sub.add_parser('dataset', help='build or import a dataset')
    dsub = ds.add_subparsers(dest='builder', required=True)
    tr = dsub.add_parser('traces', help='thought-boundary labels from reasoning traces')
    pr = dsub.add_parser('pairs', help='CAA pairs: instruction + prompt against the bare prompt')
    for b in (tr, pr):
        b.add_argument('--name', required=True)
        b.add_argument('--source', type=_source, action='append', required=True,
                       help='NAME=PROBLEMS.jsonl,ROWS.jsonl (repeatable, in order)')
        b.add_argument('--suffix', default='', help='appended to every problem text')
        b.add_argument('--max-chars', type=int, default=data.MAX_CHARS)
        b.add_argument('--exclude-ids', help='file of problem ids to leave out (JSON list or words)')
        b.add_argument('--replace', action='store_true')
    tr.add_argument('--min-each', type=int, default=data.MIN_EACH)
    tr.add_argument('--opener', default=data.RECHECK,
                    help='regex for the opener of a labelled thought (default: re-check)')
    tr.add_argument('--seed', type=int, default=0)
    tr.add_argument('--reasoning-field', default='reasoning')
    tr.add_argument('--control-name', help='default NAME-shuf')
    pr.add_argument('--instruction', help='instruction text (or --instruction-file)')
    pr.add_argument('--instruction-file', type=Path)
    pr.add_argument('--mode', choices=('answers', 'segments'), default='answers')
    pr.add_argument('--answer-field', help='answers: text, answer, content; segments: reasoning')
    pr.add_argument('--keep-unfinished', action='store_true',
                    help='answers: keep rows whose finish_reason is not stop')
    im = dsub.add_parser('import', help='an existing collector JSONL')
    im.add_argument('--name', required=True)
    im.add_argument('--path', required=True)
    im.add_argument('--mode', choices=data.MODES)
    im.add_argument('--control', help='a control JSONL (imported as NAME-control)')
    im.add_argument('--replace', action='store_true')
    sub.add_parser('datasets', help='list datasets')

    fit = sub.add_parser('fit', help='collect through gufo and fit per-layer directions')
    src = fit.add_mutually_exclusive_group(required=True)
    src.add_argument('--dataset')
    src.add_argument('--collection', help='an existing local collection (no gufo call)')
    fit.add_argument('--control-collection')
    fit.add_argument('--no-control', action='store_true')
    fit.add_argument('--collect-only', action='store_true',
                     help='stop after the gufo collections (no NumPy needed)')
    fit.add_argument('--collection-name', help='gufo collection name (default DATASET-HASH)')
    fit.add_argument('--top-n', type=int, default=4)
    fit.add_argument('--profile-layers', type=_layers, help='e.g. 32,45 (default: the top 2)')
    fit.add_argument('--alpha', type=float, default=1.0)
    fit.add_argument('--additive-kind', choices=('mean_diff', 'projected'), default='mean_diff')
    fit.add_argument('--prefix', help='profile name prefix (default the dataset)')

    te = sub.add_parser('test', help='a generation set under configurations')
    te.add_argument('--params', type=Path, help='JSON {set, configs, concurrency, ...}')
    te.add_argument('--set-kind', choices=('math', 'coding'))
    te.add_argument('--set', dest='set_path', help='math JSONL or coding set folder')
    te.add_argument('--split')
    te.add_argument('--ids-file', help='problem ids (JSON list or words)')
    te.add_argument('--max-tokens', type=int)
    te.add_argument('--config', type=_config, action='append',
                    help='NAME=JSON or NAME=@FILE {abliteration, logit_bias_profiles, '
                         'logit_bias_overrides, reasoning_effort, sampling} (repeatable)')
    te.add_argument('--concurrency', type=int)

    tn = sub.add_parser('train', help='SPSA strengths (spsa.py options after --)')
    tn.add_argument('spsa_args', nargs=argparse.REMAINDER)

    ld = sub.add_parser('load', help='upload a profile to gufo')
    ld.add_argument('--kind', choices=tuple(client.PROFILE_KINDS), default='abliteration')
    ld.add_argument('--name', required=True)
    g = ld.add_mutually_exclusive_group(required=True)
    g.add_argument('--file')
    g.add_argument('--job', help='a fit job key (with --profile, default --name)')
    ld.add_argument('--profile')
    ul = sub.add_parser('unload', help='remove a runtime profile from gufo')
    ul.add_argument('--kind', choices=tuple(client.PROFILE_KINDS), default='abliteration')
    ul.add_argument('--name', required=True)

    sv = sub.add_parser('serve', help='the HTTP job API')
    sv.add_argument('--host', default='127.0.0.1')
    sv.add_argument('--port', type=int, default=18790)

    js = sub.add_parser('jobs', help='list jobs, or show one')
    js.add_argument('key', nargs='?')
    js.add_argument('--since', type=int, default=0)
    return ap


def params_for(args):
    """(operation, params) of a job command."""
    if args.command == 'dataset':
        if args.builder == 'import':
            p = dict(builder='import', name=args.name, path=args.path, replace=args.replace)
            if args.mode:
                p['mode'] = args.mode
            if args.control:
                p['control'] = args.control
            return 'dataset', p
        p = dict(builder=args.builder, name=args.name, sources=args.source, suffix=args.suffix,
                 max_chars=args.max_chars, exclude_ids=_ids(args.exclude_ids),
                 replace=args.replace)
        if args.builder == 'traces':
            p.update(min_each=args.min_each, opener=args.opener, seed=args.seed,
                     reasoning_field=args.reasoning_field)
            if args.control_name:
                p['control_name'] = args.control_name
        else:
            instruction = args.instruction
            if args.instruction_file:
                instruction = args.instruction_file.read_text().strip()
            p.update(instruction=instruction, mode=args.mode,
                     require_stop=not args.keep_unfinished)
            if args.answer_field:
                p['answer_field'] = args.answer_field
        return 'dataset', p
    if args.command == 'fit':
        p = dict(top_n=args.top_n, alpha=args.alpha, additive_kind=args.additive_kind,
                 control=not args.no_control)
        if args.collect_only:
            p['collect_only'] = True
        for key in ('dataset', 'collection', 'control_collection', 'collection_name',
                    'profile_layers', 'prefix'):
            if getattr(args, key):
                p[key] = getattr(args, key)
        if args.collections_dir:
            p['collections_dir'] = args.collections_dir
        return 'fit', p
    if args.command == 'test':
        p = json.loads(args.params.read_text()) if args.params else {}
        spec = dict(p.get('set') or {})
        if args.set_kind:
            spec['kind'] = args.set_kind
        if args.set_path:
            spec['path'] = args.set_path
        if args.split:
            spec['split'] = args.split
        if args.ids_file:
            spec['ids'] = _ids(args.ids_file)
        if args.max_tokens:
            spec['max_tokens'] = args.max_tokens
        p['set'] = spec
        if args.config:
            p['configs'] = list(p.get('configs') or []) + args.config
        if args.concurrency:
            p['concurrency'] = args.concurrency
        return 'test', p
    if args.command == 'train':
        rest = list(args.spsa_args)
        if rest and rest[0] == '--':
            rest = rest[1:]
        return 'train', dict(args=rest)
    raise AssertionError(args.command)


def describe(item):
    """One printed line per result."""
    t = item.get('type')
    if t == 'layer':
        snr = f'{item["snr"]:.2f}' if item['snr'] is not None else '-'
        return (f'{item["dataset"]:7s} L{item["layer"]:<3d} acc {item["separated"]}/{item["pairs"]} '
                f'snr {snr} |d| {item["norm"]:.3f}')
    if t == 'row':
        if item.get('error'):
            return f'{item["config"]} {item["id"]} error {item["error"]}'
        return (f'{item["config"]} {item["id"]} correct={item.get("correct")} tokens={item["tokens"]} '
                f'hes={item["hesitations"]} trunc={item["truncated"]} {item["seconds"]}s')
    if t == 'totals':
        return (f'  {item["config"]}: {item["correct"]}/{item["n"]} correct, tokens mean '
                f'{item["tokens_mean"]} median {item["tokens_median"]}, truncated {item["truncated"]}, '
                f'hes {item["hesitations_mean"]}, sec {item["seconds_mean"]}')
    if t == 'cosine':
        return f'{item["dataset"]} cross-layer cosines: {item["layers"]} x {item["layers"]}'
    if t == 'step':
        e = item.get('eval')
        line = f'step {item["step"] + 1}' if item['step'] >= 0 else 'init'
        if 'plus' in item:
            line += f' J+ {item["plus"]["j"]:.4f} J- {item["minus"]["j"]:.4f}'
        return line + (f' eval J {e["j"]:.4f}' if e else '')
    short = {k: v for k, v in item.items() if k != 'seq' and not (
        k in ('matrix', 'separated', 'residual', 'theta') and isinstance(v, (list, dict)))}
    return json.dumps(short, ensure_ascii=False)[:400]


def run_inline(store, context, operation, params, as_json=False, out=sys.stdout):
    job = store.create(operation, params)
    print(f'job {job.key} ({job.dir})', file=sys.stderr)
    thread = threading.Thread(target=jobs.run_job, args=(job, context, ops.OPERATIONS),
                              daemon=True)
    thread.start()
    since = 0
    try:
        while True:
            thread.join(0.3)
            view = job.view(since)
            for item in view['results']:
                print(json.dumps(item, ensure_ascii=False) if as_json else describe(item),
                      file=out, flush=True)
                since = item['seq']
            if not thread.is_alive():
                break
    except KeyboardInterrupt:
        job.request_cancel()
        thread.join()
    state = job.view()
    if state['status'] == 'failed':
        print(f'failed: {state["error"]}', file=sys.stderr)
        return 1
    print(f'{state["status"]}: job {job.key}', file=sys.stderr)
    return 0 if state['status'] == 'done' else 1


def main(argv=None):
    args = parser().parse_args(argv)
    key = os.environ.get('GUFO_API_KEY') or None
    gufo = client.Gufo(args.gufo_url, key) if args.gufo_url else None
    context = ops.Context(gufo=gufo, collections_dir=args.collections_dir,
                          poll_seconds=args.poll_seconds)
    if args.command == 'serve':
        import service
        store = jobs.JobStore(args.work, recover=True)
        runner = jobs.Runner(store, context, ops.OPERATIONS)
        server = service.make_server(service.Api(store, runner, context), args.host, args.port)
        if args.host not in ('127.0.0.1', '::1', 'localhost'):
            print('warning: the job API reads local files named in requests; '
                  'expose it only to yourself', file=sys.stderr)
        print(f'steer: serving http://{args.host}:{args.port} (work {args.work})', file=sys.stderr)
        import signal

        def terminate(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, terminate)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            runner.stop()
            server.server_close()
        return 0
    store = jobs.JobStore(args.work, recover=False)
    if args.command == 'datasets':
        for meta in data.list_datasets(args.work):
            print(json.dumps({k: meta[k] for k in ('name', 'kind', 'records', 'groups', 'control')}))
        return 0
    if args.command == 'jobs':
        if args.key:
            job = store.get(args.key)
            if job is None:
                print('no such job', file=sys.stderr)
                return 1
            print(json.dumps(job.view(args.since), ensure_ascii=False, indent=1))
        else:
            for state in store.list():
                print(f'{state["key"]} {state["operation"]:7s} {state["status"]:9s} '
                      f'{state["progress"]["done"]}/{state["progress"]["total"]} {state["error"] or ""}')
        return 0
    if args.command in ('load', 'unload'):
        if gufo is None:
            print('load and unload need --gufo-url', file=sys.stderr)
            return 2
        if args.command == 'unload':
            reply = gufo.delete_profile(args.kind, args.name)
        else:
            body = dict(file=args.file) if args.file else dict(job=args.job, profile=args.profile)
            reply = gufo.upload_profile(args.kind, args.name,
                                        ops.profile_bytes(store, args.kind, args.name, body))
        print(json.dumps(reply, ensure_ascii=False))
        return 0
    operation, params = params_for(args)
    return run_inline(store, context, operation, params, args.json)


if __name__ == '__main__':
    sys.exit(main())
