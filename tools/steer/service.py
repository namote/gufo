"""The tool's own HTTP job API (standard library only; README.md lists it).

Binds 127.0.0.1 by default. It reads and writes local files named in job
parameters, so never expose it to other users or hosts.
"""
import http.server
import json
from pathlib import Path
import urllib.parse

import client
import data
import jobs
import ops

MAX_BODY = 64 << 20
PAGE_LIMIT, MAX_PAGE_LIMIT = 20, 200
ROW_FILES = ('pairs', 'keep', 'guard')


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


class Api:
    def __init__(self, store, runner, context):
        self.store, self.runner, self.context = store, runner, context

    def handle(self, method, path, query, body):
        parts = [urllib.parse.unquote(p) for p in path.strip('/').split('/') if p]
        if parts == ['health']:
            return 200, dict(status='ok', gufo=self.context.gufo.describe()
                             if self.context.gufo else None)
        if parts and parts[0] == 'jobs':
            if len(parts) >= 3:
                return self.job_files(method, parts[1], parts[2:], query)
            return self.jobs(method, parts[1:], query, body)
        if parts and parts[0] == 'datasets':
            return self.datasets(method, parts[1:], query)
        if parts == ['profiles']:
            if method != 'GET':
                raise ApiError(405, 'method_not_allowed', 'GET')
            return 200, dict(object='list', data=profile_files(self.store))
        if parts and parts[0] == 'profiles' and len(parts) == 3:
            return self.profiles(method, parts[1], parts[2], body)
        if parts == ['gufo', 'props']:
            if method != 'GET':
                raise ApiError(405, 'method_not_allowed', 'GET')
            if self.context.gufo is None:
                raise ApiError(400, 'no_gufo', 'the tool runs without --gufo-url')
            return 200, self.context.gufo.props()
        raise ApiError(404, 'not_found', 'unknown path')

    def jobs(self, method, rest, query, body):
        if not rest:
            if method == 'GET':
                listed = self.store.list()
                parent = query.get('parent', [None])[0]
                if parent is not None:
                    listed = [j for j in listed if j.get('parent') == parent]
                return 200, dict(object='list', data=listed)
            if method == 'POST':
                request = parse(body)
                operation = request.get('operation')
                if operation not in jobs.OPERATIONS:
                    raise ApiError(400, 'unsupported_operation',
                                   f'operation must be one of {jobs.OPERATIONS}')
                params = request.get('params', {})
                if not isinstance(params, dict):
                    raise ApiError(400, 'invalid_params', 'params must be an object')
                job = self.runner.submit(operation, params)
                return 200, dict(job=job.key, status=job.state['status'])
            raise ApiError(405, 'method_not_allowed', 'GET or POST')
        if len(rest) != 1:
            raise ApiError(404, 'not_found', 'unknown path')
        key = rest[0]
        if method == 'GET':
            job = self.store.get(key)
            if job is None:
                raise ApiError(404, 'unknown_job', 'no such job')
            since = query.get('since', ['0'])[0]
            try:
                since = int(since)
                if since < 0:
                    raise ValueError
            except ValueError:
                raise ApiError(400, 'invalid_since', 'since is a non-negative integer') from None
            return 200, job.view(since)
        if method == 'DELETE':
            reply = self.store.delete(key)
            if reply is None:
                raise ApiError(404, 'unknown_job', 'no such job')
            return 200, reply
        raise ApiError(405, 'method_not_allowed', 'GET or DELETE')

    def datasets(self, method, rest, query=None):
        if method != 'GET':
            raise ApiError(405, 'method_not_allowed', 'GET')
        if not rest:
            return 200, dict(object='list', data=data.list_datasets(self.store.work))
        try:
            folder = data.dataset_dir(self.store.work, rest[0])
        except ValueError:
            raise ApiError(404, 'unknown_dataset', 'no such dataset') from None
        if not (folder / 'dataset.json').exists():
            raise ApiError(404, 'unknown_dataset', 'no such dataset')
        if len(rest) == 1:
            return 200, json.loads((folder / 'dataset.json').read_text())
        if rest[1:] == ['records']:
            return 200, page_jsonl(folder / 'records.jsonl', query or {})
        raise ApiError(404, 'not_found', 'unknown path')

    def job_files(self, method, key, rest, query):
        """Read-only views of a job's files: generations, objective rows and
        written profiles."""
        if method != 'GET':
            raise ApiError(405, 'method_not_allowed', 'GET')
        job = self.store.get(key)
        if job is None:
            raise ApiError(404, 'unknown_job', 'no such job')
        if rest == ['generations']:
            folder = job.dir / 'generations'
            files = sorted(folder.glob('*.jsonl')) if folder.is_dir() else []
            return 200, dict(object='list', data=[
                dict(config=f.stem, rows=count_lines(f), bytes=f.stat().st_size) for f in files])
        if len(rest) == 2 and rest[0] == 'generations':
            path = safe_child(job.dir / 'generations', rest[1], '.jsonl')
            return 200, page_jsonl(path, query)
        if rest == ['rows']:
            folder = job.dir / 'rows'
            return 200, dict(object='list', data=[
                dict(name=n, rows=count_lines(folder / f'{n}.jsonl'))
                for n in ROW_FILES if (folder / f'{n}.jsonl').exists()])
        if len(rest) == 2 and rest[0] == 'rows':
            if rest[1] not in ROW_FILES:
                raise ApiError(404, 'not_found', 'rows are pairs, keep or guard')
            return 200, page_jsonl(job.dir / 'rows' / f'{rest[1]}.jsonl', query)
        if len(rest) == 2 and rest[0] == 'profiles':
            path = safe_child(job.dir / 'profiles', rest[1], '.json')
            return 200, json.loads(path.read_text())
        raise ApiError(404, 'not_found', 'unknown path')

    def profiles(self, method, kind, name, body):
        gufo = self.context.gufo
        if gufo is None:
            raise ApiError(400, 'no_gufo', 'the tool runs without --gufo-url')
        if kind not in client.PROFILE_KINDS:
            raise ApiError(404, 'not_found', 'kind is abliteration or logit-bias')
        try:
            data.check_name(name)
        except ValueError as error:
            raise ApiError(400, 'invalid_profile_name', str(error)) from None
        if method == 'POST':
            try:
                raw = ops.profile_bytes(self.store, kind, name, parse(body))
            except (ValueError, OSError) as error:
                raise ApiError(400, 'invalid_profile_source', str(error)) from None
            return 200, gufo.upload_profile(kind, name, raw)
        if method == 'DELETE':
            return 200, gufo.delete_profile(kind, name)
        raise ApiError(405, 'method_not_allowed', 'POST or DELETE')


def safe_child(folder, name, suffix):
    """folder/NAME+suffix for a valid name that exists, else a 404."""
    try:
        data.check_name(name)
    except ValueError:
        raise ApiError(404, 'not_found', 'no such file') from None
    path = folder / f'{name}{suffix}'
    if not path.is_file():
        raise ApiError(404, 'not_found', 'no such file')
    return path


def count_lines(path):
    with open(path, 'rb') as handle:
        return sum(1 for line in handle if line.strip())


def _int_query(query, name, default, high=None):
    try:
        value = int(query.get(name, [default])[0])
    except ValueError:
        value = -1
    if value < 0:
        raise ApiError(400, f'invalid_{name}', f'{name} is a non-negative integer')
    return min(value, high) if high is not None else value


def page_jsonl(path, query):
    """A `page` of a JSONL file: ?offset (0), ?limit (20, at most 200),
    ?fields (comma-separated keys to keep), ?id (only rows with this id)."""
    if not Path(path).is_file():
        raise ApiError(404, 'not_found', 'no such file')
    offset = _int_query(query, 'offset', 0)
    limit = _int_query(query, 'limit', PAGE_LIMIT, MAX_PAGE_LIMIT)
    fields = [f for f in (query.get('fields', [''])[0]).split(',') if f]
    wanted = query.get('id', [None])[0]
    rows = []
    with open(path, encoding='utf-8') as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                rows.append(json.loads(line))
            except ValueError:
                break  # a torn last line while the job writes
    if wanted is not None:
        rows = [r for r in rows if r.get('id') == wanted]
    page = rows[offset:offset + limit]
    if fields:
        page = [{k: r[k] for k in fields if k in r} for r in page]
    return dict(object='page', offset=offset, limit=limit, total=len(rows), data=page)


def profile_files(store):
    """Every profile file a job wrote (WORK/jobs/KEY/profiles/NAME.json)."""
    out = []
    for state in store.list():
        folder = store.root / state['key'] / 'profiles'
        if not folder.is_dir():
            continue
        for path in sorted(folder.glob('*.json')):
            item = dict(name=path.stem, job=state['key'], operation=state['operation'],
                        path=str(path), bytes=path.stat().st_size, mtime=path.stat().st_mtime)
            try:
                profile = json.loads(path.read_text())
                item.update(mode=profile.get('mode', 'suppress'), layers=profile.get('layers'),
                            direction_layer=profile.get('direction_layer'),
                            direction_kind=profile.get('direction_kind'))
            except ValueError:
                item['error'] = 'not JSON'
            out.append(item)
    return out


def parse(body):
    try:
        value = json.loads(body or b'{}')
    except ValueError as error:
        raise ApiError(400, 'invalid_json', str(error)) from None
    if not isinstance(value, dict):
        raise ApiError(400, 'invalid_json', 'the body is a JSON object')
    return value


def make_server(api, host='127.0.0.1', port=18790):
    class Handler(http.server.BaseHTTPRequestHandler):
        protocol_version = 'HTTP/1.1'

        def log_message(self, *args):
            pass

        def _serve(self, method):
            url = urllib.parse.urlsplit(self.path)
            try:
                try:
                    length = int(self.headers.get('Content-Length') or 0)
                except ValueError:
                    length = -1
                if length < 0:
                    self.close_connection = True
                    raise ApiError(400, 'invalid_request', 'bad Content-Length')
                if length > MAX_BODY:
                    self.close_connection = True
                    raise ApiError(413, 'payload_too_large', 'at most 64 MiB')
                body = self.rfile.read(length) if length else b''
                status, reply = api.handle(method, url.path, urllib.parse.parse_qs(url.query),
                                           body)
            except ApiError as error:
                status, reply = error.status, dict(error=dict(code=error.code,
                                                               message=str(error)))
            except client.GufoError as error:
                status = error.status if 400 <= error.status < 600 else 502
                reply = dict(error=dict(code=error.code or 'gufo_error', message=str(error)))
            except (ValueError, OSError) as error:
                status, reply = 400, dict(error=dict(code='invalid_request', message=str(error)))
            except Exception as error:  # always answer, never drop the connection
                status, reply = 500, dict(error=dict(code='internal_error',
                                                     message=f'{type(error).__name__}: {error}'))
            payload = json.dumps(reply, ensure_ascii=False).encode()
            self.send_response(status)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            self._serve('GET')

        def do_POST(self):
            self._serve('POST')

        def do_DELETE(self):
            self._serve('DELETE')

    return http.server.ThreadingHTTPServer((host, port), Handler)
