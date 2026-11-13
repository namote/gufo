"""The tool's own HTTP job API (standard library only; README.md lists it).

Binds 127.0.0.1 by default. It reads and writes local files named in job
parameters, so never expose it to other users or hosts.
"""
import http.server
import json
import urllib.parse

import client
import data
import jobs
import ops

MAX_BODY = 64 << 20


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
            return self.jobs(method, parts[1:], query, body)
        if parts and parts[0] == 'datasets':
            return self.datasets(method, parts[1:])
        if parts and parts[0] == 'profiles' and len(parts) == 3:
            return self.profiles(method, parts[1], parts[2], body)
        raise ApiError(404, 'not_found', 'unknown path')

    def jobs(self, method, rest, query, body):
        if not rest:
            if method == 'GET':
                return 200, dict(object='list', data=self.store.list())
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

    def datasets(self, method, rest):
        if method != 'GET':
            raise ApiError(405, 'method_not_allowed', 'GET')
        if not rest:
            return 200, dict(object='list', data=data.list_datasets(self.store.work))
        if len(rest) == 1:
            try:
                meta, _ = data.load_dataset(self.store.work, rest[0])
            except (FileNotFoundError, ValueError):
                raise ApiError(404, 'unknown_dataset', 'no such dataset') from None
            return 200, meta
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
