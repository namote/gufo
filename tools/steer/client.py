"""A small gufo HTTP client (standard library only).

Only the endpoints the tool needs: Chat Completions, the activation collection
jobs (`/v1/abliteration/jobs`) and the runtime profile updates
(`/v1/abliteration/profiles/{name}`, `/v1/logit-bias/profiles/{name}`); see
docs/SERVER.md. The API key, when given, is sent as a bearer token and never
written to job files.
"""
import json
import time
import urllib.error
import urllib.parse
import urllib.request

PROFILE_KINDS = {'abliteration': '/v1/abliteration/profiles/',
                 'logit-bias': '/v1/logit-bias/profiles/'}


class GufoError(RuntimeError):
    def __init__(self, status, message, code=None):
        super().__init__(f'HTTP {status}{" " + code if code else ""}: {message}')
        self.status, self.code = status, code


class Gufo:
    def __init__(self, base_url, api_key=None, timeout=600.0):
        if not base_url:
            raise ValueError('a gufo base URL is required (--gufo-url)')
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.timeout = timeout

    def describe(self):
        return dict(base_url=self.base_url, api_key=bool(self.api_key))

    def request(self, method, path, body=None, raw=None, timeout=None, attempts=1):
        data = raw if raw is not None else (None if body is None else json.dumps(body).encode())
        headers = {'Content-Type': 'application/json'}
        if self.api_key:
            headers['Authorization'] = f'Bearer {self.api_key}'
        for attempt in range(attempts):
            request = urllib.request.Request(self.base_url + path, data=data, headers=headers,
                                             method=method)
            try:
                with urllib.request.urlopen(request, timeout=timeout or self.timeout) as response:
                    payload = response.read()
                return json.loads(payload) if payload else {}
            except urllib.error.HTTPError as error:
                with error:
                    detail = error.read().decode(errors='replace')
                code, message = None, detail[:1000]
                try:
                    parsed = json.loads(detail).get('error') or {}
                    code, message = parsed.get('code') or parsed.get('type'), parsed.get('message', message)
                except (ValueError, AttributeError):
                    pass
                raise GufoError(error.code, message, code) from None
            except (OSError, ValueError):
                # A dropped connection or a truncated body; HTTP errors are final.
                if attempt + 1 == attempts:
                    raise
                time.sleep(0.5 * (attempt + 1))

    # Chat ---------------------------------------------------------------------
    def chat(self, body, timeout=None, attempts=3):
        return self.request('POST', '/v1/chat/completions', body, timeout=timeout,
                            attempts=attempts)

    # Collection jobs ----------------------------------------------------------
    def start_collect(self, mode, name, records, resume=True):
        body = dict(operation='collect', mode=mode, name=name, records=records, resume=resume)
        return self.request('POST', '/v1/abliteration/jobs', body)['job']

    def collect_job(self, key, since=0):
        return self.request('GET', f'/v1/abliteration/jobs/{urllib.parse.quote(key)}?since={since}',
                            attempts=3)

    def cancel_collect(self, key):
        return self.request('DELETE', f'/v1/abliteration/jobs/{urllib.parse.quote(key)}')

    # Runtime profiles ---------------------------------------------------------
    def upload_profile(self, kind, name, data):
        """Add or replace a profile; `data` is the profile file's bytes."""
        return self.request('POST', PROFILE_KINDS[kind] + urllib.parse.quote(name), raw=data)

    def delete_profile(self, kind, name):
        return self.request('DELETE', PROFILE_KINDS[kind] + urllib.parse.quote(name))
