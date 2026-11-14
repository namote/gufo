"""The lab page's static route (service.static_file) and its files.

Standard library only: python3 -I -m unittest discover -s tools/steer -p 'test_*.py'
"""
from html.parser import HTMLParser
import http.client
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parent))

import jobs  # noqa: E402
import ops  # noqa: E402
import service  # noqa: E402

WEB = Path(__file__).resolve().parent / 'web'


class StaticRouteTest(unittest.TestCase):
    def setUp(self):
        self.work = Path(tempfile.mkdtemp(prefix='steer-web-'))
        self.store = jobs.JobStore(self.work)
        self.runner = jobs.Runner(self.store, ops.Context(), ops.OPERATIONS)
        self.server = service.make_server(service.Api(self.store, self.runner, ops.Context()),
                                          port=0)
        self.port = self.server.server_address[1]
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    def tearDown(self):
        self.runner.stop()
        self.server.shutdown()
        self.server.server_close()
        shutil.rmtree(self.work, ignore_errors=True)

    def get(self, path, method='GET'):
        # http.client sends the path as given (urllib would normalize `..`).
        connection = http.client.HTTPConnection('127.0.0.1', self.port, timeout=10)
        try:
            connection.request(method, path)
            response = connection.getresponse()
            return response.status, response.getheader('Content-Type'), response.read(), \
                response.getheader('Content-Security-Policy')
        finally:
            connection.close()

    def test_index_and_types(self):
        status, kind, body, policy = self.get('/')
        self.assertEqual(status, 200)
        self.assertEqual(kind, 'text/html; charset=utf-8')
        self.assertEqual(body, (WEB / 'index.html').read_bytes())
        self.assertIn("connect-src 'self'", policy)
        expected = {'.html': 'text/html', '.css': 'text/css', '.js': 'text/javascript',
                    '.svg': 'image/svg+xml'}
        served = 0
        for path in sorted(WEB.rglob('*')):
            if path.is_file():
                self.assertIn(path.suffix, expected, f'{path.name}: add its type to the test')
                status, kind, body, policy = self.get('/web/' + path.relative_to(WEB).as_posix())
                self.assertEqual(status, 200, path)
                self.assertTrue(kind.startswith(expected[path.suffix]), (path, kind))
                self.assertEqual(body, path.read_bytes())
                self.assertIn("default-src 'none'", policy)
                served += 1
        self.assertGreaterEqual(served, 5)

    def test_traversal_and_unknown_refused(self):
        # A file the route must never serve: the job store beside the page.
        for path in ('/web/../service.py', '/web/%2e%2e/service.py', '/web/..%2fservice.py',
                     '/web/%2e%2e%2fservice.py', '/web/..%5cservice.py', '/web/.hidden.js',
                     '/web//etc/passwd', '/web/%2fetc%2fpasswd', '/web/', '/web/index.py',
                     '/web/missing.js', '/web/a%00.js', '/web/C:%5cx.js'):
            status, kind, body, _ = self.get(path)
            self.assertEqual(status, 404, path)
            self.assertEqual(kind, 'application/json', path)
            self.assertNotIn(b'import', body, path)

    def test_api_unchanged(self):
        status, kind, body, _ = self.get('/health')
        self.assertEqual(status, 200)
        self.assertEqual(kind, 'application/json')
        for method in ('POST', 'DELETE'):
            for path in ('/', '/web/app.js'):
                status, kind, _, _ = self.get(path, method=method)
                self.assertEqual(status, 404, (method, path))  # only GET serves the page
                self.assertEqual(kind, 'application/json')

    def test_symlink_escape_refused(self):
        web = self.work / 'web'
        web.mkdir()
        (web / 'ok.js').write_text('ok')
        (self.work / 'secret.js').write_text('secret')
        (web / 'link.js').symlink_to(self.work / 'secret.js')
        saved = service.WEB
        service.WEB = web
        try:
            self.assertEqual(service.static_file('/web/ok.js')[0], 200)
            self.assertEqual(service.static_file('/web/link.js')[0], 404)
        finally:
            service.WEB = saved

    def test_static_file_unit(self):
        self.assertIsNone(service.static_file('/jobs'))
        self.assertIsNone(service.static_file('/webx/app.js'))
        self.assertEqual(service.static_file('/web/../../etc/passwd')[0], 404)
        self.assertEqual(service.static_file('/web/app.js')[0], 200)


class ScriptTest(unittest.TestCase):
    """The page's scripts: valid syntax, and nothing the policy refuses or
    that would parse untrusted text (records, answers) as HTML."""

    def test_no_html_injection_or_inline_code(self):
        banned = re.compile(r'innerHTML|outerHTML|insertAdjacentHTML|document\.write|\beval\(|'
                            r'new Function|setAttribute\(\s*[\'"](?:style|on)')
        for path in sorted(WEB.glob('*.js')):
            for number, line in enumerate(path.read_text(encoding='utf-8').splitlines(), 1):
                self.assertIsNone(banned.search(line), f'{path.name}:{number}: {line.strip()}')

    @unittest.skipUnless(shutil.which('node'), 'node is not installed')
    def test_syntax(self):
        for path in sorted(WEB.glob('*.js')):
            result = subprocess.run(['node', '--check', str(path)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)


class References(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls, self.inline = [], []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        for key in ('src', 'href', 'action', 'data', 'poster'):
            if attrs.get(key) is not None:
                self.urls.append((tag, attrs[key]))
        if tag == 'script' and 'src' not in attrs:
            self.inline.append(tag)
        if 'style' in attrs or any(k.startswith('on') for k in attrs):
            self.inline.append(f'{tag} attribute')

    def handle_data(self, text):
        pass


class LocalOnlyTest(unittest.TestCase):
    """The page must work offline: it references only its own files."""

    def test_html_references_local_files(self):
        parser = References()
        parser.feed((WEB / 'index.html').read_text(encoding='utf-8'))
        self.assertTrue(parser.urls)
        for tag, url in parser.urls:
            if url.startswith('#'):
                continue
            self.assertRegex(url, r'^/web/[A-Za-z0-9_./-]+$', (tag, url))
            self.assertNotIn('..', url)
            self.assertTrue((WEB / url[len('/web/'):]).is_file(), url)
        self.assertEqual(parser.inline, [], 'the page policy refuses inline code and styles')

    def test_no_remote_urls_in_code(self):
        remote = re.compile(r'''(?:https?:)?//[A-Za-z0-9.-]+\.[a-z]{2,}[/'"`]|@import|'''
                            r'''\burl\(\s*['"]?(?:https?:|//)''')
        for path in sorted(WEB.rglob('*')):
            if path.suffix in ('.html', '.css', '.js'):
                text = path.read_text(encoding='utf-8')
                for line in text.splitlines():
                    if remote.search(line) and 'www.w3.org/2000/svg' not in line:
                        self.fail(f'{path.name}: remote reference: {line.strip()}')


if __name__ == '__main__':
    unittest.main()
