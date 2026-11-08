"""CPU tests of the pure parts of the Flash-Next serving supervisor (standard library only).

Fixtures from the live server on 2026-11-08: two recorded GFX fence readings
(fixtures/fence-samples.json) and the boot's admission manifest reduced to one
desktop client and one display infrastructure identity
(fixtures/admission-manifest.json). fixtures/host-sample.json places those
recorded identities on GPU file descriptors, plus one made-up process, to cover
each classification. Linux only (fcntl, /sys paths).
"""
import contextlib
import copy
import importlib.util
import io
import os
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

HERE = Path(__file__).resolve().parent
FIXTURES = HERE / 'fixtures'
sys.path.insert(0, str(HERE))

if os.name == 'posix':
    from ops import admission, desktop, fence, fence_read, journal  # noqa: E402
POSIX = unittest.skipUnless(os.name == 'posix', 'the supervisor is Linux only')


def load(name):
    return json.loads((FIXTURES / name).read_text())


def load_server():
    spec = importlib.util.spec_from_file_location('gufo_server', HERE / 'gufo-server.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@POSIX
class FenceTest(unittest.TestCase):
    def test_parse_recorded_gfx_ring(self):
        ring = fence.parse(load('fence-samples.json')[0]['raw'])
        self.assertEqual(ring, {'signaled': 0x023f811e, 'emitted': 0x023f811e, 'trailing_signaled': 0,
                                'trailing_emitted': 0, 'reset': 0x00260646})

    def test_parse_rejects_missing_or_inconsistent_gfx(self):
        raw = load('fence-samples.json')[0]['raw']
        with self.assertRaisesRegex(ValueError, 'GFX ring absent'):
            fence.parse(raw.replace('gfx_0.0.0', 'gfx_9.9.9'))
        with self.assertRaisesRegex(ValueError, 'inconsistent'):
            fence.parse(raw.replace('Last emitted                 0x023f811e', 'Last emitted 0x00000001', 1))
        with self.assertRaisesRegex(ValueError, 'duplicate ring'):
            fence.parse(raw + '\n--- ring 0 (gfx_0.0.0) ---\n')

    def test_pending_observations(self):
        first, second = load('fence-samples.json')
        p = fence.Pending()
        idle = p.observe(first)
        self.assertFalse(idle['pending'])
        self.assertIsNone(idle['gap_since_previous_end_s'])
        self.assertAlmostEqual(p.observe(second)['gap_since_previous_end_s'], second['begin'] - first['end'])
        busy = copy.deepcopy(second)
        busy['begin'], busy['end'] = second['end'] + 15, second['end'] + 15.0002
        busy['raw'] = busy['raw'].replace('Last emitted                 0x023f811e',
                                          'Last emitted                 0x023f8120', 1)
        self.assertTrue(p.observe(busy)['pending'])
        later = dict(busy, begin=busy['end'] + 15, end=busy['end'] + 15.0002)
        self.assertAlmostEqual(p.observe(later)['pending_sampled_lower_bound_s'], later['begin'] - busy['end'])
        with self.assertRaisesRegex(ValueError, 'overlap'):
            p.observe(dict(later))
        regress = dict(later, begin=later['end'] + 15, end=later['end'] + 15.0002, raw=first['raw'].replace(
            'Last emitted                 0x023f811e', 'Last emitted                 0x00000001', 1).replace(
            'Last signaled fence          0x023f811e', 'Last signaled fence          0x00000001', 1))
        with self.assertRaisesRegex(RuntimeError, 'regression'):
            p.observe(regress)

    def test_reader_path_validation(self):
        bdf = '0000:c6:00.0'
        ok = f'/sys/kernel/debug/dri/{bdf}/amdgpu_fence_info'
        self.assertEqual(str(fence_read.validate_path(ok, bdf)), ok)
        self.assertEqual(str(fence_read.validate_path('/sys/kernel/debug/dri/1/amdgpu_fence_info', bdf)),
                         '/sys/kernel/debug/dri/1/amdgpu_fence_info')
        for bad in ('/sys/kernel/debug/dri/../dri/1/amdgpu_fence_info', '/tmp/amdgpu_fence_info',
                    f'/sys/kernel/debug/dri/{bdf}/name', f'/sys/kernel/debug/dri/card0/amdgpu_fence_info'):
            with self.assertRaises(ValueError, msg=bad):
                fence_read.validate_path(bad, bdf)
        with self.assertRaisesRegex(ValueError, 'BDF'):
            fence_read.validate_path(ok, 'c6:00.0')


@POSIX
class HostClassificationTest(unittest.TestCase):
    def test_recorded_identities(self):
        host = desktop.classify(load('host-sample.json'), load('admission-manifest.json'))
        classes = [row['classification'] for row in host['GPU_FDs']]
        self.assertEqual(classes, ['verified_boot_desktop_client',  # exact boot desktop client
                                   'verified_boot_desktop_client',  # display infrastructure on DRI
                                   'unidentified_GPU_client',       # the same identity on /dev/kfd
                                   'unidentified_GPU_client',       # unknown process
                                   'desktop_background'])           # already classified, untouched
        summary = host['boot_desktop_classification']
        self.assertEqual(summary['PIDs'], sorted({host['GPU_FDs'][0]['PID'], host['GPU_FDs'][1]['PID']}))
        self.assertTrue(summary['no_KFD_exemption'] and summary['not_AI_KFD_exemption'])

    def test_identity_must_match_exactly(self):
        host = load('host-sample.json')
        host['GPU_FDs'][0]['verified_proc'] = dict(host['GPU_FDs'][0]['verified_proc'], starttime_ticks=1)
        host['GPU_FDs'][1]['PGID'] = -1  # outer row disagrees with its verified identity
        host = desktop.classify(host, load('admission-manifest.json'))
        self.assertEqual(host['GPU_FDs'][0]['classification'], 'unidentified_GPU_client')
        self.assertEqual(host['GPU_FDs'][1]['classification'], 'unidentified_GPU_client')

    def test_stale_boot_rejected(self):
        host = dict(load('host-sample.json'), boot_id='other-boot')
        with self.assertRaisesRegex(AssertionError, 'stale'):
            admission.classify(host, load('admission-manifest.json'))


@POSIX
class JournalTest(unittest.TestCase):
    def test_delta_flags_gpu_errors_and_keeps_raw(self):
        rows = [json.dumps({'MESSAGE': 'usb 1-1: new device'}),
                json.dumps({'MESSAGE': 'amdgpu 0000:c6:00.0: ring gfx timeout'})]
        outputs = iter(['{"MESSAGE": "boot"}\n-- cursor: s=abc\n', '\n'.join(rows) + '\n-- cursor: s=def\n'])
        run = lambda argv: subprocess.CompletedProcess(argv, 0, next(outputs), '')
        with tempfile.TemporaryDirectory() as d:
            k = journal.KernelDelta(Path(d), run)
            k.start()
            self.assertEqual(k.cursor, 's=abc')
            result = k.collect(raise_error=False)
            self.assertTrue(result['new_GPU_error'])
            self.assertEqual(result['original_rows'], 2)
            self.assertIn('ring gfx timeout', Path(result['path']).read_text())


@POSIX
class ServerTest(unittest.TestCase):
    def test_disk_admission(self):
        server = load_server()
        gib = 2**30
        self.assertTrue(server.disk_admitted(150 * gib, 0))
        self.assertTrue(server.disk_admitted(100 * gib, 50 * gib))
        self.assertFalse(server.disk_admitted(100 * gib, 49 * gib))
        self.assertFalse(server.disk_admitted(99 * gib, 500 * gib))
        self.assertTrue(server.disk_admitted(101 * gib, 500 * gib))  # credit capped at 50 GiB

    def test_command_pins_runtime_and_model(self):
        server = load_server()
        argv = server.build_command({'run_id': 'preview', 'container': 'gufo-preview'})
        self.assertEqual(argv[:3], ['docker', 'run', '--rm'])
        self.assertIn(server.IMAGE_ID, argv)
        self.assertEqual(argv[argv.index('--entrypoint') + 1], '/candidate/gufo')
        self.assertIn(f'type=bind,src={server.MODEL},dst=/models/fnx,readonly', argv)
        self.assertEqual(argv[argv.index('--kv-pool-bytes') + 1], str(server.KV_POOL_BYTES))
        self.assertEqual(argv[-1], '--log-progress')


@POSIX
class InstallTest(unittest.TestCase):
    def run_install(self, install, args):
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return install.main(args)

    def test_install_backup_and_check(self):
        install = importlib.import_module('install')
        with tempfile.TemporaryDirectory() as d:
            dest, backups = Path(d, 'gufo'), Path(d, 'backup')
            (dest / 'ops').mkdir(parents=True)
            backups.mkdir()
            (dest / 'gufo-server.py').write_text('old = 1\n')
            (dest / 'ops' / 'OWNER.lock').write_text('')
            (dest / 'ops' / 'DESKTOP-BASELINE.json').write_text('{"keep": true}\n')
            args = ['--from-dir', str(HERE), '--dest', str(dest), '--backup-root', str(backups)]
            self.assertEqual(self.run_install(install, args + ['--dry-run']), 0)
            self.assertEqual((dest / 'gufo-server.py').read_text(), 'old = 1\n')
            self.assertEqual(self.run_install(install, args), 0)
            for name in install.FILES:
                self.assertEqual((dest / name).read_bytes(), (HERE / name).read_bytes(), name)
            self.assertEqual((dest / 'ops' / 'DESKTOP-BASELINE.json').read_text(), '{"keep": true}\n')
            self.assertTrue((dest / 'ops' / 'OWNER.lock').exists())
            record = json.loads((dest / 'INSTALLED.json').read_text())
            self.assertEqual(Path(record['backup'], 'gufo-server.py').read_text(), 'old = 1\n')
            self.assertIn(Path(record['backup']).name, (backups / 'README.md').read_text())
            self.assertEqual(self.run_install(install, ['--check', '--dest', str(dest)]), 0)
            # A second install of the same files changes nothing and backs up nothing.
            self.assertEqual(self.run_install(install, args), 0)
            self.assertEqual(len([p for p in backups.iterdir() if p.is_dir()]), 1)
            (dest / 'ops' / 'fence.py').write_text('changed = True\n')
            self.assertEqual(self.run_install(install, ['--check', '--dest', str(dest)]), 1)
            self.assertEqual(self.run_install(install, ['--check', '--dest', d]), 2)  # no record there

    def test_refuses_while_a_supervisor_runs(self):
        install = importlib.import_module('install')
        with tempfile.TemporaryDirectory() as d:
            dest = Path(d, 'gufo')
            dest.mkdir()
            fake = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)',
                                     str(dest / 'gufo-server.py'), '_supervise'])
            try:
                (dest / 'server-state.json').write_text(json.dumps({'pid': fake.pid}))
                args = ['--from-dir', str(HERE), '--dest', str(dest), '--backup-root', d]
                self.assertEqual(self.run_install(install, args), 3)
                self.assertFalse((dest / 'gufo-server.py').exists())
                self.assertEqual(self.run_install(install, args + ['--allow-running']), 0)
                self.assertTrue((dest / 'gufo-server.py').exists())
            finally:
                fake.kill()
                fake.wait()


if __name__ == '__main__':
    unittest.main()
