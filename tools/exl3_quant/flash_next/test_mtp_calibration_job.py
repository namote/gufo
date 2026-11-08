"""CPU-only regressions for actual-lease parsing and isolated container policy."""
import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest

from mtp_calibration_job import (held_flock, build_command, disk_required,
                                 owned_path, OUT, TASK, RUNS, DATA, GIB)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.st = SimpleNamespace(st_dev=os.makedev(0x103, 2), st_ino=999)

    def test_requires_real_exclusive_whole_file_flock(self):
        good = '33: FLOCK ADVISORY WRITE 1319760 103:02:999 0 EOF'
        self.assertTrue(held_flock(good, 1319760, self.st))
        for bad in [good.replace('33:', '33: ->'), good.replace('FLOCK', 'POSIX'),
                    good.replace('WRITE', 'READ'), good.replace('1319760', '123'),
                    good.replace(':999', ':998'), good.replace('103:02', '103:03'),
                    good.replace('0 EOF', '1 EOF')]:
            self.assertFalse(held_flock(bad, 1319760, self.st), bad)

    def test_cpu_only_command_build_protects_all_main_mounts(self):
        prior = json.loads((OUT / 'model-job-c03/module-060/STATE.json').read_text())
        original = prior['command'].copy()
        c = {'script':'mtp_calibration_probe.py', 'argv':['--work','/mtp-run/work']}
        result = build_command(prior, c, 'unique', 'label123', DATA / 'test')
        mounts = [result[i+1] for i, x in enumerate(result) if x == '--mount']
        self.assertIn(f'type=bind,src={OUT},dst=/output,readonly', mounts)
        self.assertEqual([m for m in mounts if not m.endswith(',readonly')],
                         [f'type=bind,src={DATA / "test"},dst=/mtp-run'])
        self.assertIn('GUFO_MTP_LEASE_VERIFIED=1', result)
        self.assertIn('GUFO_STATE_SCRATCH=/mtp-run/scratch', result)
        self.assertEqual(result[-3:], ['/task/mtp_calibration_probe.py','--work','/mtp-run/work'])
        self.assertEqual(prior['command'], original)

    def test_rejects_other_writable_mount(self):
        prior = json.loads((OUT / 'model-job-c03/module-060/STATE.json').read_text())
        i = prior['command'].index('--workdir')
        prior['command'][i:i] = ['--mount','type=bind,src=/tmp,dst=/unreviewed']
        with self.assertRaises(AssertionError):
            build_command(prior, {'script':'mtp_calibration_probe.py','argv':[]}, 'x','y',DATA/'test')

    def test_budget_preserves_state_and_refs_plus_margin(self):
        self.assertEqual(disk_required('probe', 81899552737), 150*GIB)
        self.assertEqual(disk_required('full', 81899552737, int(4.05*GIB), 16*GIB),
                         116*GIB + 81899552737 + int(4.05*GIB))
        for h, m in [(0,16*GIB),(int(4.05*GIB),0)]:
            with self.assertRaises(AssertionError): disk_required('full',81899552737,h,m)

    def test_output_scope_rejects_main_and_relative_paths(self):
        self.assertEqual(owned_path(str(RUNS/'unique'), RUNS), RUNS/'unique')
        for value in (str(OUT/'work'),str(RUNS),str(RUNS/'..'/'work'),'relative'):
            with self.assertRaises(AssertionError): owned_path(value, RUNS)


if __name__ == '__main__':
    unittest.main()
