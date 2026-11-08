"""HC CLI and bounded CPU runner contracts; standard library only."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch


REPO = Path(__file__).resolve().parents[2]


class HcTools(unittest.TestCase):
    def test_namespace_help_without_optional_dependencies(self):
        result = subprocess.run([sys.executable, '-B', '-S', '-m', 'tools.exl3_quant.hc', '--help'],
                                cwd=REPO, capture_output=True, text=True, check=True)
        self.assertIn('inventory,quantize,refine', result.stdout)

    def check_module(self):
        # Hosted discovery may not have the repository root on sys.path.
        with patch.object(sys, 'path', [str(REPO), *sys.path]):
            import tools.exl3_quant.hc.check as check
        return check

    def test_runner_refuses_existing_output_and_cleans_owned_timeout(self):
        check = self.check_module()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch.object(check.subprocess, 'run') as run:
                with self.assertRaises(FileExistsError):
                    check.main(['--output', str(root)])
                run.assert_not_called()
            calls = []

            def fake_run(argv, **kwargs):
                calls.append(argv)
                if argv[:2] == ['docker', 'run']:
                    self.assertNotIn('--device', argv)
                    self.assertIn('readonly', ' '.join(argv))
                    raise subprocess.TimeoutExpired(argv, 120)
                return subprocess.CompletedProcess(argv, 0, stdout='', stderr='')

            with patch.object(check.subprocess, 'run', side_effect=fake_run):
                with self.assertRaises(subprocess.TimeoutExpired):
                    check.main(['--output', str(root / 'new')])
            name = calls[0][calls[0].index('--name') + 1]
            self.assertTrue(name.startswith('gufo-hc-cpu-'))
            self.assertEqual(calls[1], ['docker', 'rm', '-f', name])


if __name__ == '__main__':
    unittest.main()
