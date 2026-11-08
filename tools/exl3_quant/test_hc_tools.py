"""HC CLI and bounded CPU runner contracts; standard library only."""
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
import shutil
import hashlib
import json


REPO = Path(__file__).resolve().parents[2]


class HcTools(unittest.TestCase):
    def test_retained_capture_cache_requires_qualified_binary(self):
        with patch.object(sys, 'path', [str(REPO), *sys.path]):
            from tools.exl3_quant.hc.prepare_capture import cache_provenance
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); objects = root / 'objects'; objects.mkdir()
            reference = root / 'reference'; receipt = root / 'result.json'
            payload = b'synthetic retained binary'
            (objects / 'gufo').write_bytes(payload); reference.write_bytes(payload)
            digest = hashlib.sha256(payload).hexdigest()
            receipt.write_text(json.dumps({'trees': {'C': '12345678 source sha256 ' + digest}}))
            self.assertEqual(cache_provenance(objects, reference, receipt, '12345678abcd')['cache_binary_sha256'], digest)
            reference.write_bytes(b'different')
            with self.assertRaises(ValueError):
                cache_provenance(objects, reference, receipt, '12345678abcd')

    def test_capture_patch_supported_layout_and_fail_before_writing(self):
        with patch.object(sys, 'path', [str(REPO), *sys.path]):
            from tools.exl3_quant.hc.prepare_capture import instrument
        files = ['src/models/qwen38_flash_next/runtime/executor_hc.cpp',
                 'src/models/qwen38_flash_next/runtime/executor_forward.cpp',
                 'tests/models/qwen38_flash_next/rq_logits_probe.cpp']
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for rel in files:
                target = root / rel; target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(REPO / rel, target)
            changed = instrument(root)
            self.assertEqual(set(changed), set(files) | {'hc_capture_experiment.hpp'})
            self.assertIn('pilot expected first attention', (root / files[0]).read_text())
            before = {rel: (root / rel).read_bytes() for rel in files}
            with self.assertRaises(ValueError):
                instrument(root)
            self.assertEqual(before, {rel: (root / rel).read_bytes() for rel in files})

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
