"""CPU-only isolation/locks/budget regressions; does not run any GPU process."""
import errno
import fcntl
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from mtp_final_compile_job import (OUT, TASK, DATA, GIB, output_path, prelink_ngram,
                                  stat_id, required_disk, build_command, held_flock,
                                  required_pins, validate_inputs)


class FinalRunnerTests(unittest.TestCase):
    def test_lock_is_actual_exclusive_and_released(self):
        with tempfile.TemporaryDirectory(prefix="final-runner-cpu-",dir=OUT) as td:
            p=Path(td)/'lock'
            with p.open('a') as f:
                fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
                self.assertTrue(held_flock(Path('/proc/locks').read_text(),os.getpid(),p.stat()))
                with p.open('a') as other:
                    with self.assertRaises(BlockingIOError):fcntl.flock(other,fcntl.LOCK_EX|fcntl.LOCK_NB)
            self.assertFalse(held_flock(Path('/proc/locks').read_text(),os.getpid(),p.stat()))

    def test_ngram_same_inode_and_no_overwrite(self):
        with tempfile.TemporaryDirectory(prefix="final-runner-cpu-",dir=OUT) as td:
            root=Path(td);src=root/'source';src.write_bytes(b'ngram')
            work=root/'private';work.mkdir();dst=work/'ngram'
            initial=stat_id(src)
            self.assertEqual(prelink_ngram(src,dst),initial)
            self.assertEqual(stat_id(dst),initial)
            with self.assertRaises(AssertionError):prelink_ngram(src,dst)
            self.assertEqual(src.read_bytes(),b'ngram')
            # Container's second link is within this same writable directory tree.
            pack=work/'pack';pack.mkdir();os.link(dst,pack/'ngram')
            self.assertEqual(stat_id(pack/'ngram'),initial)

    def test_no_cross_device_copy_fallback(self):
        with tempfile.TemporaryDirectory(prefix="final-runner-cpu-",dir=OUT) as td:
            src=Path(td)/'source';src.write_bytes(b'x');dst=Path(td)/'dest'
            with patch('mtp_final_compile_job.os.link',side_effect=OSError(errno.EXDEV,'cross mount')):
                with self.assertRaises(OSError):prelink_ngram(src,dst)
            self.assertFalse(dst.exists());self.assertEqual(src.read_bytes(),b'x')

    def test_container_only_private_output_writable(self):
        prior=json.loads((OUT/'model-job-c03/module-060/STATE.json').read_text())
        old=prior['command'].copy()
        c=dict(assembly_review='/output/final-review.json',main_work='/output/work',
               mtp_work='/output/mtp-calibration-work/full/work',recipe='/output/MTP-MIXED-BITS-PROPOSAL.json')
        result=build_command(prior,c,'unique','label',DATA/'fresh')
        mounts=[result[i+1] for i,v in enumerate(result) if v=='--mount']
        self.assertEqual([m for m in mounts if not m.endswith(',readonly')],
                         [f'type=bind,src={DATA / "fresh"},dst=/mtp-run'])
        self.assertIn(f'type=bind,src={OUT},dst=/output,readonly',mounts)
        self.assertEqual(result.count('EXL3_SKIP_SIDE_MODELS=1'),1)
        self.assertNotIn('GUFO_MTP_LEASE_VERIFIED=1',result)
        self.assertIn('/task/mtp_final_compile_candidate.py',result)
        self.assertIn('/mtp-run/pack',result)
        self.assertEqual(prior['command'],old)

    def test_reject_other_writable_mount(self):
        prior=json.loads((OUT/'model-job-c03/module-060/STATE.json').read_text())
        i=prior['command'].index('--workdir')
        prior['command'][i:i]=['--mount','type=bind,src=/tmp,dst=/unsafe']
        c=dict(assembly_review='/output/r',main_work='/output/work',mtp_work='/output/m',recipe='/output/recipe')
        with self.assertRaises(AssertionError):build_command(prior,c,'n','l',DATA/'fresh')

    def test_budget_covers_complete_output_and_reserve(self):
        with tempfile.TemporaryDirectory(prefix="final-runner-cpu-",dir=OUT) as td:
            a=Path(td)/'main';b=Path(td)/'mtp';a.write_bytes(b'abc');b.write_bytes(b'defg')
            result=required_disk([a],[b])
            self.assertEqual(result['required_bytes'],108*GIB+897862112+7)
            self.assertEqual(result['packed_input_bytes'],7)

    def test_reject_unreviewed_paths_and_pin_all_imports(self):
        self.assertEqual(output_path('/output/work'),OUT/'work')
        for path in ['/output/../source','/output','relative','/source/file']:
            with self.assertRaises(AssertionError):output_path(path)
        c=dict(assembly_review='/output/review',recipe='/output/recipe',cpu=str(OUT/'CPU.json'))
        pins=required_pins(c)
        for name in ('mtp_final_compile_job.py','mtp_calibration_job.py','mtp_stage_disk_budget.py',
                     'owned_cleanup.py','conversion_admission.py','run_stage.py','mtp_calibration_candidate.py'):
            self.assertIn(str(TASK/name),pins)

    def test_real_decoder_receipt_is_full_not_probe(self):
        p=OUT/'mtp-calibration-work/full/work/DECODER-COMPLETE.json'
        r=json.loads(p.read_text())
        self.assertEqual(r['status'],'mtp_calibrated_pending_full_validation')
        self.assertEqual((r['rows'],r['tokens'],len(r['H'])),(2309,1997187,516))
        self.assertEqual({x['count'] for x in r['H'].values()},{1997187})


if __name__=='__main__':unittest.main()
