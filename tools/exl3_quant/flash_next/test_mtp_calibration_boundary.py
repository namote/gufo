"""CPU-only admission checks: reserve only during final block, never after collapse."""
import unittest
from mtp_calibration_boundary import classify, snapshot, notify_reserved


class Admission(unittest.TestCase):
    def decision(self, idx, running=True, marker=True, bad=False, supervisor='converting_main_model'):
        return classify({'next_module_idx': idx, 'bad_rows': [0] if bad else []},
                        {'status': supervisor}, {'status': 'running' if running else 'passed'},
                        ' -- Loading unquantized module: model.language_model.layers.47' if marker else '')

    def test_no_early_lock_or_unstarted_last_block(self):
        for idx in (0, 36, 49):
            self.assertEqual(self.decision(idx), 'wait')
        self.assertEqual(self.decision(50, marker=False), 'wait')
        self.assertEqual(self.decision(50, running=False), 'wait')

    def test_queue_during_last_block(self):
        self.assertEqual(self.decision(50), 'queue_gpu_lock')

    def test_never_claim_post_mixer_or_late_start(self):
        for idx in (51, 52, 53):
            with self.assertRaises(RuntimeError):
                self.decision(idx)

    def test_nonfinite_and_stopped_are_not_ready(self):
        with self.assertRaises(RuntimeError):
            self.decision(50, bad=True)
        with self.assertRaises(RuntimeError):
            self.decision(50, supervisor='stopped')

    def test_checkpoint_rename_gap_retry_and_consistency(self):
        values = iter([None, {'next_module_idx': 49}, {'next_module_idx': 50},
                       {'next_module_idx': 50}, {'next_module_idx': 50}])
        def read():
            value = next(values)
            if value is None:
                raise FileNotFoundError('between renames')
            return value
        got = snapshot(read, lambda: ({'status': 'converting_main_model'}, {}, ''), pause=lambda _: None)
        self.assertEqual(got[0]['next_module_idx'], 50)

    def test_persistent_missing_checkpoint_and_corruption_fail(self):
        def missing():
            raise FileNotFoundError('missing')
        with self.assertRaises(RuntimeError):
            snapshot(missing, lambda: None, retries=2, pause=lambda _: None)
        def invalid():
            raise ValueError('malformed checkpoint')
        with self.assertRaises(ValueError):
            snapshot(invalid, lambda: None, pause=lambda _: None)

    def test_notification_failure_is_recorded_without_unwinding_lease(self):
        saved = []
        def failed(_):
            raise TimeoutError('transport only')
        self.assertFalse(notify_reserved('test', send=failed, persist=saved.append, pause=lambda _: None))
        self.assertEqual(len(saved[-1]['attempts']), 3)
        self.assertFalse(saved[-1]['delivered'])
        responses = iter([{'exit': 1}, {'exit': 0}])
        self.assertTrue(notify_reserved('test', send=lambda _: next(responses), persist=saved.append, pause=lambda _: None))
        self.assertTrue(saved[-1]['delivered'])


if __name__ == '__main__':
    unittest.main()
