"""Small CPU-only fake-progress contract; no torch or GPU runtime imports."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import patch
import capture_progress


class BaseProgress:
    def __init__(self, text, count, transient=True):
        self.text, self.count, self.transient = text, count, transient
        self.calls = [('init', text, count, transient)]
        if text:
            self.progress = object()
        self.exit_return = None
        self.exit_error = None

    def __enter__(self):
        self.calls.append(('enter',))
        return self

    def update(self, value):
        self.calls.append(('update', value))
        return 'base_update_result'

    def new_task(self, text, count):
        self.calls.append(('new_task', text, count))
        if not getattr(self, 'progress', None):
            return 'base_noop'
        self.text, self.count = text, count
        return 'base_new_task_result'

    def __exit__(self, *args):
        self.calls.append(('exit', *args))
        if self.exit_error:
            raise self.exit_error
        return self.exit_return


now = [0.0]
cm = SimpleNamespace(ProgressBar=BaseProgress)
output = io.StringIO()
fake_stats = SimpleNamespace(STATS=dict(scopes=3, skipped_down_calls=512, skipped_down_rows=262144))
checks = []
with patch.object(capture_progress.time, 'monotonic', side_effect=lambda: now[0]), \
        patch.object(capture_progress.time, 'time', side_effect=lambda: 1800000000+now[0]), \
        patch.dict(sys.modules, {'capture_fast': fake_stats}), redirect_stdout(output):
    restore = capture_progress.install(cm)
    try:
        try:
            capture_progress.install(cm)
        except AssertionError:
            checks.append('duplicate_install_rejected')
        else:
            raise AssertionError('Duplicate install accepted')
        bar = cm.ProgressBar('capture', 2309, transient=False)
        assert bar.__enter__() is bar and bar.transient is False
        assert bar.update(0) == 'base_update_result'
        now[0] = 59.999
        bar.update(39)
        assert not output.getvalue()
        now[0] = 60
        bar.update(40)
        now[0] = 61
        for i in range(100):
            assert bar.new_task(f'tensor-{i}', 100) == 'base_new_task_result'
            bar.update(i)
        assert len(output.getvalue().splitlines()) == 1
        now[0] = 120
        bar.update(99)
        now[0] = 121
        assert bar.__exit__(None, None, None) is None
        assert bar.calls[0] == ('init', 'capture', 2309, False)
        assert bar.calls[1] == ('enter',)
        assert bar.calls[-1] == ('exit', None, None, None)
        assert sum(c[0] == 'new_task' for c in bar.calls) == 100
        checks += ['base_init_enter_update_new_task_exit_preserved',
                   '60_second_rate_limit_across_100_new_tasks']

        records = [json.loads(line.split(' -- GUFO_PROGRESS ', 1)[1])
                   for line in output.getvalue().splitlines()]
        assert len(records) == 3
        assert records[0]['text'] == 'capture' and records[0]['current'] == 40
        assert records[0]['count'] == 2309 and records[0]['elapsed_seconds'] == 60
        assert records[1]['text'] == 'tensor-99' and records[1]['elapsed_seconds'] == 59
        assert records[1]['bar_elapsed_seconds'] == 120
        assert records[2]['result'] == 'completed' and records[2]['event'] == 'phase_end'
        assert records[2]['timestamp'] == 1800000121
        assert records[2]['scope_stats_cumulative'] == fake_stats.STATS
        checks.append('phase_current_count_elapsed_timestamp_scope_stats')

        blank = cm.ProgressBar(None, 5)
        blank.update(2)
        assert blank.new_task('ignored', 10) == 'base_noop'
        assert blank.text is None and blank.count == 5 and blank._gufo_current == 2
        checks.append('textless_base_new_task_noop_preserved')
        failure = cm.ProgressBar('failure', 3)
        error = ValueError('fixture')
        failure.exit_return = True
        assert failure.__exit__(ValueError, error, None) is True
        record = json.loads(output.getvalue().splitlines()[-1].split(' -- GUFO_PROGRESS ', 1)[1])
        assert record['result'] == 'failed' and record['exception_type'] == 'ValueError'
        assert failure.calls[-1] == ('exit', ValueError, error, None)
        checks.append('exception_info_and_base_suppression_return_preserved')
        failure.exit_error = RuntimeError('base exit failed')
        try:
            failure.__exit__(None, None, None)
        except RuntimeError as caught:
            assert caught is failure.exit_error
        else:
            raise AssertionError('Base exception lost')
        record = json.loads(output.getvalue().splitlines()[-1].split(' -- GUFO_PROGRESS ', 1)[1])
        assert record['result'] == 'exit_error'
        checks.append('base_exit_failure_propagated')
    finally:
        restore()
assert cm.ProgressBar is BaseProgress
checks.append('restore_original_progress_class')
with patch.dict(sys.modules, {'capture_fast': None}):
    assert capture_progress._scope_stats() == {}
checks.append('no_capture_module_dependency')
result = dict(status='pass', checks=checks, gpu_used=False, torch_imported='torch' in sys.modules,
              candidate_sha256=hashlib.sha256(Path(capture_progress.__file__).read_bytes()).hexdigest(),
              test_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
destination = Path(sys.argv[1])
destination.parent.mkdir(parents=True, exist_ok=True)
destination.write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
