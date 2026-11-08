"""Opt-in host-only conversion progress; no device reads, syncs or imports."""
import json
import sys
import time


def _scope_stats():
    module = sys.modules.get('capture_fast')
    values = getattr(module, 'STATS', {}) if module is not None else {}
    return {key: values[key] for key in ('scopes', 'skipped_down_calls', 'skipped_down_rows')
            if key in values and type(values[key]) is int}


def install(cm):
    """Wrap cm's bars only; return an explicit restore callback for probes."""
    original = cm.ProgressBar
    assert not getattr(original, '_gufo_host_progress', False), 'Already installed'

    class HostProgress(original):
        _gufo_host_progress = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self._gufo_started = self._gufo_phase_started = time.monotonic()
            self._gufo_last_log = self._gufo_started
            self._gufo_current = 0

        def _gufo_emit(self, event, now, result=None, exception=None):
            record = dict(event=event, text=self.text, current=self._gufo_current,
                          count=self.count, elapsed_seconds=now-self._gufo_phase_started,
                          bar_elapsed_seconds=now-self._gufo_started,
                          timestamp=time.time(), scope_stats_cumulative=_scope_stats())
            if result is not None:
                record['result'] = result
            if exception is not None:
                record['exception_type'] = exception.__name__
            print(' -- GUFO_PROGRESS ' + json.dumps(record, separators=(',', ':')), flush=True)
            self._gufo_last_log = now

        def update(self, value):
            result = super().update(value)
            self._gufo_current = value
            now = time.monotonic()
            if now-self._gufo_last_log >= 60:
                self._gufo_emit('progress', now)
            return result

        def new_task(self, text, count):
            # Match the pinned base's no-op for bars created without text.
            has_progress = bool(getattr(self, 'progress', None))
            result = super().new_task(text, count)
            if has_progress:
                self._gufo_current = 0
                self._gufo_phase_started = time.monotonic()
            # Deliberately do not reset the rate limit or log every new task.
            return result

        def __exit__(self, exc_type, exc_value, traceback):
            try:
                result = super().__exit__(exc_type, exc_value, traceback)
            except BaseException as error:
                self._gufo_emit('phase_end', time.monotonic(), 'exit_error', type(error))
                raise
            self._gufo_emit('phase_end', time.monotonic(),
                            'failed' if exc_type is not None else 'completed', exc_type)
            return result

    cm.ProgressBar = HostProgress

    def restore():
        assert cm.ProgressBar is HostProgress, 'Progress wrapper changed since install'
        cm.ProgressBar = original

    return restore
