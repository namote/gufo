"""GFX ring pending interval lower bound from external read-only snapshots."""
import re, json, time
from .fence_read import validate_path
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RING = re.compile(r'^(?:---\s*)?ring\s+\d+\s*\(([^)]+)\)', re.I)
VALUE = re.compile(
    r'^(Last signaled trailing fence|Last signaled fence|Last emitted|Last reset)\s*[:=]?\s*(0x[0-9a-f]+|[0-9]+)',
    re.I)


def parse(raw):
    rings = {}
    current = None
    trailing = False
    for line in raw.splitlines():
        s = line.strip()
        m = RING.match(s)
        if m:
            current = m.group(1)
            if current in rings:
                raise ValueError('duplicate ring')
            rings[current] = {}
            trailing = False
            continue
        v = VALUE.match(s)
        if v and current is not None:
            label = v.group(1).lower()
            if label == 'last signaled trailing fence':
                key = 'trailing_signaled'
                trailing = True
            elif label == 'last emitted' and trailing:
                key = 'trailing_emitted'
                trailing = False
            else:
                key = {'last signaled fence': 'signaled', 'last emitted': 'emitted', 'last reset': 'reset'}[label]
            if key in rings[current]:
                raise ValueError('duplicate fence field')
            rings[current][key] = int(v.group(2), 0)
    if 'gfx_0.0.0' not in rings:
        raise ValueError('expected GFX ring absent; no substitute ring')
    r = rings['gfx_0.0.0']
    if not all(k in r for k in ['signaled', 'emitted']) or r['emitted'] < r['signaled']:
        raise ValueError('incomplete/inconsistent GFX fences')
    return r


class Pending:
    def __init__(self):
        self.last = None
        self.since = None

    def observe(self, sample):
        begin, end = sample['begin'], sample['end']
        if end < begin or end - begin > 1:
            raise ValueError('fence read bracket invalid/over1s')
        r = parse(sample['raw'])
        gap = None
        if self.last:
            if begin < self.last['end']:
                raise ValueError('fence samples overlap/backwards')
            gap = begin - self.last['end']
            old = self.last['ring']
            if (any(r[k] < old[k] for k in ['emitted', 'signaled'])
                    or ('reset' in r and 'reset' in old and r['reset'] != old['reset'])):
                raise RuntimeError('fence regression/reset; preserve and abort')
        pending = r['emitted'] > r['signaled']
        if not pending:
            self.since = None
        elif self.last is None or r['signaled'] != self.last['ring']['signaled'] or self.since is None:
            self.since = end
        lower = max(0, begin - self.since) if pending else 0
        self.last = dict(end=end, ring=r)
        return dict(ring=r, pending=pending, pending_sampled_lower_bound_s=lower, gap_since_previous_end_s=gap,
                    exact_job_latency=False, KFD_progress_observed=False, specific_GUI_job_attribution=False)


class FenceMonitor:
    def __init__(self, out, run, path, bdf):
        validate_path(path, bdf)
        self.out = out
        self.run = run
        self.path = path
        self.bdf = bdf
        self.seq = 0
        self.pending = Pending()

    def collect(self):
        self.seq += 1
        prefix = self.out / f'FENCE-{self.seq:04}'
        start = time.monotonic()
        cmd = ['sudo', '-n', '/usr/bin/timeout', '--signal=TERM', '--kill-after=0.2', '1', '/usr/bin/python3', '-I',
               str(ROOT / 'fence_read.py'), '--path', self.path, '--bdf', self.bdf]
        try:
            r = self.run(cmd)
        except BaseException as e:
            for suffix, value in [('.txt', getattr(e, 'stdout', None)), ('.stderr.txt', getattr(e, 'stderr', None))]:
                if value is not None:
                    prefix.with_suffix(suffix).write_text(
                        value.decode(errors='replace') if isinstance(value, bytes) else str(value))
            prefix.with_suffix('.json').write_text(json.dumps(dict(
                status='READ_FAILED_NO_RETRY', error=repr(e), begin=start, end=time.monotonic())) + '\n')
            raise
        # Preserve original stdout before parsing or enforcing anything.
        prefix.with_suffix('.txt').write_text(r.stdout)
        try:
            sample = json.loads(r.stdout)
            if sample['BDF'] != self.bdf or sample['requested'] != self.path:
                raise ValueError('fence reader identity changed')
            detail = self.pending.observe(sample)
            prefix.with_suffix('.json').write_text(json.dumps(dict(status='OBSERVED', sample=sample, detail=detail)) + '\n')
            return detail
        except BaseException as e:
            prefix.with_suffix('.json').write_text(json.dumps(dict(
                status='PARSE_OR_GATE_FAILED', error=repr(e), begin=start, end=time.monotonic())) + '\n')
            raise
