"""Original kernel lines are saved before raising; health != resource-return result."""
import re, json

PATTERN = re.compile(r'timeout|GPU reset|page fault|device wedged|amdgpu.*error|amdxdna.*(?:error|fault|fail)', re.I)


class KernelDelta:
    def __init__(self, out, run):
        self.out = out
        self.run = run
        self.cursor = None
        self.sequence = 0
        self.failures = []

    def start(self):
        r = self.run(['journalctl', '-k', '-n', '1', '--no-pager', '-o', 'json', '--show-cursor'])
        self.out.joinpath('KERNEL-BASELINE.txt').write_text(r.stdout)
        m = re.search(r'^-- cursor: (.+)$', r.stdout, re.M)
        if not m:
            raise ValueError('kernel journal cursor unavailable')
        self.cursor = m.group(1)

    def collect(self, raise_error=True):
        if self.cursor is None:
            raise ValueError('kernel cursor not initialized')
        r = self.run(['journalctl', '-k', '--after-cursor', self.cursor, '--no-pager', '-o', 'json', '--show-cursor'])
        self.sequence += 1
        path = self.out / f'KERNEL-DELTA-{self.sequence:04}.txt'
        path.write_text(r.stdout)
        messages = []
        for line in r.stdout.splitlines():
            if not line or line.startswith('-- cursor:'):
                continue
            row = json.loads(line)
            messages.append(row)
        errors = [m for m in messages if PATTERN.search(str(m.get('MESSAGE', '')))]
        if errors:
            self.failures = errors
        if errors and raise_error:
            raise RuntimeError('new kernel error; original preserved: ' + str(path))
        return dict(path=str(path), new_GPU_error=bool(errors), errors=errors, original_rows=len(messages))
