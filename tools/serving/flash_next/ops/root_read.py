"""Read-only bracketed GPU inventory. Never opens devices or changes processes."""
import os, time, json, runpy, hashlib, argparse, contextlib, io
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def ident(pid):
    p = Path('/proc') / str(pid)
    raw = (p / 'stat').read_text()
    tail = raw[raw.rfind(')') + 2:].split()
    st = (p / 'exe').stat()
    status = (p / 'status').read_text()
    return dict(PID=pid, PPID=int(tail[1]), starttime_ticks=int(tail[19]), PGID=int(tail[2]),
                comm=(p / 'comm').read_text().strip(), cgroup=(p / 'cgroup').read_text().strip(),
                UIDs=list(map(int, next(l for l in status.splitlines() if l.startswith('Uid:')).split()[1:])),
                exe=os.readlink(p / 'exe'),
                exe_stat=[st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_uid, st.st_mode])


def registry():
    start = time.monotonic()
    rows = []
    errors = []
    for p in Path('/sys/devices/virtual/kfd/kfd/proc').iterdir():
        if p.name.isdigit():
            try:
                rows.append(ident(int(p.name)))
            except (FileNotFoundError, ProcessLookupError):
                errors.append(dict(PID=int(p.name), kind='registry_PID_disappeared', at=time.monotonic()))
    return dict(start=start, end=time.monotonic(), rows=rows, events=errors)


def fdset(pid):
    result = {}
    for p in (Path('/proc') / str(pid) / 'fd').iterdir():
        try:
            dest = os.readlink(p)
            if dest == '/dev/kfd' or dest.startswith('/dev/dri/'):
                result[int(p.name)] = dest
        except FileNotFoundError:
            pass
    return result


def scan(pid):
    start = time.monotonic()
    rows = []
    error = None
    before = after = None
    first = last = None
    try:
        before = ident(pid)
        first = fdset(pid)
        for fd, dest in first.items():
            try:
                text = (Path('/proc') / str(pid) / 'fdinfo' / str(fd)).read_text()
                client = next((l.split(':', 1)[1].strip() for l in text.splitlines()
                               if l.startswith('drm-client-id:')), None)
                rows.append(dict(**{k: before[k] for k in ['PID', 'comm', 'starttime_ticks', 'PGID', 'cgroup']},
                                 fd=fd, device=dest, drm_client_id=client, fdinfo=text, verified_proc=before,
                                 classification='unidentified_GPU_client'))
            except FileNotFoundError:
                error = 'device_FD_disappeared_during_scan'
        last = fdset(pid)
        after = ident(pid)
    except (FileNotFoundError, ProcessLookupError):
        error = 'PID_disappeared_during_scan'
    except (PermissionError, OSError) as e:
        error = repr(e)
    return dict(PID=pid, start=start, end=time.monotonic(), before=before, after=after, FD_before=first,
                FD_after=last, rows=rows, error=error,
                stable=before is not None and before == after and first == last and error is None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--owned-root')
    args = ap.parse_args()
    root_before = None
    chains = []
    if args.owned_root:
        root_pid, root_start = map(int, args.owned_root.split(':'))
        try:
            root_before = ident(root_pid)
        except (FileNotFoundError, ProcessLookupError):
            root_before = None
        if root_before is not None:
            assert root_before['starttime_ticks'] == root_start
    begin = time.monotonic()
    boot_before = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    pre = registry()
    before = {r['PID']: scan(r['PID']) for r in pre['rows']}
    old = ROOT / 'legacy_root_read.py'
    ns = runpy.run_path(str(old))
    buffer = io.StringIO()
    legacy_begin = time.monotonic()
    with contextlib.redirect_stdout(buffer):
        ns['main']()
    legacy_end = time.monotonic()
    host = json.loads(buffer.getvalue())
    mid = registry()
    ids = {r['PID'] for r in pre['rows'] + mid['rows']}
    after = {pid: scan(pid) for pid in ids}
    post = registry()

    def sig(s):
        return sorted((r['PID'], r['starttime_ticks']) for r in s['rows'])
    stable = (sig(pre) == sig(mid) == sig(post) and not any(s['events'] for s in [pre, mid, post])
              and all(s['stable'] for s in list(before.values()) + list(after.values())))
    if stable:
        for pid in ids:
            a, b = before[pid], after[pid]
            stable = stable and a['before'] == b['after'] and a['FD_before'] == b['FD_after']
    # Publish the latest exact KFD identities and targeted rows; unstable brackets never pass.
    host['KFD_contexts'] = [{k: r[k] for k in ['PID', 'comm', 'starttime_ticks', 'PGID', 'cgroup']}
                            for r in post['rows']]
    host['GPU_FDs'] = ([r for r in host['GPU_FDs'] if r['PID'] not in ids]
                       + [r for s in after.values() for r in s['rows']])
    host['coherence'] = dict(version=1, boot_before=boot_before,
                             boot_after=Path('/proc/sys/kernel/random/boot_id').read_text().strip(), begin=begin,
                             end=time.monotonic(), clock_ticks=os.sysconf('SC_CLK_TCK'),
                             legacy_scan_begin=legacy_begin, legacy_scan_end=legacy_end, KFD_pre=pre, KFD_mid=mid,
                             KFD_post=post, FD_pre=list(before.values()), FD_post=list(after.values()),
                             stable=bool(stable))
    if root_before is not None:
        for row in pre['rows'] + mid['rows'] + post['rows']:
            chain = [row]
            seen = {row['PID']}
            try:
                while chain[-1]['PID'] != root_before['PID'] and len(chain) < 32:
                    parent = ident(chain[-1]['PPID'])
                    if parent['PID'] in seen:
                        break
                    seen.add(parent['PID'])
                    chain.append(parent)
                if chain[-1] == root_before:
                    chains.append(chain)
            except (FileNotFoundError, ProcessLookupError):
                pass
        try:
            root_after = ident(root_before['PID'])
        except (FileNotFoundError, ProcessLookupError):
            root_after = None
        host['owner_proof'] = dict(root_before=root_before, root_after=root_after, chains=chains)
    # Include ancestry verification in this collection's monotonic bracket.
    host['coherence']['end'] = time.monotonic()
    host['coherent_reader_SHA'] = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
    print(json.dumps(host))


if __name__ == '__main__':
    main()
