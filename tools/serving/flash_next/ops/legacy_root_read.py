"""Root read-only proc/DRM inventory. No signal, device open, or system mutation."""
import contextlib, io, json, os, runpy
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def main():
    original = ROOT / 'host_audit.py'
    namespace = runpy.run_path(str(original), run_name='readonly_audit_source')
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        namespace['audit_host']()
    host = json.loads(buffer.getvalue())
    host['boot_id'] = Path('/proc/sys/kernel/random/boot_id').read_text().strip()
    identities = {}
    for pid in sorted({r['PID'] for r in host['GPU_FDs']}):
        p = Path('/proc') / str(pid)
        try:
            status = (p / 'status').read_text()
            uids = next(l for l in status.splitlines() if l.startswith('Uid:')).split()[1:]
            exe = os.readlink(p / 'exe')
            st = (p / 'exe').stat()
            identity = namespace['proc_identity'](pid)
            identities[pid] = dict(**identity, UIDs=list(map(int, uids)), exe=exe,
                                   exe_stat=[st.st_dev, st.st_ino, st.st_size, st.st_mtime_ns, st.st_uid, st.st_mode])
        except (OSError, StopIteration) as exc:
            host['errors'].append(dict(PID=pid, identity_error=repr(exc)))
    for row in host['GPU_FDs']:
        row['verified_proc'] = identities.get(row['PID'])
    print(json.dumps(host))


if __name__ == '__main__':
    main()
