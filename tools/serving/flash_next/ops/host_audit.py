"""Root read-only GPU FD/KFD/telemetry audit (audit_host, proc_identity); stdlib only."""
import json
import os
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def proc_identity(pid):
    p = Path('/proc') / str(pid)
    stat = (p / 'stat').read_text().rsplit(')', 1)[1].split()
    return {'PID': pid, 'comm': (p / 'comm').read_text().strip(), 'starttime_ticks': int(stat[19]),
            'PGID': int(stat[2]), 'cgroup': (p / 'cgroup').read_text().strip()}


def audit_host():
    # This branch imports only stdlib and reads host state; sudo is solely for FD visibility.
    gpu_fds, errors = [], []
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():
            continue
        try:
            ident = proc_identity(int(p.name))
            for fd in (p / 'fd').iterdir():
                try:
                    dest = os.readlink(fd)
                    if dest == '/dev/kfd' or dest.startswith('/dev/dri/'):
                        fdinfo = (p / 'fdinfo' / fd.name).read_text()
                        client = next((x.split(':', 1)[1].strip() for x in fdinfo.splitlines() if x.startswith('drm-client-id:')), None)
                        gpu_fds.append({**ident, 'fd': int(fd.name), 'device': dest, 'drm_client_id': client, 'fdinfo': fdinfo})
                except FileNotFoundError:
                    continue
        except (FileNotFoundError, ProcessLookupError):
            continue
        except (PermissionError, OSError) as exc:
            errors.append({'PID': int(p.name), 'error': str(exc)})
    kfd_dir = Path('/sys/devices/virtual/kfd/kfd/proc')
    if not kfd_dir.is_dir():
        errors.append({'error': 'KFD process telemetry directory unavailable'})
    kfd = [proc_identity(int(p.name)) for p in kfd_dir.iterdir() if p.name.isdigit() and Path('/proc', p.name).exists()]
    desktop = {'kwin_wayland', 'kwin_x11', 'plasmashell', 'Xwayland', 'Xorg', 'gnome-shell', 'kded6', 'kglobalacceld', 'firefox', 'chrome', 'chromium', 'electron'}
    desktop_receipt = ROOT / 'DESKTOP-BASELINE.json'
    baseline = json.loads(desktop_receipt.read_text())['known_desktop_identities'] if desktop_receipt.exists() else []
    def known_desktop(row):
        return row['comm'] in desktop or any(all(row[k] == saved[k] for k in ['PID', 'comm', 'starttime_ticks', 'cgroup']) for saved in baseline)
    known_clients = {r['drm_client_id'] for r in gpu_fds if known_desktop(r) and r['drm_client_id'] is not None}
    for row in gpu_fds:
        if known_desktop(row):
            row['classification'] = 'desktop_background'
        elif row['device'] == '/dev/kfd' and row['PID'] not in {r['PID'] for r in kfd}:
            row['classification'] = 'FD_present_no_registered_KFD_context'
        elif row['drm_client_id'] is not None and row['drm_client_id'] in known_clients:
            row['classification'] = 'shared_desktop_DRM_client'
        else:
            row['classification'] = 'unidentified_GPU_client'
    devices = list(Path('/sys/class/drm').glob('card*/device/gpu_busy_percent'))
    if len(devices) != 1:
        errors.append({'error': 'ambiguous GPU telemetry'})
    telemetry = {}
    if len(devices) == 1:
        for name in ['gpu_busy_percent', 'mem_info_vram_used', 'mem_info_gtt_used']:
            p = devices[0].parent / name
            try:
                telemetry[name] = int(p.read_text())
            except OSError as exc:
                errors.append({'error': 'GPU telemetry read: ' + str(exc)})
    mem = dict(x.split(':', 1) for x in Path('/proc/meminfo').read_text().splitlines())
    result = {'at_utc': datetime.now(timezone.utc).isoformat(), 'GPU_FDs': gpu_fds, 'KFD_contexts': kfd,
              'errors': errors, 'telemetry': telemetry, 'available_bytes': int(mem['MemAvailable'].split()[0]) * 1024,
              'global_busy_zero_required': False}
    print(json.dumps(result), flush=True)
