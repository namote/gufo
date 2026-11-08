"""Exact approved display-infrastructure DRI only. No KFD/AI ownership exemption."""
from .admission import classify as original, FIELDS

OUTER = ('PID', 'comm', 'starttime_ticks', 'cgroup', 'PGID')


def classify(host, manifest):
    host = original(host, manifest)
    applied = set(host['boot_desktop_classification']['PIDs'])
    for row in host['GPU_FDs']:
        if row['classification'] != 'unidentified_GPU_client' or not row.get('device', '').startswith('/dev/dri/'):
            continue
        ident = row.get('verified_proc')
        if ident is None or any(row.get(k) != ident.get(k) for k in OUTER):
            continue
        if any(all(ident.get(k) == known[k] for k in FIELDS)
               for known in manifest.get('verified_display_infrastructure', [])):
            row['classification'] = 'verified_boot_desktop_client'
            applied.add(row['PID'])
    host['boot_desktop_classification'].update(
        PIDs=sorted(applied), exact_display_infrastructure_DRI_only=True, no_KFD_exemption=True)
    return host
