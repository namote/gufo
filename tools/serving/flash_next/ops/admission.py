"""Boot-scoped exact identity classification; original AI/KFD and quiet guards remain."""
FIELDS = ('PID', 'comm', 'starttime_ticks', 'cgroup', 'UIDs', 'exe', 'exe_stat')


def classify(host, manifest):
    assert host['boot_id'] == manifest['boot_id'], 'boot baseline stale'
    known = manifest['verified_desktop_clients']
    applied = set()
    for row in host['GPU_FDs']:
        if row['classification'] != 'unidentified_GPU_client':
            continue
        identity = row.get('verified_proc')
        if identity and any(all(identity.get(k) == saved[k] for k in FIELDS) for saved in known):
            row['classification'] = 'verified_boot_desktop_client'
            applied.add(row['PID'])
    host['boot_desktop_classification'] = dict(
        PIDs=sorted(applied), not_AI_KFD_exemption=True,
        baseline='exact boot/PID/start/UID/exe stat/cgroup', engine_counters_available=True)
    return host
