"""Conversion-only attribution of reviewed desktop graphics; not a timing gate."""
import json
from pathlib import Path
import re
import time

FIELDS = ('PID', 'comm', 'starttime_ticks', 'cgroup', 'PGID', 'UIDs', 'exe', 'exe_stat')

def clients(h):
    result = {}
    for row in h['GPU_FDs']:
        proc = row['verified_proc']
        assert all(row[k] == proc[k] for k in ('PID','comm','starttime_ticks','cgroup','PGID'))
        ident = {k: proc[k] for k in FIELDS}
        key = (row['PID'], row['drm_client_id'], row['device'])
        assert key[1] is not None
        engines = {}; metadata = {}
        for line in row['fdinfo'].splitlines():
            if line.startswith('drm-engine-'):
                match = re.fullmatch(r'drm-engine-([a-zA-Z0-9_-]+):\s*([0-9]+)\s+ns\s*', line)
                assert match, 'Malformed engine counter'
                name, value = match.groups()
                assert name not in engines, 'Duplicate engine counter'
                engines[name] = int(value)
            elif line.startswith(('drm-driver:', 'drm-pdev:')):
                name, value = line.split(':', 1)
                assert name not in metadata, 'Duplicate DRM metadata'
                metadata[name] = value.strip()
        assert metadata['drm-driver'] == 'amdgpu'
        assert re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', metadata['drm-pdev'])
        value = dict(proc=ident, client=key[1], device=key[2], pdev=metadata['drm-pdev'], engines=engines)
        if key in result:
            assert result[key] == value, 'Inconsistent duplicated DRM client'
        result[key] = value
    return result

def desktop_decision(a, b, manifest):
    """Reject unexplained utilization, non-graphics work, or changed identity."""
    try:
        assert manifest['purpose'] == 'conversion_desktop_graphics_only'
        for h in (a,b):
            assert h['boot_id'] == manifest['boot_id']
            assert not h['KFD_contexts'] and not h['errors']
            assert h['coherence']['stable']
            assert h['coherence']['boot_before'] == h['coherence']['boot_after'] == h['boot_id']
            assert 0 <= h['telemetry']['gpu_busy_percent'] <= 20
            assert 0 <= h['coherence']['end']-h['coherence']['begin'] <= 0.5
        dt = b['coherence']['end'] - a['coherence']['end']
        assert 2 <= dt <= 10
        assert a['coherence']['end'] <= b['coherence']['begin']
        aa,bb = clients(a),clients(b)
        known = {(r['proc']['PID'],r['client'],r['device']):r for r in manifest['clients']}
        assert aa.keys() == bb.keys() == known.keys(), 'DRM client set changed'
        seen = {}; gfx = 0
        for k,left in aa.items():
            right,ref = bb[k],known[k]
            assert left['proc'] == right['proc'] == ref['proc'], 'Process identity changed'
            assert left['pdev'] == right['pdev'] == ref['pdev'] == manifest['pdev']
            assert left['engines'].keys() == right['engines'].keys()
            deltas = {e:right['engines'][e]-v for e,v in left['engines'].items()}
            assert all(v >= 0 for v in deltas.values()), 'Counter reset'
            assert all(v == 0 for e,v in deltas.items() if e != 'gfx'), 'Non-graphics work'
            if deltas.get('gfx',0):
                assert ref['allow_gfx'], 'Unapproved graphics owner'
            cid = (left['pdev'],left['client'])
            if cid in seen:
                assert seen[cid] == deltas, 'Shared DRM counter inconsistency'
            else:
                seen[cid] = deltas; gfx += deltas.get('gfx',0)
        pct = gfx / (dt*1e9) * 100
        assert 0 < pct <= 20, 'No attributable desktop activity or excessive activity'
        assert all(abs(h['telemetry']['gpu_busy_percent']-pct) <= 5 for h in (a,b)), 'Unexplained GPU activity'
        return dict(admitted=True, mode='reviewed_desktop_graphics', gfx_percent=pct, seconds=dt)
    except (AssertionError,KeyError,TypeError,ValueError) as e:
        return dict(admitted=False, reason=str(e) or type(e).__name__)

def wait_conversion_admission(host, quiet, manifest_path, *, pause=time.sleep, attempts=30):
    manifest = json.loads(Path(manifest_path).read_text())
    previous = None; samples=[]
    for _ in range(attempts):
        h=host()
        assert not h['KFD_contexts'], 'GPU compute owner active'
        if quiet(h):
            return h,dict(admitted=True,mode='original_quiet',samples=samples)
        if previous is not None:
            decision=desktop_decision(previous,h,manifest)
            samples.append(dict(busy=h['telemetry']['gpu_busy_percent'],decision=decision.copy()))
            if decision['admitted']:
                decision['samples']=samples
                decision['snapshots']=[previous,h]
                return h,decision
        previous=h;pause(2)
    raise RuntimeError('No quiet or reviewed desktop-only conversion admission: '+json.dumps(samples[-3:]))
