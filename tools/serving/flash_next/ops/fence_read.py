"""Explicit existing DRM debugfs file reader. No mount, ioctl, reset or device open."""
from pathlib import Path
import argparse, json, time, os, re

LIMIT = 262144


def validate_path(value, bdf):
    p = Path(value)
    if not re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]', bdf):
        raise ValueError('explicit PCI BDF required')
    if (not p.is_absolute() or '..' in p.parts or p.name != 'amdgpu_fence_info'
            or p.parent.parent != Path('/sys/kernel/debug/dri')):
        raise ValueError('only existing DRM debugfs fence leaf allowed')
    if not (p.parent.name.isdecimal() or p.parent.name == bdf):
        raise ValueError('explicit DRM index/BDF required')
    return p


def read_existing(value, bdf, clock=time.monotonic):
    p = validate_path(value, bdf)
    begin = clock()
    resolved = p.resolve(strict=True)
    if resolved.name != 'amdgpu_fence_info' or resolved.parent.parent != Path('/sys/kernel/debug/dri'):
        raise ValueError('fence resolution escaped DRM debugfs')
    name = (resolved.parent / 'name').read_text()
    if 'amdgpu' not in name or not re.search(r'(?<![0-9a-f:.])' + re.escape(bdf) + r'(?![0-9a-f:.])', name):
        raise ValueError('DRM fence device BDF mismatch')
    st = resolved.stat()
    with resolved.open('r') as f:
        raw = f.read(LIMIT + 1)
    if len(raw) > LIMIT:
        raise ValueError('fence response exceeds bound')
    return dict(begin=begin, end=clock(), epoch=time.time(), reader_PID=os.getpid(), requested=str(p),
                resolved=str(resolved), BDF=bdf, debugfs_name=name, stat_dev=st.st_dev, stat_ino=st.st_ino, raw=raw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--path', required=True)
    ap.add_argument('--bdf', required=True)
    a = ap.parse_args()
    print(json.dumps(read_existing(a.path, a.bdf)))


if __name__ == '__main__':
    main()
