"""CPU-only Kyojin preparation with shared owner and bounded disk/memory checks."""
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from test.experiments.gufo_upstream_port_20261023 import window


def main():
    out = window.OUT_ROOT / 'kyojin-local-prepare'
    with window.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        h = window.host()
        assert window.quiet(h) and h['available_bytes'] >= 80 * 1024**3
        assert not window.npu()
        assert shutil.disk_usage(window.ROOT).free >= window.DISK_FLOOR
        name = 'gufo-kyojin-cpu-build'
        argv = ['docker', 'run', '--rm', '--name', name, '--network', 'none',
                '--user', '1000:1000', '--memory', '20g', '--memory-swap', '20g',
                '--tmpfs', '/tmp:rw,size=2g', '--entrypoint', '/bin/bash',
                '-v', f'{window.ROOT}:/workspace', '-v', f'{out}:/receipt',
                'ai-tools/gufo-eval:f3-native-b4ebe27c-rocm101',
                '/workspace/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/build.sh']
        report = {'argv': argv, 'GPU_devices': False, 'started': time.time(), 'max_seconds': 1200}
        with (out / 'build.log').open('w') as log:
            p = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT)
            try:
                while p.poll() is None:
                    assert time.time() - report['started'] < 1200, 'build deadline'
                    assert shutil.disk_usage(window.ROOT).free >= window.DISK_FLOOR, 'disk floor'
                    time.sleep(2)
            finally:
                if p.poll() is None:
                    subprocess.run(['docker', 'stop', '-t', '5', name], capture_output=True, timeout=15)
                    p.wait(timeout=15)
        report.update(returncode=p.returncode, seconds=time.time() - report['started'])
        (out / 'BUILD.json').write_text(json.dumps(report, indent=2) + '\n')
        assert p.returncode == 0


if __name__ == '__main__':
    main()
