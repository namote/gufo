"""User-authorized replacement of internal legacy Flash-Next GGUF by Kyojin EXL3."""
import fcntl
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from test.experiments.gufo_upstream_port_20261023 import window

ROOT = window.ROOT
OLD = ROOT / 'storage/models/llm/qwen-3.8-flash-next-ud-q4-k-xl-gguf'
BACKUP = Path('/run/media/m2on/T5 EVO/AI/model/originals-flash-next-20261029/unsloth-Qwen3.8-Flash-Next-GGUF')
SOURCE = Path('/run/media/m2on/T5 EVO/AI/model/flash-next-exl3-kyojin-3.05')
DEST = ROOT / 'storage/models/llm/flash-next-exl3-kyojin'
OUT = window.OUT_ROOT / 'kyojin-local-prepare'


def main():
    with window.LOCK.open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        health = window.host()
        assert window.quiet(health) and health['available_bytes'] >= 80 * 1024**3
        assert not subprocess.check_output(['docker', 'ps', '-q']).strip()
        expected = json.loads((SOURCE / 'HF-TREE.json').read_text())
        files = [r for r in expected if r['type'] == 'file']
        payloads = []
        for line in (OLD / 'SHA256SUMS').read_text().splitlines():
            digest, name = line.split(maxsplit=1)
            path = OLD / name
            assert path.is_file() and not path.is_symlink() and path.suffix == '.gguf'
            backup = BACKUP / (name if name.startswith('MTP/') else 'UD-Q4_K_XL/' + name)
            assert backup.is_file() and backup.stat().st_size == path.stat().st_size
            payloads.append({'path': str(path), 'bytes': path.stat().st_size,
                             'prior_verified_sha256': digest, 'external_backup': str(backup)})
        assert len(payloads) == 5
        free = shutil.disk_usage(ROOT).free
        assert free + sum(x['bytes'] for x in payloads) - sum(x['size'] for x in files) > window.DISK_FLOOR
        assert not DEST.exists(), 'No overwrites of existing destination'
        check = subprocess.run(['sudo', '-n', 'fuser', *[p['path'] for p in payloads]],
                               capture_output=True, text=True, timeout=30)
        assert check.returncode == 1 and not check.stdout.strip(), check
        report = {'authorization': 'User: bring tensors to local SSD; removing our GGUF should provide space.',
                  'source': str(SOURCE), 'destination': str(DEST), 'deleted_payloads': payloads,
                  'full_old_gguf_rehash': False, 'free_before': free, 'files': [], 'status': 'starting'}
        def save():
            (OUT / 'LOCAL-COPY.json').write_text(json.dumps(report, indent=2) + '\n')
        save()
        for item in payloads:
            Path(item['path']).unlink()
        report['free_after_removal'] = shutil.disk_usage(ROOT).free
        report['status'] = 'copying'
        save()
        subprocess.run(['sudo', '-n', 'fstrim', '-v', '/'], check=True, stdout=(OUT / 'model-fstrim.log').open('w'))
        DEST.mkdir()
        begin = time.monotonic()
        copied = 0
        for row in files:
            src, dst = SOURCE / row['path'], DEST / row['path']
            assert src.is_file() and not src.is_symlink() and src.stat().st_size == row['size']
            assert shutil.disk_usage(ROOT).free - row['size'] >= window.DISK_FLOOR
            dst.parent.mkdir(parents=True, exist_ok=True)
            partial = dst.with_name(dst.name + '.partial')
            sha256 = hashlib.sha256()
            git = hashlib.sha1(f"blob {row['size']}\0".encode())
            file_start = time.monotonic()
            with src.open('rb', buffering=0) as reader, partial.open('xb', buffering=0) as writer:
                while data := reader.read(4 * 1024**2):
                    writer.write(data)
                    sha256.update(data)
                    if 'lfs' not in row:
                        git.update(data)
                    copied += len(data)
                writer.flush()
            got = sha256.hexdigest() if 'lfs' in row else git.hexdigest()
            want = row['lfs']['oid'] if 'lfs' in row else row['oid']
            assert got == want and partial.stat().st_size == row['size'], row['path']
            partial.replace(dst)
            shutil.copystat(src, dst)
            record = {'path': row['path'], 'size': row['size'], 'sha256': sha256.hexdigest(),
                      'expected_identity': want, 'verified': True,
                      'seconds': time.monotonic() - file_start}
            report['files'].append(record)
            report['copied_bytes'] = copied
            report['elapsed_seconds'] = time.monotonic() - begin
            save()
            print(json.dumps({'file': row['path'], 'GiB_copied': copied / 2**30,
                              'elapsed_s': round(report['elapsed_seconds'], 1)}), flush=True)
        for name in ['HF-TREE.json', 'VERIFY-SHA256.log', 'PROVENANCE.md']:
            shutil.copy2(SOURCE / name, DEST / name)
        report.update(status='COMPLETE', free_after=shutil.disk_usage(ROOT).free)
        save()
        (DEST / 'LOCAL-COPY.json').write_text(json.dumps(report, indent=2) + '\n')


if __name__ == '__main__':
    main()
