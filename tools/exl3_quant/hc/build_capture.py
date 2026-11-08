"""CPU-only build, inside the shared pinned container with frozen source.

Mount build root at /build, frozen source at /build/source and active hc tools
read-only at /hc-tools. This driver verifies the source before configuring.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess

from tools.docker.pinned_rocm101 import configure_args, environment


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    if Path('/dev/kfd').exists() or Path('/dev/dri').exists():
        raise RuntimeError('CPU collector build must not expose GPU devices')
    manifest = json.loads(Path('/build/SOURCE.json').read_text())
    if sha(Path(__file__)) != manifest['build_driver_sha256']:
        raise RuntimeError('capture build driver changed after prepare')
    for rel, digest in manifest['files_sha256'].items():
        if sha(Path('/build/source') / rel) != digest:
            raise RuntimeError('frozen source changed: ' + rel)
    env, flags = environment()
    env['GUFO_HC_CAPTURE_DIR'] = ''
    for d in ('tmp', 'cache'):
        Path('/build', d).mkdir(exist_ok=True)
    revision = manifest['revision'] + '-hc-capture'
    subprocess.run(configure_args(flags, revision), cwd='/build/source', env=env, check=True)
    subprocess.run(['cmake', '--build', '/build/obj', '--parallel', '4', '--target',
                    'qwen38_flash_next_rq_logits_probe'], cwd='/build/source', env=env, check=True)
    p = Path('/build/obj/tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits_probe')
    Path('/build/BUILD.json').write_text(json.dumps({'status': 'built_pending_gpu_validation',
        'revision': manifest['revision'], 'source_manifest_sha256': sha(Path('/build/SOURCE.json')),
        'binary_sha256': sha(p), 'bytes': p.stat().st_size, 'gpu_used': False}, indent=2) + '\n')
    if 'object_cache_provenance' in manifest:
        proof = dict(manifest['object_cache_provenance'],
                     source_manifest_sha256=sha(Path('/build/SOURCE.json')),
                     build_receipt_sha256=sha(Path('/build/BUILD.json')), gpu_used=False)
        Path('/build/CACHE-PROVENANCE.json').write_text(json.dumps(proof, indent=2) + '\n')


if __name__ == '__main__':
    main()
