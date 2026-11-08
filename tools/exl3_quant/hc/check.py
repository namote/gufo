"""Run optional HC numerical unit tests in the shared CPU-only container."""
import argparse
import os
from pathlib import Path
import subprocess
import uuid

from tools.docker.pinned_rocm101 import PYTHON, docker_argv


def main(argv=None):
    repo = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=repo / 'build/exl3-hc/cpu')
    args = parser.parse_args(argv)
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / 'tmp').mkdir(exist_ok=True)
    command = ['PYTHONPATH=/build/source', 'PYTHONDONTWRITEBYTECODE=1',
               'OPENBLAS_NUM_THREADS=1', 'OMP_NUM_THREADS=1', 'TMPDIR=/build/tmp',
               PYTHON, '-m', 'unittest',
               'tools.exl3_quant.hc.test_cpu', 'tools.exl3_quant.hc.test_refine', '-v']
    argv = docker_argv(out, repo, command, gpu=False, source_readonly=True,
                       raw_entrypoint='/usr/bin/env')
    name = 'gufo-hc-cpu-' + uuid.uuid4().hex
    argv[2:2] = ['--name', name]
    with (out / 'cpu.log').open('x') as log:
        try:
            return subprocess.run(argv, stdout=log, stderr=subprocess.STDOUT,
                                  env=os.environ, timeout=120).returncode
        finally:
            # The Docker client timing out does not guarantee container exit.
            # This unique name owns only this check, never another GPU job.
            cleanup = subprocess.run(['docker', 'rm', '-f', name], capture_output=True,
                                     text=True, timeout=30)
            (out / 'cleanup.log').write_text(cleanup.stdout + cleanup.stderr)
            if cleanup.returncode and 'No such container' not in cleanup.stderr:
                raise RuntimeError('owned CPU container cleanup failed; see cleanup.log')


if __name__ == '__main__':
    raise SystemExit(main())
