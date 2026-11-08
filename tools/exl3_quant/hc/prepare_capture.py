"""Freeze an explicit B10-layout revision with first-attention HC export.

Only the new build/source snapshot is patched; no production checkout edits,
model access, GPU use, automatic rebuild or service operations.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile


TASK = Path(__file__).resolve().parent
REPO = TASK.parents[2]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def replace_once(text, old, new):
    if text.count(old) != 1:
        raise ValueError(f'capture patch anchor count {text.count(old)}: {old[:100]}')
    return text.replace(old, new, 1)


def instrument(source):
    """Fail on unsupported layout; never modify a different matching site."""
    source = Path(source)
    patches = {}
    p = source / 'src/models/qwen38_flash_next/runtime/executor_forward.cpp'
    s = '#include "hc_capture_experiment.hpp"\n' + p.read_text()
    s = replace_once(s, 'graph = graph && !diagnostics_active_;',
                     'graph = graph && !diagnostics_active_ && !gufo_hc_capture::Enabled();')
    patches[p] = s
    p = source / 'src/models/qwen38_flash_next/runtime/executor_hc.cpp'
    s = '#include "hc_capture_experiment.hpp"\n' + p.read_text()
    anchor = '  const bool extras = n_tokens <= options_.max_batch &&'
    s = replace_once(s, anchor, '''  const bool hc_capture = gufo_hc_capture::Enabled() &&
      m.down.data == model_->layers()[0].hc_attn.down.data;
  if (hc_capture) {
    if (normed || xn_half_ || norm_from_res)
      throw std::runtime_error("pilot expected first attention's explicit F32 norm");
    gufo_hc_capture::Save("down", s_.xn, n_tokens, c.HcDim(), stream_);
  }
''' + anchor)
    anchor = '      const Q8Input lo{s_.lo, s_.lo_q8_1, n_tokens, c.hc_low_rank};'
    s = replace_once(s, anchor, '''      if (hc_capture)
        gufo_hc_capture::Save("up", s_.lo, n_tokens, c.hc_low_rank, stream_);
''' + anchor)
    anchor = '''      if (half_gate) {
        if (diagnostics_active_)'''
    s = replace_once(s, anchor, '''      if (hc_capture)
        gufo_hc_capture::Save("up", s_.lo, n_tokens, c.hc_low_rank, stream_);
''' + anchor)
    patches[p] = s
    p = source / 'tests/models/qwen38_flash_next/rq_logits_probe.cpp'
    s = '#include "hc_capture_experiment.hpp"\n' + p.read_text()
    anchor = '''        Require(
            executor->Forward(*session, std::span(tokens).subspan(p, n), rows,'''
    s = replace_once(s, anchor, '''        gufo_hc_capture::Context(w, p, decode);
''' + anchor)
    patches[p] = s
    # All anchors must validate before writing any changed source.
    for p, s in patches.items():
        p.write_text(s)
    shutil.copyfile(TASK / 'capture_hook.hpp', source / 'hc_capture_experiment.hpp')
    return [str(p.relative_to(source)) for p in patches] + ['hc_capture_experiment.hpp']


def cache_provenance(objects, reference, receipt, revision):
    objects, reference, receipt = map(Path, (objects, reference, receipt))
    binary_sha = digest(objects / 'gufo')
    if digest(reference) != binary_sha:
        raise ValueError('object cache binary differs from qualified reference')
    record = json.loads(receipt.read_text())
    declaration = record.get('trees', {}).get('C', '')
    if binary_sha not in declaration or revision[:8] not in declaration:
        raise ValueError('B10 receipt does not identify this revision and binary')
    return {'cache_binary_sha256': binary_sha, 'reference': str(reference.resolve()),
            'receipt': str(receipt.resolve()), 'receipt_sha256': digest(receipt),
            'declaration': declaration}


def prepare(revision, output, objects, cache_reference, cache_receipt):
    output = Path(output).resolve()
    objects = Path(objects).resolve()
    if output.exists() or not output.is_relative_to(REPO / 'build'):
        raise ValueError('capture output must be a new directory under gufo/build')
    full_revision = subprocess.check_output(['git', '-C', str(REPO), 'rev-parse', revision + '^{commit}'], text=True).strip()
    provenance = cache_provenance(objects, cache_reference, cache_receipt, full_revision)
    cache = (objects / 'CMakeCache.txt').read_text()
    for value in ('CMAKE_HOME_DIRECTORY:INTERNAL=/build/source',
                  'CMAKE_BUILD_TYPE:STRING=RelWithDebInfo',
                  'CMAKE_HIP_ARCHITECTURES:STRING=gfx1151'):
        if value not in cache:
            raise ValueError('object cache does not match retained pinned build: ' + value)
    if shutil.disk_usage(REPO).free < 150 * 1024**3:
        raise ValueError('need >=150GiB free before copying retained build objects')
    output.mkdir(parents=True, exist_ok=False)
    source = output / 'source'; source.mkdir()
    with tempfile.TemporaryFile() as f:
        subprocess.run(['git', '-C', str(REPO), 'archive', full_revision], stdout=f, check=True)
        f.seek(0)
        with tarfile.open(fileobj=f) as tar:
            tar.extractall(source, filter='data')
    # Retain mtimes only for identical source; changed inputs rebuild.
    for p in source.rglob('*'):
        live = REPO / p.relative_to(source)
        if p.is_file() and not p.is_symlink() and live.is_file() and not live.is_symlink():
            if p.read_bytes() == live.read_bytes():
                shutil.copystat(live, p)
            else:
                p.touch()
    changed = instrument(source)
    shutil.copytree(objects, output / 'obj', copy_function=shutil.copy2)
    files = {str(p.relative_to(source)): digest(p) for p in source.rglob('*')
             if p.is_file() and not p.is_symlink()}
    manifest = {'revision': full_revision, 'status': 'prepared_not_built_or_gpu_verified',
                'scope': 'first attention HC only; no final mixer or MTP',
                'changed_files': changed, 'files_sha256': files,
                'prepare_sha256': digest(Path(__file__)),
                'hook_sha256': digest(TASK / 'capture_hook.hpp'),
                'build_driver_sha256': digest(TASK / 'build_capture.py'),
                'object_cache': str(objects), 'object_cache_cmake_sha256': digest(objects / 'CMakeCache.txt'),
                'object_cache_provenance': provenance,
                'gpu_used': False}
    (output / 'SOURCE.json').write_text(json.dumps(manifest, indent=2) + '\n')
    return {'output': str(output), 'revision': full_revision, 'files': len(files)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--revision', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--objects', required=True)
    parser.add_argument('--cache-reference', required=True)
    parser.add_argument('--cache-receipt', required=True)
    print(json.dumps(prepare(**vars(parser.parse_args()))))


if __name__ == '__main__':
    main()
