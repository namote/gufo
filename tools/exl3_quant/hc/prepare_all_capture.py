"""Freeze B10 with all100 HC sites and scalar teacher-forced MTP collection."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import tarfile
import tempfile

from .prepare_capture import REPO, TASK, cache_provenance, digest, replace_once


def instrument(source):
    source = Path(source)
    changed = {}
    forward = source/'src/models/qwen38_flash_next/runtime/executor_forward.cpp'
    s = '#include "hc_all_experiment.hpp"\n'+forward.read_text()
    s = replace_once(s, 'graph = graph && !diagnostics_active_;',
                     'graph = graph && !diagnostics_active_ && !gufo_hc_all::Enabled();')
    changed[forward] = s
    hc = source/'src/models/qwen38_flash_next/runtime/executor_hc.cpp'
    s = '#include "hc_all_experiment.hpp"\n'+hc.read_text()
    anchor = '  const bool extras = n_tokens <= options_.max_batch &&'
    s = replace_once(s, anchor, '''  if (gufo_hc_all::active) {
    gufo_hc_all::Require(n_tokens == 1 && !prefill_phase && !xn_half_ &&
                         !norm_from_res && exl3_res_pending_.res == nullptr,
                         "unsupported active HC capture route");
    gufo_hc_all::Save(m.down.data, false, s_.xn, n_tokens, c.HcDim(), stream_);
  }
'''+anchor)
    anchor = '      const Q8Input lo{s_.lo, s_.lo_q8_1, n_tokens, c.hc_low_rank};'
    s = replace_once(s, anchor, '''      gufo_hc_all::Save(m.down.data, true, s_.lo, n_tokens, c.hc_low_rank, stream_);
'''+anchor)
    anchor = '      if (half_gate) {\n        if (diagnostics_active_)'
    s = replace_once(s, anchor, '''      gufo_hc_all::Save(m.down.data, true, s_.lo, n_tokens, c.hc_low_rank, stream_);
'''+anchor)
    changed[hc] = s
    batch = source/'src/models/qwen38_flash_next/runtime/executor_batch.cpp'
    s = '#include "hc_all_experiment.hpp"\n'+batch.read_text()
    anchor = '  const bool fused_down = model_->output().exl3 != nullptr &&'
    s = replace_once(s, anchor, '''  gufo_hc_all::Save(m.down.data, false, s_.xn, rows, c.HcDim(), stream_);
'''+anchor)
    anchor = '  if (!DenseBatch(m.up, s_.lo, s_.hc_gate, rows, error))'
    s = replace_once(s, anchor, '''  gufo_hc_all::Save(m.down.data, true, s_.lo, rows, c.hc_low_rank, stream_);
'''+anchor)
    changed[batch] = s
    probe = source/'tests/models/qwen38_flash_next/rq_logits_probe.cpp'
    s = '#include "hc_all_experiment.hpp"\n'+probe.read_text()
    old = '''    auto device = q::rocm::DeviceModel::Upload(*weights, *reader, nullptr,
                                               nullptr, &error);'''
    s = replace_once(s, old, '''    auto mtp = q::LoadExl3Mtp(model_path, *reader, *weights, &error);
    Require(bool(mtp), error);
    auto device = q::rocm::DeviceModel::Upload(*weights, *reader, &*mtp, reader.get(), &error);''')
    s = replace_once(s, '    Require(bool(device), error);', '''    Require(bool(device), error);
    gufo_hc_all::Register(*device);''')
    s = replace_once(s, 'gufo::core::SessionMode::kAutoregressive, context, &error);',
                     'gufo::core::SessionMode::kSpeculative, context, &error);')
    s = replace_once(s, '    std::size_t head_checks = 0;', '''    Require(path_name == "decode" && prefix == 32 && position_limit <= 128 && window_limit <= 16,
            "all-HC collector requires bounded scalar decode");
    gufo::crypto::Sha256Hasher main_hash, mtp_hash;
    std::vector<float> mtp_head(c.hidden_size);
    q::MtpTrace trace;
    trace.head = mtp_head;
    std::size_t head_checks = 0;''')
    start = s.index('      for (std::size_t p = 0; p < end;) {')
    stop = s.index('      auto entry = json::Value::object();', start)
    s = s[:start]+'''      gufo_hc_all::active = false;
      Require(executor->Forward(*session, std::span(tokens).first(prefix), 0, nullptr,
              q::rocm::Executor::ForwardMode::kPrefill, &error), error);
      Require(executor->MtpPosition(*session) == prefix-1, "MTP prefix alignment differs");
      Require(executor->MtpForward(*session, std::span(tokens).subspan(prefix, 1), 0,
                                   {.trace = &trace}, &error), error);
      Require(executor->MtpPosition(*session) == prefix, "MTP anchor alignment differs");
      gufo_hc_all::Begin(w, first);
      for (std::size_t p = first; p < end; ++p) {
        gufo_hc_all::position = p;
        const auto t = Clock::now();
        Require(executor->Forward(*session, std::span(tokens).subspan(p, 1), 1,
                  logits.data(), q::rocm::Executor::ForwardMode::kDecode, &error), error);
        main_hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(logits.data()),
                                   c.vocab_size*sizeof(float)));
        writer.Add(static_cast<std::uint32_t>(w), static_cast<std::uint32_t>(p), tokens[p+1],
                   std::span(logits).first(c.vocab_size), reference.get());
        Require(executor->MtpForward(*session, std::span(tokens).subspan(p+1, 1), 0,
                                     {.trace = &trace}, &error), error);
        for (float value : mtp_head) Require(std::isfinite(value), "nonfinite MTP head");
        mtp_hash.Update(std::span(reinterpret_cast<const std::uint8_t*>(mtp_head.data()),
                                 mtp_head.size()*sizeof(float)));
        Require(session->position() == p+1 && executor->MtpPosition(*session) == p+1,
                "main/MTP teacher-force positions diverged");
        forward_ms += Ms(t);
      }
      gufo_hc_all::End(end);
'''+s[stop:]
    s = replace_once(s, '    meta["bin_sha256"] = writer.Finish();', '''    meta["bin_sha256"] = writer.Finish();
    meta["full_logits_sha256"] = main_hash.FinishHex();
    meta["mtp_head_sha256"] = mtp_hash.FinishHex();
    meta["scope"] = "all100HC scalar decode, teacher-forced MTP; no recursive draft or prefill qualification";''')
    s = replace_once(s, '  } catch (const std::exception& e) {', '''  } catch (const std::exception& e) {
    gufo_hc_all::active = false;''')
    changed[probe] = s
    for path, text in changed.items(): path.write_text(text)
    shutil.copy2(TASK/'all_capture_hook.hpp', source/'hc_all_experiment.hpp')
    return [str(p.relative_to(source)) for p in changed]+['hc_all_experiment.hpp']


def prepare(output, objects):
    output, objects = Path(output).resolve(), Path(objects).resolve()
    if output.exists() or not output.is_relative_to(REPO/'build/exl3-hc'):
        raise ValueError('new output under build/exl3-hc required')
    previous = json.loads((objects.parent/'SOURCE.json').read_text())
    revision = previous['revision']
    proof = previous['object_cache_provenance']
    cache_provenance(objects, proof['reference'], proof['receipt'], revision)
    cache_bytes = sum(p.stat().st_size for p in objects.rglob('*') if p.is_file())
    if shutil.disk_usage(REPO).free < cache_bytes+20*2**30:
        raise ValueError('insufficient space for retained objects and bounded captures')
    output.mkdir(parents=True); source=output/'source'; source.mkdir()
    with tempfile.TemporaryFile() as f:
        subprocess.run(['git','-C',str(REPO),'archive',revision],stdout=f,check=True);f.seek(0)
        with tarfile.open(fileobj=f) as t: t.extractall(source,filter='data')
    old_source=objects.parent/'source'
    for path in source.rglob('*'):
        old=old_source/path.relative_to(source)
        if path.is_file() and old.is_file() and path.read_bytes()==old.read_bytes(): shutil.copystat(old,path)
    changed=instrument(source)
    shutil.copytree(objects,output/'obj',copy_function=shutil.copy2)
    manifest={'revision':revision,'scope':'all100HC scalar teacher-forced decode',
              'files_sha256':{str(p.relative_to(source)):digest(p) for p in source.rglob('*') if p.is_file() and not p.is_symlink()},
              'changed_files':changed,'build_driver_sha256':digest(TASK/'build_capture.py'),
              'prepare_sha256':digest(Path(__file__)),'object_cache_provenance':proof,'gpu_used':False}
    (output/'SOURCE.json').write_text(json.dumps(manifest,indent=2)+'\n')
    return {'output':str(output),'scope':manifest['scope'],'revision':revision}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output',required=True);p.add_argument('--objects',required=True)
    print(json.dumps(prepare(**vars(p.parse_args())),indent=2))
