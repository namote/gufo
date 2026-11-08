"""Explicit one-shot c03 prefill quality collection or CPU scoring.

Parent owns GPU lease/host memory and container lifecycle. No service, model
build, baseline collection, fallback, automatic retry or threshold invention.
"""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(8*2**20),b''):h.update(b)
    return h.hexdigest()


def validate_meta(m,ref):
    assert m['schema']=='gufo.rq-logits.v1' and m['backend']=='exl3'
    assert m['path']=='prefill' and m['kv_cache']=='fp16'
    assert m['records']==32760 and m['record_bytes']==3096 and m['top_k']==256
    assert m['vocab_size']==248320 and m['max_prompt_batch']==512
    assert m['head_replay_checks']==64
    assert m['windows_index_sha256']==ref['windows_index_sha256']
    assert m['windows_tokens_sha256']==ref['windows_tokens_sha256']
    assert m['reference']['bin_sha256']==ref['bin_sha256']
    assert len(m['windows'])==8 and all(w['scored']==4095 for w in m['windows'])


def runtime_paths(maps):
    paths=sorted({l.split()[-1] for l in maps.splitlines() if any(k in l for k in ('libamdhip64','libhsa-runtime64','librocblas','libhipblas'))})
    assert any('/nix/store/' in p and 'clr-10.1' in p and 'libamdhip64' in p for p in paths)
    assert any('/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/' in p and 'libhsa-runtime64' in p for p in paths)
    assert not any('/opt/rocm/' in p for p in paths)
    assert not any('.gguf' in l for l in maps.splitlines()),'Unexpected GGUF mapping'
    return paths


def finite_json(x):
    if isinstance(x,float):assert math.isfinite(x)
    elif isinstance(x,dict):
        for v in x.values():finite_json(v)
    elif isinstance(x,list):
        for v in x:finite_json(v)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True)
    ap.add_argument('--phase',choices=('collect','score'),required=True);a=ap.parse_args()
    c=json.loads(a.contract.read_text());assert c['status']=='reviewed_final_quality_stage'
    assert c['script_sha256']==sha(__file__)
    required={c['metrics'],str(Path(c['metrics']).with_name('logits_io.py')),c['build_receipt'],
              str(Path(c['reference']).with_suffix('.json')),str(Path(c['model'])/'GUFO-COMPILE.json')}
    assert required<=c['files_sha256'].keys(),'Unpinned scorer/build/reference/pack evidence'
    for p,d in c['files_sha256'].items():assert sha(p)==d,p
    out=Path(c['out']);assert out.is_absolute() and out==out.resolve()
    out.mkdir(parents=True,exist_ok=True);prefix=out/'c03-prefill'
    reference=Path(c['reference']);corpus=Path(c['windows'])
    ref=json.loads(reference.with_suffix('.json').read_text())
    assert ref['label']=='flash-next-hf-bf16' and ref['path']=='prefill' and ref['records']==32760
    assert sha(reference.with_suffix('.json'))==c['reference_metadata_sha256']
    assert sha(corpus)==ref['windows_index_sha256']
    assert sha(corpus.with_name('windows.bin'))==ref['windows_tokens_sha256']
    assert Path(c['model']).is_dir()
    assert sha(Path(c['model'])/'GUFO-COMPILE.json')==c['compile_receipt_sha256']
    if a.phase=='collect':
        assert not prefix.with_suffix('.bin').exists() and not prefix.with_suffix('.json').exists()
        binary=Path(c['binary']);assert sha(binary)==c['binary_sha256']
        assert os.environ.get('LD_LIBRARY_PATH')==c['ld_library_path']
        argv=[str(binary),c['model'],'--windows',str(corpus),'--path','prefill','--batch','512',
              '--prefix','2048','--top-k','256','--reference',str(reference),'--out',str(prefix),
              '--label','c03-bf16source-fp16kv','--toolchain',c['toolchain']]
        seconds=c.get('seconds',900);assert 0<seconds<=1200
        started=time.monotonic();maps='';state={'status':'running','argv':argv,'contract_sha256':sha(a.contract)}
        with (out/'collect.log').open('x') as log:
            p=subprocess.Popen(argv,stdout=log,stderr=subprocess.STDOUT)
            try:
                while p.poll() is None:
                    try:
                        current=Path(f'/proc/{p.pid}/maps').read_text()
                        if 'libhsa-runtime64' in current and 'libamdhip64' in current:maps=current
                    except FileNotFoundError:pass
                    assert time.monotonic()-started<seconds,'Quality deadline exceeded'
                    time.sleep(.2)
                assert p.returncode==0,f'Collector failed {p.returncode}'
                paths=runtime_paths(maps);m=json.loads(prefix.with_suffix('.json').read_text());validate_meta(m,ref)
                assert prefix.with_suffix('.bin').stat().st_size==32760*3096
                assert sha(prefix.with_suffix('.bin'))==m['bin_sha256']
                state.update(status='collected_pending_cpu_scoring',runtime_paths=paths,metadata_sha256=sha(prefix.with_suffix('.json')),
                             binary_sha256=c['binary_sha256'],records=32760)
            except BaseException as e:
                state.update(status='failed',error=repr(e));raise
            finally:
                if p.poll() is None:p.kill();p.wait(timeout=30)
                state.update(returncode=p.returncode,seconds=time.monotonic()-started)
                (out/'runtime-maps.txt').write_text(maps)
                (out/'COLLECTION.json').write_text(json.dumps(state,indent=2)+'\n')
    else:
        collected=json.loads((out/'COLLECTION.json').read_text())
        assert collected['status']=='collected_pending_cpu_scoring'
        assert collected['contract_sha256']==sha(a.contract)
        assert collected['metadata_sha256']==sha(prefix.with_suffix('.json'))
        validate_meta(json.loads(prefix.with_suffix('.json').read_text()),ref)
        score=out/'score-prefill.json';assert not score.exists(),'Reuse saved score, no automatic overwrite'
        argv=[sys.executable,c['metrics'],'score','--reference',str(reference),'--bootstrap','2000',
              '--block-tokens','256','--out',str(score),str(prefix)]
        with (out/'score.log').open('x') as log:subprocess.run(argv,stdout=log,stderr=subprocess.STDOUT,check=True,timeout=600)
        metrics=json.loads(score.read_text());finite_json(metrics)
        assert metrics['schema']=='gufo.rq-quality-score.v1' and len(metrics['reports'])==1
        (out/'QUALITY.json').write_text(json.dumps(dict(status='scored_pending_independent_quality_review',
            score_sha256=sha(score),collection_sha256=sha(out/'COLLECTION.json'),reference_kind='saved HF BF16 prefill',
            scope='32760 teacher-forced prefill positions, FP16 KV, MTP off; not decode/vision/MTP quality or throughput',
            full_KL=False,local_RFN_is_not_quality_gate=True),indent=2)+'\n')

if __name__=='__main__':main()
