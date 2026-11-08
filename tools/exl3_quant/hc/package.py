"""Assemble a complete, model-bound runtime HC sidecar in a new directory.

This only packages already calibrated files; it never quantizes, overwrites a
model or starts a service. Install the resulting directory as model/hc_q8 only
after native loader/quality/throughput qualification. Also install its manifest.json
as model/hc_q8.json: that root receipt participates in continuation identity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
from .capture_window import save,sha
from .run_all import names
from .weights import read_header,read_weight_payload


def package(candidate, variant, target_model, output):
    candidate,target_model,output=map(lambda p:Path(p).resolve(),(candidate,target_model,output))
    if output.exists() or output.is_relative_to(candidate) or output.is_relative_to(target_model):
        raise ValueError('require a fresh output outside model and candidate')
    if variant not in ('gptq','refined_up'):raise ValueError('unknown variant')
    receipt=json.loads((candidate/variant/'PACK.json').read_text())
    state=json.loads((candidate/'STATE.json').read_text())
    expected={n+'.input_mix_weight_'+p+'.weight' for n in names() for p in ('down','up')}
    if (state['status']!='completed_pending_native_validation' or receipt['tensor_count']!=200
            or receipt['calibrated_tensor_count']!=200 or set(receipt['tensors'])!=expected
            or receipt['target_index_sha256']!=sha(target_model/'model.safetensors.index.json')):
        raise ValueError('incomplete/mismatched full calibration')
    pack=candidate/(variant+'.safetensors')
    if sha(pack)!=receipt['pack_sha256']:raise ValueError('changed packed file')
    header,start=read_header(pack)
    if set(header)-{'__metadata__'}!=expected:raise ValueError('pack key mismatch')
    manifest={'schema':'gufo.flash-next.hc-q8.v1','model_index_sha256':receipt['target_index_sha256'],
              'variant':variant,'tensor_count':200,'candidate_pack_sha256':sha(candidate/variant/'PACK.json'),
              'calibration_scope':'1024 scalar teacher-forced rows, 8 documents; holdout873 rows',
              'reference':'original BF16; exact FP32 expansion; FP64 GPTQ/refinement',
              'status':'pending_runtime_qualification','tensors':{}}
    spans=[]
    with pack.open('rb') as stream:
        for key in sorted(expected):
            t=receipt['tensors'][key];entry=header[key]
            _,identity=read_weight_payload(target_model,key,'F16');rows,cols=identity['shape']
            if (not t['calibrated'] or t['target_identity']!=identity or entry['dtype']!='U8'
                    or entry['shape']!=[rows,cols//32,34]):raise ValueError('wrong tensor identity/layout')
            first,last=entry['data_offsets'];size=rows*(cols//32)*34
            if first<0 or last-first!=size or start+last>pack.stat().st_size:raise ValueError('invalid tensor offsets')
            spans.append((first,last));stream.seek(start+first);raw=stream.read(size)
            digest=hashlib.sha256(raw).hexdigest()
            if len(raw)!=size or digest!=t['sha256']:raise ValueError('packed tensor payload changed')
            manifest['tensors'][key]={'target_weight_sha256':identity['weight_sha256'],
                 'packed_weight_sha256':digest,'shape':[rows,cols],
                 'source_bf16_identity':t['source_identity']}
    if any(a[1]>b[0] for a,b in zip(sorted(spans),sorted(spans)[1:])):raise ValueError('overlapping tensors')
    output.mkdir(parents=True)
    shutil.copyfile(pack,output/'model.safetensors')
    shutil.copyfile(target_model/'config.json',output/'config.json')
    save(output/'manifest.json',manifest)
    save(output/'ASSEMBLY.json',{'status':'packaged_pending_runtime_qualification','gpu_used':False,
        'candidate':str(candidate),'candidate_run_sha256':sha(candidate/'RUN.json'),
        'files':{p.name:sha(p) for p in output.iterdir() if p.is_file()}})
    return manifest


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('candidate','variant','target-model','output'):p.add_argument('--'+name,required=True)
    result=package(**vars(p.parse_args(argv)));print(json.dumps({'tensor_count':result['tensor_count'],'status':result['status']}))


if __name__=='__main__':main()
