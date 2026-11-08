"""Immutable, pre-quantization MTP Hessian/reference cache.

Capture math/state identity is independent of a target bit recipe. A changed
recipe needs a separate explicit approval binding both recipes and this cache.
No pickle, silent partial reads, fallback, or overwrite of earlier evidence.
"""
import hashlib
import json
import os
from pathlib import Path


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda:f.read(8*2**20),b''):h.update(part)
    return h.hexdigest()


def object_sha(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def validate_h(value):
    import torch
    H=value['H'];bad=value['inf_nan']
    assert value['finalized'] is False and not H.is_meta
    assert H.dtype==torch.float32 and H.ndim==2 and H.shape[0]==H.shape[1] and H.shape[0]>0
    assert torch.isfinite(H).all()
    assert bad.dtype==torch.int64 and bad.shape==(2,) and torch.equal(bad,torch.zeros_like(bad))
    assert type(value['count']) is int and value['count']>=0
    if 'count_dev' in value:
        assert value['count_dev'].dtype==torch.int64 and value['count_dev'].ndim==0
    assert int(value.get('count_dev',value['count']))>0
    assert type(value['num_total']) is int and value['num_total']>0
    assert isinstance(value['first_key'],str) and value['first_key']
    assert torch.device(value['device'])==torch.device(value['H_swap_device'])


def save_capture(path, capture, refs, binding, strategy):
    import torch
    from safetensors.torch import save_file
    path=Path(path);path.mkdir()
    tensors={};maps=[]
    allowed={'H','count','count_dev','first_key','finalized','num_total','inf_nan','device','H_swap_device'}
    for index,(key,value) in enumerate(sorted(capture.items())):
        assert not (set(value)-allowed),('Unrecognized pre-quantization H fields',key,set(value)-allowed)
        validate_h(value)
        record={'key':key,'values':{},'tensors':{},'devices':{}}
        for name,entry in value.items():
            if isinstance(entry,torch.Tensor):
                tensor_key=f'h{index}.{name}'
                tensors[tensor_key]=entry.detach().cpu().contiguous().clone()
                record['tensors'][name]={'key':tensor_key,'device':str(entry.device)}
            elif isinstance(entry,torch.device):
                record['devices'][name]=str(entry)
            else:
                assert type(entry) in (int,str,bool,float,type(None)),(key,name,type(entry))
                record['values'][name]=entry
        maps.append(record)
    for i,value in refs.items():
        assert type(i) is int and torch.isfinite(value).all()
        tensors[f'ref.{i}']=value.detach().cpu().contiguous().clone()
    payload=path/'numeric.safetensors';save_file(tensors,str(payload))
    with payload.open('rb') as f:os.fsync(f.fileno())
    manifest={'format':'gufo.mtp.capture.v1','status':'complete_pre_quantization',
              'binding':binding,'source_strategy':strategy,'maps':maps,
              'reference_indices':sorted(refs),'tensor_keys':sorted(tensors),
              'payload_sha256':sha(payload)}
    pending=path/'MANIFEST.pending'
    with pending.open('x') as f:
        json.dump(manifest,f,indent=2);f.flush();os.fsync(f.fileno())
    os.replace(pending,path/'MANIFEST.json')
    fd=os.open(path,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)
    return manifest


def load_capture(path, binding, strategy, approved_remap=None):
    import torch
    from safetensors.torch import load_file
    path=Path(path)
    manifest=json.loads((path/'MANIFEST.json').read_text())
    assert manifest['format']=='gufo.mtp.capture.v1' and manifest['status']=='complete_pre_quantization'
    assert manifest['binding']==binding,'Capture source/math/state identity changed'
    if manifest['source_strategy']!=strategy:
        assert approved_remap is not None,'Changed recipe requires explicit approved remap'
        approval=json.loads(Path(approved_remap).read_text())
        assert approval=={'status':'approved_mtp_capture_recipe_remap',
                         'cache_manifest_sha256':sha(path/'MANIFEST.json'),
                         'source_strategy_sha256':object_sha(manifest['source_strategy']),
                         'target_strategy_sha256':object_sha(strategy)},'Remap approval mismatch'
    assert sha(path/'numeric.safetensors')==manifest['payload_sha256'],'Capture payload changed/partial'
    tensors=load_file(str(path/'numeric.safetensors'))
    assert set(tensors)==set(manifest['tensor_keys'])
    capture={}
    for record in manifest['maps']:
        value=dict(record['values'])
        value.update({name:torch.device(device) for name,device in record['devices'].items()})
        value.update({name:tensors[info['key']].to(torch.device(info['device']))
                      for name,info in record['tensors'].items()})
        validate_h(value)
        assert record['key'] not in capture
        capture[record['key']]=value
    refs={i:tensors[f'ref.{i}'] for i in manifest['reference_indices']}
    return capture,refs,manifest
