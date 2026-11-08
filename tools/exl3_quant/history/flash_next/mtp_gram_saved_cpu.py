"""CPU-only independent diagnosis of preserved failed Gram operands.

No torch/numpy import, no GPU execution, no original/control recomputation.
The complete stored candidate/control matrices are compared; FP64 math.fsum
oracles cover all diagonals, the prior 8x8 sample, and 256 fixed off-diagonals.
"""
import argparse
from collections import OrderedDict
import hashlib
import io
import json
import math
from pathlib import Path
import pickle
import random
import statistics
import struct
import time
import zipfile


def rebuild(storage, offset, shape, stride, requires_grad, hooks):
    assert not requires_grad and not hooks and offset == 0
    return {'storage':storage,'shape':shape,'stride':stride}


class MetadataOnly(pickle.Unpickler):
    def find_class(self, module, name):
        allowed = {('torch._utils','_rebuild_tensor_v2'):rebuild,
                   ('torch','HalfStorage'):'half',('torch','FloatStorage'):'float',
                   ('collections','OrderedDict'):OrderedDict}
        assert (module,name) in allowed
        return allowed[module,name]

    def persistent_load(self, value):
        kind,dtype,key,location,count=value
        assert kind=='storage' and dtype in ('half','float') and location=='cpu'
        assert str(int(key))==key and count>0
        return (dtype,key,count)


def metrics(ref, candidate):
    assert len(ref)==len(candidate) and ref
    difference=[b-a for a,b in zip(ref,candidate)]
    assert all(math.isfinite(v) for v in [*ref,*candidate])
    norm=math.fsum(v*v for v in ref)
    return {'relative_l2':math.sqrt(math.fsum(v*v for v in difference)/max(norm,1e-300)),
            'max_abs':max(abs(v) for v in difference),'signed_mean':statistics.fmean(difference),
            'reference_rms':math.sqrt(norm/len(ref))}


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--saved',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    assert not a.out.exists();started=time.monotonic()
    payload=a.saved.read_bytes()
    with zipfile.ZipFile(io.BytesIO(payload)) as z:
        metadata=next(n for n in z.namelist() if n.endswith('/data.pkl'))
        prefix=metadata.removesuffix('data.pkl')
        assert z.read(prefix+'byteorder')==b'little'
        desc=MetadataOnly(io.BytesIO(z.read(metadata))).load()
        assert set(desc)=={'x','control','candidate'}
        arrays={}
        for name,entry in desc.items():
            dtype,key,count=entry['storage'];data=z.read(prefix+'data/'+key)
            fmt='e' if dtype=='half' else 'f'
            assert len(data)==count*struct.calcsize(fmt)
            arrays[name]=struct.unpack('<'+str(count)+fmt,data)
    assert desc['x']['shape']==(1,8191,640) and desc['x']['stride']==(8191*640,640,1)
    assert all(desc[k]['shape']==(1,640,640) and desc[k]['stride']==(640*640,640,1) for k in ('control','candidate'))
    x,control,candidate=(arrays[k] for k in ('x','control','candidate'));rows=8191;dim=640
    assert all(math.isfinite(v) for v in x)
    columns=[x[j::dim] for j in range(dim)]
    diagonal=[math.fsum(v*v for v in col) for col in columns]
    dc=[control[j*dim+j] for j in range(dim)];dy=[candidate[j*dim+j] for j in range(dim)]
    positions=[int(i*(dim-1)/7) for i in range(8)]
    pairs={(i,j) for i in positions for j in positions}
    rng=random.Random(71303)
    off=set()
    while len(off)<256:
        i,j=rng.randrange(dim),rng.randrange(dim)
        if i!=j:off.add((i,j))
    pairs |= off
    teachers={}
    for i,j in sorted(pairs):
        assert time.monotonic()-started<60,'CPU diagnosis deadline'
        teachers[i,j]=math.fsum(a*b for a,b in zip(columns[i],columns[j]))
    def compare_pairs(selected):
        selected=sorted(selected);ref=[teachers[p] for p in selected]
        return {name:metrics(ref,[values[i*dim+j] for i,j in selected]) for name,values in [('control',control),('candidate',candidate)]}
    diag_relative=[(b-a)/a for a,b in zip(diagonal,dy)]
    result={'status':'saved_gram_cpu_diagnosis_complete_not_qualified','gpu_used':False,
            'source_sha256':hashlib.sha256(payload).hexdigest(),'code_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'shape':[1,rows,dim],'input_dtype':'float16','oracle':'Python FP64 products and math.fsum of saved exact FP16 values',
            'full_stored_candidate_vs_control':metrics(control,candidate),
            'full_diagonal_fp64':{'control':metrics(diagonal,dc),'candidate':metrics(diagonal,dy),
                                 'candidate_relative_bias_min':min(diag_relative),'candidate_relative_bias_max':max(diag_relative),
                                 'candidate_relative_bias_mean':statistics.fmean(diag_relative),
                                 'candidate_negative_diagonals':sum(v<0 for v in diag_relative)},
            'prior_8x8_positions':positions,'prior_8x8_fp64':compare_pairs({(i,j) for i in positions for j in positions}),
            'fixed_256_offdiagonal_fp64':compare_pairs(off),
            'max_stored_asymmetry':{name:max(abs(values[i*dim+j]-values[j*dim+i]) for i in range(dim) for j in range(i)) for name,values in [('control',control),('candidate',candidate)]},
            'saved_trace':{'teacher_fp64':math.fsum(diagonal),'control':math.fsum(dc),'candidate':math.fsum(dy)},
            'scope':'Full stored matrix comparison and partial FP64 oracle only; no full FP64 Gram, no new GPU/kernel/control execution, no threshold change or quality qualification',
            'seconds':time.monotonic()-started}
    a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))


if __name__=='__main__':main()
