"""Read-only CPU audit of completed full decoder; no tensor recomputation or GPU."""
import json,hashlib,struct,sys,time,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,'/task')
from mtp_stage_disk_budget import inspect_derived
from mtp_final_compile_candidate import validate_causal_mtp_receipts,validate_calibration_identity
O=Path('/output');W=O/'mtp-calibration-work/full/work';start=time.monotonic()
def sha(p):
 h=hashlib.sha256()
 with p.open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def scan(p):
 with p.open('rb') as f:n=struct.unpack('<Q',f.read(8))[0];h=json.loads(f.read(n))
 result={}
 for k,v in h.items():
  if k=='__metadata__':continue
  typ=v['dtype'];off,end=v['data_offsets'];d={'F32':'<f4','F16':'<f2','BF16':'<u2','I64':'<i8','I32':'<i4','I16':'<i2','U16':'<u2'}[typ]
  assert 0<=off<=end<=p.stat().st_size-8-n
  a=np.memmap(p,mode='r',dtype=d,offset=8+n+off,shape=(int(np.prod(v['shape'])),));assert a.nbytes==end-off
  if typ.startswith('F') or typ=='BF16':
   for i in range(0,len(a),1<<20):
    part=a[i:i+(1<<20)]
    assert (not ((part & 0x7f80)==0x7f80).any()) if typ=='BF16' else np.isfinite(part).all(),k
  result[k]={'dtype':typ,'shape':v['shape'],'bytes':end-off}
  if a.size<=2:result[k]['values']=a.tolist()
  del a
 return result
read=lambda n:json.loads((W/n).read_text())
names=('INPUTS.json','INPUT-COMPLETE.json','DECODER-COMPLETE.json','mtp.PACKED.json','mtp.layers.0.PACKED.json','mtp.capture/MANIFEST.json','mtp.layers.0.capture/MANIFEST.json')
receipts={n:read(n) for n in names};hashes={n:sha(W/n) for n in names};dc=receipts['DECODER-COMPLETE.json'];ic=receipts['INPUT-COMPLETE.json'];dp=receipts['mtp.layers.0.PACKED.json'];m=receipts['mtp.layers.0.capture/MANIFEST.json'];inp=receipts['INPUTS.json']
recipe=json.loads((O/'MTP-MIXED-BITS-PROPOSAL.json').read_text())['tensors'];cal=json.loads(Path('/task/c03-input-ids.json').read_text())
packs=sorted((W/'qtensors').glob('*.safetensors'));packsha={p.name:sha(p) for p in packs}
assert set(packsha)=={'mtp.safetensors','mtp.layers.0.safetensors','mtp.hyper_connection_mixer.safetensors'}
assert dp['refs_sha256']==sha(W/'mtp.layers.0.references.safetensors');assert m['payload_sha256']==sha(W/'mtp.layers.0.capture/numeric.safetensors')
original=inspect_derived(O/'work/ckpt/state.safetensors');derived=inspect_derived(W/'input-checkpoint/state.safetensors')
validate_causal_mtp_receipts(receipts,hashes,packsha,recipe,original['identity'])
assert derived['identity']==ic['derived_state_identity'];assert len(derived['infos'])==2309
assert all(t=='F32' and tuple(s)==(1,cal['sequences'][i]['tokens']-1,4,2560) for i,(t,s) in enumerate(derived['infos']))
assert sum(s[1] for t,s in derived['infos'])==1997187
assert len(dc['H'])==len(m['maps'])==516 and all(v['count']==1997187 for v in dc['H'].values())
assert dc['status']=='mtp_calibrated_pending_full_validation' and dc['advanced_all_rows'] is False
H=scan(W/'mtp.layers.0.capture/numeric.safetensors');refs=scan(W/'mtp.layers.0.references.safetensors');assert set(refs)==set(map(str,range(5)))
shapes=collections.Counter()
for rec in m['maps']:
 assert rec['values']['finalized'] is False
 assert H[rec['tensors']['count_dev']['key']]['values']==[1997187]
 assert H[rec['tensors']['inf_nan']['key']]['values']==[0,0]
 shape=H[rec['tensors']['H']['key']]['shape'];assert len(shape)==2 and shape[0]==shape[1];shapes[shape[0]]+=1
P={}
for p in packs:
 part=scan(p);assert not P.keys()&part.keys();P.update(part)
trellis={k[:-8]:v for k,v in P.items() if k.endswith('.trellis')};assert set(trellis)==set(recipe)
for key,bits in recipe.items():
 assert trellis[key]['dtype'] in ('I16','U16') and trellis[key]['shape'][-1]==16*bits
 assert P[key+'.suh']['dtype']==P[key+'.svh']['dtype']=='F16'
 assert key+'.weight' not in P
terminal=W/'qtensors/mtp.hyper_connection_mixer.safetensors';pilot=O/'mtp-calibration-work/main/probe/qtensors/mtp.hyper_connection_mixer.safetensors'
assert sha(terminal)==sha(pilot)
print(json.dumps({'status':'cpu_full_decoder_artifacts_pass','seconds':time.monotonic()-start,'GPU_used':False,'rows':2309,'tokens':1997187,'H_maps':516,'H_count_all':1997187,'H_shapes':dict(shapes),'H_nonfinite':0,'inf_nan_all_zero':True,'saved_refs':5,'saved_refs_finite':True,'all_packed_float_tensors_finite':True,'actual_projection_count':len(trellis),'actual_bits':dict(collections.Counter(recipe.values())),'terminal_exact_prior_reviewed_raw_payload':True,'pack_sha256':packsha,'refs_sha256':dp['refs_sha256'],'Hpayload_sha256':m['payload_sha256'],'receipt_sha256':hashes,'derived_rows_identity':derived['identity'],'derived_stored_bytes':derived['stored_bytes'],'source_rows_identity':original['identity'],'source_or_derived_payload_read':False,'causal_receipts_pass':True,'normal_reference_errors':dc['normal_reference_errors']},indent=2))
