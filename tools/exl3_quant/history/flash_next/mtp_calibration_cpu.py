"""CPU-only input alignment, graph allocation and calibration audit checks."""
import json
from pathlib import Path
import types
import torch

from mtp_calibration_contract import shifted_prompt_state, expected_count, configure_mtp, attach_prompt_prepare
from mtp_calibration_candidate import checked_quantization, run_module, digest, write_json, rows_identity, candidate_identity, preserve_scratch_until_commit, swap_captured_h_cpu

checks = []


def fails(fn):
    try:
        fn()
    except (AssertionError, ValueError, RuntimeError):
        return
    raise AssertionError('Expected fail-closed guard')


from mtp_calibration_gufo import (SEMANTICS, prepare_gufo_prompt, rms_norm_f32,
    hadamard128_f32, gufo_input_rotation, packed_fc_f32, raw_fc_f32, derived_gram_shapes)
for length in (2,33,512,8192):
    original_ids=torch.arange(length).reshape(1,length)
    ids=original_ids[:,1:].contiguous()
    state=torch.arange(length*4*8,dtype=torch.float32).reshape(1,length,4,8)*.001
    before=state.clone()
    got=prepare_gufo_prompt(state,ids,4,8)
    assert torch.equal(got.view(torch.uint8),state[:,:-1].contiguous().view(torch.uint8))
    assert torch.equal(before,state) and got.dtype==torch.float32 and ids[0,0]==1
    fails(lambda:prepare_gufo_prompt(state.half(),ids,4,8))
checks.append('Gufo_F32_previous_HC_paired_next_ID_no_zero_prefix_source_immutable')
x=torch.tensor([1.,2.,100.,1000.]).repeat_interleave(8).reshape(1,32)
gamma=torch.linspace(-.2,.2,32).half();eps=1e-6
y=rms_norm_f32(x,gamma,eps)
ref=(x.double()/(x.double().square().mean(-1,keepdim=True)+eps).sqrt())*(gamma.double()+1)
assert torch.allclose(y.double(),ref,rtol=2e-7,atol=1e-7)
wrong=rms_norm_f32(x.reshape(1,4,8),torch.zeros(8),eps).flatten(-2)
assert not torch.allclose(y,wrong,atol=.1)
checks.append('Gufo_full_HC_norm_unequal_streams_vs_independent_FP64_rejects_grouped_norm')
# All basis vectors against independent Walsh parity matrix; includes all stages/signs.
walsh=torch.tensor([[(-1.)**((i&j).bit_count()) for j in range(128)] for i in range(128)])
scale=torch.tensor(0.0883883461356163,dtype=torch.float32)
assert torch.equal(hadamard128_f32(torch.eye(128)),walsh*scale)
x=torch.linspace(-1.1,1.3,384).reshape(3,128);su=torch.linspace(.5,1.5,128).half();sv=torch.linspace(.7,1.2,128).half()
w=(torch.arange(128*128).reshape(128,128)%17/100).half()
xrot=((x*su.float()).double()@walsh.double()*float(scale)).half()
assert torch.equal(gufo_input_rotation(torch.eye(128),su),((torch.eye(128)*su.float())@walsh*scale).half())
got=packed_fc_f32(x,w,su,sv,row_chunk=2)
ref=((xrot.double()@w.double())@walsh.double()*float(scale))*sv.double()
assert torch.allclose(got.double(),ref,rtol=.01,atol=.003)
assert torch.equal(raw_fc_f32(x,w,row_chunk=2),x@w.float())
checks.append('all128_FWHT_basis_rotation_order_and_synthetic_rotated_F16_weight_F32_output_oracle_not_K8_codec')
assert expected_count('mtp.fc_hidden',1997187)==7988748
assert expected_count('mtp.fc_embedding',1997187)==1997187
assert expected_count('mtp.layers.0.mlp.experts.0.down_proj',1997187)==1997187
checks.append('T_minus_1_counts_1997187_hidden7988748')
cal=json.loads(Path('/task/c03-input-ids.json').read_text());old=json.loads(Path('/task/capture-lt-contract.json').read_text())
shapes=derived_gram_shapes(cal,old)
assert len({shape[1] for shape in shapes})==151 and min(x[1] for x in shapes)==511 and max(x[1] for x in shapes)==8191
assert not any(x[1]==32 for x in shapes)
checks.append('derived_Lt_151_actual_lengths_511_to8191_excludes_operator_test_extra32')

from exllamav3 import Config, Model
from exllamav3.modules import Linear
config = Config.from_directory('/source')
model = Model.from_config(config, component='mtp')
recipe = json.loads(Path('/output/MTP-MIXED-BITS-PROPOSAL.json').read_text())['tensors']
targets = configure_mtp(model, recipe, Linear)
assert len(targets) == 1546 and model.calibration_all_experts
assert targets['mtp.fc_hidden'].qmap != targets['mtp.fc_embedding'].qmap
assert recipe['mtp.fc_hidden'] == recipe['mtp.fc_embedding'] == 8
assert recipe['mtp.layers.0.self_attn.q_proj'] == 6
assert recipe['mtp.layers.0.self_attn.k_proj'] == 8
assert recipe['mtp.layers.0.mlp.experts.0.down_proj'] == 4
# Exact H allocation by distinct qmap and first owner; aliases counted once.
footprint=[]
for index,module in enumerate(model.modules[:2]):
    owners={}
    for layer in module:
        if isinstance(layer,Linear) and layer.qmap:
            owners.setdefault(layer.qmap,layer)
    first5=sum(x['tokens']-1 for x in cal['sequences'][:5])
    footprint.append({'index':index,'H_maps':len(owners),'H_bytes':sum(x.in_features**2*4 for x in owners.values()),
        'owner_shape_counts':{str(dim):sum(x.in_features==dim for x in owners.values()) for dim in sorted({x.in_features for x in owners.values()})},
        'reference_bytes_one_copy':first5*4*2560*4})
Path('/output/mtp-calibration-candidate/FOOTPRINT.json').write_text(json.dumps({'source':'CPU actual MTP graph distinct qmap first owners and c03 first5 lengths',
    'stages':footprint,'derived_state_payload_bytes':1997187*4*2560*4,
    'H_bytes':sum(s['H_bytes'] for s in footprint),
    'reference_bytes_cache_plus_packed':sum(s['reference_bytes_one_copy']*2 for s in footprint),
    'notes':'Excludes safetensor metadata, completed packed weights, failed scratch, probe artifacts; H cache metadata adds only map counts/scalars.'},indent=2))
missing = dict(recipe); del missing['mtp.fc_hidden']
fails(lambda: configure_mtp(model, missing, Linear))
extra = dict(recipe); extra['invalid'] = 4
fails(lambda: configure_mtp(model, extra, Linear))
checks.append('actual_mtp_graph_all_1546_recipe_keys_separate_input_H_guard')

class DummyLinear:
    def __init__(self, key):
        self.key = key
        self.inner = types.SimpleNamespace(suh=torch.ones(2).half(),svh=torch.ones(2).half(),bias=None)
    def convert_exl3(self, H, args):
        args['g_scale'] = 1.
        return .001

original = DummyLinear.convert_exl3
def h(count):
    return {'H':torch.eye(2), 'count':count,'inf_nan':torch.tensor([0,0])}
with checked_quantization(DummyLinear,3,4):
    assert DummyLinear('mtp.fc_hidden').convert_exl3(h(12),{}) == .001
    assert DummyLinear('mtp.fc_embedding').convert_exl3(h(3),{}) == .001
    fails(lambda: DummyLinear('mtp.fc_hidden').convert_exl3(h(3),{}))
    fails(lambda: DummyLinear('mtp.fc_embedding').convert_exl3(h(3),{'q_fallback':True}))
    meta = h(3); meta['H'] = torch.empty(2,2,device='meta')
    fails(lambda: DummyLinear('mtp.fc_embedding').convert_exl3(meta,{}))
assert DummyLinear.convert_exl3 is original
checks.append('audit_meta_count_and_fallback_rejection_restores_method')

import tempfile
import shutil
import gc
from safetensors.torch import save_file
out = Path('/output/mtp-calibration-candidate'); out.mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(dir=out) as temporary:
    work = Path(temporary); (work/'qtensors').mkdir()
    (work/'INPUTS.json').write_text('{}')
    save_file({'dummy':torch.ones(1)}, str(work/'qtensors/mtp.safetensors'))
    save_file({'0':torch.ones(1,2)}, str(work/'mtp.references.safetensors'))
    recipe = {'mtp.fc_hidden':8};identity={'fixture':'state-A'}
    write_json(work/'mtp.PACKED.json',{
        'module':'mtp','index':0,'rows':1,'tokens':2,'strategy':recipe,
        'inputs_sha256':digest(work/'INPUTS.json'),'state_identity':identity,
        'code_sha256':candidate_identity(),'qualification_sha256':None,
        'packed_sha256':digest(work/'qtensors/mtp.safetensors'),
        'refs_sha256':digest(work/'mtp.references.safetensors'),'H':{'fixture':{'count':8}}})
    class Bar:
        def __init__(self,*a):pass
        def __enter__(self):return self
        def __exit__(self,*a):pass
        def update(self,*a):pass
    class Module:
        key='mtp'
        def load(self,*a,**kw):pass
        def unload(self):pass
        def prepare_for_device(self,x,p):return x
        def forward(self,x,p):return x+1
    cm = types.SimpleNamespace(ProgressBar=Bar,get_state_error=lambda x,y:(0.,0.,99.))
    config = types.SimpleNamespace(stc=types.SimpleNamespace(set_new_tensors=lambda x:None))
    args={'work_dir':str(work),'mtp_state_identity':identity}
    result=run_module(cm,Module(),config,[torch.zeros(1,2)],[torch.zeros(1,2,dtype=torch.int64)],args,recipe,0,2,4,advance_all=False,resume_packed=True)
    assert len(result['normal_reference_errors'])==1
    args['mtp_state_identity']={'fixture':'state-B'}
    fails(lambda: run_module(cm,Module(),config,[torch.zeros(1,2)],[torch.zeros(1,2,dtype=torch.int64)],args,recipe,0,2,4,advance_all=False,resume_packed=True))
checks.append('packed_resume_reuses_refs_without_capture_or_quant_rejects_changed_state')

from disk_state_large import DiskRows
from row_checkpoint import link_rows
with tempfile.TemporaryDirectory(dir=out) as temporary:
    root=Path(temporary);source=root/'source.safetensors';save_file({'tensor':torch.ones(1,2,4,8)},str(source))
    link_rows([source],root/'state.safetensors',[('F32',(1,2,4,8))])
    identity=rows_identity(root/'state.safetensors')
    store=DiskRows(root/'state.safetensors',root,ahead=1,read_bytes=2**20,write_bytes=2**20)
    private=Path(store.scratch.name)
    preserve_scratch_until_commit(store)
    store[0]=torch.full((1,2,4,8),2.)
    store.close();del store;gc.collect()
    assert private.exists() and (private/'0.safetensors').exists()
    assert rows_identity(root/'state.safetensors')==identity
    # Same shape but replaced input inode must change identity.
    changed=root/'replacement.safetensors';save_file({'tensor':torch.full((1,2,4,8),3.)},str(changed))
    changed.replace(root/'state.rows/0.safetensors')
    assert rows_identity(root/'state.safetensors')!=identity
checks.append('failed_async_rows_survive_close_GC_source_intact_derived_identity_detects_replacement')

from mtp_calibration_cache import save_capture, load_capture, sha as cache_sha, object_sha, validate_h
swapped={'H':torch.eye(2),'device':torch.device('cuda:0'),'H_swap_device':torch.device('cuda:0')}
swap_captured_h_cpu(swapped,torch.device('cuda:0'),True)
assert swapped['H'].device==torch.device('cpu') and swapped['H_swap_device']==torch.device('cuda:0')
fails(lambda:swap_captured_h_cpu(swapped,torch.device('cpu'),True))
with tempfile.TemporaryDirectory(dir=out) as temporary:
    path=Path(temporary)/'cache'
    h_original={'hidden':{'H':torch.tensor([[1.,.25],[.25,2.]],dtype=torch.float32),
        'count':0,'count_dev':torch.tensor(12),'first_key':'mtp.fc_hidden','finalized':False,
        'num_total':24,'inf_nan':torch.tensor([0,0]),'device':torch.device('cpu'),'H_swap_device':torch.device('cpu')}}
    refs={0:torch.tensor([[[1.,-0.,.5]]],dtype=torch.float16)}
    binding={'source':'fixed','state':'exact','math':'pinned','input_fc_identity':{'bits':{'mtp.fc_hidden':8}}}
    strategy={'mtp.fc_hidden':8}
    save_capture(path,h_original,refs,binding,strategy)
    malformed=dict(h_original['hidden']);malformed['inf_nan']=torch.tensor([1,-1])
    fails(lambda:validate_h(malformed))
    malformed=dict(h_original['hidden']);malformed['count_dev']=torch.tensor(0)
    fails(lambda:validate_h(malformed))
    malformed=dict(h_original['hidden']);malformed['H']=torch.ones(2,3)
    fails(lambda:validate_h(malformed))
    h_original['hidden']['H'].zero_();h_original['hidden']['finalized']=True
    restored,restored_refs,manifest=load_capture(path,binding,strategy)
    assert torch.equal(restored['hidden']['H'],torch.tensor([[1.,.25],[.25,2.]]))
    assert restored['hidden']['finalized'] is False and int(restored['hidden']['count_dev'])==12
    assert restored['hidden']['device']==torch.device('cpu') and restored['hidden']['num_total']==24
    assert torch.equal(restored_refs[0].view(torch.uint8),refs[0].view(torch.uint8))
    fails(lambda:load_capture(path,{**binding,'state':'changed'},strategy))
    fails(lambda:load_capture(path,{**binding,'input_fc_identity':{'bits':{'mtp.fc_hidden':4}}},strategy))
    altered={'mtp.fc_hidden':4}
    fails(lambda:load_capture(path,binding,altered))
    approval=Path(temporary)/'APPROVED.json'
    write_json(approval,{'status':'approved_mtp_capture_recipe_remap','cache_manifest_sha256':cache_sha(path/'MANIFEST.json'),
                        'source_strategy_sha256':object_sha(strategy),'target_strategy_sha256':object_sha(altered)})
    remapped,_,_=load_capture(path,binding,altered,approval)
    assert torch.equal(remapped['hidden']['H'],restored['hidden']['H'])
    # Absence of commit marker or changed payload is never silently accepted.
    marker=path/'MANIFEST.json';marker.rename(path/'MANIFEST.saved')
    try:load_capture(path,binding,strategy)
    except FileNotFoundError:pass
    else:raise AssertionError('Accepted incomplete cache')
    (path/'MANIFEST.saved').rename(marker)
    with (path/'numeric.safetensors').open('ab') as f:f.write(b'corrupt')
    fails(lambda:load_capture(path,binding,strategy))
checks.append('immutable_prequant_H_refs_exact_roundtrip_partial_payload_and_binding_guards_explicit_recipe_remap')

(out/'CPU.json').write_text(json.dumps({'status':'cpu_contract_pass','checks':checks,
    'GPU_tested':False,'production_integrated':False},indent=2))
print(json.dumps({'status':'cpu_contract_pass','checks':checks}))
