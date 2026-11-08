"""CPU stdlib fixtures: no model payload, GPU, compiler or runtime import."""
import json,struct,tempfile,hashlib,ast
from pathlib import Path
import mtp_final_compile_candidate as candidate
checks=[]
def reject(fn):
    try:fn()
    except (AssertionError,RuntimeError):return
    raise AssertionError('Expected rejection')
def fixture(path,tensors):
    offset=0;h={};payload=b''
    for key,(dtype,shape,data) in tensors.items():
        h[key]={'dtype':dtype,'shape':shape,'data_offsets':[offset,offset+len(data)]};offset+=len(data);payload+=data
    meta=json.dumps(h).encode()
    path.write_bytes(struct.pack('<Q',len(meta))+meta+payload)
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);src=root/'source';out=root/'out';bad=root/'bad'
    fixture(src,{'model.visual.test':('BF16',[2],b'\x00\x01\x02\x03')})
    fixture(out,{'model.visual.test':('BF16',[2],b'\x00\x01\x02\x03')})
    sm=candidate.tensor_map([src]);om=candidate.tensor_map([out])
    assert candidate.verify_vision(sm,om,count=1,parameters=2,bytes_expected=4)['payload_exact']
    fixture(bad,{'model.visual.test':('F16',[2],b'\x00\x01\x02\x03')})
    reject(lambda:candidate.verify_vision(sm,candidate.tensor_map([bad]),count=1,parameters=2,bytes_expected=4))
    fixture(bad,{'model.visual.test':('BF16',[2],b'\x00\x01\x02\x04')})
    reject(lambda:candidate.verify_vision(sm,candidate.tensor_map([bad]),count=1,parameters=2,bytes_expected=4))
    reject(lambda:candidate.tensor_map([src,out]))
    checks.append('Vision BF16 key_shape_dtype_payload equality; altered dtype/payload and duplicate source reject')
    p=root/'mtp';fixture(p,{'mtp.fc_hidden.trellis':('I16',[1,1,128],bytes(256)),
        'mtp.fc_hidden.suh':('F16',[16],bytes(32)),'mtp.fc_hidden.svh':('F16',[16],bytes(32))})
    mapping=candidate.tensor_map([p]);candidate.validate_mtp(mapping,{'mtp.fc_hidden':8})
    reject(lambda:candidate.validate_mtp(mapping,{'mtp.fc_hidden':4}))
    reject(lambda:candidate.validate_mtp(mapping,{'mtp.fc_hidden':8,'mtp.fc_embedding':8}))
    checks.append('MTP exact projection coverage and actual trellis shape bit allocation')
    for doc in ({'quant_method':'exl3','vision_bits':8},{'mtp_bits':4,'tensor_storage':{'x':'keep'}}):
        result=candidate.mixed_metadata(doc,{'mtp.fc_hidden':8},{'valid_pairs':1997187})
        assert result['vision_bits']==16 and result['mtp_bits_mode']=='mixed'
        assert result['mtp_bits_legacy_default_only'] and result['mtp_bits_by_projection']['mtp.fc_hidden']==8
        assert result['mtp_calibration']['valid_pairs']==1997187
        if 'tensor_storage' in result:assert result['tensor_storage']=={'x':'keep'}
    checks.append('Mixed metadata explicitly labels legacy scalar and retains tensor storage')
import copy
recipe={'mtp.fc_hidden':8,'mtp.fc_embedding':8,'mtp.layers.0.q':6}
fc={k:recipe[k] for k in ('mtp.fc_hidden','mtp.fc_embedding')};original_state={'rows_stat_sha256':'idx51'};derived={'rows_stat_sha256':'fc8states'}
inputs={'code_sha256':{'helper':'same'},'qualification_sha256':'qualified','input_fc_bits':fc,
        'source_identity':{'trunk':'protected'},'calibration_sha256':'c03','alignment':'gufo_mtp_f32_full_hc_shifted_v1','total_tokens':1997187,'rows':2309}
context={k:inputs[k] for k in ('source_identity','calibration_sha256','alignment','total_tokens','rows')}
ic={'input_fc_pack_sha256':'FC8','input_fc_bits':fc,'derived_state_identity':derived,'H':{'input':'counts'}}
dc={'H':{'decoder':'counts'}}
def packed(state,payload,H):return {'packed_sha256':payload,'inputs_sha256':'INPUTS','strategy':recipe,'code_sha256':inputs['code_sha256'],
    'qualification_sha256':'qualified','H':H,'state_identity':state}
ip=packed(original_state,'FC8',ic['H']);dp=packed(derived,'DECODER',dc['H'])
def capture(index,state,fc_identity):return {'status':'complete_pre_quantization','format':'gufo.mtp.capture.v1','reference_indices':[0,1,2,3,4],
    'source_strategy':recipe,'binding':{'index':index,'rows':2309,'tokens':1997187,'context':context,'state_identity':state,'math_code':inputs['code_sha256'],'input_fc_identity':fc_identity}}
ih=capture(0,original_state,None);dh=capture(1,derived,{'input_fc_pack_sha256':'FC8','derived_state_identity':derived,'bits':fc})
receipts={'INPUTS.json':inputs,'INPUT-COMPLETE.json':ic,'DECODER-COMPLETE.json':dc,'mtp.PACKED.json':ip,'mtp.layers.0.PACKED.json':dp,
          'mtp.capture/MANIFEST.json':ih,'mtp.layers.0.capture/MANIFEST.json':dh}
def validate(r):candidate.validate_causal_mtp_receipts(r,{'INPUTS.json':'INPUTS'},{'mtp.safetensors':'FC8','mtp.layers.0.safetensors':'DECODER'},recipe,original_state)
validate(receipts)
for name,key,value in [('INPUT-COMPLETE.json','input_fc_pack_sha256','otherFC'),('mtp.layers.0.PACKED.json','state_identity',{'rows_stat_sha256':'otherderived'}),
                       ('mtp.layers.0.PACKED.json','inputs_sha256','otherINPUTS'),('mtp.layers.0.PACKED.json','H',{'wrong':'counts'})]:
    altered=copy.deepcopy(receipts);altered[name][key]=value;reject(lambda:validate(altered))
altered=copy.deepcopy(receipts);altered['mtp.layers.0.capture/MANIFEST.json']['binding']['input_fc_identity']['input_fc_pack_sha256']='otherFC';reject(lambda:validate(altered))
checks.append('Cross-receipt causal binding rejects same-count but different FCpack, derived state, source INPUTS, H counters and decoder cache dependency')
identity={'recipe_sha256':'actual-recipe','calibration_sha256':'actual-c03','rows':2309,'source_total_tokens':1999496,'total_tokens':1997187}
main_done={'calibration_sha256':'actual-c03','calibration_rows':2309,'calibration_tokens':1999496}
actual_cal={'sha256':'actual-c03','rows':2309,'total_tokens':1999496}
candidate.validate_calibration_identity(identity,'actual-recipe',main_done,actual_cal)
for key,value in [('recipe_sha256','other-recipe'),('calibration_sha256','other-same-size-corpus'),('source_total_tokens',1997187)]:
    bad={**identity,key:value};reject(lambda:candidate.validate_calibration_identity(bad,'actual-recipe',main_done,actual_cal))
checks.append('Actual recipe and current main/c03 corpus binding rejects internally-consistent same-size alternative calibration')
base=Path(__file__).parent
original=(base/'convert_main.py').read_text();new=(base/'convert_main_final_candidate.py').read_text()
assert original.count("'--vision_bits','8'")==1 and new.count("'--vision_bits','16'")==1
assert "os.environ['EXL3_SKIP_SIDE_MODELS']='1'" in new
assert "status='main_complete_side_models_skipped'" in new and "sync_calibration_metadata(root/'pack',cal)" not in new
# The untouched code around the final return block must remain byte-for-byte.
prefix_original=original[:original.index('else:\n    assert target==count-1\n',original.index('    cm.main(in_args,job)'))]
prefix_expected=prefix_original.replace('import os\n',"import os\nos.environ['EXL3_SKIP_SIDE_MODELS']='1'  # Final assembly runs separately from completed artifacts.\n",1).replace("'--vision_bits','8'","'--vision_bits','16'")
assert new.startswith(prefix_expected)
assert new[new.index('\nfinally:',new.index('    cm.main(in_args,job)')):]==original[original.index('\nfinally:',original.index('    cm.main(in_args,job)')):]
checks.append('Final-main clone changes only skip_side_models, vision16 and completion receipts; guards/cleanup unchanged')
tree=ast.parse((base/'mtp_final_compile_candidate.py').read_text())
assert not any(isinstance(n,ast.Call) and isinstance(n.func,ast.Attribute) and n.func.attr in ('main','quantize_linears_parallel','quantize_linears_single') for n in ast.walk(tree))
checks.append('Final assembly does not invoke quantization or converter main')
runtime=Path('/home/m2on/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/source/exllamav3/conversion/convert_model.py')
tree=ast.parse(runtime.read_text());main=next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='main')
skip=next(n for n in main.body if isinstance(n,ast.If) and 'EXL3_SKIP_SIDE_MODELS' in ast.unparse(n.test))
assert any(isinstance(n,ast.Return) for n in skip.body)
sidecalls=[n for n in ast.walk(main) if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id=='quantize_side_model']
assert sidecalls and all(n.lineno>skip.end_lineno for n in sidecalls)
checks.append('Actual pinned converter skip returns before all old uncalibrated side quant calls')
print(json.dumps({'status':'cpu_fixture_pass','checks':checks,'executed_compile':False,'GPU_used':False}))
