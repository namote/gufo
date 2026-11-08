"""CPU-only stage budget/provenance and mount mapping fixtures; no GPU/large rows."""
import copy,json,struct,tempfile
from pathlib import Path
from mtp_stage_disk_budget import required_budget,map_path,digest,inspect_derived,GIB,DECODER_HREFS_MIN

checks=[]
def reject(fn):
    try:fn()
    except (AssertionError,ValueError,FileNotFoundError):return
    raise AssertionError('Expected stage-budget refusal')
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp).resolve();data=root/'data';out=root/'out';task=root/'task'
    for p in (data,out,task):p.mkdir()
    work=data/'work';(work/'qtensors').mkdir(parents=True);(out/'work/ckpt').mkdir(parents=True)
    source=out/'work/ckpt';(source/'job.json').write_text(json.dumps({'next_module_idx':51,'q_strategy':None,'bad_rows':[]}))
    (source/'state.rows.json').write_text('{}');(source/'original_input_ids.safetensors').write_bytes(b'ids-fixture')
    (task/'recipe.json').write_text('{}');(task/'qualified.json').write_text('{}')
    cal=json.loads(Path(__file__).with_name('c03-input-ids.json').read_text());(task/'c03-input-ids.json').write_text(json.dumps(cal))
    actual_infos=[('F32',(1,s['tokens']-1,4,2560)) for s in cal['sequences']]
    identity={'manifest_sha256':'derived-manifest','rows_stat_sha256':'derived-inodes'}
    derived={'identity':identity,'infos':actual_infos,'stored_bytes':81804779520}
    bits={'mtp.fc_hidden':8,'mtp.fc_embedding':8}
    inputs={'rows':2309,'total_tokens':1997187,'source_total_tokens':1999496,'alignment':'gufo_mtp_f32_full_hc_shifted_v1',
        'input_fc_bits':bits,'code_sha256':{'candidate':'fixed'},'qualification_sha256':digest(task/'qualified.json'),
        'recipe_sha256':digest(task/'recipe.json'),'calibration_sha256':cal['sha256'],
        'source_identity':{n:digest(source/n) for n in ('job.json','state.rows.json','original_input_ids.safetensors')}}
    (work/'INPUTS.json').write_text(json.dumps(inputs));(work/'qtensors/mtp.safetensors').write_bytes(b'packed-fixture')
    H={'hidden':{'first_key':'mtp.fc_hidden','count':7988748},'embedding':{'first_key':'mtp.fc_embedding','count':1997187}}
    pack={'rows':2309,'tokens':1997187,'H':H,'inputs_sha256':digest(work/'INPUTS.json'),'strategy':bits,
          'packed_sha256':digest(work/'qtensors/mtp.safetensors'),'code_sha256':inputs['code_sha256'],'qualification_sha256':inputs['qualification_sha256']}
    done={'status':'input_calibrated','advanced_all_rows':True,'rows':2309,'tokens':1997187,'H':H,
          'input_fc_bits':bits,'input_fc_pack_sha256':pack['packed_sha256'],'derived_state_identity':identity}
    (work/'mtp.PACKED.json').write_text(json.dumps(pack));(work/'INPUT-COMPLETE.json').write_text(json.dumps(done))
    contract={'phase':'full','argv':['--stage','decoder','--source-checkpoint','/output/work/ckpt','--work','/mtp-run/work',
        '--recipe','/task/recipe.json','--qualification','/task/qualified.json','--gram-contract','/task/gram.json'],
        'full_hrefs_bytes':int(4.05*GIB),'full_margin_bytes':16*GIB,'decoder_packed_upper_bound_bytes':2*GIB,
        'decoder_input_receipts_sha256':{n:digest(work/n) for n in ('INPUTS.json','INPUT-COMPLETE.json','mtp.PACKED.json')},
        'decoder_derived_state_identity':identity}
    def budget(c=contract,d=derived):return required_budget(c,81899552737,data,out,task,inspect_rows=lambda _:d)
    result=budget();assert result['required_bytes']==100*GIB+int(4.05*GIB)+2*GIB+16*GIB
    assert result['state_allocation_bytes']==0 and result['reused_derived_bytes']==81804779520
    assert result['required_bytes']<143*GIB
    c=copy.deepcopy(contract);c['argv'][1]='input'
    assert budget(c)['required_bytes']==100*GIB+81899552737+int(4.05*GIB)+16*GIB
    assert budget({'phase':'probe'})['required_bytes']==150*GIB
    checks.append('Input budget unchanged196.33GiB; verified decoder122.05GiB reuses76.19GiB; probe150GiB unchanged')
    for field,value in [('decoder_packed_upper_bound_bytes',GIB),('full_hrefs_bytes',DECODER_HREFS_MIN-1),('full_margin_bytes',0),('decoder_derived_state_identity',{'wrong':True})]:
        c=copy.deepcopy(contract);c[field]=value;reject(lambda:budget(c))
    d=copy.deepcopy(derived);d['identity']['rows_stat_sha256']='changed';reject(lambda:budget(d=d))
    d=copy.deepcopy(derived);d['infos'][0]=('F16',d['infos'][0][1]);reject(lambda:budget(d=d))
    d=copy.deepcopy(derived);dtype,shape=d['infos'][0];d['infos'][0]=(dtype,(1,shape[1]+1,4,2560));reject(lambda:budget(d=d))
    (work/'INPUT-COMPLETE.json').rename(work/'INPUT-COMPLETE.saved');reject(budget);(work/'INPUT-COMPLETE.saved').rename(work/'INPUT-COMPLETE.json')
    (work/'qtensors/mtp.safetensors').write_bytes(b'changed-pack');reject(budget);(work/'qtensors/mtp.safetensors').write_bytes(b'packed-fixture')
    checks.append('Missing completion, changed FCpack/state identity/dtype/per-row length and undersized Hrefs/pack/margin reject')
    for path in ('/output/work','/mtp-run','/mtp-run/../outside','relative','/mtp-run//work'):
        c=copy.deepcopy(contract);c['argv'][c['argv'].index('--work')+1]=path;reject(lambda:budget(c))
    (data/'escape').symlink_to(out/'work',target_is_directory=True)
    reject(lambda:map_path('/mtp-run/escape',data,out,task,work=True))
    for extra in (['--work=/output/work'],['--stage','input'],['--stag','input']):
        c=copy.deepcopy(contract);c['argv']+=extra;reject(lambda:budget(c))
    checks.append('Read-only/outside/root/symlink/noncanonical work and duplicate/abbreviated flag mappings reject')
    # Default metadata inspector reads real tiny row files and detects replacement
    # without reading tensor payloads; full2309 admission above uses only metadata.
    p=work/'tiny/state.safetensors';p.parent.mkdir();rows=p.with_suffix('.rows');rows.mkdir();records=[]
    for i in range(2):
        shape=[1,2,4,8];size=1*2*4*8*4;h=json.dumps({'tensor':{'dtype':'F32','shape':shape,'data_offsets':[0,size]}}).encode()
        row=rows/f'{i}.safetensors';row.write_bytes(struct.pack('<Q',len(h))+h+bytes(size));records.append({'dtype':'F32','shape':shape,'bytes':row.stat().st_size})
    p.with_suffix('.rows.json').write_text(json.dumps({'format':'gufo.calibration-rows.v1','rows':records}))
    before=inspect_derived(p);replacement=rows/'new';replacement.write_bytes((rows/'0.safetensors').read_bytes());replacement.replace(rows/'0.safetensors')
    after=inspect_derived(p);assert before['identity']!=after['identity']
    from mtp_calibration_candidate import rows_identity
    assert after['identity']==rows_identity(p)
    checks.append('Real tiny row header/stat inspector matches candidateidentity and rejects same-shape inode replacement')
print(json.dumps({'status':'cpu_stage_budget_pass','checks':checks,'GPU_used':False,'payload_hashed_bytes':'small FCfixture only; no77GiBstate hashing','runner_executed':False}))
