"""CPU-only tiny disk budget fixtures; no cleanup, Docker or GPU work."""
import json,tempfile,hashlib
from pathlib import Path
import main_only_disk_boundary as candidate
checks=[]
def reject(fn):
    try:fn()
    except AssertionError:return
    raise AssertionError('Expected explicit main-only guard')
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);task=root/'task';out=root/'out';task.mkdir();(out/'work/ckpt').mkdir(parents=True);(out/'work/qtensors').mkdir()
    (task/'model_job.py').write_text('reviewed-supervisor')
    (task/'convert_main.py').write_text("os.environ['EXL3_SKIP_SIDE_MODELS']='1'\n'--vision_bits','16'\nstatus='main_complete_side_models_skipped'")
    (task/'c03-input-ids.json').write_text(json.dumps({'sha256':'c03'}))
    main=[{'component':'main','index':i,'key':str(i)} for i in range(51)]+[{'component':'main','index':51,'key':'model.language_model.hyper_connection_mixer'},{'component':'main','index':52,'key':'lm_head'}]
    (out/'model-inventory.json').write_text(json.dumps({'modules':main}));(out/'work/qtensors/a.safetensors').write_bytes(bytes(100))
    contract={'status':'reviewed_main_only_skip_compile','skip_side_models':True,'compile_separately':True,'main_modules':53,'allowed_indices':[51,52],
              'calibration_sha256':'c03','supervisor_sha256':candidate.sha(task/'model_job.py'),'converter_sha256':candidate.sha(task/'convert_main.py'),
              'disk_boundary_sha256':candidate.sha(candidate.original.__file__)}
    def state(i):(out/'work/ckpt/job.json').write_text(json.dumps({'next_module_idx':i}))
    state(51);old=candidate.original.budget(out,81900000000);new=candidate.reviewed_budget(out,81900000000,contract,task)
    assert old['required_bytes']==new['required_bytes'] and new['compile_extra_bytes']==0
    checks.append('Mixer100GiB+state+16GiB admission unchanged')
    state(52);old=candidate.original.budget(out,20475000000);new=candidate.reviewed_budget(out,20475000000,contract,task)
    assert old['compile_extra_bytes']==20*2**30+100 and new['required_bytes']==116*2**30+20475000000
    assert new['runtime_reserve_bytes']==old['runtime_reserve_bytes']==100*2**30
    assert new['packed_margin_bytes']==old['packed_margin_bytes']==16*2**30
    assert candidate.original.budget(out,20475000000)==old
    checks.append('Head defers only packed+20GiB compile allowance; original mode and100GiB floor/16GiB margin unchanged')
    for k,v in [('status','unreviewed'),('skip_side_models',False),('compile_separately',False),('converter_sha256','wrong'),('supervisor_sha256','wrong'),('disk_boundary_sha256','wrong'),('allowed_indices',[50,51,52]),('calibration_sha256','other')]:
        reject(lambda:candidate.reviewed_budget(out,1,{**contract,k:v},task))
    state(50);reject(lambda:candidate.reviewed_budget(out,1,contract,task))
    checks.append('Unreviewed/mismatched caller/source/calibration contracts and nonfinal indices reject')
    original=candidate.original.prepare;budget=candidate.original.budget;marker=object();clone=candidate.isolated_prepare(lambda *args:marker)
    assert clone.__code__ is original.__code__ and clone.__kwdefaults__ is original.__kwdefaults__
    assert clone.__globals__ is not original.__globals__ and clone.__globals__['budget'](None,None) is marker
    assert original.__globals__['budget'] is budget and candidate.original.prepare is original
    assert all(clone.__globals__[k] is v for k,v in original.__globals__.items() if k!='budget')
    checks.append('Isolated globals reuse exact original housekeeping bytecode; no global monkeypatch/default weakening')
print(json.dumps({'status':'cpu_main_only_disk_pass','checks':checks,'GPU_used':False,'cleanup_executed':False}))
