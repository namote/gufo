"""Narrow final-main disk admission; original housekeeping and defaults unchanged."""
import hashlib,json,types
from pathlib import Path
import disk_boundary as original

TASK=Path(__file__).resolve().parent

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def reviewed_budget(out,state_bytes,contract,task=TASK):
    assert contract['status']=='reviewed_main_only_skip_compile'
    assert contract['skip_side_models'] is True and contract['compile_separately'] is True
    assert sha(task/'model_job.py')==contract['supervisor_sha256']
    assert sha(task/'convert_main.py')==contract['converter_sha256']
    assert sha(original.__file__)==contract['disk_boundary_sha256']
    converter=(task/'convert_main.py').read_text()
    assert "os.environ['EXL3_SKIP_SIDE_MODELS']='1'" in converter
    assert "'--vision_bits','16'" in converter and "status='main_complete_side_models_skipped'" in converter
    inventory=json.loads((out/'model-inventory.json').read_text())
    main=[m for m in inventory['modules'] if m['component']=='main']
    assert len(main)==contract['main_modules']==53
    assert [(m['index'],m['key']) for m in main[-2:]]==[(51,'model.language_model.hyper_connection_mixer'),(52,'lm_head')]
    assert contract['allowed_indices']==[51,52]
    index=json.loads((out/'work/ckpt/job.json').read_text())['next_module_idx']
    assert index in contract['allowed_indices'],'Main-only override applies only to final mixer/head'
    cal=json.loads((task/'c03-input-ids.json').read_text())
    assert cal['sha256']==contract['calibration_sha256']
    result=original.budget(out,state_bytes)
    removed=result['compile_extra_bytes']
    result.update(compile_extra_bytes=0,required_bytes=result['required_bytes']-removed,
                  deferred_compile_bytes=removed,policy='reviewed_main_only_compile_runs_separately')
    return result

def isolated_prepare(budget):
    """Same original bytecode/cleanup guards; changed budget binding is local only."""
    fn=original.prepare
    namespace={**fn.__globals__,'budget':budget}
    clone=types.FunctionType(fn.__code__,namespace,fn.__name__,fn.__defaults__,fn.__closure__)
    clone.__kwdefaults__=fn.__kwdefaults__
    return clone

def prepare(out,root,**kwargs):
    contract=json.loads((TASK/'main-only-disk-contract.json').read_text())
    # Validate source and mode before any original housekeeping can remove files.
    job,paths,state_bytes=original.checkpoint(Path(out),kwargs.get('calibration_path'))
    reviewed_budget(Path(out),state_bytes,contract)
    fn=isolated_prepare(lambda p,n:reviewed_budget(p,n,contract))
    return fn(out,root,**kwargs)
