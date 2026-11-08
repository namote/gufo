"""No runner import/execution: execute only its completion predicate with fixture files."""
import ast,json,tempfile
from pathlib import Path
source=Path(__file__).with_name('model_job_final_candidate.py').read_text();tree=ast.parse(source)
function=next(x for x in tree.body if isinstance(x,ast.FunctionDef) and x.name=='main_completion_ready')
namespace={'json':json,'Path':Path};exec(compile(ast.Module(body=[function],type_ignores=[]),'<completion-predicate>','exec'),namespace)
ready=namespace['main_completion_ready'];checks=[]
def rejected(fn):
    try:fn()
    except (AssertionError,KeyError):return
    raise AssertionError('Expected fail-closed completion')
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);(root/'work/ckpt').mkdir(parents=True);job=root/'work/ckpt/job.json';receipt=root/'MAIN-CONVERSION.json'
    def state(idx):job.write_text(json.dumps({'next_module_idx':idx,'bad_rows':[]}))
    def done(**overrides):
        value={'status':'main_complete_side_models_skipped','skip_side_models':True,'vision_bits':16,
               'next_module_idx':53,'main_modules':53,'calibration_sha256':'c03','calibration_rows':2309,'calibration_tokens':1999496}
        value.update(overrides);receipt.write_text(json.dumps(value))
    state(51);assert not ready(root,53,'c03')
    state(52);assert not ready(root,53,'c03')
    (root/'MODEL-CONVERSION.json').write_text('{}');assert not ready(root,53,'c03')
    checks.append('Remaining mixer/head continue; stale old MODEL-CONVERSION marker does not terminate new main flow')
    state(53);rejected(lambda:ready(root,53,'c03'))
    done();rejected(lambda:ready(root,53,'c03'))
    (root/'MAIN-COMPILE-ARGS.json').write_text('{}');assert ready(root,53,'c03')
    launched=[]
    while not ready(root,53,'c03'):launched.append('must_not_launch');break
    assert not launched
    checks.append('Valid MAIN-CONVERSION exits immediately without launch; missing receipt/args never replays final module')
    for bad in ({'status':'converted'},{'skip_side_models':False},{'vision_bits':8},{'main_modules':52},{'calibration_sha256':'wrong'}):
        done(**bad);rejected(lambda:ready(root,53,'c03'))
    done();state(52);rejected(lambda:ready(root,53,'c03'))
    state(54);rejected(lambda:ready(root,53,'c03'))
    checks.append('Stale/unmatched/uncalibrated marker and checkpoint beyond end reject')
loops=[n for n in ast.walk(tree) if isinstance(n,ast.While)]
assert any('main_completion_ready' in ast.unparse(n.test) for n in loops)
assert not any('MODEL-CONVERSION.json' in ast.unparse(n.test) for n in loops)
assert "status='main_complete_pending_calibrated_mtp_vision_compile'" in source
assert 'final_converter=' in source
checks.append('Runner loop uses tested predicate, coordinated converter guard and distinct main-only completion status')
print(json.dumps({'status':'cpu_fixture_pass','checks':checks,'runner_executed':False,'GPU_used':False}))
