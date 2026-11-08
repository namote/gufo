"""CPU contract test of the actual subset wrapper using stubbed model arithmetic."""
from contextlib import contextmanager
import hashlib
import inspect
import json
from pathlib import Path
from types import SimpleNamespace
import torch
from exllamav3.modules.transformer import TransformerBlock
from exllamav3.modules.block_sparse_mlp import BlockSparseMLP
import capture_fast
import capture_subset

checks=[];active=False;trace=[]
block=TransformerBlock.__new__(TransformerBlock);block.key='fixture';block.mlp=BlockSparseMLP.__new__(BlockSparseMLP)
ids=[torch.zeros(1,i+2,dtype=torch.int32) for i in range(5)]
xs=[torch.tensor([float(i)]) for i in range(5)]
def arithmetic(self,x,params,out_dtype=None):
    trace.append(('capture' if 'capture' in params else 'normal',active))
    if 'capture' in params:params['capture']['calls']=params['capture'].get('calls',0)+1
    return x+1
@contextmanager
def scope(self,params):
    global active
    assert not active;active=True
    try:yield
    finally:active=False

def fixture_main():
    bad_rows=set();state=xs;original_input_ids=ids
    H={};references=[];outputs=[]
    for i in range(len(state)):
        params=dict(capture=H,activate_all_experts=True,input_ids=original_input_ids[i],quant_preserve={})
        outputs.append(block.forward(state[i],params))
        references.append(block.forward(state[i],dict(input_ids=original_input_ids[i])))
    advanced=[block.forward(x,dict(input_ids=original_input_ids[i])) for i,x in enumerate(state)]
    return H,references,advanced,outputs

saved=(TransformerBlock.forward,capture_fast.install,capture_fast._targets,capture_fast._discarded_capture,capture_fast._MAIN_SHA)
try:
    TransformerBlock.forward=arithmetic
    capture_fast.install=lambda cm,automatic: (lambda:None)
    capture_fast._targets=lambda b,p: frozenset()
    capture_fast._discarded_capture=scope
    capture_fast._MAIN_SHA=hashlib.sha256(inspect.getsource(fixture_main).encode()).hexdigest()
    restore,complete,stats=capture_subset.install(SimpleNamespace(main=fixture_main),[1,3],5,8)
    try:
        H,refs,advance,outputs=fixture_main();complete()
        assert H['calls']==2 and stats['fixture']==dict(rows=5,selected_rows=2,selected_tokens=8,skipped_rows=3)
        assert all(torch.equal(y,x+1) for y,x in zip(refs,xs))
        assert all(torch.equal(y,x+1) for y,x in zip(advance,xs))
        assert all(outputs[i] is xs[i] for i in (0,2,4))
        assert all(enabled==(kind=='capture') for kind,enabled in trace)
        checks+=['selected_H_count_and_token_total','all_normal_refs_kept','all_state_advance_kept','skipped_input_identity','selected_explicit_fast_scope']
    finally:restore();restore()
    assert TransformerBlock.forward is arithmetic;checks.append('restore_idempotent')
    try:capture_subset.selection(2,{2},5,xs,ids,dict(input_ids=ids[1]))
    except AssertionError:pass
    else:raise AssertionError('Wrong ID mapping accepted')
    checks.append('ID_identity_mismatch_fails')
    try:capture_subset.install(SimpleNamespace(main=fixture_main),[3,1],5,8)
    except AssertionError:pass
    else:raise AssertionError('Unsorted selection accepted')
    checks.append('unsorted_manifest_fails')
finally:
    TransformerBlock.forward,capture_fast.install,capture_fast._targets,capture_fast._discarded_capture,capture_fast._MAIN_SHA=saved
root=Path('/output/capture-subset');root.mkdir(exist_ok=True)
names=('capture_subset.py','capture_subset_probe.py','capture_subset_job.py','test_capture_subset.py')
identity={str(Path('/task')/n):hashlib.sha256((Path('/task')/n).read_bytes()).hexdigest() for n in names}
(root/'CPU.json').write_text(json.dumps(dict(status='pass',gpu_used=False,checks=checks,identity=identity,
 scope='Actual wrapper with stub model arithmetic/scope; real cm.main arithmetic and guards require the separate GPU check'),indent=2));print(checks)
