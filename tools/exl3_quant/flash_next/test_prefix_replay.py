import ast,hashlib,inspect,json,sys
from pathlib import Path
from exllamav3.conversion import convert_model as cm
from prefix_replay import replay_source,make_replay,END
old=inspect.getsource(inspect.unwrap(cm.main));new=replay_source(cm.main)
assert new[new.index(END):]==old[old.index(END):]
assert new[:new.index('        assert 2 <= idx')] == old[:old.index('        # Collect output tensors\n')]
tree=ast.parse(new);main=tree.body[0]
loop=next(n for n in main.body if isinstance(n,ast.For) and ast.unparse(n.target)=='(idx, module)')
calls=[ast.unparse(n.func) for n in ast.walk(loop) if isinstance(n,ast.Call)]
for name in ('quantize_linears_parallel','quantize_linears_single','_flush_capture_H','image_dump'):assert name not in calls
assert calls.count('_gufo_load_packed')==1 and 'module.forward' in calls
assert 'save_tensor' in calls and 'torch.isfinite' in calls
assert 'get_preserve' in calls and 'put_preserve' in calls
assert callable(make_replay(cm.main,lambda module:{}))
r=dict(status='pass',gpu_used=False,checks=['pinned_source','original_advance_and_checkpoint_identical','original_setup_identical','no_capture_quant_calls','preserves_finite_and_checkpoint_retained','clone_compiles'],source_sha256=hashlib.sha256(new.encode()).hexdigest())
Path(sys.argv[1]).write_text(json.dumps(r,indent=2));print(r)
