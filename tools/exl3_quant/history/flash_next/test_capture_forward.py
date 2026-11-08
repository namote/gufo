"""CPU contract checks for ABI metadata/error classification and actual model keys."""
import ctypes as C,hashlib,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch
from exllamav3 import Config,Model
from capture_forward_ops import Native,Unsupported
checks=[]
def tensor(shape,dtype=torch.float16,device='cuda:0',contiguous=True):
    return NS(shape=shape,ndim=len(shape),dtype=dtype,device=device,is_cuda=True,is_contiguous=lambda:contiguous)
x=tensor((33,2560));w=tensor((2,2560,640));y=tensor((2,33,640),torch.float32)
assert Native.shape(x,w,y)==(2,33,2560,640,0)
assert Native.shape(x,w,tensor((2,33,640),torch.float16))==(2,33,2560,640,1)
checks.append('shape_batch_shared_input_output_dtype')
for xx,ww,yy in [(tensor((33,2560),torch.float32),w,y),(x,tensor((2,2559,640)),y),(x,w,tensor((2,33,640),torch.bfloat16)),(tensor((33,2560),contiguous=False),w,y)]:
    try:Native.shape(xx,ww,yy)
    except AssertionError:pass
    else:raise AssertionError('Unsupported contract accepted')
checks.append('dtype_shape_stride_rejected')
class Library:
    def __init__(self,status):self.status=status
    def forward_create(self,*a):a[-2]._obj.value=self.status;return None
for status,expected in [(-2,Unsupported),(7,Unsupported),(2,RuntimeError),(3,RuntimeError),(-1,RuntimeError)]:
    n=object.__new__(Native);n.plans=[];n.lib=Library(status)
    try:n.create(x,w,y)
    except expected as e:
        if expected is RuntimeError:assert not isinstance(e,Unsupported)
    else:raise AssertionError('Create failure ignored')
checks.append('unsupported_distinct_from_errors')
n.plans=[None]*8
try:n.create(x,w,y)
except AssertionError:pass
else:raise AssertionError('Workspace limit ignored')
checks.append('workspace_plan_bound')
cfg=Config.from_directory('/source');model=Model.from_config(cfg);block=model.modules[2]
assert block.key=='model.language_model.layers.0'
assert block.mlp.ups[0].key.endswith('.mlp.experts.0.up_proj')
assert block.mlp.gates[0].key.endswith('.mlp.experts.0.gate_proj')
assert block.attn.o_proj.key.endswith('.linear_attn.out_proj')
assert block.mlp.ups[0].out_dtype==block.mlp.gates[0].out_dtype
assert (block.mlp.ups[0].out_dtype or torch.float16) in (torch.float16,torch.float32)
assert block.attn.o_proj.in_features==6144
cfg.stc.close();checks.append('actual_model_keys_and_precision')
lib=C.CDLL('/output/capture-forward/forward.so')
for symbol in ['forward_create','forward_run','forward_destroy','forward_wmma_run']:assert getattr(lib,symbol)
checks.append('exported_native_symbols')
T=Path('/task');r=dict(status='pass',gpu_used=False,checks=checks,identity={str(T/n):hashlib.sha256((T/n).read_bytes()).hexdigest() for n in ['capture_forward.hip','capture_forward_ops.py','capture_forward_probe.py','test_capture_forward.py']})
Path('/output/capture-forward/CPU.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
