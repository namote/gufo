import ast,copy,hashlib,json,sys,time
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from capture_owner import install
source=Path(sys.argv[1]);tree=ast.parse(source.read_text());cls=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Linear');methods=[n for n in cls.body if getattr(n,'name',None) in ('init_H_data','capture_H')]
flush=next(n for n in tree.body if getattr(n,'name',None)=='flush_capture_H')
ns=dict(torch=torch,np=np,ext=NS(count_inf_nan=lambda x,out:None),_CAP_PROFILE=False,_CAP_LEGACY=False,_time=time)
exec(compile(ast.fix_missing_locations(ast.Module(body=[ast.ClassDef(name='FakeLinear',bases=[],keywords=[],body=methods,decorator_list=[]),flush],type_ignores=[])),str(source),'exec'),ns)
L=ns['FakeLinear']
def linear(key):
 x=L();x.key=key;x.qmap='input';x.in_features=4;x.device='cpu';x.cap_rkey=None;return x
class Attention:
 def __init__(self):self.q_proj=linear('q');self.qsa_indexer=NS(index_qk_proj=linear('index'))
 def decode_flash_attn_nc(self,x,b,n,p):
  if 'capture' in p:
   if n>4:self.qsa_indexer.index_qk_proj.capture_H(x,p)
   self.q_proj.capture_H(x,p)
  return x*2
checks=[];rows={7:torch.arange(28,dtype=torch.float32).view(1,7,4),3:torch.arange(12,dtype=torch.float32).view(1,3,4)}
def run(order,fix):
 a=Attention();H={};restore=install(Attention) if fix else None
 try:
  for n in order:
   p={'capture':H};y=a.decode_flash_attn_nc(rows[n],1,n,p);assert torch.equal(y,rows[n]*2);ns['flush_capture_H'](p)
  return H['input']
 finally:
  if restore:restore()
old=run((7,3),False);fixed=run((7,3),True);reverse=run((3,7),True)
assert int(old['count_dev'])==7 and int(fixed['count_dev'])==int(reverse['count_dev'])==10
xx=torch.cat([rows[n].reshape(-1,4) for n in (7,3)]);assert torch.equal(fixed['H'],xx.T@xx) and torch.equal(fixed['H'],reverse['H'])
checks+=['reproduces_conditional_owner_omission','all_rows_both_orders','independent_gram','outputs_unchanged']
a=Attention();restore=install(Attention)
try:
 assert torch.equal(a.decode_flash_attn_nc(rows[3],1,3,{}),rows[3]*2)
 for fault in ('qmap','history'):
  p={'capture':{}};a=Attention()
  if fault=='qmap':a.q_proj.qmap='other'
  else:p['capture']['input']={'first_key':'index'}
  try:a.decode_flash_attn_nc(rows[3],1,3,p)
  except AssertionError:pass
  else:raise AssertionError(fault)
finally:restore()
checks+=['normal_forward_unchanged','reject_mismatched_qmap','reject_mixed_history']
Path(sys.argv[2]).write_text(json.dumps(dict(status='pass',gpu_used=False,checks=checks),indent=2));print(checks)
