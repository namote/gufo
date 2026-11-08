"""Reuse I/O contracts and validate the byte/row policy without allocating 10GiB."""
import hashlib,json,sys,types,unittest
from pathlib import Path
from concurrent.futures import Future
import disk_state_large

T=Path('/task');R=Path('/output/capture-io-large');R.mkdir(exist_ok=True)
source=(T/'test_disk_state_async.py').read_text().replace('from disk_state_async import DiskRows','from disk_state_large import DiskRows')
ns={'__name__':'large_policy_tests'};exec(compile(source,'test_disk_state_async.py','exec'),ns)
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(ns['Tests']))
assert result.wasSuccessful()
cal=json.loads((T/'c03-input-ids.json').read_text())
class Pool:
    def submit(self,*args):
        f=Future();f.set_result(None);return f
def policy(shapes):
    s=object.__new__(disk_state_large.DiskRows)
    s.count=len(shapes);s.infos=[('F32',(1,n,4,2560)) for n in shapes]
    s.read_limit=10*2**30;s.ahead=800;s.prefetched={};s.read_pool=Pool()
    s.stats={'peak_prefetch_bytes':0};s._read_args=lambda i:(Path('unused'),'tensor',None)
    s._prime(0)
    n=len(s.prefetched);size=sum(x[1] for x in s.prefetched.values())
    assert n<=800 and size<=10*2**30
    return {'rows':n,'bytes':size}
actual=policy([v['tokens'] for v in cal['sequences']])
short=policy([512]*1000);assert short==dict(rows=512,bytes=10*2**30)
tiny=policy([1]*1000);assert tiny['rows']==800
sig=__import__('inspect').signature(disk_state_large.DiskRows)
assert sig.parameters['read_bytes'].default==10*2**30 and sig.parameters['ahead'].default==800
r=dict(status='pass',gpu_used=False,tests=result.testsRun,policy_checks=3,actual_first_window=actual,all_short_window=short,tiny_window=tiny,identity={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [T/'disk_state_large.py',T/'test_state_large.py',T/'disk_state_async.py',T/'test_disk_state_async.py']})
(R/'CPU.json').write_text(json.dumps(r,indent=2));print(json.dumps(r))
