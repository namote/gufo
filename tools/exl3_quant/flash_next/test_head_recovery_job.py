"""CPU-only fresh-lock, isolation, owned-cleanup and checkpoint guards."""
import fcntl,json,os
from pathlib import Path
import tempfile,unittest
from types import SimpleNamespace
from unittest.mock import patch
import head_recovery_job as h
from owned_cleanup import cleanup_owned

class Tests(unittest.TestCase):
    def test_command_uses_only_private_output_and_preserves_arithmetic(self):
        prior=json.loads((h.OUT/'model-job-c03/module-060/STATE.json').read_text())
        actual=json.loads((h.OUT/'model-job-c03/module-062/STATE.json').read_text())['command']
        before=prior['command'].copy();cmd=h.build_command(prior,'unique','label',h.DATA/'test')
        mounts=[cmd[i+1] for i,v in enumerate(cmd) if v=='--mount']
        self.assertEqual([m for m in mounts if not m.endswith(',readonly')],[f'type=bind,src={h.DATA / "test"},dst=/mtp-run'])
        self.assertIn(f'type=bind,src={h.OUT},dst=/output,readonly',mounts)
        env=lambda c:dict(c[i+1].split('=',1) for i,x in enumerate(c) if x=='-e')
        old,new=env(actual),env(cmd)
        for k,v in old.items():
            if k not in ['HOME','XDG_CACHE_HOME','TRITON_CACHE_DIR','GUFO_STATE_SCRATCH']:
                self.assertEqual(new[k],v,k)
        self.assertEqual(new['EXL3_HIP_F32OUT_VIA_F16'],'0')
        self.assertEqual(new['EXL3_SKIP_SIDE_MODELS'],'1')
        self.assertNotIn('GUFO_MTP_LEASE_VERIFIED',new)
        self.assertEqual(cmd[-5:],['/task/head_recovery.py','--approval','/mtp-run/APPROVAL.json','--out','/mtp-run/result'])
        self.assertEqual(prior['command'],before)

    def test_reject_other_writable_mount(self):
        p=json.loads((h.OUT/'model-job-c03/module-060/STATE.json').read_text())
        i=p['command'].index('--workdir');p['command'][i:i]=['--mount','type=bind,src=/tmp,dst=/unsafe']
        with self.assertRaises(AssertionError):h.build_command(p,'n','l',h.DATA/'test')

    def test_actual_idx52_locks_stop_and_advancement(self):
        with tempfile.TemporaryDirectory(prefix='head-job-cpu-',dir=h.OUT) as t:
            root=Path(t);ck=root/'work/ckpt';ck.mkdir(parents=True)
            (ck/'job.json').write_text(json.dumps(h.JOB52));(root/'WATCH-CONTROL.json').write_text('{"enabled":true}')
            lock=root/'GPU.lock';job=root/'JOB.lock'
            with lock.open('a') as f,job.open('a') as g,patch.object(h,'OUT',root),patch.object(h,'LOCK',lock),patch.object(h,'JOB',job):
                for stream in (f,g):fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
                h.guard()
                (ck/'job.json').write_text(json.dumps({**h.JOB52,'next_module_idx':53}))
                with self.assertRaises(AssertionError):h.guard()
                (ck/'job.json').write_text(json.dumps(h.JOB52));(root/'WATCH-CONTROL.json').write_text('{"enabled":false}')
                with self.assertRaises(AssertionError):h.guard()
                (root/'WATCH-CONTROL.json').write_text('{"enabled":true}');fcntl.flock(g,fcntl.LOCK_UN)
                with self.assertRaises(AssertionError):h.guard()

    def test_budget_covers_five_logits_and_100GiB_reserve(self):
        self.assertEqual(h.OUTPUT_BYTES,9857310720)
        self.assertGreaterEqual(h.disk_required(),108*h.GIB+h.OUTPUT_BYTES)
        self.assertGreaterEqual(h.disk_required(),150*h.GIB)

    def test_owned_cleanup_rejects_wrong_owner(self):
        calls=[]
        def run(argv,**kwargs):
            calls.append(argv)
            return SimpleNamespace(returncode=0,stdout=json.dumps([{'Config':{'Labels':{'gufo.quant.run':'OTHER'}},'State':{'Running':True},'Id':'foreign'}]),stderr='')
        child=SimpleNamespace(wait=lambda timeout:0)
        result=cleanup_owned('our-name','ours',child,lambda:{'KFD_contexts':[]},run=run)
        self.assertTrue(result['cleanup_errors'])
        self.assertFalse(any('stop' in x or 'kill' in x for x in calls))

    def test_owned_cleanup_only_stops_labeled_container(self):
        calls=[];inspections=0
        def run(argv,**kwargs):
            nonlocal inspections
            calls.append(argv)
            if argv[1]=='inspect':
                inspections+=1
                if inspections>1:return SimpleNamespace(returncode=1,stdout='',stderr='No such object: owned')
                return SimpleNamespace(returncode=0,stdout=json.dumps([{'Config':{'Labels':{'gufo.quant.run':'ours'}},'State':{'Running':True},'Id':'owned-id'}]),stderr='')
            return SimpleNamespace(returncode=0,stdout='',stderr='')
        result=cleanup_owned('our-name','ours',SimpleNamespace(wait=lambda timeout:0),lambda:{'KFD_contexts':[]},run=run)
        self.assertFalse(result['cleanup_errors'])
        self.assertEqual([x for x in calls if x[1]=='stop'],[['docker','stop','--time','10','owned-id']])

    def test_no_checkpoint_or_main_marker_write(self):
        import ast
        tree=ast.parse(Path(h.__file__).read_text())
        calls={ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)}
        self.assertTrue({'link_rows','os.replace','os.rename','os.unlink','shutil.rmtree','cm.main','cm.compile_model'}.isdisjoint(calls))
        self.assertIn('check_completion',calls)

if __name__=='__main__':unittest.main()
