import json,subprocess,unittest
from types import SimpleNamespace
from unittest.mock import Mock
import final_current_build_job as j
import final_current_build as d
class Tests(unittest.TestCase):
 def response(self,live=True,label='label'):
  return SimpleNamespace(returncode=0,stderr='',stdout=json.dumps([{'Id':'ownedid','Config':{'Labels':{'gufo.quant.run':label}},'State':{'Running':live}}]))
 def absent(self):return SimpleNamespace(returncode=1,stderr='No such object: ownedid',stdout='')
 def test_stop_timeout_falls_back_kill(self):
  run=Mock(side_effect=[self.response(),subprocess.TimeoutExpired('stop',20),self.response(),self.response(False),self.absent()])
  child=Mock();r=j.cpu_cleanup('name','label',child,run)
  self.assertEqual(r,{'cleanup_errors':[]});self.assertTrue(any(x.args[0][:2]==['docker','kill'] for x in run.call_args_list));child.wait.assert_called_once()
 def test_daemon_failure_and_wrong_owner(self):
  for reply in (SimpleNamespace(returncode=1,stderr='daemon unavailable',stdout=''),self.response(label='other')):
   run=Mock(return_value=reply);r=j.cpu_cleanup('name','label',Mock(),run);self.assertTrue(r['cleanup_errors']);self.assertEqual(run.call_count,1)
 def test_autoremove_verified(self):
  run=Mock(return_value=self.absent());child=Mock();self.assertEqual(j.cpu_cleanup('name','label',child,run),{'cleanup_errors':[]})
 def test_paths_flags_phases(self):
  plan=json.loads((j.B/'BUILD-PLAN.json').read_text());self.assertEqual(plan['output'],str(j.TASK/'final-current-nix101/build'));self.assertEqual(plan['parallel'],8)
  self.assertIn('-DCMAKE_BUILD_TYPE=RelWithDebInfo',d.configure_args('',plan['source_revision']))
  for phase in ('configure','build','cpu'):
   c=j.command(plan,phase,'name','label');self.assertNotIn('--device',c)
   mounts=[c[i+1] for i,v in enumerate(c) if v=='--mount'];self.assertEqual(len([x for x in mounts if not x.endswith(',readonly')]),2)
   self.assertEqual(c[-1],phase)
if __name__=='__main__':unittest.main()
