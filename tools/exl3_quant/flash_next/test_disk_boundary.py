"""CPU-only filesystem tests, tiny payloads; no real deletion or Docker/GPU."""
import json,struct,tempfile,unittest,os,hashlib
from pathlib import Path
import disk_boundary as d
from row_checkpoint import link_rows
R=Path('/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/disk-boundary');R.mkdir(exist_ok=True)
class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=R);self.out=Path(self.tmp.name);self.root=self.out/'model-job-c03';self.root.mkdir();self.ckpt=self.out/'work/ckpt';self.ckpt.mkdir(parents=True)
  raw=self.out/'row.safetensors';h=json.dumps({'tensor':dict(dtype='U8',shape=[1,4],data_offsets=[0,4])}).encode();raw.write_bytes(struct.pack('<Q',len(h))+h+b'abcd')
  link_rows([raw],self.ckpt/'state.safetensors',[('U8',(1,4))]);raw.unlink();self.row=self.ckpt/'state.rows/0.safetensors'
  self.cal=self.out/'cal.json';self.cal.write_text(json.dumps(dict(rows=1,sequences=[dict(tokens=4)])))
  (self.out/'work/args.json').write_text(json.dumps(dict(cal_rows=1)))
  h=json.dumps({'tensor.0':dict(dtype='I64',shape=[1,4],data_offsets=[0,32])}).encode();(self.ckpt/'original_input_ids.safetensors').write_bytes(struct.pack('<Q',len(h))+h+bytes(32))
  (self.ckpt/'job.json').write_text(json.dumps(dict(next_module_idx=1,bad_rows=[])))
  (self.out/'model-inventory.json').write_text(json.dumps(dict(modules=[dict(component='main',index=1,key='layer'),dict(component='main',index=2,key='head')],linears=[])))
  (self.out/'work/qtensors').mkdir();self.prunes=0
 def tearDown(self):self.tmp.cleanup()
 def attempt(self,status='passed',idx=0,cleanup=None):
  a=self.root/'module-001';a.mkdir();(a/'STATE.json').write_text(json.dumps(dict(status=status,cleanup_errors=[] if cleanup is None else cleanup,returned_KFD_contexts=[])))
  (a/'run.log').write_text(f'MODULE_BOUNDARY_COMPLETE {idx}\n');s=a/'row-scratch';s.mkdir();os.link(self.row,s/'0.safetensors');return a,s
 def prune(self):self.prunes+=1;return dict(returncode=0,stdout='fake',stderr='')
 def run_ready(self):return d.prepare(self.out,self.root,free=lambda:300*d.GiB,prune=self.prune,calibration_path=self.cal)
 def test_pass_cleanup_retains_current_inode(self):
  a,s=self.attempt();ino=self.row.stat().st_ino;r=self.run_ready();self.assertFalse(s.exists());self.assertEqual(self.row.stat().st_ino,ino);self.assertTrue((a/'STATE.json').exists());self.assertEqual(r['status'],'ready')
 def test_failed_unresolved_preserved(self):
  a,s=self.attempt('failed');self.run_ready();self.assertTrue(s.exists())
 def test_running_preserved(self):
  a,s=self.attempt('running');self.run_ready();self.assertTrue(s.exists())
 def test_cleanup_error_preserved(self):
  a,s=self.attempt(cleanup=['error']);self.run_ready();self.assertTrue(s.exists())
 def test_missing_marker_preserved(self):
  a,s=self.attempt();(a/'run.log').write_text('partial');self.run_ready();self.assertTrue(s.exists())
 def test_newer_attempt_blocks(self):
  a,s=self.attempt(idx=2)
  with self.assertRaises(RuntimeError):self.run_ready()
  self.assertTrue(s.exists())
 def test_newer_empty_history_does_not_block(self):
  a,s=self.attempt(idx=2);(s/'0.safetensors').unlink();s.rmdir()
  self.assertEqual(self.run_ready()['status'],'ready');self.assertTrue((a/'run.log').exists())
 def test_empty_unresolved_journal_still_blocks(self):
  a,s=self.attempt();(s/'0.safetensors').unlink();s.rmdir()
  (a/'CPU-COMMIT.json').write_text(json.dumps(dict(phase='old_archived',new_index=2)))
  with self.assertRaises(RuntimeError):self.run_ready()
 def test_dangling_cleanup_symlink_blocks(self):
  a,s=self.attempt();(s/'0.safetensors').unlink();s.rmdir();s.symlink_to(self.out/'missing',target_is_directory=True)
  with self.assertRaises(AssertionError):self.run_ready()
 def test_committed_recovery_cleanup(self):
  a,s=self.attempt('failed');(a/'CPU-COMMIT.json').write_text(json.dumps(dict(phase='committed',new_index=1)));(a/'DISK-RECOVERY.json').write_text(json.dumps(dict(status='committed',next_module_idx=1)));self.run_ready();self.assertFalse(s.exists())
 def test_pending_journal_blocks(self):
  a,s=self.attempt();(a/'CPU-COMMIT.json').write_text(json.dumps(dict(phase='old_archived',new_index=1)))
  with self.assertRaises(RuntimeError):self.run_ready()
  self.assertTrue(s.exists())
 def test_symlink_tree_rejected(self):
  a,s=self.attempt();(s/'escape').symlink_to(self.out/'work',target_is_directory=True)
  with self.assertRaises(AssertionError):self.run_ready()
  self.assertTrue(self.row.exists())
 def test_symlink_attempt_rejected(self):
  target=self.out/'external';target.mkdir();(self.root/'module-001').symlink_to(target,target_is_directory=True)
  with self.assertRaises(AssertionError):self.run_ready()
 def test_state_bytes_in_budget(self):
  r=self.run_ready();self.assertEqual(r['budget']['required_bytes'],116*d.GiB+self.row.stat().st_size);self.assertEqual(self.prunes,0)
 def test_compile_copies_budgeted(self):
  (self.ckpt/'job.json').write_text(json.dumps(dict(next_module_idx=2,bad_rows=[])));(self.out/'work/qtensors/q.safetensors').write_bytes(b'abc')
  r=self.run_ready();self.assertEqual(r['budget']['compile_extra_bytes'],20*d.GiB+3)
 def test_prune_only_when_needed(self):
  def free():return (300 if self.prunes else 110)*d.GiB
  r=d.prepare(self.out,self.root,free=free,prune=self.prune,calibration_path=self.cal);self.assertEqual(self.prunes,1);self.assertEqual(r['status'],'ready')
 def test_insufficient_prevents_start(self):
  with self.assertRaises(d.DiskSpacePending):d.prepare(self.out,self.root,free=lambda:110*d.GiB,prune=self.prune,calibration_path=self.cal)
  self.assertEqual(self.prunes,1);receipt=json.loads(next((self.root/'disk-boundaries').glob('*.json')).read_text());self.assertEqual(receipt['status'],'needs_space_before_module')
 def test_diagnostic_control_untouched(self):
  f=self.out/'baseline.pt';f.write_bytes(b'keep');self.run_ready();self.assertEqual(f.read_bytes(),b'keep')
 def test_calibration_row_count_guard(self):
  a,s=self.attempt();self.cal.write_text(json.dumps(dict(rows=2,sequences=[dict(tokens=4),dict(tokens=4)])))
  (self.out/'work/args.json').write_text(json.dumps(dict(cal_rows=2)))
  with self.assertRaises(AssertionError):self.run_ready()
  self.assertTrue(s.exists())
 def test_ids_length_guard(self):
  a,s=self.attempt();h=json.dumps({'tensor.0':dict(dtype='I64',shape=[1,3],data_offsets=[0,24])}).encode();(self.ckpt/'original_input_ids.safetensors').write_bytes(struct.pack('<Q',len(h))+h+bytes(24))
  with self.assertRaises(AssertionError):self.run_ready()
  self.assertTrue(s.exists())
 def test_symlink_checkpoint_guard(self):
  a,s=self.attempt();moved=self.out/'retained';self.ckpt.rename(moved);self.ckpt.symlink_to(moved,target_is_directory=True)
  with self.assertRaises(AssertionError):self.run_ready()
  self.assertTrue(s.exists())
 def test_partial_save_blocks_cleanup(self):
  a,s=self.attempt();(self.out/'work/ckpt_new').mkdir()
  with self.assertRaises(AssertionError):self.run_ready()
  self.assertTrue(s.exists())
 def test_truncated_current_blocks_cleanup(self):
  a,s=self.attempt();self.row.write_bytes(b'bad')
  with self.assertRaises(ValueError):self.run_ready()
  self.assertTrue(s.exists())
result=unittest.TextTestRunner(verbosity=1).run(unittest.defaultTestLoader.loadTestsFromTestCase(Tests))
if not result.wasSuccessful():raise SystemExit(1)
T=Path(__file__).parent
(R/'CPU.json').write_text(json.dumps(dict(status='pass',tests=result.testsRun,gpu_used=False,identity={str(T/n):hashlib.sha256((T/n).read_bytes()).hexdigest() for n in ['disk_boundary.py','test_disk_boundary.py']}),indent=2)+'\n')
