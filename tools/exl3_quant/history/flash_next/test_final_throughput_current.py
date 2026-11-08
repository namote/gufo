"""CPU-only measurement, provenance and result contract regressions."""
import copy,json,tempfile,unittest
from pathlib import Path
import final_throughput_current_job as j
import final_throughput_http as h

class Tests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory(dir=j.ROOT,prefix='throughput-cpu-');self.root=Path(self.tmp.name);self.addCleanup(self.tmp.cleanup)
 def test_case_matrix_and_explicit_cli(self):
  cases=j.cases();self.assertEqual(len(cases),8);self.assertEqual([c['mode'] for c in cases],['off']*4+['mtp']*4)
  for c in cases:
   a=c['argv'];self.assertNotIn('--draft-policy',a)
   if c['mode']=='mtp':self.assertEqual(a[a.index('--draft-tokens')+1],'7')
   if c['kind']=='pp':self.assertIn('2048,8192',a);self.assertNotIn('-n',a)
   elif c['kind']=='tg':self.assertIn('0,32768',a);self.assertNotIn('-p',a)
   else:
    self.assertEqual(a[a.index('--context')+1],'32768');self.assertEqual(a[a.index('--kv-pool-bytes')+1],str(j.POOL_BYTES))
 def table(self,kind):
  labels=['pp2048','pp8192'] if kind=='pp' else ['tg128','tg128 @ d32768']
  return '\n'.join('| Qwen3.8 Flash Next | 0.00 GiB | ROCm (HIP) | '+x+' | 50.00 ± 1.00 |' for x in labels)
 def trace(self,drafted=0):
  return '\n'.join(f'Qwen3.8-Flash-Next tg depth={depth} cycles=4 drafted={drafted} accepted=0 lookup_proposed=0 lookup_accepted=0 output_sha256='+('a'*64)+' text=x' for depth in [0,0,32768,32768])
 def test_separate_stdout_stderr(self):
  r=j.parse_bench(self.table('tg'),'tg','off',self.trace());self.assertEqual(len(r['hashes']),4)
  self.assertEqual(set(j.parse_bench(self.table('pp'),'pp','off')['rows']),{'pp2048','pp8192'})
 def test_partial_wrong_mode_nonfinite_bench_rejected(self):
  for table,kind,mode,trace in [(self.table('tg').splitlines()[0],'tg','off',self.trace()),
     (self.table('tg'),'tg','mtp',self.trace()),(self.table('tg'),'tg','off',self.trace(10)),
     (self.table('pp').replace('50.00','nan'),'pp','off','')]:
   with self.assertRaises(AssertionError):j.parse_bench(table,kind,mode,trace)
 def row(self):
  return dict(text='answer',reasoning='',finish='stop',ttft_seconds=.1,request_wall_seconds=2,
   usage=dict(completion_tokens=100,draft_tokens=90,draft_tokens_accepted=50,
    gufo=dict(decode_ms=1000,prefill_ms=100,queue_ms=0,requested_logical_concurrency=2,physical_execution_width=2)))
 def test_whole_decode_rate_not_acceptance(self):
  row=self.row();r=h.timing(row);self.assertEqual(r['full_decode_tokens_per_second'],100);self.assertEqual(r['emitted_tokens_per_request_wall'],50)
  row['usage']['draft_tokens_accepted']=1;self.assertEqual(h.timing(row)['full_decode_tokens_per_second'],100)
 def test_bad_http_timing_rejected(self):
  for change in [('completion_tokens',0),('completion_tokens',513),('draft_tokens_accepted',91)]:
   row=self.row();row['usage'][change[0]]=change[1]
   with self.assertRaises(AssertionError):h.timing(row)
  row=self.row();row['usage']['gufo']['decode_ms']=float('nan')
  with self.assertRaises(AssertionError):h.timing(row)
 def contract(self):
  return dict(runtime_manifest='runtime',identity={'runtime':'r',**{str(j.TASK/n):n for n in ('final_throughput_current_job.py','final_throughput_http.py','final_smoke_current_job.py','final_validation_http.py')}},elf_sha256='elf',pack_stats={'pack':[1,2,3,4]},compile_receipt_sha256='compile')
 def test_binding_rejects_changed_runtime_settings_and_pack(self):
  c=self.contract();case=j.cases()[0];original=j.binding(c,case)
  for modified in ({**c,'elf_sha256':'new'},{**c,'pack_stats':{}},{**c,'compile_receipt_sha256':'other'}):self.assertNotEqual(j.binding(modified,case),original)
  changed=copy.deepcopy(case);changed['argv']+=['--unknown'];self.assertNotEqual(j.binding(c,changed),original)
 def test_saved_baseline_requires_exact_completed_evidence(self):
  result=self.root/'RESULT.json';result.write_text('{"status":"passed"}')
  data=self.root/'run.log';data.write_text('original evidence')
  state=self.root/'STATE.json';state.write_text(json.dumps(dict(status='passed',measurement_binding='binding',cleanup_errors=[],returned_KFD_contexts=[],result_sha256=j.sha(result),evidence_sha256={'run.log':j.sha(data)})))
  spec=dict(path=str(self.root),state_sha256=j.sha(state),result_sha256=j.sha(result))
  self.assertEqual(j.saved_case(spec,'binding')[0]['status'],'passed')
  with self.assertRaises(AssertionError):j.saved_case(spec,'different')
  data.write_text('changed')
  with self.assertRaises(AssertionError):j.saved_case(spec,'binding')

 def test_supplement_binds_original_cases_and_independent_review(self):
  old=self.root/'original';old.mkdir();new=self.root/'supplement';new.mkdir()
  def put(path,value):path.write_text(json.dumps(value));return str(path)
  original=Path(put(old/'CONTRACT.json',dict(out=str(old),pack='pack',pack_stats={'one':1})))
  oldstate=Path(put(old/'STATE.json',dict(status='failed',error="AssertionError('No actual emitted-token hash in coherent candidate benchmark')",pack_unchanged=True,contract_sha256=j.sha(original),actual_elf_sha256='elf')))
  supplement=Path(put(new/'CONTRACT.json',dict(original_contract=str(original),original_contract_sha256=j.sha(original))))
  cases={}
  for name in ('http-off','http-mtp','tokens-off','tokens-mtp'):
   d=(new if name=='tokens-mtp' else old)/name;d.mkdir()
   put(d/'STATE.json',dict(status='passed',cleanup_errors=[],returned_KFD_contexts=[]));(d/'run.log').write_text('evidence')
   put(d/'RUNTIME.json',dict(elf_sha256='elf',no_gguf_map=True))
   cases[name]=dict(path=str(d),state_sha256=j.sha(d/'STATE.json'),log_sha256=j.sha(d/'run.log'),runtime_sha256=j.sha(d/'RUNTIME.json'))
  state=Path(put(new/'STATE.json',dict(status='passed_pending_independent_review',pack_unchanged=True,http_greedy_equal=True,vision_and_cache_pass=True,
   original_smoke_state_preserved=True,token_hashes=dict(off=[['0','digest']],mtp=[['0','digest']]),actual_elf_sha256='elf',runtime_manifest_sha256='runtime',
   contract_sha256=j.sha(supplement),original_contract_sha256=j.sha(original),original_state_sha256=j.sha(oldstate),cases=cases)))
  review=Path(put(new/'REVIEW.json',dict(status='passed',smoke_state_sha256=j.sha(state),original_state_sha256=j.sha(oldstate))))
  c=dict(smoke_state=str(state),smoke_state_sha256=j.sha(state),smoke_contract=str(supplement),smoke_result_review=str(review),
   elf_sha256='elf',runtime_manifest='manifest',identity={'manifest':'runtime'},pack='pack',pack_stats={'one':1})
  j.validate_smoke(c)
  put(review,dict(status='passed',smoke_state_sha256='other',original_state_sha256=j.sha(oldstate)))
  with self.assertRaises(AssertionError):j.validate_smoke(c)
  put(review,dict(status='passed',smoke_state_sha256=j.sha(state),original_state_sha256=j.sha(oldstate)))
  (old/'tokens-off/run.log').write_text('changed evidence')
  with self.assertRaises(AssertionError):j.validate_smoke(c)

if __name__=='__main__':unittest.main()
