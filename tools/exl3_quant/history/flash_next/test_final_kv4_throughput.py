"""CPU-only format selection, evidence, control and saved-baseline scope tests."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import final_kv4_throughput_job as k
import final_kv4_http as h
import final_throughput_current_job as b

class Tests(unittest.TestCase):
 def test_matrix_changes_only_kv(self):
  for old,new in zip(b.cases(),k.cases()):
   self.assertEqual(old['name'],new['name']);a=old['argv'].copy();a[a.index('--kv-cache-type')+1]=k.KV
   self.assertEqual(a,new['argv'])
  self.assertEqual(k.POOL_BYTES,b.POOL_BYTES)
 def test_actual_bench_format_required(self):
  text='\n'.join('| M | 0 GiB | ROCm | '+n+' | 50.00 ± 0.20 |' for n in ['pp2048','pp8192'])
  with self.assertRaises(AssertionError):k.parse_bench('KV cache: fp16\n'+text,'pp','off')
  self.assertEqual(k.parse_bench('KV cache: kvarn-k4v4\n'+text,'pp','off')['status'],'passed')
 def test_pool_geometry(self):
  r=k.pool_geometry('event=kv_pool_created model=flash-next blocks=297 block_bytes=6488064 bytes=1926955008')
  self.assertEqual(r['token_capacity'],76032)
  for text in ['', 'event=kv_pool_created model=flash-next blocks=1 block_bytes=1000 bytes=999']:
   with self.assertRaises(AssertionError):k.pool_geometry(text)
 def test_actual_http_format(self):
  row={'usage':{'gufo':{'component_cache':{'kv_cache_type':'kvarn-k4v4'}}}};h.verify_kv(row)
  row['usage']['gufo']['component_cache']['kv_cache_type']='fp16'
  with self.assertRaises(AssertionError):h.verify_kv(row)
 def test_separate_manual_stop_control(self):
  with tempfile.TemporaryDirectory(dir=k.OUT,prefix='kv4-control-test-') as d:
   p=Path(d)/'CONTROL.json';p.write_text('{"enabled":false}')
   with patch.object(k,'CONTROL',p):
    with self.assertRaisesRegex(AssertionError,'Latest user stop'):k.guard(float('inf'))
  self.assertNotIn('WATCH-CONTROL.json',Path(k.__file__).read_text());self.assertNotIn('notify(',Path(k.__file__).read_text())
 def test_fp16_comparison_does_not_require_cross_kv_output_identity(self):
  old={'tg-off':{'rows':{'tg128':{'mean_tps':20}},'hashes':['fp16']}}
  new={'tg-off':{'rows':{'tg128':{'mean_tps':30}},'hashes':['different-kv4']}}
  result=k.compare_fp16(new,old);self.assertFalse(result['fp16_reexecuted']);self.assertEqual(result['rows']['tg-off']['tg128']['ratio'],1.5)
 def test_cache_check_outside_timed_suite(self):
  source=Path(h.__file__).read_text();self.assertLess(source.index('measured_suite(url'),source.index("c=Client(url"));self.assertIn('outside_timed_groups=True',source)

if __name__=='__main__':unittest.main()
