"""CPU-only output-contract checks; no collector invocation."""
import copy
import unittest
import final_kv4_quality as q

class Tests(unittest.TestCase):
    def fixture(self):
        r={'windows_index_sha256':'index','windows_tokens_sha256':'tokens','bin_sha256':'reference'}
        m=dict(schema='gufo.rq-logits.v1',backend='exl3',path='prefill',kv_cache='kvarn-k4v4',records=32760,
            label='c03-k4v4',context=4096,record_bytes=3096,top_k=256,vocab_size=248320,max_prompt_batch=512,head_replay_checks=64,
            windows_index_sha256='index',windows_tokens_sha256='tokens',reference={'bin_sha256':'reference'},
            windows=[{'scored':4095} for _ in range(8)])
        return m,r
    def test_collector_uses_kvarn_and_new_label(self):
        c={'binary':'/runtime/rq_logits','model':'/models/fnx','windows':'/corpus/windows.json',
           'reference':'/reference/bf16-prefill','toolchain':'current'}
        a=q.collector_command(c,'/out/kv4')
        self.assertEqual(a.count('--kvarn'),1)
        self.assertNotIn('--kvarn-k6v6',a)
        self.assertEqual(a[a.index('--label')+1],'c03-k4v4')
        self.assertEqual(a[a.index('--batch')+1],'512')
        self.assertEqual(a[a.index('--reference')+1],'/reference/bf16-prefill')

    def test_complete_prefill(self):
        m,r=self.fixture();q.validate_meta(m,r)
    def test_wrong_or_partial_data_rejected(self):
        m,r=self.fixture()
        for k,v in [('records',4095),('label','old'),('context',2048),('path','decode'),('kv_cache','fp16'),('backend','gguf'),
                    ('head_replay_checks',0),('windows_tokens_sha256','wrong'),('reference',{'bin_sha256':'wrong'})]:
            with self.subTest(k=k),self.assertRaises(AssertionError):q.validate_meta({**m,k:v},r)
        bad=copy.deepcopy(m);bad['windows'][0]['scored']=8
        with self.assertRaises(AssertionError):q.validate_meta(bad,r)
    def test_runtime_paths(self):
        maps='0 /nix/store/test-clr-10.1/lib/libamdhip64.so\n0 /nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib/libhsa-runtime64.so\n'
        self.assertEqual(len(q.runtime_paths(maps)),2)
        for suffix in ['0 /opt/rocm/lib/libamdhip64.so','0 /models/old.gguf']:
            with self.assertRaises(AssertionError):q.runtime_paths(maps+suffix)
        with self.assertRaises(AssertionError):q.runtime_paths('')
    def test_nonfinite_metrics_rejected(self):
        q.finite_json({'reports':[{'metric':.001}]})
        for x in [float('inf'),float('nan'),float('-inf')]:
            with self.assertRaises(AssertionError):q.finite_json({'reports':[{'metric':x}]})

if __name__=='__main__':unittest.main()
