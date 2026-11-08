import unittest
import final_smoke_supplement_job as j

class Tests(unittest.TestCase):
    prefix='Qwen3.8-Flash-Next tg depth=0 cycles=1 drafted=7 accepted=5 lookup_proposed=0 lookup_accepted=0 output_sha256='
    def test_normal(self):
        r=j.emitted_record(self.prefix+'a'*64+' text=hello')
        self.assertEqual(r['hashes'],[['0','a'*64]]);self.assertEqual(r['drafts'][0]['drafted'],7)
        self.assertFalse(r['stdout_interleaved'])
    def test_actual_saved_off(self):
        p=j.s.ROOT/'smoke-003/tokens-off/run.log'
        r=j.emitted_record(p.read_text())
        self.assertEqual(r['hashes'],[['0','e2b3fb5a5d48c8973b9da26539a7edd0bec996ff7c49eb3f0db6a9db65b3efb5']])
        self.assertEqual(r['drafts'][0]['drafted'],0);self.assertTrue(r['stdout_interleaved'])
    def test_reject_missing_duplicate_or_unknown_interleaving(self):
        for text in (self.prefix+'nohash',self.prefix+'\n'+'a'*64+' text=untrusted',
                     self.prefix+'a'*64+' text=x\n'+self.prefix+'b'*64+' text=y',
                     self.prefix+'[Model Load]: 1 s\nKV cache: fp16\n|h|\n|--|\n'+'a'*64+' text=x\n'+'b'*64+' text=y'):
            with self.subTest(text=text),self.assertRaises(AssertionError):j.emitted_record(text)
    def test_missing_native_prefix_and_wrong_depth(self):
        for text in ('a'*64+' text=x',self.prefix.replace('depth=0','depth=1')+'a'*64+' text=x'):
            with self.assertRaises(AssertionError):j.emitted_record(text)

if __name__=='__main__':unittest.main()
