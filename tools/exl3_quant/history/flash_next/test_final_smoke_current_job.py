"""CPU command/HTTP evidence/identity guards. No Docker or GPU launch."""
import json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import final_smoke_current_job as j
import final_validation_http as h

class Tests(unittest.TestCase):
    def test_command_isolation_and_fixed_runtime(self):
        cmd=j.command(Path('/model'),Path('/receipt'),'owned','label',j.server_args('mtp'),[44,109],18888)
        mounts=[cmd[i+1] for i,x in enumerate(cmd) if x=='--mount']
        self.assertEqual(mounts,['type=volume,src=gufo-nix,dst=/nix,readonly',f'type=bind,src={j.RUNTIME},dst=/candidate,readonly','type=bind,src=/model,dst=/models/fnx,readonly','type=bind,src=/receipt,dst=/receipt'])
        self.assertIn('127.0.0.1:18888:8080',cmd);self.assertIn(j.IMAGE,cmd)
        self.assertIn('LD_LIBRARY_PATH='+j.LD,cmd)
        self.assertEqual(cmd[cmd.index('--entrypoint')+1],'/candidate/gufo')
        self.assertNotIn('/opt/gufo-native/bin/gufo',cmd)
        self.assertNotIn('--enable-yarn',cmd);self.assertNotIn('--mmproj',cmd)
        self.assertNotIn('--draft-policy',cmd)
        self.assertNotIn('--reasoning-effort',cmd)  # Native parser rejects effort with CLI think off.
        self.assertEqual(cmd[cmd.index('--think')+1],'off')
        self.assertEqual(cmd[cmd.index('--draft-tokens')+1],'7')
        self.assertEqual(cmd[cmd.index('--kv-cache-type')+1],'fp16')

    def test_coherent_runtime_manifest_rejects_old_or_changed_binary(self):
        import hashlib
        digest=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory(prefix='smoke-current-cpu-',dir=j.OUT) as t:
            root=Path(t);runtime=root/'runtime';runtime.mkdir()
            for name in ('gufo','rq_logits'):(runtime/name).write_bytes(('fake CPU identity '+name).encode())
            bins={n:{'sha256':digest(runtime/n),'bytes':(runtime/n).stat().st_size} for n in ('gufo','rq_logits')}
            build=root/'BUILD.json';cpu=root/'CPU.json'
            build.write_text(json.dumps({'status':'passed_pending_review','source_revision':j.REVISION}))
            cpu.write_text(json.dumps({'status':'passed_pending_review','source_revision':j.REVISION,
                'binaries':{'gufo':bins['gufo'],'tests/models/qwen38_flash_next/qwen38_flash_next_rq_logits':bins['rq_logits']}}))
            manifest=runtime/'RUNTIME.json'
            manifest.write_text(json.dumps({'status':'cpu_checked_current_nix101_pending_gpu','source_revision':j.REVISION,
                'base_image':j.IMAGE,'nix_volume':'gufo-nix','ld_library_path':j.LD,'binaries':bins,
                'build_receipt_sha256':digest(build),'cpu_receipt_sha256':digest(cpu)}))
            c={'runtime_manifest':str(manifest),'source_revision':j.REVISION,'image':j.IMAGE,'base_image':j.IMAGE,
               'elf_sha256':bins['gufo']['sha256'],'build_receipt':str(build),'runtime_cpu_receipt':str(cpu),
               'identity':{str(p):digest(p) for p in (manifest,build,cpu)}}
            with patch.object(j,'RUNTIME',runtime):
                self.assertEqual(j.runtime_identity(c)['binaries'],bins)
                with self.assertRaises(AssertionError):j.runtime_identity({**c,'elf_sha256':'oldproduction'})
                with self.assertRaises(AssertionError):j.runtime_identity({**c,'source_revision':'oldsource'})
                (runtime/'rq_logits').write_bytes(b'changed')
                with self.assertRaises(AssertionError):j.runtime_identity(c)

    def test_bench_uses_actual_final_tokens_not_pp_zero(self):
        cmd=j.bench_args('off');self.assertNotIn('-p',cmd)
        self.assertEqual(cmd[cmd.index('-n')+1],'32')
        a='a'*64;b='b'*64
        self.assertEqual(j.token_hashes('Qwen3.8-Flash-Next tg depth=0 cycles=1 output_sha256='+a+' text=x'),[('0',a)])
        self.assertNotEqual(j.token_hashes('Qwen3.8-Flash-Next tg depth=0 output_sha256='+a),j.token_hashes('Qwen3.8-Flash-Next tg depth=0 output_sha256='+b))
        with self.assertRaises(AssertionError):j.token_hashes('raw draft speed123')

    def test_finite_cache_and_signature_guards(self):
        with self.assertRaises(AssertionError):h.finite({'bad':[float('nan')]})
        with self.assertRaises(AssertionError):h.cached({'prompt_tokens':9})
        self.assertEqual(h.cached({'prompt_tokens_details':{'cached_tokens':12}}),12)
        row=dict(text='42',reasoning='',finish='stop',usage={'completion_tokens':1})
        self.assertNotEqual(h.signature(row),h.signature({**row,'usage':{'completion_tokens':2}}))

    def test_streamed_request_keeps_raw_usage_and_cold_semantics(self):
        events=[{'choices':[{'delta':{'content':'42'},'finish_reason':None}]},
                {'choices':[{'delta':{},'finish_reason':'stop'}]},
                {'choices':[],'usage':{'completion_tokens':1,'prompt_tokens':9,'cached_tokens':0}}]
        lines=[('data: '+json.dumps(x)+'\n\n').encode() for x in events]+[b'data: [DONE]\n\n']
        class Response:
            def __enter__(self):return iter(lines)
            def __exit__(self,*args):pass
        with tempfile.TemporaryDirectory(prefix='smoke-cpu-',dir=j.OUT) as t:
            c=h.Client('http://127.0.0.1:1',Path(t)/'http')
            with patch.object(h.urllib.request,'urlopen',return_value=Response()) as req:
                row=c.chat('arithmetic',[{'role':'user','content':'17+25'}])
            body=json.loads(req.call_args.args[0].data)
            self.assertFalse(body['cache_prompt']);self.assertEqual(body['temperature'],0)
            self.assertFalse(body['chat_template_kwargs']['enable_thinking'])
            self.assertEqual(row['text'],'42');self.assertEqual(row['usage']['completion_tokens'],1)
            self.assertEqual(len(row['events']),4);self.assertGreaterEqual(row['ttft_seconds'],0)
            self.assertEqual(json.loads((c.out/'arithmetic.json').read_text())['status'],'passed')

    def test_missing_done_or_bad_usage_is_retained_failure(self):
        class Response:
            def __enter__(self):return iter([b'data: {"choices":[]}\n\n'])
            def __exit__(self,*args):pass
        with tempfile.TemporaryDirectory(prefix='smoke-cpu-',dir=j.OUT) as t:
            c=h.Client('http://127.0.0.1:1',Path(t)/'http')
            with patch.object(h.urllib.request,'urlopen',return_value=Response()):
                with self.assertRaises(AssertionError):c.chat('bad',[])
            self.assertEqual(json.loads((c.out/'bad.json').read_text())['status'],'failed')

    def test_saved_baseline_cannot_be_overwritten(self):
        with tempfile.TemporaryDirectory(prefix='smoke-cpu-',dir=j.OUT) as t:
            p=Path(t)/'receipt.json';h.save(p,{'first':True})
            with self.assertRaises(AssertionError):h.save(p,{'first':False})
            self.assertTrue(json.loads(p.read_text())['first'])

if __name__=='__main__':unittest.main()
