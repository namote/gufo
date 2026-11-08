"""CPU-only current-runtime quality admission fixtures; no model/GPU calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import final_current_quality_job as q


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=q.OUT,prefix='current-quality-cpu-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def save(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return q.sha(p)

    def fixture(self,mode='full'):
        runtime={'binaries':{'gufo':{'sha256':'new-elf'}}}
        c={'pack':'/pack','pack_stats':{},'runtime_manifest':'/runtime/RUNTIME.json',
           'runtime_manifest_sha256':'runtime-hash','smoke_state':str(self.root/'smoke/STATE.json'),
           'smoke_contract':str(self.root/'smoke-contract.json'),'smoke_admission':mode,
           'smoke_cases_sha256':{},'main_off_review_reason':'Explicit parent main-off review',
           'main_off_requests_sha256':{}}
        contract={'pack':'/pack','pack_stats':{},'image':q.IMAGE,'elf_sha256':'new-elf',
                  'runtime_manifest':c['runtime_manifest'],'identity':{c['runtime_manifest']:'runtime-hash'}}
        digest=self.save(Path(c['smoke_contract']),contract)
        state={'status':'passed_pending_independent_review','pack_unchanged':True,'contract_sha256':digest,
               'runtime_manifest_sha256':'runtime-hash','actual_elf_sha256':'new-elf','source_revision':q.REVISION,
               'http_greedy_equal':True,'vision_and_cache_pass':True,'token_hashes':{'off':['same'],'mtp':['same']}}
        c['smoke_state_sha256']=self.save(Path(c['smoke_state']),state)
        names=('http-off','http-mtp','tokens-off','tokens-mtp') if mode=='full' else ('http-off',)
        for name in names:
            c['smoke_cases_sha256'][name]=self.save(self.root/'smoke'/name/'STATE.json',
                {'status':'passed','cleanup_errors':[],'returned_KFD_contexts':[],'command':['gufo','--speculative','off']})
        c['main_off_runtime_sha256']=self.save(self.root/'smoke/http-off/RUNTIME.json',
            {'elf_sha256':'new-elf','source_revision':q.REVISION,'no_gguf_map':True,'base_image':q.IMAGE,'nix_volume':'gufo-nix'})
        for name in ('greedy-ko','greedy-en','arithmetic'):
            c['main_off_requests_sha256'][name]=self.save(self.root/'smoke/http-off/http'/(name+'.json'),
                {'status':'passed','usage':{'completion_tokens':2,'draft_tokens':0},'finish':'stop','text':'42','reasoning':''})
        return c,state,runtime

    def test_full_smoke_current_runtime(self):
        c,_,r=self.fixture();q.validate_smoke(c,r)

    def test_old_runtime_rejected(self):
        c,s,r=self.fixture()
        for field,value in [('runtime_manifest_sha256','old'),('actual_elf_sha256','old'),('source_revision','old')]:
            c['smoke_state_sha256']=self.save(Path(c['smoke_state']),{**s,field:value})
            with self.subTest(field=field),self.assertRaises(AssertionError):q.validate_smoke(c,r)

    def test_explicit_main_only_after_later_failure(self):
        c,s,r=self.fixture('main_off_only_explicitly_reviewed')
        c['smoke_state_sha256']=self.save(Path(c['smoke_state']),{**s,'status':'failed','vision_and_cache_pass':False})
        case=self.root/'smoke/http-off/STATE.json';v=json.loads(case.read_text());v['status']='failed'
        c['smoke_cases_sha256']['http-off']=self.save(case,v)
        q.validate_smoke(c,r)
        c['smoke_admission']='full'
        with self.assertRaises(AssertionError):q.validate_smoke(c,r)

    def test_main_evidence_or_cleanup_failure_rejected(self):
        c,_,r=self.fixture('main_off_only_explicitly_reviewed')
        p=self.root/'smoke/http-off/http/arithmetic.json';original=json.loads(p.read_text())
        for update in ({'status':'failed'},{'text':'wrong'},{'usage':{'completion_tokens':2,'draft_tokens':7}}):
            c['main_off_requests_sha256']['arithmetic']=self.save(p,{**original,**update})
            with self.subTest(update=update),self.assertRaises(AssertionError):q.validate_smoke(c,r)
        c['main_off_requests_sha256']['arithmetic']=self.save(p,original)
        p=self.root/'smoke/http-off/STATE.json';case=json.loads(p.read_text());case['cleanup_errors']=['error']
        c['smoke_cases_sha256']['http-off']=self.save(p,case)
        with self.assertRaises(AssertionError):q.validate_smoke(c,r)

    def test_container_nix_runtime_and_cpu_score_no_gpu(self):
        c={'stage_contract':'/metadata/stage.json','reference_dir':'/reference','corpus_dir':'/corpus'}
        for phase in ('collect','score'):
            a=q.command(c,Path('/pack'),Path('/private'),'name','label',phase,[])
            self.assertEqual('--device' in a,phase=='collect')
            mounts=[a[i+1] for i,x in enumerate(a) if x=='--mount']
            self.assertIn('type=volume,src=gufo-nix,dst=/nix,readonly',mounts)
            self.assertIn('type=bind,src='+str(q.RUNTIME)+',dst=/runtime,readonly',mounts)
            self.assertEqual([x for x in mounts if not x.endswith(',readonly')],['type=bind,src=/private,dst=/receipt'])

    def test_runtime_binary_and_receipt_binding(self):
        runtime_dir=self.root/'runtime';runtime_dir.mkdir()
        binaries={}
        for name in ('gufo','rq_logits'):
            p=runtime_dir/name;p.write_bytes(name.encode());binaries[name]={'sha256':q.sha(p),'bytes':p.stat().st_size}
        build=self.root/'build.json';cpu=self.root/'cpu.json'
        bh=self.save(build,{'status':'passed_pending_review'});ch=self.save(cpu,{'status':'passed_pending_review'})
        c={'runtime_manifest':str(runtime_dir/'RUNTIME.json'),'build_receipt':str(build),'cpu_build_receipt':str(cpu)}
        manifest={'status':'cpu_checked_current_nix101_pending_gpu','source_revision':q.REVISION,'base_image':q.IMAGE,
                  'nix_volume':'gufo-nix','ld_library_path':q.LD,'binaries':binaries,'build_receipt_sha256':bh,'cpu_receipt_sha256':ch}
        c['runtime_manifest_sha256']=self.save(Path(c['runtime_manifest']),manifest)
        with patch.object(q,'RUNTIME',runtime_dir):
            q.validate_runtime(c)
            (runtime_dir/'gufo').write_bytes(b'old')
            with self.assertRaises(AssertionError):q.validate_runtime(c)

    def test_score_requires_successful_collection(self):
        h=self.save(self.root/'quality/COLLECTION.json',{'status':'collected_pending_cpu_scoring'})
        state={'status':'collected_pending_cpu_scoring','contract_sha256':'c','cleanup_errors':[],
               'returned_KFD_contexts':[],'pack_unchanged':True,'result_sha256':h}
        self.save(self.root/'COLLECT-STATE.json',state);q.validate_collection(self.root,'c')
        self.save(self.root/'COLLECT-STATE.json',{**state,'result_sha256':'bad'})
        with self.assertRaises(AssertionError):q.validate_collection(self.root,'c')


if __name__=='__main__':unittest.main()
