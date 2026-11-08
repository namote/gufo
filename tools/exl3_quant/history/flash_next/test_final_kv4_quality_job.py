"""CPU-only current-runtime quality admission fixtures; no model/GPU calls."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import final_kv4_quality_job as q


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=q.OUT,prefix='current-quality-cpu-')
        self.addCleanup(self.tmp.cleanup);self.root=Path(self.tmp.name)

    def save(self,p,value):
        p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return q.sha(p)

    def test_prior_runtime_pack_and_saved_fp16_identity(self):
        pack=self.root/'pack';self.save(pack/'GUFO-COMPILE.json',{'status':'fixture'})
        fp=self.root/'fp16';fp.mkdir()
        payload=fp/'c03-prefill.bin'
        with payload.open('wb') as f:f.truncate(32760*3096)
        c={'pack':str(pack),'runtime_manifest_sha256':'current-runtime','fp16_dir':str(fp),
           'prior_qualification':str(self.root/'FINAL-QUALIFICATION.json'),
           'prior_scope':'FP16 compatibility only; K4V4 not yet qualified',
           'fp16_bin_sha256':'saved-payload-sha','fp16_bin_stat':q.stat_id(payload)}
        r={'binaries':{'gufo':{'sha256':'current-elf'},'rq_logits':{'sha256':'current-rq'}}}
        prior={'status':'qualified_for_measured_scope','pack':str(pack),'runtime_manifest_sha256':'current-runtime',
               'actual_gufo_sha256':'current-elf','source_revision':q.REVISION,
               'compile_receipt_sha256':q.sha(pack/'GUFO-COMPILE.json'),'quality':{'KV':'fp16'}}
        c['prior_qualification_sha256']=self.save(Path(c['prior_qualification']),prior)
        c['fp16_metadata_sha256']=self.save(fp/'c03-prefill.json',{'kv_cache':'fp16','records':32760,'bin_sha256':'saved-payload-sha'})
        c['fp16_collection_sha256']=self.save(fp/'COLLECTION.json',{'status':'collected_pending_cpu_scoring',
            'metadata_sha256':c['fp16_metadata_sha256'],'runtime_manifest_sha256':'current-runtime',
            'binary_sha256':'current-rq','source_revision':q.REVISION})
        q.validate_prior(c,r)
        for update in ({'actual_gufo_sha256':'old'}, {'quality':{'KV':'kvarn-k4v4'}}, {'pack':'different'}):
            c['prior_qualification_sha256']=self.save(Path(c['prior_qualification']),{**prior,**update})
            with self.subTest(update=update),self.assertRaises(AssertionError):q.validate_prior(c,r)
        c['prior_qualification_sha256']=self.save(Path(c['prior_qualification']),prior)
        with payload.open('ab') as f:f.write(b'x')
        with self.assertRaises(AssertionError):q.validate_prior(c,r)

    def test_explicit_separate_control_no_watch_or_callback(self):
        c={'control':str(self.root/'kv4/CONTROL.json')}
        value={'enabled':True,'scope':'c03-kvarn-k4v4-validation','automatic_resume':False,'callbacks':False}
        self.save(Path(c['control']),value)
        with patch.object(q,'ROOT',self.root):
            q.validate_control(c)
            for update in ({'enabled':False},{'scope':'old'},{'callbacks':True},{'automatic_resume':True}):
                self.save(Path(c['control']),{**value,**update})
                with self.subTest(update=update),self.assertRaises(AssertionError):q.validate_control(c)
        source=Path(q.__file__).read_text()
        self.assertNotIn('WATCH-CONTROL',source);self.assertNotIn('notify(',source)

    def test_compare_requires_clean_score(self):
        h=self.save(self.root/'quality/QUALITY.json',{'status':'scored_pending_independent_quality_review'})
        state={'status':'scored_pending_independent_quality_review','contract_sha256':'c','cleanup_errors':[],
               'returned_KFD_contexts':[],'pack_unchanged':True,'result_sha256':h}
        self.save(self.root/'SCORE-STATE.json',state);q.validate_score(self.root,'c')
        for update in ({'cleanup_errors':['bad']},{'result_sha256':'wrong'},{'status':'failed'}):
            self.save(self.root/'SCORE-STATE.json',{**state,**update})
            with self.subTest(update=update),self.assertRaises(AssertionError):q.validate_score(self.root,'c')

    def test_container_nix_runtime_and_cpu_score_no_gpu(self):
        c={'stage_contract':'/metadata/stage.json','reference_dir':'/reference','corpus_dir':'/corpus','fp16_dir':'/saved-fp16'}
        for phase in ('collect','score','compare'):
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
