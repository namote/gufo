"""CPU guard/phase regressions; never executes Docker, collector or GPU work."""
import copy,json,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
import final_quality_job as q

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(dir=q.OUT,prefix='quality-job-cpu-');self.root=Path(self.tmp.name)
        self.addCleanup(self.tmp.cleanup)
    def write(self,p,value):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(value));return q.sha(p)
    def fixture(self):
        c={'pack':'/pack','pack_stats':{'file':[1,2,3,4]},'smoke_contract':str(self.root/'smoke-contract.json'),
           'smoke_state':str(self.root/'smoke/STATE.json'),'smoke_cases_sha256':{}}
        d=self.write(Path(c['smoke_contract']),{'pack':c['pack'],'pack_stats':c['pack_stats'],'image':q.IMAGE,'elf_sha256':q.ELF})
        state={'status':'passed_pending_independent_review','pack_unchanged':True,'http_greedy_equal':True,
               'vision_and_cache_pass':True,'token_hashes':{'off':['hash'],'mtp':['hash']},'contract_sha256':d}
        c['smoke_state_sha256']=self.write(Path(c['smoke_state']),state)
        for name in ('http-off','http-mtp','tokens-off','tokens-mtp'):
            c['smoke_cases_sha256'][name]=self.write(self.root/'smoke'/name/'STATE.json',{'status':'passed','cleanup_errors':[],'returned_KFD_contexts':[]})
        return c,state
    def test_valid_smoke(self):
        c,_=self.fixture();q.validate_smoke(c)
    def test_smoke_failed_or_changed(self):
        c,state=self.fixture()
        for update in ({'status':'failed'},{'pack_unchanged':False},{'vision_and_cache_pass':False},
                       {'http_greedy_equal':False},{'token_hashes':{'off':['a'],'mtp':['b']}}):
            d=copy.deepcopy(c);d['smoke_state_sha256']=self.write(Path(c['smoke_state']),{**state,**update})
            with self.subTest(update=update),self.assertRaises(AssertionError):q.validate_smoke(d)
        self.write(Path(c['smoke_state']),state)
        self.write(self.root/'smoke/http-off/STATE.json',{'status':'passed','cleanup_errors':['bad'],'returned_KFD_contexts':[]})
        with self.assertRaises(AssertionError):q.validate_smoke(c)
    def test_pack_and_runtime_mismatch(self):
        c,_=self.fixture()
        for key,value in [('pack','/other'),('pack_stats',{}),('smoke_cases_sha256',{})]:
            with self.subTest(key=key),self.assertRaises(AssertionError):q.validate_smoke({**c,key:value})
    def test_container_device_and_mount_boundary(self):
        c={'stage_contract':'/metadata/QUALITY-STAGE.json','reference_dir':'/ref','corpus_dir':'/corpus'}
        for phase in ('collect','score'):
            argv=q.command(c,Path('/pack'),Path('/private'),'name','label',phase,[44,109])
            self.assertEqual('--device' in argv,phase=='collect')
            self.assertEqual(argv[argv.index('--memory')+1],'100g' if phase=='collect' else '8g')
            mounts=[argv[i+1] for i,x in enumerate(argv) if x=='--mount']
            self.assertEqual([x for x in mounts if not x.endswith(',readonly')],['type=bind,src=/private,dst=/receipt'])
            self.assertIn('type=bind,src=/ref,dst=/reference,readonly',mounts)
            self.assertEqual(argv[-1],phase);self.assertIn('--network',argv)
    def test_only_successful_collection_allows_cpu_score(self):
        p=self.root/'COLLECT-STATE.json';r=self.root/'quality/COLLECTION.json'
        h=self.write(r,{'status':'collected_pending_cpu_scoring'})
        state={'status':'collected_pending_cpu_scoring','contract_sha256':'contract','cleanup_errors':[],
               'returned_KFD_contexts':[],'pack_unchanged':True,'result_sha256':h}
        self.write(p,state);q.validate_collection(self.root,'contract')
        for update in ({'status':'failed'},{'result_sha256':'changed'},{'cleanup_errors':['error']},
                       {'returned_KFD_contexts':[123]},{'pack_unchanged':False}):
            self.write(p,{**state,**update})
            with self.subTest(update=update),self.assertRaises(AssertionError):q.validate_collection(self.root,'contract')
    def test_memory_disk_stop_and_lock_guards(self):
        self.write(self.root/'WATCH-CONTROL.json',{'enabled':True})
        gpu=self.root/'gpu.lock';job=self.root/'job.lock';gpu.touch();job.touch()
        disk=type('Disk',(),{'free':101*q.GIB})()
        with patch.object(q,'OUT',self.root),patch.object(q,'LOCK',gpu),patch.object(q,'JOB',job),\
             patch.object(q,'available',return_value=5*q.GIB) as mem,\
             patch.object(q.shutil,'disk_usage',return_value=disk),patch.object(q,'held_flock',return_value=True) as held:
            self.assertEqual(q.guard(q.time.monotonic()+30),5*q.GIB)
            mem.return_value=3*q.GIB
            with self.assertRaises(AssertionError):q.guard(q.time.monotonic()+30)
            mem.return_value=10*q.GIB;disk.free=99*q.GIB
            with self.assertRaises(AssertionError):q.guard(q.time.monotonic()+30)
            disk.free=101*q.GIB;held.return_value=False
            with self.assertRaises(AssertionError):q.guard(q.time.monotonic()+30)
            held.return_value=True;self.write(self.root/'WATCH-CONTROL.json',{'enabled':False})
            with self.assertRaises(AssertionError):q.guard(q.time.monotonic()+30)

if __name__=='__main__':unittest.main()
