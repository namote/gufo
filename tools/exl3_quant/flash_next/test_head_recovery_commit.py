"""Small CPU fixtures for durable terminal checkpoint commit; never real mutation."""
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import head_recovery_commit as c

TMP=c.OUT/'head-recovery'


def tensor(p,shape):
    size=2
    for v in shape:size*=v
    h=json.dumps({'tensor':{'dtype':'F16','shape':shape,'data_offsets':[0,size]}}).encode()
    p.write_bytes(struct.pack('<Q',len(h))+h+bytes(size))


class Tests(unittest.TestCase):
    def fixture(self,root):
        old=root/'ckpt';old.mkdir();c.durable(old/'job.json',dict(next_module_idx=52,q_strategy=None,bad_rows=[]))
        ids=old/'original_input_ids.safetensors';ids.write_bytes(b'preserved-original-ids')
        paths=[];infos=[]
        for i,shape in enumerate(([1,2,8],[1,3,8],[1,4,2])):
            p=root/f'source-{i}.safetensors';tensor(p,list(shape));paths.append(p);infos.append(('F16',tuple(shape)))
        stage=root/'new';archive=root/'old';journal=root/'journal.json'
        return old,stage,archive,journal,paths,infos,ids
    def test_direct_hardlinks_and_repeated_stage(self):
        with tempfile.TemporaryDirectory(dir=TMP) as td:
            old,new,arc,j,paths,infos,ids=self.fixture(Path(td))
            c.stage_checkpoint(new,paths,infos,ids);c.stage_checkpoint(new,paths,infos,ids)
            rows,ri=c.open_rows(new/'state.safetensors');self.assertEqual(ri,infos)
            self.assertEqual([p.stat().st_ino for p in rows],[p.stat().st_ino for p in paths])
    def test_partial_stage_wrong_link_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as td:
            old,new,arc,j,paths,infos,ids=self.fixture(Path(td));new.mkdir();(new/'state.rows').mkdir()
            tensor(new/'state.rows/0.safetensors',[1,2,8])
            with self.assertRaises(AssertionError):c.stage_checkpoint(new,paths,infos,ids)
    def test_resume_both_rename_crashes(self):
        for at in ('old_archived','new_promoted'):
            with self.subTest(at=at),tempfile.TemporaryDirectory(dir=TMP) as td:
                old,new,arc,j,paths,infos,ids=self.fixture(Path(td));c.stage_checkpoint(new,paths,infos,ids)
                def crash(phase):
                    if phase==at:raise RuntimeError('crash fixture')
                with self.assertRaises(RuntimeError):c.commit_directories(old,new,arc,j,{'review':'fixed'},after_rename=crash)
                c.commit_directories(old,new,arc,j,{'review':'fixed'})
                self.assertEqual(json.loads((old/'job.json').read_text())['next_module_idx'],53)
                self.assertEqual(json.loads((arc/'job.json').read_text())['next_module_idx'],52)
                self.assertEqual((arc/'original_input_ids.safetensors').read_bytes(),b'preserved-original-ids')
    def test_completed_transaction_idempotent(self):
        with tempfile.TemporaryDirectory(dir=TMP) as td:
            old,new,arc,j,paths,infos,ids=self.fixture(Path(td));c.stage_checkpoint(new,paths,infos,ids)
            c.commit_directories(old,new,arc,j,{'review':'fixed'});c.commit_directories(old,new,arc,j,{'review':'fixed'})
            self.assertEqual(json.loads(j.read_text())['phase'],'checkpoint_committed')
    def test_source_metadata_binding(self):
        with tempfile.TemporaryDirectory(dir=TMP) as td:
            source=Path(td)
            names=('job.json','state.rows.json','original_input_ids.safetensors')
            for n in names:(source/n).write_bytes(n.encode())
            audit={'source_metadata_sha256':{n:c.sha(source/n) for n in names}}
            c.check_source_metadata(source,audit)
            for n in names:
                original=(source/n).read_bytes();(source/n).write_bytes(b'changed')
                with self.assertRaises(AssertionError):c.check_source_metadata(source,audit)
                (source/n).write_bytes(original)
    def test_symlink_stage_refused(self):
        with tempfile.TemporaryDirectory(dir=TMP) as td:
            p=Path(td);source=p/'source';source.write_bytes(b'data');target=p/'target';target.symlink_to(source)
            with self.assertRaises(AssertionError):c.identical_link(source,target)
    def test_actual_runner_completion_status(self):
        run={'status':'completed_pending_independent_review','cleanup_errors':[],
             'returned_KFD_contexts':[],'protected_unchanged':True,'summary_sha256':'fixed-summary'}
        c.validate_runner_state(run,'fixed-summary')
        for key,value in [('status','completed_pending_review'),('status','failed'),
                          ('summary_sha256','other-summary'),('cleanup_errors',['bad']),('returned_KFD_contexts',[123]),('protected_unchanged',False)]:
            with self.subTest(key=key,value=value),self.assertRaises(AssertionError):
                c.validate_runner_state({**run,key:value},'fixed-summary')
    def test_existing_archive_rejected(self):
        with tempfile.TemporaryDirectory(dir=TMP) as td:
            old,new,arc,j,paths,infos,ids=self.fixture(Path(td));c.stage_checkpoint(new,paths,infos,ids);arc.mkdir()
            with self.assertRaises(AssertionError):c.commit_directories(old,new,arc,j,{})
            self.assertTrue(old.exists());self.assertTrue(new.exists())

if __name__=='__main__':unittest.main()
