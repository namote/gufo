"""CPU cross-mount regression: source borrowing never links or copies payloads."""
import errno
import gc
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from exllamav3.loader.safetensors_alt import save_file
from row_checkpoint import link_rows, open_rows
import mtp_borrowed_rows as m
from mtp_calibration_candidate import preserve_scratch_until_commit


class BorrowedTests(unittest.TestCase):
    def setUp(self):
        self.src_tmp = tempfile.TemporaryDirectory(dir='/dev/shm')
        self.dst_tmp = tempfile.TemporaryDirectory(dir='/tmp')
        self.addCleanup(self.src_tmp.cleanup)
        self.addCleanup(self.dst_tmp.cleanup)
        self.source = Path(self.src_tmp.name)/'ckpt'
        self.source.mkdir()
        self.work = Path(self.dst_tmp.name)/'work'
        (self.work/'scratch').mkdir(parents=True)
        assert self.source.stat().st_dev != self.work.stat().st_dev
        paths=[];infos=[]
        self.values=[torch.arange(t*4*8,dtype=torch.float32).reshape(1,t,4,8) for t in (2,3,4)]
        for i,v in enumerate(self.values):
            path=Path(self.src_tmp.name)/f'input-{i}.safetensors'
            save_file({'tensor':v},str(path));paths.append(path);infos.append(('F32',tuple(v.shape)))
        link_rows(paths,self.source/'state.safetensors',infos)
        (self.source/'job.json').write_text(json.dumps({'next_module_idx':51,'q_strategy':None,'bad_rows':[]}))
        self.before={p.name:p.read_bytes() for p in (self.source/'state.rows').iterdir()}
        # Tiny fixture substitutes geometry and mount-readonly observation only;
        # actual I/O still crosses /dev/shm and /tmp mounts.
        for context in (patch.multiple(m,PROTECTED_CHECKPOINT=self.source,EXPECTED_ROWS=3,
                                      EXPECTED_TOKENS=9,EXPECTED_HIDDEN=8),
                        patch.object(m,'_readonly',return_value=True),
                        patch.dict(os.environ,{'GUFO_MTP_LEASE_VERIFIED':'1'})):
            context.start();self.addCleanup(context.stop)

    def source_intact(self):
        self.assertEqual(self.before,{p.name:p.read_bytes() for p in (self.source/'state.rows').iterdir()})

    def test_cross_mount_link_fails_borrowed_read_does_not_link(self):
        with self.assertRaises(OSError) as caught:
            os.link(self.source/'state.rows/0.safetensors',self.work/'would-link')
        self.assertEqual(caught.exception.errno,errno.EXDEV)
        with patch('os.link',side_effect=AssertionError('unexpected source hardlink')):
            store=m.BorrowedRows(self.source/'state.safetensors',self.work/'scratch')
            try:
                self.assertTrue(torch.equal(store[2],self.values[2]))
                self.assertFalse((Path(store.scratch.name)/'base').exists())
                self.assertEqual(list(Path(store.scratch.name).iterdir()),[])
            finally:store.close()
        self.source_intact()

    def test_serial_full_advance_checkpoint_links_only_private_rows(self):
        store=m.BorrowedRows(self.source/'state.safetensors',self.work/'scratch')
        try:
            with self.assertRaises(AssertionError):store.save_checkpoint(self.work/'partial.safetensors')
            for i in range(3):store[i]=store[i][:,:-1].clone()+2
            original_link=os.link;links=[]
            def link(src,dst,**kwargs):
                self.assertIn(Path(store.scratch.name),Path(src).parents)
                links.append((src,dst));return original_link(src,dst,**kwargs)
            with patch('os.link',side_effect=link):store.save_checkpoint(self.work/'done.safetensors')
            self.assertEqual(len(links),3)
            _,infos=open_rows(self.work/'done.safetensors')
            self.assertEqual([s[1] for _,s in infos],[1,2,3])
            with self.assertRaises(RuntimeError):store.save(self.work/'aggregate.safetensors')
        finally:store.close()
        self.source_intact()

    def test_async_prefetch_write_checkpoint_and_source_unchanged(self):
        store=m.BorrowedLargeRows(self.source/'state.safetensors',self.work/'scratch',
                                  read_bytes=4096,write_bytes=4096,ahead=3)
        try:
            for i in range(3):
                self.assertTrue(torch.equal(store[i],self.values[i]))
                store[i]=store[i][:,:-1].clone()+4
            store.save_checkpoint(self.work/'async.safetensors')
            self.assertEqual(store.changed,{0,1,2})
            self.assertEqual(store.stats['written_rows'],3)
            _,infos=open_rows(self.work/'async.safetensors')
            self.assertEqual([s[1] for _,s in infos],[1,2,3])
        finally:store.close()
        self.source_intact()

    def test_partial_async_output_survives_close_and_gc_when_preserved(self):
        store=m.BorrowedLargeRows(self.source/'state.safetensors',self.work/'scratch',
                                  read_bytes=4096,write_bytes=4096,ahead=3)
        cleanup=preserve_scratch_until_commit(store)
        directory=Path(store.scratch.name)
        store[0]=store[0]+3
        with self.assertRaises(AssertionError):store.save_checkpoint(self.work/'incomplete.safetensors')
        store.close();del store;gc.collect()
        self.assertTrue((directory/'0.safetensors').is_file())
        self.assertFalse((self.work/'incomplete.rows').exists())
        self.source_intact();cleanup()

    def test_guard_rejects_wrong_boundary_unleased_writable_and_mutated_source(self):
        with patch.dict(os.environ,{'GUFO_MTP_LEASE_VERIFIED':'0'}):
            with self.assertRaises(AssertionError):m.BorrowedRows(self.source/'state.safetensors',self.work/'scratch')
        with patch.object(m,'_readonly',return_value=False):
            with self.assertRaises(AssertionError):m.BorrowedRows(self.source/'state.safetensors',self.work/'scratch')
        with self.assertRaises(AssertionError):m.BorrowedRows(self.source/'other.safetensors',self.work/'scratch')
        store=m.BorrowedRows(self.source/'state.safetensors',self.work/'scratch')
        (self.source/'job.json').write_text(json.dumps({'next_module_idx':52,'q_strategy':None,'bad_rows':[]}))
        with self.assertRaises(AssertionError):store.check_source()
        with self.assertRaises(AssertionError):store.close()
        with self.assertRaises(AssertionError):m.BorrowedRows(self.source/'state.safetensors',self.work/'scratch')


if __name__=='__main__':unittest.main()
