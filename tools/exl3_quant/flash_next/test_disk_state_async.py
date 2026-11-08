"""CPU real-serializer contracts: exact rows, ownership, overlap and error drain."""
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch
import torch
from disk_state import DiskRows as SerialRows, _read_row
from disk_state_async import DiskRows
from exllamav3.loader.safetensors_alt import safe_open, save_file
from row_checkpoint import open_rows


class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(dir='/output/capture-io')
        self.root = Path(self.tmp.name)
        self.source = self.root/'state.safetensors'
        self.values = [torch.arange((i+1)*32, dtype=torch.float32).reshape(1,i+1,32) for i in range(10)]
        save_file({f'tensor.{i}':x for i,x in enumerate(self.values)}, str(self.source))
        self.stores = []
    def tearDown(self):
        for s in self.stores:
            try:s.close()
            except OSError:pass  # Explicit injected worker-failure cases below.
        self.tmp.cleanup()
    def store(self, cls=DiskRows, source=None, **kw):
        s=cls(source or self.source, self.root, **kw);self.stores.append(s);return s
    def test_exact_sequential_random_repeated_and_checkpoint(self):
        a=self.store(SerialRows);b=self.store()
        for i in [0,1,2,3,7,2,9,0,4,5,6,8]:
            torch.testing.assert_close(a[i],b[i],rtol=0,atol=0)
            a[i]=a[i]*1.25+3;b[i]=b[i]*1.25+3
            torch.testing.assert_close(a[i],b[i],rtol=0,atol=0)
        for s,name in [(a,'serial'),(b,'async')]:s.save_checkpoint(self.root/f'{name}.safetensors')
        pa,ia=open_rows(self.root/'serial.safetensors');pb,ib=open_rows(self.root/'async.safetensors')
        self.assertEqual(ia,ib)
        for p,q in zip(pa,pb):self.assertEqual(p.read_bytes(),q.read_bytes())
        c=self.store(source=self.root/'async.safetensors')
        for i in range(10):torch.testing.assert_close(c[i],a[i],rtol=0,atol=0)
        self.assertGreater(b.stats['prefetched_rows'],0)
    def test_snapshot_and_same_row_write_order(self):
        b=self.store();x=self.values[0].clone();b[0]=x;x.add_(999)
        b[1];torch.testing.assert_close(b[0],self.values[0],rtol=0,atol=0)
        for n in range(7):b[0]=self.values[0]+n
        b[1];torch.testing.assert_close(b[0],self.values[0]+6,rtol=0,atol=0)
    def test_prefetch_invalidated_on_assignment(self):
        b=self.store();b[0];self.assertIn(2,b.prefetched)
        b[2]=torch.full_like(self.values[2],42);b[4]
        self.assertTrue(torch.equal(b[2],torch.full_like(self.values[2],42)))
    def test_aggregate_rotation_and_unmodified_rows(self):
        b=self.store();self.source.rename(self.root/'old.safetensors')
        b.save_checkpoint(self.root/'new.safetensors')
        c=self.store(source=self.root/'new.safetensors')
        for i in range(10):torch.testing.assert_close(c[i],self.values[i],rtol=0,atol=0)
    def test_write_overlap_and_checkpoint_barrier(self):
        b=self.store();entered=threading.Event();release=threading.Event();done=threading.Event();errors=[]
        original=b._write
        def delayed(path,value):
            entered.set()
            if not release.wait(5):raise RuntimeError('test write release timeout')
            return original(path,value)
        with patch.object(b,'_write',delayed):
            b[0]=self.values[0]+5;self.assertTrue(entered.wait(2))
            # Caller can read/compute next input while writer is blocked.
            torch.testing.assert_close(b[1],self.values[1],rtol=0,atol=0)
            def checkpoint():
                try:b.save_checkpoint(self.root/'barrier.safetensors')
                except BaseException as e:errors.append(e)
                finally:done.set()
            t=threading.Thread(target=checkpoint);t.start()
            try:self.assertFalse(done.wait(.05))
            finally:release.set();t.join(5)
            self.assertTrue(done.is_set());self.assertFalse(errors)
        self.assertTrue((self.root/'barrier.rows.json').exists())
    def test_queue_memory_bounds(self):
        b=self.store(read_bytes=2048,write_bytes=2048,ahead=3)
        for i in range(10):b[i]=b[i]+1
        b.save_checkpoint(self.root/'bounded.safetensors')
        self.assertLessEqual(b.stats['peak_prefetch_bytes'],2048)
        self.assertLessEqual(b.stats['peak_write_bytes'],2048)
    def test_write_error_prevents_checkpoint(self):
        b=self.store()
        with patch.object(b,'_write',side_effect=OSError('injected write failure')):
            b[0]=self.values[0]
            with self.assertRaises(OSError):b.save_checkpoint(self.root/'bad.safetensors')
            self.assertFalse((self.root/'bad.rows.json').exists())
            with self.assertRaises(OSError):b.close()
    def test_read_error_propagates(self):
        b=self.store();original=b._read
        def failed(path,key,dependency):
            if key=='tensor.1':raise OSError('injected read failure')
            return original(path,key,dependency)
        with patch.object(b,'_read',failed):
            b[0]
            with self.assertRaises(OSError):b[1]

    def test_speculative_read_error_prevents_checkpoint(self):
        a=self.store(SerialRows);a.save_checkpoint(self.root/'rows.safetensors')
        b=self.store(source=self.root/'rows.safetensors');original=b._read
        entered=threading.Event()
        def failed(path,key,dependency):
            if path.name=='1.safetensors':
                entered.set()
                raise OSError('injected speculative failure')
            return original(path,key,dependency)
        with patch.object(b,'_read',failed):
            b[0];self.assertTrue(entered.wait(2))
            with self.assertRaises(OSError):b.save_checkpoint(self.root/'badread.safetensors')
            self.assertFalse((self.root/'badread.rows.json').exists())

if __name__=='__main__':unittest.main()
