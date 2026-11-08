"""CPU-only recovery contract checks; no torch import, GPU or model work."""
import ast
import json
from pathlib import Path
import struct
import tempfile
import unittest
from unittest.mock import patch
import head_recovery as h

from paths import OUT
TMP=OUT/'head-recovery';TMP.mkdir(parents=True,exist_ok=True)

class Tests(unittest.TestCase):
    def test_numeric(self):
        good=dict(finite_raw=True,finite_quant=True,rfn=.001,cos=.0001,sqnr=50.)
        h.validate_metrics(good)
        for k,v in [('finite_raw',False),('finite_quant',False),('rfn',float('nan')),('cos',float('inf')),('sqnr',float('-inf')),('rfn',-1)]:
            with self.subTest(k=k,v=v),self.assertRaises(AssertionError):h.validate_metrics({**good,k:v})
    def test_output_shape_dtype_extent(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t,patch.object(h,'LENGTHS',[2]),patch.object(h,'VOCAB',4):
            p=Path(t)/'row.safetensors'
            header={'tensor':{'dtype':'F16','shape':[1,2,4],'data_offsets':[0,16]}}
            data=json.dumps(header).encode();p.write_bytes(struct.pack('<Q',len(data))+data+bytes(16))
            h.check_output_header(p,0)
            p.write_bytes(p.read_bytes()[:-1])
            with self.assertRaises(ValueError):h.check_output_header(p,0)
            with patch.object(h,'tensor_header',return_value={'tensor':{'dtype':'F32','shape':[1,2,4]}}):
                with self.assertRaises(AssertionError):h.check_output_header(p,0)
            with patch.object(h,'tensor_header',return_value={'tensor':{'dtype':'F16','shape':[1,3,4]}}):
                with self.assertRaises(AssertionError):h.check_output_header(p,0)
    def test_row_identity_guards(self):
        info=[('F16',(1,512,2560))]*2309
        with patch.object(h,'open_rows',return_value=([Path('/no-file')]*2309,info)):
            with self.assertRaises(AssertionError):h.row_identity('/no-file')
        with patch.object(h,'open_rows',return_value=([],[])):
            with self.assertRaises(AssertionError):h.row_identity('/no-file')
    def test_atomic_json_partial_refused(self):
        with tempfile.TemporaryDirectory(dir=TMP) as t:
            p=Path(t)/'ROW.json';h.write_json(p,{'done':True})
            self.assertEqual(json.loads(p.read_text()),{'done':True})
            p.with_suffix('.pending').write_text('uncommitted')
            with self.assertRaises(FileExistsError):h.write_json(p,{'done':False})
            self.assertEqual(json.loads(p.read_text()),{'done':True})
    def test_scope_no_capture_quant_commit(self):
        tree=ast.parse(Path(h.__file__).read_text())
        calls={ast.unparse(x.func) for x in ast.walk(tree) if isinstance(x,ast.Call)}
        self.assertTrue({'cm.main','cm.prepare','cm.prepare_state','cm.quantize_linears_parallel','raw.capture_H','raw.convert_exl3','subprocess.run','os.link','link_rows'}.isdisjoint(calls))
        self.assertIn('cm.get_state_error',calls);self.assertIn('save_file',calls)

if __name__=='__main__':unittest.main()
