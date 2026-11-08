"""CPU-only provenance/migration regressions; no torch/native/GPU imports."""
import ast
import copy
import hashlib
import inspect
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from mtp_capture_identity import (dynamic_flush_identity,validate_capture_migration,
                                  check_probe_inputs,digest)


def source_flush(params):
    return params


def dynamic(body='return x'):
    namespace={};exec(compile('def flush_capture_H(x):\n    '+body+'\n','<lean-Gram-flush>','exec'),namespace)
    return namespace['flush_capture_H']


class IdentityTests(unittest.TestCase):
    def setUp(self):
        self.old={'source_identity':{'idx':'51'},'tokens':8702,'recipe_sha256':'recipe',
                  'code_sha256':{'mtp_calibration_candidate.py':'oldcandidate',
                                 'mtp_calibration_staged_probe.py':'oldstaged','unchanged.py':'same'}}
        self.new=copy.deepcopy(self.old);self.new['code_sha256'].update(
            {'mtp_calibration_candidate.py':'newcandidate','mtp_calibration_staged_probe.py':'newstaged',
             'mtp_capture_identity.py':'newhelper'})
        self.migration={'status':'reviewed_input_preserving_decoder_capture_migration',
            'reason':'index1_dynamic_provenance_and_explicit_discarded_scope_input_math_unchanged',
            'input_arithmetic_unchanged':True,'decoder_discarded_output_optimization_enabled':True,
            'old_inputs_sha256':'input','input_packed_sha256':'packed','derived_states_sha256':'derived',
            'old_code_sha256':self.old['code_sha256'],'current_code_sha256':self.new['code_sha256'],
            'unchanged_input_pack_and_derived_states':True}
        self.kw=dict(stage='decoder',inputs_sha256='input',packed_sha256='packed',derived_sha256='derived')

    def test_dynamic_code_has_real_stable_changed_identity(self):
        fn=dynamic()
        with self.assertRaises(OSError):inspect.getsource(fn)
        self.assertEqual(dynamic_flush_identity(fn),dynamic_flush_identity(dynamic()))
        self.assertNotEqual(dynamic_flush_identity(fn)['flush_code_sha256'],dynamic_flush_identity(dynamic('return x+1'))['flush_code_sha256'])
        with self.assertRaises(AssertionError):dynamic_flush_identity(source_flush)

    def test_index0_keeps_real_source_hash_index1_needs_no_source(self):
        tree=ast.parse(Path(__file__).with_name('mtp_calibration_candidate.py').read_text())
        node=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and isinstance(n.test,ast.Compare)
                  and isinstance(n.test.left,ast.Name) and n.test.left.id=='index'
                  and any(isinstance(a,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='backend' for t in a.targets) for a in n.body))
        code=compile(ast.fix_missing_locations(ast.Module(body=[copy.deepcopy(node)],type_ignores=[])),'<backend-cpu-test>','exec')
        env={'index':0,'cm':SimpleNamespace(_flush_capture_H=source_flush),'hashlib':hashlib,'inspect':inspect}
        exec(code,env)
        self.assertEqual(env['backend'],{'flush_source_sha256':hashlib.sha256(inspect.getsource(source_flush).encode()).hexdigest(),'choice':'original_fp32_bmm'})
        env.update(index=1,cm=SimpleNamespace(_flush_capture_H=dynamic()));exec(code,env)
        self.assertEqual(env['backend']['choice'],'qualified_lt_static_decoder')
        self.assertNotIn('flush_source_sha256',env['backend'])

    def test_explicit_scope_only_wraps_discarded_index1_forward(self):
        tree=ast.parse(Path(__file__).with_name('mtp_calibration_candidate.py').read_text())
        scopes=[n for n in ast.walk(tree) if isinstance(n,ast.With) and any(isinstance(i.context_expr,ast.Call) and isinstance(i.context_expr.func,ast.Name) and i.context_expr.func.id=='_discarded_capture' for i in n.items)]
        self.assertEqual(len(scopes),1)
        guard=next(n for n in ast.walk(tree) if isinstance(n,ast.If) and scopes[0] in n.body)
        self.assertEqual(ast.unparse(guard.test),'index == 1')
        self.assertEqual(len(scopes[0].body),1)
        self.assertEqual(ast.unparse(scopes[0].body[0]),'y = module.forward(x, params)')
        self.assertEqual(ast.unparse(guard.orelse[0]),'y = module.forward(x, params)')

    def test_valid_migration_and_no_mutations(self):
        before=copy.deepcopy((self.old,self.new,self.migration))
        self.assertTrue(validate_capture_migration(self.old,self.new,self.migration,**self.kw))
        self.assertEqual(before,(self.old,self.new,self.migration))

    def test_reject_stage_approval_and_evidence_changes(self):
        for field,value in [('stage','input'),('inputs_sha256','changed'),('packed_sha256','changed'),('derived_sha256','changed')]:
            with self.assertRaises(AssertionError):validate_capture_migration(self.old,self.new,self.migration,**{**self.kw,field:value})
        for field,value in [('status','pending'),('input_arithmetic_unchanged',False),('decoder_discarded_output_optimization_enabled',False)]:
            with self.assertRaises(AssertionError):validate_capture_migration(self.old,self.new,{**self.migration,field:value},**self.kw)

    def test_reject_recipe_source_or_unapproved_code_changes(self):
        for field,value in [('recipe_sha256','different'),('source_identity',{'idx':'52'})]:
            with self.assertRaises(AssertionError):validate_capture_migration(self.old,{**self.new,field:value},self.migration,**self.kw)
        changed=copy.deepcopy(self.new);changed['code_sha256']['unchanged.py']='different'
        receipt={**self.migration,'current_code_sha256':changed['code_sha256']}
        with self.assertRaises(AssertionError):validate_capture_migration(self.old,changed,receipt,**self.kw)

    def test_real_file_validation_preserves_input_receipts(self):
        with tempfile.TemporaryDirectory() as td:
            w=Path(td);(w/'qtensors').mkdir();(w/'INPUTS.json').write_text(json.dumps(self.old))
            (w/'qtensors/mtp.safetensors').write_bytes(b'fake small CPU fixture pack');(w/'derived-states.safetensors').write_bytes(b'fake small CPU fixture states')
            before=(w/'INPUTS.json').read_bytes()
            with self.assertRaises(FileNotFoundError):check_probe_inputs(w,self.new,'decoder')
            receipt={**self.migration,'old_inputs_sha256':digest(w/'INPUTS.json'),'input_packed_sha256':digest(w/'qtensors/mtp.safetensors'),'derived_states_sha256':digest(w/'derived-states.safetensors')}
            (w/'DECODER-CAPTURE-MIGRATION.json').write_text(json.dumps(receipt))
            self.assertEqual(check_probe_inputs(w,self.new,'decoder'),digest(w/'DECODER-CAPTURE-MIGRATION.json'))
            self.assertEqual((w/'INPUTS.json').read_bytes(),before)


if __name__=='__main__':unittest.main()
