"""Dynamic flush provenance; guarded input-preserving decoder capture migration.

No tensor or numerical code in this helper. Existing input receipts are immutable.
Caller change also enables the already guarded discarded-output capture scope;
the migration expressly does not call that decoder optimization metadata-only.
The runtime bytecode identity complements the caller's pinned leaf dependencies;
it is not invented source text and is explicitly Python-version-specific.
"""
import hashlib
import json
import marshal
from pathlib import Path
import sys


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for data in iter(lambda:f.read(8*2**20),b''):h.update(data)
    return h.hexdigest()


def dynamic_flush_identity(function):
    code=function.__code__
    assert function.__name__=='flush_capture_H'
    assert code.co_filename=='<lean-Gram-flush>', 'Unexpected dynamic flush origin'
    return {'flush_identity_kind':'python_marshaled_runtime_code_with_pinned_leaf_dependencies',
            'flush_code_sha256':hashlib.sha256(marshal.dumps(code)).hexdigest(),
            'flush_code_filename':code.co_filename,'marshal_version':marshal.version,
            'python_version':list(sys.version_info[:3])}


MIGRATABLE_CODE={'mtp_calibration_candidate.py','mtp_calibration_staged_probe.py','mtp_capture_identity.py'}


def validate_capture_migration(previous,current,migration,*,stage,inputs_sha256,packed_sha256,derived_sha256):
    """Pure check; no approval creation or rewriting of historical input evidence."""
    assert stage=='decoder', 'Capture migration is decoder-only'
    assert migration['status']=='reviewed_input_preserving_decoder_capture_migration'
    assert migration['reason']=='index1_dynamic_provenance_and_explicit_discarded_scope_input_math_unchanged'
    assert migration['input_arithmetic_unchanged'] is True
    assert migration['decoder_discarded_output_optimization_enabled'] is True
    assert migration['old_inputs_sha256']==inputs_sha256
    assert migration['input_packed_sha256']==packed_sha256
    assert migration['derived_states_sha256']==derived_sha256
    assert migration['old_code_sha256']==previous['code_sha256']
    assert migration['current_code_sha256']==current['code_sha256']
    assert {k:v for k,v in previous.items() if k!='code_sha256'}=={k:v for k,v in current.items() if k!='code_sha256'}, 'Source/calibration/recipe/row alignment changed'
    old,new=previous['code_sha256'],current['code_sha256']
    changed={k for k in old.keys()|new.keys() if old.get(k)!=new.get(k)}
    assert changed==MIGRATABLE_CODE, 'Unexpected code migration'
    assert set(new)-set(old)=={'mtp_capture_identity.py'} and not set(old)-set(new)
    assert migration['unchanged_input_pack_and_derived_states'] is True
    return True


def check_probe_inputs(work,current,stage):
    work=Path(work);previous=json.loads((work/'INPUTS.json').read_text())
    if previous==current:return None
    path=work/'DECODER-CAPTURE-MIGRATION.json'
    migration=json.loads(path.read_text())
    validate_capture_migration(previous,current,migration,stage=stage,
        inputs_sha256=digest(work/'INPUTS.json'),
        packed_sha256=digest(work/'qtensors/mtp.safetensors'),
        derived_sha256=digest(work/'derived-states.safetensors'))
    return digest(path)
