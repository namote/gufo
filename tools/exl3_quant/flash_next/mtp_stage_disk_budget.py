"""CPU-only disk admission for full MTP input versus decoder, no mutation.

Input retains the existing conservative budget. Decoder may omit a second
76GiB state allocation only after verifying its existing complete derived rows,
input FC pack and full-calibration receipt chain under the actual mount mapping.
"""
import argparse,hashlib,json
from pathlib import Path,PurePosixPath

GIB=2**30
DECODER_HREFS_MIN=1043922944+2*812769280
DECODER_PACK_MIN=2*GIB

def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda:f.read(8<<20),b''):h.update(block)
    return h.hexdigest()

def parse_full_argv(argv):
    parser=argparse.ArgumentParser(allow_abbrev=False,exit_on_error=False)
    parser.add_argument('--stage',choices=['input','decoder'],required=True)
    resume=parser.add_mutually_exclusive_group();resume.add_argument('--resume-packed',action='store_true');resume.add_argument('--resume-capture',action='store_true')
    for name in ('source-checkpoint','work','recipe','qualification','gram-contract'):parser.add_argument('--'+name,required=True)
    parser.add_argument('--calibration',default='/task/c03-input-ids.json')
    options={x for x in argv if x.startswith('--')}
    for flag in options:
        assert '=' not in flag and argv.count(flag)==1,'Canonical unique option spelling required'
    try:return parser.parse_args(argv)
    except (SystemExit,argparse.ArgumentError) as error:raise ValueError('Unsupported/full-stage argv') from error

def map_path(raw,data,out,task,*,work=False):
    parsed=PurePosixPath(raw)
    assert parsed.is_absolute() and str(parsed)==raw and '..' not in parsed.parts,'Noncanonical container path'
    roots={'/mtp-run':Path(data),'/output':Path(out),'/task':Path(task)}
    prefix=next((key for key in roots if raw.startswith(key+'/')),None)
    assert prefix is not None and (not work or prefix=='/mtp-run'),'Invalid full-work mount'
    mapped=roots[prefix].joinpath(*parsed.relative_to(prefix).parts)
    assert mapped.resolve()==mapped and roots[prefix].resolve()==roots[prefix],'Symlink path outside reviewed mount'
    assert roots[prefix] in mapped.parents and mapped!=roots[prefix]
    return mapped

def inspect_derived(path):
    from row_checkpoint import open_rows
    path=Path(path);manifest=path.with_suffix('.rows.json')
    assert not path.exists() and not manifest.with_suffix('.pending').exists()
    paths,infos=open_rows(path);records=[];total=0
    for filename,info in zip(paths,infos,strict=True):
        assert filename.resolve()==filename and filename.parent==path.with_suffix('.rows'),'Derived row symlink/escape'
        stat=filename.stat();records.append([stat.st_dev,stat.st_ino,stat.st_size,stat.st_mtime_ns,info]);total+=stat.st_size
    identity={'manifest_sha256':digest(manifest),
              'rows_stat_sha256':hashlib.sha256(json.dumps(records,sort_keys=True).encode()).hexdigest()}
    return {'identity':identity,'infos':infos,'stored_bytes':total}

def required_budget(contract,state_bytes,data,out,task,*,inspect_rows=inspect_derived):
    phase=contract['phase']
    if phase!='full':
        assert phase in ('qualifier','probe')
        return {'required_bytes':150*GIB,'phase':phase,'policy':'unchanged_bounded_probe'}
    args=parse_full_argv(contract['argv']);hrefs=contract.get('full_hrefs_bytes',0);margin=contract.get('full_margin_bytes',0)
    assert type(hrefs) is int and type(margin) is int and margin>=16*GIB
    assert args.source_checkpoint=='/output/work/ckpt','Only protected idx51 source is admitted'
    work=map_path(args.work,data,out,task,work=True)
    if args.stage=='input':
        assert hrefs>=int(4.05*GIB)
        return {'required_bytes':100*GIB+state_bytes+hrefs+margin,'phase':'full','stage':'input',
                'state_allocation_bytes':state_bytes,'Hrefs_upper_bytes':hrefs,'margin_bytes':margin,
                'policy':'unchanged_full_input_budget'}
    # All existing allocations are already reflected by current free disk. No
    # credit is taken unless full input artifacts and every derived row match.
    packed_upper=contract.get('decoder_packed_upper_bound_bytes',0)
    assert type(packed_upper) is int and packed_upper>=DECODER_PACK_MIN
    assert hrefs>=DECODER_HREFS_MIN
    required_receipts=('INPUTS.json','INPUT-COMPLETE.json','mtp.PACKED.json')
    hashes={name:digest(work/name) for name in required_receipts}
    assert hashes==contract['decoder_input_receipts_sha256'],'Input-stage provenance changed'
    inputs=json.loads((work/'INPUTS.json').read_text());done=json.loads((work/'INPUT-COMPLETE.json').read_text());pack=json.loads((work/'mtp.PACKED.json').read_text())
    assert done['status']=='input_calibrated' and done['advanced_all_rows'] is True
    assert done['rows']==pack['rows']==inputs['rows']==2309
    assert done['tokens']==pack['tokens']==inputs['total_tokens']==1997187
    assert inputs['source_total_tokens']==1999496 and inputs['alignment']=='gufo_mtp_f32_full_hc_shifted_v1'
    assert len(done['H'])==2 and done['H']==pack['H']
    assert {v['first_key']:v['count'] for v in done['H'].values()}=={'mtp.fc_hidden':7988748,'mtp.fc_embedding':1997187}
    assert pack['inputs_sha256']==hashes['INPUTS.json']
    fc_bits={k:pack['strategy'][k] for k in ('mtp.fc_hidden','mtp.fc_embedding')}
    assert inputs['input_fc_bits']==done['input_fc_bits']==fc_bits
    assert pack['packed_sha256']==done['input_fc_pack_sha256']==digest(work/'qtensors/mtp.safetensors')
    assert pack['code_sha256']==inputs['code_sha256'] and pack['qualification_sha256']==inputs['qualification_sha256']
    for flag,field in ((args.recipe,'recipe_sha256'),(args.qualification,'qualification_sha256')):
        assert digest(map_path(flag,data,out,task))==inputs[field]
    source=Path(out)/'work/ckpt'
    assert inputs['source_identity']=={n:digest(source/n) for n in ('job.json','state.rows.json','original_input_ids.safetensors')}
    assert json.loads((source/'job.json').read_text())=={'next_module_idx':51,'q_strategy':None,'bad_rows':[]}
    cal=json.loads(map_path(args.calibration,data,out,task).read_text())
    assert cal['sha256']==inputs['calibration_sha256'] and cal['rows']==2309 and cal['total_tokens']==1999496
    derived=inspect_rows(work/'input-checkpoint/state.safetensors')
    assert derived['identity']==done['derived_state_identity']==contract['decoder_derived_state_identity']
    infos=derived['infos'];assert len(infos)==2309
    assert all(dtype=='F32' and tuple(shape)==(1,cal['sequences'][i]['tokens']-1,4,2560) for i,(dtype,shape) in enumerate(infos))
    assert sum(shape[1] for _,shape in infos)==1997187
    assert not (work/'DECODER-COMPLETE.json').exists(),'Completed decoder must not be rerun'
    return {'required_bytes':100*GIB+hrefs+packed_upper+margin,'phase':'full','stage':'decoder',
            'state_allocation_bytes':0,'reused_derived_bytes':derived['stored_bytes'],
            'derived_state_identity':derived['identity'],'input_receipts_sha256':hashes,
            'Hrefs_upper_bytes':hrefs,'packed_upper_bytes':packed_upper,'margin_bytes':margin,
            'policy':'validated_existing_input_state_no_decoder_advance'}
