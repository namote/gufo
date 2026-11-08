"""Bounded real-row I/O + GPU transfer/elementwise pipeline, not model timing."""
import hashlib,json,time
from pathlib import Path
import torch
from disk_state import DiskRows as SerialRows
from disk_state_async import DiskRows as AsyncRows
from row_checkpoint import open_rows,link_rows

O=Path('/output');R=O/'capture-io'
paths,infos=open_rows(O/'work/ckpt/state.safetensors')
assert len(paths)==2309
def protection():
    files=[O/'work/ckpt/state.rows.json',O/'work/ckpt/original_input_ids.safetensors']
    return dict(metadata={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in files},rows=[(p.stat().st_ino,p.stat().st_size,p.stat().st_mtime_ns) for p in paths])
before=protection()
root=R/'probe-work';root.mkdir()
source=R/'fixture/state.safetensors'
if True:
    # Small kernel/runtime initialization outside measured windows.
    (torch.ones(32,device='cuda')*1.25+3).cpu();torch.cuda.synchronize()
    results=[];control=None
    for number,cls in enumerate([SerialRows,AsyncRows,AsyncRows,SerialRows]):
        start=time.monotonic();s=cls(source,root)
        try:
            for i in range(len(s)):
                s[i]=s[i].to('cuda')
                out=s[i]*1.25+3
                assert torch.isfinite(out).all().item()
                s[i]=out.cpu()
            output=root/f'pass{number}.safetensors'
            s.save_checkpoint(output);torch.cuda.synchronize()
            elapsed=time.monotonic()-start
            result=dict(mode=cls.__module__,seconds=elapsed,rows=len(s),io_stats=getattr(s,'stats',{}).copy())
        finally:s.close()
        files,_=open_rows(output)
        hashes=[]
        for p in files:
            h=hashlib.sha256()
            with p.open('rb') as f:
                while b:=f.read(4*2**20):h.update(b)
            hashes.append(h.hexdigest())
        if control is None:control=hashes
        result['hashes']=hashes
        result['saved_control_byte_exact']=hashes==control;results.append(result)
        (R/'PROGRESS.json').write_text(json.dumps(results,indent=2))
        assert hashes==control,'GPU state I/O byte mismatch'
        print(json.dumps(result),flush=True)
        (R/'PROGRESS.json').write_text(json.dumps(results,indent=2))
    assert before==protection(),'Source checkpoint metadata changed'
    (R/'SOURCE-PROTECTION.json').write_text(json.dumps(before,indent=2))
    (R/'RESULTS.json').write_text(json.dumps(dict(status='pass',scope='16 real idx6 rows; GPU transfer and fixed elementwise only; not model speed',results=results,hashes=control,source_checkpoint_unchanged=True),indent=2))
