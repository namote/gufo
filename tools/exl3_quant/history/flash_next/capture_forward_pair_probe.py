"""Prepared GPU qualification, NOT launched during a live converter.

Reuse retained layer6/idx9 64-row inputs and original-half Lt numerical control.
Two candidate passes, no baseline recapture or quantization. Read-only row reader
changes I/O plumbing, so timings are diagnostic, not a matched speed claim.
"""
import copy,hashlib,inspect,json,time,sys
from pathlib import Path
import torch
from exllamav3.conversion import convert_model as cm
from exllamav3.loader.safetensors_alt import safe_open
from exllamav3.modules.linear import Linear
from disk_state import _read_row
from row_checkpoint import open_rows
import capture_fast,capture_lt
from capture_forward_pair import LtPairs,prepare,installed,cleanup_all
R=Path('/output/capture-forward-pair');P=Path('/output/capture-pipeline');L=Path('/output/capture-lt-evaluate')
R.mkdir(exist_ok=True)
# Runner must provide protected read-only bind mounts and exclusive GPU lease.
mounts=[s.split() for s in Path('/proc/self/mountinfo').read_text().splitlines()]
for protected in ['/source','/output/work','/output/pack',str(P),str(L)]:
    assert any(s[4]==protected and 'ro' in s[5].split(',') for s in mounts),protected
assert not (R/'GPU-PROVENANCE.json').exists(),'Reuse completed/failed candidate evidence, never overwrite'
assert hashlib.sha256(inspect.getsource(inspect.unwrap(cm.main)).encode()).hexdigest()==capture_fast._MAIN_SHA
cpu=json.loads((R/'CPU.json').read_text());assert cpu['status']=='pass' and not cpu['gpu_used']
for n,h in cpu['identity'].items():assert hashlib.sha256((Path('/task')/n).read_bytes()).hexdigest()==h
build=json.loads(Path('/output/capture-forward/BUILD.json').read_text())
assert hashlib.sha256(Path('/output/capture-forward/forward.so').read_bytes()).hexdigest()==build['binary_sha256']
prior_state=json.loads((L/'STATE.json').read_text());assert prior_state['status']=='completed_pending_review' and not prior_state['cleanup_errors'] and not prior_state['returned_KFD_contexts']
previous=json.loads((L/'SUMMARY.json').read_text());assert previous['status']=='lt_candidate_evaluation_complete'
input_receipt=json.loads((R/'INPUT.json').read_text())
for name,expected in input_receipt['artifacts'].items():assert hashlib.sha256(Path(name).read_bytes()).hexdigest()==expected,name
control_path=Path(input_receipt['control_path']);assert control_path==L/'1-original_f16/numeric.pt'
control_hash=hashlib.sha256()
with control_path.open('rb') as stream:
    while data:=stream.read(8*2**20):control_hash.update(data)
assert control_hash.hexdigest()==input_receipt['control_sha256']
control=torch.load(control_path,map_location='cpu',weights_only=True)
assert len(control['capture_H'])==516 and len(control['ref_states'])==5
source=P/'input/state.safetensors';paths,infos=open_rows(source)
assert len(paths)==64
provenance=json.loads((P/'PROVENANCE.json').read_text());assert provenance['target']==9
job=provenance['checkpoint'];assert job==dict(next_module_idx=9,q_strategy=None,bad_rows=[])
contract=json.loads(Path('/task/capture-forward-pair-contract.json').read_text())
args0=json.loads(Path('/output/work/args.json').read_text())
assert args0['cal_rows']==2309
lengths=[info[1][1] for info in infos];assert sum(lengths)==50056

def digest(t):return hashlib.sha256(memoryview(t.contiguous().view(torch.uint8).numpy())).hexdigest()
def tree(v):
    if isinstance(v,torch.Tensor):return v.detach().to('cpu',copy=True)
    if isinstance(v,dict):return {k:tree(x) for k,x in v.items()}
    if isinstance(v,(list,tuple)):return type(v)(tree(x) for x in v)
    return copy.deepcopy(v)
def exact(a,b):
    if isinstance(a,torch.Tensor):assert a.shape==b.shape and a.dtype==b.dtype and torch.equal(a.contiguous().reshape(-1).view(torch.uint8),b.contiguous().reshape(-1).view(torch.uint8))
    elif isinstance(a,dict):
        assert a.keys()==b.keys()
        for k in a:exact(a[k],b[k])
    elif isinstance(a,(tuple,list)):
        assert type(a)==type(b) and len(a)==len(b)
        for x,y in zip(a,b):exact(x,y)
    else:assert a==b,(a,b)
def write(n,v):(R/n).write_text(json.dumps(v,indent=2)+'\n')
class Rows:
    def __init__(self):self.index=None;self.value=None;self.hashes={}
    def __len__(self):return len(paths)
    def __getitem__(self,i):
        if not isinstance(i,int) or not 0<=i<len(paths):raise IndexError(i)
        if self.index!=i:
            with safe_open(str(paths[i])) as f:self.value=_read_row(f,'tensor')
            h=digest(self.value);assert h==control['input_hashes'][i]
            self.hashes[i]=h;self.index=i
        return self.value
    def __setitem__(self,i,v):raise AssertionError('Capture qualification must never advance state')
class Done(BaseException):pass
def forbidden(*a,**kw):raise AssertionError('No quantization or checkpoint writes in diagnostic')
original_prepare,original_bar,original_flush=cm.prepare_state,cm.ProgressBar,cm._flush_capture_H
names=('save_tensor','save_dict','quantize_linears_single','quantize_linears_parallel','compile_model')
originals={n:getattr(cm,n) for n in names};original_convert=Linear.convert_exl3
old_cpu,old_gpu=torch.random.get_rng_state(),torch.cuda.get_rng_state(0)
records=[];start=time.monotonic()
write('GPU-PROVENANCE.json',dict(status='started',saved_control=str(L/'1-original_f16/numeric.pt'),input_source=str(source),target=9,rows=64,tokens=50056,production_changed=False,timing_scope='diagnostic only; row reader differs from saved baseline; packing reported separately',script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
try:
    for n in names:setattr(cm,n,forbidden)
    Linear.convert_exl3=forbidden
    for repetition in range(2):
        assert time.monotonic()-start<480
        case=R/f'case-{repetition}';case.mkdir()
        args=copy.deepcopy(args0);args.update(work_dir=str(case),out_dir='/output/pack',ngram_file='/output/pack/ngram_embedding.safetensors',cal_rows=64,cal_cols=max(lengths),max_module=9);args.pop('clear_files',None)
        state_rows=Rows();live=[];collected={};pair_context=None;pair_backend=None
        gram=capture_lt.LtGram('/output/capture-gram/gram.so','original_f16')
        restore_fast=capture_fast.install(cm,automatic=True)
        cm._flush_capture_H=capture_lt.make_flush(original_flush,gram)
        def prepare_state(in_args,in_job,config,model,tokenizer):
            assert in_args is args and in_job==job
            assert model.modules[9].key=='model.language_model.layers.6'
            live.append((config,model));return state_rows,[x.clone() for x in control['original_input_ids']]
        cm.prepare_state=prepare_state
        class Bar(original_bar):
            def __enter__(self):
                global pair_context,pair_backend
                assert self.text.startswith(' -- Capturing: ') and self.count==64
                assert torch.is_inference_mode_enabled()
                exact(torch.random.get_rng_state(),control['cpu_rng']);exact(torch.cuda.get_rng_state(0),control['gpu_rng'])
                torch.cuda.reset_peak_memory_stats();torch.cuda.synchronize();t=time.perf_counter()
                pair_backend=LtPairs('/output/capture-forward/forward.so')
                pair_state,packed_bytes=prepare(live[0][1].modules[9],pair_backend,contract)
                pair_context=installed(live[0][1].modules[9],pair_state);pair_context.__enter__()
                collected['pair_state']=pair_state;collected['packed_bytes']=packed_bytes
                collected['allowed_changed_H']={d.qmap for d in live[0][1].modules[9].mlp.downs}
                torch.cuda.synchronize();collected['packing_seconds']=time.perf_counter()-t
                self.started=time.perf_counter();return super().__enter__()
            def __exit__(self,typ,val,tb):
                if typ is not None:
                    frame=inspect.currentframe().f_back
                    try:
                        partial=tree({k:frame.f_locals[k] for k in ('capture_H','ref_states','quant_preserves') if k in frame.f_locals})
                        torch.save(partial,case/'failed-partial.pt')
                    except BaseException as save_error:
                        if val is not None:val.add_note('Partial evidence save failed: '+repr(save_error))
                    finally:del frame
                    return super().__exit__(typ,val,tb)
                super().__exit__(typ,val,tb);torch.cuda.synchronize();collected['seconds']=time.perf_counter()-self.started
                frame=inspect.currentframe().f_back;local=frame.f_locals
                assert local['idx']==9 and not local['bad_rows']
                numeric=tree({k:local[k] for k in ('capture_H','ref_states','quant_preserves','original_input_ids')})
                del local,frame
                numeric.update(cpu_rng=torch.random.get_rng_state(),gpu_rng=torch.cuda.get_rng_state(0),input_hashes=[state_rows.hashes[i] for i in range(64)])
                collected['numeric']=numeric
                torch.save(numeric,case/'numeric.pt') # preserve before any assertion
                raise Done()
        cm.ProgressBar=Bar
        torch.random.set_rng_state(control['cpu_rng']);torch.cuda.set_rng_state(control['gpu_rng'],0)
        try:
            try:cm.main(args,copy.deepcopy(job))
            except Done:pass
            else:raise AssertionError('Capture boundary missed')
            actual=collected['numeric'];assert len(actual['capture_H'])==516
            exact({k:v for k,v in control.items() if k!='capture_H'},{k:v for k,v in actual.items() if k!='capture_H'})
            assert actual['capture_H'].keys()==control['capture_H'].keys()
            differences={}
            for k,h in actual['capture_H'].items():
                ref=control['capture_H'][k]
                exact({n:v for n,v in ref.items() if n!='H'},{n:v for n,v in h.items() if n!='H'})
                assert torch.isfinite(h['H']).all() and int(h['inf_nan'].sum())==0 and int(h['count_dev'])==50056
                d=h['H'].double()-ref['H'].double();differences[k]=dict(exact=torch.equal(h['H'].view(torch.uint8),ref['H'].view(torch.uint8)),max_abs=float(d.abs().max()),relative_l2=float(d.norm()/ref['H'].double().norm().clamp_min(1e-30)))
            write(f'DIFFERENCES-{repetition}.json',differences)
            assert all(v['exact'] or k in collected['allowed_changed_H'] for k,v in differences.items()), 'Unexpected non-down H change'
            if repetition:exact(torch.load(R/'case-0/numeric.pt',weights_only=True,map_location='cpu'),actual)
            collected['pair_state'].drained();assert collected['pair_state'].stats==dict(up_calls=512*64,gate_calls=512*64)
            records.append(dict(repetition=repetition,seconds=collected['seconds'],packing_seconds=collected['packing_seconds'],packed_bytes=collected['packed_bytes'],peak_gpu_bytes=torch.cuda.max_memory_allocated(),counters_refs_inputs_rng_exact=True,pair_stats=collected['pair_state'].stats,plan_cache=pair_backend.cache.stats.copy(),max_H_relative_l2=max(d['relative_l2'] for d in differences.values()),repeat_exact=bool(repetition)))
            write('PROGRESS.json',records)
        finally:
            exc = sys.exc_info()
            actions=[]
            if pair_context:actions.append(('pair_scope',lambda:pair_context.__exit__(*exc)))
            if pair_backend:actions.append(('pair_plans',pair_backend.close))
            actions += [('gram_plans',gram.close),('fast_hook',restore_fast)]
            for config,model in live:
                actions.extend([('unload',lambda model=model:model.modules[9].unload()),('source_close',config.stc.close)])
            try:
                errors=cleanup_all(actions,exc[1])
                if errors:write(f'CLEANUP-{repetition}.json',errors)
            finally:
                cm._flush_capture_H=original_flush;cm.ProgressBar=original_bar;cm.prepare_state=original_prepare
    write('SUMMARY.json',dict(status='pair_capture_screen_complete',results=records,scope='Changed H is diagnostic only; does not qualify quantized quality or whole conversion speed',production_changed=False))
except BaseException as e:
    write('FAILURE.json',dict(error=repr(e),saved_cases=[str(p) for p in R.glob('case-*/numeric.pt')]))
    raise
finally:
    for n,v in originals.items():setattr(cm,n,v)
    Linear.convert_exl3=original_convert;cm.prepare_state=original_prepare;cm.ProgressBar=original_bar;cm._flush_capture_H=original_flush
    torch.random.set_rng_state(old_cpu);torch.cuda.set_rng_state(old_gpu,0)
