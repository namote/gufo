"""Complete only the missing MTP token case; preserve smoke003 and saved off baseline."""
import argparse,fcntl,json,re,shutil,signal,time
from contextlib import ExitStack
from pathlib import Path
import final_smoke_current_job as s

HEADER=re.compile(r'Qwen3\.8-Flash-Next tg depth=(\d+) cycles=(\d+) drafted=(\d+) accepted=(\d+) lookup_proposed=(\d+) lookup_accepted=(\d+) output_sha256=')
ERROR="AssertionError('No actual emitted-token hash in coherent candidate benchmark')"

def emitted_record(log):
    # One smoke depth/repeat only. No arbitrary generated-text hash search.
    headers=list(HEADER.finditer(log));assert len(headers)==1,'Expected exactly one native token record'
    h=headers[0];tail=log[h.end():]
    direct=re.match(r'([0-9a-f]{64}) text=',tail)
    if direct:digest=direct[1]
    else:
        rows=list(re.finditer(r'^([0-9a-f]{64}) text=',tail,re.M))
        assert len(rows)==1,'Missing or ambiguous standalone emitted-token hash'
        before=tail[:rows[0].start()].splitlines()
        assert before and re.fullmatch(r'\[Model Load\]: [0-9.]+ s',before[0]),'Unrecognized interleaving'
        assert before[1:2]==['KV cache: fp16']
        assert len(before)==4 and all(x.startswith('|') and x.endswith('|') for x in before[2:]),'Unexpected stdout before hash'
        digest=rows[0][1]
    d,c,p,a,lp,la=map(int,h.groups())
    assert d==0,'Smoke depth changed'
    return {'hashes':[['0',digest]],'drafts':[{'depth':d,'cycles':c,'drafted':p,'accepted':a}],
            'lookup_proposed':lp,'lookup_accepted':la,'stdout_interleaved':direct is None}

def evidence(path):
    return {'path':str(path),'state_sha256':s.sha(path/'STATE.json'),
            'log_sha256':s.sha(path/'run.log'),'runtime_sha256':s.sha(path/'RUNTIME.json')}

def validate_saved(c):
    old=Path(c['original_contract']);assert s.sha(old)==c['original_contract_sha256']
    original=json.loads(old.read_text());root=Path(original['out'])
    assert root==s.ROOT/'smoke-003'
    for p,h in c['identity'].items():assert s.sha(p)==h,p
    state=json.loads((root/'STATE.json').read_text())
    assert state['status']=='failed' and state['error']==ERROR and state['pack_unchanged']
    assert state['contract_sha256']==s.sha(old)
    assert state['runtime_manifest_sha256']==s.sha(original['runtime_manifest']) and state['actual_elf_sha256']==original['elf_sha256']
    summaries={};cases={}
    for name in ('http-off','http-mtp','tokens-off'):
        p=root/name;case=json.loads((p/'STATE.json').read_text())
        assert case['status']=='passed' and case['cleanup_errors']==[] and case['returned_KFD_contexts']==[]
        runtime=json.loads((p/'RUNTIME.json').read_text());assert runtime['elf_sha256']==original['elf_sha256'] and runtime['no_gguf_map']
        assert all(str(p/f) in c['identity'] for f in ('STATE.json','run.log','RUNTIME.json'))
        cases[name]=evidence(p)
        if name.startswith('http-'):
            assert str(p/'http/SUMMARY.json') in c['identity']
            summary=json.loads((p/'http/SUMMARY.json').read_text());assert summary['status']=='passed'
            summaries[name]=summary
    assert str(root/'STATE.json') in c['identity']
    for key in summaries['http-off']['greedy']:
        assert s.signature(summaries['http-off']['greedy'][key])==s.signature(summaries['http-mtp']['greedy'][key])
    for mode in ('off','mtp'):
        draft=sum(r['usage']['draft_tokens'] for r in summaries['http-'+mode]['greedy'].values())
        assert (draft==0 if mode=='off' else draft>0)
    offcase=json.loads((root/'tokens-off/STATE.json').read_text())
    assert offcase['command'][-len(s.bench_args('off')):]==s.bench_args('off')
    off=emitted_record((root/'tokens-off/run.log').read_text());assert off['drafts'][0]['drafted']==0
    return original,state,cases,off

def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);a=p.parse_args();c=json.loads(a.contract.read_text())
    assert c['status']=='reviewed_smoke_token_supplement' and c['seconds']==300
    review=json.loads(Path(c['review']).read_text());assert review['status']==c['status'] and review['contract_sha256']==s.sha(a.contract)
    required={str(s.TASK/n) for n in ('final_smoke_supplement_job.py','test_final_smoke_supplement_job.py')}
    assert required|{c['cpu']}<=c['identity'].keys()
    cpu=json.loads(Path(c['cpu']).read_text());assert cpu['status']=='pass' and cpu['gpu_used'] is False
    out=s.owned_path(c['out'],s.ROOT);assert out==s.ROOT/'smoke-003-supplement-001' and not out.exists()
    original,old,cases,off=validate_saved(c)
    with ExitStack() as stack:
        for path in (s.ROOT/'OWNER.lock',s.LOCK,s.JOB):
            f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert s.available()>=28*s.GIB and shutil.disk_usage(s.OUT).free>=150*s.GIB
        pack=s.check_contract(original)
        from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
        from conversion_admission import wait_conversion_admission
        assert not npu();_,admission=wait_conversion_admission(host,quiet,s.TASK/'conversion-desktop-contract.json')
        assert not host()['KFD_contexts'] and not npu()
        deadline=time.monotonic()+300;s.guard(deadline)
        def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        out.mkdir();state={'status':'running','contract_sha256':s.sha(a.contract),'original_contract_sha256':s.sha(c['original_contract']),
            'original_state_sha256':s.sha(Path(original['out'])/'STATE.json'),'runtime_manifest_sha256':old['runtime_manifest_sha256'],
            'actual_elf_sha256':old['actual_elf_sha256'],'source_revision':s.REVISION,'base_image':s.IMAGE,
            'admission':admission,'timing_scope':'smoke_only_not_throughput','cases':cases,'reused_off':off}
        s.save(out/'STATE.json',state)
        try:
            groups=sorted({Path('/dev/kfd').stat().st_gid,*[x.stat().st_gid for x in Path('/dev/dri').glob('renderD*')]})
            casepath=out/'tokens-mtp'
            s.run_case(original,pack,casepath,s.bench_args('mtp'),deadline,host,npu,groups)
            mtp=emitted_record((casepath/'run.log').read_text());assert mtp['drafts'][0]['drafted']>0
            assert off['hashes']==mtp['hashes'],'Exact emitted-token IDs differ'
            cases['tokens-mtp']=evidence(casepath)
            state.update(status='passed_pending_independent_review',cases=cases,token_hashes={'off':off['hashes'],'mtp':mtp['hashes']},
                draft_counters={'off':off['drafts'],'mtp':mtp['drafts']},http_greedy_equal=True,vision_and_cache_pass=True,
                quality_and_throughput_pending=True,original_smoke_state_preserved=True)
        except BaseException as e:state.update(status='failed',error=repr(e));raise
        finally:
            try:s.check_contract(original);validate_saved(c);state['pack_unchanged']=True
            except BaseException as e:state.update(status='protection_failed',protection_error=repr(e))
            state['finished']=time.time();s.save(out/'STATE.json',state)
        assert state['status']=='passed_pending_independent_review'

if __name__=='__main__':main()
