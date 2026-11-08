"""K4V4 comparison only; frozen FP16 baseline is read, never executed again.

Only explicit reviewed reuse_cases may reuse successful measurements. Failed case
paths never overwrite; prepare a new run contract reusing successful cases only.
No automatic retries, baseline substitution, deployment or model mutation.
"""
import argparse,fcntl,hashlib,json,math,os,re,shutil,signal,socket,subprocess,threading,time,uuid
from contextlib import ExitStack
from pathlib import Path
import final_smoke_current_job as s
from final_kv4_http import run_suite
import final_throughput_current_job as baseline
from final_validation_http import signature
from owned_cleanup import cleanup_owned
from run_stage import OUT,TASK,LOCK,available,save
from mtp_calibration_job import sha,held_flock,owned_path
ROOT=OUT/'final-validation';JOB=OUT/'model-job-c03/JOB.lock';GIB=2**30
POOL_BYTES=1932735283
KV='kvarn-k4v4';KVROOT=ROOT/'kv4';CONTROL=KVROOT/'CONTROL.json'


def bench_args(mode,kind):
    assert kind in ('pp','tg')
    argv=['bench','-m','/models/fnx','-r','2','--kv-cache-type',KV,'--temperature','0','--top-p','1',
        '--top-k','0','--min-p','0','--repeat-penalty','1','--seed','314159','--verbose',*s.speculative(mode)]
    return argv+(['-p','2048,8192','-d','0'] if kind=='pp' else ['-n','128','-d','0,32768'])


def server_args(mode,concurrency):
    argv=s.server_args(mode,32768,concurrency,POOL_BYTES)
    argv[argv.index('--kv-cache-type')+1]=KV
    return argv


def cases():
    result=[]
    for mode in ('off','mtp'):
        for kind in ('pp','tg'):result.append(dict(name=kind+'-'+mode,mode=mode,kind=kind,argv=bench_args(mode,kind)))
        for concurrency in (1,2):
            result.append(dict(name=f'http-c{concurrency}-{mode}',mode=mode,kind='http',concurrency=concurrency,
                argv=server_args(mode,concurrency)))
    return result


def parse_bench(text,kind,mode,trace=None):
    assert 'KV cache: '+KV in text,'Actual benchmark KV differs'
    trace=text if trace is None else trace
    rows={}
    for line in text.splitlines():
        cells=[x.strip() for x in line.split('|')]
        if len(cells)!=7 or not re.fullmatch(r'(pp|tg)\d+( @ d\d+)?',cells[4]):continue
        match=re.fullmatch(r'([\d.]+)\s*±\s*([\d.]+)',cells[5]);assert match,line
        mean,stdev=map(float,match.groups());assert math.isfinite(mean) and mean>0 and math.isfinite(stdev) and stdev>=0
        assert cells[4] not in rows;rows[cells[4]]=dict(mean_tps=mean,stddev_tps=stdev,repetitions=2)
    expected={'pp2048','pp8192'} if kind=='pp' else {'tg128','tg128 @ d32768'}
    assert set(rows)==expected,(set(rows),expected)
    result=dict(status='passed',rows=rows,scope='CLImean of two whole prefill/DecodeStep-loop token rates; generation includes draft,verify,rollback; not process wall')
    if kind=='tg':
        result['hashes']=s.token_hashes(trace);result['drafts']=s.draft_counters(trace)
        assert [d for d,h in result['hashes']]==['0','0','32768','32768']
        assert len(result['drafts'])==4
        assert all(x['drafted']==0 if mode=='off' else x['drafted']>0 for x in result['drafts'])
    return result


def binding(c,case):
    value=dict(runtime_manifest_sha256=c['identity'][c['runtime_manifest']],elf_sha256=c['elf_sha256'],pack_stats=c['pack_stats'],
        compile_receipt_sha256=c['compile_receipt_sha256'],case=case,pool_bytes=POOL_BYTES,
        helper_sha256={n:c['identity'][str(TASK/n)] for n in ('final_kv4_throughput_job.py','final_kv4_http.py','final_throughput_http.py','final_throughput_current_job.py','final_smoke_current_job.py','final_validation_http.py')})
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def saved_case(spec,expected):
    path=owned_path(spec['path'],ROOT);statepath=path/'STATE.json';resultpath=path/'RESULT.json'
    assert sha(statepath)==spec['state_sha256'] and sha(resultpath)==spec['result_sha256']
    state=json.loads(statepath.read_text());assert state['status']=='passed' and state['measurement_binding']==expected
    assert not state['cleanup_errors'] and not state['returned_KFD_contexts']
    assert state['result_sha256']==sha(resultpath)
    for relative,digest in state['evidence_sha256'].items():
        p=path/relative;assert p.resolve().is_relative_to(path.resolve()) and sha(p)==digest
    return json.loads(resultpath.read_text()),str(path)


def validate_smoke(c):
    """Validate preserved failed003 plus its explicit one-case completion receipt."""
    p=owned_path(c['smoke_state'],ROOT);assert sha(p)==c['smoke_state_sha256']
    state=json.loads(p.read_text());assert state['status']=='passed_pending_independent_review' and state['pack_unchanged']
    assert state['http_greedy_equal'] and state['vision_and_cache_pass'] and state['original_smoke_state_preserved']
    assert state['token_hashes']['off']==state['token_hashes']['mtp'] and state['token_hashes']['off']
    assert state['actual_elf_sha256']==c['elf_sha256'] and state['runtime_manifest_sha256']==c['identity'][c['runtime_manifest']]
    assert state['contract_sha256']==sha(c['smoke_contract'])
    supplement=json.loads(Path(c['smoke_contract']).read_text())
    original_path=Path(supplement['original_contract']);assert sha(original_path)==state['original_contract_sha256']==supplement['original_contract_sha256']
    original=json.loads(original_path.read_text());assert original['pack_stats']==c['pack_stats'] and original['pack']==c['pack']
    oldroot=owned_path(original['out'],ROOT);oldpath=oldroot/'STATE.json'
    assert sha(oldpath)==state['original_state_sha256']
    old=json.loads(oldpath.read_text())
    assert old['status']=='failed' and old['error']=="AssertionError('No actual emitted-token hash in coherent candidate benchmark')" and old['pack_unchanged']
    assert old['contract_sha256']==sha(original_path) and old['actual_elf_sha256']==c['elf_sha256']
    review=json.loads(Path(c['smoke_result_review']).read_text())
    assert review['status']=='passed' and review['smoke_state_sha256']==sha(p) and review['original_state_sha256']==sha(oldpath)
    assert set(state['cases'])=={'http-off','http-mtp','tokens-off','tokens-mtp'}
    for name,proof in state['cases'].items():
        path=owned_path(proof['path'],ROOT)
        assert path==(p.parent if name=='tokens-mtp' else oldroot)/name
        for filename,key in [('STATE.json','state_sha256'),('run.log','log_sha256'),('RUNTIME.json','runtime_sha256')]:assert sha(path/filename)==proof[key]
        row=json.loads((path/'STATE.json').read_text());assert row['status']=='passed' and not row['cleanup_errors'] and not row['returned_KFD_contexts']
        runtime=json.loads((path/'RUNTIME.json').read_text());assert runtime['elf_sha256']==c['elf_sha256'] and runtime['no_gguf_map']


def check_contract(c):
    assert c['status']=='reviewed_final_kv4_throughput' and c['seconds']==3600
    required={str(TASK/x) for x in ('final_kv4_throughput_job.py','final_kv4_http.py','test_final_kv4_throughput.py','final_throughput_current_job.py','final_throughput_http.py')}
    required|={c['cli_cpu'],c['smoke_contract'],c['smoke_result_review'],c['kv_layout_source']}
    assert 'kKvBlockTokens = 256;' in Path(c['kv_layout_source']).read_text();assert required<=c['identity'].keys()
    cpu=json.loads(Path(c['cpu']).read_text());assert cpu['status']=='pass' and not cpu['gpu_used']
    for p,h in cpu['identity'].items():assert sha(p)==h
    cli=json.loads(Path(c['cli_cpu']).read_text());assert cli['status']=='pass' and not cli['gpu_used']
    assert cli['runtime_manifest_sha256']==c['identity'][c['runtime_manifest']]
    assert cli['argv']==[x['argv'] for x in cases()]
    assert set(c['reuse_cases'])<={x['name'] for x in cases()}
    pack=s.check_contract(c);validate_smoke(c);validate_baseline(c);return pack


def guard(deadline):
    assert time.monotonic()<deadline,'Final throughput3600s deadline'
    control=json.loads(CONTROL.read_text());assert control['enabled'],'Latest user stop'
    assert control['scope']=='c03-kvarn-k4v4-validation' and control['automatic_resume'] is False and control['callbacks'] is False
    mem=available();assert mem>=4*GIB,'Memory floor4GiB';assert shutil.disk_usage(OUT).free>=100*GIB
    locks=Path('/proc/locks').read_text()
    for p in (LOCK,JOB):assert held_flock(locks,os.getpid(),p.stat())
    return mem


def admit(host,npu,deadline):
    stop=min(deadline,time.monotonic()+60)
    while True:
        guard(deadline);assert not npu();h=host()
        if not h['KFD_contexts'] and h['telemetry']['gpu_busy_percent']<=4:return h
        assert time.monotonic()<stop,'No quiet GPU for fair timing; no other work stopped'
        time.sleep(1)


def run_case(c,pack,out,case,deadline,host,npu,groups):
    expected=binding(c,case);out.mkdir();name='gufo-final-kv4-'+uuid.uuid4().hex[:10];label=uuid.uuid4().hex
    port=None
    if case['kind']=='http':
        with socket.socket() as sock:sock.bind(('127.0.0.1',0));port=sock.getsockname()[1]
    cmd=s.command(pack,out,name,label,case['argv'],groups,port)
    state=dict(status='running',measurement_binding=expected,started=time.time(),command=cmd,container=name,label=label,
        admission=admit(host,npu,deadline),cleanup_errors=[],returned_KFD_contexts=[])
    save(out/'STATE.json',state);child=None;thread=None;response={};mapped=False
    def requests():
        try:
            import urllib.request
            url=f'http://127.0.0.1:{port}';until=min(deadline,time.monotonic()+300)
            while True:
                assert child.poll() is None,'Owned server exited before readiness';assert time.monotonic()<until
                try:
                    with urllib.request.urlopen(url+'/ready',timeout=2) as r:
                        if r.status==200:break
                except Exception:time.sleep(.5)
            response['value']=run_suite(url,out/'http',case['concurrency'])
        except BaseException as e:response['error']=repr(e)
    try:
        with (out/'run.log').open('x') as log,(out/'stderr.log').open('x') as err:
            child=subprocess.Popen(cmd,stdout=log,stderr=err)
            if case['kind']=='http':thread=threading.Thread(target=requests,daemon=True);thread.start()
            while child.poll() is None:
                mem=guard(deadline);assert not npu()
                state['minimum_available_bytes']=min(state.get('minimum_available_bytes',mem),mem)
                state['memory_warning_observed']=state.get('memory_warning_observed',False) or mem<8*GIB
                if not mapped:mapped=s.runtime_maps(name,label,out,c['elf_sha256'])
                state['updated']=time.time();save(out/'STATE.json',state)
                if thread and not thread.is_alive():break
                time.sleep(1)
            if thread:
                assert not thread.is_alive() and 'error' not in response,response.get('error')
                result=response['value'];result['pool']=pool_geometry((out/'stderr.log').read_text())
                drafts=sum(t['draft_tokens'] for g in result['groups'].values() for t in g['timings'].values())
                assert drafts==0 if case['mode']=='off' else drafts>0
            else:
                assert child.returncode==0,child.returncode
                result=parse_bench((out/'run.log').read_text(),case['kind'],case['mode'],(out/'stderr.log').read_text())
            assert mapped,'Running coherentruntime notverified'
            save(out/'RESULT.json',result);state.update(status='passed',result_sha256=sha(out/'RESULT.json'))
    except BaseException as e:state.update(status='failed',error=repr(e));raise
    finally:
        if child is not None:
            state.update(cleanup_owned(name,label,child,host))
            if state['cleanup_errors'] or state['returned_KFD_contexts']:state['status']='cleanup_failed'
        if thread:thread.join(timeout=15)
        if thread and thread.is_alive():state['status']='http_thread_cleanup_failed'
        state['finished']=time.time()
        state['evidence_sha256']={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file() and p.name!='STATE.json'}
        save(out/'STATE.json',state)
    assert state['status']=='passed',state
    return result


def compare(results):
    assert results['tg-off']['hashes']==results['tg-mtp']['hashes'],'Exact greedy token SHA differs'
    output={'bench':{},'http':{}}
    for kind in ('pp','tg'):
        output['bench'][kind]={key:dict(off=v['mean_tps'],mtp=results[kind+'-mtp']['rows'][key]['mean_tps'],
            ratio=results[kind+'-mtp']['rows'][key]['mean_tps']/v['mean_tps'],off_stddev=v['stddev_tps'],
            mtp_stddev=results[kind+'-mtp']['rows'][key]['stddev_tps']) for key,v in results[kind+'-off']['rows'].items()}
    for concurrency in (1,2):
        off=results[f'http-c{concurrency}-off'];mtp=results[f'http-c{concurrency}-mtp']
        for prompt,row in off['groups']['greedy']['rows'].items():assert signature(row)==signature(mtp['groups']['greedy']['rows'][prompt]),f'HTTPgreedy mismatch C{concurrency} {prompt}'
        output['http'][str(concurrency)]={name:dict(off=g['aggregate_emitted_tokens_per_group_wall'],
            mtp=mtp['groups'][name]['aggregate_emitted_tokens_per_group_wall'],
            ratio=mtp['groups'][name]['aggregate_emitted_tokens_per_group_wall']/g['aggregate_emitted_tokens_per_group_wall'],
            per_request_off=g['timings'],per_request_mtp=mtp['groups'][name]['timings']) for name,g in off['groups'].items()}
    return output


def pool_geometry(text):
    rows=re.findall(r'event=kv_pool_created model=flash-next blocks=(\d+) block_bytes=(\d+) bytes=(\d+)',text)
    assert len(rows)==1,'Actual shared KV pool evidence missing/ambiguous'
    blocks,block_bytes,total=map(int,rows[0]);assert blocks>0 and block_bytes>0 and blocks==POOL_BYTES//block_bytes and blocks*block_bytes==total<=POOL_BYTES
    return dict(blocks=blocks,block_bytes=block_bytes,allocated_bytes=total,configured_bytes=POOL_BYTES,
                tokens_per_block=256,token_capacity=blocks*256,capacity_source='Actual logged blocks times pinned runtime kKvBlockTokens=256')


def validate_baseline(c):
    p=Path(c['fp16_contract']);assert str(p) in c['identity'] and sha(p)==c['identity'][str(p)]
    old=json.loads(p.read_text());assert old['elf_sha256']==c['elf_sha256'] and old['pack_stats']==c['pack_stats'] and old['pack']==c['pack']
    assert old['runtime_manifest']==c['runtime_manifest'] and old['identity'][old['runtime_manifest']]==c['identity'][c['runtime_manifest']]
    root=owned_path(old['out'],ROOT);reviewpath=Path(c['fp16_review'])
    assert str(reviewpath) in c['identity'] and sha(reviewpath)==c['identity'][str(reviewpath)]
    review=json.loads(reviewpath.read_text());assert review['status']=='passed' and review['contract_sha256']==sha(p)
    assert review['state_sha256']==sha(root/'STATE.json') and review['results_sha256']==sha(root/'RESULTS.json')
    results={}
    for case in baseline.cases():
        proof=review['checks'][case['name']]
        results[case['name']],path=baseline.saved_case(proof,baseline.binding(old,case))
        if case['kind']=='http':results[case['name']]['pool']=pool_geometry((Path(path)/'stderr.log').read_text())
    q=Path(c['fp16_qualification']);assert str(q) in c['identity'] and sha(q)==c['identity'][str(q)]
    qualified=json.loads(q.read_text());assert qualified['status']=='qualified_for_measured_scope' and qualified['actual_gufo_sha256']==c['elf_sha256']
    return results


def compare_fp16(kv4,fp16):
    out={}
    for key,new in kv4.items():
        old=fp16[key]
        if key.startswith('http'):
            out[key]=dict(groups={g:dict(fp16=old['groups'][g]['aggregate_emitted_tokens_per_group_wall'],kv4=x['aggregate_emitted_tokens_per_group_wall'],ratio=x['aggregate_emitted_tokens_per_group_wall']/old['groups'][g]['aggregate_emitted_tokens_per_group_wall']) for g,x in new['groups'].items()},kv4_pool=new['pool'],fp16_pool=old['pool'])
        else:out[key]={n:dict(fp16=old['rows'][n]['mean_tps'],kv4=x['mean_tps'],ratio=x['mean_tps']/old['rows'][n]['mean_tps']) for n,x in new['rows'].items()}
    return dict(rows=out,fp16_reexecuted=False,quality_scope='Cross-KV output differences are allowed; numerical quality is a separate collector',pool_policy='Same configured bytes, actual token capacity differs. FP16 sink/tail and MTP draft KV remain; not a 4x whole-memory reduction.')


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True);a=ap.parse_args();c=json.loads(a.contract.read_text())
    review=json.loads(Path(c['review']).read_text());assert review['status']=='reviewed_final_kv4_throughput' and review['contract_sha256']==sha(a.contract)
    out=owned_path(c['out'],KVROOT);assert not out.exists();recorded=False
    state=dict(status='admitting',contract_sha256=sha(a.contract),started=time.time(),case_paths={},reused_cases=[],
        kv_cache_type=KV,prior_qualification_scope='FP16 only; reused smoke proves runtime compatibility, not K4 correctness',
        kv4_quality_is_separate=True,notifications_enabled=False)
    try:
        with ExitStack() as stack:
            for p in (KVROOT/'THROUGHPUT-OWNER.lock',LOCK,JOB):
                f=stack.enter_context(p.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
            pack=check_contract(c);assert available()>=28*GIB and shutil.disk_usage(OUT).free>=150*GIB
            from test.experiments.gufo_upstream_port_20261023.window import host,npu
            out.mkdir();recorded=True;state['status']='running';save(out/'STATE.json',state);deadline=time.monotonic()+3600
            def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
            signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
            groups=sorted({Path('/dev/kfd').stat().st_gid,*[p.stat().st_gid for p in Path('/dev/dri').glob('renderD*')]})
            try:
                results={}
                for case in cases():
                    key=case['name']
                    if key in c['reuse_cases']:
                        results[key],path=saved_case(c['reuse_cases'][key],binding(c,case));state['reused_cases'].append(key)
                    else:
                        path=str(out/key);results[key]=run_case(c,pack,Path(path),case,deadline,host,npu,groups)
                    state['case_paths'][key]=path;save(out/'STATE.json',state)
                    if len(results)==4:save(out/'OFF-BASELINE.json',dict(status='saved_once',case_paths=state['case_paths'].copy(),measurement_binding={x['name']:binding(c,x) for x in cases()[:4]}))
                comparisons=compare(results)
                fp16=validate_baseline(c)
                for mode in ('off','mtp'):
                    for prompt,row in results[f'http-c1-{mode}']['groups']['greedy']['rows'].items():
                        assert signature(row)==signature(results[f'http-c2-{mode}']['groups']['greedy']['rows'][prompt]),'K4 C1/C2 greedy mismatch'
                against_fp16=compare_fp16(results,fp16)
                save(out/'RESULTS.json',dict(status='completed_pending_independent_review',comparisons=comparisons,against_saved_fp16=against_fp16,kv_cache_type=KV,
                    primary='Final emittedtokens/full decode and request/group wall; acceptance is diagnostic only',
                    limitations=['Two repeats may be noisy; observed ratios are not automatic improvement qualification.','HTTP sampled outputs may differ and are not a quality-equivalence check.','Same pool bytes gives different token capacity. FP16 timings are historical retained measurements; cross-KV output identity is not required and this is not a quality proof.']))
                state.update(status='completed_pending_independent_review',result_sha256=sha(out/'RESULTS.json'))
            except BaseException as e:state.update(status='failed',error=repr(e));raise
            finally:
                try:check_contract(c);state['pack_unchanged']=True
                except BaseException as e:state.update(status='protection_failed',protection_error=repr(e))
                state['finished']=time.time();save(out/'STATE.json',state)
            assert state['status']=='completed_pending_independent_review'
    finally:
        if recorded:save(out/'LOCAL-COMPLETION.json',dict(notifications_sent=False,status=state['status'],state_sha256=sha(out/'STATE.json')))

if __name__=='__main__':main()
