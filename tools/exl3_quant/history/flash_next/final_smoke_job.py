"""One reviewed production-image smoke window, no deployment/default changes.

Off/on HTTP text+image+cache and 32 emitted-token bench hashes. No quality oracle
or throughput claim. All requests/logs retained; any failure stops the window.
"""
import argparse,fcntl,hashlib,json,os,re,shutil,signal,socket,subprocess,threading,time,uuid
from contextlib import ExitStack
from pathlib import Path
from run_stage import OUT,TASK,LOCK,available,save
from mtp_calibration_job import sha,held_flock,owned_path
from mtp_final_compile_job import stat_id
from owned_cleanup import cleanup_owned
from final_validation_http import run_suite,signature

IMAGE='sha256:e7a761a40187efb13515a0275362ab09d084a129f4544aa4300ec5cccb8c7468'
ELF='62f0c59a15f3aee3aeb5336ab8e85a15406629dc89bc1cdce95d9a95672a3341'
LD='/nix/store/88jjsz00km321bcl9851aiwwdig79b4l-rocm-runtime-10.1/lib'
ROOT=OUT/'final-validation';GIB=2**30;JOB=OUT/'model-job-c03/JOB.lock'


def pack_stats(pack):return {str(p.relative_to(pack)):stat_id(p) for p in sorted(pack.rglob('*')) if p.is_file()}


def check_contract(c):
    required={str(TASK/n) for n in ['final_smoke_job.py','test_final_smoke_job.py','final_validation_http.py',
       'run_stage.py','mtp_calibration_job.py','mtp_stage_disk_budget.py','mtp_final_compile_job.py',
       'owned_cleanup.py','conversion_admission.py','conversion-desktop-contract.json']}
    required.add(str(TASK.parent/'gufo_upstream_port_20261023/window.py'))
    required|={c['cpu'],c['assembly_review'],c['image_fixture']}
    assert required<=c['identity'].keys()
    for p,d in c['identity'].items():assert sha(p)==d,p
    pack=Path(c['pack']);assert pack.is_dir() and pack==pack.resolve()
    assert pack_stats(pack)==c['pack_stats'],'Final pack changed'
    compiled=json.loads((pack/'GUFO-COMPILE.json').read_text())
    assert compiled['status']=='compiled_pending_final_quality_and_performance_validation'
    assert compiled['vision']['payload_exact'] and compiled['mtp_payload_exact'] and compiled['mtp_projection_count']==1546
    assert sha(pack/'GUFO-COMPILE.json')==c['compile_receipt_sha256']
    assert c['image']==IMAGE and c['elf_sha256']==ELF
    actual=subprocess.run(['docker','image','inspect',IMAGE,'--format','{{.Id}}'],capture_output=True,text=True,check=True,timeout=15).stdout.strip()
    assert actual==IMAGE
    return pack


def command(pack,out,name,label,argv,groups,port=None):
    cmd=['docker','run','--rm','--pull','never','--name',name,'--label','gufo.quant.run='+label,
         '--read-only','--user','1000:1000','--memory','100g','--memory-swap','100g',
         '--device','/dev/kfd','--device','/dev/dri','--ipc','host','--tmpfs','/tmp:rw,size=1g,mode=1777',
         '-e','HOME=/tmp','-e','LD_LIBRARY_PATH='+LD,
         '--mount',f'type=bind,src={pack},dst=/models/fnx,readonly',
         '--mount',f'type=bind,src={out},dst=/receipt']
    if port is None:cmd+=['--network','none']
    else:cmd+=['--network','bridge','--publish',f'127.0.0.1:{port}:8080']
    for group in groups:cmd+=['--group-add',str(group)]
    return cmd+['--entrypoint','/opt/gufo-native/bin/gufo',IMAGE,*argv]


def speculative(mode):
    assert mode in ('off','mtp')
    return ['--speculative',mode]+(['--mtp-model','/models/fnx','--draft-tokens','7'] if mode=='mtp' else [])


def server_args(mode,context=4096,concurrency=1,pool=256*2**20):
    return ['serve','llm','-m','/models/fnx','--served-model-name','gufo-final-c03','--host','0.0.0.0','--port','8080',
            '--sessions',str(concurrency),'--context',str(context),'--kv-cache-type','fp16','--kv-pool-bytes',str(pool),
            '--cache-ram-bytes',str(GIB),'--cache-tail-checkpoints','1','--think','off',
            '--temperature','0','--top-p','1','--top-k','0','--min-p','0','--repeat-penalty','1','--seed','314159',
            '--request-timeout-ms','180000','--log-progress','--trace','/receipt/trace.jsonl',*speculative(mode)]


def bench_args(mode,n=32,depth='0',repetitions=1):
    # Explicit -n with no -p selects generation only; avoid a meaningless pp0 row.
    return ['bench','-m','/models/fnx','-n',str(n),'-d',depth,'-r',str(repetitions),'--kv-cache-type','fp16',
            '--temperature','0','--top-p','1','--top-k','0','--min-p','0','--repeat-penalty','1','--seed','314159','--verbose',*speculative(mode)]


def token_hashes(log):
    rows=re.findall(r'Qwen3\.8-Flash-Next tg depth=(\d+).*?output_sha256=([0-9a-f]{64})',log)
    assert rows,'No actual emitted-token hash in production benchmark'
    return rows


def draft_counters(log):
    rows=re.findall(r'Qwen3\.8-Flash-Next tg depth=(\d+) cycles=(\d+) drafted=(\d+) accepted=(\d+)',log)
    assert rows,'Missing actual draft/verify evidence'
    return [{'depth':int(d),'cycles':int(c),'drafted':int(p),'accepted':int(a)} for d,c,p,a in rows]


def guard(deadline):
    assert time.monotonic()<deadline,'Smoke window deadline'
    assert json.loads((OUT/'WATCH-CONTROL.json').read_text())['enabled'],'Latest user stop'
    assert available()>=8*GIB,'Host memory reserve below8GiB'
    assert shutil.disk_usage(OUT).free>=100*GIB,'Disk reserve below100GiB'
    locks=Path('/proc/locks').read_text()
    for p in (LOCK,JOB):assert held_flock(locks,os.getpid(),p.stat())


def runtime_maps(name,label,out):
    r=subprocess.run(['docker','inspect',name],capture_output=True,text=True,timeout=10)
    if r.returncode:return False
    d=json.loads(r.stdout)[0];assert d['Config']['Labels']['gufo.quant.run']==label
    pid=d['State']['Pid']
    if not pid:return False
    try:text=Path(f'/proc/{pid}/maps').read_text()
    except FileNotFoundError:return False
    if 'libamdhip64' not in text or 'libhsa-runtime64' not in text:return False
    assert '/opt/rocm' not in text,'Unintended ROCm runtime'
    libs=[line for line in text.splitlines() if any(x in line for x in ('libamdhip64','libhsa-runtime64','librocblas','libhipblaslt'))]
    assert all('/nix/store/' in line for line in libs)
    assert any('rocm-runtime-10.1' in line for line in libs)
    with Path(f'/proc/{pid}/exe').open('rb') as f:
        h=hashlib.file_digest(f,'sha256').hexdigest()
    assert h==ELF,'Running ELF differs from production baseline'
    (out/'runtime-maps.txt').write_text(text)
    save(out/'RUNTIME.json',{'pid':pid,'elf_sha256':h,'image':IMAGE,'mapped_libraries':libs,'no_gguf_map':'.gguf' not in text})
    assert '.gguf' not in text,'Unexpected GGUF dependency'
    return True


def run_case(c,pack,out,argv,deadline,host,npu,groups,port=None,http=None):
    out.mkdir();name='gufo-final-test-'+uuid.uuid4().hex[:12];label=uuid.uuid4().hex
    cmd=command(pack,out,name,label,argv,groups,port);case=dict(status='running',command=cmd,name=name,label=label,started=time.time())
    save(out/'STATE.json',case);child=None;thread=None;result={};mapped=False
    def requests():
        try:
            url=f'http://127.0.0.1:{port}'
            import urllib.request
            until=time.monotonic()+300
            while True:
                assert child.poll() is None,'Owned server exited before ready'
                assert time.monotonic()<until,'Server readiness deadline'
                try:
                    with urllib.request.urlopen(url+'/ready',timeout=2) as r:
                        if r.status==200:break
                except Exception:time.sleep(.5)
            result['value']=run_suite(url,out/'http',c['image_fixture'],http['phase'],http['concurrency'])
        except BaseException as e:result['error']=repr(e)
    try:
        with (out/'run.log').open('x') as log:
            child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT)
            if http:
                thread=threading.Thread(target=requests,daemon=True);thread.start()
            while child.poll() is None:
                guard(deadline);assert not npu(),'NPU owner appeared'
                if not mapped:mapped=runtime_maps(name,label,out)
                case['minimum_available_bytes']=min(case.get('minimum_available_bytes',available()),available())
                case['updated']=time.time();save(out/'STATE.json',case)
                if thread and not thread.is_alive():break
                time.sleep(1)
            if http:
                assert not thread.is_alive(),'Owned server exited during requests'
                assert 'error' not in result,result.get('error')
                case['http_summary']=result['value']
            else:assert child.returncode==0,f'Benchmark exit {child.returncode}'
            assert mapped,'No verified production process/runtime map'
            case['status']='passed'
    except BaseException as e:case.update(status='failed',error=repr(e));raise
    finally:
        if child is not None:
            case.update(cleanup_owned(name,label,child,host))
            if case['cleanup_errors'] or case['returned_KFD_contexts']:case['status']='cleanup_failed'
        if thread:thread.join(timeout=15)
        if thread and thread.is_alive():case['status']='http_thread_cleanup_failed'
        case['finished']=time.time();save(out/'STATE.json',case)
    assert case['status']=='passed',case
    return case


def main():
    p=argparse.ArgumentParser();p.add_argument('--contract',type=Path,required=True);a=p.parse_args();c=json.loads(a.contract.read_text())
    assert c['status']=='reviewed_final_smoke' and c['seconds']==1200
    r=json.loads(Path(c['review']).read_text());assert r['contract_sha256']==sha(a.contract) and r['status']=='reviewed_final_smoke'
    cpu=json.loads(Path(c['cpu']).read_text());assert cpu['status']=='pass' and cpu['gpu_used'] is False
    for f,d in cpu['identity'].items():assert sha(f)==d
    out=owned_path(c['out'],ROOT);assert not out.exists()
    ROOT.mkdir(exist_ok=True)
    with ExitStack() as stack:
        for path in (ROOT/'OWNER.lock',LOCK,JOB):
            f=stack.enter_context(path.open('a'));fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB)
        assert available()>=28*GIB and shutil.disk_usage(OUT).free>=150*GIB
        pack=check_contract(c)
        from test.experiments.gufo_upstream_port_20261023.window import host,npu,quiet
        from conversion_admission import wait_conversion_admission
        assert not npu();_,admission=wait_conversion_admission(host,quiet,TASK/'conversion-desktop-contract.json')
        assert not host()['KFD_contexts'] and not npu()
        # This admission is a correctness smoke, never a fair timing claim.
        out.mkdir();state={'status':'running','admission':admission,'timing_scope':'smoke_only_not_throughput','contract_sha256':sha(a.contract)}
        save(out/'STATE.json',state);deadline=time.monotonic()+1200
        def stop(sig,frame):raise KeyboardInterrupt(f'signal {sig}')
        signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
        groups=sorted({Path('/dev/kfd').stat().st_gid,*[x.stat().st_gid for x in Path('/dev/dri').glob('renderD*')]})
        with socket.socket() as s:s.bind(('127.0.0.1',0));port=s.getsockname()[1]
        try:
            summaries={};hashes={};drafts={}
            for mode in ('off','mtp'):
                case=run_case(c,pack,out/f'http-{mode}',server_args(mode),deadline,host,npu,groups,port,{'phase':'smoke','concurrency':1})
                summaries[mode]=case['http_summary']
                proposed=sum(r['usage']['draft_tokens'] for r in summaries[mode]['greedy'].values())
                assert (proposed==0 if mode=='off' else proposed>0),'HTTP actual MTP mode mismatch'
            for key in summaries['off']['greedy']:
                assert signature(summaries['off']['greedy'][key])==signature(summaries['mtp']['greedy'][key]),'HTTP greedy mismatch: '+key
            for mode in ('off','mtp'):
                d=out/f'tokens-{mode}';run_case(c,pack,d,bench_args(mode),deadline,host,npu,groups)
                hashes[mode]=token_hashes((d/'run.log').read_text());assert len(hashes[mode])==1
                drafts[mode]=draft_counters((d/'run.log').read_text())
                assert all((r['drafted']==0 if mode=='off' else r['drafted']>0) for r in drafts[mode]),'Benchmark actual MTP mode mismatch'
            assert hashes['off']==hashes['mtp'],'Exact greedy emitted-token IDs differ'
            state.update(status='passed_pending_independent_review',token_hashes=hashes,draft_counters=drafts,
                         http_greedy_equal=True,vision_and_cache_pass=True,quality_and_throughput_pending=True)
        except BaseException as e:state.update(status='failed',error=repr(e));raise
        finally:
            try:check_contract(c);state['pack_unchanged']=True
            except BaseException as e:state.update(status='protection_failed',protection_error=repr(e))
            state['finished']=time.time();save(out/'STATE.json',state)
        assert state['status']=='passed_pending_independent_review'


def callback():
    import sys
    try:
        path=Path(sys.argv[sys.argv.index('--contract')+1]);c=json.loads(path.read_text());out=Path(c['out'])
        if not (out/'STATE.json').exists():return
        control=json.loads((OUT/'WATCH-CONTROL.json').read_text())
        if not control['enabled']:return
        message=f'[Gufo 최종 smoke 종료] {out}/STATE.json 및 각 case 결과를 독립 검토하라. 원본 보호/cleanup/HTTP text-count/bench emitted-token SHA/실제 draft 활성/vision/cache를 확인하고 통과하면 승인된 품질·성능 검증을 이어가라. 실패면 저장 증거로 진단하고 전체 baseline 반복 및 자동 서비스 실행 금지.'
        r=subprocess.run(['/home/m2on/.local/bin/codex','queue','--thread',control['thread'],'--message',message,'--resume'],capture_output=True,text=True,timeout=45)
        save(out/'FOLLOWUP.json',dict(exit=r.returncode,stdout=r.stdout,stderr=r.stderr))
    except BaseException as error:
        if 'out' in locals() and out.exists():save(out/'FOLLOWUP.json',{'error':repr(error)})


if __name__=='__main__':
    try:main()
    finally:callback()
