"""One full-MTP contract, then durable completion/failure notice to current chat.

No next-phase launch, boundary release, GPU operation or artifact cleanup. A
systemd caller should use Restart=no and a stop grace period sufficient for the
existing runner's owned-child cleanup. WATCH control remains authoritative.
"""
import argparse,fcntl,hashlib,json,os,signal,subprocess,sys,time
from pathlib import Path

TASK=Path(__file__).resolve().parent
OUT=Path('/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):
    temp=Path(path).with_suffix('.tmp');temp.write_text(json.dumps(value,indent=2));temp.replace(path)
def watch_enabled(path):return json.loads(Path(path).read_text())['enabled'] is True

def launch_runner(command,logfile):
    """Forward termination only to our runner; it owns GPU cleanup itself."""
    child=None;received=[]
    def stop(signum,frame):
        received.append(signum)
        if child is not None and child.poll() is None:child.send_signal(signal.SIGTERM)
    original={sig:signal.getsignal(sig) for sig in (signal.SIGTERM,signal.SIGINT)}
    for sig in original:signal.signal(sig,stop)
    try:
        child=subprocess.Popen(command,stdout=logfile,stderr=subprocess.STDOUT)
        if received:child.send_signal(signal.SIGTERM)
        code=child.wait()
        return {'returncode':code,'received_signals':received,'runner_pid':child.pid}
    finally:
        for sig,handler in original.items():signal.signal(sig,handler)

def execute_once(contract_path,receipt,*,watch_path,runner_path,run=launch_runner,send=None,pause=time.sleep):
    """Injectable CPU-tested dispatch. Never retries or chains runner execution."""
    contract_path=Path(contract_path);receipt=Path(receipt);runner_path=Path(runner_path)
    receipt.mkdir(parents=True,exist_ok=False)
    state={'status':'starting','started':time.time(),'pid':os.getpid(),
           'wrapper_sha256':sha(__file__),'contract_path':str(contract_path),'runner_path':str(runner_path)}
    save(receipt/'STATE.json',state)
    try:
        if not watch_enabled(watch_path):
            state.update(status='stopped_by_user',runner_started=False)
            return 0
        contract=json.loads(contract_path.read_text())
        assert contract['status']=='reviewed_for_mtp_run' and contract['phase']=='full'
        assert contract['script']=='mtp_calibration_candidate.py'
        argv=contract['argv'];stage=argv[argv.index('--stage')+1]
        assert stage in ('input','decoder')
        runner_state_path=Path(contract['receipt_dir'])/'STATE.json'
        assert not runner_state_path.exists(),'Runner receipt already exists; no repeat of a completed/failed contract'
        state.update(contract_sha256=sha(contract_path),runner_sha256=sha(runner_path),stage=stage,
                     runner_state_path=str(runner_state_path),runner_started=False)
        command=[sys.executable,'-u',str(runner_path),'--contract',str(contract_path)]
        state.update(status='running',command=command,runner_started=True);save(receipt/'STATE.json',state)
        with (receipt/'runner.log').open('x') as logfile:
            result=run(command,logfile)
        state['runner_result']=result
        if runner_state_path.exists():state['runner_state']=json.loads(runner_state_path.read_text())
        runner_state=state.get('runner_state',{})
        success=(result['returncode']==0 and runner_state.get('status')=='completed_pending_review'
                 and runner_state.get('cleanup_errors')==[] and runner_state.get('returned_KFD_contexts')==[]
                 and runner_state.get('protected_unchanged') is True)
        state['status']='completed_pending_independent_review' if success else 'failed_pending_independent_review'
        return 0 if success else 1
    except BaseException as error:
        state.update(status='failed_pending_independent_review',error=repr(error))
        return 1
    finally:
        state['finished']=time.time()
        try:enabled=watch_enabled(watch_path)
        except Exception as error:
            enabled=False;state['watch_error']=repr(error)
        if not enabled:
            state['notification_status']='suppressed_by_user_stop_or_unreadable_watch'
            # Retain runner failure/cleanup evidence without claiming success.
            if state.get('runner_started'):state['user_stop_observed']=True
        else:
            # Commit termination before queueing the follow-up; the follow-up can
            # immediately open this stable STATE even if transport later fails.
            state['notification_status']='pending';save(receipt/'STATE.json',state)
            message=(f"[사용자가 승인한 MTP 전체 보정 {state.get('stage','unknown')} 단계 종료] "
                '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md와 '
                f"{receipt/'STATE.json'}, {state.get('runner_state_path','runner 상태 생성 전 실패')}, "
                f"계약 {contract_path}를 읽고 독립 리뷰하라. GPU cleanup/원본 idx51 보호와 완료 산출물을 확인하고 "
                '기존 H·packed·정상 참조를 재사용하라. Gufo FP32 전체 HC 정규화, 이전 hidden과 다음 토큰 정렬, '
                '2309행·1997187쌍·hidden H 7988748 기준이다. 실패한 작은 검사만 수정하며 중복 GPU/전체 baseline 반복 금지. '
                'FP32/FP16 및 초안 정확도에 따른 배포 선택은 마지막 전체 추론 tokens/s 비교로 판단한다. '
                '이 래퍼는 다음 단계 실행이나 예약 락 해제를 하지 않았다. 최신 사용자 중지·변경 지시가 우선이다.')
            attempts=[]
            for attempt in range(3):
                try:
                    if not watch_enabled(watch_path):
                        result={'status':'disabled_by_user'}
                    else:
                        if send is None:
                            from mtp_calibration_boundary import notify
                            send=notify
                        result=send(message)
                except Exception as error:result={'error':repr(error)}
                attempts.append(result)
                save(receipt/'FOLLOWUP.json',{'attempts':attempts,'delivered':result.get('exit')==0,'message':message})
                if result.get('exit')==0:
                    state['notification_status']='delivered';break
                if result.get('status')=='disabled_by_user':
                    state['notification_status']='suppressed_by_user_stop';break
                if attempt<2:pause(5)
            else:state['notification_status']='delivery_failed_runner_not_repeated'
        save(receipt/'STATE.json',state)

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--contract',type=Path,required=True)
    ap.add_argument('--receipt-dir',type=Path,required=True);a=ap.parse_args()
    root=OUT/'mtp-calibration-callback';root.mkdir(exist_ok=True)
    receipt=a.receipt_dir.resolve();assert receipt!=root and root.resolve() in receipt.parents
    with (root/'OWNER.lock').open('a') as owner:
        fcntl.flock(owner,fcntl.LOCK_EX|fcntl.LOCK_NB)
        return execute_once(a.contract.resolve(),receipt,watch_path=OUT/'WATCH-CONTROL.json',runner_path=TASK/'mtp_calibration_job.py')

if __name__=='__main__':sys.exit(main())
