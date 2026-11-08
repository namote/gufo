"""CPU-only mocked one-shot runner and chat transport; no real subprocess/notify."""
import json,tempfile
from pathlib import Path
from mtp_calibration_callback import execute_once
checks=[]
with tempfile.TemporaryDirectory() as tmp:
    root=Path(tmp);watch=root/'watch.json';runner=root/'runner.py';runner.write_text('pass\n')
    def enabled(value):watch.write_text(json.dumps({'enabled':value}))
    def case(name,*,fail=False,disabled=False,disable_after=False,transport_fail=False,raise_runner=False):
        contract=root/f'{name}.json';runreceipt=root/f'{name}-runner';receipt=root/name
        contract.write_text(json.dumps({'status':'reviewed_for_mtp_run','phase':'full','script':'mtp_calibration_candidate.py',
          'argv':['--stage','decoder'],'receipt_dir':str(runreceipt)}))
        enabled(not disabled);runs=[];sent=[]
        def run(command,log):
            runs.append(command)
            if raise_runner:raise OSError('mock launch failed')
            runreceipt.mkdir();(runreceipt/'STATE.json').write_text(json.dumps({'status':'failed' if fail else 'completed_pending_review',
                'cleanup_errors':[],'returned_KFD_contexts':[],'protected_unchanged':True}))
            if disable_after:enabled(False)
            return {'returncode':1 if fail else 0}
        def send(message):sent.append(message);return {'exit':1 if transport_fail else 0}
        rc=execute_once(contract,receipt,watch_path=watch,runner_path=runner,run=run,send=send,pause=lambda _:None)
        state=json.loads((receipt/'STATE.json').read_text())
        assert len(runs)==(0 if disabled else 1)
        assert not any('RELEASE' in str(p) for p in root.glob('*'))
        if disabled or disable_after:assert not sent and state['notification_status'].startswith('suppressed')
        else:
            assert len(sent)==(3 if transport_fail else 1)
            assert str(receipt/'STATE.json') in sent[0] and '전체 추론 tokens/s' in sent[0]
        if not disabled:assert rc==(1 if fail or raise_runner else 0)
        return state,receipt,contract
    state,receipt,contract=case('success');assert state['status']=='completed_pending_independent_review'
    try:execute_once(contract,receipt,watch_path=watch,runner_path=runner,run=lambda *a:(_ for _ in ()).throw(AssertionError('repeat')))
    except FileExistsError:pass
    else:raise AssertionError('Overwrote callback receipt')
    checks.append('One contract runs once; completed evidence and current-chat callback paths saved; receipt replay rejected')
    assert case('failed',fail=True)[0]['status']=='failed_pending_independent_review'
    assert case('launchfailed',raise_runner=True)[0]['status']=='failed_pending_independent_review'
    checks.append('Runner failure and spawn failure both retain durable failure receipt and notify')
    assert case('stopped',disabled=True)[0]['runner_started'] is False
    assert case('stopduring',fail=True,disable_after=True)[0]['user_stop_observed'] is True
    checks.append('Disabled WATCH prevents initial launch/notify and suppresses post-stop callback; runner retains stop cleanup responsibility')
    assert case('transport',transport_fail=True)[0]['notification_status']=='delivery_failed_runner_not_repeated'
    checks.append('Transport retries at most3; runner never retried or next phase launched')
print(json.dumps({'status':'cpu_mock_pass','checks':checks,'GPU_used':False,'real_subprocess':False,'real_notification':False}))
