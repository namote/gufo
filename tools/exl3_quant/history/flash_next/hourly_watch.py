"""Hourly local status receipt and a queued follow-up in the existing Codex chat."""
import json
from pathlib import Path
import subprocess
import time

ROOT = Path('/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104')
control = json.loads((ROOT / 'WATCH-CONTROL.json').read_text())
if not control['enabled']:
    raise SystemExit(0)
out = ROOT / 'hourly-watch'
out.mkdir(exist_ok=True)
now = time.time()
snapshot = dict(timestamp=now, thread=control['thread'], states={})
for name in ['long-row-pilot/STATE.json', 'model-job-c03/STATE.json', 'work/ckpt/job.json', 'MODEL-CONVERSION.json']:
    path = ROOT / name
    if path.exists():
        snapshot['states'][name] = json.loads(path.read_text())
snapshot['memory'] = [line for line in Path('/proc/meminfo').read_text().splitlines()
                      if line.startswith(('MemAvailable:', 'SwapFree:'))]
message = ('[사용자가 요청한 Gufo 변환 1시간 점검] '
           '/home/m2on/projects/gufo/plan/exl3/F3-CONVERSION-WATCH.md를 읽고 '
           '현재 c03 변환 진행/로그/메모리/디스크를 확인해라. 정상 실행 중이면 건드리지 말고 '
           '진행을 보고한다. 실패한 경우 원인을 진단·수정·작은 검증 후 기존 승인 범위에서 '
           '체크포인트를 재사용해 이어간다. 실행 중 코드는 수정하지 않고, 다른 GPU 작업은 '
           '중지하지 않는다. 완료 시 마지막 정확성·성능 검증을 묶어서 처리하고 감시를 종료한다. '
           '이후 사용자의 중지·변경 지시가 있으면 그것이 우선이다. 중복 변환을 띄우지 않는다.')
try:
    result = subprocess.run(['/home/m2on/.local/bin/codex', 'queue', '--thread', control['thread'],
                             '--message', message, '--resume'], capture_output=True, text=True, timeout=90)
    snapshot.update(queue_exit=result.returncode, stdout=result.stdout, stderr=result.stderr)
except (OSError, subprocess.TimeoutExpired) as error:
    snapshot.update(queue_exit=1, stdout='', stderr=repr(error))
(out / (time.strftime('%Y%m%d-%H%M%S', time.localtime(now)) + '.json')).write_text(json.dumps(snapshot, indent=2))
print(json.dumps({k: snapshot[k] for k in ('timestamp', 'queue_exit', 'stdout', 'stderr')}), flush=True)
if snapshot['queue_exit']:
    raise SystemExit(snapshot['queue_exit'])
