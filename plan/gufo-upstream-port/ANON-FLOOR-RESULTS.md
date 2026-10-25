# gufo 원본 이식: anon 바닥의 정체 (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [U4 결과](U4-RESULTS.md), [U4c 결과](U4C-RESULTS.md).

사용자 요청: "이제 남은거 진행해보자". U4 결과에 남은 항목이었던 "예산과 관계없는 anon 바닥 +1.5 GiB"를 측정했다.

## 측정

- 창: `memprobe-a01`. 운영 `210c7633`(운영 플래그), 이식본 U4c `852b6c72`(RAM 예산 2 GiB·8 GiB).
- 구성은 F 창과 같다(256K × 4, 풀 40 GiB).
- 실행기는 `memprobe.py`다(컨테이너 안, `PLAN-M.json`·`PLAN-M-BASE.json`).
  - 준비 10 s 뒤, 그리고 8K 첫 턴 3번 각각의 뒤에 기록한다.
  - 기록 내용: cgroup `memory.stat`, `smaps_rollup`, 매핑별 `smaps` 합.

anon(GiB):

|시점|운영|이식본 2 GiB|이식본 8 GiB|
|---|---|---|---|
|준비 직후(요청 없음)|15.63|19.47|19.81|
|8K 첫 턴 1번 뒤|16.04|19.54|20.76|
|3번 뒤|16.44|19.55|23.63|

- **차이는 요청 전부터 있다(+3.84 GiB, 2 GiB 예산).** 모두 익명 매핑이다. 파일 캐시·shmem은 0이다.
  - 운영은 요청마다 늘어난다(지연 할당). 이식본 2 GiB는 거의 그대로이고, 8 GiB는 예산까지 채워 간다.
- 서버 로그 RSS도 같다. `load_phase=sessions` 직전은 둘 다 18,025 MiB이고, 준비 완료 때 운영 18,097, 이식본 22,026 MiB다.

## 내역 (이식본 − 운영, 준비 직후, 2 GiB 예산)

|항목|크기|출처|
|---|---|---|
|DFlash2 prompt feature pinned 버퍼|1.56 GiB(슬롯 4 × 400 MiB)|원본 #688 `b0eb5d47`. 슬롯을 만들 때 `GetMaxPromptBatch(4096) × target 층 × hidden × FP32`를 pinned host로 잡는다(`QwenGpuExecutor::ReservePromptHiddenCapture`). 로그 `slot_pinned_bytes=1677721600`. 운영은 요청마다 `std::vector`로 잡는다|
|RAM checkpoint backing 미리 커밋|1.77 GiB|원본 `GrowAhead(2 GiB)`. 첫 capture에서 등록 비용을 치르지 않으려고 준비 전에 커밋한다. 로그 `backing_prepared committed_bytes=1899233280`. **RAM 예산 안의 몫**이다(8 GiB 예산이면 2.12 GiB)|
|나머지|약 0.5 GiB|메타데이터·ledger·스레드 등. 나누지 않았다|

- **U4 결과의 "예산과 관계없는 바닥"은 정정한다.**
  - 예산과 관계없는 부분은 DFlash2 pinned 1.56 GiB + 나머지 약 0.5 GiB이다.
  - backing은 예산 중 최대 2 GiB를 시작 때 미리 커밋하는 것이다. 예산을 1 GiB로 낮추면 이 몫도 1 GiB로 줄어든다(f3 1 GiB 18.7 GiB와 맞다).
- 운영의 RAM 예산(8 GiB)은 쓸 때만 할당된다. 이 측정 범위에서 운영 anon은 16.4 GiB를 넘지 않았다.

## 후보 (미결정)

- **DFlash2 버퍼를 슬롯 사이에 나눠 쓰기:** prefill은 한 번에 한 슬롯만 돈다(`physical_execution_width=1`). 이 버퍼를 하나만 두면 약 −1.17 GiB다. 원본 #688의 설계(슬롯마다 따로)를 바꾸는 일이라, 동시 prefill 계획과 함께 원본에 물어야 한다.
- **backing 미리 커밋 줄이기:** 2 GiB를 미리 커밋하면 첫 capture가 빠르다. 줄이면 첫 capture에서 등록 비용(`backing_grown acquire_ms`)이 든다. 측정 뒤 정한다.
- **RAM 예산 결정(1–2 GiB):** 위 내역으로 예상치를 셈할 수 있다.
  - 2 GiB 예산이면 운영 대비 +3.8 GiB다(측정).
  - 1 GiB면 약 +2.8 GiB다.
  - DFlash2 버퍼를 나눠 쓰면 각각 약 1.2 GiB씩 더 줄어든다.

증거: `storage/evaluations/gufo-upstream-port-final-20261023/memprobe-a01/`(창마다 `receipt/run/RESULT.json`·`server-1.log`). cache-disk는 지웠고 `CACHE-REMOVED.json`에 기록했다.
