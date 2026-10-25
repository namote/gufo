# gufo 원본 이식 U4f 마무리: 개선사항 인수와 전체 점검 F16 (2026-10-25)

상위 문서: [이식 계획](PLAN.md), [U4f 결과](U4F-RESULTS.md), [hoist 경계 검증](HOIST-PREFIX-REVIEW.md).

사용자 지시: "오케이 이제 니가 수정해줘 개선사항 턴키 받아서 가자". 제안한 쪽이 만든 최적화·system 경계 커밋을 역검토한 뒤 남은 개선사항을 이 세션이 넘겨받았다.

## 수정 (gufo `ai-tools/port-u4f`)

|커밋|내용|
|---|---|
|`7e3af59e`|복원 중 아직 매핑되지 않은 블록 표시값(`~0`)을 `Release`에 넘기지 않는다(`ReleaseMappedKvBlocks`). 이전에는 allocator의 범위 검사가 우연히 무시했다. `PrepareKvRestoreRows`·`EnsureKvRestoreBlock`·`EnsureKvRows` 문서 보강|
|`d66d5ae7`|디스크 write-back마다 `published` 로그에 `yield_ms`·`stream_wait_ms`·`filesystem_ms`·`io_lock_ms`. queue 완료 콜백이 그 작업의 `TransferTiming`을 받는다|
|`b012bfdb`|RAM·디스크 복원 후보가 실패하면 다음 후보로 가기 전에 `event=restore_failed source= tokens= reason=`을 남긴다. 이전에는 조용히 더 얕은 후보로 넘어갔다|

- 검토 중 제기했던 "`event=system_prefix` 로그가 레벨 확인 없이 포맷된다"는 틀린 지적이었다. `LogFormatted`가 포맷 전에 레벨을 거른다.
- CPU 138/138, clang-format 통과.
- 제안한 쪽의 증거(`tmp/test-results`)는 `storage/evaluations/gufo-upstream-port-final-20261023/handoff-20261025/`로 옮겼다. 201파일과 SHA manifest이며, 5 MiB를 넘는 빌드 산출물 11개는 SHA만 적었다.

## 전체 점검 F16 (`final-f16-a01`, gufo `d66d5ae7`)

- 순서는 f15와 같다. 단위 테스트(vision serving 추가) → 6개 시나리오를 운영 `210c7633`과 번갈아 → T2 → memprobe. 실행 직전 fstrim.
- 시나리오 14창은 모두 COMPLETE. 텍스트는 운영·병합본(f13)·f15와 모든 창에서 같고, T2 21/21도 같다. 서버 로그에 `ERROR`·`logic_error`·`length_error`가 없다.
- 준비 직후 anon은 18.30 GiB로 f13과 같다.

|TTFT 합(초)|운영(같은 run)|병합본 f13|f15|**F16**|
|---|---|---|---|---|
|long32k(8)|247.4|242.2|242.9|244.4|
|256K 첫 턴 포함(long256k)|889.5|875.5|881.4|875.5|
|two8k(8)|59.8|57.3|57.4|57.2|
|regress(20)|26.7|21.3|26.3|21.5|
|boundary(11)|24.4|22.7|24.8|22.8|
|shared(19)|50.9|45.3|49.0|46.6|
|T2(21)|—|65.8|67.0|65.7|

- 단위 테스트 9개 중 `inference_backend_gpu_test` 1개 실패(아래 "남은 문제").

## RAM capture 거절 증가의 원인 (확인)

- `923a4194`부터 흩어진 KV를 RAM에 모으지 않고 디스크 쓰기 때 장치에서 chunk별로 직접 읽는다.
- 새 로그로 보면, 모델이 바쁠 때 이 장치 읽기 대기(`stream_wait_ms`)가 800 MB write-back 한 번에 4–11 s다.
  - 약 1 MB chunk 파일 821개를 하나씩 읽고, 읽을 때마다 GPU 작업 뒤에서 기다린다.
  - 모델이 쉴 때 같은 크기는 2 ms, 파일시스템 시간은 30–90 ms다.
- write-back은 큐 worker 하나가 순서대로 처리한다. 그 사이 뒤에 쌓인 RAM checkpoint 작업이 RAM을 붙잡고, 보호된 원본 때문에 회수할 대상이 없다. 그래서 RAM capture가 `byte_capacity`로 거절된다. 거절은 모두 대기 중인 write-back이 있을 때만 났다(f13·f15·F16 전부).
- F16 regress는 디스크가 빨라 거절 2회(f13 수준)였고, shared는 10회(f15 수준)였다. 둘 다 TTFT는 병합본 편차 안이다.
- 개선 후보: 장치 행 write-back의 다음 chunk 읽기를 미리 걸어 대기를 겹친다(`DiskStore::Publish`의 chunk 파일 순차 루프). 디스크 게시 코드를 건드리는 변경이라 이번에는 하지 않았다.

## 동시 fork 테스트의 간헐 실패 (발견 당시)

- `inference_backend_gpu_test`의 "concurrent fork" 검사: 같은 뿌리에서 갈라진 4개 요청 중 하나가 생성 토큰을 포함한 58토큰 checkpoint 대신 prompt checkpoint(30)에서 복원했다. 출력 토큰은 같고, reuse만 7토큰 짧다(`cached=30 required=37`).
- 반복 결과(같은 이미지 런타임, `build/gpu-test` 바이너리만 바꿈):

|바이너리|실패/실행|
|---|---|
|`d66d5ae7`(F16 단위 + 재실행 4회)|2/5|
|`c0a68ddc`(제안한 쪽 마지막)|0/5|
|`7e3af59e`|0/5|
|`b012bfdb`(실패 로그 추가)|1/8|
|이전 run들(f12–f15, 제안한 쪽 3회)|0/7|

- 이 테스트의 fork backend는 디스크 cache가 없다(`disk_capacity_bytes=0`). 그래서 `d66d5ae7`의 write-back 코드는 이 경로에서 실행되지 않는다. 표본 수로는 우연과 구분되지 않는다.
- `b012bfdb` 로그로 확인한 것: 실패한 요청에 `restore_failed`가 없다. 58토큰 checkpoint는 retention에 들어간 직후였지만, lookup이 그 시점에 후보로 쓸 수 없어 30을 골랐다. 빌린 행 유효성(`Payload::IsValid`)이나 checkpoint 일관성 검사가 잠깐 거짓인 것으로 보인다. 원인 확정은 남았다.

## 후속 (사용자 지시 "오케이 진행해줘")

### 동시 fork 간헐 실패: 원인과 수정 (`b4eec7d4`)

- 원인은 원본(#540) `Request::ShareCheckpoint`의 경계 조건이다.
  - 같은 prompt의 peer가 공유할 수 있는 가장 깊은 위치는 leader의 마지막 prefill 위치(prompt − 1)다.
  - leader의 stable 경계가 바로 거기면, leader는 그 위치에 멈춰 복사(capture)가 끝나기를 기다린다.
  - `if (target <= offset) return 0`이 "복사 중이거나 막 게시된 경계를 재사용"하는 분기보다 먼저 있어서, 그 순간 도착한 peer는 대기하지 않고 prompt checkpoint(30)에서 복원했다.
- 수정: `target < offset`일 때만 먼저 0을 돌려준다. `target == offset`은 복사 중·게시 분기를 거친 뒤에 판단한다.
- 새 CPU 테스트(stable 경계 = prompt − 1)는 옛 코드에서 실패하고 수정본에서 통과한다.
- GPU: 수정 바이너리로 `inference_backend_gpu_test` 10/10 통과(`final-f16-fork-a06`). 수정 전 HEAD 계열은 3/13 실패였다.
- 원본에도 있는 문제라 PR 후보다.

### 장치 행 write-back 읽기 겹치기 (`663d50c9`)

- 디스크 쓰기가 장치에서 읽는 piece(chunk 파일 하나 = 약 1 MB)를 다음 8개까지 미리 복사해 둔다. 순서는 disk store가 파일을 쓰는 순서(공유 먼저, 파일 ID 순)다.
- 별도 pinned 영역 8 MiB를 transfer staging으로 원장에 잡는다. RAM 상한의 1/256 이하일 때만 켜고(2 GiB 이상), 작은 cache는 이전처럼 한 piece씩 읽는다. `configured` 로그의 `disk_read_ahead_bytes`에 남긴다.
- 미리 읽기는 다른 stream이 하나 남을 때만 건다. 제자리 읽기가 stream을 못 얻으면 가장 오래된 미리 읽기를 버퍼로 정착시켜 stream을 돌려받는다. 어떤 경로든 끝나기 전에 미리 걸린 복사를 모두 기다린다.
- CPU 138/138(새 테스트: 장치 piece가 둘 이상 동시에 걸리고 모두 정착, 왕복 복원 동일).
- `final-f17-a01`: 단위 9개 모두 PASS(동시 fork·vision 포함). 이어서 regress·shared를 f16(`d66d5ae7`)과 두 번씩 번갈아 돌렸다.

|창|RAM capture 거절|write-back 1회 최대 장치 대기|write-back 완료|디스크 복원|TTFT 합|
|---|---|---|---|---|---|
|f16 shared (2회)|5 / 7|9.6 / 11.6 s|10 / 10|5 / 4|44.9 / 45.4|
|f17 shared (2회)|2 / 3|4.4 / 4.9 s|12 / 11|6 / 6|45.1 / 45.1|
|f16 regress (2회)|2 / 1|0.6 / 1.6 s|13 / 13|6 / 6|21.2 / 21.2|
|f17 regress (2회)|2 / 2|0.6 / 0.4 s|13 / 13|6 / 6|21.8 / 21.3|

- 텍스트는 모든 창에서 같다.
- shared에서 거절이 절반 이하로 줄었고, 끝까지 쓰인 write-back과 디스크 복원이 늘었다. TTFT는 편차 안에서 같다.
- 장치 대기 합은 shared에서 여전히 24–31 s다(f16 37 s). 더 깊은 미리 읽기나 우선순위 stream은 다음 후보다.

## 증거

- `storage/evaluations/gufo-upstream-port-final-20261023/final-f16-a01`(cache-disk 91.4 GiB 삭제, `CACHE-REMOVED.json`), `final-f16-fork-a02`–`a06`, `final-f16-fork-a01.log`(출력 경로 오류로 실행 전 종료), `final-f17-a01`(cache-disk 63.8 GiB 삭제).
- 실행 계획: `test/experiments/gufo_upstream_port_20261023/PLAN-F16.json`(`f16-units`·시나리오·`f16-fork*`), `PLAN-F17.json`, `final.py` set `f16`·`f17`.

## 다음

- 커밋 정리·PR 나누기·`--kv-pool-bytes` 기본값은 결정·반영했다([PR 계획](PR-PLAN.md)). D2 공개는 미정.
- 원본에도 있는 `ShareCheckpoint` 경계 문제는 별도 PR 후보.
- 장치 대기 추가 감소(미리 읽기 깊이·stream 우선순위)는 필요할 때 shared A/B로.
