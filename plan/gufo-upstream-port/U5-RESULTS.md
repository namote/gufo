# gufo 원본 이식 U5: 다시 측정할 항목 (2026-10-23)

상위 문서: [이식 계획](PLAN.md), [U3 결과](U3-RESULTS.md), [U0 기준값](U0-RESULTS.md).

사용자 결정: U4 질문에 "U5 먼저, U4 보류"를 골랐다(U4 측정 근거는 PLAN 단계표).

항목마다 U3(gufo `15f27e88`) 위에서 원본 동작과 비교하고, 이득이 있는 것만 옮긴다. 브랜치는 gufo `ai-tools/port-u5`이다.

## ① 밀려난 대화의 복원 깊이 (B3·Q4) — 채택

**원본 동작 (U3 로그로 확인):**
- checkpoint는 prompt 경계(stable과 full prompt)에만 생긴다.
- 생성이 끝난 live frontier(예: 1,040 prompt + 213 생성 = 1,253)는 슬롯에만 있다.
- 다른 요청이 그 슬롯을 cold로 배정받거나 풀 회수로 비우면 frontier는 사라진다. 다음 턴은 stable checkpoint(1,033)에서 복원하고 답변 부분(약 250토큰)을 다시 prefill한다.

**변경** (`72545ffd`, `18b9e886`, `2bae00dc`):
- 슬롯을 다시 배정하거나 회수하기 직전에, live frontier 64토큰 안쪽에 checkpoint가 없으면 그 슬롯을 정확한 live lease로 잡아 frontier를 RAM checkpoint로 capture한다. 원본 `Lease::Capture`·`PublishSnapshot`을 그대로 쓴다.
- capture는 retry 우선순위로 둔다. 다른 선택적 사본만 밀어낼 수 있고, prompt·공유 checkpoint가 RAM을 원하면 가장 먼저 밀려난다.
- 선택적 동작이다. RAM이 모자라 거절되면 `live_frontier_capture_refused`를 남기고, 원본처럼 cold로 진행한다.
- 디스크 게시는 원본 `Persist` 규칙(min_step)을 따른다.

**측정:**
- capture 1회는 약 4 ms이고 RAM 약 260 MB를 쓴다(대부분 recurrent private state).
- 첫 시도(`72545ffd`, continuation 우선순위)는 RAM 1–3 GiB 창에서 손해였다.
  - capture가 stable·공유 checkpoint를 밀어냈다.
  - 그래서 regress 다음 턴이 디스크 복원이 됐고(0.73 → 0.95 s), shared 팔로워가 checkpoint를 놓쳤다.
  - 그래서 retry 우선순위로 바꿨다.

|창(b01, `18b9e886`)|U3 대비|
|---|---|
|T2|텍스트 21/21 같음, gold 19/19. C-restore 재사용 1,286 → 1,499(frontier 전체), TTFT 0.81 → 0.34 s. 나머지 요청 같음|
|T4s shared|텍스트 19/19 같음. T2·T3 c1 2.90 → 1.31 s, 3.37 → 1.87 s. T1 팔로워 c2·c3 14.6 s는 원본 RAM 예산 경합이다(아래)|
|T4s regress·boundary·two8k·long16k·density4|텍스트 전부 같음, gold 전부 정답. TTFT 합 ±1 % 안. RAM 1–3 GiB에서는 capture가 먼저 밀려나 이득도 손해도 없다|

- **shared T1 팔로워 14.6 s:** 1 GiB 예산에 공유 checkpoint 7,174(715 MB)와 8,201(783 MB)이 함께 들어가지 못해, 7,174 게시가 두 번 `capture_refused byte_capacity`였다. U0 a02·U2 a01과 같은 원본 현상이다. 새 capture는 allocate에서 거절돼 아무것도 밀어내지 않았다.
- **판정:** 채택. RAM에 여유가 있으면 다음 턴이 frontier 전체에서 복원하고, 모자라면 원본과 같다. 운영 RAM 예산(자동 최대 32 GiB)에서는 capture가 남는 쪽이다.

## ② 팔로워 공유 깊이 7,174 (P4) — 문제 아님

- U0 문서는 "system 끝 약 8.2K보다 앞"이라고 적었다. 이것은 해석 오류였다.
- 시나리오의 system prompt는 filler와 지시문으로 약 7.17K 토큰이고, 사용자 메시지(과제 약 1K 토큰, 대화마다 다름)가 뒤따른다. 7,174는 system 경계 그 자체다.
- 원본은 앞 대화와의 공통 prefix(`CommonPrefixTokens`)에 분기점 checkpoint를 잡는다. 그래서 P4가 이미 충족된다. 옮길 것 없음.

## ③ 디스크 복원 속도 (R1) — 채택

**원본 동작:**
- 복원은 chunk 파일(512 KiB)마다 열기 → 읽기 → CRC → `CopyRowsIn` → 완료 대기를 한 스레드에서 차례로 한다.
- helper 스레드 4개가 32 MiB 앞까지 `POSIX_FADV_WILLNEED` 힌트를 준다.
- U3 측정: 16K 재시작 복원은 파일 읽기 679 ms, 전체 856 ms였다.

**호스트 재현** (`test/experiments/gufo_upstream_port_20261023/diskread.py`, 16K checkpoint의 chunk 파일 2,051개, 페이지 캐시를 비운 뒤):

|방식|읽기 스레드 1개|
|---|---|
|원본(WILLNEED helper 4개)|0.65 GB/s|
|helper 8개·창 128 MiB(힌트)|0.69 GB/s|
|`readahead(2)` helper 8개|0.68 GB/s|
|**helper가 scratch로 실제 읽기 4개**|1.94 GB/s|
|**helper가 scratch로 실제 읽기 8개, 창 64 MiB**|4.09 GB/s(CRC 포함 3.5–4.1)|

- 힌트는 파일을 앞서 올려 두지 못했다. 읽기 스레드가 거의 모든 파일에서 기다린다.
- 읽기 스레드 자체를 늘려도 선형으로 빨라졌다(2개 1.2, 4개 2.4, 8개 4.3 GB/s). 하지만 그러려면 staging이 커져야 한다.
  - 원본은 staging을 1 MiB로 고정했다("크기는 속도를 바꾸지 않는다", #562).
  - 원본 테스트는 그 1 MiB가 RAM 상한에 포함된다는 계약을 검사한다.
  - 그래서 이 방법은 쓰지 않았다. 처음 시도한 읽기 병렬화 구현은 되돌렸다.

**변경** (`3e36130a`):
- `PrefetchPayload`에 선택적 scratch 인자를 더했다. scratch를 주면 파일을 `pread`로 읽어 페이지 캐시에 올리고, 없으면 지금처럼 힌트만 준다.
- Prefetcher는 helper 8개, 256 KiB scratch, 창 64 MiB로 바꿨다.
- 읽기 스레드·staging·검증·순서는 그대로다.

|창(`u5c-*-a01`, ①번 빌드 b01 대비)|디스크 복원|
|---|---|
|long16k 재시작 복원 16,379토큰|파일 읽기 679 → 105 ms, 복원 856 → 271 ms, TTFT 2.38 → 1.80 s|
|two8k 8,187토큰|복원 509 → 166 ms, TTFT 1.88 → 1.54 s|
|regress·T2 F-disk 1,033토큰|복원 184 → 80 ms|

- 텍스트는 6개 창(T2, long16k, two8k, regress, shared, density8)에서 전부 같고, gold 전부 정답이다.
- shared c1과 density8의 요청별 TTFT는 RAM 예산 경합과 입장 순서에 따라 오르내렸고, 합은 같거나 줄었다.
- u5c-t4s-shared-a01의 창 판정은 끝난 뒤 GPU busy 4 %로 `FAILED_no_fallback`이다. 기존 패턴이고 guard는 완화하지 않았다.
- **판정:** 채택. 256K로 늘려 잡으면 재시작 복원이 약 10 s에서 3 s대가 된다. 실제 길이는 최종점검 F에서 잰다.

## 남은 항목 (옮기지 않고 기록)

- **입장 예약 보수성:** `prompt + max_tokens` 최악값을 예약한다.
  - 기본 크기 풀(세션 × 전체 문맥)에서는 대기가 생기지 않는다.
  - 운영처럼 명시한 작은 풀에서만 입장이 보수적이다.
  - 생성 도중 블록이 모자라면 기다리거나 선점해야 하므로 설계 변경이 크다. 운영 측정(F) 뒤에 판단한다.
- **원본 RAM 예산 경합:** 1 GiB 예산에서는 공유 checkpoint 둘이 함께 들어가지 못해 팔로워가 놓친다(U0·U2·U5에서 재현). 원본 이슈 후보로 남긴다.
- **Q3(공유 기준 512 → 128):** 원본은 512토큰 미만의 공유 prefix에서 기다리거나 capture하지 않는다. 128–512토큰 prefill은 0.3–1 s라 작은 PR 후보다. 이 길이의 측정 시나리오가 없어 U6 준비 때 잰다.
- **B3:** ①번 capture도 원본 `Persist` 규칙으로 디스크에 게시되므로 축출 시 디스크 쓰기 일부가 생겼다. 별도 idle write-back은 옮기지 않는다. **정정(최종점검 F):** 실제 운영 설정에서는 이 판단이 틀렸다. 마지막 경계가 디스크에 없어 재시작 뒤 TTFT가 운영 대비 2–4배였다([F 결과](F-RESULTS.md)). 이식 대상으로 다시 올린다.
- **P3·Q1(내용 hash 공유):** 결정 D4대로 원본 계보 원칙을 따른다. ②번에서 계보 공유가 system 경계를 정확히 잡는 것을 확인했다.

## 판정

- U5 통과. ①·③을 옮겼다(gufo `ai-tools/port-u5` `3e36130a`). 출력은 모든 창에서 같다.
- 다음은 최종점검 F(실제 길이)이고, 그 결과로 U4 재개 여부와 운영 전환 제안을 정한다.

증거: `storage/evaluations/gufo-upstream-port-u5-20261023/`
