# gufo 원본 이식 U3: 공유 KV 블록 풀 (2026-10-23)

상위 문서: [이식 계획](PLAN.md), [API 검토](API-REVIEW.md), [U2 결과](U2-RESULTS.md). 사용자 지시: "오케이~ 추천안으로 가자"(D6 절충안), "넹~"(U3 시작).

## 구현 (gufo `ai-tools/port-u3`, U2 `7daf08d4` 위 커밋 2개, 최종 `15f27e88`)

|커밋|내용|
|---|---|
|`033c2816`|component-cache 슬롯이 공유 풀 하나에서 256토큰 블록을 필요할 때 받는다. 입장 예약·대기·회수|
|`15f27e88`|수정: 풀이 모자라 기다리는 동안 매 스케줄러 패스가 이미 비운 유휴 슬롯을 다시 리셋하던 것(아래 "기록할 것")|

- **풀:** runner가 기동 때 KV 블록 풀 하나를 만든다(plane 2개, K·V). 모든 슬롯의 executor가 같은 풀을 받는다.
  - 크기(결정 D6 절충안): `--kv-pool-bytes`를 주면 그 값 그대로, 주지 않으면 세션 수 × 전체 문맥이다(세션이 원래 쥐던 메모리와 같음). 한 세션의 전체 문맥이 안 들어가는 크기는 기동 때 거부한다.
  - 슬롯 KV는 세션 claim에서 빠지고(`EstimateMemoryUsage(..., shared_kv)`), 풀은 `ResourceClaim`이 빈 메모리를 읽기 전에 잡힌다.
  - `gufo bench`와 CLI는 component cache를 쓰지 않으므로 지금처럼 자기 풀(전체 문맥)을 쓴다.
- **필요할 때 매핑:** 공유 풀 arena는 빈 테이블로 시작한다.
  - 쓰기 전: 원본 hook `BeforeWrite(first, end)`에서 `EnsureKvRows(end)`가 모자란 블록을 받는다. prefill·decode·batched decode·검증이 모두 이 hook을 거친다. graph capture는 hook 뒤에 시작하므로 테이블 업로드가 graph에 들어가지 않는다.
  - 복원 전: `BeginRestore`가 목표 행 수에 맞춰 남는 블록을 돌려주고 모자란 블록을 받는다.
  - 리셋: arena `Reset`이 블록을 모두 돌려준다(stream drain 뒤).
  - 모자라면 `std::length_error`로 그 pass를 쓰기 전에 거부한다. 세션은 그대로 쓸 수 있다.
  - 매핑 안 된 테이블 칸은 블록 0을 가리킨다. 가려진 읽기가 풀 밖으로 나가지 않게 하기 위해서다.
- **입장 예약** (allocator의 `KvBlockReservation`, CPU에서 검사):
  - 스케줄러가 `Acquire` 전에 `AdmissionReady(prompt, max_tokens)`를 본다. 필요한 블록 = (prompt + max_tokens)/256 올림 + 1(speculative 검증 행), 문맥 상한.
  - 풀이 모자라면 `ComponentTextCache::ReclaimRows`가 유휴 슬롯의 행을 돌려받는다. 순서는 live frontier 없는 슬롯 → prompt가 이어 쓰지 않는 live frontier(checkpoint로 덮인 것 먼저, 오래된 것 먼저)다. 돌려받는 것은 cold 재배정과 같아서, 빌린 checkpoint 행은 원본대로 RAM에 보존된다.
  - 그래도 모자라면 다른 요청이 실행 중일 때 기다린다. 아무것도 실행 중이 아니면 입장시키고, `ReserveRows`가 요청을 이름으로 거부한다.
  - `Acquire` 뒤 `ReserveRows`가 (필요 − 이미 매핑된 블록)을 예약한다. 매핑은 예약에서 먼저 꺼낸다. commit·cancel·invalidate 때 남은 예약을 돌려준다.
- **원래 계획에서 U4로 넘긴 것: checkpoint의 블록 참조(Retain)와 CoW.**
  - U3에서는 블록 하나의 주인이 늘 슬롯 하나다. 그래서 CoW 없이도 맞다.
  - 원본 checkpoint chunk는 빌릴 때부터 RAM backing을 예약하고, 덮어쓰기·해제 전에 RAM으로 보존한다. 이 흐름은 그대로 두었다.
  - chunk를 장치 블록 참조로 바꾸는 이득은 다른 슬롯 복원을 복사 대신 블록 매핑으로 할 때(U4) 생긴다. 그래서 Retain·CoW·매핑 복원을 U4에서 함께 한다.
- **로그:** `event=kv_pool`(기동), `event=kv_rows_reserved`/`kv_rows_released`(요청마다), `event=live_frontier_dropped reason=row_capacity`(회수)
- **테스트:**
  - CPU `kv_block_layout_test`: 예약이 다른 할당을 막는지, 예약 먼저 쓰기, 거부 시 아무것도 안 잡기, 이동·해제 반환, 다른 풀 예약 거부
  - CPU `component_text_cache_test` `PagedRowsAdmitWaitAndReclaim`: 가짜 runner로 예약·반환, 유휴 frontier 회수, 실행 중 요청이 있으면 대기, 대기 패스가 비운 슬롯을 다시 리셋하지 않음, 이어 쓰는 frontier는 남김, 어떤 풀에도 안 들어가는 요청은 이름으로 거부
  - GPU 어댑터 테스트 `shared_kv_pool`: 40블록 풀(문맥 2개분보다 작음)에서 두 슬롯이 번갈아 써 흩어진 블록을 받는다. 자기 풀 세션과 logits·decode 토큰(블록 경계 4608 포함)·batched decode 16단계(경계 512 포함)·capture 바이트·다른 슬롯 복원이 bit 단위로 같아야 한다. 풀 부족 거부, 예약, 리셋 반환도 본다.
- **패치 보존:** `test/experiments/gufo_upstream_port_20261023/patches/u3/`(2개)

## 검증 (PLAN-U3.json)

|계층|결과|
|---|---|
|T0|릴리스·gpu-test·cpu-test 빌드 성공. 릴리스 경고는 원본 U0 빌드와 같은 61종(새 경고 없음). 포맷 PASS(clang-format 21.1.8). CPU ctest 127/127(예약·회수 새 검사 포함)|
|T1 커널(`u3-kernels-a01`)|10종 통과. 커널은 바뀌지 않았고, arena 쪽 변경 뒤 회귀 확인용|
|모델 단위(`u3-models-b01`, 최종 빌드)|5종 통과. 어댑터 테스트의 새 `shared_kv_pool` 경우 PASS: 흩어진 블록에서 단일·batched decode, 블록 경계, 복원이 bit 동일|
|T2 기본 풀(`u3-t2-b01`)|21요청 텍스트가 U2와 전부 같고 gold 19/19. 요청별 TTFT −0.6 ~ +0.3 %, 재사용 토큰 수 같음. E(5 스트림·4 세션)는 4개 입장으로 U2와 같다. 풀은 256블록(4 GiB, `sized=sessions`)|
|T2 운영 비율 풀(`u3-t2-pool-b01`)|풀 160블록(16K 2.5개분). 텍스트 21/21 같음, gold 19/19, TTFT는 기본 풀과 같다. E는 풀 기준으로 2개씩 입장하고, 먼저 들어간 스트림이 끝나 블록을 돌려준 뒤 다음 둘이 들어간다(첫 토큰 0.7 / 7.6 / 7.6 / 13.5 s). **풀 부족 시 대기와 블록 반환 확인**|
|실제 길이 입장(`u3-admit-b01`)|256K × 4를 40 GiB 풀(2,560블록)로 기동했다. 짧은 prompt + `max_tokens` 32,768 스트림 4개가 블록 130개씩 예약하고 1.5–1.9 s 안에 모두 시작했다(free 2,430 → 2,040). 닫은 뒤 미사용 129개씩 반환해 free 2,556이 됐다(유휴 frontier 4개가 1블록씩 보유). 이어서 짧은 요청 gold 정답. **256K×4·40 GiB 계산 확인**|
|T3(`u3-t3-a01`)|U2 대비 8개 측정점 −0.1 ~ +1.0 %. AR pp2048 652.4, tg 12.94. DFlash2 pp 599.4, tg @ d8192 60.63. bench는 component cache를 거치지 않아 경로가 같다|
|기동 메모리(`u3-startup-a01`)|기본 풀은 U2와 같다: 16K × 4 87,359 MiB(U2 87,478), 64K × 8 54,933 MiB(U2 54,951). 64K × 8을 24 GiB 풀로 하면 63,009 MiB로 **+8,076 MiB**(계산 +8,192). 256K × 4·40 GiB는 50,598 MiB|
|T4s 기본 풀(`u3-t4s-{long16k,two8k,regress,boundary,shared}-a01`)|텍스트가 U2·U0와 전부 같다(5/5, 8/8, 20/20, 11/11, 19/19). gold 전부 정답. TTFT ±0.1 s|
|T4s 운영 비율(`u3-t4s-density4-b01`, `density8-b01`)|텍스트가 U2·U0와 같다(8/8, 16/16). gold 전부 정답. 다음 턴은 모두 RAM checkpoint에서 재사용했다. 풀보다 큰 대화 묶음이라 일부가 입장 대기한다: density4 다음 턴 TTFT 1.9 / 1.9 / 9.3 / 9.3 s(U2 1.3–1.7), density8 5.3–7.3 s 여섯 개와 24.4–24.9 s 두 개(U2 2.2–4.2)|

### 기록할 것

- **u3-models-a01은 무효다.**
  - 원인: `qwen27b_continuation_adapter_test`가 `EXCLUDE_FROM_ALL` 타깃이다. preset 빌드가 이 바이너리를 다시 만들지 않아서, U2 때 바이너리가 돌았다(`--draft` 포함).
  - 처리: 타깃을 명시해 빌드하고 a02로 다시 돌렸다(통과). 최종 빌드에서도 b01로 다시 돌렸다.
  - 앞으로: 단계마다 이 타깃을 명시해 빌드한다.
- **결함 1건(`15f27e88`으로 수정):** 풀이 모자라 요청이 기다리는 동안, 스케줄러 패스마다 `ReclaimRows`가 이미 비운 유휴 슬롯을 다시 cold acquire로 리셋했다.
  - 규모: u3-t2-pool-a01 대기 50초 동안 26,478번, density4-a01에서 40,393번 리셋하고 그만큼 로그를 남겼다.
  - 영향: 출력은 같았다. 하지만 리셋마다 stream drain과 recurrent 상태 초기화가 GPU를 잡아서, density4-a01 다음 턴 TTFT가 27–68 s까지 늘었다.
  - 수정: ReclaimRows가 비운 슬롯은 다시 빌려 쓰일 때까지 건너뛴다. CPU 테스트에 대기 패스 검사를 더했다.
  - 수정 뒤(b01): 리셋은 5번·6번이었고, density4 다음 턴은 1.9–9.3 s였다.
- **운영 비율 풀의 대기 비용:**
  - density4: 대화 4개가 각각 약 14K 토큰(실제 매핑 55–56블록)이라 합이 약 224블록이다. 풀 160블록에 다 들어가지 않으므로 둘씩 입장한다.
  - density8: 대화 8개 × 57블록이 풀 384블록보다 크다.
  - 우리 운영 빌드(P2, 256K × 4·40 GiB)의 설계와 같다. 기본 크기(세션 수 × 전체 문맥)에서는 실행 중 요청의 예약이 많아야 한 문맥이라 이런 대기가 생길 수 없다.
- **입장 예약은 `max_tokens` 최악값이다.** 실제 생성이 짧아도 `prompt + max_tokens`를 잡는다. 생성 도중 실패하지 않게 하려는 것이고, 대신 입장이 보수적이다. U5에서 대기 시간과 함께 다시 본다.
- **u3-t3-a01:** 창 판정 `FAILED_no_fallback`. 결과는 COMPLETE였고, 원인은 끝난 뒤 GPU busy 4 %다. 이전과 같은 패턴이고 guard는 완화하지 않았다.
- **최종 빌드에서 다시 돌리지 않은 것:**
  - 대상: T1 커널, T3, 기동 메모리, 기본 풀 T4s 5창. 모두 `033c2816`에서 돌았다.
  - 이유: 수정은 풀 대기·회수 경로 한 곳뿐이다. 기본 풀에서는 실행 중 예약이 문맥 하나를 넘지 않아 대기가 생기지 않으므로, 이 창들은 수정된 코드에 닿지 않는다.

## 판정

- 슬롯이 공유 풀에서 블록을 필요할 때 받고, 입장 때 예약하며, 모자라면 기다리고, 끝나면 돌려준다.
- 출력은 모든 경우에서 같다. 원본 세션 메모리 그대로인 기본 크기에서는 TTFT·bench가 5 % 안이다.
- 운영 크기 풀로 줄일 수 있는 메모리는 계산과 맞는다: 64K × 8 −8 GiB, 256K × 4 −24 GiB.
- **U3 통과.** checkpoint 블록 참조·CoW는 매핑 복원과 함께 U4로 옮겼다.
- 증거: `storage/evaluations/gufo-upstream-port-u3-20261023/`

## 다음 (U4)

- checkpoint chunk가 RAM 복사 대신 장치 블록 참조(Retain)를 갖는다.
- 다른 슬롯 복원은 복사 대신 블록 매핑으로 한다.
- 공유 블록에 쓰기 전에 CoW한다.
- 원본 cache 계층(ledger·retention)에 장치 블록 몫을 어떻게 셀지가 설계 핵심이다([API 검토](API-REVIEW.md) "바꿀 것" 6).
