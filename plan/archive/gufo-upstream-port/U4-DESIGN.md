# gufo 원본 이식 U4: checkpoint가 KV 행을 장치 블록 참조로 갖는다 (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [U3 결과](U3-RESULTS.md), [U5b 결과](U5B-RESULTS.md), [API 검토](API-REVIEW.md) "바꿀 것" 6.

사용자 결정: U5b 뒤 남은 결정 1(메모리)에 "1번 진행", "확인하고 U4 진행". RAM 예산을 낮춘 창(f3, 2 GiB·1 GiB)으로 먼저 비용을 확인하고 U4를 한다.

## 왜

- U5b 이식본은 운영 대비 anon이 약 +8 GiB였다(24–25 / 16–17 GiB).
- 원인: RAM checkpoint가 FP16 K/V 행을 고정(pinned) 호스트 backing에 사본으로 갖는다. 행은 이미 공유 KV 풀(U3)의 블록에 있는데, 같은 바이트를 한 번 더 갖는 셈이다.
- 슬롯에서 빌린 행(`Payload::Borrowed`)도 host backing을 미리 예약하고 idle 때 채운다. 그래서 메모리는 같다.
- 운영 포크는 행을 슬롯 장치 메모리에만 두고, snapshot에는 private 상태(약 245 MB)만 둔다.

## 무엇을

1. **장치 payload(`src/cache`):**
   - `Payload::Device(metadata charge, owner, bytes)`를 둔다. owner는 장치 행 첫 바이트를 가리키고, 블록 참조를 쥔다.
   - RAM 장부에는 메타데이터만 매긴다. `RamBytes()`는 0이다.
   - 복원 pin의 바이트는 장치 주소다(`CopyRowsIn`은 이미 장치 출처를 받는다).
   - 보존(spill) 의무가 없다. 블록은 불변이고, 쓰기 전 CoW는 adapter가 한다.
2. **Adapter 확장(`cache::Adapter`):**
   - `RetainRows(slot, component, first, end)`: 완료된 chunk 하나가 장치 블록 하나에 통째로 있으면, 그 블록 참조를 쥔 owner와 주소를 돌려준다. 아니면 빈 값이고, 지금처럼 사본으로 간다.
   - `RetainsRows(component)`: 그 component의 완료 chunk를 장치에 둘 수 있는지 알린다. capture 전 RAM 바이트 계산에 쓴다.
   - 기본 구현은 둘 다 "안 함"이다. 다른 모델(Flash-Next·Gemma)은 바뀌지 않는다.
3. **Qwen adapter와 arena:**
   - FP16 K/V(공유 풀)만 대상이다. 블록 하나 = 256토큰 × 모든 층 K/V이다. capture 하나의 같은 블록 component들은 블록 참조 하나를 같이 쥔다(Retain 1회).
   - **CoW:** 슬롯이 참조 수 2 이상인 블록에 쓰려 하면(`BeforeWrite`), 새 블록을 받아 그 블록 전체(두 plane, 모든 층)를 장치 안에서 복사하고, 테이블을 바꾼 뒤 옛 블록을 놓는다.
     - 대상 경우: 공통 prefix가 live보다 짧아 그 안쪽부터 다시 prefill할 때. speculative rollback은 완료 블록 안으로 돌아가지 않으므로 해당하지 않는다.
   - **매핑 복원:** `BeginRestore`는 [0, target)의 공유 블록을 새 블록으로 바꿔 둔다. `CopyRowsIn` 출처가 같은 plane·층의 풀 블록 b 전체이고, 그 슬롯 블록에 이번 복원에서 아직 아무것도 복사되지 않았으면, 복사 대신 b를 매핑한다(Retain b, 새 블록 Release).
     - 그 밖의 경우는 지금처럼 복사한다. 이미 매핑된 공유 블록에 복사가 오면 먼저 CoW한다.
     - 테이블 업로드는 `Validate`에서 한 번 한다.
4. **cache 계층(`component_text_cache`):**
   - capture할 때 `RetainsRows` component의 완료 chunk는 장치 payload로 만든다. tail·private 상태·draft K/V는 지금처럼 RAM 사본이다.
   - `NewPayloadBytes`에 장치 component를 빼는 판을 둔다. RAM 예산 판단, `CaptureRefused`, retention 요청이 RAM에 실제로 드는 바이트로 계산된다.
   - 디스크 게시는 장치 payload를 `copy_observed_rows`(D2H)로 읽는다. 블록 참조가 job 동안 출처를 잡아 둔다.
   - retention의 RAM 압박 계산(`UniquePayloadBytes`)은 RAM 바이트만 센다. 장치 참조만 가진 checkpoint를 RAM을 비우려고 지우지 않는다.
5. **풀 압박:**
   - checkpoint만 쥔 블록은 입장 블록을 줄인다. 입장이 모자라면(`ReclaimRows`) 기존 순서(유휴 슬롯 행 반환) 전에 checkpoint만 쥔 블록을 가진 RAM checkpoint를 지운다.
   - 순서: 디스크에 있는 것부터, 그다음 retention 순서. 지운 것은 `event=checkpoint_dropped reason=kv_pool`로 남긴다.
   - 자동 재시도나 다른 경로 전환은 없다. 그래도 모자라면 지금처럼 기다리거나 이름으로 거절한다.

## 실패 처리

- `RetainRows`가 빈 값이면(블록 경계가 아닌 chunk, FP32 정책, 풀 없음) 그 chunk는 RAM 사본이다. 이는 실행 전 명시적 경로 선택이므로 이벤트 카운터(`device_chunks`/`copied_chunks`)에 따로 센다.
- CoW 할당 실패는 쓰기 전에 throw하고 슬롯을 실패로 latch한다(지금 `EnsureKvRows` 부족과 같다). 입장 예약에 CoW 몫 1블록을 더한다.
- 장치 payload의 블록은 마지막 참조(checkpoint, reader pin, persistence job)가 놓일 때 풀로 돌아간다.

## 검증

- **단위(T0):**
  - allocator와 arena: CoW가 블록 내용을 보존하고 참조 수를 맞춘다.
  - Qwen adapter(GPU 테스트): capture → 슬롯 reset → 다른 슬롯 매핑 복원 → 출력 동일. 같은 슬롯 짧은 복원, 공유 블록 위 쓰기 CoW, 장치 payload 디스크 게시.
  - cache 계층: fake adapter에 `RetainRows`를 두고 RAM 장부·retention·`NewPayloadBytes`·풀 압박 drop을 검사한다.
  - 각 테스트는 수정 없이 실패하는 것을 확인한다.
- **GPU 창:** F2와 같은 창(f4)과 T2를 운영 빌드와 번갈아 돌린다.
  - 판정: 텍스트·gold 동일, T2 21/21.
  - anon이 운영 수준(±1–2 GiB)인지, RAM 복원 TTFT(다음 턴·shared·T2 C-restore)가 U5b 이하인지, 256K 재시작이 U5b와 같은지 본다.
- f3(RAM 예산 2·1 GiB) 결과는 비교 기준으로만 쓴다. 예산만 낮추면 무엇을 잃는지 보여 준다.

## f3 결과: RAM 예산만 낮추면 (2026-10-24, `final-f3-a01`)

같은 이식본(`3942939b`)에서 `--cache-ram-bytes`만 2 GiB와 1 GiB로 바꿨다. 비교 기준은 `final-f2-a06`(이식본 8 GiB, 운영 `210c7633`)이다.

- **정확성:** 텍스트는 모든 창에서 8 GiB와 같다. T2는 21/21 같다.
- 시간은 TTFT(초)이고, 순서는 운영 / 8 GiB / 2 GiB / 1 GiB이다.

|항목|운영|8 GiB|2 GiB|1 GiB|
|---|---|---|---|---|
|anon 최대 (GiB)|16.3–17.0|23.1–24.9|19.5–19.6|18.7|
|256K 재시작|5.12|4.99|7.24|6.40|
|32K 재시작 뒤 T3|1.36|1.62|0.85|4.16|
|two8k 재시작 뒤|0.80|0.60|1.85|0.60|
|shared 첫 팔로워 c1|2.30|7.88|14.42|14.34|
|shared 두 번째 팔로워 c2|2.48|2.32|2.33|14.44|
|shared TTFT 합|52.4|55.5|63.1|75.1|

- **예산을 1 GiB까지 낮춰도 anon은 운영보다 약 2 GiB 많다.**
- **비용이 생긴다.**
  - 256K 재시작이 1.4–2.3 s 늦어진다.
  - 공유 대화에서 학습한 분기점 checkpoint(7,174, 약 715 MB)가 예산에 들어가지 못해, 팔로워가 처음부터 prefill한다(14 s).
- 판정: 예산만 낮추는 것으로는 메모리가 운영 수준이 되지 않고 TTFT도 잃는다. U4로 간다.
