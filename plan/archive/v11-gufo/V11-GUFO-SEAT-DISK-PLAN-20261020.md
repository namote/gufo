# v11 gufo 캐시 B: 자리(seat)에서 disk로 직접 저장 계획 (2026-10-20)

상태: B1·B2 완료, B3은 사용자 결정 A(축출 시점 write-back)로 구현·측정 완료(2026-10-20). 남은 선택: 미리 비워 두기(아래). 사용자 결정(2026-10-20) "응 가자": component cache 이전(S1)보다 먼저 이 구조로 간다.
- 세션 KV를 GTT에 미리 잡는다(`hipMalloc`, CPU 접근 불가).
- offload는 작은 pinned bounce 버퍼를 거쳐(`hipMemcpy` DMA) disk로 간다.
- 같은 RAM 안에 중간 전체 사본을 두지 않는다.

[S1 계획](/home/m2on/projects/ai.tools/archive/plan/2026-10-23-gufo-s1-component/V11-GUFO-S1-PLAN-20261020.md)은 보류한다. 코드는 gufo 작업 트리 `~/projects/gufo`(브랜치 `ai-tools/cache-s4`)에서 고치고, 실험은 `test/experiments/gufo_eval_20261020/`에 둔다. 실험 기본값·Compose는 바꾸지 않는다.

## 측정·확인된 사실

- `integrated=1`. `hipMalloc` 메모리는 CPU가 읽으면 signal로 죽는다(`hostPointer` 없음). `hipHostMalloc`은 host·device 포인터가 같다.
- `hipMemcpy` 장치→host 256 MiB는 첫 회 18 GB/s, 이후 68 GB/s다(probe 1회, window guard 밖).
- 대상: Qwen 27B는 옛 경로(`ContinuationCache` + `ContinuationDiskStore`)다. Gemma4·Flash-Next는 component cache다(pinned host backing 사본 → disk). 그쪽은 별도 작업이다.
- 지금 Qwen 저장 경로(코드 확인): 자리 → `SaveSnapshot` 장치 전체 사본(D2D) → 기본 `StreamPersistentSnapshot`의 host 전체 vector → 1 MiB 조각 write + SHA. 전체 크기 사본이 두 개다.
- 복원 경로: 파일 → 파일 크기 host vector → SHA 검증 → 자리로 H2D. 장치 사본은 없다.
- 생성된 마지막 경계는 Commit 때 disk에 쓰이지 않는다. 같은 대화가 다음 턴에 live 적중으로 돌아올 때만 저장된다(frontier freeze). 다른 대화가 자리를 먼저 가져가면 사라진다.
- write-back 지점: `ContinuationCache::Acquire`가 자리를 고른 뒤, `live_tokens.clear()`와 상태 덮어쓰기 전.
- 종료 순서: `TextRunnerPool::Impl`에서 `disk_store`가 `cache`(자리)보다 먼저 정리된다. 종료 write-back을 하려면 순서를 바꿔야 한다.
- 위 저장·복원 흐름 분석의 상세는 서브에이전트 보고(2026-10-20)에서 가져왔고 file:line 근거가 있다.

## 단계

|단계|내용|통과 기준|
|---|---|---|
|B1|공통 bounce 실행기(`src/core/hip`, 장치 2D 구간을 pinned bounce 조각으로 내보내기·들여오기, double buffer). Qwen 직렬화를 구간 목록으로 바꾸고 `StreamPersistentSnapshot`을 override한다|GPU 테스트: 같은 snapshot의 기존 `SerializePersistentSnapshot` 바이트와 스트림 바이트가 같음(AR·DFlash2). 저장 시 host 최대 할당 = bounce(+verifier payload)|
|B2|스트리밍 복원: store가 header를 읽은 뒤 payload를 reader로 runner에 넘긴다. runner는 bounce → 자리에 쓰고, 끝에서 SHA를 대조한다. 틀리면 자리 Invalidate 후 항목 삭제(기존 실패 처리와 같음)|기존 복원과 같은 상태(다음 토큰 logits 동일), 손상 파일은 miss·삭제, 복원 시 host 최대 할당 = bounce|
|B3|자리 직접 write-back: 자리 재배정 직전, 종료(순서 수정) 시 그 자리의 최신 경계를 상태에서 바로 스트리밍한다. 쓰는 동안 자리를 persisting으로 잠근다(재배정·Invalidate·복원이 대기). min-step·중복 판단은 enqueue 때 한다|세 대화 이상 순환에서 재시작 후 각 대화의 마지막 경계 복원, write-back 경로 장치 사본 0|
|B4|GPU 검증: 2차·5차 시나리오, 3개 이상 대화 순환(sessions 2), 같은 빌드의 B 이전과 비교|텍스트 동일·정답, 복원 토큰 ≥ 기존, TTFT 요청별 회귀 ≤ 5 %, checkpoint당 복사 바이트·host 최대 할당·disk 기록량 기록|

## 원칙

- 공통 실행기는 모델 이름과 무관하다. 모델은 payload 순서의 구간 목록만 제공한다.
- 실패는 그대로 기록하고 다른 경로로 자동 전환하지 않는다. 스트리밍을 구현하지 않은 runner는 기존 경로를 쓰고, 어느 경로였는지 이벤트에 남긴다.
- 프롬프트 경계 disk capture(장치 사본 사용)는 B3 뒤 측정을 보고 정한다. S2 보존 정책과 함께 다룬다.
- S1a(upstream 병합, gufo `939d0375`, Gemma4 backing API 수정 `3e0297aa`)는 이미지 빌드와 CPU 129 테스트(실패 0, 8 건너뜀)를 통과했다. GPU 회귀는 B4 기준선에서 같이 잰다. Gemma4 쪽 수정의 GPU 회귀는 미검증이다.

## B1·B2 결과 (2026-10-20)

gufo 커밋은 upstream 병합 `939d0375` 이후 순서대로 다음과 같다. 패치는 `test/experiments/gufo_eval_20261020/patches/b/`에 있다.
- `3e0297aa`: Gemma4 backing API
- `d27de205`: B1 스트리밍 저장
- `9f66b879`: B2 스트리밍 복원·조각 검증
- `a155bb0e`: bounce 4칸 ring

이미지는 `ai-tools/gufo-eval:a155bb0e148a`다.

- **구현:**
  - 공통 `hip::BounceStream`: pinned 32 MiB × 4칸 ring, 장치 2D 구간을 조각으로 내보내기·들여오기.
  - Qwen `QwenGpuSnapshot::CompactSegments`: 기존 `SerializeCompact`와 새 `StreamCompact`가 같은 구간 목록을 쓴다.
  - runner `StreamPersistentSnapshot` override.
  - 새 hook `RestorePersistentSnapshotStream`: 기본은 기존처럼 전체를 읽고, Qwen은 bounce로 자리에 직접 쓴다.
  - disk store `ImageReader`: metadata만 통째로 읽고, payload는 조각 단위로 SHA를 계산하며 넘긴다. 시작 시 색인 검증도 조각 단위로 한다. checksum이 틀리면 자리를 Invalidate하고 항목을 지운다.
- **CPU:** 129 테스트 중 실패 0(8 건너뜀). disk store의 손상·checksum·staging 테스트를 포함한다.
- **GPU 단위** (`qwen27b_snapshot_stream_test`, 실제 Qwen 27B, 1,213토큰, compact 238,354,512 B, window guard 밖 단독 실행):
  - bounce 기본·1 MiB·12,289 B(KV 한 행보다 작음)에서 스트림 바이트가 `SerializeCompact`와 같다.
  - 다른 대화 내용이 남아 있던 자리에 스트리밍 복원한 뒤, 다시 직렬화한 바이트가 원본과 같다.
  - 이어서 생성한 토큰이 기존 복원·cold 실행과 같다.
- **GPU 서버** (5차 `qwen27b-boundary` 그대로, 보존본 `storage/evaluations/gufo-eval-20261020/b-a01/`): 세 빌드 모두 텍스트 11/11 동일·정답, 적중(T2 8,187, 재시작 후 T3 11,509) 동일, 저장·생략 이벤트 수 동일.

|빌드|disk 기록 8K / 11.5K (ms)|재시작 후 T3 TTFT (s)|cgroup 최대 메모리|health|
|---|---|---|---|---|
|`3e0297aa` 기준(B 전)|904–927 / 1,145–1,154|1.95–2.16|17.98 GiB|FAIL(busy 4 %)|
|`9f66b879` B, 2칸|1,092–1,270 / 1,878–2,006|1.76–1.80|17.21 GiB|FAIL(busy 4 %)|
|`a155bb0e` B, 4칸 ring|718–803 / 1,120–1,146|1.78–1.83|17.40 GiB|PASS|

- 2칸일 때는 조각마다 장치→host 복사가 동시에 도는 prefill 뒤에서 기다려 기록이 느렸다. 4칸 ring으로 복사를 세 조각 앞서 걸어 두니 기준과 같거나 더 빨라졌다.
- 다른 TTFT는 기준 대비 ±1.5 % 안이다.
- 최대 메모리 감소는 −0.58 GiB다(8K payload 한 개분 host vector 제거, ring 2개 0.25 GiB 추가). 32K에서는 감소분이 약 2.5 GB로 커진다(미측정).
- 남은 사본: 장치 snapshot(D2D)은 저장 전에 아직 만든다. B3에서 제거한다.

## B3 1차: 유휴 write-back 결과 (2026-10-20)

gufo 커밋은 다음과 같다. 이미지는 `ai-tools/gufo-eval:edb48b112ef3`이고, 패치는 `patches/b/0005-`와 `0006-`이다.
- live arena 직렬화 `4e4b99a5`
- 자리 잠금·write-back 스레드 `edb48b11`

 보존본은 `storage/evaluations/gufo-eval-20261020/b-a02/`다.

- **구현:**
  - runner `PreparePersistentState`: logits·verifier는 미리 뜨고, KV·recurrent는 자리에서 bounce로 스트리밍한다. `PersistsStateInPlace`.
  - cache `BeginPersist`/`EndPersist`/`PersistCancelled`:
    - 쓰는 중인 자리는 재배정하지 않는다.
    - live 적중이면 쓰기를 취소하고 기다린다.
    - 재배정은 frontier가 비었거나 이미 저장된 자리를 우선한다.
    - 저장 안 된 frontier를 덮으면 `live_frontier_dropped`를 기록한다.
  - pool write-back 스레드: 반납 후 2초 유휴면 쓴다. 반납된 경계마다 1회 시도한다. 종료 시 남은 자리를 모두 쓴다(디스크 store보다 먼저).
  - store `SaveState`: min-step을 적용하지 않고, 같은 대화의 얕은 항목은 step 안이면 교체한다(`replaced`).
- **CPU:** 129 통과. 새 테스트 `TestReleasedFrontierIsWrittenBackInPlace`(종료 write-back, 재시작 복원, 교체)를 포함한다.
- **GPU:**
  - 같은 upstream 병합 위 B 이전 `3e0297aa`와 비교했다.
  - 텍스트는 모두 같고 정답이다(regress 20/20, boundary 11/11).
  - `live_frontier_dropped` 0, `write_back_failed` 0.

|시나리오|항목|B 이전|B3|
|---|---|---|---|
|2차(대화 6·자리 2)|T2 적중/TTFT|8,187 / 1.66–1.69 s|8,407 / 1.15–1.22 s|
||재시작 후 T3|8,187 / 2.21–2.44 s|8,653 / 1.18–1.31 s|
||cold T1 TTFT|13.67–13.93 s|13.94–14.20 s (+2–3 %)|
||disk 기록|saved 6|saved 8 + write_back 18 (0.77–3.06 s), replaced 19|
||cgroup 최대|17.58 GiB|17.23 GiB|
|5차 boundary|T2 / T3 TTFT|7.21–7.27 / 1.95–2.16 s|6.76–6.83 / 1.37–1.40 s (T3 11,729 복원)|
||cold TTFT|13.60–19.89 s|+1.5–3 %|
||write_back|—|9건, 0.95–4.83 s|

- **health:** regress-B FAIL(busy 4 %), boundary-B PASS, 기준 regress PASS. 기존 캐시 창과 같은 양상이다.
- **해석:**
  - 매 턴 경계를 쓰므로 대화가 자리보다 많을 때 기록량이 4배 이상이다.
  - write-back이 다음 대화 prefill과 겹쳐 cold TTFT가 2–3 % 늘었다(5 % 기준 안이지만 일관됨).
  - write-back 자체도 경합으로 길어졌다.
- **사용자 결정 필요:** 기록 시점 대안은 다음 셋이다. 추천은 A다.
  - A: 다른 대화가 자리를 필요로 할 때만 쓴다.
  - B: 기존 항목 대비 N토큰 이상 늘었을 때만 쓴다.
  - C: 현행 유지.

## B3 결정 A: 축출 시점 write-back (2026-10-20)

사용자 결정은 "추천대로"다. 즉 다른 대화가 자리를 필요로 할 때만 쓴다.
- gufo `be818621`, 이미지 `ai-tools/gufo-eval:be8186210a72`, 패치 `patches/b/0007-`.
- 보존본 `storage/evaluations/gufo-eval-20261020/b-a03/`.

- **구현:**
  - 유휴 타이머를 없앴다.
  - scheduler가 `Acquire` 전에 `TextRunnerPool::AdmissionReady(prompt)`를 묻는다.
  - 남은 자리가 모두 저장 안 된 frontier를 들고 있으면, 가장 오래된 자리를 급히 쓰기로 표시하고 그 요청만 대기열 앞에 되돌린다. 이미 실행 중인 요청은 계속 decode한다.
  - write-back 스레드가 5 ms 이내에 쓰기를 시작한다.
  - 같은 대화가 돌아오면 쓰지 않는다. 쓰는 중이면 취소한다.
  - 실패한 쓰기는 다시 시도하지 않는다. 그 frontier는 덮이고 `live_frontier_dropped`로 남는다.
  - 종료 시에는 남은 frontier를 모두 쓴다.
- **CPU:** 129 통과. 새 테스트 `TestEvictionWritesTheFrontierBeforeReuse`를 포함한다.
- **GPU:** 텍스트는 모두 같고 정답이다. `live_frontier_dropped` 0, `write_back_failed` 0, health 3/3 PASS.

|시나리오|항목|B 이전 `3e0297aa`|C 유휴 2 s `edb48b11`|A 축출 `be818621`|
|---|---|---|---|---|
|2차(대화 6·자리 2)|write-back 수 / 시간|—|18 / 0.77–3.06 s|18 / 0.61–0.67 s|
||T1 c2–c5 TTFT|13.80–13.93 s|14.11–14.20 s|14.41–14.51 s (+0.6 s, 축출 쓰기 대기)|
||T2 적중 / TTFT|8,187 / 1.66–1.69 s|8,407 / 1.15–1.22 s|8,407 / 1.75–1.80 s|
||재시작 후 T3|8,187 / 2.21–2.44 s|8,653 / 1.18–1.31 s|8,653 / 1.23–1.33 s(c0·c1), 1.81–1.84 s(c2–c5)|
|5차 boundary|T3 TTFT|1.95–2.16 s|1.37–1.40 s|1.41–2.15 s|
||write-back 수 / 시간|—|9 / 0.95–4.83 s|9 / 0.63–0.80 s|
|대화 2·자리 2|write-back 수|—|6 (1.0–2.9 s)|4 (0.62–0.67 s, 재시작·종료 때만)|
||T2 / T3 TTFT|—|0.44–0.45 / 1.16–1.27 s|0.44 / 1.17–1.31 s|

- **A의 효과:**
  - 실행 중 낭비 쓰기가 0이다. 대화 2개는 재시작·종료 때만 썼다.
  - 쓰기가 다른 prefill과 겹치지 않아 0.6–0.8 s로 짧다.
  - 대기 중인 요청 말고는 cold TTFT 증가가 없다(T1 c0·c1 13.67 대 13.67).
- **A의 비용:**
  - 축출이 필요한 요청은 이전 대화 쓰기 시간만큼 첫 토큰이 늦다. 8K 기준 +0.6 s이고, 32K는 약 2–2.5 s로 추정한다(미측정).
  - 대화가 자리보다 훨씬 많은 순환에서는 C보다 TTFT가 나쁘다(T2 1.78 대 1.18 s). 쓰기 수는 같다.
- **남은 선택(미구현):** 미리 비워 두기. 모든 자리가 저장 안 된 frontier로 차 있고 오래(예: 30 s 이상) 유휴면, 가장 오래된 자리를 미리 쓴다. 바로 돌아오는 대화에는 비용이 0이고, 나중에 오는 새 대화의 대기를 감춘다.
