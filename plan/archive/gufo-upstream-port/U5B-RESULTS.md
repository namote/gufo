# gufo 원본 이식 U5b: 재시작 경로 보완 (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [F 결과](F-RESULTS.md), [U5 결과](U5-RESULTS.md).

사용자 확인: F 결과의 다음 할 일 추천(B3 idle write-back 이식, 메모리 anon·file 분리, 256K 다음 턴 확인, 재시작 창 재실행)에 "오케이 가자".

브랜치는 gufo `ai-tools/port-u5b`(`3e36130a`에서 분기), 최종 `3942939b`이다. 패치 6개는 `test/experiments/gufo_upstream_port_20261023/patches/u5b/`에 있다.

## 변경 (gufo 커밋)

|커밋|내용|
|---|---|
|`76a731de`|유휴 live frontier를 디스크에 쓰는 write-back, 거절될 stable 경계에서 prefill을 멈추지 않음. **결함 둘**(디스크 복원이 끝나기 전 사전 검사가 예외 → HTTP 500, 스케줄러 전체가 유휴일 때만 실행). 이 빌드의 F2-a01은 무효|
|`82327bd4`|세션별 유휴 판정(다른 세션이 돌고 있어도 실행), 복원 중에는 사전 검사가 거절로 보지 않음|
|`19c7e019`|RAM checkpoint가 디스크로 게시되면 그 실행의 durable 조상으로 삼아, 뒤의 write-back이 같은 chunk 파일을 다시 쓰지 않음(F2-a02 32K에서 SSD 14.8 GB를 보고 찾음)|
|`72e451b7`|① 종료 시: 큐를 비운 뒤 덮이지 않은 live frontier를 한 번 더 write-back(방금 쓴 checkpoint의 chunk 공유). ② 같은 계보 checkpoint가 아직 큐에 있으면 write-back을 미룸. ③ 슬롯 재배정 capture는 durable checkpoint가 frontier를 덮지 않으면 disk step 없이 디스크로(그 슬롯의 대기 중 write-back은 원본 계약상 reset 때 실패한다). ④ 거절될 prompt snapshot 경계에서도 prefill을 멈추지 않음|
|`85890ae1`|in-pass 예측이 없는 어댑터(Qwen 27B)에서 사전 검사가 0바이트로 계산되던 결함. 경계 앞의 frontier 크기를 하한으로 써서, 그것만으로 예산을 넘으면 멈추지 않음(`82327bd4`·④가 실제로는 동작하지 않던 원인)|
|`3942939b`|같은 공유 checkpoint에서 갈라진 대화는 계보가 같다. 그래서 한 대화의 checkpoint가 다른 대화의 durable 조상이 되고, 길이만 비교해 그 대화의 frontier를 "디스크에 있음"으로 봤다(종료 때 건너뜀). durable은 frontier의 prefix일 때만 덮은 것으로 센다|

- 원본 구조를 그대로 쓴다: 원본 `Lease`·`CaptureDurable`·`PublishSnapshot`·`PersistenceQueue`.
- write-back은 선택적 동작이다. 실패하면 기록하고 원본처럼 진행한다.
- 정책 필드 `idle_write_back_delay`는 2 s다(운영 `--cache-idle-write-ms 2000`과 같은 값).
- **CPU 테스트:** 127개 전부 통과. 새 테스트 6개를 넣었고, 각 테스트가 해당 수정 없이 실패하는 것을 확인했다.
  - IdleWriteBackResumesFromLastTurn
  - WriteBackReusesPersistedChunks
  - ShutdownWriteBackSharesQueuedCheckpoint
  - ReassignmentWritesUncoveredFrontier
  - RefusedCaptureIsKnownBeforePrefill(예측 없는 어댑터 경우 포함, fake adapter에 `SetCheckpointPrediction` 추가)
  - SharedLineageCoversOnlyItsOwnFrontier
- ④의 runner 경로(prefill 멈춤 생략)는 CPU 테스트가 덮지 않는다. 실제 창(256K 다음 턴)으로 확인했다.

## 측정 (F2, 이식본과 운영 빌드 `210c7633`를 번갈아)

창은 F와 같은 운영 구성이다(256K × 4, 풀 40 GiB, RAM 8 GiB, 디스크 64 GiB). 메모리는 호스트에서 컨테이너 cgroup `memory.stat`을 2초마다 읽어 anon(프로세스)과 file(페이지 캐시)로 나눴다. 실행기는 `final.py --set f2`, 계획은 `PLAN-F2.json`이다.

|실행|이식본 빌드|창|비고|
|---|---|---|---|
|`final-f2-a01`|`76a731de`|중단|무효(결함), `VOID.json`|
|`final-f2-a02`|`82327bd4`|32K 한 쌍|SSD 14.8 GB 보고 중단, 부분(`PARTIAL.json`)|
|`final-f2-a03`|`19c7e019`|12창 + T2|전부 COMPLETE|
|`final-f2-a04`|`72e451b7`|12창 + T2|전부 COMPLETE. T2 시간은 CPU 빌드와 겹쳐 쓰지 않음(`NOTE.json`)|
|`final-f2-a05`|`85890ae1`|256K 한 쌍 + T2|전부 COMPLETE|
|**`final-f2-a06`**|**`3942939b`(최종)**|12창 + T2|전부 COMPLETE|

**정확도:** a03–a06 모든 창에서 이식본과 운영 빌드의 텍스트가 같다(창 69요청). gold는 전부 정답이다. T2 21요청은 네 실행 모두 U5c 기준 출력과 21/21 같다.

**TTFT(이식본 / 운영 빌드, 초):**

|창|F(`3e36130a`)|a03|a04|**a06(최종)**|
|---|---|---|---|---|
|32K 재시작 T3|2.4–2.5 / 1.1–1.2|1.11·1.14 / 1.1|1.10·1.12 / 1.1|1.62·1.23 / 1.36·1.08|
|256K 재시작 T3|9.98 / 4.7|13.34 / 4.43|9.17 / 9.86|**4.99 / 5.12**|
|256K 다음 턴|1.79 / 1.28|1.80 / 1.28|1.80 / 1.27|**0.94 / 1.27**|
|two8k 재시작 T3|—|0.60·0.54 / 0.79·0.72|1.80·1.82 / 0.79·0.72|0.60·0.60 / 0.80·0.73|
|regress 다음 턴(6요청)|—|0.41 / 0.81|0.41–0.50 / 0.81|0.41–0.51 / 0.80–0.82|
|boundary 재시작 T3|—|0.43–0.46 / 0.67–0.70|0.97–1.16 / 0.62–1.03|1.03–1.19 / 0.66–0.86|
|shared 재시작 T3(6대화)|1.3–2.9 / 0.6–0.8|c1·c5 4.19·2.94|c5 1.30|0.62–1.68 / 0.74–1.12|
|TTFT 합 two8k·regress·boundary·shared|61.1·28.3·30.8·65.1 / 60.0·26.7·24.3·50.8|59.2·24.4·23.6·60.6 / 59.8·27.2·24.4·51.8|61.7·24.1·25.4·56.5 / 59.7·26.9·24.7·50.9|59.1·24.3·25.6·55.5 / 59.7·26.8·24.6·52.4|

- **256K 재시작:** F·a03에서는 시나리오가 마지막 턴 직후 종료해 write-back 기회가 없었다(254,873에서 복원, 499토큰 재계산). 운영 포크는 종료 때 frontier를 쓴다. ①로 같은 동작이 되어 255,339에서 복원하고 33토큰만 계산한다. 남는 차이는 복원 읽기 시간이다(아래).
- **256K 다음 턴:** 255,093 frontier 바로 뒤 stable 경계(255,119)에서 prefill이 한 번 멈추고 12.8 GB capture가 거절되었다(255K 깊이에서 pass 두 번). `85890ae1`로 멈추지 않아 운영보다 빠르다.
- **shared 재시작:** a06에서 6대화 모두 자기 마지막 frontier(8,667)에서 복원한다.
  - a03: c1·c5가 7,174 공유 경계에서 1,526토큰을 다시 계산했다. 재배정 슬롯의 대기 중 write-back이 실패했고, 재배정 capture는 min_step으로 거절됐다 → ③.
  - a04: c5가 8,421에서 복원했다. 다른 대화의 checkpoint가 덮은 것으로 셌다 → `3942939b`.
- **shared 첫 팔로워(7.9 / 2.3 s):** F와 같다. 원본의 학습 분기점 규칙(D4) 때문이고, U5b 범위 밖이다.
- **재시작 디스크 읽기의 흔들림:** 원본은 디스크에 쓴 checkpoint 파일을 페이지 캐시에서 버린다(`bff0eec6`, #477 "keep disk-cache snapshots out of the page cache").
  - 그래서 재시작 뒤 복원이 SSD에서 읽힐 때가 있다. 같은 8K checkpoint 812 MB가 페이지 캐시에 있으면 70–200 ms, SSD에서 읽으면 1.1–1.3 s(약 0.65 GB/s)다. boundary 2.5K(407 MB)는 0.55–0.76 s다.
  - 실행마다 다르다: two8k는 a03·a06에서 캐시, a04에서 SSD.
  - 운영 포크는 쓴 파일을 페이지 캐시에 남겨 매번 약 100 ms에 읽는다(그 대신 file 메모리로 잡힌다).
  - 256K는 두 빌드 모두 흔들린다: 이식본 1.8–7.1 s, 운영 2.8–8.2 s.

**메모리(a06 cgroup 최대, GiB, 이식본 / 운영 빌드; a03·a04도 같은 수준):**

|창|anon|file|memory.current 최대|
|---|---|---|---|
|32K|24.9 / 17.0|0.0 / 1.9|25.0 / 18.7|
|256K|24.1 / 16.2|0.1 / 10.3|24.2 / 26.9|
|two8k·regress·boundary·shared|23.1–24.9 / 16.3–16.6|0.0 / 0.2–0.5|23.2–25.0 / 16.6–16.7|

- 이식본의 차이는 거의 전부 anon(프로세스)이다. RSS도 이식본 26.7–27.6 GiB, 운영 18.7–19.2 GiB다(a03 로그).
- 이식본 로그의 RAM checkpoint 원장이 8.5–9.4 GB로 RAM 예산(8 GiB)을 채운다. 이것이 약 8 GiB 차이다.
  - 같은 `--cache-ram-bytes 8 GiB`라도 두 빌드의 의미가 다르다.
  - 원본 RAM checkpoint는 KV 행 사본을 갖는다.
  - 운영 포크의 snapshot은 private 상태만 갖고(약 245 MB), 행은 블록 공유로 디스크 파일과 페이지 캐시에 둔다.
- 256K 창은 운영 빌드가 페이지 캐시를 10–12 GiB 써서 memory.current 최대는 오히려 이식본이 작다.

**SSD 쓰기(GB, 이식본 / 운영 빌드):**

|창|F|a03|a06|
|---|---|---|---|
|32K|5.2 / 6.6|12.5 / 6.6|12.9 / 6.6|
|256K|18.3 / 17.5|18.5 / 17.5|18.9 / 17.6|
|two8k·regress·boundary·shared|1.7·2.0·—·— / 3.6·6.4·—·—|4.2·7.0·3.7·8.0 / 3.6·6.6·3.8·6.1|4.6·7.7·4.6·9.5 / 3.6·6.4·3.8·6.1|

- 32K 창은 두 대화와 `cache_prompt: false` 대조 요청이 모두 같은 33K prompt다.
  - 운영 포크는 행 블록을 내용 hash로 중복 제거해 한 번만 쓴다(`skipped` 블록).
  - 원본은 계보 안에서만 chunk를 공유하므로(결정 D4) 세 번 쓴다.
  - 실제 대화는 system prompt만 공유하고, 그것은 원본 분기점 checkpoint가 공유한다.
- 짧은 대화 창은 write-back과 재배정 시 디스크 쓰기를 넣은 만큼 F보다 늘어, 운영 빌드보다 1–3 GB 많다.

## 판정

- **재시작·다음 턴 TTFT:** 운영 빌드 수준이 됐다. 256K는 재시작 4.99 / 5.12 s, 다음 턴 0.94 / 1.27 s다. two8k·regress는 운영보다 빠르다. 남는 0.2–0.5 s 차이는 원본의 페이지 캐시 버림(#477)으로 복원이 SSD에서 읽힐 때만 생긴다.
- **정확도:** 텍스트 동일, gold 전부 정답, T2 21/21.
- **메모리:** +8 GiB anon의 원인을 확인했다(RAM checkpoint 예산을 행 사본으로 채움). 남은 항목 1.
- 운영 전환 제안 여부는 아래 결정 뒤에 정한다.

## 남은 항목 (사용자 결정 필요)

1. **RAM 예산과 메모리:** 이식본의 `--cache-ram-bytes`를 운영 포크와 같은 메모리가 되게 낮출지(예: 1–2 GiB, 다음 턴·shared TTFT를 함께 잼), 아니면 U4로 checkpoint가 행을 블록 참조로 갖게 할지.
2. **작은 checkpoint의 SSD 복원:** 8K에서 약 0.65 GB/s다. helper 실제 읽기(U5 ③)가 작은 복원에서 앞서 가지 못하는 이유를 호스트 재현으로 확인한다. 또는 원본 #477을 가장 최근 checkpoint에는 적용하지 않는 방안을 원본에 물어본다(D4와 같은 원본 정책 문제).
3. **SSD 쓰기:** 같은 prompt를 여러 대화가 보내는 경우의 중복은 D4(계보 공유)의 결과로 둔다.

증거: `storage/evaluations/gufo-upstream-port-final-20261023/final-f2-a0{1..6}/`(cache-disk 제외, 지운 기록은 각 실행의 `CACHE-REMOVED.json`).
