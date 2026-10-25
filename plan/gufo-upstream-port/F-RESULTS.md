# gufo 원본 이식 최종점검 F (2026-10-23)

상위 문서: [이식 계획](PLAN.md), [테스트 설계](TESTS.md), [U5 결과](U5-RESULTS.md).

사용자 확인: F 질문에 "넹 ~". 실행 `final-a02`(20:36–22:03, 20창, 자동 재시도 없음). `final-a01`은 실행기를 셸에서 분리하려고 첫 창 11초 만에 멈춘 무효 실행이다(결과 없음, `VOID.json`).

## 구성

- **이식본:** gufo `ai-tools/port-u5` `3e36130a`, 운영 구성 그대로다.
  - 설정: 256K × 4, 풀 40 GiB, RAM 8 GiB, 디스크 64 GiB, `max_tokens` 32768, reasoning 16384, mmproj.
  - 운영 포크에만 있는 `--cache-disk-block-bytes`·`--cache-idle-write-ms`는 원본에 없어서 뺐다.
- **기준:** 운영 빌드 `210c7633`, 운영 플래그 그대로다.
- **실행 순서:** 시나리오마다 두 빌드를 번갈아(ABBA) 돌렸다.
- **원본 suite:** `long-context`·`cache`를 이식본과 원본 main `79c2dcc5`에서 돌렸다.
- **실행 묶음:** `test/experiments/gufo_upstream_port_20261023/`.
  - 실행기 `final.py`, suite 드라이버 `suite.py`.
  - `PLAN-F.json`(이식본), `PLAN-F-BASE.json`(운영 빌드), `PLAN-F-UP.json`(원본).
  - TESTS.md가 적은 별도 디렉터리 대신 기존 묶음의 window.py·LOCKS를 그대로 쓰도록 이 묶음에 넣었다.

## 결과

**정확도:** 20창 모두 결과 COMPLETE. 이식본과 운영 빌드의 텍스트는 비교한 모든 요청에서 같다. gold는 전부 정답이다.

|창|이식본|운영 빌드|비고|
|---|---|---|---|
|32K 대화 2개|첫 턴 58.7–60.4 s, 다음 턴 0.51 s, **재시작 뒤 2.4–2.5 s**|60.8–61.8 s, 0.48–0.60 s, **1.1–1.2 s**|디스크 읽기 자체는 이식본이 빠름(474–508 ms, 운영 634–694 ms)|
|256K 문서|첫 턴 899 s, 다음 턴 1.79 s, **재시작 뒤 9.98 s**|884 s, 1.28 s, **4.7 s**|복원 3.96 s / 3.43 s(16.9 GB, 16,213조각)|
|256K×4 동시|짧은 4개 19.6 s, 긴·짧은 섞음 154.5 s|19.1 s, 139.6 s|섞음에서 16K 95.8 s / 62.3 s, q0 29.8 s / 50.7 s(prefill 배치 순서 차이)|
|64K×8 동시|37.9 s, 189.5 s|38.3 s, 186.1 s|같은 수준|
|two8k·regress·boundary·shared(실제 운영 설정, 58요청)|TTFT 합 61.1 / 28.3 / 30.8 / 65.1 s|60.0 / 26.7 / 24.3 / 50.8 s|차이는 거의 전부 재시작 뒤 T3(이식본 1.3–2.9 s, 운영 0.6–0.8 s)와 shared 팔로워 첫 턴|
|bench(R4 측정점)|AR pp 649/643, tg 12.94; DFlash2 tg 26.78, d32768 51.30|642/639, 12.90; 26.75, 51.28|모든 측정점 0–1 % 안. 이식본이 같거나 조금 빠름|
|원본 suite|`long-context` 통과, `cache` 8경우 통과|—|원본 `79c2dcc5`도 같은 두 suite 통과|

- **컨테이너 최대 메모리**(cgroup peak, 페이지 캐시 포함): 이식본이 거의 모든 창에서 약 6–7 GiB 많다(예: 64K×8 26.6 / 19.6 GiB). 256K 창만 반대다(24.4 / 30.9 GiB).
  - 이식본 로그에는 원본 구조의 고정 확보분이 보인다(RAM backing 2.27 GB, 슬롯 pinned 1.68 GB).
  - 나머지는 anon과 file(페이지 캐시)을 나눠 기록하지 않아 확인하지 못했다.
- **SSD 쓰기:**
  - 256K는 18.3 / 17.5 GB, 32K는 5.2 / 6.6 GB.
  - 짧은 대화 창은 이식본이 적다(two8k 1.7 / 3.6, regress 2.0 / 6.4 GB). 운영 빌드는 idle write-back으로 더 자주 쓴다.
- **디스크 사용:** 창마다 최대 17 GB. 여유가 60 GB 아래로 내려가 regress 창부터 끝난 창의 cache-disk를 지웠다(기록됨).
- **창 판정:** `f-bench` 하나는 끝난 뒤 GPU busy 4 %로 `FAILED_no_fallback`이다(측정은 완료, 기존 패턴, guard 완화 없음). 나머지 19창은 COMPLETE.

## 원인 분석

1. **재시작 뒤 TTFT가 2–4배 (가장 큰 차이):**
   - 이식본은 원본 `Persist` 규칙(min_step)을 따른다. 그래서 둘째 턴 끝이 기존 checkpoint와 가까우면 디스크 게시를 거절한다(`persistence_refused reason=min_step`). 재시작 뒤에는 첫 턴 prompt 경계에서 복원하고, 그 뒤 약 500토큰을 다시 계산한다. 256K 깊이에서는 이 재계산이 약 5 s다.
   - 운영 빌드는 idle write-back(`--cache-idle-write-ms 2000`)으로 마지막 경계를 디스크에 써 둔다.
   - U5에서 "B3 idle write-back은 옮기지 않는다"고 한 판단이 실제 길이에서는 틀렸다.
2. **shared 첫 팔로워(7.9 s / 2.3 s):**
   - 원본은 공유 분기점을 두 요청의 공통 prefix로 배운다. 그래서 둘째 대화는 4,096 grid까지만 재사용하고, 셋째부터 7,174를 쓴다.
   - 운영 포크는 처음부터 7,171을 재사용한다.
   - 원본 계보 원칙(D4) 안에서 바꾸려면 원본과 논의가 필요하다.
3. **256K 다음 턴(1.79 / 1.28 s):** 원인을 확인하지 못했다. 재시작 경로가 아니라서 1번과 별개다.
4. **메모리 +6 GiB:** 위와 같다. 원본 구조의 고정 확보분이 약 4 GB이고, 나머지는 미확인이다.

## 판정

- **정확도·처리량:** 통과. 텍스트 동일, gold 전부, bench 동일, 원본 suite 통과.
- **운영 전환:** 아직 제안하지 않는다. 재시작 뒤 TTFT(운영 대비 2–4배)와 메모리(+6 GiB)가 운영 빌드보다 나쁘다.
- **U4:** 계속 보류한다. 256K 재시작 비용은 디스크 읽기(3.96 s)와 경계 재계산(약 5 s)이 차지한다. U4(RAM→슬롯 블록 매핑)로는 줄지 않는다.
- **다음 할 일 (추천, 사용자 확인 뒤):**
  1. B3 idle write-back을 원본 구조 위로 옮긴다. 원본 `Persist`·min_step과 맞물리게 하고, 유휴 상태일 때 live frontier checkpoint를 디스크에 게시한다.
  2. 메모리 차이를 cgroup `memory.stat`(anon과 file)으로 나눠 잰다.
  3. 256K 다음 턴 0.5 s 차이를 로그로 확인한다.
  4. 그 뒤 F 중 재시작이 있는 창(32K, 256K, regress, boundary, shared)만 다시 돈다.
- **F에 빠진 것:**
  - 원본 `cache-*` 계열 suite(edits, growth, depth, rotation, concurrency, shared-prefix)는 돌리지 않았다. PR 직전 F에서 main과 번갈아 돈다.
  - 원본 suite는 이미지의 클라이언트 패키지로 돌렸다. openai 2.6.1로, 원본 고정 버전 2.41.1과 다르다.

**후속(U5b, 2026-10-24):** 다음 할 일 1–4를 진행했다. 재시작·다음 턴 TTFT는 운영 수준이 됐고, 메모리 차이는 RAM checkpoint 예산을 행 사본으로 채우는 anon 8 GiB로 확인했다([U5b 결과](U5B-RESULTS.md)).

증거: `storage/evaluations/gufo-upstream-port-final-20261023/`
