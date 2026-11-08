# gufo 원본 이식 U4b 결과: 긴 prompt 끝 앞쪽 tail checkpoint (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [U4b 설계](U4B-DESIGN.md), [U4 결과](U4-RESULTS.md).

## 구현

- gufo `ai-tools/port-u4b` `9a5cacee`(`17ef4326`에서 분기)이다. 운영 `PlanTailCheckpoints` 규칙을 원본 `PlanCaptures` 선택 후보로 옮겼다. 내용은 [설계](U4B-DESIGN.md) "무엇을"과 같다.
- CPU 테스트는 127/127 통과다. 새 검사 2개는 수정을 빼면 실패하는 것을 확인했다.

## GPU 창

- 창은 U4와 같다(256K × 4, 풀 40 GiB, RAM 8 GiB, 디스크 64 GiB).
  - `final-f6-a01`: 운영 `210c7633`과 번갈아 돌렸고, T2를 포함한다(`--set f6`).
  - `final-f7-a01`: RAM 예산 2 GiB, 이식본만 돌렸다(`--set f7`).
- 모델 GPU 코드는 U4 이후 바뀌지 않았다. 그래서 GPU 단위 테스트는 다시 돌리지 않았다(`final-f4-b01` PASS).

**정확성**
- 텍스트는 모든 창에서 운영·U5b(a06)·U4와 같다.
- T2는 21/21 같다(f6, f7 모두).

**TTFT(초)**

운영 값은 같은 run(f6)의 `fb-*`이다.

|항목|운영|U4 8G|U4b 8G|U4 2G|U4b 2G|
|---|---|---|---|---|---|
|shared 첫 팔로워 c1|2.29|7.86|**4.65**|14.34|**4.66**|
|shared 두 번째 이후 c2–c5|2.30–2.48|2.31–2.33|2.31–2.32|2.30–2.33|2.30–2.33|
|shared 합(19)|51.9|56.5|53.3|60.4|**50.8**|
|shared 첫 prompt c0(8,208 prefill)|14.49|13.83|14.24|13.45|14.06|
|long32k T1(32K prefill)|59.27|59.37|58.65|59.62|59.81|
|long256k T1(256K prefill)|883.0|902.3|903.3|898.9|899.6|
|32K 재시작 뒤 T3|1.15|0.84|0.85|3.91|2.13|
|256K 재시작|6.53|8.53|4.66|7.64|5.12|
|regress 합(20)|26.9|22.6|22.2|23.2|22.4|
|boundary 합(11)|25.1|23.2|25.3|24.6|23.6|
|two8k 합(8)|60.7|58.2|58.6|59.0|60.1|
|T2 합(21)|—|66.6|66.5|—|65.8|

- **첫 팔로워는 절반쯤 고쳐졌다.** 7.86 → 4.65 s이고, 2 GiB에서는 14.34 → 4.66 s이다.
  - 첫 팔로워는 앞 대화의 tail 6,153을 RAM에서 복원하고(7 ms) 2,055토큰을 prefill한다.
  - 2 GiB에서 shared 합은 운영보다 작다(50.8 / 51.9 s).
- **운영 수준(2.3 s)에는 못 미쳤다.** 원인은 아래 "운영이 빠른 진짜 이유"에 있다.
- pass 분할 비용은 보이지 않는다.
  - 첫 prefill 차이는 −0.7 ~ +1.0 s로 run 편차 안이다(shared c0 +0.4 s, 32K −0.7 s, 256K +1.0 s / 900 s).
- 256K 재시작은 4.66 s다. U4 b01의 8.53 s는 디스크 읽기 편차였다는 해석과 맞다([U4 결과](U4-RESULTS.md)).
- boundary 7–9번(0.3 → 1.0 s)은 U4b 때문이 아니다.
  - 모든 빌드가 같은 2,466토큰을 복원하고 33토큰만 prefill한다.
  - 운영(0.78–0.99 s)·U5b(1.03–1.19 s)도 1 s대다. U4 b01의 0.3 s가 좋게 나온 쪽이다.

**메모리**

최대 anon(GiB)이다.

|창|운영|U4b 8G|U4b 2G|
|---|---|---|---|
|long32k|17.0|24.7|19.6|
|long256k|16.2|21.1|19.6|
|two8k|16.7|24.1|19.5|
|regress|16.3|24.3|19.5|
|boundary|16.4|22.8|19.5|
|shared|16.4|24.3|19.6|
|t2|—|24.9|19.6|

- U4와 같다. 예산이 상한이고 이식본은 그 상한까지 채운다. tail checkpoint는 그 안에서 자리를 받는다.

## 운영이 빠른 진짜 이유 (설계 정정)

- **설계의 원인 분석이 틀렸다.** [설계](U4B-DESIGN.md)는 운영의 첫 팔로워가 tail checkpoint를 복원한다고 봤다.
  - 운영의 tail도 같은 규칙이라 7,177·6,153·5,129에 잡힌다. 분기점은 7,174라서 7,177은 쓸 수 없다.
- **운영은 system prompt 끝에 checkpoint를 둔다.**
  - 운영 `210c7633` `text_model_runner.cpp:2751`: `system_prefix_tokens`(system 메시지만 렌더링한 토큰 수, `inference_backend.cpp:5578`)가 `shared_prefix_min_tokens`(128) 이상이면 `PlanBoundary(system_prefix_tokens, branch=true)`.
  - continuation 우선순위의 RAM checkpoint이고, 디스크에도 shared prefix로 저장한다.
  - 이 창에서는 7,171이다. 첫 prompt c0에서 `cache_shared_prefix_snapshots=1`, `disk_cache stored reason=saved tokens=7171`이다.
  - 팔로워 c1–c5는 모두 7,171을 복원하고 1,037토큰을 prefill한다.
- 이식본(원본 기반)에는 이 경계가 없다. 원본도 없다. 그래서 첫 팔로워는 tail 6,153을 쓰고, c1이 prefill하는 동안 분기점 7,174를 배운다.
- **tail은 여전히 쓸모가 있다.** 분기점이 메시지 경계가 아닌 경우(같은 문서에 다른 질문)를 덮는다. 운영도 두 가지를 같이 쓴다.

## 판정

- **채택:** 정확성이 같고, 첫 팔로워가 7.9 → 4.7 s(2 GiB에서 14.3 → 4.7 s)로 빨라진다. 다른 창의 손해는 없다.
- **1번(첫 팔로워 지연)은 아직 끝나지 않았다.** 다음은 U4c: system prompt 경계 checkpoint를 원본 구조로 옮긴다.

증거는 `storage/evaluations/gufo-upstream-port-final-20261023/`의 `final-f6-a01`, `final-f7-a01`(각 `.log`)이다. 각 창의 cache-disk는 `final.py`가 공간이 부족할 때 지웠고(FINAL.json `cache_disk_removed`), 나머지는 `final-f6-a01/CACHE-REMOVED.json`에 기록하고 지웠다.
