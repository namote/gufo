# gufo 원본 이식 U4d 결과: DFlash2 prompt feature pinned 버퍼 공유 (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [U4d 설계](U4D-DESIGN.md), [anon 바닥](ANON-FLOOR-RESULTS.md), [U4c 결과](U4C-RESULTS.md).

## 구현

- gufo `ai-tools/port-u4d` `d42b292c`(`852b6c72`에서 분기)이다. 내용은 [설계](U4D-DESIGN.md) "무엇을"과 같다.
- CPU 테스트 127/127, clang-format 통과다.

## GPU 창

- `final-f10-a01`(`--set f10`)이다. RAM 예산 2 GiB, 256K × 4, 풀 40 GiB, 디스크 64 GiB.
  - run 직전에 `fstrim`(26.3 GiB)을 했다.
  - 순서: GPU 단위 테스트 → 6개 창을 운영 `210c7633`(`fb-*`)과 번갈아 → T2 → `memprobe`.
- 15개 창 모두 COMPLETE, health PASS다.

**GPU 단위 테스트:** 모두 PASS다.
- 테스트: `qwen_prompt_hidden_buffer_test`(새 `TestSharedPages` 포함), continuation adapter(DFlash2 draft 포함), target, DFlash GPU, inference backend.

**정확성**
- 텍스트는 모든 창에서 같다(운영 같은 run, U5b a06, U4c 2G f9).
- gold는 모두 통과했다. T2는 21/21 같다(a06, f9 대비).
- 서버 로그에 `logic_error`, capture 불완전 오류, 예외가 없다.
- 시작 로그는 `slot_pinned_bytes=419430400`이다. 슬롯 4개가 400 MiB를 한 번만 잡는다.

**메모리(GiB)**

memprobe는 준비 직후 10 s와 대화 3개 뒤에 잰다.

|항목|운영|U4c 2G|U4d 2G|
|---|---|---|---|
|memprobe 준비 직후 anon|15.63|19.47|**18.30**|
|memprobe 대화 3개 뒤 anon|—|—|18.37|
|창별 최대 anon|16.3–17.1|19.5–19.6|**18.37–18.55**|

- 준비 직후 anon은 **−1.17 GiB**로, 설계 예상과 같다(1.56 GiB 중 공유 400 MiB만 남음).
- 운영과의 차이 +2.67 GiB는 다음과 같다([anon 바닥](ANON-FLOOR-RESULTS.md)).
  - 공유 pinned 0.39
  - RAM 예산 안 backing 미리 커밋 1.77
  - 기타 약 0.5
- units 창 28 GiB는 GPU 테스트 바이너리 값이다. 서버 값이 아니다.

**TTFT(초)**

운영은 같은 run의 `fb-*`다.

|항목|운영|U4c 2G(f9)|U4d 2G|
|---|---|---|---|
|shared 첫 팔로워 c1|2.30|2.29|2.30|
|shared 둘째 턴(6개)|0.87 × 6|0.38–1.60|**0.38–0.46**|
|shared 합(19)|50.8|50.1|**46.0**|
|long32k 합(8)|245.0|247.3|245.6|
|32K 재시작 뒤 T3|1.17|1.06|0.86|
|256K 첫 턴(cold prefill)|883.4|898.4|901.4|
|256K 재시작|4.46|6.31|4.54|
|regress 합(20)|26.7|22.5|21.7|
|boundary 합(11)|24.4|24.7|22.8|
|two8k 합(8)|59.9|60.2|57.9|
|T2 합(21)|—|66.6|66.3|

- **TTFT 손해는 없다.** prefill pass는 원래 한 번에 하나다. 모든 합이 f9와 같거나 작다.
- **디스크 복원 편차가 없었다.** TRIM한 SSD에서 shared 둘째 턴이 모두 0.38–0.46 s였다. 운영 RAM 경로는 0.87 s다.
  - 256K 재시작은 4.54 s로 운영(4.46 s)과 같다.
  - [디스크 편차 결과](DISKREAD-RESULTS.md)와 맞는다.
- 256K cold prefill은 이식본이 운영보다 약 2 % 느리다(901 / 883 s).
  - a06 900, f9 898 s에서도 같았다. U4d와는 관계없다.

## 판정

- **채택:** 정확성이 같고, 준비 직후 anon이 −1.17 GiB다. TTFT 손해는 없다.
- **운영과 비교한 이식본(U4d, RAM 예산 2 GiB):**
  - 텍스트·gold·T2는 같다.
  - TTFT는 shared·regress·boundary·two8k·재시작에서 운영과 같거나 빠르다. 256K cold prefill만 약 2 % 느리다.
  - anon은 +2.1–2.7 GiB 많다(최대 18.4–18.6 / 16.3–17.1). 대부분은 RAM 예산 안의 backing 미리 커밋이다.
  - SSD 쓰기는 창마다 운영의 1.1–1.8배다(D4 결정 비용). TRIM이 하루 한 번 돈다.
- 운영 전환 제안 여부와 D2–D4·D7–D9는 사용자가 정한다.

증거는 `storage/evaluations/gufo-upstream-port-final-20261023/`의 `final-f10-a01`(`.log` 포함)이다. cache-disk 14개(91.9 GiB)는 지우고 `final-f10-a01/CACHE-REMOVED.json`에 기록했다. 그 뒤 `fstrim`으로 117.7 GiB를 풀었다.
