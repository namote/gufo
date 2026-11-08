# gufo 원본 이식 U0: 원본 기준값 (2026-10-23)

상위 문서: [이식 계획](PLAN.md), [테스트 설계](TESTS.md). 사용자 지시: "gufo 원본 이식·제안 계획 U0 시작".

## 구성

- **원본:** gufo 작업 트리 `~/projects/gufo`에서 원본 main `79c2dcc5`로 브랜치 `ai-tools/port-u0`를 만들었다. 코드는 바꾸지 않았다.
- **툴체인(결정 D1):** 추천안대로 우리 ROCm 7.2.4 이미지(GCC 13.3)를 썼고, 호스트에 Nix는 설치하지 않았다.
  - 이미지: `ai-tools/gufo-eval:79c2dcc56f00`. 기존 `gufo_eval_20261020/Dockerfile`을 그대로 썼다.
  - 릴리스 빌드 269단계가 약 2분에 성공했다(T0 빌드). 로그: `tmp/test-results/gufo-upstream-port-20261023/build-release.log`.
- **실행 묶음:** `test/experiments/gufo_upstream_port_20261023/`
  - `window.py`: 기존 창을 복사했다. 이미지의 revision label이 PLAN의 `gufo_revision`과 같은지 검사한다.
  - `t2.py`: T2 경우와 T4s 밀도 시나리오를 돈다.
  - 그 밖의 파일: `PLAN-U0.json`, `LOCKS.json`(`LOCKS.py`로 생성), 이미지 fixture
  - 장문·회귀 시나리오와 bench는 `gufo_eval_20261020`의 `scenario.py`·`bench.py`를 다시 썼다.
- **서버 설정:** 원본에는 `--kv-pool-bytes`·`--cache-disk-block-bytes`·`--cache-idle-write-ms`가 없다. 그래서 원본 옵션만 썼다.
  - 슬롯마다 문맥 전체 KV를 미리 잡는다.
  - Qwen 27B는 component cache로 돈다. 응답 timings에 `component-cache-v1`이 찍혀 확인했다.
- **기준 출력:** 원본 API는 토큰 ID를 주지 않는다(logprobs 미지원). 그래서 기준은 greedy 출력 텍스트와 usage 토큰 수로 기록했다.
  - 생성은 `PROMPTS.json`의 `max_tokens` 384를 그대로 썼다. gold JSON이 약 250토큰이라서다. TESTS.md의 "32토큰 이하"와 다르며, 이 차이를 여기 기록해 둔다.
- **GPU 사용:** 창 9개를 차례로 돌렸고 합계 약 23분 걸렸다. 창마다 시작 전 GPU quiet(KFD 0, busy 0 %)를 확인했고, 끝난 뒤 KFD 0, 새 커널 오류 0, 남은 컨테이너 0이었다.

## 결과 요약

| 창 | 내용 | 결과 |
|---|---|---|
| u0-t2 | T2 경우 A–H. 16K × 4, DFlash2, think off, mmproj. 재시작 1회, 이어서 AR 서버 | **전부 통과**. 21요청 gold 정답, cache 끈 재요청과 텍스트 동일, AR 텍스트가 DFlash2와 동일 |
| u0-t3 | bench pp2048·tg128, 깊이 0·8K, 3회 | 기록함. 창 판정은 FAILED(아래 "창 판정") |
| u0-t4s-density4 | 16K × 4 동시(약 14.3K prompt) → 다음 턴 | 4/4 정답, 다음 턴 재사용 14,253 |
| u0-t4s-density8 | 16K × 8 동시 → 다음 턴 | 8/8 정답, 다음 턴 재사용 14,253 |
| u0-t4s-long16k | 16K 장문 cold → 다음 턴 → 재시작 → 셋째 턴 | 정답, cold와 텍스트 동일 |
| u0-t4s-two8k | 8K 대화 2개, 세션 1 | 정답, cold와 동일 |
| u0-t4s-regress | 1K 대화 6개, 세션 2 | 정답, cold와 동일 |
| u0-t4s-boundary | 1K 대화 3개, 둘째 턴 +1K | 정답, cold와 동일 |
| u0-t4s-shared | 8K system 공유 대화 6개, 세션 2 | 정답, cold와 동일. 창 판정은 FAILED(아래 "창 판정") |

운영 구성(DFlash2·이미지·think off·디스크 캐시·재시작)이 원본 `79c2dcc5`에서 지금 정상으로 돈다. 계획 "위험"에 적은 걱정(원본 Qwen 27B component cache가 아직 검증 중)은 짧은 경우와 16K 축소판에서는 나타나지 않았다.

### 창 판정

u0-t3·u0-t4s-shared 두 창은 결과가 COMPLETE인데 창 판정이 `FAILED_no_fallback`이다.
- 원인: 끝난 직후 GPU busy가 4 %였다. guard 기준은 3 % 이하다.
- 그 밖의 확인: KFD 0, 새 GPU 오류 없음, fence 정상.
- 이전 평가에서도 같은 패턴이 있었다. guard는 완화하지 않고 판정을 그대로 남긴다. 수치는 기준값으로 쓴다.

## T2 기준 (u0-t2)

| 경우 | 기준 |
|---|---|
| A cold | prompt 1,040, TTFT 2.08 s, 2/2 정답 |
| B 다음 턴 | 1,286 중 1,253 재사용(live), TTFT 0.33 s |
| C 다른 슬롯 복원 | 대화 4개를 더 채운 뒤 c0. 1,532 중 1,286 재사용, TTFT 0.80 s(cold 2.73 s), 텍스트 동일 |
| D 공통 system 2개 동시 | 리더가 prefill 5,053, 팔로워가 4,930 재사용. 둘 다 TTFT 9.74 s(팔로워는 리더를 기다림). cold와 텍스트 동일 |
| E 입장 | 세션 4개에 스트림 5개, 각 `max_tokens` 16,000. 4개가 1.0–1.4 s에 첫 토큰을 냈고, 5번째는 앞 스트림을 닫은 뒤(7.47 s) 시작했다. 원본은 슬롯 수만큼만 입장한다. 블록 예약은 없다. 닫은 뒤 요청은 A와 텍스트 동일 |
| F 재시작 디스크 복원 | 1,532 중 1,033 재사용, TTFT 1.43 s(cold 2.70 s), 텍스트 동일 |
| G 이미지 | "4827" 정답(prompt 251) |
| H AR | A 2건과 이미지가 DFlash2와 텍스트 동일. AR 생성 wall 18.5 s, DFlash2 5.3 s |

- host 최대 메모리(cgroup): 25.7 GiB
- 디스크 캐시: 3.9 GB

## T3 기준 (u0-t3, `gufo bench`, 3회)

| 모드 | pp2048 | tg128 | pp2048 @ d8192 | tg128 @ d8192 |
|---|---|---|---|---|
| AR | 650.87 ± 0.90 | 12.95 ± 0.00 | 607.45 ± 0.81 | 12.63 ± 0.00 |
| DFlash2 | 603.91 ± 6.36 | 26.95 ± 0.11 | 558.07 ± 0.11 | 61.47 ± 0.02 |

- 공개 카드 수치(AR pp2048 663, DFlash2 607)와 같은 수준이다.
- DFlash2 tg는 bench 합성 입력의 draft 수락률에 따라 깊이마다 크게 다르다. 그래서 비교는 같은 측정점끼리만 한다.

## T4s 기준

| 창 | TTFT·재사용 | host 최대 | 디스크 |
|---|---|---|---|
| density4 | T1 동시 4개 TTFT 83–108 s(wall 118 s). T2 재사용 14,253/14,499, TTFT 1.2–1.6 s | 25.0 GiB | 4.7 GB |
| density8 | T1 동시 8개 TTFT 162–220 s(wall 238 s). T2 재사용 14,253, TTFT 2.1–4.0 s | 26.6 GiB | 9.3 GB |
| long16k | cold 16,386 → 27.9 s. 다음 턴 16,599 재사용 0.38 s. 재시작 뒤 16,379 재사용 2.36 s(cold 29.2 s) | 19.7 GiB | 1.3 GB |
| two8k | T2 8,194/8,187 재사용 0.8–0.9 s. 재시작 뒤 8,187 재사용 1.85 s(cold 14.8 s) | 19.3 GiB | 1.6 GB |
| regress | T2 c0–c5 모두 1,033 재사용(prompt 1,286). 재시작 뒤도 1,033 | 19.7 GiB | 1.9 GB |
| boundary | T2·T3 모두 1,033 재사용(prompt 2,253·2,499) | 17.9 GiB | 0.9 GB |
| shared | 팔로워 T1 7,174 재사용(system 약 8.2K). T2·T3에서 c0만 8,201(live), 나머지는 7,174 | 17.9 GiB | 1.5 GB |

### 원본 동작에서 보인 것 (U5 재측정 대상)

- **슬롯에서 밀려난 대화의 복원 깊이:** 1,024 grid checkpoint(1,033)로만 돌아간다. regress·boundary·F 경우에서 그랬다. 직전 턴 끝까지 돌아가지 못해 끝부분 250–1,200토큰을 다시 계산한다. 우리 B3(축출 시 write-back)·Q4(끝 checkpoint)가 겨냥한 차이이며, U5에서 원본 위로 이득을 잰다.
- **공유 system의 팔로워 재사용:** 7,174토큰이다(system 끝 약 8.2K보다 앞). P4 "system 경계 checkpoint"와 같은지 U5에서 확인한다. → U5에서 확인: system prompt가 약 7.17K라 7,174가 system 경계 그 자체다(8.2K는 사용자 메시지까지 포함한 길이, [U5 결과](U5-RESULTS.md) ②).
- **밀도:** 16K × 8에서도 원본은 문맥 전체 KV를 미리 잡고 돈다. 16K라서 메모리 문제가 드러나지 않는다. 256K × 4·64K × 8의 실제 메모리 차이(P2의 근거)는 최종점검 F 항목이다.

## 다음

- U1(Q2 공유 pass scratch 이식)은 사용자 확인 뒤 시작한다.
- 기준 텍스트는 `tmp/test-results/gufo-upstream-port-20261023/u0-*-a01/receipt/run/RESULT.json`에 있다. 원본 최신으로 다시 맞출 때만 새로 기록한다.
- 운영은 그대로 `210c7633`이다. compose·Manager는 건드리지 않았다.
