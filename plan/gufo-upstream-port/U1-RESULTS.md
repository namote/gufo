# gufo 원본 이식 U1: 공유 prompt pass scratch (2026-10-23)

상위 문서: [이식 계획](PLAN.md), [U0 기준값](U0-RESULTS.md). 사용자 지시: "가자~"(U1 시작).

## 구현 (gufo `ai-tools/port-u1` `14bcd820`, 원본 `79c2dcc5` 위 커밋 1개)

이식한 것은 우리 Q2(`46020c3a`·`b15592c0`)다.

- **그대로 옮긴 것:**
  - `QwenPassScratchSet`, `QwenSharedPassScratch`: 4,096행, 2.15 GiB, 로드 때 페이지를 미리 잡음
  - `PassScratchLease`가 빌리는 pass: prompt, 검증, batched decode
  - 돌려줄 때 hidden 행을 세션 몫으로 옮김
  - 세션 몫: 64행
  - `GetScratchView` 행 수 검사
- **원본 구조에 맞춘 것:**
  - 원본에는 우리 paged KV의 `QwenKvPlacement`가 없다. 그래서 공유 scratch를 이렇게 넘긴다:
    - `QwenTextRunner`가 만든다.
    - `ContinuationAdapter`(생성자 인자 추가)를 거쳐 `QwenGpuExecutor`(생성자 인자 추가)를 지나 `QwenGpuArena`로 간다.
  - 원본에서 Qwen 27B executor를 만드는 서빙 경로는 `ContinuationAdapter::CreateSlot` 하나뿐이다.
  - `gufo bench`와 CLI(`model_loader`)는 공유 scratch 없이 지금처럼 세션마다 전체 set을 둔다.
- **플래그를 뺐다.** 원본 규칙은 "성공한 최적화는 별도 스위치 없이 기본값"이다. 그래서 `--no-shared-pass-scratch`를 없애고 항상 켰다. 비교는 같은 바이너리 A/B 대신 U0 빌드와 번갈아 돌리는 원본 방식으로 했다.
- **테스트 추가(`continuation_adapter_test`):**
  - FP16 boundaries·복원·guard 시나리오 전체를 공유 scratch 어댑터로 한 번 더 돈다.
  - 새 경우 `shared_pass_scratch`: 공유 어댑터 slot이 4,500토큰 prefill(pass 2개)을 한다. 그 사이 다른 slot이 prefill을 끼워 넣는다. 이어서 decode 8토큰을 한다. 공유 scratch가 없는 어댑터와 logits·토큰·마지막 hidden이 bit 단위로 같아야 한다.
- **포맷:**
  - 원본 `tools/ci/check-format.py`가 PASS했다. 우리 gufo 변경에 포맷 검사를 돌린 것은 처음이다.
  - 도구: clang-format 21.1.8을 `~/.cache/ai-tools/clang-format-venv`(uv)에 설치했다.
  - 버전 선택: 22·23은 원본 main 자체에서도 위반이 나서 21을 골랐다.
- **패치 보존:** `test/experiments/gufo_upstream_port_20261023/patches/u1/`

## 검증

PLAN-U1.json의 창 6개를 돌렸고 GPU 합계는 약 25분이다.

| 계층 | 결과 |
|---|---|
| T0 | 릴리스 빌드에서 새 경고 없음. gpu-test 대상 4개 빌드. 포맷 PASS. CPU ctest 126/126(모델 자산이 없는 8개는 skip) |
| GPU 단위 | 5종 통과: `continuation_adapter_test`(공유 경우 포함), 같은 테스트 `--draft`, `qwen27b_target_test`, `qwen_dflash_gpu_test`, `inference_backend_gpu_test`(HTTP parity·DFlash2·batched) |
| T2 | 21요청 텍스트가 U0와 전부 같고 gold 정답. 요청별 TTFT −0.3 ~ +2.9 %(5 % 안). 입장 4개+1 대기는 U0와 같음. cgroup 최고치 25.7 → 25.0 GiB |
| 기동 메모리(U0·U1 번갈아) | 기동 5초 뒤 host MemAvailable: 16K × 4는 80,842 → 87,406 MiB(**+6.4 GiB**), 64K × 8은 39,765 → 54,773 MiB(**+14.7 GiB**). Q2 실측(64K × 8 +14.5 GiB)과 같다 |
| T3 bench | 전 측정점이 U0 대비 5 % 안. AR pp2048 −0.3 %, pp2048 @ d8192 −2.2 %, tg 0 %. DFlash2 pp −0.8 %, @ d8192 −0.2 %, tg −0.1 %, tg @ d8192 −0.9 % |

### 기록할 것

- **u1-tests 창:** 결과가 `FAILED`다. 테스트 4개는 통과했지만, `inference_backend_gpu_test`가 시작되지 못했다.
  - 원인: 이 테스트에 잡은 1,500초가 창에 남은 1,400초보다 길었다. 테스트 결과가 아니라 계획 실수다.
  - 처리: 같은 테스트만 `u1-tests-backend` 창으로 따로 돌렸고 통과했다.
- **u1-t3 창:** 판정이 `FAILED_no_fallback`이다. 결과는 COMPLETE였고, 끝난 뒤 GPU busy 4 %(기준 3 %) 때문이다. U0와 같은 패턴이며 guard는 완화하지 않았다.
- **T2 cold prefill 차이:** +1–2 %(20–40 ms)로 고르게 늘었다. 5 % 안이지만 방향이 한쪽이다. U0는 약 1시간 전에 잰 값이다. 원본 PR 직전 최종점검 F에서 main과 번갈아 다시 재고, 넘으면 원인을 본다.

## 판정

- 출력이 같고 bench·TTFT가 5 % 안이다.
- 세션 하나에 약 2.1 GiB를 아낀다. 원본에서도 우리 Q2와 같은 이득이다.
- U1 통과로 판정한다. 원본 PR(U6)에 낼 첫 후보다.
- 증거: `storage/evaluations/gufo-upstream-port-u1-20261023/`(창 결과, 빌드·CPU 로그, `u0-startup` 대조)

## 다음

- U2(P1 블록 테이블 커널)는 [모델 확장](MODELS.md)의 풀·테이블 API 설계 검토를 먼저 거친다(PLAN 단계표).
- 결정 D5(블록 크기 256행 vs 64행)는 U2 전에 사용자 확인이 필요하다.
