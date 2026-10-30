# W9 결과: 투기 디코딩 개선 (2026-10-29)

상위 문서: [SPEC-DECODE.md](SPEC-DECODE.md)(설계), [W9-BRIEF.md](W9-BRIEF.md). 표기: **[확인]**은 측정이나 코드에서 확인한 것, **[짐작]**은 추론이다.

## 상태

- 완료:
  - 1단계 설계(`adf9c5d3`).
  - 도구(`f7fc9aa9`, `8d227e90`):
    - `tools/qwen-flash/spec/{build_suites.py, spec_stats.py, suite_k.json, suite_d.json}`
    - ai.tools `rq_serving.py`의 `record_responses` 옵션(커밋하지 않음, sha256 `c0f1b5eacd2c5e9b83d370cade0e4950f893d26dd0b27070e1f4b7ddb58e4274`, 원본 `92a79dc7…` 백업은 W9 세션 scratchpad에 있음)
  - 27B R0 종류별 재집계.
  - **M0 창 2개(mtp, ar).**
- 완료(2026-10-30): M1 trace 도구와 시뮬레이터(`ba85850b`, 수정 `09b10ccd`), 빌드, M1 창. A+K 81/81 재생 일치. 정책 비교 결과가 아래 "M1 결과"에 있다.
- 다음:
  - M1b(D 8개 다시 trace, 창 약 15분)
  - I1 구현 승인

## 도구 검증 [확인]

- **suite_k:** Kyojin 원문과 같다. `build/w9-src/kyojin`의 `tools/bench.py`, `tools/qwen/spec_prompts.py`를 AST로 읽어 17/17이 바이트 단위로 같았다.
- **프록시 자체 시험:** 가짜 SSE 서버(0.2 s 간격 3 이벤트)로 시험했다. 클라이언트 도착 시각이 0.0/0.2/0.4 s였으므로 버퍼링 없이 흘려보낸다. 기록과 `/metrics` 저장도 확인했다.
- **spec_stats:**
  - 합성 기록 2건으로 사이클당 토큰, AR 비율, lookup 비율을 손계산과 대조해 같았다.
  - R0 27B 보고서의 "all" 행이 R0-RESULTS 값(0.453, 0.663, 29.5, 2.28×)과 같다.
- **M0 첫 분석 실패와 수정:** serving_bench warmup 요청(`max_tokens` 16)이 기록에 섞여 결정성 검사가 실패했다. 측정 요청과 예산이 다른 기록은 건너뛰도록 고쳤다(`8d227e90`, 건너뛴 수를 출력에 남김. A와 K에서 각 1건).

## 27B R0 종류별 재집계 (CPU, 기존 `rq0-accept-a01`, UD-Q4_K_M DFlash2 대 AR)

| 종류 | n | 수락률 | 토큰당 수락 | DFlash2 tok/s | AR 대비 |
|---|---:|---|---|---:|---|
| repetitive | 6 | 0.822 [0.703, 0.946] | 0.837 | 58.8 | 4.54× [3.79, 5.50] |
| reasoning | 5 | 0.560 | 0.745 | 38.1 | 2.94× [2.32, 4.02] |
| code | 9 | 0.544 | 0.729 | 36.0 | 2.79× [2.27, 3.69] |
| d4_summary | 5 | 0.501 | 0.712 | 33.5 | 2.60× |
| structured | 5 | 0.477 | 0.678 | 30.7 | 2.38× |
| **all** | 64 | 0.453 [0.409, 0.503] | 0.663 | 29.5 | 2.28× [2.07, 2.53] |
| summarization | 5 | 0.382 | 0.633 | 26.9 | 2.10× |
| expository | 7 | 0.409 | 0.634 | 27.2 | 2.09× |
| d4_code_explain | 5 | 0.386 | 0.624 | 26.7 | 2.06× |
| multilingual | 2 | 0.367 | 0.604 | 25.6 | 1.97× |
| instruction | 5 | 0.347 | 0.562 | 23.2 | 1.79× [1.42, 2.32] |
| creative | 5 | 0.285 | 0.489 | 20.0 | 1.55× [1.19, 1.86] |
| d4_instruction | 5 | 0.269 | 0.465 | 19.1 | 1.47× [1.32, 1.66] |

- 64/64 완성이 AR과 바이트 단위로 같다(greedy 동일성).
- 약한 곳은 창작과 지시문(1.5–1.8×)이다. 27B는 이미 Kyojin 27B(3090, 1.55–1.59×)보다 배율이 높다.
- 보고서: `~/projects/ai.tools/storage/evaluations/w9-m0-20261029/r0_udq4km_categories.json`(`spec_stats.py summary <dflash2 bench-0> --ar <ar bench-0>`).

## M0: Flash-Next 기준선 (2026-10-29)

### 창 [확인]

- **계획:** `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-W9-M0.json`(sha256 `1860e7fc…`, LOCKS.json에 포함).
- **이미지:** `ai-tools/gufo-eval:rqw3-base-rocm101`(id `0afd4cea`, gufo `c94572b3`, nix ROCm 10.1 / Clang 24 release). c94572b3에서 HEAD 사이에 Flash-Next, speculative, serve 소스 변경이 없다.
- **모델:** 외장 exFAT(`/run/media/m2on/T5 EVO/…/unsloth-Qwen3.8-Flash-Next-GGUF/`, 읽기 전용). UD-Q4_K_XL 4분할 + `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf`이고, 감독이 sha256 17/17을 확인했다(외장 `PROVENANCE.md`, `VERIFY-SHA256.log`).
- **서버:** `--think off --context 49152 --sessions 1`. mtp는 `--speculative mtp --mtp-model …`(draft-tokens 기본 7), ar는 `--speculative off`다.
- **bench:** C=1, `--max-tokens 256`, greedy, `--no-cache-prompt`.
  - A(`accept_prompts.json` 64개)와 K(`suite_k.json` 17개): warmup 1, 반복 mtp 2 / ar 1.
  - D(`suite_d.json` 8개, 실제 prompt 7,274–7,509 / 27,158–39,256 토큰): warmup 0, 반복 mtp 2 / ar 1.
- **결과 폴더:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/w9-m0/{w9-m0-fn-mtp, w9-m0-fn-ar}/`. 두 창 모두 COMPLETE, health_PASS, 재시도 없음.
  - 서버 준비: mtp 285 s, ar 328 s. **OOM 없음.**
  - bench 시간: mtp 516/183/333 s, ar 425/144/191 s.
- **첫 실행 시도:** `window.py`를 스크립트로 불러 import 오류로 preflight 전에 멈췄다(컨테이너 없음, GPU 사용 없음). 빈 출력 폴더는 지웠고, 로그는 `w9-m0.launch-error-1.log`로 남겼다. 모듈로 다시 불렀다(`python3 -m test.experiments.gufo_upstream_port_20261023.window`).
- **분석:** `spec_stats.py summary`. 결과는 `~/projects/ai.tools/storage/evaluations/w9-m0-20261029/summary-{0,1,2}.json`(sha256 `2340d4cd…`, `6d173d31…`, `0ed0dee7…`).
- **원시 파일 sha256:**
  - mtp bench `7d2701f3…`/`113a1fbd…`/`b23386ae…`, responses `7e98be9a…`/`3b2a913f…`/`640e013d…`
  - ar bench `87605f46…`/`f205c5da…`/`9eb75343…`
- **tok/s는 참고용이다.** PLE 표를 USB에서 읽는다. 다만 AR tg는 27.0–28.9로, 공개값 28.29(NVMe)와 차이가 작다 [짐작: 오염은 수 % 안].

### 결과 (bootstrap 2000회, 프롬프트 단위 95 % 구간)

**세트 K (Kyojin 비교 프롬프트):**

| 종류 | n | 수락률 | 사이클당 드래프트 | 사이클당 토큰 | lookup 비율 / 수락 | MTP tok/s (참고) | AR tok/s (참고) | AR 대비 | AR과 동일 |
|---|---:|---|---:|---|---|---:|---:|---|---|
| chat | 3 | 0.659 | 1.39 | 1.92 [1.80, 2.04] | 0 | 40.5 | 28.6 | 1.42× [1.35, 1.46] | 3/3 |
| prose | 3 | 0.602 | 1.51 | 1.91 [1.83, 2.03] | 0 | 38.6 | 27.3 | 1.41× [1.28, 1.58] | 3/3 |
| code | 3 | 0.791 | 3.18 | 3.52 [3.32, 3.86] | 0.016 / 0.45 | 56.4 | 27.7 | 2.04× [1.97, 2.12] | 3/3 |
| multi | 4 | 0.615 | 1.60 | 1.99 [1.75, 2.30] | 0 | 38.3 | 27.6 | 1.39× [1.30, 1.46] | 4/4 |
| copy | 4 | 0.790 | 5.23 | 5.13 [4.70, 5.91] | 0.413 / 0.79 | 69.6 | 28.9 | 2.41× [2.28, 2.64] | 4/4 |
| **all** | 17 | 0.691 [0.647, 0.734] | 2.00 | 2.38 [2.08, 2.83] | 0.080 / 0.78 | 44.7 | 27.9 | 1.60× [1.47, 1.77] | 17/17 |

**세트 A (64개) 전체:**

- 수락률 0.667 [0.638, 0.698], 사이클당 드래프트 2.54, 사이클당 토큰 2.70 [2.48, 2.96]
- AR 토큰 비율 0.031, lookup 0.039/0.64
- tok/s 45.3(AR 27.0), **1.67× [1.53, 1.85]**, AR과 동일 64/64

종류별 AR 대비:

| repetitive | reasoning | code | structured | d4_summary | expository | summarization | d4_code_explain | instruction | d4_instruction | creative | multilingual |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 2.38× | 2.00× | 2.17× | 1.70× | 1.65× | 1.54× | 1.53× | 1.51× | 1.49× | 1.43× | **1.21×** | **1.09×** |

- creative는 사이클당 드래프트 1.46개이고, AR 토큰 비율이 0.189다(백오프).

**세트 D (문맥 깊이):**

| 깊이 | n | 수락률 | 사이클당 드래프트 | 사이클당 토큰 | MTP tok/s (참고) | AR tok/s (참고) | AR 대비 | 동일 |
|---|---:|---|---:|---|---:|---:|---|---|
| 약 7.4K | 4 | 0.637 | 1.71 | 2.09 [1.83, 2.54] | 39.0 | 27.1 | 1.44× [1.28, 1.68] | 4/4 |
| 27–39K | 4 | 0.636 | 1.63 | 2.04 [1.77, 2.45] | 39.8 | 26.4 | 1.51× [1.44, 1.58] | 4/4 |

### Kyojin과 비교 [Kyojin 수치는 공개 문서, 측정 조건이 다름]

| 종류 | gufo HEAD MTP (USB, 참고) | gufo v0.8.0 (Kyojin 측정) | Kyojin 카드 1.5.10 | Kyojin SERVE.md (구 팩, plain 27.3) |
|---|---:|---:|---:|---:|
| chat | 40.5 (1.42×) | 37.3 | 52.8 | 42.2 (1.55×) |
| prose | 38.6 (1.41×) | 32.5 | 47.5 | 38.7 (1.42×) |
| code | 56.4 (2.04×) | 52.3 | 73.2 | 52.2 (1.91×) |
| multi | 38.3 (1.39×) | — | — | 42.6 (1.56×) |
| copy | 69.6 (2.41×) | — | — | 54.7 (2.01×) |
| 8K / 32K 문맥 | 39.0 / 39.8 | 35.3 / 31.9 | 52.2 / 51.1 | — |

- 같은 카드 프롬프트(K의 chat/prose/code = Kyojin `DECODE_PROMPTS`)에서 HEAD는 Kyojin이 잰 v0.8.0보다 9 %, 19 %, 8 % 빠르다. 32K 문맥에서는 25 % 빠르다 [확인: 같은 프롬프트, 측정 방법은 다름. Kyojin은 클라이언트 시간 중앙값이고 우리는 서버 decode_ms 합이다].
- **그래도 Kyojin 1.5.10 카드보다 1.23–1.34× 느리다(8K 1.34, 32K 1.28).** 배율만 보면 구 팩 SERVE.md(plain 27.3)와 chat/prose/multi에서 비슷하거나 약간 낮고, code/copy는 높다. Kyojin 1.5.10 카드의 plain 기준은 공개되지 않았다.
- [짐작] 따라서 남은 격차는 (1) Kyojin 쪽 plain decode 향상(1.5.10 팩과 커널), (2) chat/prose/multi/creative에서 짧은 드래프트(사이클당 1.4–1.6개)다.

### 읽은 것

1. **[확인] greedy 출력 동일성:** 89/89 프롬프트(A 64, K 17, D 8)에서 MTP 완성이 AR과 바이트 단위로 같다. 반복 사이 드래프트 수도 모두 같다(기록 2건/프롬프트, 결정성).
2. **[확인] 약점은 chat/prose/multi/creative/긴 문맥이다.**
   - 수락률은 0.60–0.66으로 나쁘지 않다. 그런데 컨트롤러가 **사이클당 1.4–1.7개만** 드래프트한다. 사이클당 토큰은 1.9–2.1이다.
   - code/copy/reasoning/structured는 3–5개를 드래프트하고, 사이클당 토큰이 3.2–5.1이다.
   - [짐작] 깊이별 EMA는 "이 요청의 평균"만 보므로 확신이 높은 구간(관용구, 반복 표현)에서도 길이를 늘리지 못한다. 반대로 확신이 낮은 토큰에서도 정해진 길이를 다 쓴다. 토큰별 확신도 정지(I1)가 겨냥하는 바로 그 경우다. 이득의 크기는 M1에서 잰다.
3. **[확인] lookup은 copy에서만 크다**(드래프트의 41 %, 수락 0.79). A 전체는 3.9 %다. I2의 기대 이득은 작다.
4. **[확인] 백오프:** creative는 출력 토큰의 18.9 %를 AR로 냈다. 다른 종류는 0–6 %다.

## M1 계획 (승인 대기)

### 목적

토큰별로 MTP 사슬 깊이 7의 예측과 확신도를 teacher forcing으로 기록한다. 그 기록으로 길이 정책과 lookup 정책을 **GPU 없이** 비교한다.

### 도구 (승인된 경계 `tests/models/qwen38_flash_next/`, `tools/qwen-flash/`)

1. **`qwen38_flash_next_gpu_probe --mtp-trace`**(새 모드, 운영 소스는 바꾸지 않음):
   - suite JSON의 각 프롬프트를 서버와 같은 Qwen chat template(think off)로 토큰화하고 prefill한 뒤, greedy AR로 최대 256토큰을 낸다.
   - 위치 t마다 다음을 한다.
     - Session과 같은 catch-up(`engine.cpp:623-667` `DraftReplay`/`DraftCatchUp`과 같은 행 선택)을 한다.
     - 그 위에서 `MtpForward(-1)`를 6번 재귀 호출해 깊이 1..7의 후보 top-64 logits를 받는다(`MtpOutput.candidates`, 이미 있는 출력이라 커널 변경 없음).
     - 깊이마다 (argmax id, top-64 softmax p, top-2 차이)를 기록한다.
     - `MtpRewind`로 되돌린다.
   - 헤드 선택은 운영 `MtpDraftVocab` 규칙(`draft_vocab.hpp`)을 그대로 쓴다. `--full-head` 진단 선택지도 둔다.
   - 출력: 프롬프트별 토큰열, 프롬프트 토큰 수, 위치별 7깊이 기록(바이너리 + JSON 머리).
2. **`tools/qwen-flash/spec/spec_sim.py`(새 파일):**
   - `prompt_lookup.hpp`(Find/Observe/resume)와 `MtpLengthController`(`mtp_policy.hpp`)를 Python으로 옮긴다.
   - 비용은 `kMtpCycleMilliseconds`(C1, 문맥 보간)를 쓴다.
   - 정책을 재생해 프롬프트별 draft_n, accepted, rounds, lookup_n과 예상 ms를 낸다.
   - 후보 정책:
     - P0: 현재 정책
     - P1: Kyojin `TierRule`(0.6/0.3/0.95/0.85, dsplit 3)
     - P2: 비용 인식 확신도 규칙. 커밋된 이력으로 p를 수락 확률로 보정하는 bin을 만들고(시간 비의존), 한계 기대 토큰 대 한계 ms로 계속할지 정한다.
     - P3: 사후 최적 길이(상한)
3. **검증 (합격 기준):**
   - (a) trace 토큰열과 프롬프트 토큰 수가 M0 서버 완성 sha256, `prompt_n`과 같다.
   - (b) **P0 재생이 M0 기록의 프롬프트별 `draft_n`, `draft_n_accepted`, `draft_rounds`, `lookup_n`과 정확히 같다.**
   - 다르면 그 수와 위치를 보고하고 원인을 찾는다. [짐작] 가능한 원인은 MTP catch-up 행 수(배치 1 대 k+1)에 따른 near-tie argmax, 빌드 플래그 차이다.

### 빌드 (무거운 CPU 작업, 허락 필요)

- 운영과 같은 ROCm 10.1 / Clang 24 툴체인이 필요하다. 다른 툴체인이면 logits가 달라 (a)와 (b) 비교가 어긋난다(`docs/models/qwen3.8-flash-next/ROCM.md` "Numerical differences").
- 방법: `git archive HEAD` → `build/w9-src-snap/`, nixos/nix 컨테이너(volume `gufo-nix`, `--cores 8`)에서 `nix develop -c cmake --preset gpu-test -B build/w9-nix-gpu-test` 뒤 `--target qwen38_flash_next_gpu_probe`만 빌드한다.
- gpu-test 프리셋(RelWithDebInfo + assert)과 운영 release의 최적화 플래그 차이는 (b)로 확인한다. 어긋나면 release 플래그 + 테스트 타깃 구성을 감독과 정한다.
- 포맷 검사(`tools/ci/check-format.py`, nixos/nix)도 같은 단계에서 한다.
- 예상 [짐작]: 첫 nix develop 환경 준비 + 타깃 하나 빌드에 20–40분.

### 창 `PLAN-W9-M1.json` (1개)

- **이미지:** 위 빌드 결과를 F23 방식으로 `FROM ai-tools/gufo-eval:64849ca6ec7b` + closure로 묶거나, 빌드 트리를 `/src`로 읽기 전용 마운트해 unit 드라이버로 부른다(RQ2G/unit.py 선례). 방식은 빌드 결과를 보고 고른다.
- **모델:** M0와 같이 외장을 `:ro`로 쓴다. trace 결과는 디스크 속도와 무관하다.
- **입력:** A, K, D(완성 약 16.5K 위치 [짐작: M0 사이클 수 × 사이클당 토큰]).
- **예상 시간 [짐작]:** 위치마다 AR 약 36 ms + MTP 7회 약 2–3 ms ≈ 57 ms이므로 약 16분. 여기에 D prefill 약 3분과 USB 로드 약 5분을 더해 약 25분이다. seconds는 3600으로 잡는다.
- **산출물:** receipt에 trace 파일(약 16.5K × 7 × 16 B ≈ 2 MB + 토큰열)을 남긴다. 분석 결과는 이 문서에 적는다.

### 판단점

- P0 재생이 M0와 맞고, 최선 후보(P1/P2)의 예상 이득이 A와 K 전체에서 **+3 % 이상**이면 I1 구현으로 간다.
- 깊이 1 조건부 수락이 Kyojin 수치(ndt 2에서 87.6 %, 유리한 프롬프트)보다 확실히 낮으면, `--full-head`와 BF16 사이드카 trace로 I5(드래프트 품질 원인)를 먼저 본다.

## M1 결과 (2026-10-30)

### 빌드와 창 [확인]

- **코드:** `ba85850b`, `09b10ccd`.
  - `Session::ProbeDraftChain`(진단 메서드만 추가했고, 운영 경로는 그대로다)
  - `gpu_probe --mtp-trace`
  - `tools/qwen-flash/spec/spec_sim.py`
- **빌드:**
  - **첫 시도 실패:** `nix develop` gpu-test devShell은 torch, aotriton, rocmlir 같은 개발 도구를 소스부터 빌드하기 시작했다. 90분 timeout에 걸려 중단했다(`build/w9-nix-gpu-test.log`).
  - **대신 운영 nix derivation을 override했다(`build/w9-nix-probe.nix`).** 같은 nixpkgs, ROCm 10.1, HIP Clang 24, RelWithDebInfo를 쓰고, `BUILD_TESTING=ON`(-UNDEBUG)으로 `qwen38_flash_next_gpu_probe`만 빌드했다.
  - 소스는 `git archive c94572b3`(M0 바이너리와 같은 리비전) + `build/w9-trace.patch`(sha256 `a18b6df6…`)이고, `build/w9-src-snap`에 있다.
  - 이미지 `ai-tools/gufo-eval:w9-trace-rocm101`(id `1ffe6c47`). `FROM 64849ca6ec7b` + runtime closure이고, 운영과 같은 patched ROCr wrapper가 붙는다.
  - 빌드 중에 E10 창(품질 창)이 돌고 있었다. 감독 규칙상 허용되는 겹침이다.
- **창:** `PLAN-W9-M1.json`, `w9-m1-trace`. COMPLETE, 1,369 s, 재시도 없음.
  - 출력 `tmp/test-results/gufo-upstream-port-20261023/w9-m1/w9-m1-trace/receipt/run/trace.jsonl`(sha256 `923330bc6bb04cc179fbaa874c52c9aab87a10fc24bb75cf2337a91a60fc31a3`, 89개 프롬프트)

### 검증: P0 재생 대 M0 서버

| 세트 | 프롬프트 토큰 수 일치 | 완성 sha256 일치 | P0의 사이클/드래프트/수락/lookup 정확히 일치 |
|---|---|---|---|
| A | 64/64 | 64/64 | **64/64** |
| K | 17/17 | 17/17 | **17/17** |
| D | 8/8 | **1/8** | 1/8 |

- **D가 어긋난 원인 [확인].**
  - trace는 `Session::Sync` 한 번으로 prefill했다. 그래서 엔진 청크 4096 단위로 나뉘었다(`kPrefillChunkTokens`).
  - HTTP 서버는 단일 세션 프롬프트를 2048 단위 pass로 넣는다(`kAdmissionPrefillTokens`, `text_generation_scheduler.hpp:31`). M0 서버 로그의 component-cache 캡처도 2048/4096/6144/…에 있다.
  - pass 경계가 다르면 prefill 산술이 달라진다. 그래서 greedy 출력이 바뀌었다. d8k_2는 우연히 같았다.
  - A/K 프롬프트는 모두 2048토큰보다 짧아서 영향이 없다.
- **수정 `09b10ccd`:** `--prefill-step 2048`(기본값)으로 서버와 같은 prefix를 차례로 Sync한다.
  - 같은 방식으로 다시 빌드해 `/nix/store/wib32lyl…`, 이미지 `w9-trace2-rocm101`(id `864dbcbf`)를 만들었다.
  - D만 다시 도는 창 `PLAN-W9-M1B.json`(LOCKS 반영)이 승인을 기다린다.
- 정책 비교는 **정확히 재생되는 A+K 81개**로만 했다. D는 M1b 뒤에 더한다.

### 정책 비교 (A로 고르고, K로 따로 확인. C1 비용표 모델)

- **스크립트:** `tools/qwen-flash/spec/{spec_sim.py, spec_sweep.py, spec_sensitivity.py}`
- **결과 파일:** `~/projects/ai.tools/storage/evaluations/w9-m1-20261030/{validate.json, sweep_ak.json, sensitivity_ak.json}`
- **모델 검증:** P0의 모델 tok/s는 A 43.4, K 41.8이다. M0 실측은 A 45.3, K 44.7이다(USB 참고값). 수 % 안에서 맞는다.

| 정책 | A (고르는 데 씀) | K (따로 확인) |
|---|---|---|
| P0 현재 | 1.000 | 1.000 |
| P1 Kyojin TierRule 기본값(0.6/0.3/0.95/0.85, 3) | 1.109 | **1.090** |
| P1 격자에서 고른 값(thf 0.7, thv 0.4, thfd 0.8, thvd 0.4, dsplit 3) | 1.120 | **1.103** |
| P2 비용 인식 확신도(decay 0.995, prior 8) | 1.082 | 1.064 |
| P3 사후 최적(상한) | 1.341 | 1.326 |

종류별 모델 tok/s (P0 → P1 고른 값 → P3):

| chat | prose | creative | multi | code | copy | reasoning | structured |
|---|---|---|---|---|---|---|---|
| 37.7 → 42.5 → 51.2 | 36.1 → 38.6 → 47.5 | 32.7 → 36.6 → 45.6 | 36.9 → 41.2 → 51.1 | 49.8 → 55.4 → 65.4 | 61.5 → 64.5 → 70.7 | 51.3 → 57.5 → 67.7 | 48.9 → 56.9 → 63.2 |

- **민감도 (A+K):** 계산하고 버린 드래프트 비용을 1–6 ms, 확률을 만드는 추가 비용을 드래프트마다 0–0.3 ms로 바꿔도 P1 고른 값은 +8.9 %에서 +11.8 % 사이, Kyojin 기본값은 +7.8 %에서 +10.7 % 사이다.
- **[짐작] 해석:**
  - 지금 컨트롤러는 드래프트 전에 길이를 정한다. 그래서 확신이 높은 구간에서는 짧게 끊고, 낮은 토큰에서는 정해진 길이를 다 쓴다.
  - 토큰별 확률 누적곱으로 계속/정지를 정하면, 수락이 0.67에서 0.74로 오르면서 사이클당 드래프트도 늘어난다.
  - 사후 최적과의 남은 거리(+33 %)는 확신도 신호의 한계(top-64 softmax)와 드래프트 품질에 있다.
- **판단점(+3 % 이상)을 넘었다. I1 구현을 제안한다.**

### I1 구현 제안 (승인 대기)

- **정책:** C1 greedy와 sampled 요청의 MTP 사슬 길이를 토큰별 확률 누적곱(TierRule, 고른 값 또는 Kyojin 기본값. 감독이 고른다)으로 정한다. 상한은 7 그대로다.
  - lookup, verify, 수락, 샘플링은 바꾸지 않는다. 그래서 greedy 출력은 바이트 단위로 같다.
  - C>1 계획 배치(`MtpBatchController`)는 1단계에서 바꾸지 않는다.
  - sampled는 제안 분포가 바뀌므로 seeded 재현은 달라지지만, 분포는 정확하다. 적용 범위는 감독이 정한다.
- **확률 신호:** greedy 드래프트 스텝에서 지금은 `.token`만 요청한다. 같은 헤드 logits로 max softmax를 내는 작은 커널(argmax와 logsumexp)을 더하거나, 기존 top-64 후보 경로를 쓴다.
  - 스텝마다 이미 토큰을 호스트로 읽으므로 동기화는 늘지 않는다(`executor.cpp:4411`).
  - 추가 비용은 민감도 0.3 ms 안에 들어야 한다(op 테스트와 profile로 확인).
- **AR 백오프:** TierRule에는 백오프가 없다. 기존 `retry_tokens` 구조를 유지할지는 시뮬레이터로 따로 비교한다.
- **검증:**
  - 컨트롤러 단위 테스트
  - `spec_sim`을 새 정책으로 다시 돌려 서버 draft 수와 정확히 맞추기(구현과 모델의 동일성)
  - greedy 출력 동일성(M0 세트 89개 sha, `tests/functional`의 `long-context`와 `cache`)
  - 속도: main과 짝지은 대조. 요청·단계별 5 %·10 ms, ABBA. NVMe에 모델 복사가 필요하다(디스크 결정 필요).

## I1 구현 (2026-10-30, 감독 승인: Kyojin TierRule 기본값, C1 greedy만, AR 백오프 유지)

### 코드 (`f1cf33ac`)

- `mtp_policy.hpp`:
  - `DraftConfidence`: top-64 후보 위의 softmax. trace(`gpu_probe --mtp-trace`)와 같은 계산이다.
  - `MtpConfidenceRule`: TierRule 0.6/0.3/0.95/0.85, 얕은 깊이 3.
  - `MtpLengthController::Limit()`
- `engine.cpp` `PrepareDecode`:
  - 적용 조건은 greedy, 계획 배치 아님, `defer_head` 아님, occupancy 1, `DecodeConcurrency() == 1`이다(서버 `--sessions 1`의 단일 요청).
  - 길이 컨트롤러의 `Choose`가 0이면 AR 백오프를 그대로 한다. 아니면 상한 `min(cap - 1, Limit())`까지 드래프트하고, 매 드래프트의 확신도로 남길지와 계속할지를 정한다.
  - catch-up과 재귀 MtpForward가 토큰과 함께 top-64 후보를 낸다. 스텝마다 이미 호스트로 토큰을 읽고 있으므로 동기화는 늘지 않는다.
  - `Observe`는 그대로 둔다. 그래서 백오프 판단은 유지된다.
  - verify, 수락, lookup, sampled 경로, C>1은 바꾸지 않았다.
- `spec_sim.py`: `TierBackoffPolicy`(I1), `validate --policy I1`
- `mtp_sampling_test.cpp`: `CheckConfidenceRule`(경계값과 DraftConfidence)

### 시뮬레이터: AR 백오프 유무 (A+K 81개, P0 대비 모델 tok/s)

| 정책 | 전체 | chat | prose | creative | multi | code | copy | instruction |
|---|---|---|---|---|---|---|---|---|
| TierRule, 백오프 없음 | 1.104 | 1.113 | 1.056 | 1.135 | 1.109 | 1.100 | 1.039 | 1.116 |
| **I1 = TierRule + 기존 백오프** | **1.088** | 1.113 | 1.041 | 1.094 | 1.089 | 1.100 | 1.039 | 1.086 |

- 백오프를 유지하면 전체 1.6 %, creative 4 %를 잃는다(모델 기준).
- 백오프를 빼는 것은 감독 결정대로 별도 커밋으로 제안한다. 지금은 유지한다.

### 1단계: 단위 테스트 [확인]

- 빌드: `build/w9-nix-i1-tests.nix`. 운영 derivation override이고, 소스는 c94572b3 + `build/w9-i1.patch`(sha256 `90dfc545…`)다.
- `qwen38_flash_next_{mtp_sampling,prompt_lookup,draft_vocab}_test` 모두 PASS(`build/w9-nix-i1.log`).
- 포맷 검사 PASS.

### 다음 단계 준비

- 후보 운영 빌드: `nix build path:/src/build/w9-i1-src#default` → `/nix/store/4fn3q3gs…-gufo-0-unstable-dirty`. 이미지 `ai-tools/gufo-eval:w9-i1-rocm101`(R2 방식으로 묶음).
- **2·3단계 창 `PLAN-W9-I1V.json`(sha256 `806d85e7…`, LOCKS 반영):** M0 mtp와 같은 인자, 세트, 기록을 쓴다.
  - (2) `spec_sim.py validate --policy I1`(M1 A+K trace와 M1b D trace)로 서버 draft 수와 프롬프트별 정확히 일치하는지 본다.
  - (3) 89개 완성 sha256이 M0 AR과 같은지 본다.
  - D 비교에는 M1b가 먼저 필요하다.
- functional `long-context`/`cache`: `run.py`로 c94572b3 기준선을 기록한 뒤 후보를 돌린다(F23 `suite` 드라이버 방식). 창 계획은 (2)·(3) 뒤에 쓴다.

### M1b와 2·3단계 창 (2026-10-30)

#### 창 실행 [확인]

- M1b(`PLAN-W9-M1B.json`)와 I1V(`PLAN-W9-I1V.json`)를 감독의 "start" 뒤에 이어서 돌렸다. 둘 다 COMPLETE, 재시도 없음이다.
  - 첫 실행 시도는 내 사전 검사가 다른 W의 짧은 CPU 컨테이너를 보고 시작을 거부했다. 창은 돌지 않았다. 검사가 최대 15분 기다리도록 고쳐서 다시 실행했다.
- **M1b:** 529 s. 출력 `w9-m1b/w9-m1b-trace-d/receipt/run/trace-d.jsonl`(sha256 `30148eeb…`).
- **I1V:** 서버 준비 283 s, bench 460/164/302 s. 출력은 `w9-i1v/w9-i1v-fn-mtp/receipt/run/`이다.
- **분석:** `~/projects/ai.tools/storage/evaluations/w9-i1v-20261030/`(`validate_i1_a.json`, `validate_i1_ak.json`, `i1v_stats.json`, `cmp.py`, `trace_ak.jsonl`)

#### D 재생은 아직 서버와 맞지 않는다 [확인, 원인 미확인]

- M1b의 D 완성은 M1(4096 청크)과 **바이트 단위로 같다.** 그런데 서버와는 여전히 7/8이 다르다.
- 그래서 앞의 "2048 pass 경계" 설명은 **틀렸다.** prefill pass 경계는 출력을 바꾸지 않는다.
- **차이의 모양:** 생성 중간의 near-tie 지점에서 갈린다(13–782자째). 서버 AR도 서버 MTP와 같고 trace와는 다르다. 그래서 decode 경로 문제가 아니라, 긴 문맥에서 서버 쪽 상태(prefill 또는 캐시 경로)가 `Session::Sync`와 조금 다르다.
- Flash-Next KV는 FP16 하나뿐이라(`kv_cache_type.cpp:22`) KV 양자화도 원인이 아니다.
- **[짐작] 남은 후보:** 서버 component-cache의 in-pass checkpoint(프롬프트 끝 7토큰 앞)나 pass 끝 캡처(2048 배수)가 sparse attention/indexer 상태에 영향을 주고, 그 영향이 sparse 임계값을 넘는 긴 문맥에서만 드러나는 경우다. 확인하지 않았다.
- **영향:** 정책 재생 검증(구현 = 모델)은 A+K 81개로 했다. D는 서버 실측(draft 수, 동일성)만 쓴다.
- **해결 방안(감독 결정):** trace가 서버와 같은 `SyncThroughComponents`/`SyncThrough` 캡처 경계를 쓰게 고쳐 다시 빌드하고 trace를 다시 돈다(약 10분 창). 또는 D 재생 일치를 범위에서 뺀다.

#### 2단계: I1 재생과 서버의 정확한 일치 [확인]

- `spec_sim.py validate --policy I1`(M1 A+K trace) 대 I1V 서버 기록: **81/81 정확히 일치**(cycles, drafted, accepted, lookup, lookup_accepted). 프롬프트 토큰 수도 81/81 같다.
- 그래서 구현이 TierRule + 기존 백오프 모델과 같게 동작한다.

#### 3a단계: greedy 출력 동일성 [확인]

- I1V의 89개 프롬프트, 모든 반복의 완성 sha256이 M0 AR과 같다: **89/89**.
- 반복 사이 draft 수도 같다(결정성).

#### I1 대 M0 (같은 외장 디스크 조건. tok/s는 참고용이고, 판정은 NVMe ABBA로 한다)

| 세트 | I1/M0 decode tok/s (paired 95 %) | 수락률 차이 | I1 tok/s | I1 AR 대비 | 사이클당 드래프트 / 토큰 |
|---|---|---|---|---|---|
| A 64 | **1.126 [1.090, 1.175]** | +0.072 [0.055, 0.090] | 51.0 | 1.89× | 2.83 / 3.10 |
| K 17 | **1.117 [1.093, 1.146]** | +0.006 [−0.018, 0.033] | 49.9 | 1.79× | 2.67 / 2.86 |
| D 8 | **1.138 [1.070, 1.231]** | −0.004 [−0.053, 0.045] | 44.8 | 1.68× | 2.43 / 2.54 |

- **시뮬레이터 예측(A+K +8.8 %)보다 실측이 조금 크다.** [짐작] 비용표가 2026-09-20 값이라 실제 행당 비용이 더 작을 수 있다.

종류별 I1 tok/s (참고). Kyojin 카드 1.5.10 값은 괄호에 적었다.

| 종류 | M0 → I1 | Kyojin |
|---|---|---|
| chat | 40.5 → 47.0 | (52.8) |
| prose | 38.6 → 42.3 | (47.5) |
| code | 56.4 → 66.2 | (73.2) |
| multi | 38.3 → 41.5 | |
| copy | 69.6 → 75.7 | |
| 8K | 39.0 → 46.5 | (52.2) |
| 32K | 39.8 → 43.3 | (51.1) |

- Kyojin과의 격차는 M0의 1.23–1.34×에서 **1.11–1.18×**로 줄었다(측정 방법은 서로 다르다).
- AR 대비 배율: chat 1.65×, prose 1.55×, code 2.39×. Kyojin SERVE.md의 1.42–1.91×와 비슷하거나 높다.

#### 다음

- I1F(기능 suite, `PLAN-W9-I1F.json`)는 W8 FAKEQ 뒤에 돈다.
- 그 뒤 RQ2F → W8 speed → NVMe 복사 → I1 ABBA(MAIN/I1/GRID, `PLAN-W9-ABBA-*.json`, 판정 `tools/qwen-flash/spec/abba.py`).

## 열린 관찰: 긴 문맥에서 HTTP 서버와 `Session::Sync`의 greedy 출력이 다르다 (2026-10-30)

감독 결정(2026-10-30):
- D의 정확한 재생 일치는 범위에서 뺀다. I1의 D 판정은 서버 측정(draft 수, 동일성, ABBA)만으로 한다.
- 이 차이는 I1과 무관하다. 엔진의 기존 실행 경로 사이의 차이로 보고, 정확성 문제로 다루지 않는다. 다른 W에 넘길 수 있도록 기록만 남긴다.

### 무엇이 다른가 [확인]

- **같은 것:** 같은 리비전(c94572b3 기반), 같은 nix ROCm 10.1 / Clang 24, 같은 가중치(UD-Q4_K_XL + shared-Q8_0 MTP, sha256 확인), 같은 Model 옵션(context 49152, draft 7, decode concurrency 1), 같은 chat template(think off), 같은 프롬프트 토큰 수.
- **(1) HTTP 서버:** `gufo serve llm --sessions 1`, `--speculative off` 또는 `mtp`, 요청 `cache_prompt: false`, greedy.
- **(2) 엔진 직접 호출:** `Session::Sync`(한 번에, 또는 2048 prefix씩. 둘의 결과는 바이트 단위로 같다) 뒤에 `Session::Evaluate`로 greedy decode. 경로는 `qwen38_flash_next_gpu_probe --mtp-trace`이고, 이미지는 `w9-trace-rocm101`과 `w9-trace2-rocm101`이다.
- **결과:**
  - 2048토큰보다 짧은 프롬프트 81개는 (1)과 (2)가 바이트 단위로 같다.
  - 약 7.3K–39K토큰 프롬프트 8개 중 7개는 생성 도중 갈린다.
  - (1)의 AR과 MTP는 서로 같다(8/8). 그래서 decode 경로 차이가 아니라, prefill이나 상태 경로의 차이다.
- KV 타입 차이는 아니다. Flash-Next는 FP16 KV 하나뿐이다(`src/cli/kv_cache_type.cpp:22`).

### 재현 방법

- **프롬프트:** `tools/qwen-flash/spec/suite_d.json`(sha256 `82929a75…`)의 8개. 저장소 docs를 `git show c94572b3:` 기준으로 잘랐다.
- **서버 출력:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/w9-m0/w9-m0-fn-ar/receipt/run/responses-2.jsonl`(SSE 원문)
- **엔진 직접 출력:** `…/w9-m1b/w9-m1b-trace-d/receipt/run/trace-d.jsonl`의 case 줄(`completion` ids, `text`)
- **처음 갈리는 위치** (완성 텍스트 문자 위치, trace / 서버):

| 프롬프트 | 프롬프트 토큰 | 처음 갈리는 문자 | trace | 서버 |
|---|---:|---:|---|---|
| d8k_0 | 7,369 | 274 | "climbed the spiral staircase…" | "climbed the final spiral…" |
| d8k_1 | 7,274 | 782 | "that require more than just…" | "that contain not just…" |
| d8k_2 | 7,355 | 같음 | — | — |
| d8k_3 | 7,509 | 13 | "Based on the documentation provided" | "Based on the provided documentation" |
| d32k_0 | 27,158 | 171 | "the keeper, moved through…" | "the keeper, had long since…" |
| d32k_1 | 31,180 | 614 | "Retain immutable **shared chunks**" | "Retain immutable, shared prefix chunks" |
| d32k_2 | 39,256 | 124 | "tearing the old stone tower" | "tearing the lighthouse" |
| d32k_3 | 29,817 | 572 | "**Dependencies:**" | "**Model Weights:**" |

### 짐작 (확인 안 됨)

- 서버는 `cache_prompt: false`여도 component cache 캡처를 한다. M0 서버 로그(`w9-m0-fn-mtp/receipt/run/server-1.log`)에 2048 배수 위치와 프롬프트 끝 7토큰 앞(in-pass checkpoint, `SyncThrough`/`SyncThroughComponents`)의 retention 기록이 있다.
- 이 캡처 경로가 sparse attention/indexer(압축 블록) 상태를 `Session::Sync`와 조금 다르게 남기고, 그 차이가 sparse 임계값을 넘는 긴 문맥에서만 near-tie를 뒤집는 것으로 보인다.
- **확인 방법:** trace가 같은 경계에서 `SyncThrough(prompt, len − 7, &checkpoint)`와 component 캡처를 쓰게 하고 결과를 비교한다. 또는 서버에서 component cache를 끈 상태(가능하다면)로 비교한다.

### 3b단계: 기능 suite (I1F, 2026-10-30)

- **창:** `PLAN-W9-I1F.json`, `w9-i1f-suite`. 한 번 돌렸고 재시도는 없다. 결과는 FAILED_no_fallback이다. 위치는 `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/w9-i1f/w9-i1f-suite/receipt/run/`.

| 그룹 | 구성 | 결과 |
|---|---|---|
| standard-cache (`cache`) | `--sessions 2`, I1 비활성(회귀 확인) | **통과**(816 s). text/thinking/tool/image cancel, disk spacing, disk restore 사례가 모두 통과했다 |
| single-session (`long-context` + `cache`) | `--sessions 1`, I1 활성 | **시험을 돌리지 못했다**: `server startup timed out` |

- **원인 [확인: server.log].**
  - 기능 runner에서는 디스크 캐시가 켜진다. 그래서 서버가 로드할 때 모델 artifact fingerprint를 만들면서 104 GB를 외장 USB에서 읽는다(`artifact_identity` 4분 43초).
  - `suite.py`가 고정으로 넘기는 `--startup-timeout 600`을 넘겼다.
  - 두 번째 그룹은 498 s로 간신히 통과했다. [짐작] 첫 시도 덕분에 page cache가 따뜻했다.
  - I1과는 무관하다.
- **감독 결정 (b):** NVMe 복사 뒤 ABBA 전에 single-session 그룹만 다시 돈다.
  - 계획: `PLAN-W9-I1F2.json`(NVMe 모델 마운트, mmproj는 외장 읽기 전용, `runner_args: ["--startup-timeout", "1800"]`). `suite.py`의 고정 600 뒤에 붙으므로 마지막 값이 쓰인다.
  - LOCKS 반영은 W8 SPEED가 끝난 뒤에 한다.

### NVMe 복사와 I1F2 (2026-10-30)

- **NVMe 복사:** `build/w9-copy-flash-next.sh`(로그 `build/w9-copy-flash-next.log`). UD-Q4_K_XL 4분할 + MTP shared-Q8_0, 106.3 GiB를 `~/projects/ai.tools/storage/models/llm/qwen-3.8-flash-next-ud-q4-k-xl-gguf/`로 옮겼다.
  - 파일마다 복사 뒤 sha256이 외장 `EXPECTED-SHA256.txt`(HF LFS oid)와 같다(`SHA256SUMS`, `PROVENANCE.md`).
  - [짐작] 쓰기 직후에 해시했으므로 일부는 page cache에서 읽었을 수 있다.
  - 복사 뒤 루트 여유는 158 GiB다.
- **I1F2** (`PLAN-W9-I1F2.json`, sha256 `051085fc…`, LOCKS 반영): COMPLETE, 재시도 없음. NVMe에서 서버 로드는 71 s였다.
  - **single-session(`--sessions 1`, I1 활성): 통과.** 13개 사례가 모두 통과했다: `long-context`, text/thinking-tool/legacy-tool/image/image-thinking cancel(RAM과 disk), disk-spacing(RAM과 disk).
  - 결과 위치: `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/w9-i1f2/w9-i1f2-suite/receipt/run/`
- **3단계 결론:** greedy 동일성 89/89(I1V)와 기능 suite(I1 활성 그룹 I1F2, I1 비활성 그룹 I1F standard-cache)가 모두 통과했다.
- **다음:** I1 ABBA(W8 SPEED2 뒤, 감독 "start").

## I1 ABBA 속도 판정 (2026-10-30)

### 창 [확인]

- **순서:** MAIN a1 → I1 b1 → GRID c1 → GRID c2 → I1 b2 → MAIN a2. 모두 COMPLETE, health_PASS, 재시도 없음. 11:22–12:59에 돌렸다.
- **창마다 사전 검사:** `docker ps` 비어 있음, LOCKS, 디스크, 1분 load < 4(시작 때 0.6–3.8).
- **조건:** NVMe 모델(sha256 확인), 서버 준비 22–23 s. 세 arm 모두 같은 nix ROCm 10.1 / Clang 24이고, A/K/D 세트, C=1 greedy, think off, 프롬프트 캐시 꺼짐, warmup 1(D 0) + 반복 2를 썼다.
- **결과 위치:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/w9-abba/<tag>/`
- **판정:** `tools/qwen-flash/spec/abba.py`(프롬프트·단계별, t = max(10 ms, 5 %), 요청끼리 평균하지 않음). 결과는 `~/projects/ai.tools/storage/evaluations/w9-abba-20261030/abba-{i1,grid,grid_vs_i1}-{0,1,2}.json`에 있다.

### 결과

| 비교 | 세트 | decode tok/s (참고, 반복 합) | pass / faster / slower / inconclusive (프롬프트 × 단계) | 완성 동일 |
|---|---|---|---|---|
| **I1 대 MAIN** | A 64 | 49.36 → 53.53 (**1.085**) | 85 / 38 / **2** / 3 | 64/64 |
| | K 17 | 47.82 → 51.92 (**1.086**) | 22 / 10 / 0 / 2 | 17/17 |
| | D 8 | 42.01 → 45.97 (**1.094**) | 10 / 6 / 0 / 0 | 8/8 |
| GRID 대 MAIN | A | 49.36 → 53.80 (1.090) | 81 / 40 / 2 / 5 | 64/64 |
| | K | 47.82 → 52.46 (1.097) | 20 / 13 / 0 / 1 | 17/17 |
| | D | 42.01 → 46.56 (1.108) | 9 / 7 / 0 / 0 | 8/8 |
| GRID 대 I1 | A / K / D | 1.005 / 1.010 / 1.013 | A 123 / 1 / 2 / 2, K 34 / 0 / 0 / 0, D 15 / 1 / 0 / 0 | 전부 같음 |

- **prefill:** 모든 비교에서 pass다. inconclusive는 I1 대 MAIN의 `d4_ko_instr_01` prefill 하나이고, MAIN 쪽에 310 ms 이상치가 있다.
- **각 arm 안의 일관성:** 반복과 창 사이의 완성 sha와 draft 수가 모두 같다.

### 판정에 걸린 것 (I1 대 MAIN, A)

| 프롬프트 | 단계 | MAIN (ms, 4표본) | I1 (ms, 4표본) | 상태 | 원인 |
|---|---|---|---|---|---|
| `d4_ko_instr_04` | decode | 139.7–140.6 | 152.3–153.3 | **slower** (+12.9, 기준 10) | 9토큰 답. I1이 첫 사이클부터 넓게 드래프트한다(draft 9 대 8, 수락 7 대 7). 운영 컨트롤러는 "수락을 보기 전 첫 사이클은 보수적 폭"으로 이 짧은 답 문제를 이미 막고 있었다(QUALITY.md "Sampled draft-head costs") |
| `en_code_rust_err` | decode | 1912.0–1914.8 | 2067.2–2073.2 | **slower** (+155.8, 기준 95.7, +8.1 %) | 확신도 정지가 이 프롬프트에서 너무 일찍 끊는다(draft 167 → 155, 수락 146 → 141, 사이클이 늘어남) |
| `d4_ko_code_02` | decode | 5035–5503 | 4623–5237 | inconclusive | 양쪽에 이상치가 하나씩 있다. 중앙값은 I1이 412 ms 빠르다 |
| `ko_explain_fermentation` | decode | 6399–6449 | 6065–6100 | inconclusive | I1이 337 ms 빠르지만 범위 간격이 기준 320 ms에 조금 못 미친다 |
| `d4_ko_instr_01` | prefill | 198–311 | 200–265 | inconclusive | 양쪽 prefill이 흔들린다(I1과 무관한 단계) |

- K와 D에는 slower가 없다. inconclusive 2개(`k_copy_2`, `k_prose_0` decode)는 그대로 남긴다.
- **시뮬레이터 확인:** 정확히 재생되는 A+K trace에서 두 slower 프롬프트는 I1이 P0보다 각각 1.057배 느리다. 실측(+9 %, +8 %)과 방향과 크기가 맞는다. 그래서 잡음이 아니라 정책 탓이다.

### 수정안 (시뮬레이터, A+K 81개, 승인 대기)

- 스크립트: `~/projects/ai.tools/storage/evaluations/w9-abba-20261030/{variants.py, sweep_v1.py}`

| 변형 | 전체 대 P0 | `d4_ko_instr_04` | `en_code_rust_err` | 5 %·10 ms보다 느린 프롬프트 |
|---|---|---|---|---|
| I1 (지금) | 1.088 | 1.057 | 1.057 | 1(+모델 경계 1) |
| V1: 첫 사이클(`explored_depth == 0`)은 운영 컨트롤러 폭을 쓰고, 그다음부터 확신도 규칙 | 1.087 | 1.000 | 1.070 | 1 |
| V2: 운영 컨트롤러 길이를 아래 한계로 | 1.033 | 1.057 | 1.009 | 2 |
| **V1 + 깊은 임계값 0.9/0.4** (얕은 0.6/0.3/3은 Kyojin 그대로) | **1.093** | 1.000 | ≤ 1.015 | **0** |
| V1 + 격자 최선 0.7/0.3/0.9/0.4/3 | 1.095 | 1.000 | ≤ 1.03 | 0 |

- [짐작] V1은 운영 컨트롤러가 이미 쓰는 첫 사이클 보수성을 되살리는 것이라 근거가 명확하다.
- 깊은 임계값을 바꾸는 것은 81개 프롬프트로 고른 값이라 과적합 위험이 있다. 바꾸면 다시 ABBA로 확인해야 한다.
- **제안:** V1 + 깊은 임계값 0.9/0.4(자유 파라미터 2개)로 고치고, 다시 빌드하고, I1 대 MAIN ABBA를 다시 한다(MAIN 표본은 재사용하지 않고 새로 번갈아 잰다). 또는 V1만 넣고 깊은 임계값은 Kyojin 그대로 둔다.

## I1 수정판 (2026-10-30, 감독 승인: V1 + 깊은 임계값 0.9/0.4, held-out 세트 추가)

### 코드 (`92fcaf37`)

- **`engine.cpp`:** 확신도 규칙을 요청이 수락을 한 번이라도 관찰한 뒤(`explored_depth != 0`)에만 쓴다. 첫 사이클은 운영 컨트롤러의 폭과 사슬을 그대로 쓴다.
- **`mtp_policy.hpp`:** 깊은 깊이의 계속 임계값 0.9, 남김 임계값 0.4. 얕은 깊이는 0.6/0.3/3 그대로(Kyojin 값)다.
- **단위 테스트:** 새 경계를 넣었다(0.99 사슬 7개, 깊은 0.85는 남기고 멈춤, 0.4 남김, 0.3 버림).
- **`spec_sim.py`:** `ConfidencePolicy` = I1(수정판). 처음 판은 `I1a`로 남겼다. I1a는 기존 I1V 기록과 여전히 81/81 맞는다.
- **시뮬레이터 (A+K 81개, P0 대비):** I1 1.093, I1a 1.088. 5 %·10 ms보다 느린 프롬프트는 0개다.
- **포맷:** 내 파일은 clang-format PASS다. 전체 검사는 다른 W의 추적되지 않는 EXL3 작업 파일(`tests/core/quant/exl3_upstream_test.cpp`) 때문에 실패한다. 내 범위 밖이라 손대지 않았다.

### held-out 세트 (`tools/qwen-flash/spec/suite_h.json`, sha256 `a15ebc28…`)

- 32개 프롬프트, 8개 종류(explain, story, code, howto, summary, reasoning, structured, rewrite) × (영문 2 + 한글 2)이다.
- 임계값을 정한 **뒤에** W9가 새로 썼다. 튜닝에는 쓰지 않았다.
- **겹침 검사:** `check_heldout.py`(W2 `accept.check_overlap`, 40자 조각을 20자 간격으로). accept_prompts(D4 포함), suite_k, suite_d, d4_ko_prompts, 세트 안 중복을 봤고 모두 **0건**이다. 결과는 manifest에 기록했다.
- R0/R3 품질 코퍼스는 token window로만 저장돼 있어 이번 검사에서 뺐다.

### 1단계와 빌드 [확인]

- c94572b3 + `build/w9-i1b.patch`(sha256 `5a98e238…`, = ba85850b 09b10ccd f1cf33ac 92fcaf37 src/tests)
- `qwen38_flash_next_{mtp_sampling,prompt_lookup,draft_vocab}_test` 모두 PASS(`build/w9-nix-i1b.log`). 빌드는 메모리 24 GB, 4코어로 제한했다(XQ가 돌고 있었다).
- 운영 빌드 `/nix/store/dq89jrza…`, 이미지 `ai-tools/gufo-eval:w9-i1b-rocm101`(label `c94572b3…+w9-5a98e238`)

### 준비한 창 (LOCKS 반영)

1. **`PLAN-W9-HTRACE.json`**(sha256 `fe522388…`): held-out trace. trace 이미지는 `w9-trace2`를 그대로 쓰고, 모델은 NVMe다. 약 10분 [짐작]. 결과로 시뮬레이터의 held-out 검사(I1 대 P0)를 ABBA 전에 한다.
2. **`PLAN-W9-I1V2.json`**(sha256 `5bd7542e…`): 수정판 I1 서버, A/K/D/H, 기록 포함. 약 25분 [짐작].
   - `validate --policy I1`: A+K는 M1 trace, H는 HTRACE와 프롬프트별로 정확히 맞아야 한다.
   - 완성 동일성: A/K/D는 M0 AR, H는 trace와 비교한다.
3. **`PLAN-W9-ABBA2-{MAIN,I1}.json`**(sha256 `95327218…`, `732e0619…`): MAIN a1 → I1 b1 → I1 b2 → MAIN a2. MAIN은 새로 잰다. A/K/D/H를 따로 판정한다. 창마다 약 20분.
