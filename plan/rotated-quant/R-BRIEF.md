# RQ 작업 안내: R0b·R1a (에이전트용, 2026-10-28)

상위 문서: [회전 가중치 양자화 계획](PLAN.md). 사용자 결정(2026-10-28): 디스크를 확보하기 전에 디스크와 무관한 R0b(타입별 실효 대역폭)와 R1 컨버터의 합성 텐서 부분(R1a)을 먼저 진행한다. 감독(메인 세션)이 각 단계를 에이전트에 맡기고 결과를 검증한다.

## 공통 규칙

- 저장소 규칙: `CLAUDE.md`(= `AGENTS.md`). 옵션 이름 하나·별칭 없음, 모델 코드·도구는 모델 옆, 자동 fallback 없음, 한 줄 Conventional Commit.
- **호스트에는 Nix, numpy, llama.cpp가 없다.** 빌드와 Python 실행은 eval 이미지 컨테이너에서 한다. 이미지 `ai-tools/gufo-eval:7c2c5a728270`에는 `hipcc`(ROCm 7.2.4), `cmake`, `/opt/venv/bin/python3`(numpy 2.3.5)가 있다. `llama-quantize`는 없다.
  ```sh
  D="docker run --rm -v /home/m2on/projects/gufo:/src -w /src --user $(id -u):$(id -g)"; I=ai-tools/gufo-eval:7c2c5a728270
  $D --entrypoint python3 $I -I <script> ...
  $D --entrypoint hipcc $I ...
  ```
- **다른 세션이 같은 장비에서 GPU 창을 돌린다**(운영 전환 점검 P2). 그래서:
  - CPU 무거운 일(hipcc 빌드, 큰 numpy 계산)을 하기 전에 `docker ps`로 `gufo-port-*` 컨테이너가 도는지 본다. 돌고 있으면 끝날 때까지 기다린다(공유 LPDDR라 CPU 부하가 상대 GPU 측정을 오염시킨다).
  - 다른 세션(`gufo-00`)과의 약속(2026-10-28): 이쪽 GPU 창을 시작하기 전에 감독이 그 세션에 알린다. 그 세션도 자기 창을 시작하기 전에 알리고, 창 시작 전에 `ABBA-OWNER.lock`을 `flock -n`으로 잡는다. 에이전트는 GPU 창을 시작하기 전에 감독에게 보고하고 허락을 받는다.
  - 포맷 검사(nixos/nix 컨테이너), 패키지 내려받기도 CPU 작업이다. 상대 창이 도는 동안에는 하지 않는다.
  - GPU는 ai.tools 창 실행기로만 쓴다. 한 번에 창 하나. 절차는 [KVarN K1 브리프](../gufo-upstream-port/K1-BRIEF.md)의 "공통 규칙" GPU 항목을 그대로 따른다(창 계획 JSON은 `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/`에, `driver: test.experiments.gufo_eval_20261020.unit`, 실행 뒤 `LOCKS.py`). 창 하나에 한 번만 돌리고 실패해도 자동 재시도하지 않는다.
- **파일 경계:** 자기 단계의 파일만 만들고 `git add`한다. 다른 에이전트와 다른 세션이 같은 트리에서 일한다. push·PR·이슈 금지.
- 커밋 트레일러(감독이 준 그대로):
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` /
  `Claude-Session: https://claude.ai/code/session_016Kh8sxz8VLDwUyGvpXsPaK`
- C++/HIP 포맷: 커밋 전 [K1 브리프](../gufo-upstream-port/K1-BRIEF.md)의 포맷 명령으로 `tools/ci/check-format.py`를 통과시킨다.
- 끝에 보고: 바꾼 파일·커밋, 돌린 검사와 결과(실패 포함 원문 요약), 측정 표, 남은 문제. 짐작은 짐작이라고 쓴다. 막히거나 결정이 필요하면 멈추고 보고한다.

## R0b: 타입별 실효 대역폭 (GPU)

계획의 "R0b" 절이 범위다. 요약과 구체 사항:

- **파일:** `tools/bench/rq_type_bench.hip`(필요하면 같은 디렉터리 헤더 하나). 빌드 산출물은 `build/rq-bench/`. 운영 코드는 고치지 않는다.
- **가중치:** 무작위 값으로 만든 실제 블록 구조(타입별 블록 크기 그대로). 모델 파일은 쓰지 않는다.
- **크기:** 27B 행렬(K→N): 5120→10240(`attn_qkv`), 5120→6144(`attn_gate`), 5120→12288(`attn_q`), 5120→1024(`attn_k`/`attn_v`), 6144→5120(`ssm_out`, `attn_output`), 5120→17408(`ffn_gate`/`ffn_up`), 17408→5120(`ffn_down`), 5120→248320(`output`).
- **배치:** AR 1, DFlash2 verify(실제 verify 배치 크기를 `src/models/qwen/dflash_policy.hpp` 등에서 찾아 적는다), prefill 512와 2048.
- **커널:**
  - 27B 지원 타입(Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S)은 **운영 경로 커널**을 그대로 부른다(`src/models/qwen/hip/kernels/gemv_quant.hip`, `small_batch_*`, `prefill_quant_gemm.hip` 등. 어느 배치에서 어느 커널이 선택되는지 운영 코드의 선택 로직을 따라 적는다). 운영 커널을 링크하기 어려우면 소스를 포함하거나 복사하되, 복사했다면 그 사실과 원본 경로·리비전을 결과에 적는다.
  - 2비트대(Q2_K, IQ2_XXS, IQ2_XS, IQ2_S)는 upstream ggml HIP의 vec-dot(mmvq) 구현을 가져와 벤치 커널로 만든다. 라이선스(MIT)와 출처 커밋을 파일 머리에 적는다. AR과 verify만 재고, prefill은 생략해도 된다(생략하면 적는다).
  - 독립 FWHT 커널: 128점, 부호 곱, 정규화. K = 5120, 6144, 17408, 배치는 위와 같다. CPU 참고와 비교한 정확성 검사를 같이 넣는다.
- **측정:** 타입·크기·배치마다 시간(중앙값, p95)과 실효 GB/s(가중치 바이트 기준). `tools/bench/gfx1151_peak.hip` 상한을 같은 창에서 잰다. 각 측정은 워밍업 뒤 반복하고, 타입 순서를 번갈아 돌린다.
- **창:** 창 계획 `PLAN-RQ0B.json`. 실행 시간 예상을 먼저 세우고, 다른 세션의 창이 없을 때 돌린다.
- **산출:** `plan/rotated-quant/R0B-RESULTS.md`에 "타입별 bpw 대 실효 대역폭 대 토큰당 시간" 표(27B 전체 dense 한 토큰을 그 타입으로 읽을 때의 추정 시간 포함), FWHT 비용 표, 상한 대비 비율, 연산에 묶이는 것으로 보이는 타입과 근거.

## R1a: 컨버터 합성 텐서 부분 (CPU, GPU 없음)

계획의 "R1 컨버터" 절과 "GGUF 표시 규칙 (v1)", "공통과 모델별 경계" 절이 범위다. 이번에는 실제 모델 파일과 `llama-quantize` 없이 할 수 있는 것만 한다.

- **파일:** `tools/rotquant/`(공통 본체), `tools/qwen27b/rotquant_groups.py`(27B 입력 그룹 정의), 테스트는 `tools/rotquant/` 아래 또는 저장소의 Python 테스트 관례를 확인해 따른다. 기존 `tools/gufo/gguf.py`를 재사용하고, 필요한 GGUF 쓰기 기능은 `tools/rotquant/` 안에 둔다(`tools/gufo/gguf.py`를 바꿔야 하면 최소로 하고 보고한다).
- **할 것:**
  1. 회전 수학: 128점 정규화 Sylvester Hadamard, 부호 벡터 `D`, `W' = W·D·H`(GGUF 차원 규칙 `ne0 = K` 주의), 입력 회전 `x' = H·D·x`.
  2. 27B 입력 그룹 정의(계획의 표 그대로), 그룹 안 부호 공유 강제, 정의 없는 모델 거부.
  3. `rotate` 단계: BF16 GGUF(분할 포함) 입력 → 회전 BF16 GGUF + `.rot` 짝 텐서 + 전역 KV + `gufo.rotation.source_*` KV. 텐서 하나씩 스트리밍.
  4. `.rot` 저장 타입: 계획의 두 선택지(I8 또는 F16) 중 R1a에서는 **F16**으로 한다(gufo reader에 I8이 없으므로). 이유를 결과에 적는다.
  5. 레시피 형식(`--recipe <yaml>`), `--rotate-groups`, `--seed`, `--rotate-only`, `--copy-recipe-from`(헤더만 읽음), `--dry-run`(텐서별 타입·회전 여부·예상 크기·bpw 표). 27B 로더 지원 타입(`src/models/qwen/weights.cpp:59`) 밖의 타입은 거부.
  6. `quantize` 단계: `llama-quantize` 명령줄(`--tensor-type` 규칙, `--imatrix`)을 만드는 부분까지 구현하고, 실제 실행은 하지 않는다. 생성된 명령줄을 테스트로 확인한다.
- **테스트(합성 텐서):** Hadamard 직교성, 그룹 부호 일치, `W'·(H·D·x) ≈ W·x`(FP32에서 1e-5 수준, BF16 저장 뒤에는 BF16 반올림 수준), 분할 GGUF 입력, 레시피 파싱·거부, 짝 텐서와 KV 쓰기→읽기 왕복, dry-run 표, `llama-quantize` 명령줄 생성.
- **하지 않을 것:** 실제 27B 파일 변환, `llama-quantize` 실행, llama.cpp 빌드, 로더(C++) 변경. 이것들은 디스크 확보 뒤 R1b·R2에서 한다.
- **산출:** 코드·테스트 커밋, `plan/rotated-quant/R1A-RESULTS.md`(사용법, 레시피 예시, 테스트 결과, R1b에서 확인할 것 목록: `llama-quantize`가 `.rot`을 그대로 복사하는지, 이름 기반 휴리스틱, llama.cpp 로드 거부).
