# X2 브리프: EXL3 직접 로드와 decode·verify 커널 (2026-10-30)

공통 규칙은 [W-COMMON.md](../rotated-quant/W-COMMON.md)를 따른다. 계획은 [PLAN.md](PLAN.md) X2, 근거는 [XA-RESULTS.md](XA-RESULTS.md)(X0·X1 통과), [EXL3-RESULTS.md](../rotated-quant/EXL3-RESULTS.md)(W8: 속도와 커널 목록)다.

담당은 둘로 나눈다. 둘은 인터페이스(디바이스 텐서 구조)를 먼저 합의한다.

- **X2a (X-A, W8 이어서):** 직접 로드
- **X2b (X-B, W3):** 빠른 커널

## X2a: EXL3 팩 직접 로드 (X-A)

- **로더:** gufo 27B가 EXL3 safetensors 팩을 바로 읽는다.
  - `trellis`, `suh`, `svh`, `mul1`은 압축된 채로 GPU에 올린다.
  - 나머지(임베딩, norm, conv1d, A_log, dt_bias, alpha/beta 등)는 X1 매핑(`exl3_weights`)대로 변환해 올린다. V head 재배열과 norm +1이 여기에 들어간다.
  - 파일 형식은 로더가 판단한다. CLI 옵션은 더하지 않는다.
- **메타데이터:** 토크나이저와 하이퍼파라미터를 정한다.
  - 후보 (a): HF `tokenizer.json`/`config.json`에서 gufo 내부 구조를 만든다.
  - 후보 (b): 같은 모델의 GGUF 메타데이터(텐서 없는 사이드카)를 쓴다.
  - 고른 방식과 이유를 보고한다. 지금 GGUF 경로와 토크나이저 출력(토큰 id)이 같아야 한다.
- **정확성용 기준 커널:** 단순 GPU dequant 커널(trellis → FP16 타일, Hadamard)과 기존 GEMM을 조합한다. 속도는 보지 않고, X2b 커널의 비교 기준으로 쓴다.
- **통과 기준:**
  - R0 하네스로 KL(ExLlamaV3 같은 팩 ‖ gufo 직접 로드)이 X1과 같은 엔진 차이 수준이다(중앙값 ≤ 1e-4, 평균 ≤ 1e-3).
  - X1의 ΔNLL 차이(약 1–2e-3, BF16 재반올림이 원인이라는 짐작)가 줄어드는지 기록한다.
  - GGUF 경로의 fingerprint가 그대로다.
  - 메모리 상주량을 기록한다(5.00 팩, 목표는 UD-Q4_K_XL 이하).

## X2b: decode GEMV와 verify 커널 (X-B)

- **커널:**
  - trellis 디코드를 융합한 GEMV(m = 1)
  - verify/소배치(m = 2–8, DFlash2 verify, MTP)
  - 출력 헤드(K6)
- **Hadamard 융합:**
  - 입력 Hadamard(`suh`, 128점)는 활성화 생성 커널에 융합한다(RMSNorm, SwiGLU, SSM gate 등). R2의 회전 융합 지점 표를 다시 쓴다. EXL3는 텐서마다 suh가 다르므로, 같은 입력을 쓰는 텐서끼리 활성화를 공유할 수 없다. 이 문제를 어떻게 풀지 설계에 적는다.
  - 출력 Hadamard(`svh`)는 GEMV epilogue에 융합한다.
- **참고 코드:** upstream `exllamav3_ext/rocm/*`(W8 커널 목록에서 decode 시간의 81 %가 `exl3_gemv_mr_dot_*`), vcruz305(`v_sad_u8` 디코드, LDS pad), Kyojin. MIT 고지를 유지하고, 규칙은 `.agents/skills/optimize-kernel/SKILL.md`를 따른다.
- **통과 기준:**
  - X2a 기준 경로와 KL이 엔진 차이 이하다. 커널 단위로는 CPU 참조(X0)와 상대 오차가 FP16 수준이다.
  - 실효 대역폭이 피크의 80 % 이상이다(약 190 GB/s 이상, `tools/bench`, `tools/prof`).
  - tg128 d0이 ExLlamaV3 같은 팩 이상이다(5.00: 12.37 tok/s 이상, 4.00: 15.56 이상).
  - DFlash2 텍스트 수락률이 같은 출력에서 유지되고, 실효 tok/s가 UD-Q4_K_XL + DFlash2(29.16)보다 높다(5.00 기준).
  - 속도 판정은 main과 짝지은 요청·단계별 5 %·10 ms다.
- prefill GEMM은 X3 범위다. X2 동안 prefill은 X2a 기준 경로(dequant + GEMM)를 쓴다.

## 파일 경계

- **X2a:**
  - `src/core/quant/exl3/**`
  - `src/models/qwen/` 로더·바인딩(`exl3_weights` 등)
  - `tests/core/quant/exl3*`, `tests/models/qwen27b/exl3*`
  - `plan/exl3/XA-RESULTS.md`
- **X2b:**
  - `src/core/hip/detail/exl3_*`(공통 디코드)
  - `src/models/qwen/hip/kernels/exl3_*`, `src/models/qwen/hip/ops/` 해당 부분
  - 활성화 생성 커널의 Hadamard 융합 부분
  - `tests/models/qwen27b/exl3_kernel*`
  - `plan/exl3/XB-RESULTS.md`
- **함께 고치는 곳:** 바인딩과 커널 선택처럼 둘이 같이 손대는 곳은 먼저 감독에게 알리고 순서를 정한다. 공유 `CMakeLists.txt`에는 자기 hunk만 넣는다.

## 순서와 허락

1. 둘 다 먼저 설계 메모를 쓰고 `WAITING`으로 보고한다(읽기만). X2a 로더의 디바이스 텐서 구조와 X2b 커널 인터페이스를 맞춘다.
2. 빌드는 무거운 CPU 작업이라 감독 허락이 필요하다. timing 창과 겹치면 안 된다.
3. GPU 창은 감독이 정한 순서대로 연다.
