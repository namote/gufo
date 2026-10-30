# X-B 결과: EXL3 trellis decode GEMV·verify 커널 (진행 중, W3, 2026-10-30)

상위 문서: [XB-BRIEF.md](XB-BRIEF.md) X2b, 설계 [XB-DESIGN.md](XB-DESIGN.md)(`2d8fcc4f`). 이 문서는 X-B(W3)가 쓴다.

## 상태

- **설계 확정(감독 2026-10-30):**
  - m > 8은 8행 묶음으로 반복한다. 큰 배치는 X3 WMMA로 넘긴다.
  - 활성화는 FP16으로 반올림한다. 커널 기준은 X0 대비 상대 오차 약 1e-3, FP16 입력 FP64 대비 약 1e-5다. 모델 기준(X2a 기준 경로 대비 KL이 엔진 차이 수준)은 그대로다.
  - 순서: X-A가 기준 경로를 분기 지점과 arena에 먼저 연결하고, 그 뒤 X2b 빠른 커널이 같은 분기에 들어간다.
- **독립 대역폭 실험:** 빌드 완료. GPU 창(`PLAN-XB1.json`)은 F1의 P1T/K0 창 뒤에 감독이 "start"를 주면 돈다.

## 목표 (감독 확정)

- **GEMV 커널 실효 대역폭 210 GB/s 이상(DRAM read 약 240 GB/s의 88 %).** 브리프의 80 %(190 GB/s)에서 올렸다.
  - 근거: 5.00 팩은 토큰당 16.22 GB다. upstream tg128 12.37 tok/s를 넘으려면 decode 전체가 약 201 GB/s 이상이어야 한다. 190 GB/s면 약 11.9 tok/s에 그친다.
- **관건은 m = 8 verify에서 대역폭을 유지하는 것이다.** DFlash2 실효 29.16 tok/s 기준과의 여유가 작다(설계 7절).

## 독립 대역폭 실험 (`tools/bench/exl3_gemv_bench.hip`)

- **코어(설계 3.1절):**
  - lane L이 타일 위치 8L … 8L + 7을 푼다. 그 8개 상태는 타일 비트열의 64비트 창 하나 안에 있다(K ≤ 6). 이 창을 dword 세 개에서 `v_alignbit_b32` 두 번으로 잘라 내므로, 상태마다 상수 shift 하나다.
  - mul1 짝 디코드는 `v_sad_u8`/`v_sad_hi_u8` + `v_pk_fma_f16`이다. 곱은 `v_dot2_f32_f16`(FP32 누적)이다.
  - 블록 안 warp들이 K를 나누고, LDS에서 warp 순서로 감소한다.
  - 출력 epilogue는 store형이다. 128열 구간을 마지막으로 끝낸 블록이 `svh ⊙ H(acc)`를 FP32로 쓴다(자기 초기화 카운터).
- **해시 두 형태:**
  - 0: `v_mul_lo_u32`
  - 1: 24비트 곱 둘 + shift
- **check 모드:**
  - 디코드 FP16 비트를 CPU 참조(`src/core/quant/exl3`)와 비교한다(K 4/5/6).
  - GEMV를 FP64(같은 FP16 입력)와 비교한다. 상대 L2 기준은 1e-5다.
  - M = 2/4/8의 행 r이 M = 1과 비트 단위로 같은지 본다.
- **time 모드:**
  - 모양 10개: 27B 본문 K = 5 일곱, lm_head K = 6, K = 4 둘
  - M 1/2/4/8 × warp 4/8/16 × wave당 N 타일 1/2 × 해시 2형태
  - launch 20회의 중앙값. 가중치는 256 MB가 넘는 사본을 돌려 가며 읽는다.
  - 같은 버퍼의 평범한 읽기 속도도 같이 잰다(상한).
- **빌드:**
  - 명령: `docker run --rm -v <gufo>:/src -w /src --user $(id -u):$(id -g) --entrypoint sh ai-tools/gufo-eval:7c2c5a728270 build/xb-bench/build.sh`(커밋하지 않은 스크립트)
  - 플래그: `hipcc -O2 -g -DNDEBUG -std=c++20 --offload-arch=gfx1151`, CPU 참조 `src/core/quant/exl3/format.cpp`를 함께 링크
  - 결과: 바이너리 `build/xb-bench/exl3_gemv_bench`(sha256 `e0743eae…`), 소스 sha256 `5b61ff03…`
  - 커널 144개 모두 spill과 scratch가 0이다. VGPR은 25–79, occupancy는 16 waves/SIMD다.
