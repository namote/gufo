# X3 결과: EXL3 prefill GEMM (진행 중, W3, 2026-10-31)

상위 문서: [X3-DESIGN.md](X3-DESIGN.md)(0절이 외부 검토 반영), [PLAN.md](PLAN.md) X3.

## 1단계: 독립 bench (`tools/bench/exl3_prefill_bench.hip`)

- **틀:** 운영 `HalfPrefillGemmKernel<256, 256, 8, 4>`와 같다. BM 256 채널 × BN 256 토큰, 32 wave, WMMA f16, FP32 누적, 스위즐 LDS다.
- **생산자 변형:**
  - FP16 미리 디코드 대조
  - trellis v1(`v_mul_lo_u32` 해시)
  - trellis v2(pk 해시)
  - trellis v3(pk 해시, K32 단계)
  - v2 + 출력 회전(죽은 GEMM LDS 재사용)
  - 입력 회전 커널(FP32 → FP16 행 우선)과 전체 파이프라인
- **GGUF 기준:** 운영 경로를 소스에서 링크했다(prefill_fp16, prefill_quant_gemm, wave64 단위). 1024행 미만은 Q8_1 양자화 + W-quant × A8 GEMM, 1024행 이상은 FP16 변환 + FP16 GEMM이다. Q4_K, Q6_K, IQ4_XS 각각 생산자 시간을 포함한다.
- **check:**
  - pk 해시 65,536상태 전수 검사(위쪽 16비트 쓰레기 포함)
  - 모든 생산자 대 FP64(1e-5), FP16 대조와의 비트 동일성(K4/5/6)
  - 출력 회전 대 FP64, 입력 회전
- **빌드:** 스크립트는 `build/x3-bench/build.sh`(커밋하지 않음), 이미지 `ai-tools/gufo-eval:7c2c5a728270`이다.
  - 자원(X3-DESIGN 0절 표): 모든 변형에서 scratch와 spill이 0이다. VGPR 138–162, occupancy 9–10, LDS 64 KiB(K32는 32 KiB)다.
- **검토 반영(2026-10-31, Codex 확인 뒤):**
  - 전체 파이프라인은 생산자와 GEMM을 한 이벤트 쌍 안에서 잰다. 따로 잰 값도 같이 남긴다.
  - 결과는 projection 마이크로벤치로 표시한다. FFN 체인 비교를 하나 더한다(GGUF는 운영 경로와 실제 혼합 타입). EXL3 쪽은 생산자를 아직 융합하지 않은 상한이다.
  - 틀을 운영과 맞췄다: 블록 재배열(`prefill_fp16.hip:288`, 그룹 8)을 넣었다. FP16 대조는 같은 trellis에서 디코드한 가중치를 쓴다. k/v의 GGUF IQ4_XS/Q8_0 등은 운영 분기가 작은 타일을 고른다(`prefill_fp16.hip:835`).
  - MALL: 사본을 한 바퀴 다 돌게 예열하고 잰다. 지나간 가중치 바이트(MB)를 행마다 적는다.
  - check 추가:
    - K = 17408
    - 출력 유한성
    - 입력 회전을 FP32 계약 참조와 FP16 바이트 단위로 비교
    - 회전 → GEMM → 출력 회전 합성 검사
    - 65,536상태 전수 디코드 대 `DecodeState`
  - 빌드: 바이너리 sha256 `45552647e4177477df4cbdf068c177913e517e2622265562a417e4f2a60a91b1`, 소스 sha256 `f18d6d9692aece387055aaecb9c0fbe91ec8936449ff69509ad1b074e2c02c23`. scratch와 spill은 0이다. VGPR은 FP16 대조 155, trellis 138–151이다.
- **창 계획:** `PLAN-X3B1.json`(check, time 10, peak, idle). F1 단위 창 뒤에 돈다.

## 운영 구현 (사용자 2026-10-31: 먼저 다 만들고 끝에 한 번에 비교한다)

- bench 창 X3B1과 bench 다듬기는 멈췄다. 검토 반영 항목은 위 기록으로만 남긴다.
- **커밋 `9d5c937b`:**
  - `prefill_fp16.hip` `HalfPrefillGemmKernel`에 템플릿 인자 `TrellisBits`를 더했다. trellis 생산자(pk 해시, `HashMul1Packed`)와 svh 출력 회전 epilogue(죽은 s_a 재사용, store/residual)다. GGUF 인스턴스는 버려지는 `if constexpr` 가지만 늘었다. 지문은 같다(아래).
  - `LaunchExl3PrefillGemm`: prefill은 256 채널 × 256 토큰 타일이고, 64행 이하(X3b)는 128 채널 × 64 토큰 타일이다.
  - `LaunchExl3RotateRows`: 행 우선 FP16 입력 회전(X2b 산술)이다.
  - arena에 `fast_prefill_input`(max batch × 가장 넓은 입력, 142.6 MB)을 두었다. EXL3 모델일 때만 잡는다.
  - dispatch:
    - 17행 이상: WMMA GEMM
    - 16행 이하: X2b GEMV
    - 나머지: 기준 경로
    - 경계 17은 잠정값이다. 마지막 창에서 9–64행을 비교한다.
    - 묶음과 MLP 호출은 17행 이상이면 투영마다 WMMA로 간다.
- **단위 창(짧은 GPU 창, 사용자 허락):**
  - `x3u1`(FAILED):
    - 커널 시험의 prefill 두 경우가 FP64 1e-5 기준을 넘었다: k = 17408에서 5.4e-5, k = 5120에서 1.6e-5.
    - 나머지는 통과했다: 디코드 경로 전부, pk 해시 65,536상태, 행 회전 FP16 바이트 동일, k ≤ 1024 prefill 3.2e-6.
    - `exl3_reference_test`가 통과했고, **UD-Q4_K_M 지문 27줄이 rq2h와 바이트 단위로 같다.**
  - `x3u2`(진단): GPU의 회전 전 합을 K16 단계 에뮬레이션과 비교했다.
    - 최근접 반올림 에뮬레이션과는 FP64와 같은 차이가 났다. 0 방향 내림 에뮬레이션과는 그 절반이었다.
    - **오차는 WMMA FP16 누적의 성질이다.** k에 비례해 대략 3.1e-9 × k다. 디코드 오류가 아니다(행 회전은 바이트 동일이고, k ≤ 256에서는 1e-5 안).
  - `x3u3`(COMPLETE): prefill 기준을 같은 k·같은 활성화에서 운영 GGUF FP16 WMMA GEMM(Q8_0)이 낸 FP64 대비 오차의 1.25배로 바꿨다.
    - 결과: EXL3 오차가 GGUF 오차와 같은 크기다. 비율은 모두 0.80(= 1/1.25)이다. 예: k = 17408에서 5.45e-5 대 GGUF 5.42e-5, k = 1024에서 3.25e-6 대 3.20e-6.
    - **모든 검사를 통과했다.**
