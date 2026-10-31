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
