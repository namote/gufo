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
- **빌드:**
  - 스크립트: `build/x3-bench/build.sh`(커밋하지 않음), 이미지 `ai-tools/gufo-eval:7c2c5a728270`, 약 1분 50초
  - 바이너리 `build/x3-bench/exl3_prefill_bench`(sha256 `d47a1f4353e1…`), 소스 `fbdc279260a8…`
  - 자원(X3-DESIGN 0절 표): 모든 변형에서 scratch와 spill이 0이다. VGPR 138–162, occupancy 9–10, LDS 64 KiB(K32는 32 KiB)다.
- **창 계획:** `PLAN-X3B1.json`(check, time 10, peak, idle, 약 5분). F1 단위 창 뒤에 돈다.
