# R0b 결과: 타입별 실효 대역폭 (2026-10-28)

상위 문서: [회전 가중치 양자화 계획](PLAN.md) "R0b", [작업 안내](R-BRIEF.md) "R0b". 무작위 가중치(타입별 실제 블록 구조)로 Qwen 27B 행렬 크기에서 타입별 실효 대역폭과 토큰당 시간을 쟀다. 모델 파일은 쓰지 않았다.

## 요약

- **AR(1행)에서 바이트가 시간으로 이어지는 타입:** Q8_0, Q6_K, Q5_K, IQ4_XS, IQ4_NL은 DRAM 읽기 상한(240 GB/s)의 86–91 %, Q4_K는 82 %. 이 범위에서는 bpw를 내리면 토큰 시간이 거의 비례해서 준다(IQ4_XS 64.0 ms가 가장 빠른 운영 타입).
- **연산에 묶이는 타입:** Q3_K와 IQ3_S는 AR에서 상한의 54 %(130 GB/s)에 그쳐 **바이트가 24 % 적은데도 Q4_K(73.4 ms)·IQ4_XS(64.0 ms)보다 느리다(약 85 ms)**. DFlash2 verify(8행)에서는 34–38 %로 더 떨어져 step 시간이 Q8_0(128 ms)과 비슷하거나 더 길다(Q3_K 133 ms, IQ3_S 120 ms). 지금 커널로는 Q3_K·IQ3_S가 속도 이득을 주지 않는다.
- **2비트대(upstream ggml mmvq):** Q2_K는 AR에서 상한의 82 %(42.5 ms, 측정한 타입 중 가장 빠름). IQ2_XXS/XS/S는 54–59 %(51–58 ms)로 연산에 묶인다. verify 8행에서는 모두 19–27 %(Q2_K 186 ms, IQ2_* 111–126 ms)로, upstream mmvq는 열마다 디코드를 반복해 8행에서 Q8_0보다 느리거나 비슷하다. 2비트대를 verify에 쓰려면 운영 small-batch 커널처럼 디코드를 행 사이에 공유하는 커널이 필요하다.
- **prefill:** 모든 타입이 연산에 묶인다(512: int8 WMMA 14–31 TOPS, 상한 57; 2048: FP16 경로 20–37 TFLOPS, 상한 55). 토큰당 1.4–2.5 ms로 bpw와 거의 무관하고, 512에서는 Q8_0이 가장 빠르다.
- **FWHT(독립 커널, 128점):** AR·verify에서는 K와 무관하게 launch당 약 5 µs(launch에 묶임). 27B 한 토큰의 회전 257회(K 5120 129회, 6144 64회, 17408 64회)를 독립 launch로 하면 AR 토큰당 1.26 ms(IQ4_XS AR의 2.0 %), verify step당 1.35 ms, prefill 512 토큰당 0.048 ms, 2048 토큰당 0.076 ms. AR·verify의 비용은 거의 전부 launch 수라서 R2의 융합이 필요하다.

## 무엇을 어떻게 쟀나

- **바이너리:** `tools/bench/rq_type_bench.hip`, `tools/bench/rq_type_bench_mmvq.hpp`(커밋 `d19126df`). 빌드 스크립트는 아래 "빌드"(`build/rq-bench/build.sh`, 커밋하지 않음). 운영 코드는 바꾸지 않았다.
- **운영 타입(Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S):** 운영 커널 소스(`src/models/qwen/hip/kernels/` gemv, gemv_quant, qkv, ssm, swiglu, prefill_quant_gemm, prefill_fp16, prefill_quant_wave64, small_batch_wave64, small_batch_quant16_wave64)를 **복사하지 않고** 리비전 `d19126df` 트리에서 release와 같은 플래그(`-O2 -g -DNDEBUG`, CMake의 파일별 옵션: wave64, `-fno-slp-vectorize`, iterative-ilp)로 컴파일해 링크했다. 운영 라우팅을 따랐다(모델의 모든 행렬이 같은 타입이라고 가정):

  | 배치 | 운영 경로(선택 로직) | 부른 함수 |
  |---|---|---|
  | AR 1 | `decode_step.cpp`: 행렬별 `LaunchGEMV`(decode quant → `kHipQuantDirect` → `LaunchQ8KBlockGEMV`), Q/K/V·SSM 입력·FFN gate/up은 융합 launch | `LaunchGEMV`, `LaunchFusedQKVProjections`, `LaunchFusedSSMInputProjections`, `LaunchFusedSwiGLUGEMV` |
  | verify 8 | `batched_decode.cpp` `LaunchProjection` → `LaunchBatchedQuantGEMMFp32`. 8 = `DFlashLengthController::kMaxDraftTokens`(7) + anchor(`dflash/executor.cpp` `block_count = draft_count + 1`). adaptive 정책은 2–8행을 쓴다 | `LaunchBatchedQuantGEMMFp32` |
  | prefill 512 | `prefill_chunk.cpp`: Q8_1 활성화 + `LaunchBatchedQuantGEMMPreQuantized` | 같음(활성화 양자화는 시간 밖) |
  | prefill 2048 | `UseQwen27bFp16Prefill`(batch ≥ 1024, 저비트 FFN): `LaunchBatchedQuantGEMMFp16`. 전부 Q8_0인 모델은 저비트 FFN이 없어 Q8_1 경로 | Q8_0: PreQuantized, 그 밖: Fp16 |

- **2비트대(Q2_K, IQ2_XXS, IQ2_XS, IQ2_S):** upstream llama.cpp `ggml/src/ggml-cuda/mmvq.cu`·`vecdotq.cuh`·`quantize.cu`(MIT, The ggml authors, 커밋 `5c0e9468378eba6bf3cc1989ff5d62fbbe4d9e3a`). 복사하지 않고 저장소에 이미 있는 HIP 사본 `src/models/deepseek_v4_flash/kernels/rocm/mmq/`(VENDOR.md에 같은 커밋)의 `mul_mat_vec_q_switch_type`과 `ds4_quantize_row_q8_1_hip`를 그 타깃의 플래그(`-O3 -ffast-math -fno-finite-math-only -std=c++17 -DGGML_USE_HIP`)로 컴파일해 불렀다. gfx1151(RDNA3.5)에서 upstream은 RDNA2 매개변수 표(출력 행·블록당 wave32 하나)를 고른다. 헤더 `rq_type_bench_mmvq.hpp`에 출처와 라이선스를 적었다. **prefill은 생략했다**(이 타입들의 prefill 커널이 gufo에 없다). 활성화 Q8_1 양자화는 시간 밖이다.
- **가중치:** 무작위 바이트 + 블록마다 유한한 FP16 scale(d, dmin). 운영과 같이 익명 huge-page mmap + `hipHostRegister(Mapped|ReadOnly)`(`model_loader.cpp`). 행렬마다 복사본을 두어 합이 96 MiB(MALL 32 MiB의 3배) 이상이 되게 하고 launch마다 돌렸다(output 행렬은 복사본 1개로 이미 그보다 크다).
- **크기:** attn_qkv 5120→10240(×48/토큰), attn_gate 5120→6144(×48), attn_q 5120→12288(×16), attn_k·attn_v 5120→1024(×32), ssm_out·attn_output 6144→5120(×64), ffn_gate·ffn_up 5120→17408(×128), ffn_down 17408→5120(×64), output 5120→248320(×1, AR·verify만. prefill은 마지막 토큰만 logits를 낸다). ssm_alpha·ssm_beta(5120→48, 토큰당 0.05 GB 미만)는 AR 융합 launch 안에서만 읽었고 행렬별 합에는 넣지 않았다.
- **측정:** 항목마다 워밍업 3회 뒤 launch마다 hipEvent 쌍으로 쟀다. 4라운드 × 8회 = 32표본(FWHT 포함). 라운드마다 타입 순서를 바꾼다(짝수 라운드 정방향, 홀수 역방향, 라운드마다 시작점 회전). 실효 GB/s = 가중치 바이트 / 중앙값. 표의 괄호는 (중앙값 µs / p95 µs).
- **토큰 추정:** 행렬별 중앙값 × 토큰당 개수의 합(가중치 GEMM만; attention, DeltaNet 재귀, norm, 임베딩은 빠짐). AR은 운영 융합 launch를 쓴 "decode-fused" 합, 2비트대는 행렬별 합. verify는 8행 step 시간과 토큰당(= step / 8). prefill은 output 없이 step / 배치.
- **정확성 검사(`check`):** 운영 타입은 `src/core/quant/ggml_dequant.cpp`의 CPU 역양자화로, 2비트대는 ggml-quants.c `dequantize_row_*`를 따라 새로 쓴 스칼라 역양자화로 첫·끝 32개 출력 행 × 활성화 행 0, 1, b−1의 내적을 만들어 |gpu − ref| / Σ|w·x|를 봤다. 기준: FP32 경로 1e-4, Q8_1 경로 1e-2, FP16 경로 3e-3. FWHT는 CPU Hadamard(double)와 비교하고, D = 1로 두 번 적용해 원래 값으로 돌아오는지 봤다.

## 창

- 창 계획: `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-RQ0B.json`(model `r0b`: check, time `4 8`, peak, idle 20 s). 이미지 `ai-tools/gufo-eval:7c2c5a728270`(ROCm 7.2.4). LOCKS.py 갱신.
- 실행: `rq0b-a01`, `rq0b-a02`는 다른 세션 창이 lock을 잡고 있어 **시작 전 lock 거절**(실행 아님). `rq0b-a03`(2026-10-28 14:23 KST)을 한 번 돌렸다: status COMPLETE, health_PASS true, pre/post KFD 컨텍스트 없음·busy 0 %. 테스트 시간: check 3.0 s, time 31.3 s, peak 3.3 s.
- 기록: `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/rq0b-a03/r0b/receipt/run/{check,time,peak}.log`. 바이너리 SHA-256: `rq_type_bench` `2a4c69e7ce8a665c…`, `gfx1151_peak` `a3205d8bfdef4791…`. RESULT.json의 `source_revision`(`40aa6393`)은 `/src/.gufo-revision` 파일 값으로, 다른 세션 파일이라 건드리지 않았다(이 실행의 소스는 `d19126df`).
- `time.log`에서는 stderr의 "round n/4 done" 줄이 버퍼된 stdout 줄 중간에 끼어 있다. 표는 그 줄을 지운 뒤 읽었다(RESULT 328줄 모두 읽힘).

## 검사 결과

`check`: **PASS**(67개 항목 모두 ok). 최대 상대 오차: FP32 경로(AR·verify) 1.1e-8, Q8_1 prefill 2.9e-4, FP16 prefill 1.6e-5, 2비트 mmvq 2.9e-4. FWHT: K 5120/6144/17408에서 최대 절대 오차 7.0e-7–8.1e-7(참값 최대 6.1–6.9), 왕복 오차 7.2e-7.

## 상한 (같은 창, `tools/bench/gfx1151_peak.hip`)

| 항목 | 값 |
|---|---|
| DRAM read | 240.07 GB/s |
| DRAM write / copy | 218.37 / 209.23 GB/s |
| WMMA int8 / fp16 / bf16 | 57.02 TOPS / 55.34 / 55.29 TFLOPS |
| VALU fp32 FMA | 24.82 TFLOPS |

상한 비율은 DRAM read 240.07 GB/s 기준이다. 상한 커널은 hipMalloc 메모리를 읽고, 이 벤치의 가중치는 운영처럼 host-registered 메모리다(Strix Halo는 둘 다 같은 LPDDR).

## 타입별 bpw 대 실효 대역폭 대 토큰당 시간 (27B 가중치 GEMM 전체)

weight GB는 output을 포함한 27B dense 행렬 전체(임베딩 제외). ideal = weight GB / 240.07 GB/s. AR은 운영 융합 경로(2비트대는 행렬별 합). prefill 512는 Q8_1 경로, 2048은 운영 선택(Q8_0만 Q8_1, 나머지 FP16).

| type | bpw | weight GB | ideal ms | AR ms/token | AR GB/s (상한 %) | verify8 step ms | verify8 ms/token | verify8 GB/s (상한 %) | pp512 ms/token | pp2048 ms/token |
|---|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | 8.500 | 27.20 | 113.3 | 126.2 | 216 (90 %) | 128.2 | 16.03 | 212 (88 %) | 1.592 | 1.570 |
| Q6_K | 6.562 | 21.00 | 87.5 | 96.7 | 217 (91 %) | 109.6 | 13.71 | 192 (80 %) | 2.110 | 1.503 |
| Q5_K | 5.500 | 17.60 | 73.3 | 83.4 | 211 (88 %) | 90.2 | 11.28 | 195 (81 %) | 1.771 | 1.367 |
| Q4_K | 4.500 | 14.40 | 60.0 | 73.4 | 196 (82 %) | 89.0 | 11.13 | 162 (67 %) | 1.731 | 1.375 |
| IQ4_XS | 4.250 | 13.60 | 56.6 | 64.0 | 212 (89 %) | 89.1 | 11.14 | 153 (64 %) | 1.685 | 1.422 |
| IQ4_NL | 4.500 | 14.40 | 60.0 | 70.1 | 206 (86 %) | 91.4 | 11.43 | 158 (66 %) | 1.651 | 1.464 |
| Q3_K | 3.438 | 11.00 | 45.8 | 84.8 | 130 (54 %) | 133.2 | 16.66 | 83 (34 %) | 2.490 | 1.623 |
| IQ3_S | 3.438 | 11.00 | 45.8 | 85.0 | 130 (54 %) | 120.5 | 15.06 | 91 (38 %) | 1.856 | 1.720 |
| Q2_K | 2.625 | 8.40 | 35.0 | 42.5 (행렬별) | 198 (82 %) | 186.1 | 23.26 | 45 (19 %) | — | — |
| IQ2_XXS | 2.062 | 6.60 | 27.5 | 51.1 (행렬별) | 129 (54 %) | 110.9 | 13.87 | 60 (25 %) | — | — |
| IQ2_XS | 2.312 | 7.40 | 30.8 | 52.3 (행렬별) | 142 (59 %) | 119.0 | 14.88 | 62 (26 %) | — | — |
| IQ2_S | 2.562 | 8.20 | 34.2 | 57.9 (행렬별) | 142 (59 %) | 126.5 | 15.81 | 65 (27 %) | — | — |

운영 타입의 AR 행렬별 합(융합 launch 대신 행렬마다 `LaunchGEMV`): Q8_0 124.5, Q6_K 98.2, Q5_K 84.0, Q4_K 75.2, IQ4_XS 62.9, IQ4_NL 71.3, Q3_K 89.6, IQ3_S 87.4 ms. IQ4_XS만 융합 쪽이 느리다(융합 SSM 입력 187 GB/s, 융합 QKV 182 GB/s 대 행렬별 199–217 GB/s. 융합 커널이 IQ4_XS를 정적 특수화하지 않는 것으로 보인다 — 짐작, ISA는 보지 않았다).

## 행렬별 측정

열 이름 attn_k_v는 attn_k·attn_v 각각 하나, ssm_out_attn_out은 ssm_out·attn_output 각각 하나, ffn_gate_up은 ffn_gate·ffn_up 각각 하나의 시간이다. 배치 1의 fused 열은 운영 decode 융합 launch(alpha·beta 포함)다.

### 배치 1: 실효 GB/s (중앙값 µs / p95 µs)

| type | kernel | attn_qkv | attn_gate | attn_q | attn_k_v | ssm_out_attn_out | ffn_gate_up | ffn_down | output | fused_ssm_in(qkv+gate+a+b) | fused_qkv(q+k+v) | fused_ffn(gate+up) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | LaunchGEMV | 218 (256/257) | 213 (157/163) | 218 (306/329) | 175 (32/49) | 214 (156/157) | 221 (428/456) | 216 (438/460) | 229 (5909/5993) | 222 (403/405) | 221 (353/377) | 212 (895/903) |
| Q6_K | LaunchGEMV | 213 (202/210) | 194 (133/134) | 213 (242/247) | 137 (31/43) | 210 (123/125) | 216 (338/351) | 217 (336/345) | 226 (4604/4617) | 214 (324/332) | 208 (289/302) | 220 (665/675) |
| Q5_K | LaunchGEMV | 210 (172/178) | 186 (116/118) | 212 (204/206) | 117 (31/34) | 183 (118/125) | 218 (281/289) | 210 (291/311) | 230 (3803/3851) | 202 (287/297) | 193 (261/277) | 221 (553/565) |
| Q4_K | LaunchGEMV | 193 (152/160) | 160 (111/120) | 195 (182/198) | 97 (30/41) | 176 (101/110) | 199 (252/257) | 192 (261/265) | 212 (3375/3405) | 185 (257/261) | 177 (233/246) | 208 (482/496) |
| IQ4_XS | LaunchGEMV | 216 (129/136) | 199 (84/86) | 217 (154/162) | 115 (24/36) | 209 (80/86) | 221 (214/219) | 217 (218/220) | 228 (2968/2976) | 187 (239/242) | 182 (214/216) | 224 (424/433) |
| IQ4_NL | LaunchGEMV | 203 (145/149) | 165 (107/109) | 204 (174/181) | 101 (29/41) | 188 (94/109) | 207 (243/249) | 211 (237/249) | 221 (3231/3264) | 201 (236/239) | 194 (212/227) | 208 (483/493) |
| Q3_K | LaunchGEMV | 118 (190/197) | 97 (140/153) | 121 (224/236) | 65 (35/38) | 117 (115/125) | 126 (305/319) | 130 (295/306) | 142 (3838/3922) | 125 (290/297) | 123 (257/263) | 134 (573/580) |
| IQ3_S | LaunchGEMV | 122 (184/200) | 100 (135/142) | 120 (225/235) | 62 (37/39) | 120 (113/134) | 129 (296/312) | 136 (282/286) | 132 (4129/4161) | 128 (283/290) | 126 (250/264) | 129 (594/618) |
| Q2_K | ggml-mmvq | 192 (90/96) | 181 (57/59) | 201 (103/107) | 110 (16/16) | 177 (58/59) | 205 (143/146) | 200 (146/150) | 216 (1934/1982) | — | — | — |
| IQ2_XXS | ggml-mmvq | 126 (108/110) | 117 (69/70) | 127 (127/129) | 74 (18/19) | 121 (67/69) | 134 (172/178) | 130 (176/183) | 134 (2444/2487) | — | — | — |
| IQ2_XS | ggml-mmvq | 137 (110/112) | 126 (72/73) | 140 (130/132) | 82 (19/19) | 132 (69/71) | 146 (177/183) | 146 (177/182) | 146 (2508/2528) | — | — | — |
| IQ2_S | ggml-mmvq | 141 (119/123) | 130 (78/80) | 140 (144/148) | 83 (20/21) | 136 (74/76) | 145 (197/202) | 144 (199/206) | 147 (2768/2791) | — | — | — |

### 배치 8: 실효 GB/s (중앙값 µs / p95 µs)

| type | kernel | attn_qkv | attn_gate | attn_q | attn_k_v | ssm_out_attn_out | ffn_gate_up | ffn_down | output |
|---|---|---|---|---|---|---|---|---|---|
| Q8_0 | BatchedQuantGEMMFp32 | 209 (266/277) | 188 (178/188) | 213 (314/319) | 125 (44/49) | 202 (166/172) | 216 (438/445) | 217 (436/452) | 229 (5904/5939) |
| Q6_K | BatchedQuantGEMMFp32 | 185 (233/239) | 161 (160/173) | 179 (289/298) | 102 (42/43) | 191 (135/145) | 195 (375/393) | 200 (365/372) | 218 (4794/4862) |
| Q5_K | BatchedQuantGEMMFp32 | 191 (188/204) | 154 (140/151) | 192 (225/231) | 90 (40/41) | 180 (120/126) | 201 (305/324) | 212 (290/304) | 201 (4341/4481) |
| Q4_K | BatchedQuantGEMMFp32 | 148 (199/203) | 128 (138/146) | 140 (253/263) | 57 (52/53) | 114 (155/177) | 179 (280/299) | 191 (262/268) | 156 (4572/4713) |
| IQ4_XS | BatchedQuantGEMMFp32 | 138 (202/210) | 102 (164/174) | 135 (247/262) | 53 (53/54) | 142 (118/124) | 168 (282/289) | 175 (271/276) | 137 (4931/5017) |
| IQ4_NL | BatchedQuantGEMMFp32 | 152 (195/203) | 135 (131/149) | 149 (238/244) | 63 (47/48) | 148 (120/146) | 162 (309/320) | 168 (299/317) | 173 (4139/4223) |
| Q3_K | BatchedQuantGEMMFp32 | 80 (281/308) | 74 (184/190) | 82 (330/343) | 36 (62/63) | 76 (179/196) | 88 (433/458) | 80 (482/527) | 92 (5954/6001) |
| IQ3_S | BatchedQuantGEMMFp32 | 86 (263/275) | 78 (174/180) | 86 (315/325) | 40 (56/57) | 81 (167/184) | 94 (408/427) | 102 (375/394) | 96 (5694/5762) |
| Q2_K | ggml-mmvq | 44 (388/401) | 44 (235/245) | 45 (456/467) | 34 (51/52) | 44 (232/241) | 45 (646/659) | 47 (627/633) | 44 (9571/9662) |
| IQ2_XXS | ggml-mmvq | 61 (223/233) | 58 (139/150) | 60 (270/283) | 40 (34/36) | 58 (140/146) | 60 (384/394) | 61 (377/384) | 56 (5864/5910) |
| IQ2_XS | ggml-mmvq | 63 (240/248) | 59 (154/156) | 62 (292/299) | 42 (36/40) | 61 (150/159) | 62 (415/419) | 65 (396/410) | 59 (6243/6293) |
| IQ2_S | ggml-mmvq | 65 (259/273) | 64 (158/169) | 65 (310/321) | 45 (37/38) | 63 (159/168) | 65 (440/449) | 68 (421/428) | 60 (6783/6840) |

### 배치 512: 실효 GB/s (중앙값 µs / p95 µs)

| type | kernel | attn_qkv | attn_gate | attn_q | attn_k_v | ssm_out_attn_out | ffn_gate_up | ffn_down |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | Q8_1+PreQuantized | 32 (1736/1776) | 30 (1110/1169) | 31 (2142/2237) | 20 (273/293) | 32 (1027/1061) | 32 (2991/3024) | 32 (2924/2985) |
| Q6_K | Q8_1+PreQuantized | 19 (2281/2479) | 17 (1506/1574) | 18 (2799/2837) | 14 (316/327) | 19 (1376/1464) | 18 (3978/4172) | 19 (3848/3904) |
| Q5_K | Q8_1+PreQuantized | 19 (1938/1964) | 17 (1262/1340) | 18 (2380/2448) | 12 (292/309) | 19 (1123/1200) | 18 (3330/3404) | 19 (3245/3288) |
| Q4_K | Q8_1+PreQuantized | 16 (1874/1919) | 14 (1274/1332) | 15 (2335/2412) | 10 (292/295) | 16 (1094/1137) | 15 (3247/3294) | 16 (3171/3204) |
| IQ4_XS | Q8_1+PreQuantized | 15 (1853/1958) | 14 (1202/1286) | 15 (2269/2301) | 10 (287/296) | 16 (1046/1106) | 15 (3196/3245) | 16 (3045/3087) |
| IQ4_NL | Q8_1+PreQuantized | 16 (1798/1902) | 15 (1176/1275) | 16 (2207/2240) | 12 (254/258) | 17 (1044/1158) | 16 (3126/3164) | 17 (3004/3132) |
| Q3_K | Q8_1+PreQuantized | 8 (2731/2815) | 8 (1710/1761) | 8 (3272/3357) | 6 (377/382) | 8 (1667/1699) | 8 (4577/4713) | 8 (4759/4808) |
| IQ3_S | Q8_1+PreQuantized | 11 (2054/2193) | 11 (1267/1369) | 11 (2492/2550) | 8 (301/308) | 11 (1190/1312) | 11 (3497/3571) | 11 (3400/3655) |

배치 512 연산률 TOPS (2·b·M·K / 중앙값):

| type | attn_qkv | attn_gate | attn_q | attn_k_v | ssm_out_attn_out | ffn_gate_up | ffn_down |
|---|---|---|---|---|---|---|---|
| Q8_0 | 30.9 | 29.0 | 30.1 | 19.7 | 31.4 | 30.5 | 31.2 |
| Q6_K | 23.5 | 21.4 | 23.0 | 17.0 | 23.4 | 22.9 | 23.7 |
| Q5_K | 27.7 | 25.5 | 27.1 | 18.4 | 28.7 | 27.4 | 28.1 |
| Q4_K | 28.6 | 25.3 | 27.6 | 18.4 | 29.4 | 28.1 | 28.8 |
| IQ4_XS | 29.0 | 26.8 | 28.4 | 18.7 | 30.8 | 28.6 | 30.0 |
| IQ4_NL | 29.9 | 27.4 | 29.2 | 21.1 | 30.8 | 29.2 | 30.4 |
| Q3_K | 19.7 | 18.8 | 19.7 | 14.3 | 19.3 | 19.9 | 19.2 |
| IQ3_S | 26.1 | 25.4 | 25.9 | 17.9 | 27.1 | 26.1 | 26.8 |

### 배치 2048: 실효 GB/s (중앙값 µs / p95 µs)

| type | kernel | attn_qkv | attn_gate | attn_q | attn_k_v | ssm_out_attn_out | ffn_gate_up | ffn_down |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | Q8_1+PreQuantized | 8 (6914/7063) | 8 (4265/4490) | 8 (8420/8540) | 7 (820/858) | 8 (4128/4171) | 8 (11833/12043) | 8 (11556/11872) |
| Q6_K | BatchedQuantGEMMFp16 | 7 (6554/6727) | 6 (4100/4271) | 6 (8045/8359) | 5 (837/878) | 6 (4713/4875) | 6 (11184/11382) | 7 (10604/10857) |
| Q5_K | BatchedQuantGEMMFp16 | 6 (5852/6017) | 6 (3611/3702) | 6 (7289/7439) | 5 (760/788) | 5 (4158/4312) | 6 (10223/10456) | 6 (9853/10223) |
| Q4_K | BatchedQuantGEMMFp16 | 5 (5988/6135) | 5 (3709/3810) | 5 (7438/7605) | 4 (749/781) | 5 (3858/4201) | 5 (10391/10680) | 5 (9866/10051) |
| IQ4_XS | BatchedQuantGEMMFp16 | 4 (6140/6319) | 4 (3798/3918) | 4 (7556/7752) | 3 (913/962) | 4 (4520/4613) | 4 (10630/10826) | 5 (9926/10143) |
| IQ4_NL | BatchedQuantGEMMFp16 | 5 (6318/6551) | 4 (3936/4128) | 4 (7812/8010) | 3 (864/894) | 4 (4757/5043) | 5 (10871/11041) | 5 (10274/10429) |
| Q3_K | BatchedQuantGEMMFp16 | 3 (6937/7034) | 3 (4291/4386) | 3 (8543/8750) | 2 (1029/1067) | 2 (5593/5928) | 3 (11828/12008) | 3 (11627/11728) |
| IQ3_S | BatchedQuantGEMMFp16 | 3 (7358/7449) | 3 (4567/4633) | 3 (8981/9082) | 2 (1074/1099) | 2 (5747/6017) | 3 (12568/12657) | 3 (12448/12584) |

배치 2048 연산률 TFLOPS (2·b·M·K / 중앙값):

| type | attn_qkv | attn_gate | attn_q | attn_k_v | ssm_out_attn_out | ffn_gate_up | ffn_down |
|---|---|---|---|---|---|---|---|
| Q8_0 | 31.1 | 30.2 | 30.6 | 26.2 | 31.2 | 30.9 | 31.6 |
| Q6_K | 32.8 | 31.4 | 32.0 | 25.6 | 27.3 | 32.6 | 34.4 |
| Q5_K | 36.7 | 35.7 | 35.4 | 28.3 | 31.0 | 35.7 | 37.0 |
| Q4_K | 35.9 | 34.7 | 34.6 | 28.6 | 33.4 | 35.1 | 37.0 |
| IQ4_XS | 35.0 | 33.9 | 34.1 | 23.5 | 28.5 | 34.3 | 36.8 |
| IQ4_NL | 34.0 | 32.7 | 33.0 | 24.9 | 27.1 | 33.6 | 35.5 |
| Q3_K | 31.0 | 30.0 | 30.2 | 20.9 | 23.0 | 30.9 | 31.4 |
| IQ3_S | 29.2 | 28.2 | 28.7 | 20.0 | 22.4 | 29.1 | 29.3 |


## FWHT 비용 (독립 커널, 128점, 부호 곱·정규화, FP32 제자리)

커널: wave32 하나가 128점 묶음 하나(레인당 4값, 레인 안 2단계 + `__shfl_xor` 5단계), 256스레드 블록. I/O GB/s = 읽기+쓰기 바이트 / 중앙값.

| K | batch | 중앙값 µs | p95 µs | I/O GB/s |
|---|---|---|---|---|
| 5120 | 1 | 4.87 | 5.41 | 8.4 |
| 5120 | 8 | 5.07 | 5.29 | 64.6 |
| 5120 | 512 | 25.81 | 30.87 | 812.5 |
| 5120 | 2048 | 374.27 | 380.64 | 224.1 |
| 6144 | 1 | 4.89 | 5.18 | 10.1 |
| 6144 | 8 | 5.21 | 5.49 | 75.5 |
| 6144 | 512 | 31.22 | 36.89 | 806.1 |
| 6144 | 2048 | 454.06 | 457.88 | 221.7 |
| 17408 | 1 | 4.97 | 5.28 | 28.0 |
| 17408 | 8 | 5.73 | 6.09 | 194.4 |
| 17408 | 512 | 301.37 | 312.37 | 236.6 |
| 17408 | 2048 | 1227.87 | 1243.06 | 232.3 |

27B 한 토큰의 입력 회전: K 5120 129회(층마다 attn_norm 출력 1, ffn_norm 출력 1, 그리고 output_norm 1), K 6144 64회(ssm_out·attn_output 입력), K 17408 64회(ffn_down 입력). 그룹 안 행렬이 같은 회전 입력을 공유한다고 가정했다.

| 배치 | step당 FWHT ms | 토큰당 ms | 같은 배치 IQ4_XS GEMM 대비 |
|---|---|---|---|
| AR 1 | 1.259 | 1.259 | 2.0 %(64.0 ms) |
| verify 8 | 1.354 | 0.169 | 1.5 %(89.1 ms step) |
| prefill 512 | 24.6 | 0.048 | 2.9 %(1.685 ms/token) |
| prefill 2048 | 155.9 | 0.076 | 5.4 %(1.422 ms/token) |

- AR·verify는 K와 무관하게 launch당 약 5 µs로 launch·지연에 묶인다. 이 비용은 독립 launch 257회의 값이고, R2에서 norm이나 활성화 양자화 커널에 융합하면 대부분 사라질 것으로 본다(짐작; 융합 비용은 R2에서 잰다).
- prefill 512의 K 5120·6144는 활성화(10–12 MB)가 MALL에 남아 800 GB/s 이상, K 17408(36 MB)부터와 2048은 DRAM에 묶여 약 220–237 GB/s다. 운영 prefill 활성화는 BF16이라 실제 I/O는 이 FP32 측정의 절반 정도일 것이다(짐작).

## 연산에 묶이는 타입과 근거

- **Q3_K, IQ3_S (운영 커널):** 같은 AR 커널(`QuantWarpBlockDot` 기반 GEMV)에서 큰 행렬(attn_k·attn_v 제외)의 실효 대역폭은 Q5_K·Q6_K·Q8_0·IQ4_XS·IQ4_NL 165–230 GB/s, Q4_K 160–212 GB/s인데 Q3_K·IQ3_S는 97–142 GB/s다. 시간이 바이트에 비례하지 않는다: AR 토큰당 Q3_K 84.8 ms, IQ3_S 85.0 ms로 바이트가 1.31배 많은 Q4_K(73.4 ms)보다 느리다. verify 8행에서는 80–102 GB/s로, Q8_0(212 GB/s)보다 step이 길거나 비슷하다. prefill 512에서도 Q3_K는 19–20 TOPS(Q8_0 30–31)로 가장 느리다. 원인은 Q3_K의 hmask·6비트 scale 풀기와 IQ3_S의 grid 조회 디코드로 보인다(짐작; `tools/prof/isa_mix.py`로 확인하지 않았다).
- **IQ2_XXS, IQ2_XS, IQ2_S (upstream mmvq):** AR 117–147 GB/s(큰 행렬, 상한의 49–61 %), 같은 mmvq 경로의 Q2_K는 177–216 GB/s라서 IQ2 grid 조회·부호 디코드가 병목으로 보인다. verify 8행: upstream mmvq는 열(활성화 행)마다 같은 블록을 다시 디코드하므로(`ncols_dst` 루프 안에서 `vec_dot` 호출) 8행에서 19–27 %로 떨어진다(큰 행렬에서 Q2_K 44–47 GB/s, IQ2_* 56–68 GB/s).
- **Q4_K, IQ4_XS, IQ4_NL의 verify:** AR에서는 82–91 %인데 verify 8행에서 64–67 %(Q5_K·Q6_K 80–81 %). 8행 공유 커널에서 디코드·FMA 비율이 커지는 것으로 보인다(짐작).
- **작은 행렬(attn_k·attn_v, 5120→1024, 1.4–5.6 MB):** AR에서 모든 타입이 16–37 µs에 머물러 GB/s가 낮다(62–175 GB/s). 대역폭보다 launch·꼬리 지연에 묶인다. 토큰당 32회라 AR 토큰에서 약 1 ms다.
- **prefill:** 모든 타입이 WMMA 연산에 묶인다. 512(Q8_1 int8 WMMA)는 상한 57 TOPS 대비 14–31 TOPS, 2048(FP16 경로)은 55 TFLOPS 대비 20–37 TFLOPS. bpw를 내려도 prefill 토큰 시간은 줄지 않는다(오히려 512에서 Q6_K·Q3_K가 Q8_0보다 33–56 % 느리다).

## R4·E7·E8에 주는 함의 (짐작 포함)

- R4 레시피 비용: 지금 커널에서 AR "싸게 읽히는" 운영 타입은 IQ4_XS(바이트당 시간이 가장 좋음), Q5_K, Q6_K, Q8_0. Q3_K·IQ3_S를 섞으면 바이트는 줄어도 AR·verify 시간이 늘 수 있으므로, E8 비용 모델에는 bpw가 아니라 이 표의 시간을 써야 한다.
- E7(2비트): Q2_K는 AR에서 대역폭에 가깝게 읽히므로 AR 전용이라면 후보다. 그러나 DFlash2 verify에서는 upstream mmvq로는 Q8_0보다 느리다. IQ2_*는 AR에서도 연산에 묶인다. E7b 진행 전에 2비트 small-batch 커널(행 공유 디코드)을 만들지 판단이 필요하다 — 사용자 결정 사항.
- 이 표는 단일 타입 모델 가정이다. 실제 혼합 레시피는 행렬별 표(위)로 합산한다.

## 빌드

eval 이미지 안에서(`/src` = gufo 트리). 호스트에서는
`docker run --rm -v /home/m2on/projects/gufo:/src -w /src --user $(id -u):$(id -g) --entrypoint sh ai-tools/gufo-eval:7c2c5a728270 build/rq-bench/build.sh`

```sh
#!/bin/sh
# R0b bench build. Runs inside the eval image with the gufo tree at /src.
# Production units use the release flags (-O2 -g -DNDEBUG) and their CMake
# source options; the vendored ggml units use the DeepSeek mmq target flags.
set -e
cd /src
out=build/rq-bench
obj=$out/obj
mkdir -p $obj
P="hipcc -O2 -g -DNDEBUG -std=c++20 --offload-arch=gfx1151 -I. -DENGINE_ENABLE_HIP=1 -Rpass-analysis=kernel-resource-usage"
Q=src/models/qwen/hip/kernels
build() { # object source extra-flags...
  o=$1; src=$2; shift 2
  $P "$@" -c $src -o $obj/$o.o 2> $obj/$o.log || { tail -40 $obj/$o.log; exit 1; }
}
build main tools/bench/rq_type_bench.hip
build gemv $Q/gemv.hip
build gemv_quant $Q/gemv_quant.hip
build qkv $Q/qkv.hip
build ssm $Q/ssm.hip
build swiglu $Q/swiglu.hip
build prefill_quant_gemm $Q/prefill_quant_gemm.hip -fno-slp-vectorize
build prefill_fp16 $Q/prefill_fp16.hip
build prefill_quant_wave64 $Q/prefill_quant_wave64.hip -mwavefrontsize64 -fno-slp-vectorize
build small_batch_wave64 $Q/small_batch_wave64.hip -mwavefrontsize64
build small_batch_quant16_wave64 $Q/small_batch_quant16_wave64.hip -mwavefrontsize64 -gline-tables-only -Xarch_device -mllvm=-amdgpu-sched-strategy=iterative-ilp
hipcc -O2 -g -DNDEBUG -std=c++20 -I. -c src/core/quant/ggml_dequant.cpp -o $obj/ggml_dequant.o
M=src/models/deepseek_v4_flash/kernels/rocm/mmq
V="hipcc -x hip -O3 -ffast-math -fno-finite-math-only -std=c++17 --offload-arch=gfx1151 -DGGML_USE_HIP -DGGML_HIP_NO_VMM -DDS4_ROCM_WMMA_MMQ_Y=64 -Wno-unused-value -I$M"
for u in mmvq quantize ds4_ggml_stubs; do
  $V -c $M/$u.hip.cpp -o $obj/v_$u.o 2> $obj/v_$u.log || { tail -40 $obj/v_$u.log; exit 1; }
done
$V -DRQ_MMVQ_IMPLEMENTATION -c tools/bench/rq_type_bench_mmvq.hpp -o $obj/rq_mmvq.o 2> $obj/rq_mmvq.log || { tail -40 $obj/rq_mmvq.log; exit 1; }
hipcc --offload-arch=gfx1151 $obj/*.o -lhipblas -o $out/rq_type_bench
hipcc -O3 --offload-arch=gfx1151 -std=c++20 tools/bench/gfx1151_peak.hip -o $out/gfx1151_peak 2> $obj/peak.log || { tail -20 $obj/peak.log; exit 1; }
cat $obj/*.log | grep -E "Function Name|VGPRs:|Occupancy|Spill|LDS Size|ScratchSize" | sed 's/.*remark: //; s/ \[-Rpass.*//' > $out/res.txt || true
echo built
```

## 남은 문제

- 측정은 창 하나(`rq0b-a03`) 한 번이다. 반복 창은 돌리지 않았다. 라운드 사이 퍼짐은 p95로만 보인다(대부분 중앙값의 +2–10 %, 작은 행렬은 +50 %까지).
- 운영 경로 중 재지 않은 것: verify의 packed gate/up SwiGLU(`TryLaunchPackedQuantSwiGLUFp32`, gate·up이 메모리에서 붙어 있을 때), prefill의 gate/up 융합(`LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1`, `TryLaunchBatchedDualQuantGEMMSwiGLUFp16`), prefill 2048의 residual 융합, 여러 세션 verify(9–64행). verify·prefill 토큰 추정은 행렬별 launch 합이다.
- Q3_K·IQ3_S·IQ2_*의 연산 병목 원인은 짐작이다. ISA 구성(`tools/prof/isa_mix.py`)이나 프로파일로 확인하지 않았다.
- 2비트 prefill 커널이 없어 prefill 칸이 비었다.
- FWHT는 FP32 제자리 독립 커널이다. 운영 prefill 입력(BF16)과 융합 비용은 R2에서 잰다.
- 작업 중 규칙 위반 1건: 포맷 검사(nixos/nix 컨테이너에서 nix 패키지 내려받기 + clang-format)를 다른 세션의 `gufo-port-bfe3783c15` 창이 도는 동안 실행했다(`docker ps` 출력에 그 컨테이너가 있었는데 멈추지 않았다). 빌드(hipcc)는 창이 없을 때만 했다. 그 창의 측정이 오염되었을 수 있으니 감독이 확인해야 한다.

## 감독 확인과 기록 (2026-10-28)

- 감독이 창 `rq0b-a03`의 `WINDOW.json`(COMPLETE, health_PASS), `check.log`(`CHECK PASS`), `peak.log`(DRAM read 240.07 GB/s), `time.log` RESULT 줄 일부(`ffn_down` AR: Q4_K 260.5 µs, IQ4_XS 217.8 µs, Q3_K 295.0 µs)를 원본과 대조했다.
- **커밋 경계:** `d19126df`에는 이 단계의 벤치 파일 외에 R1a 컨버터 파일 9개도 들어 있다(두 에이전트가 같은 트리에서 거의 동시에 커밋). 이력은 고치지 않았다(사용자 결정 2026-10-28).
- **규칙 위반 1건의 영향 범위:** 겹친 다른 세션 창은 `gufo-upstream-port-20261023/um2-a01/um2b-long32k`이다. 그 세션에 알려 타이밍 재확인 여부를 정하게 한다.
- 계획 반영: [계획](PLAN.md)의 "R0b 결과가 계획에 주는 것", E3 전제 수정, E9(3비트 커널 최적화) 추가, E7b 범위.
