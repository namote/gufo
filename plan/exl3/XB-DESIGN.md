# X2b 설계 메모: EXL3 trellis decode GEMV·verify 커널 (W3, 2026-10-30)

상위 문서: [XB-BRIEF.md](XB-BRIEF.md) X2b, [PLAN.md](PLAN.md) X2. 근거: [XA-RESULTS.md](XA-RESULTS.md)(X0·X1, X2a 설계 메모 2절), [EXL3-RESULTS.md](../rotated-quant/EXL3-RESULTS.md)(속도, 실행 커널), R2 경로 표([R2-RESULTS.md](../rotated-quant/R2-RESULTS.md) "경로 표"). 읽기만 하고 쓴 설계다. 코드, 빌드, GPU 창은 아직 없다.

upstream은 exllamav3 v1.6.5 `cdc785db`(MIT, Copyright 2025 Turboderp)이고, 클론은 `build/w9-src/exllamav3/exllamav3/exllamav3_ext/rocm/quant/`다. 아래 `upstream:파일:줄`은 그 트리를 가리킨다.

## 1. 요약

- **코어:** upstream RDNA "tiles" 코어와 같은 lane 배치를 쓴다. 한 wave가 16 × 16 타일을 레지스터에서 바로 풀고(LDS 없음, 배리어 없음), FP16 활성화와 `v_dot2_f32_f16`으로 FP32에 누적한다. WMMA는 m ≥ 16(X3, 배치 verify)에만 쓴다. 근거는 4절의 VALU 예산이다.
- **형태:** m = 1(행 타일 1), m = 2–8(행 타일 M ∈ {2, 4, 8}, 가중치를 한 번 읽어 M행에 씀), 여러 행렬 한 launch(같은 입력을 쓰는 gate/up, qkv/z, q/k/v), lm_head(K = 6, n = 248,320). split-K는 블록 안 warp들이 나눈다.
- **입력 Hadamard:** 활성화를 만드는 커널(RMSNorm, SSM post-norm gate, SwiGLU)이 **소비 텐서마다 따로** `H(suh_c ⊙ x)`를 FP16 슬랩으로 쓴다. 텐서마다 suh가 다르므로 공유하지 않고 복사본을 만든다. 쓰는 양은 행당 수십 KB로, 가중치 읽기(수십–수백 MB)에 비해 무시할 수 있다.
- **출력 Hadamard:** 128열 구간이 다 끝나면 마지막에 도착한 warp가 그 구간을 `svh ⊙ H(acc)`로 회전한다(upstream의 자기 초기화 카운터 방식). 같은 epilogue에 residual 더하기, gate·up → SwiGLU → down 입력 회전을 융합한다.
- **목표 대역폭:** 5.00 팩은 토큰당 16.2 GB다. upstream tg128 12.37 tok/s를 넘으려면 decode 전체 실효 대역폭이 약 201 GB/s 이상이어야 한다. 그래서 GEMV 커널 단위 목표를 210 GB/s 이상(DRAM read 240 GB/s의 약 88 %)으로 둔다.
- **X-A 인터페이스:** X2a 메모 2절의 `quant::exl3::DeviceLinear`(파일 배치, 256 B 정렬, FP16 suh/svh)를 그대로 받는다. 덧붙이는 요청 셋은 8절에 있다.

## 2. 27B decode의 EXL3 투영과 입력 공유

`qwen35` GGUF 헤더에서 읽은 모양이다(hidden 5120, FFN 17408, SSM inner 6144, q 헤드 24 × 256, kv 헤드 4 × 256). 5.00 팩 본문은 K = 5, lm_head는 K = 6이다.

| 입력(생성 지점) | 소비 텐서 [k → n] | 층 | suh 복사본 | 토큰당 바이트(5.00) |
|---|---|---|---:|---:|
| attn_norm(DeltaNet 층) | in_proj_qkv [5120 → 10240], in_proj_z [5120 → 6144]; in_proj_a/b(비양자화, 회전 없음) | 48 | 2 | 48 × 52.4 MB |
| attn_norm(full attention 층) | q [5120 → 12288], k [5120 → 1024], v [5120 → 1024] | 16 | 3 | 16 × 45.9 MB |
| SSM post-norm gate(헤드 128 = Hadamard 구간 1개) | out_proj [6144 → 5120] | 48 | 1 | 48 × 19.7 MB |
| attention 출력 | o_proj [6144 → 5120] | 16 | 1 | 16 × 19.7 MB |
| post_attention_norm | gate, up [5120 → 17408] | 64 | 2 | 64 × 111.4 MB |
| SwiGLU(gate·up epilogue) | down [17408 → 5120] | 64 | 1 | 64 × 55.7 MB |
| output_norm | lm_head [5120 → 248320], K = 6 | 1 | 1 | 953.5 MB |
| 합계 | | | | 약 16.2 GB(W8 계산 16.22 GB와 같음) |

- 복사본은 행당 k × 2 B다. 층당 최대 3 × 10 KB, 행 8개면 240 KB다. L2(2 MB)와 MALL에 머무는 크기다.
- in_proj_a/b는 X-A가 `ssm_alpha/beta` BF16으로 바꿔 지금 gufo 경로(평범한 정규화 x)를 탄다. 그래서 attn_norm은 FP32 정규화 x도 계속 낸다.

## 3. 커널 구조

### 3.1 타일 코어 (모든 형태가 같이 씀)

- **lane 배치:** upstream `exl3_gemv_kernel_rdna.cuh:215-240`을 그대로 따른다.
  - lane L은 타일의 위치 8L … 8L + 7을 푼다. 그러면 `B[r0][cA], B[r0+1][cA], B[r0+8][cA], B[r0+9][cA]`와 `cB = cA + 8` 열의 같은 네 행을 갖는다. 여기서 `r0 = (L % 4) · 2`, `cA = (L / 8) · 2 + ((L >> 2) & 1)`이다.
  - 열 c는 lane 넷(4c … 4c + 3)이 나눠 갖는다. lane마다 A 행 (r0, r0+1)과 (r0+8, r0+9)에 대해 `v_dot2_f32_f16` 넷을 한다. quad 감소(`__shfl_xor` 두 번)는 k 구간이 끝날 때 **한 번만** 한다.
- **B 읽기:** 타일 하나는 32 · K B 연속이다. warp가 모든 바이트를 한 번씩 만지고, 겹치는 lane 읽기는 L0이 받는다. 주소는 wave에 균일하다(SGPR base + lane 고정 offset). upstream `exl3_gemv_tiles_rdna.cuh:36-44`와 같다.
- **디코드 산술** (upstream tiles 코어, 모두 정확한 정수 연산이라 X0 CPU 참조와 FP16 비트가 같다):
  - 해시 곱셈 x · C mod 2³²는 `v_pk_mul_lo_u16` + `v_mad_u32_u16` 두 개(full rate)로 한다. `v_mul_lo_u32`(quarter rate)는 쓰지 않는다.
  - mul1 바이트 합은 `v_sad_u8` + `v_sad_hi_u8`로 한다. 짝 두 값이 바로 `[0x6400 + s0 | 0x6400 + s1]`로 묶여 나오고, `v_pk_fma_f16` 하나로 값이 된다.
  - 상태 창은 `v_alignbit_b32` 하나로 뽑는다.
- **K별 디코더:** upstream은 K = 1, 2, 4만 손으로 짠 정렬 디코더이고, 나머지(3, 5–8)는 일반 `dq8` 비트 산술이다. 우리 목표는 본문 **K = 5와 head K = 6**이다. 그래서 이 둘의 정렬 디코더를 새로 짠다.
  - 근거: K = 5에서 lane의 8위치는 40비트이고, 앞 16비트 창까지 56비트가 필요하다. 정렬된 dword 둘–셋에서 `alignbit`로 뽑을 수 있다.
  - K = 3, 4와 반정수 K는 일반 경로로 시작한다. 4.00 팩(K = 4)과 XQ 팩 비교 때 필요하면 정렬 디코더를 더한다.
- **N 타일 수:** wave당 N 타일 2개(upstream `EXL3_GEMV_TILES_TMAX 2`). 이웃한 타일을 같이 읽어 A 읽기를 나누고 B를 64 · K B씩 연속으로 읽는다. 블록 수가 모자라면 1개로 한다.

### 3.2 형태

| 형태 | 쓰는 곳 | 구조 |
|---|---|---|
| **m = 1, 행렬 하나** | out_proj, o_proj, down(SwiGLU 융합 형태는 아래) | 블록 = (N 타일 2개, k 전체). 블록 안 W개 warp가 k16 구간을 나눈다(split-K, W ∈ {4, 8, 16}, upstream `exl3_gemv_splitk_warps` 규칙에서 시작). warp 결과를 LDS 16 × W float로 감소하고 출력 epilogue로 간다 |
| **m = 1, 여러 행렬** | qkv + z, q + k + v, gate + up | 한 launch, `blockIdx.y` = 행렬 슬롯. 슬롯마다 자기 A 슬랩, trellis, svh를 쓴다. 인자는 `DeviceLinear` 값 배열(커널 인자, 그래프 캡처 안전) |
| **m = 2–8** | DFlash2 verify(최대 1 + 7행), MTP | 행 타일 M ∈ {2, 4, 8} 중 m 이상인 가장 작은 것. 타일을 한 번 풀어 M행 × fdot2 4개를 누적한다. 행 r의 누적 순서가 m = 1과 같으므로, **행 r 결과는 m = 1 호출과 비트 단위로 같다**(upstream multirow의 성질과 같고, 테스트로 고정한다). m을 넘는 행은 읽지 않는다 |
| **lm_head** | output | n = 248,320, N 타일 15,520개라 블록 수가 충분하다. split-K를 쓰지 않는다(upstream도 2,048 타일이 넘으면 single-warp form). epilogue는 FP32 logits를 쓴다. DFlash2 드래프터도 같은 head를 쓴다(5절) |
| **m > 8** | 다중 세션 decode·verify(C × 1–8행, 최대 64) | X2 동안: 8행 묶음으로 반복 launch. 가중치를 묶음마다 다시 읽는다(C배). X3의 WMMA 경로가 생기면 그쪽으로 넘긴다. **결정 필요(10절)** |

- **출력 형식:** FP32다(gufo hidden이 FP32). upstream 기본은 FP16 C다.
- **그래프 캡처:** 모든 launch 모양은 모델과 m에 따라 고정이다. 카운터와 슬랩은 로드 때 arena에 잡는다. 캡처 중에는 할당하지 않는다.

## 4. WMMA인가 fdot2인가: VALU 예산 [짐작, isa_mix로 확인할 것]

- **타일당 wave VALU (K = 5, m = 1):** lane당 위치 8개, 즉 짝 4개다. 짝마다 해시 2 × 2, 바이트 합 2, `pk_fma` 1, 창 뽑기 약 2, `dot2` 2이고, 주소 계산은 없다. 그러면 lane당 약 44 VALU이고, wave32는 SIMD에서 한 사이클에 하나씩 낸다.
- **토큰 전체:** 타일 수는 본문 약 9,500만 개 + head 500만 개다. 약 1억 × 44사이클을 SIMD 80개(CU 40 × 2)에 나누면 SIMD당 5,500만 사이클, 약 2.9 GHz에서 약 19 ms다. 메모리 시간은 16.2 GB ÷ 210 GB/s = 77 ms다. **VALU는 메모리 시간의 약 25 %다.**
- **m = 8:** 행마다 `dot2` 4 × 2를 더하면 타일당 약 100 VALU, 약 43 ms다. 아직 메모리에 묶이지만 여유가 줄어든다. latency 숨기기와 dual-issue 여부에 따라 달라지므로 측정으로 정한다.
- **WMMA를 m ≤ 8에 쓰지 않는 이유:**
  - RDNA3 WMMA의 B 조각은 lane마다 열 하나의 16개 k 값을 들고, lane 16–31이 같은 값을 반복해서 든다. trellis를 그 배치로 풀면 디코드 일이 두 배가 된다.
  - A 행을 16으로 채우면 m = 1–8에서 계산이 2–16배 낭비된다.
  - fdot2 형태가 메모리에 묶여 있는 동안에는 WMMA가 이득이 없다. m ≥ 16(배치 verify, prefill)은 X3에서 WMMA로 간다.
- **결론:** decode와 verify(m ≤ 8)는 fdot2를 쓴다. 측정에서 m = 8이 VALU에 묶이면(대역폭이 m = 1보다 5 % 넘게 떨어지면) 그때 WMMA 변형을 다시 본다.

## 5. 입력 Hadamard 융합

### 5.1 산술 계약 (R2 FWHT 계약을 일반화)

- 입력: 생성 커널이 이미 가진 FP32 값 `x`(정규화 값, SwiGLU 값 등)다.
- 단계:
  1. `x ⊙ suh_c`: FP16 suh를 FP32로 바꿔 곱한다(반올림 한 번).
  2. FWHT: 길이 1, 2, …, 64 순서의 butterfly이고 쌍마다 add/sub 한 번이다. R2 `RotateSegment*` 그대로다.
  3. `kHadamard128Scale`을 한 번 곱한다. 그 뒤에 `asm volatile` 레지스터 배리어를 둔다(R2에서 scale과 FP16 변환이 `v_fma_mix`로 합쳐지는 것을 막았다).
  4. FP16 RNE로 바꿔 슬랩에 쓴다.
- R2와 다른 점은 1단계 하나다. R2는 부호를 XOR로 정확히 적용했고, EXL3는 부호 × 스케일이라 곱셈이다. `hadamard.hpp`에 앞 스케일 functor(부호 XOR | suh 곱)를 템플릿 인자로 더한다. R2 인스턴스의 산술은 바뀌지 않는다.
- upstream `had_hf_r_128`도 FP32 butterfly(XOR 부호 + add)다. 차이는 입력이 FP16이냐 FP32냐뿐이다. gufo는 FP32 값에서 바로 회전하므로 반올림이 한 번 적다.

### 5.2 생성 지점 (R2 경로 표 재사용)

| 생성 지점 | R2의 회전 변형 | EXL3 변형 |
|---|---|---|
| attn_norm, decode `LaunchRMSNorm` / verify `LaunchBatchedRMSNorm` | `LaunchRMSNormRotated`(LDS 행, wave당 구간) | `…Exl3`: 같은 LDS 행에서 소비자 c마다 회전해 슬랩 c에 FP16으로 쓴다(DeltaNet 2개, full attention 3개). FP32 정규화 x는 alpha/beta용으로 계속 쓴다 |
| post_attention_norm | `LaunchRMSNormRotated` | 위와 같고 소비자 2개(gate, up) |
| SSM post-norm gate(헤드 = 128) | `BatchedSSMPostNormGate…<Rotate>`(prefill), decode는 단독 | 헤드 하나가 구간 하나다. gate 곱 뒤 `suh_out ⊙` → FWHT → FP16. decode와 verify 모두 융합한다 |
| attention 출력(헤드 256 = 구간 2개) | 단독 `LaunchRotateRowsFp32` | 단독 회전 + FP16(16층, 작음). 나중에 split-K combine 커널에 융합할 수 있다 |
| SwiGLU | `LaunchFusedSwiGLUGEMV`(GEMV 안 SwiGLU) | gate·up 출력 epilogue에 융합한다(6.2) |
| output_norm | `LaunchQwenOutputHead`(R-norm + GEMV) | 같은 구조. suh_head로 회전 → FP16 → lm_head GEMV |
| DFlash2 드래프터 head 입력 | 단독(`output_rotation`) | selector 투영 뒤 suh_head 회전 + FP16(행 ≤ 8) |

### 5.3 다른 방법을 고르지 않은 이유

- **GEMV 프롤로그에서 블록마다 회전:** 블록 수(N 타일 / 2)만큼 같은 회전을 되풀이한다. upstream도 m = 1 융합 커널의 LDS 프롤로그가 occupancy를 깎아 별도 경로로 돌렸다(`exl3_gemv_multirow_rdna.cu` 상단 주석, "the fused kernels' LDS prologue costs occupancy").
- **lane 안 회전:** 코어 lane은 A의 4행(r0, r0+1, r0+8, r0+9)만 읽는다. 128점 회전에 필요한 행이 lane 안에 없다.
- **한 번 회전 후 suh 비율 곱하기:** `x ⊙ suh_c = (x ⊙ suh_0) ⊙ (suh_c / suh_0)`이지만 H 뒤에서는 대각 행렬이 교환되지 않는다. 쓸 수 없다.
- **비용:** 소비자 c마다 회전 1번(FP32 butterfly 7단, 행 5120이면 구간 40개)과 FP16 쓰기 10 KB다. 층당 μs 이하이고, launch 수는 늘지 않는다(같은 커널 안).

## 6. 출력 svh epilogue

### 6.1 기본형

- N 타일 16열은 128열 구간의 1/8이다. 구간 카운터 하나에 8타일(m행이면 행마다 같은 카운터)이 도착한다.
- 마지막에 도착한 warp가 구간의 FP32 결과를 다시 읽는다(방금 쓴 L2). 그다음 FWHT → scale → `⊙ svh` → FP32를 쓰고 카운터를 0으로 되돌린다.
- 이것은 upstream `exl3_gemv_fused_epilogue`(`exl3_gemv_kernel_rdna.cuh:771-800`)와 같은 방식이다. fence는 wave에 균일하게 두고, 카운터는 자기 초기화라 그래프 재생에 안전하다.
- **카운터:** launch 지점마다 고유 영역을 쓴다(lm_head 1,940구간, 나머지 ≤ 136구간 × 슬롯). 로드 때 arena에 0으로 잡는다.
- **더 단순한 대안:** 블록 하나가 구간 하나(8 N 타일)를 끝까지 맡으면 카운터가 없다. 하지만 n = 1024(k/v)는 블록이 8개뿐이라 split-K로도 wave가 모자란다. 그래서 기본은 카운터 방식으로 하고, 블록 수가 충분한 모양(lm_head)은 측정으로 비교한다.

### 6.2 융합 변형

| 변형 | 소비 | 동작 |
|---|---|---|
| **store** | qkv, z, q, k, v | 회전 + svh 뒤 FP32로 쓴다. 다음 커널(conv1d, q/k norm·RoPE·KV write)은 그대로 둔다 |
| **residual** | out_proj, o_proj, down | 회전 + svh 뒤 `hidden += y`. 지금 따로 있는 residual add launch를 없앤다. 다음 RMSNorm은 행 전체 합이 필요하므로 따로 둔다 |
| **swiglu→down** | gate + up(슬롯 2개, 같은 구간 카운터) | 구간 s의 gate·up이 모두 끝나면 마지막 warp가: gate 회전 + svh_g, up 회전 + svh_u, `silu(g) · u`(FP32, 지금 SwiGLU 산술 그대로), `⊙ suh_down` → FWHT → scale → FP16을 down 슬랩 구간 s에 쓴다. intermediate 17408 = 128 × 136이라 구간이 정확히 맞는다. upstream MoE의 "activation folded into the down projection's input rotation"과 같은 생각이다 |
| **logits** | lm_head | 회전 + svh → FP32 logits. 샘플링은 그대로 둔다 |

## 7. 기대 대역폭

| 항목 | 값 |
|---|---|
| DRAM read 피크(R0b, `gfx1151_peak`) | 약 240 GB/s |
| 5.00 팩 토큰당 가중치 바이트 | 16.22 GB(본문 15.27 + head 0.95) |
| upstream ExLlamaV3 5.00 tg128 d0 | 12.37 tok/s, 실효 201 GB/s |
| gufo UD-Q4_K_XL tg128 d0 | 12.32 tok/s, 실효 203 GB/s(GEMV 밖 시간 포함) |
| gufo 지금 4비트 GEMV(R0b) | IQ4_XS 89 %, Q4_K 82 % |

- **필요 대역폭:** 12.37 tok/s를 넘으려면 decode 전체가 16.22 GB ÷ 80.8 ms = 201 GB/s 이상이어야 한다.
- gufo의 GEMV 밖 시간(DeltaNet 재귀, attention, norm, 샘플링)은 UD-Q4_K_XL과 같다고 본다. 그러면 **GEMV 커널이 약 210 GB/s(피크의 88 %)이면 약 12.6 tok/s**가 된다[짐작].
- 브리프의 80 %(190 GB/s)는 최소선이다. 190 GB/s면 약 11.9 tok/s라 upstream보다 느리다. **실제 목표는 210 GB/s다.**
- **4.00 팩:** 13.18 GB이고, 210 GB/s면 약 15.6 tok/s다(upstream 15.56).
- **DFlash2:** 실효 tok/s 29.16 이상이 기준이다. AR이 12.6이고 verify(m = 8)가 메모리에 묶여 m = 1과 비슷하면, UD-Q4_K_XL의 배율(2.36×)로 약 29.7이 된다. 여유가 작다. verify m = 8의 대역폭 유지가 관건이다[짐작].
- **측정:**
  - 커널 단위: `tools/bench` 독립 HIP 실험. 모양 6개(5120 → 10240/6144/12288/1024/17408, 6144 → 5120, 17408 → 5120, 5120 → 248320 K6) × m ∈ {1, 2, 4, 8}.
  - 파이프라인: `tools/prof/prof.py --passes`, ISA는 `tools/prof/isa_mix.py`.

## 8. X-A와 맞출 인터페이스

X2a 메모 2절(`src/core/quant/exl3/device.hpp`의 `DeviceLinear`, 파일 배치, 256 B 정렬 arena, `QwenTensorRef::exl3`)을 그대로 받는다. 덧붙이는 요청:

1. **수치 계약 한 줄 수정:** 메모의 "활성화 x는 FP32"는 기준 경로(X2a)에 맞다. X2b 빠른 커널은 **회전된 활성화를 FP16으로 반올림**해 fdot2로 곱한다(upstream과 같다). 커널 단위 기준은 다음 둘이다.
   - X0 `GemvReference` 대비 상대 오차가 FP16 수준(약 1e-3)이다.
   - FP16 반올림 입력을 넣은 FP64 참조와는 1e-5 수준이다.
   - FP32 활성화 변형(가중치를 FP32로 풀어 FMA)은 VALU가 약 2배라, 품질 문제가 보일 때만 만든다.
2. **trellis는 파일 배치 그대로 둔다(재배치 없음).** [k/16][n/16][16K]이고, 커널은 타일 (kt, nt)를 `trellis + (kt · n_tiles + nt) · TileWords`로 찾는다. X0 덤프·upstream과 비트 대조가 그대로 되기 때문이다. N-major 재배치(warp의 k 구간이 연속이 됨)는 측정 뒤 이득이 있으면 다시 제안한다. 그때를 위해 `DeviceLinear`에 배치 표시(enum) 하나만 예약해 두기를 제안한다.
3. **arena 공동 영역:** X2b가 쓸 스크래치를 로더가 같이 잡는다. 크기는 X2b가 함수로 준다.
   - FP16 입력 슬랩: 소비자별, 최대 행 8 × k
   - 출력 카운터: launch 지점별
   - gate·up 융합용 down 슬랩
   - 위치는 `QwenGpuArena`이고 공유 파일이다. 그래서 감독에게 순서를 받는다.
4. **바인딩 이름:** 슬롯 묶음(qkv + z, q + k + v, gate + up)은 `QwenLayerWeights`의 기존 텐서 쌍에서 고른다. 로더가 따로 묶지 않아도 된다(인접 배치 불필요, 포인터 표로 처리).

분기 지점 다섯 곳(decode GEMV, small-batch verify, prefill route, output head, DFlash2 head)은 X2a가 기준 경로로 먼저 연결하고, X2b가 같은 분기 안에 빠른 커널을 넣는 순서를 제안한다. 그러면 X2b의 각 커널을 X2a 경로와 바로 A/B할 수 있다.

## 9. upstream과 비교

| 항목 | upstream 1.6.5 RDNA | X2b 계획 | 이유 |
|---|---|---|---|
| 타일 코어 | tiles 코어: 레지스터 디코드, fdot2, quad 감소 | 같다 | 이미 대역폭 한계(201 GB/s)를 낸다 |
| K = 5, 6 디코드 | 일반 `dq8` 산술 | 정렬 디코더를 새로 짠다 | 5.00 팩 본문과 head가 이 K다 |
| 입력 회전 | 별도 `had_in` launch(행렬마다), A_had는 전역(L2) | 생성 커널 안에서 소비자마다 회전 | launch 수와 FP16 왕복을 줄인다. 생성 값이 FP32라 반올림이 한 번 적다 |
| 출력 회전 | 마지막 도착 warp 융합, FP16 C 기본 | 같은 방식, FP32 C, residual·SwiGLU→down 융합 | gufo hidden이 FP32다. launch를 줄인다 |
| m = 2–8 | 행 타일 {2, 4, 8}, 행별 비트 동일 | 같다 | DFlash2 verify 최대 8행 |
| split-K warp 수 | 4/8/16 규칙 | 같은 규칙에서 시작해 측정으로 고친다 | |
| 런타임 선택 | 환경 변수 여러 개 | 없다(성공한 최적화가 기본) | gufo 규칙(옵션 별칭 금지, 기본 경로 하나) |
| 그래프 | 패치 지점 + 파라미터 블록 | gufo 스트림 캡처, 포인터는 로드 때 고정 | 패치가 필요 없다 |

- MIT 고지: 가져온 코드에는 파일 머리에 upstream 출처와 Turboderp 저작권 고지를 둔다. vcruz305(`v_sad_u8` 디코드, LDS pad)와 Kyojin은 비교 자료로만 쓴다. upstream tiles 코어에 이미 `sad_u8`이 들어가 있다.

## 10. 시험과 단계

- **CPU:** 없다(X0 참조와 fixture를 재사용).
- **GPU 단위(`tests/models/qwen27b/exl3_kernel_test.hip`):**
  - 무작위 trellis(모든 16비트 상태가 유효)로 K 1–8, 반정수 K, 코드북 셋을 돈다. 디코드 비트는 X0와 같아야 한다.
  - GEMV m = 1–8 대 X0 FP64 참조(위 두 기준), 행 r 결과 = m = 1 결과(비트).
  - epilogue 네 변형, 여러 행렬 슬롯, 그래프 캡처 재생 2회 일치.
- **모델(GPU 창, 허락 뒤):**
  - X2a 기준 경로와 R0 하네스 KL을 비교한다(엔진 차이 이하).
  - 속도: main과 짝지은 요청·단계별 5 %·10 ms. tg128 d0과 d32K, DFlash2 수락률은 `accept.py`로 본다.
- **순서(제안):**
  1. 코어 + m = 1 + epilogue(store), K = 5. 독립 HIP 실험으로 대역폭을 먼저 잰다.
  2. 생성 커널 회전(RMSNorm, SSM gate) + residual 변형
  3. gate·up swiglu→down
  4. m = 2–8, lm_head K = 6, DFlash2 head
  5. 모델 연결(X2a 분기 안)과 창
- **빌드와 창은 각각 감독 허락 뒤에 한다.**

## 11. 결정이 필요한 것

1. **m > 8(다중 세션 decode·verify):** X2 동안 (a) 8행 묶음 반복(가중치를 C배 읽음)과 (b) X2a 기준 경로(dequant + GEMM) 중 어느 쪽인가. 나는 (a)를 제안한다. 운영 기본은 C = 1이고, X3 WMMA가 생기면 바뀐다.
2. **활성화 FP16 반올림(8절 1):** upstream과 같은 산술이다. 커널 기준을 위 두 가지로 해도 되는지.
3. **arena 공동 영역과 분기 지점의 순서(8절 3, 끝 문단):** X2a가 기준 경로를 먼저 연결하는 순서에 동의하는지.
