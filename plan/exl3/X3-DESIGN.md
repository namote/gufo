# X3 설계 메모: EXL3 trellis 디코드 + WMMA prefill GEMM (W3, 2026-10-31)

상위 문서: [PLAN.md](PLAN.md) X3(사용자 2026-10-31: prefill도 속도 목표), [XB-DESIGN.md](XB-DESIGN.md), [XB-RESULTS.md](XB-RESULTS.md). 읽기만 하고 쓴 설계다. 코드, 빌드, GPU 창은 아직 없다.

## 1. 목표와 기준

- **통과 기준(PLAN X3):** 같은 운영 바이너리의 UD-Q4_K_XL과 ABBA로 짝지어 비교한다. pp512, pp2048, pp8192와 깊이 32K prefill이 모두 요청·단계별 5 %·10 ms 안이거나 더 빠르다.
- **기준값:** XB6 창에서 같은 바이너리로 다시 잰다. 지금 알고 있는 값은 아래와 같다.

| 경로 | pp512 | pp2048 | pp512 @ d32K | pp2048 @ d32K | 출처 |
|---|---:|---:|---:|---:|---|
| gufo UD-Q4_K_M | 490–510 | 650–682 | 378–382 | 497–500 | `rq2f`(cand2) |
| gufo UD-Q4_K_XL | — | 639.8 | — | — | R0 속도 |
| ExLlamaV3 EXL3 4.00/5.00 | — | 411–434 | 잴 수 없음(LDS 초과) | 잴 수 없음 | W8 |
| gufo EXL3(X2a 기준 경로) | XB6에서 잰다 | | | | |

- 깊이 32K에서 줄어드는 몫(약 −25 %)은 attention(kvarn KV)이다. 형식과 상관이 없다. 그래서 **선형층이 d0에서 같으면 d32K에서도 같다.** X3는 선형층 GEMM만 바꾼다.

## 2. 한계: 계산과 메모리

- **계산:**
  - 토큰당 선형층 FLOP은 2 × 24.3e9 ≈ 48.7 GFLOP이다. lm_head는 prefill에서 마지막 토큰만 계산한다.
  - 같은 창에서 잰 WMMA FP16 피크는 55.5 TFLOPS다(`gfx1151_peak`; int8 57.2 TOPS로 사실상 같다). 선형층만 보면 상한은 약 1,140 tok/s다.
  - 지금 gufo pp2048 약 650 tok/s는 선형층 기준 약 32 TFLOPS, 피크의 약 57 %다. 나머지는 attention, norm, SSM과 파이프라인 비효율이다.
- **메모리:** prefill GEMM은 가중치를 M 블록(BM 행)마다 다시 읽는다.
  - 5.00 팩(16.2 GB)을 BM = 256으로 하면, 2048행은 8번(130 GB), 0.54 s다. 같은 2048토큰의 계산은 약 3.15 s다. **prefill은 계산에 묶인다.**
  - UD-Q4_K_XL(16.5 GB)도 같은 구조이고 바이트 수도 거의 같다. 메모리 쪽 차이는 없다.
- **EXL3만의 추가 비용은 디코드 VALU다.**
  - trellis 타일(256개 가중치) 하나를 wave가 푸는 데 lane당 약 44 VALU가 든다(X2b 코어, K = 5).
  - BM = 256 블록에서 B 타일 하나는 WMMA 16회(행 타일 16개)에 쓰인다. 16 × 약 34.5사이클 = 552사이클이다. 디코드는 그 약 8 %다[짐작, VALU와 WMMA가 겹치지 않는다고 가정].
  - GGUF 경로도 K-quant 역양자화 VALU를 같은 자리에서 쓴다(Q4_K/Q6_K: 가중치 8개에 약 20 VALU로 짐작). 그래서 **UD-Q4_K_XL 대비 차이는 약 3–5 %로 본다**[짐작].
  - 기준(5 %)의 경계다. 그래서 디코드를 줄이는 설계(5절)를 처음부터 넣는다.

## 3. 커널 구조

### 3.1 틀: gufo FP16 prefill GEMM을 재사용한다

- `src/models/qwen/hip/kernels/prefill_fp16.hip`의 FP16 WMMA GEMM(`BM = BN = 256`, `WMMA f32_16x16x16_f16`, FP32 누적, K16 순서)을 바탕으로 한다. 이 경로가 지금 gufo pp2048 약 650 tok/s를 낸다.
  - 바뀌는 것은 **B 생산자 하나**다. K-quant 블록 역양자화 대신 trellis 디코드를 쓴다.
  - A(FP16 활성화) 적재, LDS 배치, WMMA 루프, FP32 누적은 그대로 둔다.
- **trellis 타일과 WMMA B 조각:**
  - trellis 타일은 16(k) × 16(n)이고, WMMA B 피연산자도 16 × 16이다. 타일 하나가 B 조각 하나다.
  - X2b 디코드는 lane이 위치 8개(열 2개 × 행 4개)를 낸다. RDNA WMMA B는 lane이 열 하나의 k 값 16개를 들어야 한다(lane 16–31은 같은 값). 그래서 디코드 값을 LDS B 타일(16 × 16 FP16, 512 B)에 썼다가 B 조각으로 읽는다.
  - upstream도 같은 이유로 B를 LDS로 거친다(`exl3_gemm_inner_rdna.cuh` 머리 주석 2번).
- **타일 재사용:**
  - 블록은 K 단계마다 B 타일들(BK × BN)을 한 번 풀어 LDS에 둔다. 블록 안 모든 행 wave가 그것을 쓴다(BM / 16 = 16 행 타일).
  - M 블록이 다르면 다시 푼다. 청크가 4096행이면 16번이다. GGUF 경로가 역양자화를 M 블록마다 다시 하는 것과 같다.
  - 디코드 비용을 더 줄이는 후보는 5절에 있다.
- **분할 없음:** prefill은 블록이 BN = 256열(Hadamard 구간 2개)의 전체 K를 맡는다(split-K 없음). 출력 회전에 카운터가 필요 없다(3.3절).

### 3.2 입력 회전 (prefill 행)

- R2의 FP16 prefill 생성 커널을 일반화한다. 바꾸는 것은 두 가지다.
  - 부호 XOR → `suh` 곱(곱 반올림 고정, X2b 계약)
  - 출력 하나 → 소비자마다 하나(최대 3개)

| 생성 지점 | R2 커널 | EXL3 소비자 |
|---|---|---|
| attn_norm | `LaunchBatchedRMSNormRotatedFp16`(HalfNorm5120) | q/k/v 또는 qkv/z(+ alpha/beta는 FP32 그대로) |
| ffn_norm | 같은 커널(잔차 융합형) | gate/up |
| SSM post-norm gate | `BatchedSSMPostNormGateFp16Kernel<Rotate>`(헤드 128 = 구간) | out_proj |
| attention 출력 | 단독 회전 + `LaunchFloatToFp16` | o_proj |
| SwiGLU | `LaunchBatchedSwiGLURotateFp16` | down(아니면 3.3절의 epilogue 융합) |

- **슬랩:** 행 M × k FP16을 소비자마다 둔다. 청크 4096행이면 q/k/v 세 벌은 3 × 4096 × 5120 × 2 B = 126 MB다. 통합 메모리 120 GB에 비하면 작다. 위치는 arena다(크기는 `max_batch_`에서 정한다).
- 쓰는 양(청크당 수백 MB)은 GEMM 계산 시간(초 단위)에 비해 무시할 수 있다.

### 3.3 출력 회전과 epilogue 융합

- **블록이 128열 구간을 통째로 갖는다**(BN = 256 = 구간 2개). K 루프가 끝나면 16행 띠마다 C를 LDS(16 × 256 FP32 = 16 KB)로 모아 butterfly → scale → `svh`를 적용하고 쓴다. 이 띠를 16번 반복한다.
  - 산술은 X2b store epilogue와 같다(FP32 butterfly 같은 순서). 그래서 같은 입력의 decode/verify 결과와 행 단위로 비교할 수 있다(합 순서가 달라 비트 동일은 아니다).
- **gate + up → SwiGLU → down 입력:**
  - 블록의 BN = 256을 gate 128열 + up 128열(같은 열 구간)로 짠다.
  - epilogue에서 두 구간을 회전 → SwiGLU → `suh_down` 곱 → butterfly → FP16 down 입력 슬랩에 쓴다.
  - X2b 3a와 같은 산술이다. 활성화 FP32가 메모리에 가지 않는다.
- **residual:** o_proj/out_proj/down epilogue에서 hidden에 바로 더할 수 있다. X2b와 마찬가지로 측정 뒤에 정한다.

## 4. M에 따른 경로 나누기

| M(행) | 쓰는 곳 | 커널 | 근거 |
|---|---|---|---|
| 1–8 | decode, DFlash2 verify 1세션, 드래프터 head | X2b fdot2 GEMV(완료) | 메모리에 묶인다. 가중치를 한 번 읽는다(m = 8 207–209 GB/s) |
| 9–64 | 다중 세션 decode·verify(C × 1–8행), 짧은 prefill | **새 "skinny" WMMA:** BM = M을 16으로 올림(16/32/48/64), BN = 128(구간 1개), split-K, X2b 카운터 epilogue 재사용 | 블록 하나가 M 전체를 가져 가중치를 한 번만 푼다. M = 64에서 계산은 토큰당 약 63 ms, 메모리 77 ms라 아직 메모리에 묶인다[짐작]. 지금의 8행 반복(가중치를 C번 다시 읽음)보다 C배까지 빠르다 |
| 65 이상 | prefill 청크 | 3.1절의 BM = BN = 256 GEMM(M이 512 미만이면 BM = 128) | 계산에 묶인다 |

- 경계값(8/9, 64/65)은 실험(6절)으로 정한다. GGUF 쪽에는 이미 "batch < 1024 → Q8_1 경로, ≥ 1024 → FP16 경로" 분기가 있다(`UseQwen27bFp16Prefill`). EXL3는 모든 M에서 FP16 활성화를 쓴다. Q8_1 경로는 쓰지 않는다. int8 WMMA가 FP16보다 빠르지 않기 때문이다(57.2 대 55.5).

## 5. 디코드 비용을 줄이는 방법 (기준 5 %를 위해)

1. **해시 곱:** upstream tiles 코어의 `v_pk_mul_lo_u16` + `v_mad_u32_u16`(full rate 두 개)를 쓴다. X2b는 `v_mul_lo_u32`(quarter rate) 그대로다. GEMV에서는 영향이 없었지만(xb1 ±1 %, 메모리에 묶임), 계산에 묶이는 prefill에서는 디코드 VALU의 약 30 %가 줄어든다[짐작].
2. **BM = 512(wave 16개):** 디코드 한 번이 두 배의 행에 쓰인다. 디코드 비율이 약 8 %에서 4 %로 준다. LDS는 A 512 × 32 × 2 = 32 KB, B 32 × 256 × 2 = 16 KB로 64 KB 안이다. 레지스터와 occupancy는 실험으로 본다.
3. **wave 역할 나누기:** 일부 wave는 다음 K 단계의 B를 풀고, 나머지는 WMMA를 돈다(이중 버퍼 LDS). VALU와 WMMA가 서로 다른 wave에서 겹친다[짐작: RDNA3.5에서 WMMA와 다른 wave의 VALU가 동시에 issue되는지 측정 필요].
4. 1–3을 측정한 뒤에도 모자라면, 디코드한 FP16 B 띠를 MALL(32 MB)에 두고 같은 N 띠의 M 블록들이 나눠 읽는 생산자–소비자 구조를 검토한다. 복잡도가 커서 마지막 후보다.

## 6. 참고 구현

| 출처 | 내용 | X3에 쓰는 것 |
|---|---|---|
| upstream exllamav3 1.6.5 `rocm/quant/exl3_gemm_inner_rdna.cuh` | WMMA 재구성, 3단 파이프라인, split-K + 전역 lock, 출력 Hadamard 융합 선택. **M 타일 16만 지원**(`static_assert(TILESIZE_M == 16)`, 32/64행 MoE 타일 미이식). 큰 m은 `reconstruct` + hipBLASLt로 간다(W8 커널 목록) | B를 LDS로 전치하는 방식, FP32 누적. 큰 M 구조는 따르지 않는다(ExLlamaV3 prefill 411–434 tok/s, d32K 실행 불가가 그 결과다) |
| vcruz305/exllamav3-amd | RDNA3.5 WMMA GEMV, `v_sad_u8` 디코드, LDS bank conflict pad(+10 %), 그룹 MoE prefill GEMV 131–135 GB/s | LDS pad를 B 타일 배치에 적용한다 |
| Kyojin | Flash-Next prefill 8K 1,621 tok/s(MoE). 27B dense 수치는 없다 | 비교값만. gufo Flash-Next pp2048 1,801(`fec4759d`)이 이미 위다 |
| gufo `prefill_fp16.hip` | FP16 WMMA GEMM, BM = BN = 256, K-quant 역양자화 B | 틀 전체(3.1절) |

## 7. 기존 GGUF 경로와 합치는 방법

- **분기 지점:** `prefill_chunk.cpp` `gemm_weight`의 `IsExl3(w)` 가지(X-A가 기준 경로로 연결함)와 `LaunchExl3Projection`(dispatch). rows > 64면 새 prefill GEMM으로, 9–64는 skinny WMMA로 보낸다.
- **코드 공유:** `prefill_fp16.hip` GEMM 템플릿에 B 생산자 정책(K-quant | trellis)을 템플릿 인자로 더한다.
  - GGUF 인스턴스의 산술은 바뀌지 않는다. 비회전 GGUF 지문 27줄의 바이트 동일로 확인한다(R2·X2a와 같은 방식).
  - 공유 파일이므로 감독에게 순서를 받는다. 대안은 같은 틀을 EXL3 전용 파일(`kernels/exl3_gemm.hip`)에 복제하는 것이다. 이 경우 GGUF 경로를 전혀 건드리지 않지만, 중복이 생긴다. **결정 필요(9절).**
- **활성화:** EXL3 층은 생성 커널(3.2절)이 소비자별 FP16 슬랩을 쓴다. FP32 활성화는 alpha/beta 같은 비양자화 텐서용으로만 남긴다.
- **깊이 32K:** attention·KV 경로는 그대로이므로 바꿀 것이 없다.

## 8. 단계, 시험, 기대치

1. **독립 bench**(`tools/bench`, 커밋만 하고 운영 코드는 바꾸지 않음):
   - 27B 모양 × M 512/2048/4096에서 trellis B GEMM과 같은 틀의 Q4_K/Q6_K B GEMM(지금 커널)을 같은 창에서 TFLOPS로 비교한다.
   - 5절의 1–3을 변형으로 넣는다.
   - M 16/32/64 skinny WMMA 대 X2b 8행 반복의 GB/s도 잰다.
2. **정확성:**
   - GEMM 대 FP64(같은 FP16 입력, 커널 회전 산술 재현) 상대 L2 1e-5 이하
   - epilogue 융합(SwiGLU → down)과 비융합의 비트 동일
   - 행별 결과가 X2b decode 결과와 FP32 누적 차이 수준인지
3. **모델 연결:** dispatch와 `prefill_chunk.cpp`, 생성 커널 일반화. GGUF 지문 동일, X2a 기준 경로 대비 KL.
4. **창:** UD-Q4_K_XL과 ABBA로 pp512/2048/8192와 d32K를 판정한다(XB6와 같은 형식).

- **기대[짐작]:**
  - EXL3 pp는 같은 틀의 GGUF보다 디코드 VALU 차이만큼(약 3–5 %) 느린 데서 시작한다. 5절의 1–2를 넣으면 기준(5 %) 안에 들어갈 것으로 본다.
  - 다중 세션 decode·verify(M 9–64)는 skinny WMMA로 지금보다 크게 빨라진다.
  - 이 둘은 bench 1단계에서 먼저 숫자로 확인한다. 기준에 못 미치면 그 자리에서 보고한다.

## 9. 결정이 필요한 것

1. `prefill_fp16.hip` 템플릿에 B 생산자 정책을 더할지(공유, GGUF 지문으로 확인), EXL3 전용 파일로 복제할지
2. prefill 소비자별 FP16 슬랩(청크 4096에서 최대 약 126 MB)을 arena에 두는 데 동의하는지
3. skinny WMMA(M 9–64)를 X3 범위에 넣을지. 다중 세션 성능이고, prefill 기준에는 들어가지 않는다
