# Flash-Next 확장 계획 초안 (W7 준비 조사, 2026-10-29)

[W7 브리프](W7-BRIEF.md)의 산출물이다. 읽기만 하고 조사했다. 코드 변경, 빌드, GPU 실행, 내려받기는 하지 않았다. 구현은 27B R5 뒤에 사용자 확인을 받고 시작한다([HANDOFF](HANDOFF.md) "결정이 필요한 것": Flash-Next 착수).

조사 기준: 브랜치 `ai-tools/port-u4f`, HEAD `f95d1302`. HF 정보는 API 메타데이터로만 확인했다(2026-10-29).

**표기.** **[확인]**은 소스, API 응답, 문서에서 직접 읽은 것이다. **[짐작]**은 계산하거나 추론한 것이고, 검증할 단계를 함께 적는다. 텐서 형상은 `config.hpp` 주석에 적힌 값(hidden 2560 등)으로 계산했다. 파일 헤더로는 아직 확인하지 않았다(F0 단계에서 확인한다).

## 0. 요약

- **로더 [확인]:** dense 행렬은 지금 Q8_0, BF16, F16, F32만 받는다. 4비트 dense 커널은 없다. 4비트 커널(Q4_K, Q5_K, Q5_1)은 전문가(routed) 경로에만 있다.
- **목표 범위 [짐작]:** UD-Q4_K_XL에서 Q8_0인 dense 행렬은 약 42억 개다. decode 한 토큰이 읽는 약 6.1 GB 중 약 4.5 GB가 이 행렬이다. 1단계 대상(`attn_in`, `ssm_out_in`, `attn_out_in`, `head_in`, 약 33억 개)을 4.25–4.5 bpw로 내리면 토큰당 약 1.7 GB(약 29 %)가 줄어든다.
- **회전 융합 [확인]:** 27B와 달리 Flash-Next의 활성화 생성 커널은 대부분 FP32 행을 쓰고, 경로에 따라 Q8_1, 타일 Q8, F16 사본을 함께 쓴다. 회전을 융합할 곳은 경로마다 다르다(5절).
- **입력 그룹 [확인]:** `ffn_in`(전문가 gate/up, 공유 전문가, 라우터)은 decode에서 공유 전문가와 전문가가 같은 Q8_1 사본을 읽는다. 그래서 전문가를 다시 양자화하지 않고서는 이 그룹을 회전할 수 없다. 1단계에서는 제외한다.
- **변환 경로 [확인]:** 지금 컨버터(`rotate` → `llama-quantize`)로는 Flash-Next를 변환할 수 없다. 회전 BF16 전체(약 354 GB)를 중간 파일로 쓰기 때문이다. UD 파일을 바탕으로 dense만 바꿔 끼우는 "graft" 모드가 필요하다(7.3절).
- **품질 기준 [확인]:** BF16 전체(약 360 GB)는 128 GB 장비에 올라가지 않는다. Q8_0(238 GB)도 마찬가지다. 그래서 27B처럼 BF16 기준으로 잴 수 없다. 기준은 "UD 전문가 + dense BF16" 혼합 파일이나 기존 F32 CPU 참고(`reference_probe`)로 한다(8절).

## 1. 모델 구조 요약 (`src/models/qwen38_flash_next/config.hpp`)

[확인] 구조 (값은 주석 기준):

- `qwen4exp` 아키텍처, trunk 48층, hidden 2560
- hyper-connection(HC): residual stream 4개(`hc_dim` 10240), low rank 320
- 4층마다 full attention(12층): 24 head × 256, KV 2 head, QSA indexer 4 × 128
- 나머지 36층은 Gated DeltaNet: k head 16, v head 48, head dim 128, value dim 6144
- MoE: 전문가 512개 중 10개 사용, `expert_ff` 640, 공유 전문가 1개(`ff` 640)
- PLE n-gram 임베딩: 한 층에 주입한다. 16 head × 160 = 2560
- MTP: 별도 사이드카의 `blk.48`. 임베딩과 출력 헤드는 trunk 것을 빌려 쓴다

각 층의 흐름은 다음과 같다.

- `hc_attn` mixer가 residual 4줄에서 grouped RMSNorm `xn`([10240])을 만든다.
- `xn`에서 down(10240→320)과 up(320→10240)을 거쳐 gate를 얻고, epilogue가 `mixed`([2560])를 만든다.
- `mixed`가 attention이나 SSM의 입력이 된다. 결과는 combine 단계에서 residual에 다시 들어간다.
- FFN 쪽도 `hc_ffn`으로 같은 구조를 거친다.
- 출력 헤드 앞의 norm은 `output_hc` mixer가 맡는다.

## 2. 로더 지원 타입과 커널 (조사 항목 1)

### 2.1 텐서 계열별 허용 타입 [확인: `weights.cpp`, `device_model.cpp`]

| 계열 | GGUF 이름 | 모양 (K → N) | 로더 허용 | 실제로 쓰이는 커널 (decode ≤8행 / prefill) |
|---|---|---|---|---|
| DeltaNet 입력 | `attn_qkv`, `attn_gate` | 2560 → 10240, 6144 | Q8_0, BF16, F16, F32 | Q8_0 둘이면 16384행 하나로 쌓는다(`ssm_in`). decode는 `qfn_mmq_q8_0_dense_vec_preq`(Q8_1 입력). prefill은 `DenseF16SsmGemm`(conv 융합, F16 입력)이나 `W8A8Gemm`(타일 Q8) |
| DeltaNet α/β | `ssm_alpha`, `ssm_beta` | 2560 → 48 | F32, BF16, Q8_0 | 쌓아서 F16으로 좁힌다. Q8_0이 들어오면 `Stack`이 F32/Q8_0 혼합을 거절한다. decode `SmallGemm`, prefill hipBLASLt |
| DeltaNet 출력 | `ssm_out` | 6144 → 2560 | Q8_0, BF16, F16, F32 | decode는 Q8_1 GEMV(GDN epilogue가 Q8_1을 쓴다). prefill은 `DenseF16Gemm`(F16 출력 경로) 또는 `W8A8Gemm` |
| attention 입력 | `attn_q`(q·gate), `attn_k`, `attn_v` | 2560 → 12288, 512, 512 | Q8_0, BF16, F16, F32 | Q8_0이면 13312행으로 쌓는다. decode는 Q8 GEMV. prefill은 `AttentionF16Gemm`(norm, rope, KV 저장 융합)이나 W8A8 |
| attention 출력 | `attn_output` | 6144 → 2560 | Q8_0, BF16, F16, F32 | decode는 `SigmoidMulQ8_1` 뒤 Q8 GEMV. prefill은 F16 또는 W8A8 |
| indexer | `indexer.q_proj`, `indexer.k_proj` | 2560 → 512, 128 | Q8_0, BF16, F16, F32 | UD에서는 BF16이다(EXPERIMENTS #681 "keeps the indexer in BF16"). decode `SmallGemm`(FP32 입력), prefill hipBLASLt |
| HC down | `hc_{attn,ffn}_down`, `output_hc_down`, `nextn.hc_head_down` | 10240 → 320 | Q8_0, BF16, F16, F32 | decode는 Q8 GEMV(`HcCombineQ8_1`이 `xn`의 Q8_1을 쓴다). prefill은 `HcDownF16Gemm`이나 W8A8(`xn_q8t`) |
| HC up | `hc_*_up` | 320 → 10240 | Q8_0, BF16, F16, F32 | decode는 `SiluScaleQ8_1` 뒤 Q8 GEMV. prefill은 `HcMixF16Gemm`(epilogue 융합) |
| HC inject | `hc_*_inject` | 10240 → 4 | F32, Q8_0, BF16 | F32이면 mix epilogue 안에서 FP32로 계산한다 |
| PLE | `ple_key`, `ple_value` | 2560 → 10240, 2560 | Q8_0, BF16, F16, F32 | `Quantize` 뒤 Q8 GEMV |
| PLE 표 | `per_layer_token_embd` | 160 × 행 | IQ4_NL, BF16 | 디바이스에 올리지 않고 디스크에서 읽는다(`ngram_->StartRead`) |
| 라우터 | `ffn_gate_inp`, `ffn_gate_inp_shexp` | 2560 → 512, 1 | F32 | 둘을 쌓아 F16으로 좁힌다 |
| 공유 전문가 | `ffn_{gate,up,down}_shexp` | 2560 → 640, 640 → 2560 | Q8_0, BF16, F16, F32 | gate·up은 decode에서 `qfn_mmq_q8_0_dense_vec_preq`(gate 융합). down은 Q8 GEMV. prefill은 `SwigluHalf`/`SwigluQ8Tiled` 뒤 F16이나 W8A8. Q8_0일 때만 side stream에 겹쳐 돈다 |
| 전문가 gate/up | `ffn_{gate,up}_exps` | 2560 → 640, ×512 | Q4_K, Q5_K, Q6_K, Q5_1, Q8_0 (Q6_K는 업로드에서 거절) | decode `qfn_mmq_moe_gated_vec(_preq)`(Q4_K, Q5_K, Q8_0), prefill `RoutedGatedF16Gemm`(Q4_K, Q5_K) / mmq `*_moe_raw` |
| 전문가 down | `ffn_down_exps` | 640 → 2560, ×512 | 위와 같음 | decode `qfn_mmq_moe_vec`, prefill `RoutedF16Gemm`(Q5_1, Q8_0) |
| 임베딩, 헤드 | `token_embd`, `output` | 2560 → 248320 | Q8_0, Q6_K, BF16, F16, F32 (Q6_K는 업로드에서 거절) | 헤드는 `Dense`: decode Q8 GEMV, prefill W8A8. MTP도 같은 헤드를 쓰고, 세션마다 헤드 행 일부를 draft 표로 복사한다(`AppendDraftRows`) |
| MTP | `nextn.eh_proj` | 5120 → 2560 (임베딩 2560 + hidden 2560으로 나눈다) | Q8_0, BF16, F16, F32 | `DenseBatch`(`QuantizeBatch` + Q8 GEMV) |

### 2.2 4비트 관련 사실

- **[확인]** `executor.cpp` `Dense()`는 Q8_0만 양자화 경로로 보낸다. 나머지 타입은 `SmallType()`(BF16, F16, Q8_0, F32)이 처리하고, 그 밖의 타입이면 `logic_error`를 던진다. 쌓기(`Stack`)도 F32와 Q8_0만 받는다.
- **[확인]** UD-Q4_K_XL의 dense는 Q8_0이다. 근거는 두 가지다.
  - #474 포크 README: "r8: 145 tensors, 103.68 → 102.37 GiB". 145 = `attn_qkv` 36 + `attn_gate` 36 + `ssm_out` 36 + `attn_q`/`attn_k`/`attn_v` 각 12 + `output` 1이다.
  - 그 크기 감소(1.31 GiB)가 아래 계산과 맞는다: Q8_0 → Q4_1로 바꾼 2,484 M개 × 0.4375 B, 그리고 Q8_0 → Q4_K로 바꾼 헤드 636 M개 × 0.5 B.
- **[짐작]** HC, 공유 전문가, PLE의 key/value, MTP 사이드카의 dense도 Q8_0으로 보인다. 근거는 로더 기본 경로와 포크 레시피 주석("tuned Q8 kernels win")이다. 전문가 타입(gate/up Q4_K 또는 Q5_K, down Q5_1 또는 Q8_0)과 층별 배분은 헤더로 확인해야 한다(F0).

## 3. 토큰당 읽는 바이트 [짐작, F0에서 헤더로 확인]

| 묶음 | 원소 수 (trunk) | UD 타입 (추정) | decode 토큰당 바이트 |
|---|---:|---|---:|
| `attn_in` 양자화 행렬 (DeltaNet `ssm_in` 36 × 41.9 M, attention `attn_qkv` 12 × 34.1 M) | 1,918 M | Q8_0 | 2.04 GB |
| `ssm_out` (36 × 15.7 M) | 566 M | Q8_0 | 0.60 GB |
| `attn_output` (12 × 15.7 M) | 189 M | Q8_0 | 0.20 GB |
| 출력 헤드 | 636 M | Q8_0 | 0.68 GB |
| HC down / up (97 mixer × 3.28 M씩) | 318 M / 318 M | Q8_0 | 0.34 / 0.34 GB |
| 공유 전문가 (48 × 4.9 M) | 236 M | Q8_0 | 0.25 GB |
| PLE key/value | 33 M | Q8_0 | 0.04 GB |
| 라우터, α/β (F16), indexer (BF16) | — | F16/BF16 | 약 0.17 GB |
| 활성 전문가 10개 × 48층 | 2,360 M | Q4_K/Q5_1 | 약 1.47 GB |
| **합계** | | | **약 6.1 GB** |

- #474의 AR 토큰당 약 32 ms와 맞춰 보면 실효 약 190 GB/s다. 대역폭에 묶여 있다는 가정과 맞는다.
- 1단계 대상(`attn_in`, `ssm_out_in`, `attn_out_in`, `head_in`, 3,309 M)을 IQ4_XS(4.25 bpw)로 내리면 토큰당 약 1.76 GB가 줄어든다. Q4_K라면 약 1.66 GB다. 토큰당 시간의 상한 추정은 32 → 약 23 ms다. 실제 이득은 4비트 커널의 실효 대역폭에 달려 있다(6절).

## 4. 입력 그룹 표 초안 (조사 항목 2)

v1 규칙(PLAN "GGUF 표시 규칙"): 회전 블록 128, 입력 쪽만 회전, 그룹 안에서는 부호를 공유한다. "활성화 정밀도"는 decode(≤8행) / verify(≤8행, decode와 같은 경로) / prefill 순이다. 모두 [확인]이고, 형상은 `config.hpp` 기준이다.

| 그룹 | 입력을 만드는 곳 | K | 소비 텐서 | 활성화 정밀도 | 128 배수 / 256 배수 | 1단계 |
|---|---|---:|---|---|---|---|
| `attn_in` | `HcMix`(hc_attn) epilogue `mixed`: `HcMixEpilogue`(decode, `mixed_q8_1` 동시 기록), `HcMixEpilogueVec4F16`/`HcMixF16Gemm`(prefill, `x_half`+`x_q8t` 동시 기록), `Quantize`/`PrepareHalfInput`, MTP `DenseBatch` | 2560 | DeltaNet `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` / attention `attn_q`, `attn_k`, `attn_v`, `indexer.q_proj`, `indexer.k_proj` | Q8 행렬: Q8_1 / Q8_1 / F16 WMMA 또는 W8A8. α/β: FP32→F16. indexer: FP32 / BF16 | 예 / 예 | **대상.** `mixed`는 projection만 읽는다. 그래서 epilogue에서 FP32 `mixed` 자체를 회전하고, α/β(F32)와 indexer(BF16)도 오프라인에서 같이 회전하는 것이 가장 단순하다 |
| `ssm_out_in` | `GatedDeltaNet` epilogue(head별 RMSNorm + SiLU gate): `gdn_out` F32, `gdn_q8_1`(decode), `x_q8t`(prefill W8A8), `out_half`(prefill F16) | 6144 | `ssm_out` | Q8_1 / Q8_1 / F16 또는 W8A8 | 예 (head 128 = 회전 블록 하나) / 예 | **대상.** #474에서 가장 민감했다. 27B E2 결과로 4비트 여부를 정한다 |
| `attn_out_in` | attention 출력 × sigmoid gate: `SigmoidMulQ8_1`(decode), `SigmoidMul` 뒤 `Dense`(prefill) | 6144 | `attn_output` | Q8_1 / Q8_1 / F16 또는 W8A8 | 예 / 예 | **대상.** #474와 포크는 Q8_0으로 남겼다 |
| `head_in` | `HcMix`(output_hc) `mixed` → `Dense(output)`. **MTP 헤드도 공유한다:** `MtpHead`의 `HcMix`(nextn.hc_head), `DraftLogits`(prefix 행과 세션 draft 표) | 2560 | `output` (MTP draft 표와 prefix 포함) | Q8_1 / Q8_1 / W8A8 | 예 / 예 | **대상.** 드래프트 경로에서 빠뜨리면 수락률만 조용히 떨어진다(27B DFlash2와 같은 위험). tied 헤드(`output` 없음 → `token_embd`)는 거부한다 |
| `hc_xn_in` | mixer grouped RMSNorm `xn`: `RmsNormRows`, `HcCombineQ8_1`(decode, `xn_q8_1`), `HcCombineF16`/`HcCombineMoeF16`(prefill, `xn_half`+`xn_q8t`) | 10240 | `hc_*_down`, `output_hc_down`, `nextn.hc_head_down` | Q8_1 / Q8_1 / W8A8(`xn_q8t`) 또는 `HcDownF16Gemm` | 예 / 예 | **2단계.** `xn`은 mix epilogue(`mixed = Σ gate·xn`)와 F32 `inject`도 읽는다. 그래서 FP32 `xn`이 아니라 양자화 사본만 회전해야 한다. `inject`는 그룹 밖에 둔다(회전하지 않은 F32). halogen v2는 mixing 층을 6비트로 둔다 |
| `hc_lo_in` | `SiluScaleQ8_1` / `SiluScale` (low rank) | 320 | `hc_*_up` | Q8_1 / Q8_1 / F16 (`HcMixF16Gemm`) | **아니오** (320 = 2.5 × 128) / 아니오 | **제외.** v1 블록 128로는 회전할 수 없다. 1단 포맷만 가능하다. Q8_0으로 둔다 |
| `ffn_in` | `HcMix`(hc_ffn) `mixed`: decode `ReadyQ8`(공유 전문가와 전문가가 같은 Q8_1 사본을 읽는다), 전문가 vec 커널 내부 양자화(`qfn_mmq_moe_vec`는 FP32 x를 받는다), prefill `x_half`(라우터, 공유 전문가, 전문가가 함께 쓴다) | 2560 | `ffn_gate_inp`(라우터 F32), `ffn_gate_inp_shexp`(F32), `ffn_{gate,up}_shexp`, `ffn_{gate,up}_exps`(3차원, `.rot` 하나를 공유) | 라우터 FP32→F16. 공유 전문가 Q8_1 / Q8_1 / F16·W8A8. 전문가 Q8_1(커널 내부) / F16 WMMA | 예 / 예 | **3단계.** 공유 전문가만 회전하면 decode에서 회전된 사본과 회전 안 된 사본이 둘 다 필요하다. 그룹 전체를 회전하려면 전문가 gate/up(약 80 B개)을 BF16에서 다시 양자화해야 한다 |
| `shexp_down_in` | `GatedDense` 출력(`qfn_mmq_q8_0_dense_vec_preq` gated, FP32) 뒤 `Quantize`, prefill `SwigluHalf`/`SwigluQ8Tiled` | 640 | `ffn_down_shexp` | Q8_1 / Q8_1 / F16 또는 W8A8 | 예 (5 × 128) / **아니오** | 2단계 후보. 1단 4비트 포맷만 가능하다(E6) |
| `down_exps_in` | 전문가 SwiGLU: gated vec 커널 출력 → `qfn_mmq_moe_vec` 내부 양자화, prefill `RoutedGatedF16Gemm`의 `up_half` | 640 | `ffn_down_exps` (3차원) | 커널 내부 Q8_1 / 같음 / F16 | 예 / 아니오 | 3단계. 1단 포맷만 가능하다(Q5_1, Q8_0, IQ4_NL, Q4_0/Q4_1). #474에서도 확인했다 |
| `ple_in` | PLE 임베딩 수집 `ple_emb` → `Quantize` | 2560 | `ple_key`, `ple_value` | Q8_1 (모든 경로) | 예 / 예 | 2단계. 한 층뿐이라 작다(0.04 GB/토큰) |
| `mtp_emb_in` | `nextn.enorm`(임베딩 RMSNorm) | 2560 | `nextn.eh_proj`의 임베딩 절반 | `DenseBatch` Q8_1 | 예 / 예 | 사이드카 2단계. 지금 GGUF는 [emb | hidden]을 행마다 이어 붙인 5120 열이다. 두 절반이 다른 입력이므로 `.rot`([5120])에 두 그룹의 부호를 이어 붙여야 한다. v1 규칙에서는 이것이 한 텐서 두 그룹이 된다(9절) |
| `mtp_hidden_in` | `nextn.hnorm`(hc_dim 전체 RMSNorm, stream별 2560) | 2560 | `nextn.eh_proj`의 hidden 절반 | `DenseBatch` Q8_1 | 예 / 예 | 위와 같다 |
| MTP 블록 내부 | 사이드카 `blk.48`의 hc mixer, attention, 공유 전문가, 전문가 | trunk와 같다 | trunk와 같은 구성 | trunk와 같다 | — | 사이드카는 따로 변환한다. 1단계에서는 그대로 두고 `head_in`만 맞춘다 |

"gated residual"은 Flash-Next에서 세 가지를 가리킨다고 본다[짐작]: 공유 전문가의 sigmoid gate(`ffn_gate_inp_shexp`, F32, 라우터와 같이 쌓인다), attention 출력 gate(`attn_q`의 gate 절반), PLE gate(`PleGate`). 이 중 별도 행렬 입력 그룹이 되는 것은 없다. 셋 다 위 표의 `ffn_in`, `attn_in`, `ple_in`, `attn_out_in`에 들어 있다.

## 5. 활성화 생성 커널과 회전 융합 위치 (조사 항목 3)

### 5.1 경로별 목록 [확인: `executor.cpp`, `batch.cpp`, `kernels.hpp`]

| 그룹 | decode (1행, graph 캡처) | MTP verify (2–8행) | MTP 드래프트/catch-up (`batch.cpp`) | prefill (넓은 배치) |
|---|---|---|---|---|
| `attn_in` | `HcMixEpilogue` (+`mixed_q8_1`) | 같음 (`MatrixRows` ≤ 8은 decode와 같은 경로) | `DenseBatch` → `QuantizeBatch` | `HcMixEpilogueVec4F16`, `HcMixF16Gemm`(`x_half`, `x_q8t`), `PrepareHalfInput`, `QuantizeQ8Tiled` |
| `ssm_out_in` | `GatedDeltaNet` epilogue → `gdn_q8_1` | 같음 | `GatedDeltaNetBatch` | `GatedDeltaNet`(`x_q8t` 또는 `out_half`) |
| `attn_out_in` | `SigmoidMulQ8_1` | 같음 | `DenseBatch` | `SigmoidMul` + `QuantizeQ8Tiled` / `PrepareHalfInput` |
| `head_in` | `HcMix(output_hc)` → `Quantize` | 같음 | `MtpHead` → `HcMix(nextn.hc_head)` → `DraftLogits` | `Quantize` 또는 `QuantizeQ8Tiled` |
| `hc_xn_in` | `HcCombineQ8_1` | 같음 | `DenseBatch`(10240×320 전용 분기) | `HcCombineF16`, `HcCombineMoeF16`(`xn_q8t`) |
| `shexp_down_in` | `Quantize` (GatedDense 출력) | 같음 | — | `SwigluHalf`, `SwigluQ8Tiled` |

- **graph:** decode graph는 위 커널을 그대로 캡처한다(`Run(session, key, graph, body)`). 따로 고칠 생산 커널은 없다. 다만 부호 벡터는 업로드 때 고정된 디바이스 포인터여야 한다(재생할 때 주소가 바뀌면 안 된다).
- **공통 양자화기:** 위 융합 커널을 쓰지 않는 호출은 모두 `qfn_mmq_quantize_q8_1`(`Executor::Quantize`), `QuantizeQ8Tiled`, `NarrowActivations`(`PrepareHalfInput`)로 모인다. 그런데 이 세 함수는 어느 그룹의 입력인지 모른다. 그래서 그룹별 부호를 넘기는 인자가 필요하다.

### 5.2 27B와 다른 점

1. **FP32 중간값이 있다.** 27B는 활성화 생성 커널이 Q8_1을 곧바로 쓰기 때문에 회전을 그 안에 넣어야 한다(PLAN R2). Flash-Next는 대부분 FP32 행(`mixed`, `gdn_out`, `ctx`)을 먼저 쓴다. 그래서 두 가지 방법이 있다.
   - (a) FP32 행 자체를 회전한다. 소비자 전원이 회전 가중치일 때만 된다. `attn_in`, `ssm_out_in`, `attn_out_in`, `head_in`이 이 경우다.
   - (b) 양자화 사본만 회전한다. `hc_xn_in`처럼 같은 행을 elementwise로도 읽는 경우다.
2. **같은 입력의 사본이 여러 정밀도로 생긴다.** prefill에서는 F16(`x_half`, F16 WMMA)과 타일 Q8(`x_q8t`, W8A8)이 한 epilogue에서 함께 나온다. 회전은 그 epilogue 안에서 FP32에 한 번 적용하고, 두 사본에 같은 값을 써야 한다.
3. **W8A8 활성화 형식은 27B와 같다.** `x_q8t`는 27B `prefill_quant_gemm.hpp`의 타일 Q8_1 레이아웃 그대로다(`w8a8_wave64.hip`가 그 헤더를 include해 Q8_0으로 인스턴스화한다). 다만 Flash-Next `QuantizeQ8Tiled`는 activation-sum sidecar를 쓰지 않는다. 27B 템플릿의 Q4_K/Q5_K(`HasOffset`)에는 그 sidecar가 필요하다. IQ4_XS나 Q8_0에는 필요 없다.
4. **decode 활성화가 다르다.** 27B decode GEMV(`gemv_quant.hip` `LaunchGEMV`)는 FP32 활성화를 받는다. Flash-Next decode는 mmvq 형식의 행 단위 `block_q8_1`(`MATRIX_ROW_PADDING`)을 받는다. 같은 가중치 타입이라도 활성화 형식이 달라 커널을 그대로 바꿔 끼울 수 없다(6.3절).
5. **shape 전용 융합 prefill 커널이 많다.** `DenseF16SsmGemm`(16384×2560, conv 융합), `AttentionF16Gemm`(13312×2560, norm/rope/KV 융합), `HcDownF16Gemm`, `HcMixF16Gemm`은 모두 Q8_0 가중치를 F16으로 풀어 WMMA에 넣는다. 4비트로 바꾸면 이 커널들의 가중치 디코드도 바꾸거나 일반 경로로 돌려야 한다.
6. **드래프트 헤드 경로.** 27B DFlash2와 같은 위험이 있다. MTP는 trunk 헤드 행을 세션 draft 표로 복사한다(`AppendDraftRows`). 행을 복사하는 것은 회전과 상관없지만, 그 입력 `mixed`(nextn.hc_head)는 회전해야 한다.

## 6. 4비트 dense 커널 (조사 항목 4)

### 6.1 지금 있는 것 [확인]

| 커널 | 타입 | 용도 | dense에 쓸 수 있나 |
|---|---|---|---|
| `qfn_mmq_q8_0_dense_vec_preq` (mmvq) | Q8_0 | dense decode/verify, gated 변형 | Q8_0만 |
| `W8A8Gemm` (27B 템플릿 Q8_0 인스턴스) | Q8_0 | dense prefill | 템플릿은 다른 타입을 지원한다(6.3절) |
| `DenseF16Gemm`, `DenseF16SsmGemm`, `AttentionF16Gemm`, `HcMixF16Gemm` | Q8_0 → F16 | dense prefill | Q8_0만 |
| `qfn_mmq_moe_vec`, `qfn_mmq_moe_gated_vec(_preq)` | Q4_K, Q5_K, Q5_1, Q8_0 | 전문가 decode | "전문가 1개짜리 MoE"로 쓸 수는 있지만 활성화 행마다 가중치를 다시 읽는다 |
| `qfn_mmq_*_moe_raw`, `RoutedF16Gemm`, `RoutedGatedF16Gemm` | Q4_K, Q5_K, Q5_1, Q8_0 | 전문가 prefill | 포크가 이 방식으로 prefill을 처리했다(`QuantDense`, `zero_ids`) |

### 6.2 zengjiajun0623/gufo `gufo-opt/flash-next` 검토 [확인: `gh`로 읽기만 함]

- **출처:** <https://github.com/zengjiajun0623/gufo/tree/gufo-opt/flash-next>, HEAD `b3f4202eb26080b37dc66f28d4a9d8bbff456209` (2026-10-07). gufo 0.8.0(`e03bb91`) 위의 12개 커밋이다. 저장소 라이선스는 **MIT**(gufo-org/gufo도 MIT)이고, gufo-org/gufo의 포크다. 작성자는 #474에서 "Feel free to take anything"라고 했다. 커밋에 "Co-Authored-By: Claude" 트레일러가 있다. 가져다 쓸 경우 커밋 메시지에 출처 URL과 SHA를 남긴다.
- **dense 4비트 변경 내용 (diff 요약):**
  - `weights.cpp`: dense, HC, 헤드 허용 타입에 Q4_K, Q5_K, Q5_1, Q4_1을 더하고, `FormatOf`에 Q4_1(32, 20)을 더했다.
  - `device_model.cpp`: `Stack`과 `stackable`이 Q4_K, Q5_K, Q4_1도 받게 했다.
  - `mmvq.hip.cpp`: 새 `mul_mat_vec_kq<type, ncols, has_gate, rows_per_wave>`를 추가했다. wave32 하나가 가중치 행(기본 1행)을 1–8개 입력에 대해 한 번 읽는다. Q4_K는 기존 `Q4MoeFragment`를 쓰고, Q4_1/Q5_1/Q5_K는 ggml `vec_dot`을 쓴다. 이 kernel의 진입점이 `qfn_mmq_kq_dense_vec_preq`다. 환경 변수 `GUFO_KQ_ROWS`가 있다.
  - prefill: `Executor::QuantDense`가 dense 행렬을 "전문가 1개"로 보고 routed mmq(`qfn_mmq_q4_1_moe_raw` 추가)로 처리한다.
  - verify(≤64행)는 8행씩 vec 커널로 처리한다.
- **#474 측정:** Q4_1 dense 약 210 GB/s, Q4_K dense 약 150 GB/s(N = 4)다. "단순한 레이아웃이 이겼다." 2블록 unroll과 다행 wave는 더 느렸다.
- **평가 [짐작]:**
  - 0.8.0 이후 `Executor::Dense`, `DenseBatch`, F16 prefill 경로, decode Q8_1 생산 커널이 많이 바뀌었다. 패치를 그대로 적용할 수는 없고 다시 구현해야 한다.
  - prefill을 routed mmq로 돌리는 방식은 지금 Q8_0의 F16 WMMA 융합 경로(`DenseF16SsmGemm` 등)보다 느릴 가능성이 크다. #474 시점의 prefill은 550–660 tok/s였고, 지금 Q8_0은 pp2048이 약 1,900 tok/s다.
  - 커널 단위 op 테스트는 dense 쪽에 없다(`flash-next-experts` 브랜치에는 전문가 op 테스트가 있다).
  - 쓸 만한 것은 decode vec 커널의 구조(입력 1–8행 공유, gate 융합)와 Q4_1 측정 결과다.
- **#474의 다른 산출물:** `score` 서브커맨드(paired NLL / top-1 / 절단 KL), `requant.py`(ctypes `ggml_quantize_chunk` + imatrix로 텐서별 재양자화, UD 파일 위에 덮어쓰기), `sidecar_fix.py`, 레시피들. `requant.py` 방식은 7.3절 graft 모드와 같은 구조다(다만 Q8_0 원본에서 시작했다).

### 6.3 27B 커널(`src/models/qwen/hip/kernels/`) 재사용 판단 [확인 + 짐작]

- **prefill: 재사용 가능.** `WKQuantA8BlockedWmmaGEMMKernel<..., WType>`는 Q4_K, Q5_K, Q6_K, Q3_K, Q8_0을 다룬다. 27B 커널 전체로는 IQ4_XS, IQ4_NL, IQ3_S까지 지원한다. Flash-Next는 이미 이 헤더를 Q8_0으로 쓰고 있다. 4비트 타입으로 인스턴스화하고 Flash-Next 형상에서 다시 재면 된다.
  - Q4_K는 activation-sum sidecar가 필요하다. 그래서 `QuantizeQ8Tiled`와 융합 epilogue가 sidecar도 쓰도록 바꾸거나, sidecar가 필요 없는 IQ4_XS를 고른다.
  - R0b에서 prefill은 모든 타입이 WMMA 연산에 묶였다. 그래서 prefill의 목표는 "빨라짐"이 아니라 "느려지지 않음"이다.
- **decode/verify: 그대로는 재사용할 수 없다.** 27B GEMV는 FP32 활성화를 받고(`QuantWarpBlockDot`), small-batch wave64 커널은 "measured Qwen27B verification shapes" 전용이다(`TryLaunchKQuantSmallBatchWave64`). 선택지는 셋이다.
  - (a) Flash-Next decode에서 27B FP32-활성화 GEMV를 부른다. `mixed` FP32가 이미 있다. 활성화 산술이 Q8_1에서 FP32로 바뀌어 기존 비트 동일성이 깨지지만, R0b에서 IQ4_XS AR 89 %를 낸 커널이다.
  - (b) mmvq에 4비트 dense vec 커널을 더한다(포크 방식, Q8_1 활성화).
  - (c) 27B `QuantWarpBlockDot`의 블록 디코드를 Q8_1 입력용으로 옮긴다.
  - F3 단계에서 셋을 같은 창에서 비교하자고 제안한다. 27B 커널을 Flash-Next에서 include하는 것은 이미 선례가 있다(`w8a8_wave64.hip`). 공통 위치로 옮길지는 구현 때 정한다.
- **타입 후보 [짐작]:**
  - K가 256의 배수인 그룹(2560, 6144)은 IQ4_XS(27B R0b 최고), Q4_K, Q4_1(포크 최고) 셋이다.
  - K = 640인 `shexp_down_in`과 `down_exps_in`은 1단 포맷만 가능하다: IQ4_NL, Q4_0/Q4_1, Q5_1.
  - Q4_0/Q4_1은 27B 로더에도 없다. HANDOFF 결정 항목 "Q4_0/Q4_1 커널 추가"와 함께 정한다.

## 7. 모델 파일과 디스크 계획 (조사 항목 5)

### 7.1 HF 파일 [확인: HF API, 2026-10-29]

`unsloth/Qwen3.8-Flash-Next-GGUF`:

- main은 `766911a6b7369840a91dbcd95f9f997acaab6cd6`(2026-10-06), 문서 고정 리비전은 `38bb39ee97821de2c9009abb7e93950eec396e66`이다.
- 두 리비전 사이에서 UD-Q4_K_XL, BF16, `MTP/*shared*`의 sha256은 같다.
- 바뀐 것은 비공유 MTP 파일(`MTP/mtp-…-BF16/Q4_K_M/Q8_0.gguf`)뿐이다. 커밋 `efc50b7c` "Update shards due to upstream changes"가 그 변경이다.
- 모델 카드의 base_model은 `Qwen/Qwen3.8-Flash-Next`, 라이선스는 other다.

| 용도 | 파일 | 크기 (바이트) | sha256 (HF lfs oid) |
|---|---|---:|---|
| 기준선 | `UD-Q4_K_XL/…-00001-of-00004.gguf` | 10,946,624 | `4448186216b3af4cc558bbce2c3213f01608f8f8b2e5267a9767971dd3ec8082` |
| | `…-00002-of-00004.gguf` | 49,859,583,136 | `3f342f1c1580473f1ee94ddd5b28206e8c07a70fa1a366f59d1d6c922919a6c9` |
| | `…-00003-of-00004.gguf` | 49,376,141,504 | `56758f40269cad5cd9b0d3d6fbae0f40f6d5be6de49e4ab392dbe83157d9cbd3` |
| | `…-00004-of-00004.gguf` | 12,087,983,520 | `753bda48b98ba4f1636134a90a967de1b2d3908a236c026e464777342e53510a` |
| | 합계 | **111,333,654,784 (103.7 GiB)** | |
| MTP (운영) | `MTP/mtp-Qwen3.8-Flash-Next-shared-Q8_0.gguf` | 2,786,568,256 | `5ff54097406a905cf3a724c709124ceb0e3e10235ee862298969e91c96fa96e6` |
| MTP 원본 | `MTP/mtp-Qwen3.8-Flash-Next-shared-BF16.gguf` | 5,227,963,456 | `32473fa74e8ed0f0026219b09b43d303f52f328c0f8b0caf4b24bf8834cc596b` |
| 회전 원본 | `BF16/…-00001-of-00008.gguf` | 10,946,368 | `348986b2568af284eb16a0be0ccd17aa4cf0bc823de4a1c9da8e5a24365e3fd9` |
| | `…-00002` | 10,447,201,664 | `905f21f256847501da84872c526791539a660328b85b701049d859a1c3dfa1cd` |
| | `…-00003` | 102,400,491,712 | `fd6926a9142df95273cb92623b151f09d260f85b7c02e9469a1b610b552c6d0e` |
| | `…-00004` | 48,394,236,672 | `e4ca944e85c400a5fecb6fd7b0ae0358b67e9bb39ded38a9a8596f8850e7e6dc` |
| | `…-00005` | 48,367,899,040 | `c110361d1bbe8064e15c58c456cc35f0fe3d1b1329869e51b067909f83b95e6e` |
| | `…-00006` | 48,615,233,984 | `cbce912e81f2d983e0a302ceb48554a45ecade9b083f4827d53fc8191af3aeb6` |
| | `…-00007` | 48,367,899,040 | `23fbbbef0920c451c2fa4d9cf96d3dd167ed38c1cead608457464812d8930bc8` |
| | `…-00008` | 47,426,022,016 | `b6e4e5c646f8f708bd471e922004aa752acbcdac1d3abb27c9d42037630916f7` |
| | 합계 | **354,029,130,496 (329.7 GiB)** | |
| 참고 | `imatrix_unsloth.gguf_file` | 580,038,720 | (감독이 받는 중) |

- **원본 선택: BF16 GGUF.**
  - [확인] HF safetensors(`Qwen/Qwen3.8-Flash-Next`, 리비전 `de4b8e4d`)는 약 1,800억 개, 360 GB, 약 130개 파일(각 약 3.3 GB)이다.
  - [짐작] safetensors를 쓰면 GGUF 이름으로 바꾸는 변환(HC, PLE, MTP 분리 포함)을 따로 만들어야 한다. BF16 GGUF는 unsloth가 이미 변환해 둔 것이다. 그래서 27B D2처럼 BF16 GGUF를 원본으로 쓴다.
  - [짐작] BF16 `00003`(102 GB, 텐서 하나 크기)은 PLE 표다. Q8_0과 UD-Q5/Q6의 `00003`이 54.4 GB로 같은 것이 근거다. dense 텐서가 어느 분할에 있는지는 헤더를 읽어 확인한다(F0).
- **감독 진행 상황 (2026-10-29 통지):** 감독이 외장 디스크 `/run/media/m2on/T5 EVO/AI/model/originals-flash-next-20261029/unsloth-Qwen3.8-Flash-Next-GGUF/`로 리비전 `766911a6`의 BF16 8분할, MTP BF16과 shared-BF16, imatrix를 받고 있다. 받은 뒤 외장 여유는 약 228 GB다.
  - 이 조사 중에 확인했을 때 외장 디스크는 exFAT, **`ro`로 마운트**되어 있었고 여유는 596 GB였다. 브리프의 "약 2 TB"와 다르다. 쓰기는 감독이 다시 마운트해서 진행한 것으로 본다.

### 7.2 디스크 예산

- 루트(NVMe): 여유 약 394 GB(`df -h`, 2026-10-29). `window.py` 하한은 150 GiB다. 그래서 실제로 쓸 수 있는 공간은 **약 244 GiB**다.
- 27B 작업이 곧 쓸 공간: 회전 BF16 55 GB와 시험 Q4 2개(각 약 17 GB). 이 동안 남는 공간은 약 150 GiB 안팎이다[짐작].
- Flash-Next GPU 단계에 루트에 있어야 하는 것:
  - 기준선 UD 103.7 GiB
  - 후보 하나 약 102 GiB[짐작: 1단계 대상 4비트화로 약 1.7 GB 감소]
  - MTP shared-Q8_0 2.6 GiB
  - 합계 **약 208 GiB**
- **결론 [짐작]:**
  - Flash-Next GPU 단계는 27B 회전 BF16과 실험 GGUF를 지운 뒤에야 할 수 있다. 그때 여유는 약 36 GiB다. 후보는 **한 번에 하나만** 루트에 둔다.
  - 후보 파일을 외장에서 실행하는 것은 안 된다. PLE 표를 실행 중에 디스크에서 읽고(`ngram_->StartRead`), USB exFAT은 측정에 섞이기 때문이다.
  - 다 쓴 후보는 외장으로 옮겨 보관하거나 지운다. 외장 여유 약 228 GB에는 후보 2개까지 들어간다.
- **회전 BF16 전체(약 354 GB)는 어디에도 두지 않는다.** 루트에는 하한 때문에, 외장에는 남는 공간 228 GB 때문에 들어가지 않는다. 그래서 7.3절의 graft 경로가 필요하다. 중간 파일은 회전한 dense만 담은 BF16(약 7–9 GB) 하나만 허용한다.

### 7.3 지금 컨버터로 가능한가, 필요한 변경 (감독 요청)

**지금 `tools/rotquant/convert.py`로는 안 된다 [확인]:**

1. `rotate_file()`은 회전한 그룹뿐 아니라 **모든 텐서**를 복사해 회전 BF16 GGUF 하나를 쓴다. Flash-Next라면 전문가 BF16 약 242 GB와 PLE 102 GB를 포함해 약 354 GB가 된다.
2. 2단계 `llama-quantize`는 그 파일 전체를 읽어 모든 텐서를 다시 양자화한다. `--copy-recipe-from UD`를 써도 전문가가 BF16에서 다시 양자화된다. 그러면 UD 전문가와 바이트가 달라지고(보정 데이터도 다르다) 비교가 섞인다. 전문가 1,200억 개를 양자화하는 CPU 비용도 크다.
3. `file_sha256(model.paths)`가 입력 분할 전부를 해시한다. 외장 USB에서 매번 354 GB를 읽게 된다.
4. 출처 KV(`gufo.rotation.source_sha256`)가 원본 하나만 담는다. 바탕 파일과 원본 두 개를 기록할 자리가 없다.

**필요한 변경: "graft" 모드 (공통 `tools/rotquant/`) [짐작, 구현 때 확정]:**

- **입력:**
  - `--base <UD-Q4_K_XL 첫 분할>`(루트): 레시피가 바꾸지 않는 텐서는 여기서 바이트 그대로 복사한다. 전문가, PLE, 임베딩, HC, 공유 전문가가 UD와 정확히 같다.
  - `--source <BF16 첫 분할>`(외장, `:ro` 마운트): 레시피가 바꾸는 텐서만 읽는다. 분할 헤더로 텐서와 분할의 대응을 만든다(dense 약 8.4 GB만 USB에서 읽는다).
- **텐서별 스트리밍 양자화:**
  - 회전 대상 텐서는 행 묶음 단위로 읽고 회전한 뒤, 메모리에서 바로 양자화해 출력에 쓴다. 중간 회전 BF16 파일이 없다.
  - 양자화기는 기준 llama.cpp(`68d9053a`, `qwen4exp` 지원은 `src/llama-arch.cpp`에서 확인)의 `libggml-base`를 ctypes로 불러 `ggml_quantize_chunk`(+imatrix 행)를 쓴다. #474 `requant.py`와 같은 방식이다.
  - 대안은 회전 dense만 담은 작은 BF16 GGUF(약 7 GB)에 `llama-quantize`를 돌리고 결과를 바꿔 끼우는 것이다. 다만 텐서 일부만 있는 `qwen4exp` 파일을 `llama-quantize`가 받는지 확인하지 않았다. 층별 텐서 개수 검사에 걸릴 수 있다.
  - 두 양자화기가 같은 비트를 내는지는 27B 텐서 하나로 W1 산출물과 비교해 확인한다.
- **출력:** 루트에 단일 GGUF(약 110 GB)를 쓴다. 회전 텐서마다 `.rot`, 전역 `gufo.rotation.*` KV를 넣는다. 출처 KV는 `source_*`(BF16, HF sha256 기록값 사용)와 `base_*`(UD sha256)로 둘을 남긴다. 출력 쓰기는 한 번에 약 110 GB이므로 "10 GB를 넘는 디스크 쓰기"로 감독 허락을 받는다.
- **해시:** 원본 전체를 해시하지 않고, 받을 때 검증한 HF sha256(`SHA256SUMS`)을 KV에 기록한다. 실제로 읽은 텐서 바이트의 해시를 따로 남긴다.
- **모델 정의 `tools/qwen-flash/rotquant_groups.py`:**
  - `ARCHITECTURE = "qwen4exp"`
  - 4절의 그룹 표
  - 역할별 허용 타입은 `weights.cpp`에서 그대로 가져오고, F3에서 커널이 생기면 넓힌다
  - 3차원 전문가 텐서는 `.rot` 하나를 공유한다(3단계)
  - F32/BF16 소비자(α/β, indexer, 라우터)는 회전한 뒤에도 float 타입을 유지한다
- **확인할 기존 코드 [확인]:** `plan.py`의 `tensor.ne[0] % 128`과 `rotate_last_axis`는 3차원(행 = 원소 수 / K)도 처리한다. 이름을 소문자로 요구하는 규칙(`tensor_type_lines`)은 `llama-quantize` 경로에만 해당하므로, graft 모드에서는 필요 없다.
- **MTP 사이드카:** 같은 graft로 처리한다. base는 shared-Q8_0, source는 shared-BF16이다. 1단계에서는 사이드카를 바꾸지 않는다.

### 7.4 받는 순서

1. **(진행 중, 감독)** BF16 8분할, MTP BF16과 shared-BF16, imatrix를 외장에 받는다. sha256은 위 표와 대조한다.
2. **F0 (GPU와 루트 공간 필요 없음):** 외장 BF16 분할들의 헤더를 읽어 텐서 이름, 타입, 형상, 분할 위치를 정리하고 `config.hpp` 값을 확인한다. UD 헤더도 필요하다. 방법은 둘 중 하나를 감독이 정한다.
   - (a) UD를 먼저 받아 헤더를 읽는다.
   - (b) HF에서 각 분할 앞부분 몇 MB만 Range로 읽는다. 이것도 내려받기이므로 감독 결정이다.
3. **UD-Q4_K_XL(111 GB) + MTP shared-Q8_0(2.8 GB):**
   - 권장 [짐작]: 외장에 먼저 받아 둔다. 228 GB에서 114 GB를 쓰면 114 GB가 남는다.
   - Flash-Next GPU 단계가 시작될 때(27B R5 뒤, 27B 실험 파일을 지운 뒤) 루트로 복사한다. USB 읽기 111 GB에 몇 분 걸린다.
   - 이렇게 하면 네트워크 내려받기를 루트 공간 사정과 떼어 놓을 수 있다.
   - 외장에 둘 공간이 없으면 GPU 단계 직전에 루트로 바로 받는다.
4. **후보 GGUF:** graft로 루트에 하나씩 쓴다. 측정한 뒤 외장으로 옮기거나 지운다(지우기 전에 감독 확인).
5. 외장 원본(BF16 등)은 Flash-Next 단계가 끝날 때까지 둔다. 다시 받는 비용(354 GB)이 크기 때문이다.

## 8. #474의 측정 방법과 하네스 (조사 항목 6)

### 8.1 #474 [확인: 이슈 본문, 포크 README]

- **에이전트 벤치 "pibench":** pi 코딩 에이전트로 24개 작업을 돌린다. 자체 채점이고 단일 세션, greedy다. 지표는 wall, decode 중앙값 tok/s, prefill 초, 통과 수다. 실제 세션 198건 재생(중앙값 16.6K 토큰)도 했다.
  - 작업 세트와 재생 데이터는 비공개이고, 도구만 공개했다. 그래서 그대로 재현할 수는 없다.
  - 단일 프롬프트 MTP tok/s는 잡음이라고 보고 10개 프롬프트 `acc_probe.py`(`/metrics`)를 썼다.
- **품질 기준:** "Q8-trunk 기준 파일(UD)" 대비 top-1 ≥ 0.97, 절단 KL ≤ 0.02였다.
  - 코퍼스: 기준 파일의 greedy 출력 30개, 11,193 위치. 도구는 `score` 서브커맨드다.
  - 전문가 변경은 KL로 평가할 수 없다고 보고 paired NLL(약 11K 위치, 게이트 dNLL ≤ +0.05 %)을 썼다.
- **`ssm_out` 민감도 (P4-2):** 7개 텐서 계열을 하나씩 되돌리는 sweep을 했다.
  - r8(`ssm_out`도 Q4_1): top-1 0.9669, KL 0.0119
  - A2(`ssm_out` Q8_0): 0.9720 / 0.0083으로 통과, decode −1.2 %
  - A1(헤드 Q8_0): 0.9735 / 0.0100이지만 decode −5.9 %라 버렸다
- **imatrix:** llama-imatrix로 UD 첫 분할 위에서 만들었다. 코퍼스는 330 KB(코드, 영문 위키, 중문 위키), `-c 512`다.

### 8.2 우리 하네스로 재현할 수 있나 [확인 + 짐작]

- **BF16 기준은 불가능하다.**
  - BF16 전체(약 330 GiB)와 Q8_0(238 GB)은 128 GB 장비에 올라가지 않는다. 디바이스에서 돌릴 수 있는 가장 큰 파일은 UD-Q4_K_XL이다.
  - 로더의 전문가 허용 타입에 BF16이 없다. 그래서 BF16 전문가 파일은 CPU 참고로도 읽을 수 없다.
- **기준 제안:** **"UD 전문가 + dense BF16" 혼합 파일.** graft로 dense만 BF16 원본(회전하지 않음)으로 바꾼 파일이다.
  - 로더가 dense BF16을 받으므로 로드할 수 있다. 크기는 약 115 GB로 추정한다(dense가 약 4 GB 늘어난다)[짐작].
  - 메모리 여유가 빠듯하다. RFC는 UD + Q8 MTP가 약 114 GB/125 GiB라고 적었다. 짧은 문맥, MTP 없이 돌린다.
  - 이 기준은 전문가가 같으므로 **dense 변경의 효과만** 깔끔하게 잰다(#474의 지적과 같다).
  - 메모리가 모자라면 기준을 회전 대상 그룹만 BF16으로 둔 파일로 줄인다.
  - BF16 dense라 prefill이 hipBLASLt/SmallGemm 경로를 탄다. 기준값으로만 쓰므로 속도는 상관없다.
- **이미 있는 Flash-Next 도구:** `tests/models/qwen38_flash_next/`
  - `reference_probe`(F32 CPU 참고, 위치별 logits dump)
  - `gpu_probe --logit-rows N` / `--nll`
  - `f32_reference.py`(top-K + logsumexp fixture, 절단 KL, top-1, NLL, `--against` paired bootstrap 95 % 구간)
  - W2 하네스의 지표 설계(top-k + logsumexp 저장, paired bootstrap)와 같은 구조다. 따라서 **W2 하네스 코드를 그대로 쓰기보다 이 도구들을 쓰고, 코퍼스와 판정 기준만 W2와 맞추자고 제안한다.**
  - W2 코드는 브리프상 `tools/qwen27b/rq_quality/`에 있고 27B 엔진에 묶인다. 지표 계산 부분을 공통으로 뺄지는 W2가 끝난 뒤 감독이 정한다.
- **코퍼스:** W2의 32K 평가 코퍼스(해시 고정)를 그대로 쓴다. Flash-Next 토크나이저로 다시 토큰화하고 해시를 따로 기록한다. 회전 기저 imatrix의 보정 코퍼스는 W4 것과 같은 규칙으로 분리한다.
- **에이전트 벤치:** pibench는 재현할 수 없다. 대신 CLAUDE.md 규칙대로 `tests/functional/pi_agent.py`(리플레이), `agent_long.py`, tool-call suite, `test_long_context.py`, D4 한국어 세트를 쓴다. 속도는 BENCHMARKS.md의 HTTP 측정(greedy, MTP 적응)과 요청·단계별 5 %·10 ms 판정을 따른다.
- **`ssm_out` 민감도:** 27B E2 결과를 먼저 본다. Flash-Next F4에서는 `ssm_out`, `attn_output`, 헤드를 각각 Q8_0과 4비트로 두는 sweep을 혼합 기준 대비 KL과 top-1로 잰다.

## 9. 공통 코드 재사용 시 막히는 곳

| 공통 부분 | 막히는 곳 | 제안 |
|---|---|---|
| `tools/rotquant/convert.py` | 전체 회전 BF16을 중간 파일로 쓴다. `llama-quantize`로 파일 전체를 다시 양자화한다. 입력 전체를 해시한다(7.3절) | graft 모드(`--base`, `--source`, 텐서별 스트리밍 양자화)를 공통에 추가한다. 27B도 실험 GGUF를 만드는 시간을 줄일 수 있다 |
| GGUF 표시 규칙 v1 | (1) `nextn.eh_proj`는 한 텐서 안에 서로 다른 입력 두 개를 이어 붙였다([emb \| hidden]). "텐서 하나 = 그룹 하나"가 아니다. (2) `hc_xn_in`은 양자화 사본만 회전하고 F32 `inject`는 그대로 둔다. "같은 입력이면 같은 그룹" 규칙과 어긋난다. (3) K = 320(`hc_*_up`)은 회전할 수 없다 | (1) 사이드카 단계에서 `.rot` 앞뒤 절반을 각 그룹 부호로 두는 규칙을 v1에 덧붙일지 정한다. 아니면 eh_proj를 회전하지 않는다. (2) 그룹을 "같은 양자화 활성화 사본을 읽는 행렬"로 정의한다. float elementwise 소비자는 그룹 밖이라고 문서에 적는다. (3) 회전하지 않음(v1 범위) |
| `src/core/quant/rotation.{hpp,cpp}` (W3) | 아직 없다. W3이 27B 기준으로 만든다. Flash-Next 바인딩은 `TensorRef`/`Binder`가 따로 있다 | 공통 검사 API가 "그룹 표 + 텐서 이름 → 부호 벡터"만 다루고 27B 타입에 의존하지 않도록 W3 리뷰 때 확인한다 |
| `src/core/hip/detail/hadamard.hpp` (W3) | 아직 없다. Flash-Next 융합 지점은 wave32 epilogue(`HcMixEpilogue*`, `GatedDeltaNet`)와 mmq 양자화기(`qfn_mmq_quantize_q8_1`, ggml 계열 코드)다 | device 함수가 128점을 한 wave(32 lane × 4원소) 안에서 처리하고, FP32 레지스터 입출력과 LDS 없는 변형을 제공하면 둘 다 쓸 수 있다. W3 설계 때 요구 사항으로 전달한다 |
| 27B 커널 | decode 활성화 형식이 다르다(FP32 대 Q8_1). Q4_K prefill에 sidecar가 필요하다. 27B small-batch는 27B shape 전용이다 | 6.3절 |
| W2 하네스 | BF16 기준이 불가능하다. 27B 엔진 전용이다 | 8.2절 |
| imatrix (W4) | `imatrix_unsloth`는 원래 기저다. 회전 기저에는 쓸 수 없다. W4 수집기는 27B forward 훅이다 | Flash-Next forward에 같은 훅을 단다(그룹별 FP32 활성화 위치는 5.1절). 원래 기저 imatrix는 E1식 대조군에 unsloth 것을 그대로 쓸 수 있다[짐작: 텐서 이름이 대응하는지 F0에서 확인] |

## 10. 구현 단계 제안 (27B R5 뒤, 사용자 확인 후)

| 단계 | 27B 대응 | 내용 | 통과 기준 |
|---|---|---|---|
| F0 헤더와 계획 확정 | — | BF16, UD, MTP 헤더 dump(텐서, 타입, 형상, 분할). `config.hpp` 값과 3절 바이트 표를 확인한다. 2–4절의 [짐작]을 [확인]으로 바꾼다 | 헤더 표가 이 문서와 맞는다. 다르면 문서를 고치고 보고한다 |
| F1 컨버터 | R1 | graft 모드, `tools/qwen-flash/rotquant_groups.py`, CPU 테스트(합성 + 실제 텐서 표본 `W'·(H·D·x) ≈ W·x`), `ggml_quantize_chunk`와 `llama-quantize`의 비트 동일 확인(27B 텐서 하나) | 단위 테스트 통과. `--dry-run` 표. 비회전 graft(레시피 = UD 타입)의 출력이 UD와 텐서 바이트 단위로 같다 |
| F2 로더와 회전 경로 | R2 | Flash-Next 바인딩에서 공통 검사를 호출한다. 부호 업로드. 1단계 그룹의 FP32 생산 지점(5.1절)에 회전을 융합한다. MTP 헤드 경로. 회전을 지원하지 않는 경로는 거부한다(CPU 참고 경로 포함, 또는 참고 경로에도 회전 구현) | ① FWHT op 테스트 ② **회전 BF16-dense 혼합 파일이 비회전 혼합 파일에 충분히 가깝다.** decode, verify, prefill 각각에서 KL이 (Q8 dense ‖ BF16 dense)의 1/10 이하이고 top-1이 그 이상 ③ 같은 파일에서 MTP 수락률이 회전 전과 신뢰구간 안에서 같다 ④ 비회전 UD는 비트 단위로 이전과 같다(기존 logits 동일성 테스트) ⑤ 거부 테스트 |
| F3 4비트 dense 커널 | R0b + E9 | 6.3절 (a)/(b)/(c) 후보와 IQ4_XS/Q4_K/Q4_1을 Flash-Next 형상(16384×2560, 13312×2560, 2560×6144, 248320×2560)에서 AR, verify 1–8행, prefill로 잰다(`tools/qwen-flash/` 벤치). 고른 커널에 op 테스트를 추가한다. 융합 prefill 커널은 4비트용으로 바꾸거나 일반 경로로 보낸다 | AR 실효 대역폭이 상한의 80 % 이상(27B IQ4_XS 수준). verify 8행이 Q8_0보다 느리지 않다. prefill pp2048이 Q8_0 대비 5 % 넘게 느려지지 않는다 |
| F4 레시피 실험 | R4 (E1, E2, E6) | 혼합 기준 대비: (i) 1단계 그룹 4비트 ± 회전 (ii) `ssm_out`, `attn_output`, 헤드를 각각 Q8_0/4비트 (iii) 회전 기저 imatrix 있음/없음. 같은 날 HTTP 속도(d0/d32K, MTP 수락률) | 결과표(KL, NLL, top-1 신뢰구간, 토큰당 시간, 수락률). 채택 기준은 27B D4처럼 사용자와 정한다 |
| F5 실제 작업과 채택 | R5 + D4 | `pi_agent.py` 리플레이, tool-call, long-context, 한국어 세트. 기준선은 UD-Q4_K_XL + shared-Q8_0 MTP다 | D4 게이트 통과. decode 이득이 5 %·10 ms 기준으로 확인된다. 채택하면 기본 배포 파일과 문서를 고친다 |
| F6 (선택) 2·3단계 그룹 | E6, E7 | `hc_xn_in`, `shexp_down_in`, `ple_in`, MTP 사이드카 dense. 그다음 `ffn_in`과 `down_exps_in`(전문가 다시 양자화) | F4와 같다. 전문가 변경은 KL이 아니라 paired NLL로 판정한다(#474) |

## 11. 사용자 결정이 필요한 것

| 항목 | 시점 | 선택지 |
|---|---|---|
| Flash-Next 착수와 범위 | 27B R5 뒤 | 1단계 그룹(`attn_in`, `ssm_out_in`, `attn_out_in`, `head_in`)만, 또는 HC와 공유 전문가까지 |
| 품질 기준 파일 | F2 전 | UD 전문가 + dense BF16 혼합(제안), 또는 UD 자체(#474 방식), 또는 F32 CPU 참고 |
| 4비트 dense 타입과 커널 방식 | F3 결과 뒤 | IQ4_XS / Q4_K / Q4_1. 27B FP32 GEMV 재사용 대 mmvq Q8_1 커널. Q4_0/Q4_1을 받는다면 27B 결정 항목과 함께 정한다 |
| decode 산술 변경 허용 | F3 | 27B FP32-활성화 GEMV를 쓰면 Q8_1 → FP32로 바뀌어 기존 비트 동일 fixture가 바뀐다 |
| 포크 코드 차용 | F3 | MIT. 구조만 참고할지, 일부 코드를 가져올지(출처를 기록한다) |
| UD 받는 위치와 시점 | F0 전 | 외장에 먼저 받고 GPU 단계에 루트로 복사(제안), 또는 GPU 단계 직전에 루트로 바로 받기 |
| v1 규칙 보강 | F1 | eh_proj 이중 그룹 `.rot`, "양자화 사본 기준 그룹" 정의 |
| 공통 graft 모드 | F1 | 공통 `tools/rotquant/`에 추가해 27B도 쓸지 |

## 12. 비용 추정 [짐작]

### 디스크

- 외장: BF16 원본 등 약 362 GB(진행 중). UD를 먼저 받으면 +114 GB.
- 루트:
  - UD 기준선 103.7 GiB + MTP 2.6 GiB
  - 후보 1개 약 102 GiB
  - 혼합 기준 파일 약 107 GiB
  - 후보와 혼합 기준은 동시에 둘 수 없다(하한 150 GiB). 그래서 품질 창 순서는 "혼합 기준 실행 → 지움 → 후보 생성 → 후보 실행"이다. 기준 logits는 top-K fixture(수십 MB)로 저장하므로 기준 파일을 다시 둘 필요가 없다.

### CPU (감독 허락 필요)

| 작업 | 예상 비용 |
|---|---|
| F0 헤더 읽기 | 가볍다 (분할마다 수 MB) |
| graft 1회 | USB 읽기 약 8 GB + 텐서별 양자화(포크 기록: 145개 텐서 263초) + 루트 쓰기 약 110 GB. 수 분에서 15분 |
| imatrix 수집 | GPU 창 1회 |
| 빌드 | 27B와 같다 |

### GPU 창 (창 하나 약 45–60분 가정)

| 단계 | 창 수 |
|---|---|
| F2 | 1–2 (회전 동일성 ②③, 비회전 동일성 ④) |
| F3 | 1 (커널 벤치) |
| F4 | 3–5 (기준 fixture, 레시피별 품질과 속도, imatrix) |
| F5 | 1–2 (실제 작업 게이트, A/B 속도) |
| **합계** | **약 6–10개** |

- 모델 로드(약 111 GB)에 창마다 몇 분이 든다.
- 모든 창은 ai.tools 플랫폼을 끈 상태에서만 한다. 모델이 약 111 GB라 플랫폼이 켜져 있으면 메모리가 모자란다.

## 13. EXL3 이식 타당성 (W7 추가 조사, 2026-10-29)

배경: PLAN "외부 기준과 목표"(`76b5daee`). 이 절은 읽기만 하고 조사한 것이다.

- GitHub는 `gh`로 읽었고, HF는 API 메타데이터와 모델 카드, 카드에 실린 차트 이미지만 봤다.
- 가중치는 받지 않았고, 코드를 빌드하거나 실행하지 않았다.
- 표기는 앞 절과 같다([확인] / [짐작]).

### 13.0 요약

- **라이선스 [확인]:** 세 저장소 모두 MIT다. gufo(MIT)로 가져오는 데 법적인 걸림은 없다. 저작권 고지를 남기면 된다. 다만 팩 가중치(Yamz, turboderp Flash-Next)는 Qwen Community License 1.0이다.
- **Yamz 팩은 우리 손으로 다시 만들 수 없다 [확인].** Kyojin이 쓰는 Yamz 팩의 변환 도구는 공개되지 않았다. 우리가 직접 만들 수 있는 것은 upstream `convert.py`(PyTorch, GPU 양자화기)로 만든 팩이다.
- **속도 차이의 출처 [짐작, 근거는 13.4절].**
  - Kyojin 자체 기록에서 Flash-Next plain decode는 27.3 tok/s다(3.47 bpw 팩). 지금 gufo AR은 28.3 tok/s다. 둘은 비슷하다.
  - 바이트는 EXL3이 대략 절반이다. 그런데 커널 실효 대역폭도 gufo의 절반 정도다(EXL3 dense GEMV는 피크의 55–65 %, "실용 단일 커널 상한" 약 135 GB/s. gufo는 27B R0b에서 80–91 %).
  - 그러니 52 대 35의 차이는 주로 **투기 디코드**(MTP + n-gram "mix" 규칙 + 2단 draft head)에서 온다. 그리고 비교에 쓰인 gufo 숫자 35는 v0.8.0 기준이다.
- **품질 [확인: turboderp 차트].** 같은 bpw에서 EXL3의 KL이 GGUF보다 확실히 낮다. Flash-Next에서는 EXL3 4.05 bpw(KL 0.0066)가 UD-Q4_K_XL(약 5.2 bpw, 0.0084)보다 낮다.
  - Kyojin 자체 측정(Yamz 95 GB 0.0216 대 gufo 0.0156)과 순위가 반대다. Yamz 팩이 더 낮은 bpw 쪽이거나 레시피가 다른 것으로 보인다[짐작].
  - 그래서 "gufo 품질 유지 + 바이트 감소"는 EXL3 형식으로 가능해 보인다.
- **이식 규모 [짐작]:** 형식 리더, 커널(trellis 디코드 GEMV/다중 행/MoE/GEMM, 입출력 Hadamard), 두 모델의 바인딩과 실행 경로, 그리고 양자화기 운용(GPU, HF safetensors 원본 필요)까지다. 회전 + k-quant 경로보다 한 자릿수 크다.
  - 바이트 감소만 보면 회전 + 3–4비트 k-quant로도 상당 부분을 얻을 수 있다(R4 결과 대기).
  - 투기 디코드 개선은 형식과 독립이다. 그래서 먼저 그쪽(드래프트 정책, n-gram 혼합, verify 비용)을 따로 보는 것이 맞다고 본다.

### 13.1 출처와 라이선스 [확인]

| 저장소 | 커밋 / 릴리스 | 라이선스 | 내용 | gufo로 가져오기 |
|---|---|---|---|---|
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3) | v1.6.5 `cdc785db`(2026-10-25). 릴리스 자산에 `exllamav3-1.6.5+rocm10.0.torch2.13.0-cp31{2,3,4}` wheel이 있다 | MIT (Copyright 2025 Turboderp) | EXL3 형식, 양자화기 `convert.py`, CUDA 커널 `exllamav3_ext/quant/*`, ROCm 커널 `exllamav3_ext/rocm/*` | 가능(MIT 고지 유지). ROCm 쪽은 vendored `rocm/vendor/rocm_wmma_gemm`(MIT, Copyright 2024 Adel Johar)도 포함한다 |
| [Yamz-Labs/kyojin](https://github.com/Yamz-Labs/kyojin) | 1.8.3 `2132bd24`(2026-10-29). 벤치 수치는 1.5.10 `db6103b`(`doc/figures.json`) | MIT. `LICENSE`는 Turboderp 것 그대로이고, 추가분도 MIT라고 `doc/credits.md`에 적혀 있다. `exllamav3/vendor/fla`도 MIT | exllamav3 포크(Python + `exllamav3_ext/` 전체). gfx1151 커널, Qwen/GLM/MiMo 서버, 벤치 도구. **팩을 만든 양자화 파이프라인은 저장소에 없다**(credits.md, 팩 카드 "The conversion tooling is not published") | 코드는 가능(MIT). 팩 레시피는 재현할 수 없다 |
| [vcruz305/exllamav3-amd](https://github.com/vcruz305/exllamav3-amd) | `e0ec553b`(2026-10-07) | MIT (Turboderp 고지) | 첫 gfx1151 이식. 기반은 [sdougbrown/exllamav3](https://github.com/sdougbrown/exllamav3)(MIT, upstream 포크, gfx12 WMMA decode). RDNA3.5 WMMA GEMV, 그룹 MoE, `v_sad_u8` trellis 디코드, int8 mixer, LDS bank conflict 수정 | 가능. Kyojin `README.strix-halo.md`에 같은 내용이 들어가 있다 |

- 세 저장소 모두 PyTorch C++ 확장(`bindings.cpp`, `libtorch/`)이다. 커널 본문은 HIP로 옮길 수 있지만, 호스트 쪽(텐서 할당, 디스패치, 그래프)은 torch에 묶여 있어 그대로 쓸 수 없다[짐작: 파일 구성 기준].
- gufo에 넣을 때는 출처 URL, 커밋, 라이선스 고지를 `THIRD_PARTY` 또는 해당 파일 머리에 남긴다(gufo 기존 vendored mmq와 같은 방식, `mmq/VENDOR.md`).

### 13.2 EXL3 형식 [확인: `exllamav3/modules/quant/exl3.py`, `doc/convert.md`, README]

- **기반:** QTIP(Cornell RelaxML)의 간소화판이다. 가중치를 16×16 타일 단위로 trellis 부호화하고, 타일 하나는 `16·K`개의 uint16이다(K = 비트/가중치). half-integer K(예 4.5)는 `mul1` 코드북에서만 된다.
- **코드북:** `mul1`(기본), `mcg`, `3inst`. 모두 표를 읽지 않고 계산하는 procedural 코드북이다. int8 GEMV와 CPU 전문가 오프로드는 `mul1`을 요구한다.
- **회전:** **입력 쪽과 출력 쪽 모두** Hadamard를 쓴다.
  - `suh`(입력 부호, fp16)와 `svh`(출력 부호와 스케일, fp16)를 텐서마다 저장한다. 실행은 `had_r_128`(128점)이다.
  - `in_features`와 `out_features`가 모두 128의 배수여야 한다.
  - 실행 시 입력을 회전하고(`x' = H·diag(suh)·x`), 출력에도 역회전(`H`, `svh`)을 적용한다. 그래서 GEMV 앞뒤에 Hadamard 커널이 붙는다(RDNA에서는 3 launch, 또는 프롤로그와 에필로그에 융합).
- **비트 선택:** K는 1–8(정수, mul1이면 반정수도)이다.
  - `--bits`로 평균 bpw를 정하고, 텐서별 배분은 측정(Hessian, calibration) 기반 예산이다. `--hq`는 attention과 공유 전문가를 올린다.
  - `--recipe`로 텐서별 비트를 정할 수 있다. 헤드는 `--head_bits`(기본 6), MTP는 `--mtp_bits`(기본 4, 보정 없음), n-gram 표는 `--ngram_bits`다.
  - 임베딩은 기본이 비양자화이고, 256 그룹으로 회전·부호화하는 선택지도 있다.
  - 출력 채널 스케일(`--out_scales`)도 있다.
- **양자화 방식:** 층별 calibration forward로 입력 Hessian을 즉석 계산하고, LDLQ와 Viterbi trellis 부호화를 한다. 보정은 250행 × 2048토큰이 기본이다.
- **저장:** HF 구조 그대로 safetensors 분할(`model.safetensors.index.json`) + `config.json`의 `quantization_config` + 텐서별 `trellis`/`suh`/`svh`/(`mcg`|`mul1`)이다.
  - Flash-Next는 `ngram_embedding.safetensors`를 따로 둔다(팩 사이에 바꿔 쓸 수 있고, 보통 디스크에서 스트리밍한다).
  - Yamz 팩에는 추가로 `hc_gs_sidecar.safetensors`(633 MB, HC mixer int8로 보인다 — 짐작), `draft2s.st`(2단 draft head, 331 MB), `quantization_config.json`(97 MB, 텐서별 설정)이 있다.

**우리 회전과 `prism.hadamard` 비교:**

| 항목 | EXL3 | 우리 v1 (`gufo.rotation.*`) | upstream `prism.hadamard` (D8) |
|---|---|---|---|
| 회전 블록 | 128 | 128 | 128 |
| 위치 | 입력 + 출력 양쪽 (`suh`, `svh`) | 입력만 | 입력만. `y = W'·(H·(s·P·x))`, GDN 순열 P 포함 |
| 부호 | 텐서마다 따로 (fp16, 스케일 포함) | 입력 그룹이 공유 (±1) | 입력 폭마다 (W1이 확인 중) |
| 양자화 | trellis(QTIP), 16×16 타일, 1–8 비트 | 기존 GGUF 타입(K-quant, IQ) | 기존 GGUF 타입 |
| 실행 비용 | 행렬마다 입력 회전과 출력 역회전 | 그룹마다 입력 회전 한 번(생산 커널에 융합) | 우리와 같음 |
| 컨테이너 | safetensors + config | GGUF | GGUF |

- EXL3 회전은 행렬마다 독립이다. 출력 쪽 회전 때문에 matmul 출력 형식이 바뀌지 않는 대신, 행렬마다 추가 Hadamard가 붙는다.
- 우리 방식은 "그룹 공유 + 입력만"이라 activation 생산 커널에 한 번만 융합하면 된다.
- 두 방식을 같은 파일에 섞을 수는 없다. 형식 변환도 불가능하다. EXL3 가중치를 GGUF 타입으로 다시 양자화하면 이중 양자화가 된다.

### 13.3 커널 [확인: 파일 머리 주석과 문서 / 짐작: 비교]

| 커널 | upstream 1.6.5 `exllamav3_ext/rocm/quant/` | gfx1151 튜닝 상태 | gufo 대응 |
|---|---|---|---|
| decode GEMV (m = 1) | `exl3_gemv_rdna.cu`(+`exl3_gemv_kernel_rdna.cuh`, `exl3_gemv_tiles_rdna.cuh`): fp16 활성화 × trellis 디코드, `fdot2` 누적. CUDA의 cooperative 융합 대신 입력 Hadamard / GEMV / 출력 Hadamard 3 launch(일부 형태는 프롤로그·에필로그 융합). K 1–8, 세 코드북 모두 지원 | vcruz/Kyojin: WMMA GEMV(`v_wmma_f16_16x16x16_f16`), `v_sad_u8` trellis 디코드(−3–4 %), LDS bank conflict 수정 `EXL3_HIP_STG_PAD`(+10 %). **dense GEMV는 DRAM 피크(236 GB/s)의 55–65 %, lm_head 약 200 GB/s.** "단일 커널 실용 스트리밍 상한 약 135 GB/s"(`read_bw_vs_size.py`) | 27B: `gemv_quant.hip` FP32 활성화, R0b에서 IQ4_XS 89 %, Q4_K 82 %(DRAM read 240 GB/s 기준). Flash-Next: Q8_1 mmvq. **vcruz/Kyojin의 "135 GB/s 실용 상한" 주장은 우리 측정과 맞지 않는다.** 우리 커널은 200 GB/s 이상을 낸다 |
| verify 다중 행 (m = 2–8) | `exl3_gemv_multirow_rdna.cu`: 가중치를 한 번 읽어 M ∈ {2,4,8}행에 쓴다. 행 결과는 m = 1과 비트 동일. 입력 회전 1 launch + split-K dot 1 launch | Kyojin: "ms/forward = 32.4 + 9.04 × rows". 행이 늘 때 비용 대부분은 고유 전문가 수 증가에서 온다 | 27B small-batch wave64(27B shape 전용), Flash-Next Q8 mmvq 1–8행. R0b에서 27B verify 8행 Q8_0은 212 GB/s |
| MoE decode | `exl3_mgemv_rdna.cu`: (전문가 × N 타일) 단위 warp, 입력 회전은 프롤로그에, 출력 회전과 가중 합은 에필로그에 융합 | vcruz: grouped MoE prefill GEMV 131–135 GB/s(고유 전문가 바이트 기준). Kyojin: K6/K8 fused MoE(`tools/qwen/K6K8.md`, 실험) | Flash-Next `qfn_mmq_moe_gated_vec`(Q4_K, Q5_K), `mmvq` moe |
| prefill GEMM | `exl3_gemm_inner_rdna.cuh`: WMMA(`wmma_f32_16x16x16_f16`)로 재구성, 3단 파이프라인, split-K. dense GEMM 튜너(첫 실행 때, `EXL3_DENSE_GEMM_TUNE`). `exl3_moe_inner_rdna.cuh` 파이프라인 MoE | Kyojin Flash-Next prefill 8K 1621 tok/s | gufo `fec4759d` pp2048 1,801(d0)과 1,721(d8K)이 이미 이보다 높다(측정 방법은 다르다). prefill은 이식 이유가 아니다 |
| 기타 | HC mixer(`hc_mix.cu`, int8 `hc_mix_q8r.cuh`), GatedResidual `gr_mix`, GDN, PLE, n-gram | — | Flash-Next에 다 있다 |

- 정리 [짐작]: EXL3 커널의 강점은 **바이트당 품질**(trellis + 양쪽 회전)이다. 바이트를 읽는 효율은 gufo 커널이 높다.
- 그래서 EXL3 형식을 가져올 때 커널을 그대로 가져오면 gufo의 대역폭 효율을 잃을 수 있다. trellis 디코드를 gufo 커널 구조(FP32/Q8_1 활성화, 행 공유) 안에 다시 짜야 할 가능성이 크다.

### 13.4 속도는 어디서 오는가 (Kyojin 52 대 gufo 35) [짐작, 근거 표시]

| 요소 | Kyojin (Yamz 팩) | gufo | 근거 |
|---|---|---|---|
| 팩 크기 | 95.37 GB (n-gram 표 26.24 GB 포함, 표는 디스크 스트리밍) | 111.33 GB (PLE 표 IQ4_NL 약 28.8 GB, 디스크) | HF API sizes [확인]. gufo PLE 크기는 [짐작: 7.1절] |
| trunk 평균 bpw | 약 4 bpw[짐작: 비-n-gram 68 GB ÷ 약 1,290억 개]. 카드의 다른 문서에는 "qwen38-yamz-v1, 3.47 bpw"라는 기록도 있어(`tools/qwen/SERVE.md`) 버전마다 다를 수 있다 | 약 5.1 bpw (dense Q8_0 많음) | [짐작] |
| decode 토큰당 읽는 바이트 | 약 3–3.5 GB[짐작: 활성 약 66억 개 × 약 4 bpw + 헤드 6비트 + HC int8] | 약 6.1 GB (3절) | [짐작] |
| plain decode | 27.3 tok/s (3.47 bpw 팩, 1행 36.6 ms → 약 85–95 GB/s 실효) | AR 28.3 tok/s (`fec4759d`, 35.3 ms → 약 173 GB/s) | Kyojin `tools/qwen/SERVE.md` [확인], gufo BENCHMARKS.md [확인]. 실효 GB/s는 [짐작] |
| 투기 decode | 52.2(8K), card chat/prose/code 52.8/47.5/73.2. MTP + n-gram mix 규칙 + `draft2s` 2단 draft head, `--ndt 7`(대부분 3까지) | v0.8.0 35.3(8K, Kyojin 측정). 우리 `fec4759d` 측정 mixed 36.9(d8K)–40.5 | 둘 다 [확인]. 측정 방법이 달라 직접 비교는 안 된다 |
| 실효 증가율 (투기/plain) | 약 1.7–1.9× (chat 42.2/27.3 등, SERVE.md) | 약 1.3–1.4× (36.9/28.3) | [짐작: 위 수치에서 계산] |

해석 [짐작]:

1. plain decode는 거의 같다. 바이트는 EXL3이 약 절반이고, 커널 효율은 gufo가 약 2배라 서로 상쇄된다.
2. 투기 decode의 차이(약 1.4×)는 **드래프트 정책과 verify 비용**에서 온다. Kyojin은 n-gram을 MTP와 섞고 2단 draft head를 쓴다. gufo도 컨텍스트 복사를 하지만 조건이 보수적이다(README: greedy 12, sampled 24토큰 일치).
3. verify 행 비용: Kyojin은 행마다 9 ms가 늘고 기본 32 ms다. #474의 gufo verify 1 step은 약 57.5 ms였다(3.15 토큰/step). 같은 단위로 재지 않으면 비교가 안 된다.
4. 따라서 "EXL3 형식 이식"만으로 52에 닿는다는 보장은 없다. 바이트가 절반이 되더라도 gufo 커널 효율을 유지해야 plain 약 45 tok/s 이상이 나온다(이론상). 그 위에 드래프트 개선이 따로 필요하다.

### 13.5 품질 (fidelity) [확인: 공개 차트와 표 / 짐작: 해석]

- **Kyojin 측정** ([benchmarks.md](https://github.com/Yamz-Labs/kyojin/blob/main/doc/benchmarks.md)): FP8 공식판 대비 200개 중립 텍스트, 3,084 위치.

  | 엔진 | 팩 | top-1 | KL | 반복 간 KL |
  |---|---|---|---|---|
  | Kyojin | 95 GB | 93.81 % | 0.0216 | 0.0008 |
  | Strata | 93.7 GB | 93.64 % | 0.0316 | 0.0074 |
  | gufo v0.8.0 | 111.3 GB | 95.36 % | 0.0156 | 0.0000 |

- **turboderp 공개 bpw–KL 차트.** Flash-Next 카드 `qb_kld.png`(HF `turboderp/Qwen3.8-Flash-Next-exl3` `8e1af104`). 자체 생성 in-domain trace(입력 14,496 + 출력 45,588 토큰), 원본 대비 KL. x축은 임베딩과 헤드를 뺀 bpw다.

  | 형식 | bpw | KL |
  |---|---:|---:|
  | EXL3 3.05 (H5 NG5) | 3.05 | 0.0174 |
  | EXL3-SC 3.18 | 3.2 | 0.0138 |
  | EXL3-SC 3.68 | 3.7 | 0.0087 |
  | **EXL3 4.05 (H6 NG6)** | 4.05 | **0.0066** |
  | EXL3-SC 4.18 | 4.2 | 0.0061 |
  | EXL3 5.05 | 5.05 | 0.0040 |
  | EXL3 6.05 | 6.05 | 0.0031 (노이즈 바닥 0.0026) |
  | UD-IQ4_XS | 약 4.05 | 0.0162 |
  | **UD-Q4_K_XL** | 약 5.2 | **0.0084** |
  | NVFP4 W4A16 | 약 4.8 | 0.0102 |

  - 이 차트로는 **EXL3 약 3.7–4 bpw가 UD-Q4_K_XL 품질과 같거나 낫다.** 27B 차트(`turboderp/Qwen3.8-27B-exl3` `d23340a1`)도 같은 경향이다: EXL3 5.00 bpw 0.0023, SC 4.00 H5 0.0062, 4.00 bpw 0.0082 대 UD-Q4_K_XL 0.0051, IQ4_XS 0.0101.
- **두 측정이 서로 반대인 이유 [짐작]:**
  - (1) Yamz 팩이 turboderp 4.05보다 낮은 bpw이거나 레시피가 다르다. 비-n-gram 크기는 비슷하지만 SERVE.md에는 3.47 bpw 기록도 있다.
  - (2) 기준(FP8 대 원본 BF16), 코퍼스(중립 텍스트 대 자체 생성 trace), 지표(top-20 절단 대 전체)가 다르다.
  - (3) Kyojin의 int8 HC mixer 같은 추가 근사가 있다.
- **고-bpw 팩으로 gufo 품질을 맞출 수 있나 [짐작]:**
  - 차트상 turboderp 4.05 bpw(팩 107.3 GB, n-gram 39 GB 포함)는 이미 UD-Q4_K_XL보다 KL이 낮다.
  - n-gram 표를 NG4(약 26 GB)로 바꾸면 약 95 GB가 된다. n-gram 표는 팩끼리 바꿔 쓸 수 있다고 카드에 적혀 있다.
  - 즉 **같은 95 GB 안에서 gufo 이상의 품질이 가능해 보인다.** 우리 하네스로 확인해야 한다.
- 메모리: 5.05 bpw(123 GB, n-gram 39 GB)는 trunk가 약 84 GB다. n-gram 표를 스트리밍하면 128 GB 장비에 들어갈 수도 있다[짐작: Yamz 95 GB 팩이 113 GiB를 쓴다고 했으므로 여유가 작다].

### 13.6 이식 범위와 위험 [짐작]

| 부분 | 내용 | 규모 | 위험 |
|---|---|---|---|
| 형식 리더 | safetensors 분할 + `quantization_config`(텐서별 K, 코드북) + `trellis`/`suh`/`svh` 텐서. 모델별 이름 대응(HF → gufo 역할)이 27B와 Flash-Next 각각 필요하다. gufo에는 safetensors 로더 선례가 있다(`src/models/qwen3_tts/loader.hpp`, `minimax_h3`) | 중 | 중. 팩마다 부가 파일이 다르다(Yamz `hc_gs_sidecar`, `draft2s.st`는 Kyojin 전용) |
| 양자화기 | upstream `convert.py`(PyTorch, GPU). README는 "70B+ 수 시간(고성능 소비자 GPU)"이라고 한다. **입력은 HF safetensors**(Flash-Next 360 GB, 우리가 받는 BF16 GGUF는 못 쓴다). ROCm 양자화: Kyojin 트리에 `quantize_tiles_rdna.cuh`가 있지만 gfx1151 양자화 경로가 검증됐는지는 확인하지 못했다. `doc/roc_hip_decode.md`는 "the quantizer"를 범위 밖으로 둔다 | 대 | **대.** gfx1151에서 Flash-Next(1,800억 개) 양자화 시간을 알 수 없다(수십 시간 가능). 디스크도 360 GB 원본 + work dir(출력 1벌). 외부 팩을 쓰면 이 단계를 건너뛸 수 있다(turboderp 팩은 레시피 공개, Yamz는 비공개) |
| 커널 | trellis 디코드(K 1–8, mul1), m = 1 GEMV, m = 2–8 verify, MoE gated/down decode, prefill GEMM과 MoE GEMM, 입출력 128점 Hadamard. 활성화는 fp16 | 대 (수천 줄, upstream rocm만 수십 KB 파일 10여 개) | 대. torch 의존 제거. gufo 대역폭 효율(80–90 %)에 맞추려면 재설계가 필요하다. 그래프 캡처와 결정성 기준(bit 동일 fixture)도 다시 세워야 한다 |
| 통합 (27B) | `src/models/qwen/` 바인딩, decode/verify(DFlash2)/prefill 경로에 새 weight 타입, 출력 쪽 Hadamard 에필로그 | 중–대 | 중. DFlash2 드래프터 헤드 공유 문제는 회전과 같다 |
| 통합 (Flash-Next) | 13개 계열(4절) 전부, 3차원 전문가, PLE/n-gram 표, MTP 사이드카 | 대 | 대. 현재 fused prefill 커널(Q8_0 shape 전용)을 모두 대체해야 한다 |
| 검증 | 형식 오라클(CPU 디코드), op 테스트, 품질 하네스, 속도 A/B | 중 | 중 |

**회전 + k-quant 경로와 비교:**

| 기준 | 회전 + k-quant | EXL3 이식 |
|---|---|---|
| 새 코드 | 컨버터(Python, 대부분 완료), 생산 커널에 FWHT 융합, 일부 4비트 dense 커널 | 형식, 양자화기 운용, 커널 일체, 두 모델 통합 |
| 양자화 | llama-quantize/ggml(CPU, 분 단위) | PyTorch GPU 양자화기(시간 단위), HF 원본 필요 |
| 바이트당 품질 | 4비트 이상에서 회전 이득은 작을 수 있다(PLAN 위험). 3비트 이하는 E3/E7 결과를 봐야 안다 | 공개 차트상 같은 KL에서 약 1–1.5 bpw 적다 |
| 커널 효율 | 기존 gufo 커널(80–91 %) | 지금 공개 커널은 55–65 %. 끌어올리려면 별도 작업 |
| 외부 호환 | llama.cpp `prism.hadamard`(D8) | exllamav3/TabbyAPI/Kyojin 팩 |

판단 [짐작]: 결정은 R4(E3: 3–3.5 bpw 회전)의 품질과 속도를 본 뒤에 내린다.

- 회전 k-quant 약 3.5 bpw가 UD-Q4_K_XL 품질에 근접하면, 이식의 품질 이득은 작고 비용만 크다.
- 근접하지 못하면 EXL3의 바이트당 품질 우위가 결정적이다. 그때는 "27B에서 형식 리더 + m = 1 GEMV 하나"의 작은 시범으로 gufo 커널 효율 안에서 trellis 디코드가 대역폭에 묶이는지부터 확인하자고 제안한다.
- 드래프트 정책 개선(n-gram 혼합, 2단 draft head)은 형식과 무관하므로 별도 작업으로 먼저 검토할 가치가 있다.

### 13.7 공개 Qwen3.8-27B EXL3 팩 [확인: HF API]

- **Kyojin/Yamz 27B 팩은 없다.** `yamz-labs` 작성자의 모델은 GLM, MiMo, Qwen3.8-Flash-Next, DeepSeek-V4.1, Flint, Nanbeige뿐이다.
  - Kyojin README의 "Qwen3.8-27B (EXL3 4.0 bpw, MTP + n-gram)" 측정(RTX 3090)은 팩 출처를 적지 않았다.
  - 테스트 코드 기본 경로는 `~/Models/Qwen3.8-27B-exl3`(`tests/hip_gemv_logits_oracle.py`)이다. turboderp 팩으로 보인다[짐작].
- **turboderp/Qwen3.8-27B-exl3** (Apache-2.0, main `d23340a1`, 브랜치별 분기):

  | 브랜치 | 커밋 | safetensors 합계 |
  |---|---|---:|
  | `3.50bpw` | `8351c54e` | 15.34 GB |
  | `4.00bpw` | `113cf7ab` | 16.86 GB |
  | `5.00bpw` | `a35e75a7` | 19.90 GB |
  | `6.00bpw` | `d32ba0bb` | 22.94 GB |
  | `SC_4.00bpw_H5` | `4acd9ad5` | 16.71 GB |
  | `SC_5.00bpw_H6` | `dfe2f4fd` | 19.91 GB |
  | `SC_6.00bpw_H6` | `9e4c23cf` | 22.95 GB |

  - 2.00, 2.50, 3.00과 SC 1.40–3.00, `_V*`(비전 양자화)도 있다.
  - main에는 `kld_table.json`, `recipe_*.yaml`, `cal_trace.*`, `measurement.json`이 있다.
- **turboderp/Qwen3.8-Flash-Next-exl3** (Qwen Community 1.0, main `8e1af104`):

  | 브랜치 | 크기 (n-gram 표) |
  |---|---|
  | `3.05bpw_h5_ng5` | 85.0 GB (32.6) |
  | `4.05bpw_h6_ng6` | 107.3 GB (39.0) |
  | `5.05bpw_h6_ng6` | 123.1 GB (39.0) |
  | `6.05bpw_h6_ng6` | 138.9 GB |
  | `4.18bpw_h6_ng8` | 121.8 GB (51.8) |

  - SC 1.66–3.68 브랜치도 있다.
- **Yamz Flash-Next 팩:** `yamz-labs/Qwen3.8-Flash-Next-EXL3-Yamz` main `247ef39b`, 95,371,323,041 B. Qwen Community 1.0.
- **우리 하네스로 같은 27B를 잴 수 있나:** 가능하다. turboderp 4.00/5.00 bpw(16.9/19.9 GB)를 받아 Kyojin이나 upstream exllamav3(ROCm wheel)으로 돌리고, 우리 W2 코퍼스 위치별 top-k logits를 뽑아 BF16 기준과 비교하면 된다.
  - 실행 환경(PyTorch ROCm, Kyojin 빌드)이 따로 필요하다.
  - logits dump 훅이 있는지 확인해야 한다. Kyojin `eval/model_diff.py`, `eval/glm_kld.py`가 있지만 아직 읽지 않았다.
  - 내려받기와 GPU 창은 감독이 정한다.

## 참고

- 이슈: [gufo-org/gufo#474](https://github.com/gufo-org/gufo/issues/474) (2026-10-07, 닫힘. 관리자가 Discussions로 안내했다)
- 포크: `zengjiajun0623/gufo` `gufo-opt/flash-next` `b3f4202e`, `gufo-opt/flash-next-experts`(검토하지 않았다. 전문가 Q4_1, IQ4_KS, IQ4_NL, IQ4_K 커널과 op 테스트가 있다)
- 소스:
  - `src/models/qwen38_flash_next/{config.hpp,weights.cpp}`
  - `kernels/rocm/{device_model.cpp,executor.cpp,batch.cpp,kernels.hpp,w8a8_wave64.hip,mmq/}`
  - `src/models/qwen/hip/kernels/{prefill_quant_gemm.hpp,gemv_quant.hip}`
  - `tools/rotquant/{convert.py,plan.py}`
- 문서: `docs/models/qwen3.8-flash-next/{README,EXPERIMENTS,QUALITY}.md`, `docs/cache-redesign/RFC.md`(메모리 여유)
