# X-A 결과: EXL3 형식 명세, CPU 참조, 27B 정확성 경로 (진행 중, 2026-10-30)

상위 문서: [X-A 브리프](XA-BRIEF.md), [EXL3 도입 계획](PLAN.md). 이 문서는 X-A(W8)가 쓴다.

## 상태

- **X0 완료.** CPU 부분 `f26f800d`, upstream 비트 대조 창 `xa-ref-a01` 통과(2026-10-30).
- **X1 완료(통과).** 매핑 `04ca9c54`, BF16 GGUF 4.00/5.00, 정확성 창 `xa-x1-a01` 통과(2026-10-30).

## X0: 형식 명세와 CPU 참조

- **명세:** [docs/EXL3-FORMAT.md](../../docs/EXL3-FORMAT.md). 기준은 upstream exllamav3 v1.6.5 `cdc785db`이고, 근거 파일:줄을 달았다.
- **코드:** `src/core/quant/exl3/`
  - `format.{hpp,cpp}`: trellis 상태 추출(정수 K와 반정수 K), 코드북 3종(3INST, mcg, mul1, FP16 RNE를 double로 재현), 16×16 타일 순열, 행렬 dequant(전체와 열 구간), 128점 Hadamard, FP64 참조 GEMV.
  - `checkpoint.{hpp,cpp}`: 모델과 무관한 safetensors 팩 리더. gufo `src/core`에는 공용 safetensors 리더가 없어(모델별 리더만 있음) 새로 만들었다.
- **CPU 자체 검사** `quant_exl3_format_test`(CTest, `cpu;core;quant;exl3`): 통과.
  - FP16 65,536개 왕복, 동률 처리
  - TileState 대 단순 비트 읽기(모든 K와 반정수 K, 무작위 타일)
  - 타일 순열
  - 코드북: 3 × 65,536 상태 전부의 출력 checksum을 독립 Python 재현(struct `e` 반올림)과 대조
  - Hadamard, 참조 GEMV 대 명시적 회전 가중치(상대 오차 1e-12 미만)
- **원본 대비 확인(명세 검증):** 4.00/3.00/5.00 팩의 3층 `k_proj`를 명세대로 numpy로 풀고 회전해, HF BF16 원본과 비교했다.

| 팩 | K | 상대 오차 | 상관 |
|---|---:|---:|---:|
| 3.00 | 3 | 0.143 | 0.9898 |
| 4.00 | 4 | 0.072 | 0.9974 |
| 5.00 | 5 | 0.037 | 0.9993 |

  - 비트가 하나 늘 때마다 오차가 절반이 된다. 그래서 비트 순서, 타일 순열, 코드북, suh/svh 쪽, Hadamard 순서가 맞다. 배치가 틀리면 오차가 1 근처가 된다.
- **upstream 비트 대조(창 `xa-ref-a01/xa-ref`, 2026-10-30 14:45, COMPLETE, health_PASS):**
  - 덤프: `dump_ref.py`(이미지 `exl3-eval:1.6.5-rocm100`, 라벨 `f26f800d`, gfx1151, exllamav3 1.6.5). `synth` 2 s, `pack` 11 s.
  - 대조: CPU에서 `quant_exl3_upstream_test`.
  - 원시 JSON: `…/xa-ref-a01/xa-ref/receipt/run/exl3_upstream_{synth,pack}.json`(sha256 `016611e2a609…`, `21957d0465df…`).

| 대상 | 결과 |
|---|---|
| 합성 27경우: K 1–8 × 3INST/mcg/mul1, K 1.5/2.5/3.5 mul1(256 × 128) | inner W' FP16 비트 **27/27 동일**(sha256). FP64 forward 대 upstream FP64 상대 오차 4e-16–1.1e-15. upstream FP16 런타임(`LinearEXL3.forward`) 대 FP64 오차 3.5e-4–8.3e-4(FP16 반올림 수준) |
| 실제 텐서 8개 | 비트 **8/8 동일**: 3.00 k_proj(K3), 3.50 k_proj(K4), 3.50 layers.10 down_proj(K3), 4.00 k_proj와 layers.0 down_proj(K4), 4.00 lm_head(K6, 5120 × 248320), 4.00 mtp.fc(K4), 5.00 k_proj(K5) |

  - 합성 JSON을 fixture `tests/core/quant/exl3_upstream_synth.json`으로 넣었다. 이제 CTest `quant_exl3_upstream_test`(`cpu;core;quant;exl3`)가 돈다. `quant_exl3_format_test`와 함께 2/2 통과다.
  - mcg와 3INST는 실제 팩이 없다. 합성 비트 대조로만 확인했다(XQ가 다른 코드북 팩을 만들면 추가).

## X1: 27B 정확성 경로

- **설계(감독 승인 2026-10-30):** 정확성 경로는 오프라인 BF16 GGUF다. EXL3 팩을 llama.cpp `qwen35` GGUF 배치의 BF16 GGUF로 바꾸고, 기존 gufo BF16 경로로 돌린다. 런타임 코드는 바꾸지 않는다.
  - 로드 시점에 BF16으로 풀면 호스트 54 GB와 장치 54 GB가 120 GB 통합 메모리에 겹친다. 파일 경로는 페이지 캐시 기반이라 FAKEQ에서 이미 돌았다(HIP 최대 51 GB).
  - 직접 로드(`.trellis`를 GPU에 둔 채)는 trellis를 직접 푸는 커널이 생기는 X2로 옮긴다. 토크나이저 메타데이터(HF `tokenizer.json` → GGUF 키)도 X2에서 정한다.
- **코드:** `src/models/qwen/exl3_weights.{hpp,cpp}` (llama.cpp `8ab29fae` `conversion/qwen.py`와 같은 배치)
  - EXL3 투영 → 유효 가중치 `diag(suh)·H·W'·H·diag(svh)`를 [out][in] BF16으로
  - linear attention V 헤드를 grouped에서 tiled 순서로: qkv의 V 행, z 행, alpha/beta 행, out_proj 입력 열, conv1d V 채널, A_log, dt_bias
  - norm +1(linear_attn.norm 제외)과 A_log → −exp(A_log)은 FP32로
  - `mtp.*` → blk.64 nextn(fc → eh_proj, pre_fc_norm_embedding/hidden → enorm/hnorm, norm → shared_head_norm, 층 텐서)
- **도구:** `tests/models/qwen27b/exl3_to_gguf.cpp`(`qwen27b_exl3_to_gguf`, EXCLUDE_FROM_ALL). `compare`는 BF16 GGUF와 텐서별로 대조하고, `write`는 단일 파일 BF16 GGUF를 쓴다(템플릿 GGUF의 메타데이터에서 `split.*` 제외, 텐서 순서와 타입은 템플릿과 같음).
- **전체 매핑 대조(EXL3 4.00 대 BF16 GGUF, 866 텐서, 실패 0):**

| 분류 | 개수 | 내용 |
|---|---:|---|
| EXACT | 350 | 바이트 동일: 임베딩, 모든 norm(+1), dt_bias, conv1d(V 재정렬 포함), ssm_norm |
| NEAR | 107 | ssm_alpha/beta 96개: 팩이 원본 BF16을 F16으로 저장해 작은 값의 FP16 반올림 차이(rel 약 1e-8). ssm_a 11개: −exp의 1 ULP 차이 |
| QUANT | 409 | EXL3 양자화 오차. 본문 K=4는 rel 0.067–0.086(상관 0.996–0.998), lm_head(K=6)는 0.020 |

  - V를 재정렬한 텐서(qkv, gate, ssm_out)의 오차가 나머지와 같다. 재정렬이 틀리면 오차가 1 이상이 된다.
  - **`blk.64.nextn.eh_proj`(mtp.fc)는 rel 0.224, 상관 0.976으로 다른 텐서보다 크다.** 짐작: `mtp_bits` 4를 보정 없이 양자화했기 때문이다(W7 조사의 "MTP 4비트, 보정 없음"). 절반이 뒤바뀌었다면 상관이 0 근처여야 한다. 27B 대상 forward는 이 텐서를 쓰지 않는다.
- **BF16 GGUF(완료 2026-10-30):** `w8-exl3-20261029/x1-gguf/`(T5)에 썼다. 폴더마다 `PROVENANCE-<bpw>.json`이 있다.

| 팩 | 파일 | 크기 | sha256 | 시간 |
|---|---|---:|---|---:|
| 4.00 | `Qwen3.8-27B-EXL3-4.00bpw-BF16.gguf` | 54,657,735,424 | `079cf6ab19559cfc…` | 725 s |
| 5.00 | `Qwen3.8-27B-EXL3-5.00bpw-BF16.gguf` | 54,657,735,424 | `01053e2d3b9ebde4…` | 795 s |

  - 8스레드, 이미지 64849ca6ec7b, 도구 `04ca9c54`.
  - 확인: 쓴 4.00 파일을 gufo `GgufReader`로 다시 열어 `compare`로 대조했다(blk.3 11개 텐서 EXACT).
- **정확성 창(`PLAN-EXL3-X1.json`, `xa-x1-a01/xa-x1-4.00` 17:00–17:51, `xa-x1-5.00` 17:51–18:44, 둘 다 COMPLETE, health_PASS, 한 번씩):**
  - 수집기는 고정 `b5219f1b`(이미지 64849ca6ec7b)다. R0, FAKEQ와 같다.
  - 지표는 `metrics.py score`/`compare`(bootstrap 2000, seed 20261029, 블록 256)로 냈다. 보고는 `storage/evaluations/rq-r0-20261029/metrics-w8/x1-*.json`, `compare-x1-*.json`에 있다.

**엔진 차이: KL(ExLlamaV3 같은 팩 ‖ gufo-EXL3). 기준은 중앙값 ≤ 1e-4, 평균 ≤ 1e-3.**

| 팩 | 경로 | n | KL 평균 [토큰 95 %] | 중앙값 | 최대 1개 제외 | top-1 | 뒤집힘, KL > 1 | 판정 |
|---|---|---:|---|---:|---:|---:|---|---|
| 4.00 | prefill | 32,760 | 2.38e-4 [1.40e-4, 3.79e-4] | 2.98e-5 | 1.82e-4 | 0.9937 | 208, 0 | 통과 |
| 4.00 | decode | 4,096 | 1.18e-4 [0.86e-4, 1.62e-4] | 3.16e-5 | 1.05e-4 | 0.9949 | 21, 0 | 통과 |
| 5.00 | prefill | 32,760 | 4.28e-4 [1.70e-4, 8.27e-4] | 2.94e-5 | 2.61e-4 | 0.9929 | 233, 3 | 통과 |
| 5.00 | decode | 4,096 | 0.99e-4 [0.74e-4, 1.34e-4] | 3.17e-5 | 0.84e-4 | 0.9941 | 24, 0 | 통과 |

  - 참고로, 같은 엔진끼리의 바닥(ExLlamaV3 BF16 대 gufo BF16, W8)은 평균 6.9e-4, 중앙값 2e-5였다. X1 값은 이와 같은 크기다. 엔진 차이의 원인은 gufo의 BF16 반올림 유효 가중치 대 ExLlamaV3의 FP16 활성화와 FP16 logits로 본다[짐작].
  - 5.00 prefill의 KL > 1 세 위치는 chat 창 6이다(예: 위치 405, KL 5.5, 뒤집힘). 위치에 민감한 chat 위치라는 RQ2B 관찰과 같은 유형이다[짐작].

**BF16 기준 KL(prefill, `rq0-bf16` 대비)이 W8 ExLlamaV3 값과 같은지**

| 팩 | gufo-EXL3 KL | W8 ExLlamaV3 KL | paired 차이(gufo − ExLlamaV3) [토큰 95 % / 블록 95 %] | top-1 차이 | ΔNLL 차이 |
|---|---|---|---|---|---|
| 4.00 | 18.86e-3 [17.33, 20.54] | 18.84e-3 | +0.02e-3 [−0.51, +0.65] / [−0.54, +0.65], 구분 안 됨 | −0.0006, 구분 안 됨 | +2.3e-3 [+1.6, +3.2] |
| 5.00 | 7.79e-3 [6.23, 9.85] | 7.64e-3 | +0.14e-3 [−0.08, +0.38] / [−0.07, +0.41], 구분 안 됨 | −0.0002, 구분 안 됨 | +1.4e-3 [+0.5, +2.2] |

  - KL과 top-1은 W8 값과 신뢰구간 안에서 같다. 통과다.
  - ΔNLL만 gufo 쪽이 1.4–2.3e-3 nats/token 크다(구분됨). 같은 크기의 엔진 차이(BF16 대 FP16 경로)로 보이며 양자화 차이보다 한 자릿수 작다[짐작]. X2 커널(FP16/FP32 활성화, 가중치를 BF16로 다시 반올림하지 않음)에서 다시 본다.

**GGUF fingerprint(RQ2H 방식):**
  - `build/xa-gpu-test`의 `qwen27b_target_test --fingerprints-only`를 UD-Q4_K_M으로 돌렸다(27줄).
  - 결과는 `rq2h/units`의 같은 로그와 **바이트 단위로 같다**(`cmp`). EXL3 코드가 기존 GGUF 경로 출력을 바꾸지 않았다.

**판정: X1 통과.** 4.00과 5.00 모두 엔진 차이가 기준 안이고, BF16 기준 KL이 W8과 같으며, GGUF 경로 fingerprint가 그대로다. 다음은 X2(직접 로드와 trellis 커널, X-B)와 토크나이저 메타데이터 결정이다.

## X2a 설계 메모 (2026-10-30, 읽기만 하고 쓴 초안, X2b와 합의 전)

목표([XB-BRIEF.md](XB-BRIEF.md) X2a): gufo 27B가 EXL3 팩을 BF16으로 풀지 않고 읽는다. `trellis`/`suh`/`svh`는 압축된 채로 GPU에 둔다. 정확성용 기준 경로(단순 dequant 커널 + 기존 GEMM)를 만들고, 이를 X2b 커널의 비교 기준으로 쓴다.

### 1. 메타데이터: (b) GGUF 사이드카를 고른다

**방식:**
- 팩 디렉터리 안에 텐서 없는 GGUF `gufo.gguf`를 둔다.
- 내용은 같은 모델 GGUF의 KV 전부(`split.*` 제외: 하이퍼파라미터, `tokenizer.ggml.*`, `tokenizer.chat_template`)와 새 키 `gufo.exl3.checkpoint = "."`(팩 디렉터리, 사이드카 기준 상대 경로)다.
- 텐서는 0개다.

**이유:**
- **토큰 id 동일성이 구성상 보장된다.** 지금 gufo의 토크나이저(`QwenTokenizer::CreateFromGguf`), 채팅 템플릿 검사(`QwenChatTemplate::ValidateGgufTemplate`), 설정(`GgufReader::ExtractModelConfig`)은 모두 GGUF 메타데이터에서 만들어진다. 같은 KV 바이트를 쓰면 GGUF 경로와 토크나이저 출력이 같을 수밖에 없다.
- 반면 (a) HF `tokenizer.json` 경로는 BPE 어휘·merges·token_type·pre-tokenizer(`qwen35`)·특수 토큰을 다시 구현해야 하고 동일성 검사가 따로 필요하다. 이는 별도 작업으로 미룬다.
- **CLI를 바꾸지 않는다.**
  - 지금 CLI 진입점(`src/cli/{prompt,bench,serve}`, `rq_logits`)은 모두 "GGUF 경로 → `GgufReader` → `QwenGpuModel/Executor::CreateFromGguf`"다.
  - 사이드카 경로를 `--model`로 주면 `CreateFromGguf`가 `gufo.exl3.checkpoint` 키를 보고 EXL3 로드로 가지를 친다. 파일 형식은 로더가 판단한다. 옵션은 더하지 않는다.
- **만들기:** `qwen27b_exl3_to_gguf metadata EXL3_DIR OUT.gguf TEMPLATE.gguf`(X1 도구에 모드 추가)로 만든다.
  - 템플릿 KV를 복사하고 키 하나를 더한다.
  - 일관성 검사:
    - config.json(text_config) 대 GGUF 하이퍼파라미터: layers, hidden, heads, ssm 크기, vocab
    - tokenizer.json 어휘 대 `tokenizer.ggml.tokens`: 같은 토큰열
  - 틀리면 이름을 들어 거부한다.
- **단점:** 새 모델마다 같은 모델의 GGUF(메타데이터만)가 필요하다. 27B에는 이미 있다. Flash-Next도 unsloth GGUF가 있다.

### 2. 디바이스 텐서 구조 (X2b와 맞출 인터페이스)

```cpp
// src/core/quant/exl3/device.hpp (X2a)
namespace gufo::quant::exl3 {
struct DeviceLinear {                 // one EXL3 projection y = x W, W [k][n]
  const std::uint16_t* trellis;       // device; [k/16][n/16][rate.TileWords()] uint16, file order
  const std::uint16_t* suh;           // device FP16 [k]
  const std::uint16_t* svh;           // device FP16 [n]
  std::uint32_t k, n;                 // in_features, out_features (multiples of 128)
  BitRate rate;                       // bits 1..8, half (1.5/2.5/3.5)
  Codebook codebook;                  // kMul1 for the turboderp packs
};
}
```

- **배치:** 파일 그대로다. 타일 순서는 [k/16][n/16]이고 타일 안은 `16·K` uint16 연속이다. upstream RDNA 커널(`exl3_gemv_*`, `reconstruct`)이 읽는 순서와 같다.
- **정렬 (측정):** safetensors의 텐서 오프셋은 정렬되지 않는다. 5.00 팩의 trellis 409개 모두 16 B 정렬이 아니다. 그래서 mmap 주소를 그대로 쓰지 않는다.
  - 모든 trellis/suh/svh를 장치 arena 하나에 256 B 정렬로 복사한다.
  - 호스트 쪽은 로드 동안만 mmap을 쓴다.
- **gufo 바인딩:**
  - `QwenTensorRef`에 `const quant::exl3::DeviceLinear* exl3 = nullptr`을 더한다. `exl3 != nullptr`이면 EXL3 투영이다.
  - `data`는 trellis, `num_elements`는 k·n, `type`은 바꾸지 않는다(검사·분기 시 `exl3`를 먼저 본다).
  - DeviceLinear 배열은 `QwenModelWeights`가 소유한다(지금 `rotation_signs`, `folded_weights`처럼 shared_ptr).
- **분기 지점(X2b와 함께 손댈 곳, 감독에게 순서 요청):**
  - decode GEMV(`quant_ops.hpp`의 type switch)
  - small-batch verify(`LaunchBatchedQuantGEMMFp32`)
  - prefill GEMM 경로(`gemm_route.hpp`의 `QwenGemmRoute`)
  - 출력 헤드(`LaunchQwenOutputHead`)
  - DFlash2 드래프터가 공유하는 타깃 헤드·임베딩(`dflash_weights.cpp:299`)
- **수치 계약(커널 공통):**
  - `y = svh ⊙ H_n((H_k(suh ⊙ x)) · W')`. H는 128점 자연순 Sylvester / √128이다. W'는 trellis의 FP16 값이다.
  - 활성화 x는 FP32, 출력 y는 FP32다. 가중치를 BF16 등으로 다시 반올림하지 않는다.
  - 커널 단위 정답은 X0 CPU 참조(`GemvReference`)다.
- **입력 공유 문제:** suh가 텐서마다 달라, 같은 입력(qkv/gate/alpha/beta, q/k/v, gate/up)이라도 `H(suh ⊙ x)`를 공유할 수 없다.
  - 기준 경로는 텐서마다 회전 커널을 한 번씩 부른다.
  - 융합 방식(GEMV 프롤로그에서 블록마다 128점 FWHT 계산 등)은 X2b가 정한다.

### 3. 양자화되지 않은 텐서

- **변환:** X1 매핑(`exl3_weights`)이 GGUF 배치대로 만든다. 임베딩 BF16, norm(+1) F32, conv1d/A_log/dt_bias(V 재배열) F32, `in_proj_a/b` → `ssm_alpha/beta` BF16.
- **업로드:** 호스트 버퍼 하나에 모아 weight region 하나로 올린다(`folded_weights` 패턴). 그 밖의 바인딩은 `LoadFromGguf`의 이름 대응을 그대로 쓴다.
- **mtp.\*:** 27B 대상 forward가 쓰지 않으므로 X2a에서는 올리지 않는다.
- **메모리 (5.00, 결정 필요):**
  - trellis + suh/svh(본문 + lm_head)는 16.22 GB다. 임베딩 BF16 2.54 GB를 더하면 약 18.8 GB로, 목표인 UD-Q4_K_XL 상주 17.56 GB를 넘는다.
  - 로드 때 임베딩을 Q8_0(1.35 GB)이나 Q6_K(약 1.04 GB)로 바꾸면 약 17.6/17.3 GB가 된다. gufo 임베딩 커널은 둘 다 받는다(`weights.cpp` `SupportsTensorType`).
  - 임베딩 양자화는 품질에 영향이 거의 없다고 본다[짐작]. 하지만 KL로 확인해야 한다. 하기로 하면 X2a 통과 기준에 BF16 임베딩과 Q8_0 임베딩 두 경우를 넣는다.

### 4. 정확성 기준 경로 (X2a, 속도는 보지 않음)

`src/core/quant/exl3/reference.hip`에 커널 셋을 둔다.

1. **`DequantInnerF32`:** trellis 열 구간 → W'를 FP32 [n_chunk][k](GGUF 행 배치, 전치)로 쓴다.
   - 위치 하나당 스레드 하나: 상태를 추출하고 코드북 디코드(정수 연산, FP16 `__hfma`/`__hadd`).
   - X0 CPU 참조와 FP16 비트가 같은지 테스트한다(FP32는 FP16 값을 그대로 담는다).
2. **`RotateInput`:** `x' = H_k(suh ⊙ x)`, 행마다 FP32. 128점 FWHT, 공유 메모리.
3. **`RotateOutput`:** `y = svh ⊙ H_n(t)`, FP32.
4. **GEMM:** 기존 F32 가중치 경로(F32 GEMV / hipBLAS F32 GEMM)에 스크래치 W'를 넘긴다.
   - n 구간 단위로 한다. 스크래치는 k × n_chunk × 4 B ≤ 256 MB.
   - W'를 FP32로 쓰므로 가중치 재반올림이 없다. 그래서 X1의 ΔNLL 차이(1–2e-3, BF16 재반올림 짐작)를 바로 검증할 수 있다.

**비용(짐작):**
- decode는 토큰마다 전체 가중치를 FP32로 다시 푼다(약 100 GB 쓰기). 토큰당 약 0.5–1 s이므로 decode 4,096위치는 팩당 약 1시간이다.
- prefill은 창(4,096토큰)당 한 번 풀기라 R0와 비슷하다.
- 품질 창은 prefill 전부와 decode 일부(예: 창 2개, 1,024위치)로 제안한다.

### 5. 시험

- **CPU(CTest):**
  - 사이드카 생성과 검사(합성 GGUF, 불일치 거부)
  - `DeviceLinear` 바인딩 검사(shape·rate·codebook, su/sv 거부)
- **GPU(`tests/models/qwen27b/exl3_*`):**
  - `DequantInnerF32` 대 CPU W'(비트 동일, 합성 27경우 fixture 재사용)
  - `RotateInput/Output` 대 CPU `HadamardBlocks`(상대 1e-6)
  - 투영 하나 대 `GemvReference`
- **품질 창:**
  - 수집기를 다시 빌드한다(직접 로드 경로).
  - 기준 둘: ExLlamaV3 같은 팩, gufo BF16.
  - GGUF fingerprint(RQ2H 방식).
  - 상주 메모리 기록.

### 6. 파일과 순서 (제안)

- **X2a:**
  - `src/core/quant/exl3/{device.hpp,reference.hip,…}`
  - `src/models/qwen/exl3_weights.*`(로더 함수 `LoadFromExl3`)
  - `src/models/qwen/hip/model_loader.cpp`의 `CreateFromGguf` 분기
  - `tests/models/qwen27b/exl3_*`
- **공동(X2b와 순서 협의):**
  - `QwenTensorRef` 필드 추가(`state.hpp`)
  - 투영 분기 지점 다섯 곳
- **순서:**
  1. 인터페이스 합의(이 메모 2절)
  2. 사이드카와 로더(CPU)
  3. 기준 커널과 GPU 단위 시험
  4. 투영 분기(기준 경로)
  5. 품질 창
