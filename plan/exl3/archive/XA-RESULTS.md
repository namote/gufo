# X-A 결과: EXL3 형식 명세, CPU 참조, 27B 정확성 경로 (진행 중, 2026-10-30)

상위 문서: [X-A 브리프](XA-BRIEF.md), [EXL3 도입 계획](../PLAN.md). 이 문서는 X-A(W8)가 쓴다.

## 상태

- **X0 완료.** CPU 부분 `f26f800d`, upstream 비트 대조 창 `xa-ref-a01` 통과(2026-10-30).
- **X1 완료(통과).** 매핑 `04ca9c54`, BF16 GGUF 4.00/5.00, 정확성 창 `xa-x1-a01` 통과(2026-10-30).

## X0: 형식 명세와 CPU 참조

- **명세:** [docs/EXL3-FORMAT.md](../../../docs/EXL3-FORMAT.md). 기준은 upstream exllamav3 v1.6.5 `cdc785db`이고, 근거 파일:줄을 달았다.
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

## X2a 배선 (a72c89b0, 2026-10-30)

- 기준 GPU 경로: `src/core/quant/exl3/reference.{hpp,hip}`
- 분기점 하나: `src/models/qwen/hip/exl3_dispatch.hpp`의 `LaunchExl3Projection`. X2b 빠른 커널은 이 함수 뒤에 붙는다.
- 분기 지점 다섯 곳:
  - 디코드: `decode_step.cpp`
  - 배치 디코드와 검증: `batched_decode.cpp`의 `LaunchProjection`
  - prefill: `prefill_chunk.cpp`의 `gemm_weight`
  - `LaunchQwenOutputHead`
  - DFlash2 공유 헤드: `RunBlockGemm`, `ProjectBlock`
- arena: `QwenGpuArena::AllocateExl3` / `Exl3Scratch`. 세션마다 약 540 MB(기준 경로). X2b slab 크기는 W3와 합의한다.
- 로더 사이드카 분기:
  - 임베딩은 기본 Q8_0이다. 사이드카 키 `gufo.exl3.embedding="bf16"`이면 BF16으로 둔다(비교용).
  - EXL3 텐서는 장치 할당 하나에 256 B 정렬로 올린다.
  - `GetResidentBytes`에 EXL3 바이트를 포함한다.
- GPU 창: `PLAN-EXL3-X2A.json`(단위 시험, 사이드카 target_test, X1 BF16과 비교, UD fingerprint). 결과는 여기에 적는다.

### 후속 작업 (X2/X4)

- **MTP를 쓸 수 없다.** 사이드카 이미지에 blk.64(nextn) 텐서가 없다. EXL3 팩의 `mtp.*`를 이미지에 넣거나 EXL3 투영으로 연결해야 한다.
- **DFlash 비용 선택이 EXL3 헤드를 비-Q8로 본다.** `QwenDFlashGpuDraftBackend`는 `output.type == kQ8_0`으로 Q8/Q4 검증 비용을 고른다. EXL3 참조는 type이 기본값 `kF32`라서 Q4 쪽으로 분류된다. EXL3용 비용 기준을 정해야 한다.
- **`rq_logits`/`rq_kld_base`가 헤드 scratch를 넘기지 않는다.** 이 상태로 EXL3 모델을 넣으면 `logic_error`로 멈춘다. W2가 고친다(supervisor 배정).

### X2A 창 1 (2026-10-30, `x2a-unit/x2a`, a72c89b0 바이너리, FAILED)

| 시험 | 결과 | 비고 |
|---|---|---|
| `exl3_reference_test` | FAIL | dequant는 모든 K/codebook/half-K에서 bit-exact다. projection 최대 상대오차는 4.3e-7이다. rotation 검사가 5.4e-5로 떨어졌는데, 원소마다 \|기대값\|+1e-3으로 나누는 척도 탓이다. 행 최대값 기준으로 고쳤다. |
| sidecar checks / vs X1 | FAIL | `CreateWeightRegions`가 메모리 이미지(힙)를 `CloneMappedWeightsToAnonymousPages`로 보냈고, 정렬되지 않은 `MADV_POPULATE_READ`가 EINVAL을 냈다. 성공했다면 `MADV_DONTNEED`가 힙을 지웠을 것이다. 파일 없는 영역은 장치 할당 후 `CopyMappedRegion`으로 올리게 고쳤다. |
| UD-Q4_K_M fingerprints | PASS | rq2h와 byte-equal이다. 비-EXL3 경로는 바뀌지 않았다. |

- 창 2(`PLAN-EXL3-X2A2.json`)는 F1 ABBA 뒤에 돈다.
- 빌드는 W3 08bba6ed(빠른 decode/verify 커널)를 포함해 다시 한다. 23:05 빌드에는 08bba6ed가 없다(빌드 23:05:57, 커밋 23:08:42, exl3_gemv 오브젝트 없음).
- 시험 순서: kernel_test → reference_test → sidecar 둘 → fingerprints.

### X2A 창 2 (2026-10-31, `x2a2-unit/x2a2`, 1e47ca44 바이너리 = W3 08bba6ed 포함, FAILED: W3 kernel test만 실패)

| 시험 | 결과 | 비고 |
|---|---|---|
| `qwen27b_exl3_kernel_test` (W3) | FAIL 5/10 | FP16 반올림 입력 대비 상대 L2가 0.9e-5~4.1e-5로 한계 1e-5를 넘는다. 정확값 대비는 2.1e-4~2.2e-4(FP16 활성 반올림 수준)다. W3가 판단한다. |
| `exl3_reference_test` | PASS | rotation 1.4e-7, projection 4.3e-7, dequant bit-exact |
| 5.00 사이드카 target 검사 | PASS (1055 s) | replay/검증/동시성/prefill 연속 검사를 EXL3 경로로 모두 통과했다. decode/verify는 빠른 경로다. |
| 사이드카 vs X1 5.00 BF16 GGUF | PASS (199 s) | 27행, top1 27/27, mean KL 4.9e-6, mean TV 9.2e-4, mean ΔNLL -9.1e-4. 측정값이다. 차이는 BF16 재반올림, Q8_0 임베딩, FP16 활성(빠른 경로)에서 온다. |
| UD-Q4_K_M fingerprints | PASS | rq2h와 byte-equal이다. |

- 빠른 경로를 끄는 스위치 없이 엔진 수준 결과가 나왔다.
- 기준 경로만 쓰는 대조 빌드(08bba6ed 이전)는 엔진 시험이 통과했으므로 필요하지 않다.

### X2a 품질 창 준비 (2026-10-31)

- 수집기 `e49424919532…`(`build/xa-q-gpu-test`, HEAD `30f32907`, W3 stage 2/3a/3b 포함)를 쓴다. 처음 빌드한 `dc886362…`(316b9f64)는 대체됐다. 자세한 내용은 R0-RESULTS 같은 절에 있다.
- 사이드카:
  - `gufo.gguf`(sha256 `28fbeb69…`, 두 팩이 같다)를 BF16 shard 1 템플릿으로 다시 써서 byte-equal을 확인했다.
  - `gufo-bf16emb.gguf`(sha256 `cb044d6c…`, 49키, `gufo.exl3.embedding="bf16"`)는 `metadata --bf16-embedding`(316b9f64)으로 썼다.
- 계획: `PLAN-EXL3-X2AQ.json`, 팩마다 창 하나.
  - ExLlamaV3 같은 팩 대비 prefill/decode, gufo BF16 대비 prefill.
  - BF16 임베딩 사이드카의 prefill(두 기준).
  - EXL3 fingerprints, GGUF 동일성 확인.
  - 창 뒤에 `metrics.py score`/`compare`로 임베딩 짝 비교를 하고, `model_resident_bytes`를 적는다.

### X2AQ 창 a01 (2026-10-31, 중단)

- 4.00 창 결과:
  - 통과: UD fingerprints, GGUF 동일성(BF16 prefill 창 0, 4095/4095), EXL3 4.00 fingerprints, decode-vs-exl3(4096 레코드).
  - prefill-vs-exl3: 수집기 head replay 검사에서 실패했다(R0-RESULTS 같은 날 절). 남은 prefill이 모두 같은 이유로 실패하므로 감독 승인으로 창을 멈췄다. receipt와 decode 기록은 남겼다.
  - 5.00 창은 시작 전에 막았다.
- **내 실수:** sidecar 전체 경로 검사가 계획에 들어가지 않았다. 편집 스크립트의 변수 재사용 때문이다. 이번에 다시 넣고 읽어서 확인했다.
- 수집기는 `f007ede2`로 고쳤다(`26f9b96a…`). 다시 돌리는 창은 a02다.

### X2AQ 창 a02 (2026-10-31, `xa-q-a02`, 수집기 `26f9b96a…`, FAILED: prefill 시간 초과)

| 시험 | 4.00 | 5.00 |
|---|---|---|
| UD fingerprints | PASS (rq2h와 같음) | — |
| GGUF 동일성(BF16 prefill 창 0) | PASS, 4095/4095 동일 | — |
| sidecar 전체 경로 검사 | PASS (1044 s) | PASS (1057 s) |
| EXL3 fingerprints | PASS (a01과 같은 값) | PASS |
| prefill-vs-exl3 | 1800 s 시간 초과(창 7/8 완료) | 1800 s 시간 초과(창 7/8 완료) |

- 기준 경로 prefill은 4096토큰 창 하나에 약 226 s, 창 8개에 약 1808 s 걸린다. 계획의 1800 s 한계를 넘었다(내 계획 오류).
- 시간 초과가 나면 창 드라이버가 남은 시험을 돌리지 않는다.
- 끝난 창 7개는 head replay 검사를 모두 통과했다(오류 없음). 수집기 수정(`f007ede2`)이 맞다.
- 사용자 지시(단계별 창 없음)에 따라 다시 돌리지 않는다. 품질 측정은 `PLAN-FINAL-27B`에서 한다. prefill 한계는 3600 s, 창은 12600 s로 고쳤다.

## 최종 27B 품질 창 (PLAN-FINAL-27B, c97d555d, 2026-10-31)

- **이미지:** `final-c97d555d-tests-rocm101`(id a42ab0c0)과 release `final-c97d555d-rocm101`(id e797855d). 둘 다 라벨 c97d555d다.
  - `git archive c97d555d`(ai-tools/integ)에서 운영 nix derivation으로 빌드했다(ROCm 10.1, `build/final-c97d555d/`).
  - 테스트 실행 파일은 206개다. deepseek_v4_flash HIP 테스트 3개는 운영 툴체인에서 링크가 실패해 넣지 않았다.
- 창 4개(`final-27b-a01/`)가 모두 COMPLETE, health_PASS였다. 재시도는 없었다.
  - final-units 32/32 PASS: EXL3, KVarN(K1T 5개 포함), Qwen·FN attention, FN CPU, kv_cache_type.
  - 품질 창 셋은 모든 시험이 PASS였다: UD fingerprints(rq2h와 byte-equal), GGUF 동일성 4095/4095, 팩마다 sidecar 전체 경로 검사, collector head replay.
- **품질**(a01 코퍼스, top-k 256, bootstrap 2000, seed 20261029; `final-27b-a01/analysis/`):

| 팩 | 경로 | 기준 | KL 평균 | ΔNLL | top-1 |
|---|---|---|---|---|---|
| 4.00 | prefill | ExLlamaV3 같은 팩 | 1.90e-4 | +2.17e-3 | 99.42 % |
| 4.00 | decode | ExLlamaV3 같은 팩 | 0.80e-4 | +0.89e-3 | 99.49 % |
| 4.00 | prefill | gufo BF16 | 1.90e-2 | +1.38e-2 | 93.50 % |
| 5.00 | prefill | ExLlamaV3 같은 팩 | 4.23e-4 | +1.71e-3 | 99.37 % |
| 5.00 | decode | ExLlamaV3 같은 팩 | 0.66e-4 | +0.21e-3 | 99.49 % |
| 5.00 | prefill | gufo BF16 | 7.69e-3 | +0.10e-3 | 96.11 % |
| c02 | prefill | ExLlamaV3 c02(xq-q01) | 2.92e-4 | +0.93e-3 | 99.42 % |
| c02 | decode | ExLlamaV3 c02 | 1.26e-4 | +0.54e-3 | 99.32 % |
| c02 | prefill | gufo BF16 | 6.54e-3 | −2.80e-3 | 96.25 % |

- **엔진 차이**(gufo 직접 로드 vs ExLlamaV3, 같은 팩):
  - KL 평균은 0.7–4.2e-4, ΔNLL은 +0.2–2.2e-3이다. X1(BF16 GGUF 경로)과 같은 크기다.
  - BF16 대비 품질 순서는 W8/XQ와 같다: c02 ≈ 5.00 > 4.00.
- **임베딩 짝 비교**(Q8_0 − BF16, 같은 위치, ExLlamaV3 기준):
  - 4.00: ΔKL +0.9e-5, ΔNLL +1.9e-4, top-1 −0.02 %p.
  - 5.00: ΔKL +1.7e-4(token CI [−0.0e-4, +4.5e-4]), ΔNLL +1.8e-4(CI가 0을 포함), top-1 −0.04 %p(CI [−0.064, −0.012] %p로 구분됨).
  - gufo BF16 기준이면 4.00, 5.00 모두 KL ±1e-4 이하, ΔNLL ≤ 1.9e-4이고 CI가 0을 포함한다.
  - 판정: Q8_0 임베딩의 영향은 ΔNLL로 엔진 차이의 1/10 이하다. 5.00의 KL 평균에서는 엔진 차이(2.6e-4)와 같은 크기의 꼬리 기여(+1.7e-4)가 보이고, top-1이 0.04 %p 떨어진다. 감독이 판단한다.
- **상주 메모리**(`model_resident_bytes`): 4.00 13.55 GiB(BF16 임베딩 14.66), 5.00과 c02 16.38 GiB(BF16 임베딩 17.49). Q8_0 임베딩은 BF16보다 1.11 GiB 작다.

### 결정: 토큰 임베딩은 BF16 (감독, 2026-10-31)

- **이유:**
  - 품질 유지가 목표다.
  - 5.00에서 Q8_0의 top-1 −0.04 %p가 구분됐다.
  - 임베딩 조회는 토큰마다 한 행이라 속도 영향이 없다.
  - 1.11 GiB는 128 GB 장비에서 문제가 아니다.
- **적용:**
  - c02 팩에 `gufo-bf16emb.gguf`(sha256 `cb044d6c…`, 4.00/5.00과 같은 파일)를 기본으로 두었다.
  - `PLAN-FINAL-27B`의 타이밍 창(XB6)은 EXL3 4.00과 c02를 모두 이 sidecar로 불러온다(4곳).
  - 로더의 키 없는 기본값(Q8_0)은 바꾸지 않았다. 바꾸려면 코드 변경과 이미지 재빌드가 필요하다.

### 최종 이미지 재빌드 b6e3ff82 (2026-10-31, gufo-06 요청)

- **소스:** integ `b6e3ff82` = c97d555d + 3712dbe7(BF16 임베딩 기본값) + efc396ba(FN tensorless shard digest, `gguf_reader`에 `data_offset` 기록).
  - 같은 방법으로 빌드했다(`build/final-b6e3ff82/`). 테스트 실행 파일은 207개다.
- **이미지:**
  - release `ai-tools/gufo-eval:final-b6e3ff82-rocm101` id `08fcc2b23cea`
  - tests `final-b6e3ff82-tests-rocm101` id `4db07a5bff24`
  - 둘 다 라벨 `b6e3ff827801febf5fd5ca83c3ba5042f9a02452`이다. c97d555d 이미지는 남겼다.
- **sanity 창**(`PLAN-FINAL-B6E3-SANITY.json`, `final-b6e3-sanity/`): final-units 32/32 PASS. UD fingerprints는 rq2h와 byte-equal이고, BF16 prefill 창 0은 rq0-a01과 4095/4095 같다. 둘 다 COMPLETE, health_PASS다.

### 최종 이미지 68d4e3c8과 통합 시험 (2026-10-31, gufo-06 요청)

- **이미지:** 라벨은 모두 `68d4e3c8643a39323d06ee73674c6b0d956c6997`이다.
  - release `final-68d4e3c8-rocm101` id `50ccc384a513`
  - tests `final-68d4e3c8-tests-rocm101` id `f25dd87105b6`
  - 테스트 derivation은 `/nix/store/1qrjnri6…`(`build/final-68d4e3c8/`)이고, 요청한 ctest 바이너리가 모두 있다.
- **창**(`PLAN-FINAL-68D4.json`, `final-68d4-a01/`, 한 번씩 돌렸다):
  - final-units 32/32 PASS.
  - final-identity 3/3 PASS: UD fingerprints가 rq2h와 byte-equal, BF16 prefill이 4095/4095 동일.
  - final-gpu-units 13/18 PASS. 27B `exl3_kernel_test`(TestBuckets 포함), `exl3_reference_test`, `plain_prefill_stable_test`는 모두 통과했다.
- **실패 5개.** 모두 FN의 bitwise 동일성 검사다. 로그는 `final-gpu-units/receipt/run/<이름>.log`에 있다. 고치지 않았다.

| ctest | 실패한 검사 |
|---|---|
| fn.exl3_prefill_executor | X3 prefill F=640: chunks=1의 prefill 잔차 행이 bitwise로 다르다 |
| fn.exl3_gdn_prefill | GDN dense X3 1/64/65 chunk: 잔차 행이 bitwise로 다르다 |
| fn.mtp_kv_group | MTP K7 QKV vs KV group rows=1: Q를 빼면 K/V projection 비트가 바뀐다 |
| fn.mtp_kv_executor | EXL3 scalar cache-only YaRN: ragged catch-up session 0의 K cache가 bitwise로 다르다 |
| fn.hc_down_f16 | HC fusion이 stable scalar와 31비트 다르다 |

- 추정(확인 안 함): 이 바이너리는 운영 nix 툴체인(ROCm 10.1 / HIP Clang 24, release 플래그)으로 빌드됐다. integ docker gpu-test 빌드(ROCm 7.2.4 이미지, RelWithDebInfo)와 컴파일러·플래그가 다르다.
