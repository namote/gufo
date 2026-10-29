# R1c: upstream `prism.hadamard.*` 명세와 우리 그룹의 대응 (W1c 1단계, 2026-10-29)

상위 문서: [W1c 브리프](W1C-BRIEF.md), PLAN D8. 이 단계는 읽기만 했다. 코드 변경, 빌드, 변환, GPU 사용은 없다.

## 근거로 읽은 것

| 출처 | 리비전 | 비고 |
|---|---|---|
| upstream `ggml-org/llama.cpp` master | `8ab29faec401829db0e617f40af2c942f4a9eaab`(2026-10-29) | `build/rq-llama/upstream-src/`에 shallow clone |
| 병합 커밋 #29600 | `a9d27ac693ff0a1d3cba449c912c2b891904a4c7`(2026-10-11) | `gh api …?ref=`로 `llama-model.cpp`를 대조했다 |
| 설계 PR #29077 | — | 닫힘, 본문만 |
| LoRA PR #30910 | `0682a576a772`(2026-10-22) | — |

- 병합 뒤 `src/llama-model.cpp`에서 prism과 Hadamard를 언급하는 커밋은 `a9d27ac693` 하나뿐이다.
- 병합 시점의 `load_hparams_hadamard`(version 1/2, `tied_output`, `sign_data[width]`, `gdn_v_grouped`)는 master와 같다. 그래서 형식은 병합 뒤 바뀌지 않은 것으로 본다(확인).
- 아래 줄 번호는 master `8ab29fae` 기준이다.

표기: **확인**은 코드로 확인한 것이고, **짐작**은 읽고 추론했지만 실행으로 확인하지 않은 것이다.

## 결론 요약

1. **부호 벡터는 입력 폭(K)마다 하나다.** 파일 전체에서 폭마다 하나이고, 층이나 그룹마다 두지 않는다. 우리 v1은 그룹 × 층마다 다르다.
   - 그래서 **기존 회전 파일은 가중치 바이트를 그대로 둔 채 옮길 수 없다.**
   - 원본에서 다시 회전하는 비용은 작다. 회전 BF16은 약 3분이고, Q4는 하나에 약 2.5–4.5분이다.
   - 대신 R2 ②③의 GPU 결과를 새 파일로 다시 재야 한다.
2. **`ssm_alpha`와 `ssm_beta`는 upstream 허용 목록에 없다.** 우리 `attn_in` 그룹 가운데 DeltaNet 층의 이 두 텐서를 회전하면 upstream 로더가 거부한다.
   - upstream에서 읽히는 파일을 만들려면 이 둘은 회전하지 않아야 한다. 그러면 gufo는 같은 활성화의 회전본과 원본을 둘 다 써야 한다(W3 영향). **사용자 결정이 필요하다.**
3. 나머지는 모두 대응한다.
   - 변환 정의 `x' = H·(s·x)`와 Sylvester 순서, 정규화, block 128
   - 허용 아키텍처(`qwen35`)와 나머지 텐서 종류(`attn_q/k/v/qkv/gate/output`, `ffn_*`, `ssm_out`, `output.weight`)
   - nextn 제외: `weight_names`에 넣지 않으면 된다
   - llama-quantize의 KV 보존
   - GDN 순열 `P`는 우리에게 필요 없다. 우리는 GGUF(tiled) 순서에서 회전하므로 `gdn_v_grouped`를 쓰지 않는다.
4. **잃는 것:** 지원하지 않는 llama.cpp(우리 기준 b11069 포함)는 이 파일을 평범한 `qwen35`로 읽어 조용히 틀린 출력을 낸다. #29600 본문에 따르면 master에서 PPL이 10.23에서 1,258,507로 무너진다. 형식 안에서 이것을 막을 방법은 없다.

## 1. KV 명세 (확인: `src/llama-model.cpp:2073-2234`, `src/llama-arch.cpp:486-496`, `gguf-py/gguf/gguf_writer.py:1765-1796`)

| 키 | GGUF 타입 | 허용 값 | 로더 검증(거부 조건) |
|---|---|---|---|
| `prism.hadamard.version` | u32 | 1 또는 2 | 다른 값은 거부. **이 키가 없으면 나머지 키를 모두 무시**한다(회전 없음으로 읽음). 2는 `tied_output = true`일 때만, 1은 `false`일 때만 허용 |
| `prism.hadamard.tied_output` | bool, 선택 | | true면 `output.weight`가 없어야 하고 version 2여야 하며, `inverse_weight_names`에 `token_embd.weight`가 있어야 한다 |
| `prism.hadamard.block_size` | u32, 필수 | 2의 거듭제곱 | 0이거나 2의 거듭제곱이 아니면 거부. 텐서마다 `ne[0] % block_size == 0`도 검사한다(`:2275`) |
| `prism.hadamard.transform` | string, 필수 | `"normalized-sylvester-walsh-hadamard"` | 다르면 거부 |
| `prism.hadamard.axis` | string, 필수 | `"input-last-dimension"` | 다르면 거부 |
| `prism.hadamard.sign_mode` | string, 필수 | `"identity"`, `"explicit"` | 다르면 거부 |
| `prism.hadamard.sign_widths` | i32(또는 u32) 배열 | explicit일 때 필수 | 비어 있으면 거부. 원소마다 `> 0`이고 block_size의 배수여야 하며, **같은 폭이 두 번 나오면 거부**한다 |
| `prism.hadamard.sign_values` | i32 배열 | ±1 | 폭의 합과 길이가 같아야 한다. ±1이 아닌 값은 거부 |
| `prism.hadamard.weight_names` | string 배열, 필수 | 허용 종류의 이름 | 비어 있으면 거부, 중복 거부. 허용 종류 밖이면 거부(3절). 파일에 없는 이름은 `load_tensors`에서 거부 |
| `prism.hadamard.inverse_weight_names` | string 배열, 선택 | `token_embd.weight`만 | 다른 이름은 거부. `weight_names`와 겹쳐도 거부 |
| `prism.hadamard.gdn_v_grouped` | bool, 선택 | | true면 `ssm_out`의 입력을 tiled에서 grouped 순서로 순열한다(4절). 헤드 기하가 맞지 않으면 거부 |

- **sign_mode의 뜻:**
  - `identity`: s = 1. 부호 없이 H만 쓴다. 이때 `sign_widths`와 `sign_values`는 읽지 않는다.
  - `explicit`: 폭마다 부호 벡터 하나(`sign_data[width]`)를 쓴다.
- **아키텍처 허용 목록**(`:2151-2163`): `llama`, `qwen3`, `qwen3moe`, `qwen35`, `qwen35moe`, `qwen3next`. 그 밖은 거부한다. 우리 `qwen35`는 들어 있다.
- 추가 보호: 그래프를 처음 만들 때 `llama_verify_hadamard_graph`(`src/llama-context.cpp:31`, 호출 `:2684`)가 접힌 가중치가 모두 변환된 입력을 읽는지 확인한다. 빠진 곳이 있으면 거부한다.
  - 다만 이 검사는 **그 context 모델의 `hdmd.rot`만** 본다. 6절의 DFlash 위험을 보라.
- 정수 배열은 `get_arr<int32_t>`가 INT32와 UINT32를 모두 받는다(`src/llama-model-loader.cpp:323`). 부호에 -1이 있으므로 INT32로 쓴다.

## 2. 부호 벡터 단위와 우리 그룹

- **확인:** `hdmd.sign_data`는 `std::map<uint32_t width, std::vector<int32_t>>`다(`src/llama-model.h:716`). 같은 폭이 두 번 나오면 거부한다(`:2131`).
- `load_tensors_hadamard`는 가중치마다 `sign_data.find(weight->ne[0])`로 **폭이 같은 모든 가중치에 같은 부호**를 붙인다(`:2343-2386`). 함수 주석에도 "one sign vector per width"라고 적혀 있다(`:2236`).
- 활성화 변환 캐시의 키는 `(활성화, 회전 행렬)`이다(`src/llama-graph.cpp:1622`, `get_hdmd_input`). 부호가 폭마다 하나라서 같은 활성화를 읽는 가중치끼리 결과를 공유할 수 있다.

| 우리 그룹 | K | 소비 텐서 | v1의 D | upstream의 s |
|---|---:|---|---|---|
| attn_in | 5120 | attn_qkv, attn_gate, ssm_alpha, ssm_beta / attn_q, attn_k, attn_v | 층마다 다름(`attn_in:<층>`) | **s₅₁₂₀** 하나 |
| ffn_in | 5120 | ffn_gate, ffn_up | 층마다 다름 | **s₅₁₂₀**(attn_in과 같음) |
| head_in | 5120 | output | 하나 | **s₅₁₂₀**(같음) |
| ssm_out_in | 6144 | ssm_out | 층마다 다름 | **s₆₁₄₄** 하나 |
| attn_out_in | 6144 | attn_output | 층마다 다름 | **s₆₁₄₄**(ssm_out과 같음) |
| ffn_down_in | 17408 | ffn_down | 층마다 다름 | **s₁₇₄₀₈** 하나 |

- v1 회전 BF16에는 서로 다른 D가 64층 × 5그룹 + head = 321개쯤 있다(짐작: 그룹 인스턴스 수에서 셈). upstream이 표현할 수 있는 부호는 3개뿐이다.
- **그래서 바이트를 보존하는 이전은 불가능하다.** 회전 파일에서 다시 접는 방법은 있다: `W'' = W'·H·(D_l·D_g)·H`. 하지만 BF16 반올림이 한 번 더 들어가므로 원본에서 다시 회전하는 쪽이 정확하다.
- **다시 변환하는 비용**(R1b 실측 기준):

  | 항목 | 비용 |
  |---|---|
  | 회전 BF16 | 159.5 s(입력 해시 포함), 쓰기 55 GB |
  | 회전 Q4(UD 복사) | 141.8 s, 17.6 GB |
  | 비회전 Q4 | 바뀌지 않는다(회전이 없다). 정렬만 다르다 |
  | R2 ②③ GPU 재측정 | W3/W2 창. 시간은 W3 기준 |
  | R3 | 아직 하지 않았으므로 손실 없음 |

- 폭마다 부호가 하나라는 것은 표현력의 문제일 뿐이다. 품질 차이는 짐작으로 무시할 만하다. 층마다 다른 D는 이론적 이득이 거의 없다. 같은 D를 써도 W·D·H는 층마다 다르다.

## 3. 가중치 목록, `head_in`, nextn, 허용 텐서 종류

- **허용 종류**(확인, `src/llama-model.cpp:2166-2199`; 변환 쪽 `conversion/base.py:754`): `output.weight`, 그리고 `blk.N.{attn_q, attn_k, attn_v, attn_qkv, attn_gate, attn_output, ffn_gate, ffn_up, ffn_down, *_exps, *_shexp, ssm_out}.weight`

| 우리 그룹의 텐서 | upstream 허용 | 비고 |
|---|---|---|
| attn_qkv, attn_gate, attn_q, attn_k, attn_v | 예 | |
| **ssm_alpha, ssm_beta** | **아니오** | upstream `qwen35.cpp:370,377`은 이 둘도 `build_lora_mm`으로 계산한다. 그래서 기술적으로는 접을 수 있지만, 허용 목록에 없어서 이름 때문에 거부된다 |
| ssm_out, attn_output | 예 | |
| ffn_gate, ffn_up, ffn_down | 예 | |
| output.weight | 예 | `build_lora_mm`, untied |

- **`ssm_alpha`/`ssm_beta` 선택지(결정 필요):**
  - (A) 회전하지 않는다. 파일은 upstream과 호환된다. gufo는 DeltaNet 층에서 이 두 텐서만 회전 전 활성화를 써야 한다. 지금 decode, prefill, batched 경로는 같은 정규화 활성화로 `attn_qkv`, `attn_gate`, `alpha`, `beta`를 함께 계산한다(`decode_step.cpp:229`, `prefill_chunk.cpp:508`, `batched_decode.cpp:151`).
    - (A1) 원래 활성화를 하나 더 둔다.
    - (A2) 로드할 때 gufo가 `W·D·H`로 접는다. 텐서가 작아서(48×5120) 비용은 무시할 만하다. 하지만 Q8_0이면 다시 양자화해야 해서 바이트가 달라진다. BF16/F32로 들고 있으면 정확하다.
  - (B) 회전한다. upstream 로더가 거부하므로 D8의 목적(교차 검증과 호환)에 맞지 않는다.
  - (C) upstream 허용 목록에 `ssm_alpha`/`ssm_beta`를 더하는 PR을 낸다. 변경은 작다(로더 목록과 `conversion/base.py` 정규식). 하지만 병합 시점을 우리가 정할 수 없다.
  - 제안: (A)로 가고, 구현은 W3가 정한다(A1 또는 A2). 품질 영향은 짐작으로 무시할 만하다. 두 텐서는 UD 레시피에서 Q8_0이고 원래 크기도 작다.
- **nextn `blk.64`(D6):** upstream에는 층 범위 규칙이 없다. `weight_names`에 넣은 것만 접힌 것으로 본다. 그러므로 `blk.64.*`를 목록에서 빼면 D6이 그대로 표현된다(확인: 로더는 목록 밖의 가중치를 건드리지 않는다).
- **`head_in`과 `tied_output`, `inverse_weight_names`:**
  - 27B는 untied다(`output.weight`가 있다). 그래서 version 1로 쓰고 `output.weight`를 `weight_names`에 넣는다. `tied_output`은 쓰지 않는다.
  - `inverse_weight_names`(token_embd를 회전 기저로 저장하고 조회 뒤 역변환하는 방식)는 쓰지 않는다. 우리는 임베딩을 회전하지 않는다.
  - v1의 "tied 모델에서 head 회전 거부" 규칙은 upstream의 version 2 규칙으로 대체된다. 우리 컨버터는 tied 모델에서 계속 head 회전을 거부해도 된다(지원 범위를 좁히는 것).
- 그룹 일관성: upstream은 같은 활성화의 소비자 일부만 접어도 허용한다. v1의 "인스턴스 안 일부만 회전하면 거부"는 upstream 규칙이 아니다. 그래서 gufo 로더와 컨버터가 "gufo가 실행할 수 있는 부분집합"으로 따로 지킨다.

## 4. GDN 순열 `P`(`gdn_v_grouped`)

- **확인:**
  - HF `linear_attn.out_proj`의 입력(V 헤드)은 K 헤드 기준 묶음 순서다. llama.cpp 변환은 이 열을 tiled 순서로 재배열한다(R1B 항목 12).
  - upstream 변환은 **접힌** `out_proj`를 재배열하지 않고 grouped 순서로 둔다. 그리고 `gdn_v_grouped = true`를 쓴다(`conversion/qwen.py:581-584, 646-650`). 런타임은 `ssm_out` 입력을 tiled `[hd, nk, rep]`에서 grouped `[hd, rep, nk]`로 순열한 다음 s와 H를 적용한다(`src/llama-graph.cpp:1626-1632`, `llama_hadamard_transform::set_gdn_v_perm`).
  - 이것은 HF 기저에서 접는 Prism 도구 때문에 필요한 장치다.
- **우리에게는 필요 없다.** 우리는 GGUF(이미 tiled로 재배열된) `ssm_out`의 K 순서에서 회전한다. 그래서 `gdn_v_grouped`를 쓰지 않거나 false로 둔다.
  - 이 키는 전역이라 모든 접힌 `ssm_out`에 적용된다. 우리 방식과 섞어 쓸 수 없다.
  - gufo 로더는 `gdn_v_grouped = true`인 파일(예: Prism 배포 파일)을 거부하거나 순열을 구현해야 한다. 제안은 거부다.

## 5. block 128과 변환 정의

- **확인:**
  - H는 `H[r][c] = (parity(r & c) ? −1 : 1)/√block`이다(`src/llama-model.cpp:2321-2333`). 우리 `sylvester()`(`np.block` 재귀)와 같은 natural order다.
  - 적용 순서는 `x ← P·x`(선택), `x ← s ⊙ x`, `x ← H_block·x`(`llama_mul_mat_hadamard`, `src/llama-impl.h:57`)다. 128개 묶음으로 reshape해 `mul_mat(rot, x)`를 하는데, ggml의 mul_mat은 `rotᵀ·x`이고 H는 대칭이므로 결과는 `H·x`다.
  - 따라서 `x' = H·(s·x)`이고, 우리 `x' = H·D·x`와 같다. 저장 가중치 `W' = W·D·H`도 같다(로더 주석 `:2165`).
- block_size는 128로 쓴다. 27B의 K(5120, 6144, 17408)는 모두 128의 배수다.

## 6. 양자화 경로와 실행

- **llama-quantize KV 보존:** 짐작이지만 근거가 강하다. `llama-quant.cpp:1070`이 `gguf_set_kv(ctx_out, ml.metadata)`로 입력 KV 전체를 복사한다. prism 전용 처리는 없다. R1b에서 b11069가 `gufo.*`를 그대로 보존한 것과 같은 경로다.
  - 텐서 수가 원본과 같아지므로(`.rot` 없음) b11069 llama-quantize에서도 그대로 동작할 것이다.
- **타입 제한:** 없다(확인). 변환은 활성화 쪽 `mul_mat`이고 가중치 타입과 무관하다.
  - `build_lora_mm`이 `w`의 타입을 따지지 않는다. NVFP4만 정밀도 힌트가 따로 있다.
  - `GGML_HINT_SRC0_IS_HADAMARD`는 CPU, CUDA/HIP(`ggml-cuda/fwht.cu`), Vulkan, SYCL, BLAS에서 FWHT 빠른 경로를 고르는 힌트다. 정확성과는 무관하다.
- **imatrix (R3에 영향):** 짐작이지만 근거가 강하다.
  - upstream `llama-imatrix`는 matmul의 `src0` 이름이 `blk.`(또는 output)인 노드의 `src1`을 기록한다(`tools/imatrix/imatrix.cpp:566-579`).
  - 접힌 가중치의 matmul은 `src1 = H·(s·x)`, 즉 **회전된 입력**이다. 그래서 회전 기저의 통계가 가중치 이름으로 기록된다.
  - Hadamard matmul 노드 자체(`prism.hadamard.128`)는 이름 필터에서 빠진다.
  - 그러므로 prism 형식의 회전 BF16에 upstream `llama-imatrix`를 돌리면 R3가 바라는 회전 기저 imatrix가 나온다. gufo 수집기의 기준(교차 검증)이 될 수 있고, 대안 수집기도 될 수 있다. 그 imatrix를 upstream llama-quantize에 넣는 것도 형식상 맞다.
- **DFlash2 드래프터 위험(확인, upstream 쪽 문제):**
  - upstream DFlash 디코더는 대상 모델의 `output`을 빌려 `build_lora_mm`으로 계산한다(`src/models/dflash.cpp:1062-1073`).
  - 그런데 graph의 `hdmd`는 **드래프터 모델의 것**(`src/llama-context.cpp:2727`, `model.hdmd`)이라 비어 있다. 그러면 회전된 `output.weight`가 변환 없는 입력으로 계산되고, `llama_verify_hadamard_graph`도 드래프터 context에서는 돌지 않는다.
  - 그래서 `head_in`을 회전한 파일을 upstream DFlash2와 쓰면 draft logits가 조용히 틀린다. 수락률만 떨어지고 출력은 정확하다.
  - gufo는 W3에서 드래프터 head 경로를 직접 회전한다. 교차 검증에서는 speculative 없이 비교한다.

## 7. 우리 쪽 변경 목록 (제안, 크기는 짐작)

| 영역 | 변경 | 크기 |
|---|---|---|
| `tools/rotquant/hadamard.py` | 부호 유도를 인스턴스 단위에서 폭 단위로 바꾼다(`signs(seed, f"width:{K}", K)`) | 작음 |
| `tools/qwen27b/rotquant_groups.py` | `attn_in`에서 `ssm_alpha`/`ssm_beta`를 뺀다(결정 (A)일 때). upstream 허용 종류 정규식으로 검사한다. nextn 제외는 그대로 둔다 | 작음 |
| `tools/rotquant/plan.py` | `ROTATION_KV`와 `.rot`을 `prism.hadamard.*`로 바꾼다. `check_rotation`을 upstream 규칙(1절)과 gufo 부분집합 규칙(그룹 일관성, `gdn_v_grouped`/`tied_output`/`inverse` 거부)으로 다시 쓴다. dry-run 표에서 `.rot` 줄을 sign widths 줄로 바꾼다 | 중간(150–250줄) |
| `tools/rotquant/gguf_io.py` | INT32 배열과 STRING 배열 인코딩을 추가한다 | 작음 |
| `tools/rotquant/convert.py` | `rotate_file`에서 `.rot` 쓰기를 빼고 prism KV를 쓴다. 출처 `gufo.rotation.source_*` KV는 남긴다(이름 공간을 바꿀지는 선택) | 작음 |
| `tools/rotquant/verify.py` | `.rot` 바이트 비교 대신 prism KV 바이트 보존을 검사한다. 텐서 집합은 원본과 같다 | 작음 |
| `tools/rotquant/check_rotated.py` | 부호를 KV에서 폭마다 읽는다 | 작음 |
| `tools/rotquant/realign.py` | 변경 없음 | — |
| 테스트 `tests/tools/rotquant_test.py` | `.rot` 가정을 쓰는 테스트(왕복, 거부 11종, dry-run, verify, 가짜 quantize)를 다시 쓴다. upstream 거부 규칙 테스트를 더한다 | 큼(200–300줄) |
| 테스트 `tests/models/qwen27b/rotquant_groups_test.py` | alpha/beta 제외와 허용 종류 검사 | 작음 |
| 로더 `src/core/quant/rotation.{hpp,cpp}`(W3) | `ReadRotation`이 prism KV를 읽게 한다. 폭별 부호를 그룹 인스턴스에 연결한다(모든 층이 같은 벡터를 가리킴). upstream 거부 규칙과 gufo 부분집합 규칙을 넣는다 | 중간(277줄 파일을 다시 씀) |
| 27B 바인딩 `src/models/qwen/weights.cpp`(W3) | `BindRotation`: 층별 포인터가 공유 벡터를 가리키게 한다. `GroupTable`에서 alpha/beta를 뺀다 | 작음 |
| 27B 실행 경로(W3) | (A1) 또는 (A2)에 따라 DeltaNet `alpha`/`beta`의 입력을 바꾼다(decode, prefill, batched, verify) | 중간에서 큼 |
| C++ 테스트(W3) | `quant_rotation_test`, `qwen27b_rotation_binding_test` | 중간 |
| 문서 | `docs/ROTATED-WEIGHTS.md` 새 규칙, PLAN "GGUF 표시 규칙" 절 교체, 최소 llama.cpp 버전 명시 | 작음 |
| 이전 도구 | **바이트 보존 이전은 불가능하다**(2절). 대신 원본에서 새 컨버터로 다시 변환한다. 별도 도구는 만들지 않는다 | — |

## 8. 교차 검증 계획

- **빌드:** upstream llama.cpp를 고정 리비전으로 `build/rq-llama-prism/`에 받아 eval 이미지에서 CPU로 빌드한다. 고정 리비전은 지금 받아 둔 `8ab29fae` 또는 감독이 정한 커밋이다.
  - 방법은 R1b와 같다: `-DGGML_HIP=OFF -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF`
  - 대상: `llama-completion`, `llama-perplexity`, `llama-quantize`, `llama-imatrix`
  - R1b는 b11069를 29 s에 빌드했다. 지금 트리는 더 크므로 1–2분으로 짐작한다.
  - HIP 빌드는 필요 없다. 속도가 필요할 때만(예: imatrix 기준을 크게 만들 때) HIP을 쓴다.
- **검사:**
  1. 새 회전 BF16과 Q4를 upstream이 로드하는지 본다. 로그에 `loaded N Hadamard-folded weight(s) … using 1 rotation(s) and 3 sign vector(s)`가 나와야 한다.
  2. 음성 대조: b11069로 같은 파일을 돌리면 조용히 틀린 출력이 나와야 한다(위험 기록용).
  3. 수치 동치: `llama-perplexity --kl-divergence-base`로 비회전 BF16을 기준으로 삼고, 회전 BF16을 CPU에서 비교한다. 16×512 토큰이다.
     - 27B BF16의 CPU prefill은 R1b 측정으로 짐작해 수십 tok/s다. 그래서 회당 3–10분이다.
     - 회전 BF16의 KL 기대값은 R1B 항목 9(상대 오차 1.7e-3) 수준이다.
  4. gufo와 llama.cpp: 회전 Q4(UD 복사)를 같은 코퍼스 창(W2 하네스)에서 돌려 top-k logits를 비교한다. llama.cpp는 CPU다. 27B Q4 CPU decode는 R1b 측정으로 약 6 tok/s다. logits만 보므로 prefill 위주다(짐작: 창 하나에 수 분).
     - 비교 대상: gufo GPU의 `rq_logits` 출력, 그리고 비회전 Q4의 같은 쌍 차이를 기준선으로 한다.
  5. speculative는 끈다(6절 DFlash 위험).
- 시간 합계(짐작): 빌드 2분 + 로드 확인 2분 + KL 2회 20분 + logits 비교 수십 분. CPU 단계 하나다.

## 9. 위험

- **조용한 오독(잃는 보호):**
  - v1에서는 `.rot` 때문에 텐서 수가 맞지 않아, 지원하지 않는 llama.cpp가 로드를 거부했다(R1b 항목 4).
  - prism 형식은 텐서 집합이 원본과 같다. 그래서 a9d27ac6보다 오래된 llama.cpp, 그리고 prism을 모르는 다른 런타임은 이 파일을 평범한 `qwen35`로 읽어 쓰레기 출력을 낸다(#29077과 #29600 본문, master에서 PPL 1,258,507).
  - 형식 안에서 막을 방법은 없다. 보조 표시(`general.file_type`, 이름)는 로드를 막지 못한다. 추가 텐서를 넣으면 지원하는 llama.cpp도 텐서 수 때문에 거부한다.
  - 대책:
    - 파일 이름에 표시를 넣는다(예: `-prism-hadamard`).
    - `PROVENANCE`와 배포 문서에 최소 llama.cpp 리비전 `a9d27ac6`(2026-10-11)을 적는다. 릴리스 태그 번호는 확인하지 않았다.
    - gufo 쪽은 prism 키를 읽으므로 안전하다.
    - gufo가 지원하지 않는 경로(`--cpu` 등)는 거부를 유지한다.
- **upstream DFlash 드래프터의 head 변환 누락**(6절): head를 회전한 파일을 upstream speculative와 쓰면 수락률이 떨어진다. 필요하면 upstream에 이슈를 낸다. gufo에는 영향이 없다.
- **gufo가 다른 도구로 만든 prism 파일을 받을 때:** `gdn_v_grouped`, `tied_output`/version 2, `inverse_weight_names`, `identity` 모드, block ≠ 128, 그룹 일부만 회전 같은 조합이 올 수 있다. 지원하지 않는 조합은 이름을 들어 거부한다. `identity`는 부호를 모두 +1로 두면 쉽게 받을 수 있다.
- **부호 단위 축소의 품질 영향:** 짐작으로 무시할 만하다. 다시 변환한 뒤 R2 ②로 확인한다.
- **upstream 형식이 바뀔 가능성:** version 1과 2만 허용한다. 우리는 version 1 키만 쓴다.

## 결정이 필요한 것

1. `ssm_alpha`/`ssm_beta`: (A) 회전하지 않음(제안)과 gufo 구현 방식(A1 또는 A2, W3), (B) 회전 유지(upstream 거부), (C) upstream PR
2. 기존 v1 회전 파일 처리: 원본에서 다시 변환(제안)하고 R2 ②③을 다시 잰다. v1 파일은 새 파일이 검증된 뒤 지운다.
3. 출처 KV 이름: `gufo.rotation.source_*`를 유지할지, `gufo.rotquant.source_*`로 바꿀지(작은 일)
4. upstream 고정 리비전(교차 검증과 문서의 최소 버전)
