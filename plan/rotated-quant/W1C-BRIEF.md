# W1c 브리프: 표시 규칙을 upstream `prism.hadamard.*`로 전환 — 1단계 명세 대응 (2026-10-29)

이 브리프는 [W-COMMON.md](W-COMMON.md)를 따른다. 근거는 PLAN의 D8이다(사용자 결정 2026-10-29). 이 단계에서는 **읽기와 문서 작성만** 한다. 코드는 바꾸지 않는다. 구현은 감독이 이 결과를 검토하고, 필요하면 사용자에게 물은 뒤 시작한다.

## 배경

- llama.cpp upstream이 PR #29600(merge `a9d27ac693ff0a1d3cba449c912c2b891904a4c7`, 2026-10-11)으로 Hadamard 접힘 가중치 실행을 병합했다.
  - 설계는 닫힌 PR #29077 본문에 있다: `y = W'·(H·(s·P·x))`, `prism.hadamard.*` KV.
  - LoRA 지원은 #30910이다.
  - gguf-py 키: `VERSION, TIED_OUTPUT, BLOCK_SIZE, TRANSFORM, AXIS, SIGN_MODE, SIGN_WIDTHS, SIGN_VALUES, WEIGHT_NAMES, INVERSE_WEIGHT_NAMES, GDN_V_GROUPED`
- 우리 v1 규칙(PLAN "GGUF 표시 규칙 (v1)")은 `gufo.rotation.*` 전역 KV와 텐서마다 `<name>.rot` 짝 텐서(F16 ±1)를 쓴다.
- 우리 기준 llama.cpp `68d9053a`(b11069)에는 이 지원이 없다(487커밋 앞).

## 읽을 것

- upstream `ggml-org/llama.cpp` master: `gguf-py/gguf/constants.py`, `gguf_writer.py`, `src/llama-model.cpp`, `src/llama-graph.cpp`, `src/llama-arch.cpp`, `src/llama-context.cpp`, `src/models/qwen35.cpp`, `conversion/qwen.py`, `conversion/base.py`, 그리고 llama-quantize의 KV 복사 경로
  - `gh api repos/ggml-org/llama.cpp/contents/<path> -H 'Accept: application/vnd.github.raw'` 또는 `gh pr view/diff`로 읽는다.
  - clone이 필요하면 `build/rq-llama/upstream-src/`에 shallow clone한다. 이것은 가벼운 작업으로 친다.
- PR #29077, #29600, #30910의 본문과 리뷰 토론
- 우리 쪽: `tools/rotquant/`, `tools/qwen27b/rotquant_groups.py`, `src/core/quant/rotation.*`, `src/models/qwen/weights.cpp`(회전 바인딩), `docs/ROTATED-WEIGHTS.md`, R1B/R2 결과 문서

## 답할 것 (`plan/rotated-quant/R1C-PRISM-MAPPING.md`에 표로)

1. **KV 명세.** 키마다 타입, 허용 값, 검증 규칙(로더가 무엇을 거부하는지)을 적고, `sign_mode`의 종류와 의미를 정리한다.
2. **부호 벡터 단위.** upstream은 입력 폭마다 부호 벡터 하나인가, 텐서·그룹마다인가?
   - 우리 그룹(attn_in, ssm_out_in, attn_out_in, ffn_in, ffn_down_in, head_in)과 그 K, 지금 그룹마다 D가 다른지를 표로 만든다.
   - 같은 K인 그룹끼리 D가 달라 upstream으로 표현할 수 없으면, 기존 회전 파일을 가중치 바이트 그대로 옮길 수 없다. 그 경우 다시 변환하는 비용을 적는다.
3. **가중치 목록.**
   - `weight_names`에 넣을 텐서는 nextn `blk.64` 제외(D6)를 뜻해야 한다.
   - `head_in`(`output.weight`)과 `tied_output`, `inverse_weight_names`(token_embd)의 관계를 적는다.
   - upstream이 허용하는 아키텍처(qwen35 포함 여부)와 텐서 종류(SSM `ssm_out`, attention `attn_output`, gate 등)가 우리 그룹을 모두 덮는지 확인한다.
4. **GDN 순열 `P`(`gdn_v_grouped`).** 무엇이며, 우리 `ssm_out_in`에 필요한지 적는다.
5. **block 128과 변환 정의.** `transform`·`axis` 값, 정규화, 곱하는 순서(우리 `x' = H·D·x`와 같은지)를 수식 수준에서 맞춘다.
6. **양자화 경로.**
   - upstream llama-quantize가 `prism.hadamard.*` KV를 보존하는가?
   - imatrix 수집(`llama-imatrix`)은 회전된 입력을 기록하는가? R3 설계에 영향이 있다.
   - k-quant 타입 제한이 있는가? 실행은 타입과 무관한가(`llama_mul_mat_hadamard`, `GGML_HINT_SRC0_IS_HADAMARD`)?
7. **우리 쪽 변경 목록.**
   - 컨버터(writer, verify, check_rotated, realign)에서 바꿀 곳
   - 로더(`src/core/quant/rotation.*`와 27B 바인딩)에서 바꿀 곳
   - 바꿔야 할 테스트
   - 기존 파일 이전 도구(가중치 바이트 보존, `.rot` 제거, KV 추가)를 만들 수 있는지
   - 각각의 대략적 크기
8. **교차 검증 계획.** upstream llama.cpp(`a9d27ac6` 이후 고정 리비전)를 `build/rq-llama-prism/`에 빌드한다. 회전 Q4(UD 복사 레시피)를 llama.cpp와 gufo에서 같은 입력으로 돌려 logits를 비교한다.
   - 빌드 방법, 필요한 HIP 여부, 예상 시간을 적는다.
   - CPU 빌드로 충분한지(27B Q4 CPU 실행 시간)도 판단한다.
9. **위험.** 지원하지 않는 llama.cpp가 이 파일을 조용히 잘못 읽는 문제를 적고, 막을 방법이 있는지 본다(예: upstream이 쓰는 다른 표시, `general.file_type`). 우리 v1의 "거부" 보호를 잃는 대가도 적는다.

짐작과 확인한 것을 구분한다. upstream 코드의 줄 위치나 커밋을 근거로 단다.

## 파일 경계

- 쓸 수 있는 파일: `plan/rotated-quant/R1C-PRISM-MAPPING.md` 하나
- 커밋: `docs(plan): …`로 그 파일만, 경로를 지정해 커밋한다.
- 빌드, 변환, GPU, 코드 수정은 하지 않는다.

## 보고

끝나면 `WAITING: 구현 승인`으로 보고한다. 보고에 넣을 것:

- 표현할 수 없는 것
- 사용자 결정이 필요한 것
- 기존 회전 파일을 옮길 수 있는지
- 구현 순서 제안: W1 컨버터 → W3 로더 → 교차 검증
