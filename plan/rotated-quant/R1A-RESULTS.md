# R1a 결과: 컨버터 합성 텐서 부분 (2026-10-28)

상위 문서: [회전 가중치 양자화 계획](PLAN.md), [작업 안내](R-BRIEF.md)의 "R1a" 절. GPU, 실제 모델 파일, `llama-quantize` 실행 없이 했다. 테스트는 eval 이미지(`ai-tools/gufo-eval:7c2c5a728270`, Python 3.12.3, NumPy 2.3.5, PyYAML 6.0.1)에서 `python3 -I`로 돌렸다.

## 바꾼 파일

| 파일 | 내용 |
|---|---|
| `tools/rotquant/__init__.py` | 컨버터 버전 문자열(`rotquant 1`) |
| `tools/rotquant/hadamard.py` | 128점 정규화 Sylvester Hadamard, 그룹 인스턴스별 부호 벡터, `values·D·H`(가중치 `W' = W·D·H`와 입력 `x' = H·D·x`를 같은 함수로), BF16 RNE 반올림 |
| `tools/rotquant/gguf_io.py` | GGUF 헤더 읽기(메타데이터를 원래 바이트로 보관, 분할 GGUF), 단일 파일 GGUF v3 쓰기(텐서 스트리밍 콜백). 값 해석과 타입 표는 `tools/gufo/gguf.py`의 `Reader`, `GGML_TYPE_BLOCK`, `GGML_TYPE_NAME`을 그대로 쓴다 |
| `tools/rotquant/recipe.py` | `--recipe` YAML 파싱과 거부 |
| `tools/rotquant/plan.py` | 모델 정의 로드, 텐서별 계획, 회전 메타데이터 검사(로더 거부 조건의 Python판), dry-run 표, `llama-quantize` 명령줄 |
| `tools/rotquant/convert.py` | CLI. rotate 스트리밍, quantize 호출 |
| `tools/qwen27b/rotquant_groups.py` | 27B 입력 그룹 표, 역할, 역할별 지원 타입 |
| `tests/tools/rotquant_test.py` | 공통 테스트 20개(합성 텐서) |
| `tests/models/qwen27b/rotquant_groups_test.py` | 27B 그룹 표와 로더 지원 타입 일치 테스트 2개 |

`tools/gufo/gguf.py`는 바꾸지 않았다. 그 reader는 `tokenizer.*` 값을 건너뛰고 원래 바이트를 남기지 않아 복사에 쓸 수 없으므로, 같은 `Reader`로 원래 바이트 구간을 기록하는 헤더 파서를 `tools/rotquant/gguf_io.py`에 두었다. CMake/CTest 등록과 `tools/README.md` 표 갱신은 공유 파일이라 하지 않았다(아래 "남은 일").

## 사용법

```sh
D="docker run --rm -v /home/m2on/projects/gufo:/src -w /src --user $(id -u):$(id -g)"; I=ai-tools/gufo-eval:7c2c5a728270
C="$D --entrypoint python3 $I -I tools/rotquant/convert.py --groups qwen27b"

# 1) 표만 본다 (텐서별 역할, 그룹 인스턴스, 회전 여부, 원래/목표 타입, MiB, bpw, 합계)
$C --input Qwen3.8-27B-BF16-00001-of-00002.gguf --recipe rot-q4.yaml --seed 1 --dry-run

# 2) 회전 BF16만 만든다 (R2 ② 검증용, 그리고 여러 레시피가 공유하는 중간 파일)
$C --input Qwen3.8-27B-BF16-00001-of-00002.gguf --rotate-groups all --seed 1 \
   --rotate-only --output rot-bf16.gguf

# 3) 회전된 파일에서 양자화 (회전 단계를 건너뛴다. 회전 그룹은 파일과 같아야 한다)
$C --input rot-bf16.gguf --rotate-groups all --recipe rot-q4.yaml --imatrix rot.imatrix \
   --llama-quantize "$(command -v llama-quantize)" --output rot-q4.gguf

# 4) 같은 바이트 비교: UD-Q4_K_XL 헤더의 텐서별 타입을 그대로 쓴다 (헤더만 읽는다)
$C --input rot-bf16.gguf --rotate-groups all --copy-recipe-from Qwen3.8-27B-UD-Q4_K_XL.gguf \
   --llama-quantize "$(command -v llama-quantize)" --output rot-udq4.gguf
```

옵션:

- `--groups <모델>`: `tools/<모델>/rotquant_groups.py`. 없으면 거부한다. 입력의 `general.architecture`가 정의(`qwen35`)와 다르면 거부한다.
- `--input`: BF16 GGUF(분할이면 아무 조각이나), 또는 `--rotate-only`로 만든 회전 GGUF.
- `--recipe <yaml>`, `--rotate-groups all|none|a,b,...`(레시피의 `rotate`를 덮어쓴다). 둘 다 회전 그룹을 정하지 않으면 거부한다.
- `--seed`: 새로 회전할 때 필수. 부호는 `SeedSequence([seed, crc32("<그룹>:<층>")])`로 만들고 파일에 저장한다. 이미 회전된 입력에 주면 거부한다.
- `--rotate-only`: 회전 float GGUF만 쓴다. `--imatrix`, `--copy-recipe-from`, `--llama-quantize`와 함께 쓰면 거부한다. 레시피의 `types`는 이 모드에서 쓰지 않는다.
- `--copy-recipe-from <gguf>`: 다른 GGUF 헤더의 텐서별 타입. 레시피 `types`와 함께 쓰면 거부한다. 빠진 텐서나 컨버터가 만들 수 없는 타입이 있으면 거부한다.
- `--dry-run`: 표만 출력한다. `--output` 없이 쓸 수 있다.
- `--imatrix`, `--llama-quantize`: 양자화 단계에 넘긴다. 양자화(=`--rotate-only` 아님)는 `--llama-quantize`가 필수다.
- 출력은 덮어쓰지 않는다. 양자화하면 `--output` 옆에 `<이름>.tensor-types.txt`(규칙), `<이름>.recipe.json`(유효 레시피. 이 내용의 sha256이 `gufo.rotation.source_recipe_sha256`), 원본 BF16에서 바로 회전+양자화하면 `<이름>.rotated.gguf`(중간 파일)도 생긴다.

### 레시피 형식

```yaml
version: 1
rotate: [attn_in, ffn_in, ffn_down_in]   # all / none / 그룹 목록
types:                                   # 위에서부터 처음 맞는 규칙 (Python re.search)
  - match: 'attn_output|ssm_alpha|ssm_beta|^token_embd'
    type: Q8_0
  - match: 'ssm_out'
    type: Q6_K
  - match: '\.weight$'
    type: Q4_K
```

- 규칙은 레시피 역할(임베딩, projection) 텐서에만 적용한다. norm과 SSM 파라미터는 원래 타입을 유지하고, 그 타입이 로더 지원 타입(F32)이 아니면 거부한다.
- 레시피 역할 텐서는 모두 규칙에 맞아야 한다(기본 타입 없음). 아무 텐서에도 쓰이지 않은 규칙도 거부한다(오타 방지).
- 허용 타입: F32, BF16, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, IQ4_NL, IQ4_XS, IQ3_S(대문자 이름 그대로). `weights.cpp`의 projection 목록에 있는 Q8_K는 활성화용 형식이라 `llama-quantize` 대상이 될 수 없어 제외했다. 테스트가 `weights.cpp`의 `SupportsTensorType` 목록과 이 집합을 비교한다.
- 양자화 블록이 행(K)을 나누지 않으면 거부한다. `llama-quantize`는 이때 경고만 하고 다른 타입으로 조용히 바꾸므로(`tensor_type_fallback`), 컨버터가 먼저 막는다. 예: K = 384인 텐서에 Q4_K.
- 알 수 없는 키, `version` ≠ 1, 잘못된 정규식, 모르는 그룹, 중복 그룹, 지원 밖 타입(Q2_K, Q8_K, 소문자 `q4_k` 포함)은 거부한다.

### dry-run 표 예시 (합성 2층 모델, 위 레시피)

`shape`은 NumPy 순서(N×K, GGUF `ne`의 역순)다. `rot`은 이번 실행에서 회전하면 `yes`, 이미 회전된 입력이면 `kept`다.

```text
tensor                                     shape role          group           rot  source  target         MiB    bpw
---------------------------------------------------------------------------------------------------------------------
token_embd.weight                         64x256 embedding     -               no   BF16    Q8_0          0.02  8.500
output_norm.weight                           256 norm          -               no   F32     F32           0.00 32.000
output.weight                             64x256 projection    head_in         no   BF16    Q4_K          0.01  4.500
blk.0.attn_norm.weight                       256 norm          -               no   F32     F32           0.00 32.000
blk.0.post_attention_norm.weight             256 norm          -               no   F32     F32           0.00 32.000
blk.0.attn_qkv.weight                    384x256 projection    attn_in:0       yes  BF16    Q4_K          0.05  4.500
blk.0.attn_gate.weight                   256x256 projection    attn_in:0       yes  BF16    Q4_K          0.04  4.500
blk.0.ssm_alpha.weight                     4x256 projection    attn_in:0       yes  BF16    Q8_0          0.00  8.500
blk.0.ssm_beta.weight                      4x256 projection    attn_in:0       yes  BF16    Q8_0          0.00  8.500
blk.0.ssm_out.weight                     256x256 projection    ssm_out_in:0    no   BF16    Q6_K          0.05  6.562
blk.0.ssm_a                                    4 ssm_parameter -               no   F32     F32           0.00 32.000
blk.0.ssm_conv1d.weight                    384x4 ssm_parameter -               no   F32     F32           0.01 32.000
blk.0.ssm_dt.bias                              4 ssm_parameter -               no   F32     F32           0.00 32.000
blk.0.ssm_norm.weight                         64 norm          -               no   F32     F32           0.00 32.000
blk.0.ffn_gate.weight                    512x256 projection    ffn_in:0        yes  BF16    Q4_K          0.07  4.500
blk.0.ffn_up.weight                      512x256 projection    ffn_in:0        yes  BF16    Q4_K          0.07  4.500
blk.0.ffn_down.weight                    256x512 projection    ffn_down_in:0   yes  BF16    Q4_K          0.07  4.500
blk.1.attn_norm.weight                       256 norm          -               no   F32     F32           0.00 32.000
blk.1.post_attention_norm.weight             256 norm          -               no   F32     F32           0.00 32.000
blk.1.attn_q.weight                      512x256 projection    attn_in:1       yes  BF16    Q4_K          0.07  4.500
blk.1.attn_k.weight                      128x256 projection    attn_in:1       yes  BF16    Q4_K          0.02  4.500
blk.1.attn_v.weight                      128x256 projection    attn_in:1       yes  BF16    Q4_K          0.02  4.500
blk.1.attn_output.weight                 256x384 projection    attn_out_in:1   no   BF16    Q8_0          0.10  8.500
blk.1.attn_q_norm.weight                      64 norm          -               no   F32     F32           0.00 32.000
blk.1.attn_k_norm.weight                      64 norm          -               no   F32     F32           0.00 32.000
blk.1.ffn_gate.weight                    512x256 projection    ffn_in:1        yes  BF16    Q4_K          0.07  4.500
blk.1.ffn_up.weight                      512x256 projection    ffn_in:1        yes  BF16    Q4_K          0.07  4.500
blk.1.ffn_down.weight                    256x512 projection    ffn_down_in:1   yes  BF16    Q4_K          0.07  4.500
---------------------------------------------------------------------------------------------------------------------
tensors: 28  rotated: 13  rotated groups: attn_in, ffn_in, ffn_down_in
weights: 0.001 GiB, 5.0080 bpw
.rot sign tensors (F16): 13, 0.007 MiB
expected data size: 0.001 GiB (+ header and padding)
llama-quantize base type: Q4_K
```

## 설계 결정

- **`.rot` 저장 타입 = F16.** 값 ±1을 F16(`0x3C00`/`0xBC00`)으로 저장한다. gufo reader의 `GgmlType`(`src/core/gguf_reader.hpp:23`)에 I8이 없으므로 F16이면 R2 로더가 reader를 고치지 않고 읽을 수 있다. 크기 차이는 무시할 만하다: 27B 전체에서 짝 텐서는 텐서당 K × 2바이트(5120 → 10 KiB, 17408 → 34 KiB)로 합계 수 MiB 수준이다(짐작: 64층 × 그룹 텐서 수로 계산한 값, 실제 파일로는 확인 안 함). 검사기는 F16이 아니거나 ±1 밖의 값이 있으면 거부한다.
- **그룹 인스턴스 = 그룹 × 층.** 같은 층의 같은 그룹(예: 0층 `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta`)은 부호를 공유하고, 층이 다르면 부호도 다르다. `head_in`은 인스턴스 하나다.
- **이미 회전된 입력은 회전 단계를 건너뛴다.** 회전 BF16 파일(약 55 GB)을 한 번 만들고 여러 레시피(E1~E6)가 공유하게 해서 D2의 디스크 예산을 지킨다. 이때 요청한 회전 그룹이 파일의 회전 그룹과 다르면 거부한다.
- **메타데이터.** 회전된 텐서가 하나라도 있으면 표시 규칙 v1의 네 키(`version` u32 1, `kind` "hadamard", `block_size` u32 128, `side` "input")를 쓴다. 출처 키 `gufo.rotation.source_converter`, `source_recipe_sha256`, `source_sha256`(입력 바이트, 분할이면 조각을 순서대로 이은 바이트의 sha256), `source_file`(첫 조각 파일 이름)은 항상 쓴다. 넷 다 문자열이라 `llama-quantize --override-kv`(127자 제한)로 넘길 수 있다. 회전 그룹이 없으면 네 표시 키는 쓰지 않는다. 그래서 로더의 "표시 KV ↔ `.rot`" 짝 검사는 `gufo.rotation.version` 등 네 키만 봐야 하고, `source_*`는 표시로 보지 않아야 한다(R2에서 지킬 것).
- **출력은 단일 파일.** 분할 입력의 `split.*` 키는 뺀다. 토크나이저를 포함한 나머지 메타데이터는 원래 바이트를 그대로 복사한다.
- **회전 계산.** 행 묶음(64 MiB 분량) 단위로 BF16 → FP32, `(W∘d)`를 128열 블록으로 나눠 FP32 Hadamard를 곱하고 BF16 RNE로 반올림한다. 입력 텐서가 F16이면 F16, F32이면 F32로 쓴다. 비유한 값이나 F16 범위 초과는 거부한다. 양자화된 텐서를 회전하려 하면 거부한다.
- **quantize 명령줄.** 기준 llama.cpp(`68d9053a`, b11069)의 `tools/quantize/quantize.cpp`와 `src/llama-quant.cpp`를 읽고 정했다(실행은 안 함).
  ```text
  llama-quantize --pure --tensor-type-file <out>.tensor-types.txt [--imatrix F]
      --override-kv gufo.rotation.source_converter=str:rotquant 1
      --override-kv gufo.rotation.source_recipe_sha256=str:<sha256>
      [--override-kv gufo.rotation.source_sha256=str:... --override-kv gufo.rotation.source_file=str:...]   # 회전 없이 원본에서 바로 양자화할 때만
      <rotated.gguf> <out.gguf> <기본 타입>
  ```
  - 2차원 이상 텐서마다 `^blk\.0\.attn_qkv\.weight$=Q4_K` 같은 정확한 규칙 한 줄을 쓴다(F32로 남길 `ssm_conv1d`도 `=F32`로 명시). 그래서 이름 기반 혼합 휴리스틱(`llama_tensor_get_type_impl`)이 끼어들 틈이 없다. `--pure`는 이중 안전장치다.
  - `llama-quantize`는 패턴을 소문자로 바꾸므로 이름이 소문자가 아니면 거부한다.
  - 기본 타입(위치 인자)은 바이트가 가장 큰 양자화 타입이다. `llama-quantize`는 기본 타입이 양자화 타입일 때만 `--tensor-type` 규칙을 적용하므로, 모든 텐서가 float이면 거부한다(`--rotate-only`를 쓰라고 안내).
  - 인자가 수백 개라 `--tensor-type` 대신 같은 형식의 `--tensor-type-file`을 쓴다.

## 테스트

```sh
D="docker run --rm -v /home/m2on/projects/gufo:/src -w /src --user $(id -u):$(id -g) -e PYTHONDONTWRITEBYTECODE=1"; I=ai-tools/gufo-eval:7c2c5a728270
$D --entrypoint python3 $I -I tests/tools/rotquant_test.py -v            # Ran 20 tests ... OK
$D --entrypoint python3 $I -I tests/models/qwen27b/rotquant_groups_test.py  # Ran 2 tests ... OK
```

| 테스트 | 확인하는 것 | 결과 |
|---|---|---|
| `test_orthogonal_symmetric_normalized` | `H·Hᵀ = I`, `H = Hᵀ`, `H·H = I`, 원소 ±1/√128 (1e-12) | 통과 |
| `test_signs_are_deterministic_per_instance` | 같은 시드·인스턴스는 같은 부호, 층·시드가 다르면 다름, K가 128 배수가 아니면 거부 | 통과 |
| `test_rotation_preserves_matmul_fp32` | K = 128, 640, 5120에서 FP32 `W'·(H·D·x)`와 FP64 `W·x`의 상대 오차 < 1e-5, 역변환 `W'·H·D = W` | 통과 |
| `test_bf16_storage_error_is_rounding_level` | BF16 저장 뒤 원소 오차 ≤ 2⁻⁸·|값|, matmul 상대 오차 < 4e-3(그리고 0이 아님) | 통과 |
| `test_bf16_round_to_nearest_even` | RNE 동률 처리, NaN 거부 | 통과 |
| `RecipeTest` 4개 | 레시피 파싱, 13가지 거부, `--rotate-groups` 정규화, 정의 없는 모델(`nosuch`, `../qwen27b`, 빈 이름) 거부 | 통과 |
| `test_split_input_is_read_as_one_model` | 2조각 분할 GGUF(조각 0에만 메타데이터, `split.no`/`count` u16, `split.tensors.count` i32)를 한 모델로 읽고 바이트 일치 | 통과 |
| `test_rotate_only_roundtrip` | 분할 입력 → `--rotate-only` → 독립 reader(`tools/gufo/gguf.py`)로 네 표시 KV, 출처 KV, `split.*` 제거, `.rot` 집합·타입(F16)·모양(`[K]`) 확인. 그룹 안 부호 일치(0층 `attn_in` 4개, 1층 `attn_q`/`attn_v`, `ffn_gate`/`ffn_up`), 층 간 부호 다름. 모든 회전 텐서에서 `W'·(H·D·x) ≈ W·x`(상대 < 4e-3), 저장값이 FP32 경로의 BF16 반올림과 비트 단위로 같음, 비회전 텐서와 토크나이저 KV는 바이트 그대로 | 통과 |
| `test_partial_groups_and_no_rotation` | 일부 그룹만 회전, 회전 없음이면 표시 키 없이 출처 키만 | 통과 |
| `test_rotation_metadata_rejections` | 부호 하나 뒤집기, `version` 없음, `block_size` 64, `token_embd`에 `.rot`, 그룹 인스턴스 안 일부만 회전, 한 층만 빠진 그룹, ±1 아닌 값, 길이 틀린 `.rot`, 표시 키만 있고 `.rot` 없음, `.rot`만 있고 표시 키 없음 → 모두 이름을 들어 거부 | 통과 |
| `test_plan_rejections` | 묶인(tied) 헤드 회전, 다른 아키텍처, 시드 없음, 회전 그룹 미지정, 블록이 K를 나누지 않음, 쓰이지 않은 규칙, 타입 없는 텐서, 회전 파일과 다른 그룹 요청, 회전 파일에 `--seed`, 회전 파일에 `--rotate-only`, `--llama-quantize` 없음, `--rotate-only`와 `--llama-quantize` 함께, 덮어쓰기 → 모두 종료 코드 2와 해당 메시지 | 통과 |
| `test_float_target_types_follow_loader_roles` | BF16 norm은 로더 norm 역할(F32만)에 맞지 않아 거부 | 통과 |
| `test_dry_run_table` | 표 행(모양, 역할, 인스턴스, 회전, 타입, MiB, bpw 4.500), 합계, `.rot` 수(13), 기본 타입, rotate-only 표 | 통과 |
| `test_copy_recipe_from_header` | 헤더의 타입(Q5_K, IQ4_XS, Q8_0)이 그대로 목표 타입이 됨, 레시피 `types`와 함께 쓰면 거부, 헤더에 빠진 텐서 거부 | 통과 |
| `test_quantize_command_from_rotated_file` | 가짜 `llama-quantize`(argv 기록)로 정확한 argv, `.recipe.json`의 sha256 = override 값, 규칙 파일(행렬마다 한 줄, `.rot` 없음, 소문자, `ssm_conv1d=F32`), 중간 파일 안 만듦 | 통과 |
| `test_quantize_from_bf16_rotates_first` | 원본에서 회전+양자화: 중간 `.rotated.gguf`를 만들고 그것을 입력으로 넘김. 회전 없음이면 원본을 직접 넘기고 `source_sha256`/`source_file`을 override로 넘김 | 통과 |
| `test_quantize_command_rejects_float_only_recipe` | 모든 텐서가 BF16이면 명령 생성 거부 | 통과 |
| `test_plan_table`(27B) | 64층(4층마다 attention) 이름 전체가 계획 표의 6개 그룹·역할에 맞음 | 통과 |
| `test_supported_types_follow_the_loader`(27B) | 역할별 허용 타입이 `weights.cpp` `SupportsTensorType`과 같음(projection은 Q8_K 제외) | 통과 |

처음 실행에서 `test_rotate_only_roundtrip` 하나가 실패했다. 테스트가 FP64 회전값과 저장값의 차이를 반 ulp(2⁻⁸·|값|)로만 허용했는데, 컨버터는 FP32로 회전하므로 반올림 경계 근처 값에서 FP32 누적 오차만큼 넘었다. 허용치에 FP32 오차 항(1e-6 × 최대값)을 더하고, 대신 FP32 경로와의 비트 단위 일치를 추가로 검사하도록 테스트를 고쳤다. 컨버터 코드는 바꾸지 않았다.

## R1b에서 확인할 것 (실제 파일과 llama.cpp 필요)

아래 `llama-quantize` 관련 항목은 기준 리비전 `68d9053a`의 소스를 읽고 세운 **짐작**이다. 실행으로 확인한 것은 없다.

1. **`.rot` 복사.** `.rot`은 1차원이라 `tensor_allows_quantization`이 `ggml_n_dims < 2`에서 false를 돌려 원래 타입(F16)으로 복사될 것이다(짐작). 출력에서 `.rot` 타입·값이 그대로인지 바이트로 비교한다. 출력 텐서 순서는 `weights_map`(이름 순 map) 순서로 바뀔 수 있다(짐작). 순서가 바뀌어도 이름 기준 검사는 영향이 없다.
2. **이름 기반 분류.** `tensor_get_category`는 부분 문자열로 분류하므로 `blk.N.attn_v.weight.rot`도 ATTENTION_V로 세어 `n_attention_wv`가 늘어난다(짐작). `--pure`와 전 텐서 수동 규칙에서는 이 카운터가 타입 선택에 쓰이지 않을 것이다. 실제 양자화 로그에 `applying manual override` 외의 타입 변경이나 fallback 경고(`not divisible by`)가 없는지, 출력 헤더의 타입이 `.tensor-types.txt`와 모두 같은지 확인한다(이 비교 검사를 R1b에 추가한다).
3. **로더의 추가 텐서 허용.** `llama-quantize`의 `llama_model_loader`(`check_tensors=true`)는 `load_hparams`까지만 해서 `.rot`을 거부하지 않을 것이다(짐작). 실제 실행으로 확인한다.
4. **llama.cpp 로드 거부.** 회전 GGUF(BF16과 양자화 둘 다)를 `llama-server`/`llama-cli`가 텐서 수 불일치(`wrong number of tensors`)로 거부하는지 확인하고 메시지를 기록한다.
5. **`--override-kv`.** `gufo.*` 키가 새로 추가되거나 바뀌는지, 회전 파일의 다른 `gufo.rotation.*` 키가 그대로 복사되는지 출력 헤더로 확인한다.
6. **실제 27B dry-run.** unsloth BF16 원본(분할)으로 `--dry-run`을 돌려 역할 `-`(로더가 쓰지 않는 텐서, 예: MTP나 비전 텐서가 있다면)가 있는지, norm·SSM 파라미터가 F32인지(아니면 컨버터가 거부한다), 그룹 텐서 수(층 64 기준)를 확인한다. 역할 없는 2차원 텐서가 있으면 원래 타입으로 규칙에 들어가므로 양자화되지 않는다.
7. **`--copy-recipe-from` UD-Q4_K_XL.** UD 헤더에 컨버터가 만들 수 없는 타입(예: Q8_K)이 있는지, 모든 레시피 역할 텐서가 있는지 확인한다.
8. **회전 시간과 메모리.** 27B 회전 BF16 생성 시간, 최대 RSS, 입력 sha256 시간(약 55 GB 두 번 읽기: 해시 한 번, 변환 한 번)을 잰다. 다른 세션의 GPU 창이 없을 때 돌린다.
9. **회전 BF16의 CPU 검증(R1 통과 기준).** 실제 파일에서 몇 텐서를 골라 `W'·(H·D·x) ≈ W·x`를 NumPy로 확인한다(테스트와 같은 방법).
10. **imatrix 이름.** R3 imatrix의 항목 이름이 회전 파일의 텐서 이름(`blk.N.ffn_up.weight`)과 맞는지, `.rot`에 대한 항목을 요구하지 않는지 확인한다.
11. **E7a dequant 경로.** 계획의 "E7a용 dequant 경로 확인"은 R1a 범위 밖이라 하지 않았다.

## 남은 일

- CTest 등록(`CMakeLists.txt`)과 `tools/README.md`의 `rotquant/` 행 추가는 공유 파일이라 하지 않았다. 등록하려면 NumPy·PyYAML이 있는 환경에서만 돌게 해야 한다. Nix 개발 셸의 Python에는 `ps.numpy`가 있고 PyYAML은 `transformers` 의존성으로 들어올 것이다(짐작, 확인 안 함).
- 정식 표시 규칙 문서(`docs/`)는 R2에서 로더와 함께 쓴다.

## 감독 확인과 기록 (2026-10-28)

- 감독이 GPU 창이 빈 상태에서 eval 이미지(`7c2c5a728270`)로 두 테스트를 다시 돌렸다: `tests/tools/rotquant_test.py` 20/20, `tests/models/qwen27b/rotquant_groups_test.py` 2/2 통과. 회전 수학(`hadamard.py`)을 읽어 `W' = W·D·H`가 K 축에 적용되고 BF16 반올림이 round-to-nearest-even인 것을 확인했다.
- **커밋 경계:** 이 단계의 코드와 테스트 9개 파일은 R0b 커밋 `d19126df`(`feat(bench): …`)에 함께 들어갔다. 다른 세션이 같은 브랜치에서 작업 중이고 결과 문서와 창 기록이 `d19126df`를 가리키므로 이력을 고치지 않았다(사용자 결정 2026-10-28).
- **CTest 등록은 R1b로 미룬다.** 저장소의 Python 테스트는 `CMakeLists.txt`에 `add_test`로 등록하지만, Nix 검사 환경(`flake.nix` `mkTestCheck`)의 Python에는 PR 검사에 NumPy가 없고 전체 검사에도 PyYAML이 없다. 등록 전에 레시피 형식을 JSON으로 바꿀지(PyYAML 의존 제거), 검사 환경에 패키지를 더할지 정한다.
