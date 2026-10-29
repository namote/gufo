# R1b 결과: 컨버터 실모델 (W1, 2026-10-29)

상위 문서: [W1 브리프](W1-BRIEF.md), [R1a 결과](R1A-RESULTS.md)의 "R1b에서 확인할 것" 11항목과 브리프의 12항목. CPU 단계 1은 감독의 허락을 받아 진행했다. 감독이 lock을 잡았고, GPU는 쓰지 않았다.

## 요약

- 회전 BF16 GGUF를 만들었다. 블록 0–63의 6개 그룹, 텐서 497개다. 실제 텐서 23개 표본에서 저장값이 `BF16-RNE(W·D·H)`와 비트 단위로 같았다(행 전체). `W'·(H·D·x)`와 `W·x`의 상대 L2 오차는 1.7e-3(중앙값)이고 최대 2.6e-3이다. 이는 BF16 반올림 수준이다.
- 기준 `llama-quantize`(`68d9053a`)는 회전 파일을 그대로 양자화했다.
  - `.rot` 497개는 F16 그대로 바이트가 같다.
  - 출력 타입은 레시피(UD-Q4_K_XL 복사)와 모두 같다. fallback 경고는 없다.
  - `gufo.*` KV도 모두 보존됐다.
- llama.cpp는 회전 파일(BF16, Q4)을 `wrong number of tensors; expected 1363, got 866`으로 거부한다. 비회전 대조군 Q4는 정상으로 로드하고 생성한다.
- unsloth BF16 GGUF는 HF `Qwen/Qwen3.8-27B`(리비전 `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`)와, llama.cpp Qwen3.5 변환 규칙 안에서 비트 단위로 같다. PROVENANCE의 원본 리비전 unknown이 풀렸다.
- 결정 반영(감독, 2026-10-29):
  - NextN(MTP) 블록 64는 회전하지 않는다.
  - 레시피는 JSON만 받는다.
  - CTest는 전체 점검에만 등록한다.

## 바꾼 파일과 커밋

| 커밋 | 내용 |
|---|---|
| `df84baa3` refactor(rotquant): read recipes as JSON instead of YAML | `--recipe`는 JSON만 받는다. 중복 키, `version` 1.0/true, 문법 오류는 거부한다. PyYAML 의존이 없어졌다. 테스트 레시피도 JSON으로 바꿨다 |
| `28b8c180` feat(rotquant): skip NextN blocks and check llama-quantize output against the plan | 아래 세 가지를 바꿨다 |
| `e91e30a2` test(rotquant): register the converter and 27B group tests in CTest | `rotquant_test`(루트 `CMakeLists.txt`), `qwen27b.rotquant-groups`(`tests/models/qwen27b/CMakeLists.txt`). 라벨은 `cpu;tools;quantization`이다. hosted `check-pr`에는 넣지 않았다. `flake.nix`는 바꾸지 않았다(전체 점검 Python에 이미 numpy가 있다). `tools/README.md`의 `rotquant/` 행도 고쳤다. 공유 파일이라 내 hunk만 `git apply --cached`로 담았다 |
| `9e70cbed` feat(rotquant): add a real-tensor check for rotated GGUF files | `tools/rotquant/check_rotated.py`(항목 9 도구)와 합성 파일 테스트 |

`28b8c180`의 내용:
- `rotquant_groups.rotated_layers(kv)` = `block_count - nextn_predict_layers`. `Groups.for_model`이 헤더에서 범위를 읽고, 범위 밖 블록은 그룹이 없는 것으로 본다. 그래서 그 블록에 `.rot`이 있으면 "role has no rotation path"로 거부한다. `block_count`가 없으면 거부한다.
- `tools/rotquant/verify.py`: 양자화 뒤 출력을 다시 읽어 검사한다.
  - 텐서 집합이 원본과 같은지
  - 타입이 계획과 같은지
  - 타입이 바뀌지 않은 텐서와 `.rot`의 바이트가 원본과 같은지
  - `gufo.*` KV와 `--override-kv` 값이 맞는지
  - `convert.py`가 `llama-quantize` 직후 이 검사를 자동으로 부른다. 어긋나면 종료 코드 2로 거부한다.
- 테스트: 가짜 `llama-quantize`가 실제 출력 GGUF를 쓰게 바꿨다. 잘못된 타입과 `.rot` 변조를 거부하는지 확인한다. NextN 블록 제외 테스트도 더했다.

테스트(이미지 `ai-tools/gufo-eval:7c2c5a728270`):
- `tests/tools/rotquant_test.py` 23/23, `tests/models/qwen27b/rotquant_groups_test.py` 3/3
- CTest: `ai-tools/gufo-eval:05e527d2cae5-rocm101`에서 `-DENGINE_ENABLE_HIP=OFF`로 configure했다(`build/rq-llama/gufo-ctest`). `ctest -R rotquant` 결과는 2/2 Passed다. `7c2c5a728270` 이미지에는 `libsndfile` 개발 파일이 없어 gufo configure가 실패한다(`sndfile` not found).

커밋하지 않은 보조 스크립트는 `build/rq-llama/` 아래에 있다: `hfcheck.py`(항목 12), `e7a/roundtrip.c`와 `e7a/compare.py`(항목 11). 로그는 모두 `build/rq-llama/logs/`에 있다.

## llama.cpp 기준 빌드

- 방법: eval 이미지 안에서 CMake로 빌드했다. nixos/nix 컨테이너는 쓰지 않았다.
- 소스: GitHub tarball `68d9053afd4f4d0752ced6187585f862355a40be`(sha256 `7ec6ef2e02c29a15879cd4c997b55976f1f0bbce4a042de15b71b376df52dc40`)를 `build/rq-llama/src`에 풀었다. `.devops/nix/llama-cpp-reference.nix`의 리비전과 같다. Nix 해시(NAR)와 직접 대조하지는 않았다.
- CPU 빌드다. 로드 거부만 보면 되므로 HIP은 필요 없다.
  ```
  cmake -S src -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DGGML_HIP=OFF -DGGML_NATIVE=ON -DLLAMA_OPENSSL=OFF \
        -DLLAMA_BUILD_UI=OFF -DLLAMA_USE_PREBUILT_UI=OFF -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_COMMIT=68d9053a
  cmake --build build -j16 --target llama-quantize llama-completion llama-cli llama-simple
  ```
  GCC 13.3, 29 s 걸렸다. RUNPATH가 `/w/build/bin`이라 컨테이너에서 `build/rq-llama`를 `/w`로 마운트해 실행한다.
- gguf-py는 `build/rq-llama/src/gguf-py`를 썼다. 이미지의 pip `gguf` 0.19.0은 쓰지 않았다.

## 산출물

| 파일 | 크기 | sha256 |
|---|---:|---|
| `qwen-3.8-27b-rot-bf16-gguf/Qwen3.8-27B-rot-BF16.gguf` | 54664557760 | `7b4edef728e317e4ad51f3e405eb302d108a69cbfb05650a61164c27c03402e0` |
| `qwen-3.8-27b-rot-q4-test-gguf/Qwen3.8-27B-rot-UDcopy-Q4.gguf` | 17627849920 | `76555d9a45365743ca56ce36099cb31eed19a07fde5fb3805cf83a5ecc633602` |
| `qwen-3.8-27b-rot-q4-test-gguf/Qwen3.8-27B-plain-UDcopy-Q4.gguf` | 17621027936 | `1f97624d574f772825c7809244cd324146b7eb18cc00500d3dfd2bc76481d56c` |

- 각 폴더(`~/projects/ai.tools/storage/models/llm/` 아래)에 `PROVENANCE.md`와 `SHA256SUMS`를 두었다. Q4 폴더에는 `*.tensor-types.txt`와 `*.recipe.json`도 있다.
- 회전 BF16의 KV: `gufo.rotation.source_sha256 = ac6d62ca…7d3c`(두 조각을 이은 바이트), `source_recipe_sha256 = d6232be4…9e84`
- 시드 1, 그룹 all
- 루트 여유 공간: 394 GB에서 306 GB가 됐다.

## 항목별 결과

### 1. `.rot` 복사
`llama-quantize`는 1차원 `.rot`을 `type = f16, size = 0.010 MiB`로 복사했다. 컨버터 검사에서 497개 모두 바이트가 같았다. 텐서 순서는 입력과 다르다(`output.weight`가 맨 앞에 오는 등 llama의 순서). 이름 기준 검사에는 영향이 없다.

### 2. 이름 기반 분류와 fallback
- 두 양자화 로그(회전, 비회전)에서 `applying manual override`가 각각 315회로 같다. `not divisible by`나 fallback 경고, 다른 타입 변경은 없다.
- 출력 헤더 타입은 `.tensor-types.txt`와 모두 같다(`verify.py`). 회전 Q4와 비회전 Q4의 가중치 타입은 텐서마다 같다.
- `.rot` 이름(`…attn_v.weight.rot` 등)이 `n_attention_wv` 같은 카운터를 늘리는지는 로그로는 보이지 않는다. 그래도 `--pure`와 전 텐서 수동 규칙 아래에서 결과 타입에는 영향이 없음을 출력 타입 비교로 확인했다.

### 3. `llama-quantize` 로더의 추가 텐서 허용
회전 파일(텐서 1363개)을 거부 없이 읽어 양자화했다(quantize time 141.0 s).

### 4. llama.cpp 로드 거부
`llama-completion -m <파일> -p "The capital of France is" -n 12 --temp 0 -no-cnv -t 16 --no-warmup`로 확인했다.

| 파일 | 결과 |
|---|---|
| 회전 BF16 | 거부. `llama_model_load: error loading model: done_getting_tensors: wrong number of tensors; expected 1363, got 866` |
| 회전 Q4 | 거부. 같은 메시지 |
| 비회전 Q4(대조군) | 로드. 출력 "Paris. / The capital of Germany is Berlin.", CPU decode 5.98 tok/s |

로그는 `build/rq-llama/logs/load-{rot-bf16,rot-q4,plain-q4}.log`에 있다.

### 5. `gufo.*` KV
- 회전 Q4의 키 집합은 회전 BF16과 같다. 바뀐 키는 `general.file_type`(32에서 17)과 `gufo.rotation.source_recipe_sha256`(override)뿐이다.
- 네 표시 키와 `source_sha256`, `source_file`, `source_converter`는 그대로다.
- 비회전 Q4에는 표시 키 없이 `source_*` 네 개만 있다(override). llama.cpp는 이 파일을 정상으로 로드한다.

### 6. 실제 27B dry-run
표: `build/rq-llama/logs/dry-rot-bf16.txt`

- 텐서 866개. 원래 타입은 norm 213개, SSM 파라미터 144개 모두 F32라 거부되는 것이 없다.
- 역할 없는 텐서(`-`)는 4개다. `blk.64.nextn.eh_proj.weight`(BF16 10240→5120, 100 MiB)와 `blk.64.nextn.{enorm,hnorm,shared_head_norm}.weight`(F32).
- `qwen35.block_count = 65`, `nextn_predict_layers = 1`이다. 블록 64(MTP)는 attention, FFN 텐서가 "projection" 역할이지만 그룹은 `-`이고 회전하지 않는다(결정 (a)).
- 그룹별 회전 텐서 수:

  | 그룹 | 수 | 비고 |
  |---|---:|---|
  | attn_in | 240 | 48×4 + 16×3 |
  | ssm_out_in | 48 | |
  | attn_out_in | 16 | |
  | ffn_in | 128 | |
  | ffn_down_in | 64 | |
  | head_in | 1 | |

  합계는 497개, `.rot`은 6.48 MiB다.
- 데이터는 50.900 GiB다. 결정 전 dry-run에서는 블록 64까지 포함해 504개였다.

### 7. `--copy-recipe-from` UD-Q4_K_XL
- 가능하다. 컨버터가 만들 수 없는 타입은 없다.
- 레시피 역할 텐서 505개의 타입 분포:

  | 타입 | 수 |
  |---|---:|
  | Q8_0 | 110 |
  | Q6_K | 55 |
  | Q5_K | 191 |
  | Q4_K | 68 + 임베딩 1 |
  | IQ4_XS | 70 |
  | IQ4_NL | 6 |
  | Q3_K | 3 |
  | IQ3_S | 1 |

- 기본 타입은 Q5_K이고 quant size는 16800.71 MiB(5.16 BPW)다.
- 차이 하나: UD는 `blk.64.nextn.eh_proj.weight`가 Q6_K인데, 우리 출력에서는 27B 로더 역할이 없어 BF16으로 남는다(+약 60 MiB). 회전 Q4와 비회전 Q4는 같은 조건이다.
- 시험 파일은 imatrix 없이 만들었다. UD 원본은 imatrix(항목 496개)로 만든 파일이라 UD와 품질이 같지 않다.

### 8. 시간, 메모리, 해시

| 단계 | 경과 | 최대 RSS | 비고 |
|---|---:|---:|---|
| 원본 조각별 `sha256sum -c` | 80.9 s | 7 MB | OK(디스크에서 읽음) |
| 회전 BF16(입력 sha256 + 회전 + 쓰기) | 159.5 s | 51.3 GB | user 285.5 s. RSS는 mmap 입력 파일 페이지 포함 |
| 위 입력 sha256만 따로(조각을 이어서, Python) | 92.7 s | — | 회전 시간의 대부분이 해시와 I/O라는 뜻이다(짐작: 회전 계산 자체는 약 70–100 s) |
| 회전 BF16 출력 sha256sum | 29.1 s | — | 페이지 캐시 |
| 회전 Q4(llama-quantize, 32스레드) | 141.8 s | 13.4 GB | user 3089 s |
| 비회전 Q4(원본 sha256 + llama-quantize) | 267.6 s | 13.4 GB | quantize 171.5 s |
| 항목 9 검사(16 텐서, `output.weight` 전체 포함) | 50.3 s | 42.5 GB | float32 사본 |

### 9. 회전 BF16의 CPU 검증
도구는 `tools/rotquant/check_rotated.py`다. 결과는 `build/rq-llama/logs/check-rotated{,-attn}.{log,json}`에 있다.

- 표본: 기본 표본은 그룹마다 첫·중간·끝 인스턴스에서 16개다. 큰 attn_in 텐서 7개를 따로 더했다(`blk.0.attn_qkv`, `blk.0.attn_gate`, `blk.3.attn_q`, `blk.3.attn_k`, `blk.33.attn_qkv`, `blk.63.attn_q`, `blk.62.attn_gate`). 모든 행을 검사했다(`--rows 0`).
- 부호: `.rot`은 시드 1에서 다시 만든 부호와 23개 모두 같다. 파일 전체에 대한 `check_rotation`도 통과했다(497개 `.rot`이 ±1이고 그룹 인스턴스 안에서 같으며, 일부 층만 회전한 그룹이 없다).
- 비트 일치: 저장값은 원본 BF16 → FP32 → `W·D·H`(FP32) → BF16 RNE로 다시 계산한 값과 23개 텐서의 모든 행에서 같다(불일치 0).
- 가중치 반올림 오차: `|W'_bf16 − W'_fp32| / |W'_fp32|`의 최대는 3.891e-3이고, 반 ulp 한도 2⁻⁸ = 3.906e-3 안이다. p99 최댓값은 3.46e-3이다.
- 출력 오차: 무작위 x 4개와, 이상치 x 4개(무작위 x에서 채널 8개를 100배 키움)로 쟀다.
  - 상대 L2 오차 184개 값: 최소 8.9e-4, 중앙값 1.67e-3, 최대 2.61e-3
  - 최댓값은 행이 48개뿐인 `blk.32.ssm_beta`다.
  - 큰 텐서는 모두 1.65e-3–1.89e-3이다.
  - 이상치 x와 무작위 x의 차이는 없다.
- 판단: BF16 가중치 반올림(상대 rms 약 2⁻⁸/√3 ≈ 2.3e-3 이하)과 같은 수준이다. 회전 계산의 오류로 볼 수 있는 큰 값은 없다.
- 원소별 오차를 열 RMS로 나눈 값: p50 약 1.1e-3, p99 약 4.5e-3, 최대 4.6e-2(`output.weight` 248320행의 꼬리)

### 10. imatrix 이름 (R3 수집기 설계 메모, 소스 읽기)
- llama.cpp `llama-imatrix`(`tools/imatrix/imatrix.cpp`)는 matmul의 `src0->name`, 즉 가중치 텐서 이름(`blk.N.ffn_up.weight` 등)을 키로 쓴다. 기본 형식은 GGUF이고, 텐서마다 `<name>.in_sum2`(K × 전문가 수)와 `<name>.counts`를 저장한다. 기록하는 이름은 `blk.`로 시작하는 것과(옵션이 있으면) `output.weight`다.
- `llama-quantize`는 `common_imatrix_load`로 읽어 sum2/count로 정규화한다. **양자화되는 텐서만** `imatrix_data->find(remapped_imatrix_name)`로 찾는다. 1차원 `.rot`은 복사 경로라 항목을 요구하지 않는다.
- 크기는 `ne[0] × ne[2]`(K)와 같아야 한다. 다르면 예외가 난다(`token_embd`는 예외).
- **주의:** 항목이 없는 텐서는 `did not find weights for …` INFO만 내고 imatrix 없이 양자화한다. 다만 IQ2_*/IQ1_*/IQ3_XXS는 오류로 멈춘다. 그러므로 R3 수집기가 회전 기저(`H·D·x`) 활성화를 회전 파일의 같은 가중치 이름으로 모두 기록하는지, 컨버터가 `--imatrix` 범위를 따로 검사하는지 확인해야 한다(제안).
- 회전 파일은 llama.cpp가 로드를 거부하므로 `llama-imatrix`로는 회전 기저 imatrix를 만들 수 없다. gufo 수집기가 이 형식으로 써야 한다.
- NextN 블록 64는 회전하지 않으므로 그 텐서의 imatrix 항목은 원래 기저다.

### 11. E7a dequant 경로
- 실제 가중치 64행(`blk.0.ffn_gate.weight`, K = 5120, 원래와 회전 두 가지)으로 왕복했다. ggml `ggml_quantize_chunk`(imatrix = 1)로 양자화한 뒤, gguf-py `gguf.quants.dequantize`(기준 리비전 `gguf-py`)와 ggml `to_float`로 각각 풀어 비교했다.
- Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M 다섯 타입 모두 두 결과가 **비트 단위로 같았다**.

| 타입 | bpw | 상대 RMSE(원래) | 상대 RMSE(회전) |
|---|---:|---:|---:|
| Q2_K | 2.625 | 0.268 | 0.268 |
| IQ2_XXS | 2.0625 | 0.345 | 0.344 |
| IQ2_XS | 2.3125 | 0.300 | 0.300 |
| IQ2_S | 2.5625 | 0.250 | 0.249 |
| IQ1_M | 1.75 | 0.453 | 0.450 |

상대 RMSE는 imatrix = 1, 64행 기준이라 품질 판단에는 쓰지 않는다.

- 정한 경로: E7a fake quant는 `llama-quantize`(imatrix 포함)로 2비트 GGUF를 만들고, 기준 리비전의 gguf-py `quants.dequantize`로 BF16으로 다시 푼다.
- 큰 텐서에서 Python 속도가 문제면 같은 결과를 내는 ggml `to_float`(C 도우미)를 쓴다.
- gufo C++ `ggml_dequant.cpp`는 IQ2_XS, IQ2_S, IQ1_M이 없으므로 쓰지 않는다.
- 실제 E7a 파일은 만들지 않았다.

### 12. 원본 대조 (HF safetensors, USB 읽기 전용)
- 대상: `/run/media/m2on/T5 EVO/AI/model/originals-…/llm/qwen-3.8-27b-bf16/`
- HF 리비전은 백업의 `source.json`에 있다: `Qwen/Qwen3.8-27B` `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0`. `config.json`과 `README.md`에는 리비전이 없다.
  - 백업 `manifest.json`: backup_id `originals-20261007T122040Z-101bd114`, status verified
  - `config.json` sha256은 `files.sha256`과 같다(`191e0af2…`)
  - transformers 5.8.0.dev0, dtype bfloat16
- 방법: 표본 텐서의 바이트 구간만 읽었다(2.21 GiB, 12.8 s). 외장 디스크에는 쓰지 않았다. 스크립트는 `build/rq-llama/hfcheck.py`, 결과는 `build/rq-llama/logs/hfcheck.{log,json}`이다.
- 대조한 표본(53건) 범위:
  - `token_embd`와 `output`의 앞·뒤 256행
  - DeltaNet 블록 0·62, attention 블록 3·63
  - MTP 블록(`mtp.*` → `blk.64.*`)
  - 모든 norm과 SSM 파라미터 종류
- 결과: 49/53이 아래 변환 아래에서 비트 단위로 같다. 나머지 4건은 같은 `ssm_a`를 두 방식으로 비교한 것이다.

| GGUF 텐서 | 변환 (llama.cpp `conversion/qwen.py` Qwen3.5) | 결과 |
|---|---|---|
| 임베딩, 헤드, `ffn_*`, `attn_q/k/v/output`, MTP `eh_proj`·`ffn_down`·`attn_q` | 없음 | BF16 비트 일치 |
| `attn_qkv` | V 행만 K 헤드 기준 묶음에서 tiled 순서로 재배열(`_reorder_v_heads`, 16 K 헤드, 48 V 헤드) | 비트 일치 |
| `attn_gate`(in_proj_z), `ssm_alpha`/`ssm_beta`(in_proj_a/b) | 행 재배열 | 비트 일치 |
| `ssm_out`(out_proj) | **열(입력 K) 재배열** | 비트 일치 |
| `ssm_conv1d` | squeeze, V 채널 재배열, F32 | 일치 |
| `ssm_dt.bias` | 재배열, F32 | 일치 |
| `ssm_a` | `-exp(A_log)`, 재배열, F32 | FP32 계산과 최대 1.49e-8 차이(exp 구현의 1 ulp 수준). BF16에서 계산한 값과는 다름(최대 1.1e-3). 그래서 FP32로 계산한 것이다 |
| `*_norm`(attn, post_attention, q/k, output, MTP enorm/hnorm/shared_head) | `w + 1`을 FP32로 계산(BF16 반올림 없음) | 일치 |
| `ssm_norm`(linear_attn.norm) | +1 없음, F32 | 일치 |

- RQ에 주는 뜻: `ssm_out`의 입력 K 순서는 HF와 다르다(V 헤드 tiled 순서). 회전은 GGUF 순서 기준으로 적용되고, gufo 런타임도 GGUF 순서로 활성화를 만든다. 그러므로 회전 파일은 일관되다. HF safetensors에서 직접 회전하는 경로를 만든다면 이 재배열을 먼저 해야 한다.

## 남은 일과 문제

- `.rot` 이름이 llama-quantize의 이름 기반 카운터(`n_attention_wv` 등)를 늘리는지는 확인하지 못했다(로그에 나오지 않는다). 결과 타입에는 영향이 없음을 확인했다.
- `ai-tools/gufo-eval:7c2c5a728270`에는 libsndfile 개발 파일이 없어 gufo CTest configure가 안 된다. CTest 확인은 `05e527d2cae5-rocm101` 이미지로 했다.
- `R1A-RESULTS.md`의 YAML 레시피 예시는 이제 맞지 않는다(JSON으로 바뀜). 그 문서는 R1a 기록이라 고치지 않았다. 레시피 형식의 기준은 `tools/rotquant/recipe.py`의 docstring이다.
- imatrix 범위 검사(항목 10의 주의)를 R3에서 정해야 한다.
