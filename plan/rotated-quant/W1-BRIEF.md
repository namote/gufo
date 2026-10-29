# W1 브리프: R1b 컨버터 실모델 (2026-10-29)

이 브리프는 [W-COMMON.md](W-COMMON.md)와 [R-BRIEF.md](R-BRIEF.md)를 따른다. 범위는 PLAN "R1 컨버터" 절과 [R1A-RESULTS.md](R1A-RESULTS.md)의 "R1b에서 확인할 것" 11항목이다.

## 목표

1. 27B BF16 원본으로 **회전 BF16 GGUF**를 만들고, 실제 텐서로 회전이 맞는지 확인한다.
2. 기준 llama.cpp의 `llama-quantize`가 회전 파일을 올바르게 양자화하는지 확인한다.
3. 다음 W가 쓸 파일을 남긴다.

## 입력과 출력

- 원본: `~/projects/ai.tools/storage/models/llm/qwen-3.8-27b-bf16-gguf/`(분할 2개, `SHA256SUMS`, `PROVENANCE.md`). 읽기 전용으로 마운트한다.
- 회전 BF16 출력: `~/projects/ai.tools/storage/models/llm/qwen-3.8-27b-rot-bf16-gguf/`
  - 그룹: `--rotate-groups all`, `--seed 1`, `--rotate-only`
  - 같은 폴더에 `PROVENANCE.md`(원본 sha256, 컨버터 리비전, 명령줄, 시드, 시간, 최대 RSS)와 `SHA256SUMS`를 둔다.
  - 이 파일은 W3 ②, W4, W5에서 계속 쓰므로 지우지 않는다.
- 회전 Q4 시험 파일 1개: UD-Q4_K_XL 레시피를 복사해(`--copy-recipe-from`) 만든다. imatrix는 쓰지 않는다. 위치는 `~/projects/ai.tools/storage/models/llm/qwen-3.8-27b-rot-q4-test-gguf/`. 이 파일은 W3 ③(DFlash2 수락률)과 llama.cpp 로드 거부 확인에 쓴다. 비교용으로 같은 레시피의 **비회전** Q4도 만들되, `--rotate-groups none`으로 BF16 원본에서 만든다. 두 파일은 W3 ③이 끝날 때까지 둔다.

## llama.cpp 기준 빌드

- 리비전은 `.devops/nix/llama-cpp-reference.nix`의 `68d9053a`(b11069)다.
- 호스트에는 Nix가 없다. 다음 둘 중 하나로 빌드하고, 고른 방법을 결과에 적는다.
  - eval 이미지 안에서 그 리비전 소스를 CMake로 빌드한다(`build/rq-llama/`).
  - nixos/nix 컨테이너로 flake 출력 `.#llama-cpp-reference`를 빌드한다.
- `llama-quantize`, `llama-cli`(또는 `llama-server`)와 gguf-py만 있으면 된다. 로드 거부만 확인하므로 CPU 빌드로 충분하다. HIP 빌드가 꼭 필요하면 이유를 적는다.
- 회전 GGUF가 아닌 파일(비회전 Q4)은 llama.cpp가 정상으로 로드하는지도 함께 확인한다(대조군).

## 할 일 (R1A-RESULTS의 1–11)

- 1–5: `.rot` 복사(바이트 비교), 이름 기반 분류와 fallback 경고, 출력 헤더 타입이 `.tensor-types.txt`와 같은지 검사(이 검사는 컨버터나 테스트에 추가한다), llama.cpp 로드 거부 메시지, `gufo.*` KV 보존
- 6: 실제 27B `--dry-run` 표. 역할 없는 텐서, norm·SSM 파라미터 타입, 그룹별 텐서 수
- 7: UD-Q4_K_XL 헤더 복사 가능 여부
- 8: 회전 시간, 최대 RSS, 해시 시간
- 9: CPU 검증. 실제 파일에서 그룹마다 1개 이상, 층 앞·중간·끝에서 텐서를 골라 `W'·(H·D·x) ≈ W·x`를 확인한다. 무작위 x와, 가능하면 이상치가 있는 x로 확인한다. BF16 반올림 수준의 오차인지 상대 오차 분포로 보인다.
- 10: imatrix 항목 이름(llama.cpp imatrix 형식과 회전 파일 텐서 이름의 대응). R3 수집기 설계에 넘길 메모만 남긴다.
- 11: E7a dequant 경로. Q2_K, IQ2_XXS, IQ2_XS, IQ2_S, IQ1_M을 BF16으로 다시 풀 수 있는 경로(gguf-py `quants.py` 또는 ggml)를 정하고, 작은 텐서 하나로 왕복을 확인한다. 실제 E7a 파일은 만들지 않는다.

- 12: **원본 대조(PROVENANCE의 unknown 해소).**
  - 대조 대상: 외장 디스크의 HF 원본 `Qwen/Qwen3.8-27B` BF16 safetensors. 경로는 `/run/media/m2on/T5 EVO/AI/model/originals-20261007T122040Z-101bd114/llm/qwen-3.8-27b-bf16/`이고, exFAT를 읽기 전용으로 마운트했다. 같은 폴더 상위에 `files.sha256`과 `manifest.json`이 있다.
  - 방법: 텐서 표본을 골라 unsloth BF16 GGUF와 값이 비트 단위로 같은지, 아니면 변환(예: A_log, conv1d 모양 변환, norm +1 등)이 있는지 확인한다. 표본은 그룹마다, 층 앞·끝에서 고르고, norm과 SSM 파라미터를 포함한다.
  - HF 원본의 리비전을 `config.json`, `README.md`, manifest에서 찾을 수 있으면 기록한다.
  - USB에서 읽으므로 전체 해시는 하지 않는다. 표본만 읽는다.
  - 외장 디스크에는 쓰지 않는다.

## CTest 등록

감독이 사용자에게 방식을 묻는 중이다(JSON 레시피로 바꿀지, 점검 환경에 numpy/yaml을 더할지). 답이 오기 전에는 등록하지 않는다. 답은 감독이 SendMessage로 전한다.

## 파일 경계

- 쓸 수 있는 파일:
  - `tools/rotquant/**`
  - `tools/qwen27b/rotquant_groups.py`
  - `tests/tools/rotquant_test.py`
  - `tests/models/qwen27b/rotquant_groups_test.py`
  - `plan/rotated-quant/R1B-RESULTS.md`
  - `build/rq-llama/`(커밋하지 않음)
  - 위 모델 출력 폴더
- CTest 등록 방식이 정해지면 `CMakeLists.txt`(등록 부분만)와 `tools/README.md`(`rotquant/` 행)도 고칠 수 있다.
- `src/`는 고치지 않는다.

## 순서와 허락

1. 소스를 읽고 계획을 세운다. 이 단계는 허락이 필요 없다.
2. **WAITING:** "CPU 단계 1"(llama.cpp 빌드, 원본 해시 재확인 여부, 회전 변환, Q4 두 개 생성)의 예상 시간과 디스크 사용량을 감독에게 보고하고 허락을 받는다.
3. 허락을 받으면 실행하고 검증한다(9는 무거운 numpy이므로 같은 CPU 단계 안에서 한다).
4. 커밋한 뒤 `R1B-RESULTS.md`에 결과를 쓰고 보고한다.

## 통과 기준

- 단위 테스트가 통과한다(기존 20/20, 2/2와 새로 추가한 테스트).
- 실제 텐서 CPU 검증에서 오차가 BF16 반올림 수준이다.
- llama.cpp가 회전 파일의 로드를 거부하고, 비회전 대조군은 로드한다.
- `.rot`과 KV가 양자화 출력에 그대로 있다.
- 출력 타입이 레시피와 모두 같다.
