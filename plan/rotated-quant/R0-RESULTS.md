# R0 결과: 품질 하네스와 기준선 (진행 중, 2026-10-29)

상위 문서: [회전 가중치 양자화 계획](PLAN.md) "R0 기준선"·"효율 점수판"·D4, [W2 브리프](W2-BRIEF.md). 이 문서는 W2가 쓴다. 측정 결과는 GPU 창 뒤에 채운다.

## 상태

- 완료: 기존 도구 조사, 평가 코퍼스 a01, 하네스(수집기·지표·bootstrap), 합성 logits CPU 테스트, 모델 파일 sha256, 수집기 빌드.
- 남음: smoke 창, 품질 창(BF16 먼저), 속도 창(rocm101 측정 빌드), D4 한국어 프롬프트 세트 초안, `RESULTS.md` 첫 표.

## 하네스

- 수집기 `tests/models/qwen27b/rq_logits.cpp`(target `qwen27b_rq_logits`, `EXCLUDE_FROM_ALL`, CTest 없음). 공개 엔진 API만 쓰고 `src/`는 바꾸지 않는다. 경로:
  - `prefill`: `SetPromptHiddenCapture` + `ForwardPromptBatch`로 모든 행의 최종 hidden을 받고, 운영 prefill이 마지막 행에 쓰는 헤드(`LaunchRMSNorm` eps 1e-6 + `LaunchGEMV`)를 각 행에 적용한다. 마지막 행은 `CopyLastLogits`와 비트 단위로 같은지 검사한다(`head_replay_checks`).
  - `decode`: 위치마다 `ForwardToken` + `CopyLastLogits`.
  - `verify`: `SaveState` → `ForwardVerificationChunk(≤8행, logits 캡처)` → `CopyVerificationLogits` → `FinishVerification`. 드래프트를 모두 수락한 DFlash2 verify와 같은 순서다. 드래프터용 target-layer hidden 캡처는 켜지 않는다.
- 레코드(위치마다): 다음 토큰, 그 logit, 전체 vocab logsumexp(float64), 자기 top-k id·logit, 기준 run의 top-k id에서의 자기 logit. 형식은 `tools/qwen27b/rq_quality/logits_io.py`. 후보 run은 기준 run의 `.bin` sha256에 묶인다.
- 지표 `tools/qwen27b/rq_quality/metrics.py`(BF16 대비):
  - paired ΔNLL(nats/token).
  - top-k 절단 KL: 기준 top-k 집합과 나머지 버킷 하나로 거칠게 만든 두 분포의 정확한 KL. 0 ≤ KL_k ≤ 전체 KL이고, 차이의 크기는 기준 top-k 질량(`coverage`)으로 함께 보고한다.
  - top-1 일치율(전체 vocab argmax, 동률은 낮은 id).
  - 각각 percentile bootstrap 95% 구간 세 가지: 토큰 단위, 256토큰 블록 단위, 창 단위(2000회, seed 20261029). 카테고리별 표와 두 후보의 paired 차이(`compare`)도 낸다.
- k: 처음 64로 설계했으나 smoke 결과로 256으로 정했다(아래 고정 바이너리 항목). run당 약 100 MB.
- 회전 파일도 같은 명령으로 돈다(모델 경로만 다름).
- 툴체인(감독 결정 2026-10-29): R0, R2 ②, R4 품질 비교는 모두 같은 수집기 바이너리로 한다. ROCm 7.2.4 gpu-test 빌드, 이미지 `ai-tools/gufo-eval:64849ca6ec7b`(ROCm 7.2.4, libsndfile 포함. `7c2c5a728270`은 현재 트리가 요구하는 sndfile이 없어 configure가 실패했다). 메타데이터에 `toolchain`, `compiler`, `hip_version`을 기록한다. 속도는 따로 rocm101 측정 빌드로 잰다.
- 고정 바이너리(감독 2026-10-29, v3 트리): `build/rq-w2-gpu-test/tests/models/qwen27b/qwen27b_rq_logits` sha256 `5ed601662b7b195885d8bb0ef83a94f9159396e9f6486a5cc86e0e6d6f73d0c0`. 소스는 작업 트리 스냅샷 `build/rq-w2-src`다(HEAD `32dbf0fc` + W3 미커밋 v3 변경, `git diff HEAD` sha256 `916bfef7726bdb594e6f727564739fb7e0109f22911229fda61343d8dd9a567e`, untracked 목록 sha256 `20bf604d…`, 기록 `build/rq-w2-src.provenance`). 수집기 소스는 커밋 `976f351e`과 같다. smoke 창(`rq0-smoke-a01`)은 그 전 바이너리 `c680cb88…`로 돌렸고 smoke 전용이다. `479a2a91…`은 쓰지 않았다.
- top-k(감독 2026-10-29): 256으로 R0, R2 ②, R4 전체에 고정한다. smoke에서 top-64 기준 질량이 평균 0.84, 최소 0.275였기 때문이다. 결과 표에는 기준 top-k 질량의 평균, p5, 최소를 모델·경로별로 낸다. 전체 창 평균이 0.95보다 낮으면 1024로 올릴지 감독이 정한다.

## 평가 코퍼스 a01

- 위치: `~/projects/ai.tools/storage/evaluations/rq-r0-20261029/corpus-a01/`(`SHA256SUMS` 포함). manifest 사본: `tools/qwen27b/rq_quality/corpus_manifest.json`. 빌더 `tools/qwen27b/rq_quality/corpus.py`, 소스 리비전 `f95d1302`, tokenizer는 HF `tokenizer.json`(sha256은 manifest). 두 번 빌드해 바이트 단위로 같았다.
- `windows.bin` sha256 `ee0d9ba3963bcc6c1e27b02d1e41e60712a28d343dd300788a6ea8d16779791c`, `windows.json` `51413ceb583989ac9c8e3adea74092aef56e9615f3f20369cfd2a4fecc3515ab`.
- 32,768토큰: 영문 산문, 한글 산문, 코드, 대화 각 8,192토큰. 4,096토큰 창 8개(카테고리당 2개), 겹침 0, 창마다 새 상태에서 시작한다. 채점 위치 32,760개. 문서는 1,536토큰으로 자르고, 창은 문서 경계에서 시작한다.
- 출처: 이 저장소의 MIT 파일 26개(목록·blob·추가 날짜는 manifest). 저장소는 2026-08-11에 시작했고 고른 파일은 2026-09-16 이후에 추가됐다. 그래서 Qwen3.8 27B 학습 데이터에 들어 있을 수 없다. 대화는 문서 섹션을 템플릿 질문과 답(빈 think 블록, 모델 `chat_template.jinja`와 같은 렌더링을 테스트로 확인)으로 바꾼 합성 대화다.
- **한계(감독 2026-10-29):** 영역이 기술 문서와 코드에 치우친다. 한글 문서에는 영문 식별자가 많다. 일반 산문 품질은 이 코퍼스로 판단하지 않고 D4 실제 작업 게이트로 보완한다. 사용자가 넓히라고 하면 a02를 만든다. 창이 8개뿐이라 창 단위 구간은 거칠다. 블록 단위 구간을 함께 본다.
- R3 보정 코퍼스는 manifest `sources`의 경로를 쓰지 않는다.

## 측정 대상과 파일 해시

| 표 이름 | 파일 | sha256 | 비고 |
|---|---|---|---|
| BF16(기준) | `qwen-3.8-27b-bf16-gguf/Qwen3.8-27B-BF16-0000{1,2}-of-00002.gguf` | `b9966e82…191f`, `92e3943c…fa02` | `SHA256SUMS` |
| Q8_0 | `qwen-3.8-27b-gguf/Qwen3.8-27B-Q8_0.gguf` | `aab65c67ef0dad127960efef9247f1832bca105faa1c7a052cc039b223cf86a1` | imatrix 없음, 2026-10-29 해시 |
| Q6_K(bartowski) | `qwen-3.8-27b-gguf-q6-bartowski/Qwen3.8-27B-Q6_K.gguf` | `539041d2cf400b02f53a53cae54339ea65103d276645a34a68eac1c8d69a9db3` | imatrix 있음 |
| UD-Q4_K_XL | `qwen-3.8-27b-ud-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf` | `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e` | `source-UD-Q4_K_XL.json` |
| Q4_K_M(plain, imatrix 없음) | `qwen-3.8-27b-gguf/Qwen3.8-27B-Q4_K_M.gguf` | `c600de0300ae8a0eb3a6c0b8b5561b8b96f16bd2c863c2a66c42de29d391a747` | 출처 기록 없음 |
| UD-Q4_K_M(운영) | `qwen-3.8-27b-ud-q4-k-m-gguf/Qwen3.8-27B-UD-Q4_K_M.gguf` | `322e194ff79741c7baa497c240f677f54b201b0efab44ca8e50f122b39123482` | Unsloth imatrix |
| GSQ-RCO-IQ3_S | `qwen3_8-27b-gsq-rco-iq3_s-d562806-a01/…` | `64b53b64…d810` | **로드 불가로 보류** |

- 2026-10-29 해시 시간(호스트 `sha256sum`, 순서대로): Q8_0 30 s, Q6_K 25 s, Q4_K_M 21 s, UD-Q4_K_M 16 s. 기록 `~/projects/ai.tools/storage/evaluations/rq-r0-20261029/model-sha256.txt`.
- **GSQ-RCO-IQ3_S 보류(감독 결정 2026-10-29):** IQ2_S(임베딩 포함), IQ2_XS, IQ2_XXS, IQ3_XXS, Q2_K, IQ1_M 텐서가 있어 27B 로더(`weights.cpp` `SupportsTensorType`)가 받지 않는다. W1 11항의 역양자화 경로가 확인되면 BF16으로 풀어 fake quant로 잰다(E7a와 같은 경로).

## D4 한국어 프롬프트 세트 (초안)

- 파일: `tools/qwen27b/rq_quality/d4_ko_prompts.json`. 20개 문항으로, 지시 따르기·요약·코드 설명·tool-call이 각각 5개다.
- 문항은 W2가 이 세트를 위해 새로 썼다(2026-10-29). 저장소 문서나 R0 코퍼스의 문장을 쓰지 않았으므로 평가 코퍼스와 겹치지 않는다. R3 보정 코퍼스도 이 파일을 쓰지 않는다.
- 요청 기본값: greedy(temperature 0), thinking 끔, max_tokens 768.
- 문항마다 `checks`에 자동 검사 후보를 적었다. 예: 언어, 문단·문장 수, 꼭 들어갈 말, JSON 구조, 표·번호 목록, 호출할 tool 이름과 인자. 마지막 tool 문항은 tool을 부르지 않아야 맞다.
- **판정 방식 제안(수치 기준은 사용자가 정한다):**
  1. 자동 검사: 문항별 통과/실패를 낸다. tool-call은 이름과 인자가 정확히 맞아야 통과로 본다.
  2. 기준 모델과의 비교: UD-Q4_K_XL과 같은 요청을 보내 출력의 일치 여부를 본다. 같으면 통과로 보고, 다르면 자동 검사 결과로 판단한다.
  3. 자동 검사가 갈리는 문항(요약의 사실성, 코드 설명의 정확성)은 사람이 보거나 BF16 출력을 참조 답으로 써서 비교한다.
  4. 후보 채택 조건 예시: 자동 검사 실패 수가 UD-Q4_K_XL보다 많지 않고, tool-call 5문항이 모두 통과. 숫자는 사용자가 정한다.
- 아직 실행기를 만들지 않았다. HTTP로 `gufo serve`에 보내는 작은 실행기를 `tests/functional/` 관례에 맞춰 만들지는 감독이 정한다.

## 품질 표

(창 뒤에 채운다.)

## 속도 측정 계획 (초안, 실행 전)

- **바이너리:** W3이 만든 이미지 `ai-tools/gufo-eval:rqw3-base-rocm101`에 든 release `gufo`.
  - 이미지 라벨: `org.opencontainers.image.revision` `c94572b3`, `ai.tools.toolchain` `nix-rocm-10.1`.
  - `/opt/gufo/src/build/release/gufo`는 nix store의 정적 바이너리다(`gufo-0-unstable-dirty`, ROCm 10.1 / Clang 24 운영 툴체인).
  - W3의 회전 변경이 없는 HEAD로 만들었으므로 **R0 비회전 기준선 속도에 쓸 수 있다.**
  - 조건 1: W3 (다)에서 W3 코드가 비회전 경로의 성능을 바꾸지 않는다고 확인돼야, 이 기준선을 R2·R4 이후 바이너리와 비교할 수 있다.
  - 조건 2: R4에서 회전 파일과 비교할 때는 같은 W3 바이너리로 비회전 대조군을 다시 잰다. 리비전이 다른 수치를 섞지 않는다.
- **창 계획:** `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-RQ0-SPEED.json`(초안, 실행하지 않음).
  - driver `gufo_eval_20261020.bench`가 이미지의 `gufo bench`를 차례로 부른다.
  - 측정: `-p 2048 -n 128 -d 0,32768 -r 3 --temperature 0 --seed 314159`를 AR(`--speculative off`)과 DFlash2(`--speculative dflash2 --dflash-model …DFlash2-Q4_K_M.gguf -v`, 수락률 출력)로 잰다. 기본 KV는 kvarn-k4v2다.
  - 순서: Q8_0, Q6_K, UD-Q4_K_XL, Q4_K_M, UD-Q4_K_M, BF16을 돈 뒤, BF16을 뺀 역순으로 한 번 더 돈다(ABBA). 양자화 파일마다 떨어진 두 표본으로 잡음 폭을 낸다.
  - 시간: 22회, 약 3시간 이내(짐작, `seconds` 12600).
- **한계:** `gufo bench`는 합성 토큰 id(`(pos % 1000) + 100`)를 넣는다. 그래서 여기서 나오는 DFlash2 수락률과 실효 tok/s는 텍스트 수락률이 아니다. pp/tg AR 처리량은 타입 간 비교에 쓸 수 있다.
- **텍스트 수락률:** `tools/gufo/serving_bench.py --suite docs/models/qwen3.8-27b/artifacts/speculative-corpus.json --corpus-layout distinct --concurrency 1 --max-tokens 128`로 `gufo serve llm … --speculative dflash2`에 HTTP로 보낸다.
  - 이 도구는 응답 usage의 `draft_tokens`, `draft_tokens_accepted`로 수락률을 낸다. prefill, decode, TTFT는 따로 보고한다.
  - 지금 창 driver 가운데 "서버를 띄우고 serving_bench를 돌리는" 것이 없다. 그래서 ai.tools에 작은 driver가 필요하다. W2 파일 경계 밖이므로 감독이 정한다.
  - 대안은 `benchmark-model` 스킬의 `tools/gufo/model_bench`(HTTP, 깊이별 캐시 확인)다.
- **텍스트 수락률 창(감독 결정 2026-10-29: 얇은 driver):**
  - driver: `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/rq_serving.py`. 새 측정 로직은 없다. 하는 일은 이미지 `gufo diagnose --fingerprint`, `gufo serve llm` 시작과 `/ready` 대기, `/src/tools/gufo/serving_bench.py` 실행(결과는 `/receipt/run`), 서버 종료다. ai.tools 저장소에는 커밋하지 않는다.
  - 창 계획: `PLAN-RQ0-ACCEPT.json`(초안). 대상마다 dflash2와 ar 두 항목(ar은 실효 속도 대조군)으로 모두 12개다. 설정: `--context 8192 --sessions 1 --think off`, speculative corpus(11문항), C=1, tg128, warmup 1, 반복 3, `--no-cache-prompt`.
  - R2 ③과 E5에서는 모델 경로만 바꿔 그대로 쓴다.
  - 확인한 것(GPU 없이): driver import, read-only 컨테이너 안에서 `/src`의 git source identity와 `gufo diagnose --fingerprint`. 서버와 bench 실행은 아직 확인하지 않았다.
  - 겹침 확인: `docs/models/qwen3.8-27b/artifacts/speculative-corpus.json` 11문항의 40자 조각(20자 간격)을 찾아봤다. R0 코퍼스 a01과 R3 c01에는 없다. R3 c02에 `expository_pangram` 조각 1개("The quick brown fox…" 문장)가 있다. 수락률은 품질 지표가 아니므로 기록만 한다.
- **상주 메모리:** `gufo bench`는 최대 메모리를 내지 않는다. 품질 수집기의 `model_resident_bytes`와 `hip_peak_used_bytes`를 쓴다.
- **판정:** 요청·단계별 5 %·10 ms 기준으로 본다. 두 표본의 차이가 그보다 크면 결론 없음으로 둔다.

## 속도 표

(창 뒤에 채운다.)
