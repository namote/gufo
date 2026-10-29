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
- k = 64: 레코드가 작고(run당 약 26 MB), 절단 오차를 `coverage`로 직접 확인할 수 있다. 64가 충분한지는 BF16 run의 coverage로 판단한다.
- 회전 파일도 같은 명령으로 돈다(모델 경로만 다름).
- 툴체인(감독 결정 2026-10-29): R0, R2 ②, R4 품질 비교는 모두 같은 수집기 바이너리로 한다. ROCm 7.2.4 gpu-test 빌드, 이미지 `ai-tools/gufo-eval:64849ca6ec7b`(ROCm 7.2.4, libsndfile 포함. `7c2c5a728270`은 현재 트리가 요구하는 sndfile이 없어 configure가 실패했다). 메타데이터에 `toolchain`, `compiler`, `hip_version`을 기록한다. 속도는 따로 rocm101 측정 빌드로 잰다.

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

## 품질 표

(창 뒤에 채운다.)

## 속도 표

(창 뒤에 채운다.)
