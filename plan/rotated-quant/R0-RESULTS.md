# R0 결과: 품질 하네스와 기준선 (진행 중, 2026-10-29)

상위 문서: [회전 가중치 양자화 계획](PLAN.md) "R0 기준선"·"효율 점수판"·D4, [W2 브리프](W2-BRIEF.md). 이 문서는 W2가 쓴다. 측정 결과는 GPU 창 뒤에 채운다.

## 상태

- 완료: 기존 도구 조사, 평가 코퍼스 a01, 하네스(수집기·지표·bootstrap), 합성 logits CPU 테스트, 모델 파일 sha256, 수집기 빌드.
- 완료(2026-10-29): smoke 창 `rq0-smoke-a01`, 품질 창 `rq0-a01`(6개 모델), `RESULTS.md` 품질 값, D4 한국어 세트 초안.
- 완료(2026-10-29): 속도 창 `rq0-speed-a01`(AR).
- 완료(2026-10-29): 텍스트 수락률 창 `rq0-accept-a01`.
- 남음: 없음(R0 범위). D4 실행기는 감독이 정한다.

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

## prism 로더용 수집기 재빌드 (2026-10-29)

- 로더가 upstream `prism.hadamard.*`로 바뀌었다(PLAN D8, W3 `d23a9be5`, `e8a1eeb7`, `5c3422ff`). `5ed60166`은 v1 로더 트리라 prism 파일을 읽지 못하므로 수집기를 다시 빌드했다.
- 새 바이너리: `build/rq-w2-prism-gpu-test/tests/models/qwen27b/qwen27b_rq_logits` sha256 `b5219f1b64387a45177eb77570cf352dd7e476bf2bcbd04618806094d9deb65e`.
  - 소스: `build/rq-w2-prism-src`. HEAD는 `5c3422ff2f69fc9a7de016db7d6a2fa3449704b3`이다.
  - `git diff HEAD` sha256은 `430e37acf782…`(1,307바이트, `plan/rotated-quant/R2-RESULTS.md`만)이다. untracked는 `.dockerignore`와 `.gufo-revision`이다. 기록 파일은 `build/rq-w2-prism-src.provenance`다.
  - 수집기 소스는 커밋 `976f351e`과 같다.
  - 툴체인은 이전과 같다(ROCm 7.2.4 gpu-test, 이미지 `ai-tools/gufo-eval:64849ca6ec7b`, `-j8`).
  - 이전 바이너리 `5ed60166`(`build/rq-w2-gpu-test`)은 그대로 둔다.
- **BF16 기준 재사용 조건:** 새 수집기의 비회전 출력이 `5ed60166`과 바이트 단위로 같아야 한다. W3 RQ2H는 UD-Q4_K_M fingerprint를 `rq-gpu-test` 빌드로 본다. BF16 가중치 경로와 이 바이너리는 그 범위 밖이다.
  - 그래서 `PLAN-RQ2B-PRISM.json`의 `rq2bp-same` 창(약 10분)을 먼저 돌린다. BF16 창 0의 prefill·verify(각 4,095위치)와 decode(prefix 2048, 64위치)를 `logits_io.py same`으로 `rq0-a01` 기록과 레코드 단위로 대조한다.
  - 하나라도 다르면 BF16 기준을 새 바이너리로 다시 수집해야 한다.
- 도구 추가:
  - `logits_io.py same OLD NEW`: 새 run의 레코드가 옛 run의 같은 위치 레코드와 바이트 단위로 같은지 본다.
  - `metrics.py score`의 `kl_tail`: 평균, 최대 위치 하나를 뺀 평균, 중앙값, top-1 뒤집힘 수, 그중 KL > 1인 수, 상위 위치를 낸다. W3 RQ2B 표의 값(prefill 4.86e-4 / 5.86e-5 / 1.17e-5 / 132 / 1)을 그대로 재현했다.

## RQ2B-prism: rot-prism BF16 대 BF16 (2026-10-29, 창 `rq2bp-a01`, W2 측정)

R2 ② 판정은 W3/감독이 [R2 결과](R2-RESULTS.md)에서 한다. 여기에는 측정 값만 적는다.

- **창:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/rq2bp-a01/`. 계획 `PLAN-RQ2B-PRISM.json`, 수집기 `b5219f1b…`, 이미지 `64849ca6ec7b`.
  - `rq2bp-same`: 19:50–19:57, COMPLETE, health_PASS. 새 수집기의 BF16 출력이 `rq0-a01` 기록과 **바이트 단위로 같다.** prefill 4095/4095, verify 4095/4095, decode 64/64(`logits_io.py same`). 그래서 BF16 기준 `rq0-bf16`을 그대로 쓴다.
  - `rq2bp-rotprism`: 19:57–20:51, COMPLETE, health_PASS. 실행 시간은 prefill 809 s, verify 1140 s, decode 1197 s다. 헤드 재생 검사는 8/8 통과했다. HIP 최대 사용량은 51.3–51.5 GB다.
- **파일:** `qwen-3.8-27b-rot-prism-bf16-gguf/Qwen3.8-27B-rot-prism-BF16.gguf`, sha256 `1f0f3f8b…`.
- **지표:** `metrics.py score`(bootstrap 2000, seed 20261029, 블록 256, `kl_tail` 포함). 보고는 `~/projects/ai.tools/storage/evaluations/rq-r0-20261029/metrics-rq2bp/rotprism-<path>.json`에 있다.

| 경로 | n | KL 평균 [토큰 95 %] | 최대 1개 제외 평균 | 중앙값 | top-1 [95 %] | top-1 뒤집힘(비율) | 그중 KL > 1 | ΔNLL [95 %] |
|---|---:|---|---:|---:|---|---:|---:|---|
| prefill | 32760 | 1.29e-4 [0.48e-4, 2.49e-4] | 8.56e-5 | 1.20e-5 | 0.99591 [0.99521, 0.99655] | 134 (0.41 %) | 1 | −0.0002 [−0.0007, +0.0002] |
| verify | 32760 | 1.26e-4 [0.48e-4, 2.38e-4] | 8.54e-5 | 1.19e-5 | 0.99594 [0.99524, 0.99655] | 133 (0.41 %) | 1 | −0.0002 [−0.0006, +0.0003] |
| decode | 4096 | 3.13e-4 [0.44e-4, 8.38e-4] | 5.69e-5 | 1.35e-5 | 0.99780 [0.99634, 0.99902] | 9 (0.22 %) | 0 | +0.0005 [−0.0003, +0.0018] |
| 참고: upstream llama.cpp CPU | gufo docs 코퍼스 | 1.01e-4 | — | 3.3e-5 | 0.9951 | — | — | — |

- **llama.cpp 참고값의 출처:** 감독 제공(2026-10-29)이다. 코퍼스, 엔진, 경로가 달라 직접 비교값이 아니다. 같은 차수인지 볼 때만 쓴다.
- **v1 파일(rq2b-a01)과 비교:**

| 경로 | 파일 | KL 평균 | 최대 1개 제외 평균 | 중앙값 | 뒤집힘 |
|---|---|---:|---:|---:|---:|
| prefill | v1 | 4.86e-4 | 5.86e-5 | 1.17e-5 | 132 |
| prefill | prism | 1.29e-4 | 8.56e-5 | 1.20e-5 | 134 |

  - 중앙값, 뒤집힘 수, 상위 2개를 뺀 평균(v1 5.05e-5, prism 5.38e-5)은 거의 같다.
  - 평균의 차이는 소수 위치 몇 개에서 나온다.
- **꼬리 위치**(prefill, KL):

| 위치 | v1 | prism |
|---|---|---|
| window 7 pos 3017 | 13.99(뒤집힘) | 0.0001 |
| window 7 pos 2537 | 0.27 | 1.04 |
| window 7 pos 3493 | 0.001 | 1.44(뒤집힘) |

  - v1의 붕괴 위치(3017)는 prism에서 사라졌다. 대신 다른 chat 위치(3493)가 뒤집혔다.
  - pos 2537(`<|im_start|>user\n` 뒤)은 두 부호 집합에서 모두 크다.
  - 짐작: 큰 KL 한두 개는 부호 집합에 민감한 위치(혼돈형)에서 나온다. pos 2537은 부호와 상관없이 민감한 위치다. 층별 확인은 하지 않았다.
- **카테고리**(prefill KL 평균): chat 4.1e-4, code 4.8e-5, ko_prose 3.3e-5, en_prose 2.4e-5다. chat이 다른 카테고리보다 한 차수 크고, 상위 위치도 대부분 chat window(6, 7)에 있다.
- **PLAN R2 ② 기준과 대조(값만, 판정은 W3/감독):** 기준은 KL ≤ Q8_0 KL/10이다(prefill 2.13e-4, verify 0.57e-4, decode 0.88e-4).
  - prefill 평균은 기준 아래다.
  - verify와 decode 평균은 기준보다 크다. decode 평균은 pos 2537 하나가 대부분이다.
  - top-1은 세 경로 모두 Q8_0(0.9795/0.9870/0.9827)보다 높다.

## RQ2I: gufo 대 llama.cpp 로더 교차 확인 (설계, 2026-10-29, W2 담당)

- **목적:** 같은 prism Q4 파일과 같은 토큰으로 gufo와 upstream llama.cpp의 logits를 비교한다. 회전 파일의 KL이 비회전 파일과 같은 차수면 gufo 로더가 맞다. 로더 오류는 KL 0.1 이상으로 드러난다(감독 기준).
- **실행 파일:** `qwen27b_rq_kld_base`(`tests/models/qwen27b/rq_kld_base.cpp`, 커밋 `526c4e71`). sha256 `11400fc820b635a65eda512cfdf9ecd2f273ce59ebe41a9e4df955236096d09f`이고, `build/rq-w2-prism-gpu-test`에서 빌드했다.
  - 소스 스냅샷은 HEAD `5c3422ff`의 `src`와 같다. 이후 `src` 변경이 없다.
  - 같은 트리의 `qwen27b_rq_logits`는 `b5219f1b…`로 바뀌지 않았다.
- **동작:** upstream `8ab29fae` `tools/perplexity/perplexity.cpp`의 `--kl-divergence-base` 파일을 쓴다.
  - 헤더 `_logits_`, n_ctx, n_vocab, n_chunk, tokens.
  - 위치 n_ctx/2 … n_ctx−2의 행. 행은 llama `log_softmax` uint16 인코딩이다.
  - chunk c는 코퍼스 a01 window (c % 8)의 토큰 [512·(c/8), +512)이다. chunk마다 executor를 새로 시작하고, logits는 운영 prefill 헤드(`LaunchQwenOutputHead`)로 낸다.
- **CPU 테스트:** `--encode-rows` 모드(GPU 없음)의 C++ 인코딩을 Python `logits_io.llama_encode_row`와 비교한다(`rq_quality_test.py`, `GUFO_RQ_KLD_BASE`).
  - 행 4개를 썼다: 일반, 범위 > 16, scale 0, 동률이 많은 행. q 값과 scale은 같고, min_log_prob는 4 ulp 안이다.
  - 통과했다(25개, 1개는 환경 변수가 없어 건너뜀).
- **GPU 창(`PLAN-RQ2I.json`, model `rq2i-gufo`, 약 5분):**
  - 대상: rot-prism UD-copy Q4(`107e9b11…`)와 plain-prism UD-copy Q4(`b1a830cc…`).
  - 설정: n_ctx 512, 16 chunk. 파일마다 2.03 GB(`llama_kld_base_bytes`)이고 receipt 아래에 쓴다.
- **CPU 단계(창 뒤):** W1의 `build/rq-llama-prism/build/bin/llama-perplexity`(upstream `8ab29fae`, CPU)를 eval 이미지에서 파일마다 돌린다.
  - 명령: `-m <같은 GGUF> --kl-divergence-base <gufo 파일> --kl-divergence -c 512 -b 512 --chunks 16 -t 16`
  - 토큰: llama.cpp는 KL 모드에서 토큰을 base 파일에서 읽는다(토큰화하지 않음). Qwen3.8 GGUF에는 `tokenizer.ggml.add_bos_token` 키가 없어 BPE 기본값(add_bos false)이 적용된다. 그래서 gufo가 쓴 토큰을 그대로 쓴다.
  - 예상 시간: 파일마다 약 3–6분이다(W1의 BF16 실적 342 s보다 짧을 것, 짐작). 최대 RSS 약 20 GB다.
- **판정:** llama의 Mean/Median KLD와 Same top p를 rot 파일과 plain 파일에서 비교한다.

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

## 품질 표 (2026-10-29, 창 `rq0-a01`)

- **창:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/rq0-a01/<model>/`. BF16(12:12–13:07 KST), Q8_0, Q6_K, UD-Q4_K_XL, Q4_K_M, UD-Q4_K_M 순서로 돌렸다. 모두 COMPLETE, health_PASS true, 재시도 없음. 계획은 `PLAN-RQ0.json`.
- 고정 바이너리 `5ed60166…`, top-k 256, 이미지 `ai-tools/gufo-eval:64849ca6ec7b`.
- 원시 결과는 run마다 `receipt/run/<model>-<path>.{json,bin,log}`에 있다. 지표 보고는 `~/projects/ai.tools/storage/evaluations/rq-r0-20261029/metrics-a01/`에 있다(`<model>-<path>.json`, `compare-*.json`, bootstrap 2000회, seed 20261029, 블록 256토큰).
- 경로: prefill과 verify는 32,760위치, decode는 창마다 prefix 2048 뒤 512위치(4,096위치)다. 각 경로는 같은 경로의 BF16 run과 비교한다.
- prefill 헤드 재생 검사(마지막 행이 `CopyLastLogits`와 비트 단위로 같은지)는 모든 prefill run에서 8/8 통과했다.
- **coverage:** 기준(BF16) top-256 질량 평균이 prefill·verify 0.970, decode 0.964다. 0.95 이상이므로 k 256을 유지한다. p5는 0.850/0.838, 최소는 0.382/0.571이다.

### 모델·경로별 (BF16 대비, KL과 ΔNLL은 ×10⁻³)

| 설정 | 경로 | n | ΔNLL ×10⁻³ (토큰 CI / 블록 CI) | KL_256 ×10⁻³ (토큰 CI / 블록 CI) | top-1 (토큰 CI / 블록 CI) | 기준 top-k 질량 평균/p5/최소 |
|---|---|---|---|---|---|---|
| Q8_0 | prefill | 32760 | +1.06 ([-0.60, +2.76] / [-1.78, +3.91]) | 2.13 ([+1.60, +2.87] / [+1.40, +3.15]) | 0.9795 ([+0.9779, +0.9810] / [+0.9758, +0.9829]) | 0.9697 / 0.8501 / 0.3824 |
| Q8_0 | verify | 32760 | +1.34 ([+0.37, +2.41] / [-0.03, +2.99]) | 0.57 ([+0.48, +0.67] / [+0.44, +0.71]) | 0.9870 ([+0.9857, +0.9883] / [+0.9842, +0.9895]) | 0.9697 / 0.8501 / 0.3824 |
| Q8_0 | decode | 4096 | +1.25 ([-0.46, +2.97] / [-0.30, +3.00]) | 0.88 ([+0.59, +1.25] / [+0.53, +1.48]) | 0.9827 ([+0.9785, +0.9866] / [+0.9766, +0.9878]) | 0.9637 / 0.8375 / 0.5708 |
| Q6_K(bartowski) | prefill | 32760 | +0.78 ([-1.09, +2.72] / [-2.97, +3.80]) | 2.96 ([+2.61, +3.42] / [+2.36, +3.73]) | 0.9698 ([+0.9679, +0.9716] / [+0.9648, +0.9743]) | 0.9697 / 0.8501 / 0.3824 |
| Q6_K(bartowski) | verify | 32760 | +0.28 ([-1.49, +2.40] / [-1.79, +2.50]) | 2.68 ([+1.74, +4.17] / [+1.62, +4.10]) | 0.9737 ([+0.9720, +0.9755] / [+0.9694, +0.9776]) | 0.9697 / 0.8501 / 0.3824 |
| Q6_K(bartowski) | decode | 4096 | +2.23 ([-0.05, +4.51] / [-0.72, +5.43]) | 1.93 ([+1.47, +2.56] / [+1.35, +2.97]) | 0.9741 ([+0.9692, +0.9788] / [+0.9644, +0.9822]) | 0.9637 / 0.8375 / 0.5708 |
| UD-Q4_K_XL | prefill | 32760 | +10.71 ([+6.38, +15.25] / [+4.13, +18.25]) | 15.17 ([+12.91, +18.04] / [+11.34, +19.61]) | 0.9475 ([+0.9452, +0.9500] / [+0.9420, +0.9527]) | 0.9697 / 0.8501 / 0.3824 |
| UD-Q4_K_XL | verify | 32760 | +10.62 ([+6.30, +15.19] / [+4.11, +18.03]) | 15.16 ([+12.90, +18.03] / [+11.34, +19.58]) | 0.9476 ([+0.9453, +0.9501] / [+0.9422, +0.9528]) | 0.9697 / 0.8501 / 0.3824 |
| UD-Q4_K_XL | decode | 4096 | -0.49 ([-6.21, +4.94] / [-10.86, +8.62]) | 15.00 ([+9.16, +23.71] / [+7.42, +25.94]) | 0.9473 ([+0.9402, +0.9541] / [+0.9360, +0.9585]) | 0.9637 / 0.8375 / 0.5708 |
| Q4_K_M(plain, imatrix 없음) | prefill | 32760 | +28.84 ([+24.50, +33.33] / [+20.80, +38.99]) | 19.68 ([+17.92, +21.70] / [+16.50, +23.60]) | 0.9310 ([+0.9283, +0.9337] / [+0.9239, +0.9377]) | 0.9697 / 0.8501 / 0.3824 |
| Q4_K_M(plain, imatrix 없음) | verify | 32760 | +28.77 ([+24.48, +33.27] / [+20.76, +38.80]) | 19.65 ([+17.93, +21.64] / [+16.49, +23.54]) | 0.9309 ([+0.9282, +0.9336] / [+0.9238, +0.9377]) | 0.9697 / 0.8501 / 0.3824 |
| Q4_K_M(plain, imatrix 없음) | decode | 4096 | +17.51 ([+9.83, +26.48] / [+7.86, +27.14]) | 18.54 ([+14.92, +22.79] / [+12.96, +25.82]) | 0.9304 ([+0.9226, +0.9385] / [+0.9194, +0.9431]) | 0.9637 / 0.8375 / 0.5708 |
| UD-Q4_K_M(운영) | prefill | 32760 | +1.31 ([-3.40, +6.14] / [-7.85, +8.24]) | 17.53 ([+15.39, +20.00] / [+13.84, +22.04]) | 0.9413 ([+0.9388, +0.9438] / [+0.9363, +0.9462]) | 0.9697 / 0.8501 / 0.3824 |
| UD-Q4_K_M(운영) | verify | 32760 | +1.45 ([-3.28, +6.32] / [-7.50, +8.37]) | 17.53 ([+15.40, +19.99] / [+13.85, +22.08]) | 0.9415 ([+0.9390, +0.9440] / [+0.9364, +0.9464]) | 0.9697 / 0.8501 / 0.3824 |
| UD-Q4_K_M(운영) | decode | 4096 | +4.98 ([-3.09, +12.20] / [-0.57, +10.23]) | 16.98 ([+12.33, +23.91] / [+10.45, +28.52]) | 0.9390 ([+0.9316, +0.9463] / [+0.9290, +0.9492]) | 0.9637 / 0.8375 / 0.5708 |

### 카테고리별 (prefill 경로, 토큰 CI)

| 설정 | en_prose KL ×10⁻³ | ko_prose KL ×10⁻³ | code KL ×10⁻³ | chat KL ×10⁻³ | en_prose top-1 | ko_prose top-1 | code top-1 | chat top-1 |
|---|---|---|---|---|---|---|---|---|
| Q8_0 | 0.84 [+0.80, +0.88] | 1.28 [+1.09, +1.57] | 0.94 [+0.81, +1.11] | 5.45 [+3.43, +8.33] | 0.980 | 0.973 | 0.991 | 0.974 |
| Q6_K(bartowski) | 1.83 [+1.68, +2.01] | 2.54 [+2.16, +3.04] | 1.60 [+1.42, +1.84] | 5.88 [+4.50, +7.51] | 0.969 | 0.959 | 0.987 | 0.964 |
| UD-Q4_K_XL | 7.17 [+6.67, +7.71] | 10.29 [+9.21, +11.63] | 5.18 [+4.88, +5.52] | 38.04 [+28.84, +48.87] | 0.946 | 0.931 | 0.972 | 0.941 |
| Q4_K_M(plain, imatrix 없음) | 14.45 [+13.08, +16.14] | 15.63 [+14.74, +16.59] | 8.75 [+8.14, +9.56] | 39.89 [+33.77, +47.10] | 0.929 | 0.916 | 0.965 | 0.914 |
| UD-Q4_K_M(운영) | 10.27 [+9.68, +10.92] | 13.28 [+12.19, +14.67] | 7.37 [+6.89, +7.92] | 39.20 [+30.48, +49.09] | 0.936 | 0.928 | 0.967 | 0.934 |

BF16 자체 NLL(nats/token, prefill): en_prose 2.676, ko_prose 2.362, code 1.037, chat 2.691; 전체 2.192.

### 크기, 메모리, 수집 시간

| 설정 | 평균 bpw | 상주 GB (`GetResidentBytes`) | HIP 최대 사용 GB | prefill / verify / decode 수집 분 |
|---|---|---|---|---|
| BF16 | 16.001 | 54.66 | 51.42 | 13.8 / 19.8 / 20.2 |
| Q8_0 | 8.502 | 28.60 | 25.36 | 5.6 / 10.5 / 9.6 |
| Q6_K(bartowski) | 7.045 | 23.86 | 20.63 | 5.4 / 10.1 / 9.1 |
| UD-Q4_K_XL | 5.145 | 17.56 | 14.33 | 5.0 / 8.1 / 6.2 |
| Q4_K_M(plain, imatrix 없음) | 5.696 | 18.97 | 15.75 | 5.0 / 8.3 / 6.7 |
| UD-Q4_K_M(운영) | 4.803 | 16.46 | 13.24 | 5.0 / 8.1 / 6.0 |

### paired 차이 (첫째 − 둘째, ×10⁻³, top-1은 비율)

| 비교 | 경로 | ΔNLL (토큰 / 블록 CI) | KL_256 (토큰 / 블록 CI) | top-1 (토큰 / 블록 CI) |
|---|---|---|---|---|
| UD-Q4_K_XL − Q4_K_M(plain, imatrix 없음) | prefill | -18.14 ([-23.25, -12.81] / [-26.56, -10.73]) | -4.51 ([-6.82, -2.03] / [-7.10, -1.75]) | +0.0166 ([+0.0136, +0.0194] / [+0.0128, +0.0204]) |
| UD-Q4_K_XL − Q4_K_M(plain, imatrix 없음) | verify | -18.14 ([-23.34, -12.82] / [-26.58, -10.82]) | -4.49 ([-6.76, -2.05] / [-7.05, -1.76]) | +0.0168 ([+0.0139, +0.0196] / [+0.0130, +0.0206]) |
| UD-Q4_K_XL − Q4_K_M(plain, imatrix 없음) | decode | -18.00 ([-27.81, -8.99] / [-33.61, -4.97]) | -3.54 ([-9.96, +5.06] / [-9.57, +3.94]) | +0.0168 ([+0.0088, +0.0249] / [+0.0088, +0.0249]) |
| UD-Q4_K_XL − UD-Q4_K_M(운영) | prefill | +9.40 ([+5.58, +13.24] / [+2.81, +18.05]) | -2.36 ([-4.40, +0.21] / [-4.56, +0.12]) | +0.0062 ([+0.0042, +0.0084] / [+0.0042, +0.0082]) |
| UD-Q4_K_XL − UD-Q4_K_M(운영) | verify | +9.17 ([+5.30, +13.06] / [+2.61, +17.79]) | -2.37 ([-4.42, +0.18] / [-4.60, +0.13]) | +0.0062 ([+0.0042, +0.0083] / [+0.0042, +0.0081]) |
| UD-Q4_K_XL − UD-Q4_K_M(운영) | decode | -5.46 ([-10.85, +0.19] / [-16.72, +2.48]) | -1.99 ([-6.94, +4.95] / [-6.91, +4.58]) | +0.0083 ([+0.0022, +0.0144] / [+0.0024, +0.0139]) |
| Q4_K_M(plain, imatrix 없음) − UD-Q4_K_M(운영) | prefill | +27.54 ([+21.80, +33.69] / [+16.03, +42.38]) | +2.15 ([-0.11, +4.22] / [-0.22, +4.43]) | -0.0103 ([-0.0130, -0.0074] / [-0.0143, -0.0067]) |
| Q4_K_M(plain, imatrix 없음) − UD-Q4_K_M(운영) | verify | +27.32 ([+21.60, +33.47] / [+15.88, +41.86]) | +2.12 ([-0.15, +4.18] / [-0.29, +4.39]) | -0.0106 ([-0.0134, -0.0077] / [-0.0146, -0.0069]) |
| Q4_K_M(plain, imatrix 없음) − UD-Q4_K_M(운영) | decode | +12.53 ([+2.46, +23.62] / [+4.23, +21.26]) | +1.56 ([-5.55, +6.89] / [-4.92, +6.57]) | -0.0085 ([-0.0168, -0.0005] / [-0.0154, -0.0015]) |
| Q8_0 − Q6_K(bartowski) | prefill | +0.27 ([-1.55, +2.11] / [-1.66, +2.49]) | -0.84 ([-1.34, -0.11] / [-1.33, -0.12]) | +0.0096 ([+0.0079, +0.0114] / [+0.0074, +0.0119]) |
| Q8_0 − Q6_K(bartowski) | verify | +1.06 ([-0.99, +2.82] / [-1.12, +3.65]) | -2.11 ([-3.51, -1.22] / [-3.47, -1.14]) | +0.0132 ([+0.0115, +0.0149] / [+0.0111, +0.0156]) |
| Q8_0 − Q6_K(bartowski) | decode | -0.98 ([-3.51, +1.32] / [-4.98, +2.72]) | -1.05 ([-1.37, -0.82] / [-1.53, -0.77]) | +0.0085 ([+0.0042, +0.0129] / [+0.0037, +0.0134]) |


### 읽은 것

- **순서:** KL_256 기준으로 Q8_0 < Q6_K ≪ UD-Q4_K_XL < UD-Q4_K_M < Q4_K_M(plain) 순이다. top-1도 같은 순서다. ΔNLL은 UD-Q4_K_M(운영)이 +1.3×10⁻³로 0과 구분되지 않는다. UD-Q4_K_XL은 +10.7×10⁻³, plain Q4_K_M은 +28.8×10⁻³다. KL과 NLL의 순서가 다른 원인(Unsloth imatrix가 이 영역 텍스트에 맞는지 등)은 짐작이다.
- **신뢰구간 폭(W2 브리프 기준):** UD-Q4_K_XL − Q4_K_M(plain)의 차이는 prefill·verify에서 ΔNLL, KL, top-1 모두 토큰·블록 구간 안에서 0을 벗어난다. 창 단위 구간도 마찬가지다(`compare-*.json`). 그래서 이 비교에서는 **코퍼스를 늘릴 필요가 없다.**
  - decode 부분집합(4,096위치)에서는 KL 차이가 구분되지 않는다. ΔNLL과 top-1은 구분된다.
  - UD-Q4_K_XL − UD-Q4_K_M의 KL 차이(−2.4×10⁻³)는 토큰 구간이 0에 걸친다([−4.40, +0.21]×10⁻³). top-1과 ΔNLL은 구분된다.
- **경로 차이:** 4비트 파일은 prefill과 verify가 거의 같다(차이 0.1 % 이하). Q8_0은 prefill KL(2.13×10⁻³)이 verify(0.57×10⁻³)의 약 4배다.
  - 짐작: 모든 행렬이 Q8_0인 파일은 prefill에서 Q8_1 활성화 경로를 쓴다(R0b 표). 이 활성화 양자화 오차가 BF16 경로와의 차이로 보이는 것 같다. 확인하지 않았다.
  - R2 ②의 기준인 "KL(Q8_0 ‖ BF16)"은 경로마다 다르므로, 같은 경로끼리 비교해야 한다.
- **카테고리:** 대화(chat)의 KL이 모든 모델에서 다른 카테고리보다 3–6배 크다. 예: UD-Q4_K_XL 38×10⁻³ 대 산문·코드 5–10×10⁻³. 신뢰구간도 넓다. 대화 창에 KL이 큰 위치가 몰려 있는 것으로 보인다(어느 토큰인지는 아직 보지 않았다). 코드가 가장 작다. 한글 산문은 영문 산문보다 1.2–1.5배 크다.
- 코퍼스 영역 한계(기술 문서와 코드)는 위 "평가 코퍼스 a01" 절에 적었다. 일반 산문 품질은 D4로 본다.
- **bpw:** output을 포함하고 token_embd와 NextN 블록(`blk.64`, 대상 forward가 쓰지 않음)을 뺀 2차원 가중치 기준이다. plain Q8_0과 Q4_K_M 파일에는 `blk.64`가 없다.
- **상주 메모리:** `GetResidentBytes`(가중치)와 HIP가 보고한 최대 사용량(`hipMemGetInfo` 기준 최대 total − free)이다. 수집기 컨텍스트 4,352(prefill)에서 잰 값이라 서빙 KV는 포함하지 않는다.
- **GSQ-RCO-IQ3_S:** 로드 불가로 보류했다(위 측정 대상 절).


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
- **텍스트 수락률 창(감독 결정 2026-10-29):**
  - driver `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/rq_serving.py`. 새 측정 로직은 없다. 하는 일은 `gufo diagnose --fingerprint`, `gufo serve llm` 시작과 `/ready` 대기, `/src/tools/gufo/serving_bench.py --suite` 실행, 서버 종료다. ai.tools 저장소에는 커밋하지 않는다.
  - 프롬프트 `tools/qwen27b/rq_quality/accept_prompts.json`(64개). 빌더는 `accept_prompts_src.py`다. 구성: speculative corpus 11개, D4의 tool이 아닌 15개, W2가 새로 쓴 38개("W2-authored", 영문 19, 한글 19).
  - 겹침 검사: 40자 조각을 20자 간격으로 찾았다(파일 `manifest.overlap`). R0 a01과 R3 c01에는 겹침이 없다. R3 c02에는 `expository_pangram` 조각 1개가 있다. 수락률 프롬프트는 품질 지표가 아니므로 기록만 한다.
  - 설정: C=1, `--max-tokens 256`, greedy, think off, `--no-cache-prompt`, warmup 1 + 반복 3.
    - 수락률은 결정적인 첫 반복 값을 쓴다. 반복 사이에 draft 수가 같은지 검사한다.
    - 실효 tok/s는 세 반복의 decode 합으로 낸다.
  - 분석 `tools/qwen27b/rq_quality/accept.py`: 프롬프트 단위 bootstrap(2000회, ratio 추정량)이다.
    - `summary`: 수락률 Σaccepted/Σdrafted, 출력 토큰당 수락 토큰, decode tok/s를 낸다. 같은 프롬프트의 AR 결과를 함께 주면 AR 대비 속도 향상을 paired로 낸다.
    - `compare`: 같은 프롬프트의 두 run에 대해 paired 차이(수락률)와 비율(tok/s)을 낸다. R2 ③(회전 대 비회전)과 E5가 그대로 부른다.
    - 테스트: `tests/models/qwen27b/rq_accept_test.py`(합성 보고서 10개 항목). 64프롬프트에서 수락률 95% 구간 폭은 0.1 미만, 포함률은 85/100 이상으로 확인했다.
  - R0 대상(감독 결정): UD-Q4_K_M과 UD-Q4_K_XL은 dflash2와 ar, Q8_0은 dflash2만 잰다. BF16, Q6_K, plain Q4_K_M은 뺀다. 창은 모드를 번갈아 5개다(`PLAN-RQ0-ACCEPT.json`).
  - 예상 시간(짐작): dflash2 항목은 각 약 30–40분, ar 항목은 각 약 65–75분이다(64 × 256토큰 × 4회, AR 약 12.9 tok/s). 합계 약 4시간이다.
- **상주 메모리:** `gufo bench`는 최대 메모리를 내지 않는다. 품질 수집기의 `model_resident_bytes`와 `hip_peak_used_bytes`를 쓴다.
- **판정:** 요청·단계별 5 %·10 ms 기준으로 본다. 두 표본의 차이가 그보다 크면 결론 없음으로 둔다.

## DFlash2 텍스트 수락률 (2026-10-29, 창 `rq0-accept-a01`)

- **창:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/rq0-accept-a01/<entry>/`. 5개 모두 COMPLETE, health_PASS true, driver 상태 COMPLETE, 서버 정상 종료(returncode 0), 재시도 없음. 계획은 `PLAN-RQ0-ACCEPT.json`.
  - 감독 변경: AR 항목은 `--repetitions 1`, dflash2 항목은 3이다.
  - bench 시간: UD-Q4_K_M dflash2 1239 s, ar 908 s / UD-Q4_K_XL ar 935 s, dflash2 1231 s / Q8_0 dflash2 1694 s.
- **설정:** 바이너리는 `rqw3-base-rocm101`(`c94572b3`)이다. 드래프터는 DFlash2 Q4_K_M(adaptive)이다. 프롬프트 64개, C=1, 최대 256토큰, greedy, think off, 프롬프트 캐시 꺼짐.
  - 모든 dflash2 run에서 세 반복의 draft 수가 같았다(`accept.py` 검사 통과).
- **분석:** `tools/qwen27b/rq_quality/accept.py`, 프롬프트 단위 bootstrap 2000회. 보고는 `~/projects/ai.tools/storage/evaluations/rq-r0-20261029/accept-a01/`에 있다.
- **겹침:** `expository_pangram`의 c02 조각 1개는 감독 결정(2026-10-29)대로 그대로 둔다.

| 대상 | 수락률 Σacc/Σdraft | 출력 토큰당 수락 토큰 | DFlash2 decode tok/s | AR decode tok/s(같은 프롬프트) | AR 대비 |
|---|---|---|---|---|---|
| UD-Q4_K_M(운영) | 0.453 [0.409, 0.503] | 0.663 [0.626, 0.699] | 29.50 [26.78, 32.81] | 12.95 [12.94, 12.97] | 2.28× [2.07, 2.53] |
| UD-Q4_K_XL | 0.457 [0.412, 0.508] | 0.664 [0.626, 0.701] | 29.16 [26.38, 32.50] | 12.33 [12.31, 12.35] | 2.36× [2.14, 2.64] |
| Q8_0 | 0.391 [0.350, 0.440] | 0.675 [0.634, 0.713] | 21.10 [18.82, 23.84] | (재지 않음) | 약 2.75×(짐작, 속도 창 AR tg 7.68과 짝 없이 비교) |

paired 비교(같은 64개 프롬프트, 95 % 구간):

| 비교 | 수락률 차이 | 토큰당 수락 차이 | decode tok/s 비율 | 같은 완성 출력 |
|---|---|---|---|---|
| UD-Q4_K_M − UD-Q4_K_XL (dflash2) | −0.004 [−0.011, +0.003] (구분 안 됨) | −0.001 [−0.009, +0.006] | 1.012 [0.993, 1.032] (구분 안 됨) | 10/64 |
| Q8_0 − UD-Q4_K_M (dflash2) | −0.062 [−0.074, −0.050] | +0.012 [−0.005, +0.026] | 0.715 [0.681, 0.743] | 9/64 |
| UD-Q4_K_M ÷ UD-Q4_K_XL (AR) | — | — | 1.050 [1.048, 1.053] | 10/64 |

- **읽은 것:**
  - 두 UD 4비트 파일의 수락률은 구분되지 않는다.
  - Q8_0은 Σacc/Σdraft가 낮다. 그런데 출력 토큰당 수락 토큰은 같다. adaptive 컨트롤러가 Q8_0에서 더 많이 드래프트하는 것으로 보인다(짐작, 스텝 수는 보고서에 없다).
  - **R2 ③과 E5에는 Σacc/Σdraft와 출력 토큰당 수락 토큰을 함께 보기를 권한다.**
  - AR tok/s(12.95, 12.33)는 속도 창의 tg128 d0(12.93, 12.32)과 0.2 % 안에서 맞는다.
- **구간 폭:** 단일 모델의 수락률 구간 폭은 약 0.09다(프롬프트 사이 차이가 크다). 같은 프롬프트의 paired 차이 구간 폭은 약 0.013이다. R2 ③의 회전 대 비회전 비교는 paired로 하면 약 ±0.007 수준의 변화를 볼 수 있다(짐작, UD 두 파일 비교 기준).
- **32K 깊이의 DFlash2는 재지 않았다.** 이 창은 짧은 프롬프트, 사실상 d0이다.

## 속도 표 (2026-10-29, 창 `rq0-speed-a01`, AR만)

- **창:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/rq0-speed-a01/rq0-speed/`. COMPLETE, health_PASS true, 재시도 없음. 계획 `PLAN-RQ0-SPEED.json`.
- **바이너리:** 이미지 `ai-tools/gufo-eval:rqw3-base-rocm101`의 release `gufo`(nix ROCm 10.1 / Clang 24, HEAD `c94572b3`, W3 회전 변경 없음).
- **설정:** `gufo bench -p 2048 -n 128 -d 0,32768 -r 3 --speculative off --temperature 0 --seed 314159`. KV는 기본 kvarn-k4v2다. 숫자는 bench가 낸 반복 3회의 평균 ± 표준편차(tok/s)다.
- **DFlash2를 뺀 이유(감독 2026-10-29):** `gufo bench`의 DFlash2 수락률은 합성 프롬프트 하나의 궤적이라 믿을 수 없다. DFlash2 수락률과 실효 tok/s는 텍스트 수락률 창(`rq_serving.py`)에서 낸다.
- **순서(ABBA):** Q8_0, Q6_K, UD-Q4_K_XL, Q4_K_M, UD-Q4_K_M, BF16을 돈 뒤 BF16을 뺀 역순으로 한 번 더 돌았다. 양자화 파일마다 떨어진 두 표본(A, B)이 있다.

| 설정 | pp2048 d0 A / B | tg128 d0 A / B | pp2048 d32K A / B | tg128 d32K A / B | 표본 간 최대 차이 | 판정 |
|---|---|---|---|---|---|---|
| Q8_0 | 536.9 / 530.5 | 7.68 / 7.69 | 404.5 / 412.7 | 7.55 / 7.57 | pp 2.0 % | 잡음 5 % 안 |
| Q6_K(bartowski) | 429.3 / 434.6 | 8.61 / 8.58 | 343.3 / 349.9 | 8.44 / 8.42 | pp 1.9 % | 잡음 5 % 안 |
| UD-Q4_K_XL | 639.8 / 639.5 | 12.32 / 12.32 | 484.6 / 482.5 | 12.03 / 11.99 | 0.4 % | 잡음 5 % 안 |
| Q4_K_M(plain, imatrix 없음) | 643.0 / 666.8 | 10.82 ± 0.69 / 11.38 | 488.4 / 505.7 | 10.94 / 11.10 | tg d0 5.2 %, pp 3.7 % | **tg d0 결론 없음**(A의 반복 표준편차 0.69) |
| UD-Q4_K_M(운영) | 635.5 / 661.0 | 12.94 / 12.92 | 482.7 / 494.5 | 12.58 / 12.67 | pp 4.0 % | 잡음 5 % 안 |
| BF16(참고만) | 77.1 | 4.10 | 73.9 | 4.07 | 한 번만 | 참고 |

- **읽은 것:**
  - tg는 bpw를 따라간다. UD-Q4_K_M 12.9, UD-Q4_K_XL 12.3, Q4_K_M(plain) 10.8–11.4, Q6_K 8.6, Q8_0 7.7 tok/s다. plain Q4_K_M은 텐서 288개가 Q8_0(5.70 bpw)이라 UD 4비트보다 느리다.
  - pp는 4비트 파일이 640–667 tok/s로 비슷하다. Q6_K(429–435)는 Q8_0(531–537)보다 느리다. R0b의 Q6_K prefill 결과와 같은 방향이다.
  - 32K 깊이에서 tg는 2–3 % 떨어지고, pp는 24–25 % 떨어진다.
- **BF16은 참고만 한다(감독 2026-10-29).** UD 파일은 32바이트로 정렬돼 있다(D7). BF16은 오프셋 정렬 때문에 decode가 3–15 % 느리다. 그래서 BF16 속도는 다른 파일과 같은 기준으로 비교하지 않는다.
- **잡음:** 같은 파일의 떨어진 두 표본 사이에서 pp는 최대 4.0 %, tg는 최대 0.5 % 차이가 났다. 예외는 plain Q4_K_M의 tg d0(5.2 %)로, 반복 사이 편차가 컸다. 단계별 5 %·10 ms 기준에서 이 칸만 결론 없음이다.
- **상주 메모리:** 품질 표의 "크기, 메모리, 수집 시간"을 쓴다.
- **조건:** 이 수치는 W3 회전 변경이 없는 바이너리의 값이다. R4에서 회전 파일과 비교할 때는 같은 W3 바이너리로 비회전 대조군을 다시 잰다.
