# 투기 디코딩 개선 설계 (W9 1단계, 2026-10-29)

상위 문서: [PLAN.md](PLAN.md) "방향 전환" 2번, [W9-BRIEF.md](W9-BRIEF.md), [FLASH-NEXT.md](FLASH-NEXT.md) 13.4절. 이 문서는 읽기만 하고 조사한 것이다. 코드 변경, 빌드, GPU 실행, 모델 복사는 하지 않았다. 표기: **[확인]**은 코드나 문서를 직접 읽은 것이고, **[짐작]**은 추론이다.

## 0. 요약

- **gufo Flash-Next는 Kyojin과 같은 계열의 기능을 이미 갖고 있다 [확인].**
  - MTP 1층을 사슬로 돌려 드래프트를 만든다(최대 7개).
  - prompt lookup 복사를 MTP와 섞는다.
  - 수락 이력으로 길이를 고르고, 측정한 사이클 비용표를 쓴다.
  - 작은 드래프트 헤드(98,304행 + 요청 표)를 쓴다.
  - verify는 사슬(chain) 방식이고 최대 8행이다.
  - 그리고 Kyojin이 잰 v0.8.0(2026-10-05) **뒤에** lookup 혼합(#613), 작은 헤드(#603, #733, #739), 점유율 가격(#664, #703), 샘플 드래프트 개선(#742)이 들어왔다. 그러니 35.3 대 52.2라는 격차는 지금 HEAD에서 다시 재야 한다.
- **verify 비용은 gufo가 더 싸다 [확인: 두 비용표].**

  | 드래프트 수 D | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
  |---|---|---|---|---|---|---|---|---|
  | gufo C1 d0, ms (`mtp_costs.hpp:21`) | 38.5 | 47.3 | 56.1 | 65.7 | 74.1 | 82.2 | 89.7 | 96.8 |
  | Kyojin `CYCLE_MS` (`tools/qwen/spec_policy.py:302-329`) | 32.3 | 46.7 | 58.4 | 69.9 | 83.2 | 92.6 | 105.3 | 116.6 |

  - 드래프트 하나를 더하면 gufo는 약 8.8 ms, Kyojin은 약 11–14 ms가 든다. 단, gufo의 D=0(38.5)은 plain AR(35.3)보다 3.2 ms 비싸다. MTP catch-up 때문이다.
  - **[짐작] 그래서 남는 격차는 비용이 아니라 사이클당 수락 토큰(드래프트 품질과 길이 정책)이나 작업 부하 차이에서 와야 한다.** Kyojin plain 27.3과 spec 42–55의 배율을 내려면, 사이클당 토큰이 약 2.4–3이어야 한다.
- **Kyojin과 gufo의 실제 차이 [확인]:**
  1. **길이 신호.** Kyojin은 드래프트하는 동안 **토큰마다 드래프터의 max softmax 확률**을 곱해 "reach"를 만들고, 그 값으로 멈춘다(`TierRule`). gufo는 드래프트 **전에** 깊이별 수락 EMA와 비용표로 길이를 한 번 정한다.
  2. **lookup 조건.** Kyojin은 16토큰 이상 일치 + witness ≥ 3이면 MTP를 **대체**한다. gufo greedy는 12토큰(재개 시 5), 복사가 3개 이상이면 MTP를 건너뛰고, 그 밖에는 MTP 드래프트 뒤에 붙인다.
  3. **draft2s.** 학습한 드래프트 헤드가 **아니다.** lm_head argmax를 근사하는 3단 검색(약 70 MB만 읽음, argmax 99.70 % 일치)이다. 드래프트 한 스텝의 헤드 시간만 줄인다. gufo 작은 헤드와 같은 목적이다.
- **제안 순서:**
  - **M0:** 지금 HEAD 기준선을 같은 하네스로 잰다. Kyojin 카드 프롬프트와 8K/32K 문맥을 포함한다.
  - **M1:** 오프라인 MTP 사슬 trace와 정책 시뮬레이터. 정책 후보를 GPU 창 없이 비교한다.
  - **I1:** 확률 기반 토큰별 정지(Kyojin tier 방식과 gufo 비용표의 결합).
  - **I2:** lookup 조건 조정.
  - **I3:** 드래프트 스텝 비용 절감.
  - 27B는 M0 범주 분석과 I1 아이디어의 적용 여부만 본다.

## 1. gufo 현재 구조

### 1.1 Flash-Next (`src/models/qwen38_flash_next/`) [확인]

| 항목 | 내용 | 근거 |
|---|---|---|
| 드래프트 원천 | (a) MTP 사이드카 `blk.48`(nextn 1층, shared-Q8_0). 임베딩과 출력 헤드는 trunk 것을 빌린다. 같은 층을 사슬로 반복한다: 앵커 뒤 `DraftCatchUp`, 그다음 스텝마다 `MtpForward`. (b) prompt lookup(공통 `src/core/speculative/prompt_lookup.hpp`). **주의:** `ngram.cpp`는 PLE n-gram **임베딩**이고 드래프트와 무관하다 | `engine.cpp:1072-1097`, `config.hpp:79-80,131`, `ngram.hpp:21-29` |
| 혼합 규칙 | 첫 스텝은 lookup이 없다. greedy는 일치가 12토큰 이상(최근 수락 복사의 재개는 5토큰)이고 복사가 3개 이상이면 MTP를 건너뛴다(`AppendLookup(..., 3, sampled)`). 그 밖에는 MTP 드래프트를 하나 만들 때마다 lookup을 다시 찾고, 찾으면 남은 사슬을 복사로 채우고 드래프트를 멈춘다. sampled 요청은 24토큰과 확립된 원천이 있어야 하고, 현재 답은 색인하지 않는다 | `engine.cpp:1041-1051,1084-1087`, `prompt_lookup.hpp:43-54,207-213`, `README.md:49-84` |
| 길이 결정 | `MtpLengthController`: 깊이별 성공/실패를 감쇠 0.75로 센다(prior 1.5/0.5). 기대 토큰 ÷ `MtpCycleCosts`(문맥, 점유율)를 최대화한다. 완전 수락은 censored로 보고 2배 깊이를 probe한다. `BestLength == 0`이면 AR 16토큰 뒤 재시도한다. 첫 사이클은 보수적이다. **결정은 드래프트 전에 한 번 하고, 드래프트 토큰의 확신도는 쓰지 않는다.** C>1 greedy는 `MtpBatchController`(측정 ms 셀, 탐사)를 쓴다 | `mtp_policy.hpp:39-169,177-250` |
| 상한 | `kMaxMtpDraftTokens = 7`, verify ≤ `max_speculative`(8행). 8행을 넘는 lookup 블록은 기각됐다(순환 상태 rollback 필요) | `mtp_policy.hpp:15`, `EXPERIMENTS.md` "Wider lookup blocks" |
| 헤드 | 작은 헤드: 98,304행 + 1,024행 요청 표. 전체 헤드 675 MB(약 3 ms)를 대신한다. 요청의 토큰 이력에서만 결정한다 | `draft_vocab.hpp:12-48,95-105` |
| 트리/사슬 | 사슬. 트리는 없다 | `engine.cpp:1206-1215` |
| verify 배치 | 1 + 드래프트 수(≤ 8행)를 한 번에 forward한다(`ForwardMode::kVerify`). C>1은 여러 요청 행을 합친다 | `engine.cpp:1300-1307` |
| 수락 판정 | greedy: GPU argmax(`GreedyMtpPredictions`)와 드래프트가 같으면 수락하고, 첫 불일치에서 멈춘다. 문법 금지 토큰은 전체 행을 내려받아 다시 판정한다. sampled: compact top-64 후보로 p/q 거절 샘플링과 잔차 보정을 한다. lookup 토큰은 point-mass 제안이다 | `engine.cpp:1130-1215`, `mtp_sampling.hpp:76-123` |
| 출력 동일성 | README: "greedy text is byte-identical". verify 행과 AR의 산술이 같도록 맞췄다(C2–C8, 1–8 rollback prefix 일치) | `README.md:64-65`, `QUALITY.md:938` |
| 공개 수치 | `fec4759d` HTTP greedy: AR tg 28.29(d0), 28.27(d8K). MTP mixed 35.79(d0), 36.91(d8K), 39.97(d32K). 배율은 약 1.27–1.42×. repetitive 116.9 | `docs/models/qwen3.8-flash-next/BENCHMARKS.md:15-45` |
| 진단 도구 | `qwen38_flash_next_gpu_probe --mtp-audit`(독립 오라클), `--cost-audit C [--depth N]`(사이클 비용). **깊이별 수락 trace 도구는 없다** | `tests/models/qwen38_flash_next/gpu_probe.cpp:11-17`, `mtp_audit.cpp` |
| 서버 지표 | timings: `draft_rounds`(verify 스텝 수), `draft_n`, `draft_n_accepted`, `lookup_n`, `lookup_n_accepted` | `src/cli/serve/generation_metrics.hpp:231-240`, `inference_backend.cpp:2932` |

### 1.2 27B (`src/models/qwen/`) [확인]

| 항목 | DFlash2 (운영) | MTP (nextn) |
|---|---|---|
| 드래프트 원천 | z-lab DFlash2 블록 드래프터(Q4_K_M). 타깃 hidden을 주입하고 블록을 한 번에 낸다. lookup이 성공하면 블록 전체를 **대체**한다 | `QwenMtpGpuModel`: K-quant를 BF16으로 펼친다. 사슬(`ForwardTargetHidden`, `ForwardFeedback`) |
| 혼합 규칙 | `TryLookup`: greedy는 신경 수락 EMA가 중립 3 아래(`weak_neural`)일 때만 12토큰, 그 밖에는 24토큰. 메시지 경계 토큰은 넘지 않는다 | 없음 |
| 길이 | `DFlashLengthController`: 수락 길이 평균을 EMA(0.25)로 추적하고, 기하 분포 가정 + 상대 비용표(Q4 C1은 1.0→1.29로 평평함)로 고른다. C2–C8 표와 Q8 공유 폭이 있다 | 고정 `min(max_tokens, max_draft_tokens)` |
| 트리/사슬 | 사슬 | 사슬 |
| verify | ≤ 8행/요청(`ForwardVerificationChunk`), small-batch wave64 | 같음 |
| 수락 | greedy 동일성 / sampled p/q | 같음 |
| 근거 | `dflash_policy.hpp:37-251`, `hip/dflash/backend.cpp:441-519,629-639` | `hip/mtp/backend.cpp:122-155`. HTTP는 거부한다(`inference_backend.cpp:3457`). CLI `prompt`/`bench`만 된다 |
| 수치 | 64문항 텍스트: AR 대비 2.28× [2.07, 2.53], 수락률 0.453, 출력 토큰당 수락 0.663(R0-RESULTS) | 재지 않음 |

[짐작] 27B DFlash2는 C1 Q4 비용이 7개까지 거의 평평하다(1.29×). 그래서 길이 정책에서 얻을 것이 Flash-Next보다 작다. 이득이 있다면 드래프터 품질이나 lookup 쪽에서 나온다.

## 2. 비교 대상 (모두 MIT, 읽은 커밋 고정)

| 저장소 | 커밋 | 라이선스 | 로컬 얕은 복제 |
|---|---|---|---|
| [Yamz-Labs/kyojin](https://github.com/Yamz-Labs/kyojin/tree/2132bd248e3bfe07258a408669608f3d2cfdda79) | `2132bd24` (1.8.3, 2026-10-29) | MIT (Turboderp 고지 + 추가분 MIT) | `build/w9-src/kyojin` |
| [turboderp-org/exllamav3](https://github.com/turboderp-org/exllamav3/tree/cdc785db89e263e65e5849cb53e1ada156411414) | `cdc785db` (v1.6.5) | MIT | `build/w9-src/exllamav3` |
| [vcruz305/exllamav3-amd](https://github.com/vcruz305/exllamav3-amd/tree/e0ec553be92c794eee19858ce820db8ed8715c05) | `e0ec553b` | MIT | `build/w9-src/exllamav3-amd` |
| [vcruz305 Strix Halo recipe](https://github.com/vcruz305/Qwen3.8-Flash-Next-EXL3-Framework-Strix-Halo-recipe/tree/5c83a2b99e7cfc3d50879ea215c494a5335a1a11) | `5c83a2b9` | 저장소 고지 확인 필요 | `build/w9-src/strix-recipe` |

코드를 가져오지 않고 아이디어만 쓴다면 고지 의무는 없다. 표나 상수를 옮기면 해당 파일 머리에 출처 URL과 커밋을 남긴다(`mmq/VENDOR.md` 방식).

### 2.1 Kyojin (`tools/qwen/serve.py`, `tools/qwen/spec_policy.py`) [확인]

- **구조:** 서버가 `spec_policy.install`로 `Generator.iterate_draftmodel_mtp_gen`과 드래프터의 `sample_from_state`를 덮어쓴다(`serve.py:900-905`, `spec_policy.py:904-905`). 기본값은 `--draft-policy mix`, `--ndt 7`이다.
- **MTP:** Flash-Next는 `mtp_num_hidden_layers: 1`이고, 같은 블록을 사슬로 반복한다. 각 깊이의 출력 HC 스택이 다음 깊이의 `target_hidden`이 된다(`generator.py:1288-1325`, `qwen4_exp_mtp.py:31-36`). 드래프트는 argmax다. 파일 주석은 HC 스트림 연결이 "semantic guess"라고 적는다(`qwen4_exp_mtp.py:23-24`).
- **mix 규칙:** **단일 활성 요청**에서만 동작한다(`spec_policy.py:837`).
  - 문맥 suffix 일치가 16토큰 이상이고, witness(같은 gram의 이전 출현과 공통인 연속)가 3 이상이면 lookup 연속 최대 7개로 **MTP를 대체**한다. 이때 드래프터 캐시 구멍을 막으려고 MTP forward를 1번 돌린다(`:838-880`).
  - 기본값 `QWSPEC_NGRAM=(gram 2, K 7, match 16, cmin 3)`이다(`:339-356`). 주석의 근거: 5토큰 일치는 JSON/code에서 "1.7 of 3"만 수락됐고, 신경 드래프트는 "5.1 of 5.5"였다. 16토큰 일치도 템플릿 값에서는 3–6 % 느렸다.
  - 다중 요청에서는 행 폭을 맞춘다. gfx1151 MoE에는 행 비용표가 없어 `RowAlloc`이 꺼져 있다.
- **길이(기본 `TierRule`, `:251-266`):** 깊이 i의 드래프터 max softmax 확률 p를 누적곱해 reach q를 만든다.
  - 깊이 1–3은 q ≥ 0.6이면 계속 드래프트하고, q ≥ 0.3이면 그 토큰을 남긴다.
  - 깊이 4 이상은 계속에 0.95, 남김에 0.85가 필요하다.
  - 깊이 1은 항상 남긴다.
  - 근거 주석: "a row beyond the 4th pays only when its chance of being accepted is above ~0.5"(`:256`).
  - 정지는 드래프터 안의 `StopDraft` 예외로 한다. 그래서 불필요한 MTP forward도 건너뛴다(`:717-723`).
  - 대안(기본 아님): `CostRule`, `OnlineRule`(EMA로 reach 보정), `TableRule`(gufo와 같은 깊이별 EWMA + `CYCLE_MS`), 문맥별 임계값(`QWSPEC_DEPTHRULE`).
- **draft2s(`exllamav3/modules/draft2s.py:1-17`, `tools/qwen/draft2s_pack.py:24-54`):**
  - 512축 회전 + 4비트 미리보기 점수 → 히스토그램 임계값으로 512–2048행 → 3비트 Hadamard 행 사본으로 top-16 → 실제 헤드 16블록 정확 점수.
  - 경사 학습이 아니라, 헤드 Gram 행렬과 캡처한 MTP 입력의 고유분해로 CPU에서 **맞춘** 것이다.
  - 1행 greedy 드래프트 스텝에만 쓴다. 출력은 argmax와 후보 위 softmax p이고, p가 tier 규칙의 신호가 된다.
  - 헤드 1.35 ms를 약 70 MB 읽기로 줄인다. Yamz EXL3 팩 전용이고, 27B에는 경로가 없다.
- **verify:** 사슬, ≤ 8행. 수락은 타깃 샘플과 드래프트가 같은지로 판정한다(`generator.py:2229-2236`). Qwen sampled는 p/q 거절 샘플링이 아니라 "타깃 샘플이 같으면 수락"이다. 트리는 행당 약 9 ms 때문에 "dead end"라고 결론 냈다(`README.strix-halo.md:111-119`).
- **출력 동일성:** "greedy output equals plain greedy"를 20프롬프트 토큰 id로 측정했다(SERVE.md). 이를 위해 verify 커널을 행 불변으로 맞췄다(`EXL3_QSA_ROWINV`, `EXL3_VERIFY_ROW_SPLIT` 등, `serve.py:510-584`). 다중 요청, 캐시된 턴, int8 KV에서는 다를 수 있다고 스스로 적었다.
- **수치:**
  - plain/spec, 구 팩 3.47 bpw, 160토큰, greedy(SERVE.md 끝 절): chat 27.3/42.2, prose 27.3/38.7, code 27.3/52.2, multi 27.3/42.6, copy 27.2/54.7. 배율은 1.42–2.01×다.
  - 카드(1.5.10): prose 47.5, chat 52.8, code 73.2. 문맥 곡선: 8K 52.2, 32K 51.1. plain 기준선은 공개하지 않았다.
  - Kyojin이 잰 gufo v0.8.0: 카드 chat 37.3, prose 32.5, code 52.3. 8K 35.3, 32K 31.9. gufo 실행 옵션은 저장소에 없다.
  - **방법:** greedy, thinking off, 256토큰, 3회 중앙값. decode는 클라이언트 시간으로 (completion−1)/(마지막−첫 청크)다. 카드 프롬프트는 `tools/bench.py` `DECODE_PROMPTS`(종류별 3개)이고, 문맥 곡선은 실제 텍스트 8K–256K에 "essay continuation" 256토큰이다.

### 2.2 upstream exllamav3 v1.6.5 [확인]

- MTP 사슬: `iterate_draftmodel_mtp_gen`(`generator.py:1164-1283`). 기본 `ndt`는 4다(Qwen MTP `default_draft_size`, `qwen3_5_mtp.py:171`, `qwen4_exp_mtp.py:118`).
- **draft model과 n-gram 드래프트는 함께 쓸 수 없다**(`generator.py:252-254`). 대신 `-hdt` hybrid가 있다. 접미사 자동자(SAM) 일치가 H 이상이고, MTP 짧은 드래프트와 겹치는 위치에서 일치하면 H토큰으로 넓힌다(`generator.py:1470-1516`, `job.py:1877-1897`).
- **적응 길이 `-dds`/`-dc`:** `DraftConfidenceCalibrator`(`draft_confidence.py:5-110`)는 드래프터 확신도(max logit) 1.0 폭 bin별로 감쇠 0.995 수락률을 센다. 그 추정의 누적곱이 `dc`(기본 0.4)보다 작아지면 멈춘다.
- verify는 사슬, 수락은 토큰 동일성이다. 손실 없음 테스트는 top-2 차이 0.25 미만의 near-tie만 예외로 허용한다(`tests/e2e/test_speculative.py:1-11,25`). **gufo의 바이트 동일 기준이 이보다 엄격하다.**

### 2.3 vcruz305/exllamav3-amd [확인]

- `ndt` = 최대 드래프트 수(verify K+1행). `dc` = `-dds`의 목표 누적 수락 확률. 레시피 운영값은 `-ndt 3 -dds -dc 0.6`이다.
- 확신도 신호는 Qwen4Exp MTP의 raw max logit이다(`qwen4_exp_mtp.py:198-203`).
- **동적 길이의 이득은 측정되지 않았다:** 6프롬프트 평균에서 ndt 3 동적 42.25 대 정적 42.53이었고, run 간 표류 1.30 tok/s 안이다(`README.strix-halo.md:217-234`).
- 98,304열 축소 헤드: 46.48 → 48.86 tok/s, 수락 72.8 → 73.0 %, greedy 동일. 특수 토큰을 빼면 수락이 69.5 → 35 %로 떨어졌다(`#L81-109`). **gufo `draft_vocab.hpp`와 같은 설계다.**
- **라운드 해부:** 57.1 ms에 2.84토큰. verify 약 51 ms(90 %), 드래프트 루프 4.9 ms, 드래프트 prefill 1.4 ms다(`#L115-117`).
- **정적 폭 표(한 프롬프트, 유리한 경우):** rows 1/2/3/4 → 40.6/51.5/59.1/69.0 ms, 토큰/forward 1.00/1.94/2.75/3.05. ndt 2에서 수락 87.6 %, 깊이 5에서 47.8 %다.
- **문맥 깊이(레시피):** 8K 32.9 tok/s 수락 68 %, 32K 33.1 tok/s 66 %(Q4 캐시, ndt 3).

### 2.4 비교 정리

| 항목 | gufo Flash-Next | Kyojin | exllamav3 / vcruz |
|---|---|---|---|
| 드래프트 원천 | MTP 사슬 + lookup | MTP 사슬 + lookup | MTP 사슬, 또는 n-gram(배타), hybrid |
| 혼합 | greedy: 12토큰 일치 + 3개 이상이면 대체, 아니면 MTP 뒤에 붙임 | 16토큰 + witness 3이면 대체(C1만) | hybrid H: MTP와 겹치는 위치가 일치할 때만 확장 |
| 길이 신호 | 깊이별 수락 EMA × 비용표, 드래프트 **전** | 토큰별 드래프터 p 누적곱, 드래프트 **중** | 확신도 bin 보정 누적곱, 드래프트 중 |
| 최대 | 7 | 7 (대부분 3까지) | ndt(기본 4, 레시피 3) |
| verify | 사슬 ≤ 8행 | 사슬 ≤ 8행 | 사슬 |
| 행당 추가 비용 | 약 8.8 ms(드래프트 포함) | 약 10.5–14 ms(드래프트 포함) | 약 9 ms |
| 헤드 | 98,304행 + 요청 표 | draft2s 근사 검색 | 98,304열 |
| greedy 동일성 | 바이트 동일(설계와 테스트) | 토큰 id 동일(C1, 측정) | near-tie 허용 |
| 2단 헤드 | 없음 | draft2s = 헤드 근사 검색(추가 학습 없음, 팩 전용 331 MB) | 없음 |

## 3. 격차 해석 [짐작]

1. **v0.8.0과 지금 HEAD는 다르다.** v0.8.0 → HEAD 사이에 lookup 혼합(#613)과 작은 헤드(#603) 등이 들어왔다. Kyojin 카드 code(52.3)처럼 반복이 있는 부하에서는 lookup 효과가 클 수 있다. **지금 격차가 얼마인지부터 다시 잰다(M0).**
2. **verify 비용은 원인이 아니다.** 표 0을 보면 같은 D에서 gufo 사이클이 더 싸다(D ≥ 2).
3. **남는 후보는 사이클당 수락 토큰이다.**
   - (a) MTP 깊이별 조건부 수락률. 입력 결선(HC 스트림 연결), Q8 대 BF16 사이드카, 작은 헤드 탈락이 원인일 수 있다.
   - (b) 길이 정책. 확신이 높은 사슬을 깊게, 낮은 것을 짧게 자르는 토큰별 신호가 gufo에는 없다.
   - (c) lookup 조건.
   - (a)가 크면(예: 깊이 1 수락이 Kyojin 약 0.85–0.88보다 확실히 낮으면) 정책 조정보다 그쪽을 먼저 본다.
4. **작업 부하가 다르다.** gufo 공개 "mixed"는 1.27–1.42×, Kyojin plain/spec은 1.42–2.01×(종류별)다. 같은 프롬프트로 재야 비교할 수 있다.
5. **D=0 사이클 오버헤드.** 38.5 대 35.3이다. 드래프트를 안 하는 사이클(AR backoff)에서도 MTP catch-up 3.2 ms를 낸다. 수락률이 낮은 prose에서 상대적으로 크다.

## 4. 측정 설계

### 4.1 공통

- **하네스:** W2 `rq_serving.py`(`~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/`) + `tools/gufo/serving_bench.py --suite` + `tools/qwen27b/rq_quality/accept.py`(프롬프트 단위 paired bootstrap 2000회).
- **설정:** C=1, greedy, think off, `--no-cache-prompt`, warmup 1 + 반복 3(AR은 1). 드래프트 수는 반복 사이에 같아야 한다(`accept.py` 검사).
- **지표:**
  - 수락률 Σacc/Σdraft
  - 출력 토큰당 수락 토큰
  - **verify 스텝당 토큰** = completion / `draft_rounds`
  - lookup 비율 `lookup_n`/`draft_n`과 그 수락률
  - decode tok/s, 같은 프롬프트 AR 대비 배율
  - 종류별 값, 문맥 깊이(d0, 8K, 32K)
- **하네스 보강(2단계, 승인 필요) [확인: 필드 부재]:** `serving_bench.py`의 Sample은 `draft_rounds`, `lookup_n`, `lookup_n_accepted`를 저장하지 않는다(`serving_bench.py:78-80,529-533`). 세 필드를 더하고, `accept.py`에 종류별 요약과 스텝당 토큰을 더한다. 두 파일 모두 W9 경계 밖의 공유 도구이므로 감독이 정한다. 대안은 rq_serving이 원응답 timings를 따로 저장하는 것이다(ai.tools, 커밋 안 함).
- **판정:** CLAUDE.md 규칙을 따른다. 요청·단계별 5 %·10 ms, main(후보 직전 커밋)과 짝지은 대조, 영향받는 이력만 ABBA로 번갈아 잰다. 잡음은 "결론 없음"으로 남긴다.

### 4.2 프롬프트 세트

| 세트 | 내용 | 용도 |
|---|---|---|
| A | `accept_prompts.json` 64문항(종류 12개: code 9, expository 7, repetitive 6, …) | R0과 같은 기준, 27B와 공통 |
| K | Kyojin `tools/bench.py` `DECODE_PROMPTS`(prose/chat/code 각 3개) + `tools/qwen/spec_prompts.py`(chat/prose/code/multi/copy). MIT이므로 텍스트를 suite JSON으로 옮기고 출처 커밋을 적는다 | Kyojin 수치와 직접 비교 |
| D | 8K, 32K 문맥 + "essay continuation" 256토큰. 문맥 텍스트는 R0 a01 코퍼스나 저장소 문서에서 만들고, Kyojin `serve_bench_ctx.py`의 형태를 따른다 | 문맥 깊이 |

### 4.3 M0: Flash-Next 기준선 (GPU 창 1–2개)

- **바이너리:** 운영 툴체인(ROCm 10.1 / Clang 24) release. HEAD 기준 이미지(`rqw3-base-rocm101` 계열)에 Flash-Next가 들어 있는지 확인한다.
- **서버:** `gufo serve llm --model UD-Q4_K_XL/…-00001-of-00004.gguf --speculative mtp --mtp-model mtp-…-shared-Q8_0.gguf --sessions 1 --context 40960`. AR 대조는 같은 서버 인자에서 speculative만 끈다.
- **항목:** (A, K, D) × (mtp, ar).
  - [짐작] 시간: A mtp 약 64 × 256 / 35 tok/s × 4회 ≈ 31분. A ar 1회 약 10분. K와 D는 작다(D 32K prefill 포함 각 수 분).
  - 합계 약 1.5–2시간, 창 하나(`PLAN-W9-M0.json`)다.
- **모델 파일과 디스크 [확인: `df`, `du`]:**
  - 루트 여유는 **202 GiB**다. UD-Q4_K_XL은 104 GiB, MTP shared-Q8_0은 2.6 GiB다. 복사하면 약 95 GiB가 남아 **150 GiB 하한 아래로 내려간다.**
  - 외장(exFAT USB)에서 직접 실행하면 **타이밍이 오염된다.** PLE 표를 실행 중에 디스크에서 읽기 때문이다(FLASH-NEXT 7.2절).
  - 그래서 감독 결정이 필요하다. 다음 중 하나를 고른다.
    - (a) 다른 실험 파일 약 60 GiB를 정리한 뒤 루트에 복사한다(정리 대상은 감독이 확인한다).
    - (b) **수락 지표만** 외장에서 먼저 잰다. C=1 greedy의 드래프트 길이는 수락 이력에만 의존하고 시간에 의존하지 않는다(`mtp_policy.hpp:37-38`, `MtpBatchController`는 C>1에서만 시간을 쓴다). 그래서 수락률, 스텝당 토큰, lookup 비율은 디스크 속도와 무관하게 결정적이다. tok/s는 (a) 뒤에 잰다.
    - [짐작] (b)를 먼저 하면 M1/I1 판단을 루트 공간과 떼어 놓을 수 있다. (b) 창에서는 서버 로그의 `NgramTotals.disk_reads`도 기록한다.

### 4.4 M1: 오프라인 MTP 사슬 trace + 정책 시뮬레이터 (새 도구, 2단계)

- **원리 [확인: 코드 경로로 추론]:** greedy에서 깊이 d의 드래프트는 깊이 1..d−1이 모두 수락됐을 때만 쓰인다. 그 경우 사슬의 입력 토큰은 타깃 greedy 토큰과 같다. 따라서 AR greedy 출력에 대해 위치 t마다 다음을 기록하면, 어떤 길이와 lookup 정책이든 **정확히** 재생할 수 있다.
  - 위치 t에서 catch-up 뒤 MTP 사슬 7깊이의 (argmax, 작은/전체 헤드 max softmax p, top-2 차이)
  - 타깃 토큰
  - prompt lookup 일치 길이와 복사 후보
  - 그러면 위치 t에서 시작한 사이클의 수락 길이 = 앞에서부터 일치하는 개수가 된다.
- **도구:** `tests/models/qwen38_flash_next/`에 trace 모드를 추가한다(`gpu_probe`에 `--mtp-trace` 또는 새 EXCLUDE_FROM_ALL 타깃). CTest와 운영 경로는 바꾸지 않는다. 각 t에서 7번 MTP forward + rewind를 한다.
  - [짐작] 비용: 64 × 256 위치 × (AR 35 ms + 7 × 약 2–3 ms) ≈ 15–25분.
- **시뮬레이터(Python, `tools/qwen-flash/spec_sim.py` 가칭):** trace + 비용표(`kMtpCycleMilliseconds` 또는 cost-audit 실측) → 정책별 사이클 수, 스텝당 토큰, 예상 tok/s.
  - **검증:** 현재 정책(`MtpLengthController` + lookup 규칙을 그대로 옮긴 것)의 시뮬레이션 드래프트 수가 M0 서버 `draft_n`/`draft_n_accepted`/`draft_rounds`와 **정확히 같아야** 한다. 다르면 trace나 이식이 틀린 것이다.
- **산출:**
  - 깊이별 조건부 수락률 a_d(종류별) → 3(a) 판단, Kyojin/vcruz 수치(ndt 2에서 87.6 %, 깊이 5에서 47.8 %)와 비교
  - 정책 후보별 예상 이득
  - "오라클 길이"(사후 최적) 상한
- **3(a) 진단(trace 결과가 나쁠 때):** 같은 trace를 BF16 사이드카(`mtp-…-shared-BF16.gguf`, 외장에 있음)와 전체 헤드로 반복한다. 양자화, 작은 헤드, 결선 중 어디서 오는지 가른다. 결선은 기존 `--mtp-audit` 오라클과 upstream `qwen4_exp_mtp.py`를 대조해 확인한다(둘 다 "semantic guess"라고 적었으므로, 어느 쪽이 맞는지는 수락률로만 가릴 수 있다).

### 4.5 27B

- **CPU만 필요 [2단계 첫 작업]:** 기존 `rq0-accept-a01` 보고서(UD-Q4_K_M dflash2/ar)를 `category`별로 다시 집계한다. 수락률, 토큰당 수락, 배율을 본다. GPU 창은 쓰지 않는다. 이번 조사 중에는 GPU 창 컨테이너가 돌고 있어 계산을 미뤘다.
- 32K DFlash2 텍스트 수락은 R0에서 재지 않았다. 세트 D를 27B에도 돌릴지는 감독이 정한다(창 하나, 약 30분 [짐작]).

## 5. 개선안

이득은 모두 [짐작]이고, M1 시뮬레이터로 먼저 확인한 뒤 구현한다. 모든 안은 **드래프트 제안만 바꾸고** 타깃 verify, 수락, 샘플링 산술은 그대로 둔다. 그래서 greedy 출력은 바이트 단위로 같아야 한다.

| # | 안 | 내용 | 예상 배율 (Flash-Next mixed) | 구현 크기 | 위험 | 품질 |
|---|---|---|---|---|---|---|
| I1 | **토큰별 확신도 정지** | 드래프트 스텝마다 작은 헤드의 max softmax p(argmax 커널에서 logsumexp를 함께 계산해 호스트로 읽음)를 얻는다. 기존 깊이별 EMA 대신(또는 함께) "p로 보정한 조건부 수락 추정 × 비용표 한계 이득"으로 계속/정지를 정한다. 최대 깊이는 7 그대로이고, 확신이 높으면 깊게, 낮으면 즉시 멈춘다. 보정은 gufo처럼 커밋된 수락 이력만 쓴다(시간 비의존, snapshot 상태에 포함) | 1.05–1.15× (사이클당 낭비 행 감소 + 깊은 사슬) | 중: `mtp_policy.hpp` 새 컨트롤러, executor 드래프트 경로 p 출력, snapshot 버전 올림, 단위 테스트 | 중: 호스트 동기화가 늘어날 수 있다(이미 스텝마다 토큰을 읽는지 확인 필요). vcruz `-dds`는 정적 대비 이득을 못 보였다. **시뮬레이터에서 +3 % 이상일 때만 진행한다** | 제안만 바뀜. greedy 동일 |
| I2 | **lookup 조건 조정** | 시뮬레이터로 최소 일치(12/16), witness 조건(Kyojin cmin 3), 재개 5토큰, "MTP 뒤에 붙이기 대 대체"를 스윕한다. greedy C1 기준이고, sampled 규칙(24토큰)은 건드리지 않는다 | 0.98–1.05× (부하에 따라 다름. copy/code 쪽 이득) | 소: `prompt_lookup.hpp` 상수나 술어, Flash-Next `AppendLookup` 조건 | 중: **공통 파일**이라 27B DFlash2에도 영향이 간다. 27B 쪽은 감독이 순서를 정한다(W3) | greedy 동일 |
| I3 | **D=0 사이클 오버헤드** | AR backoff 사이클의 catch-up 3.2 ms를 줄인다(KV-only catch-up은 이미 있다. 남는 비용을 프로파일로 찾음, `tools/prof/prof.py`) | 1.00–1.03× (prose나 수락이 낮은 부하) | 소–중 | 낮음 | 동일 |
| I4 | **드래프트 스텝 헤드 비용** | 작은 헤드 98,304 × 2560 Q8(약 267 MB, 약 1.3 ms/스텝 [짐작])를 draft2s 방식의 거친 선별 + 정확 재점수로 바꾼다. 선별용 저비트 사본과 회전은 오프라인에서 만든다(학습 없음). 재점수는 기존 Q8 행이라 argmax가 같은지 검사할 수 있다 | 1.02–1.04× (드래프트 2–3개 기준) | 대: 새 사이드카 형식 + 커널 2개 + 맞춤 도구 | 중: 0.3 % argmax 불일치는 수락률만 낮춘다 | 동일(제안만) |
| I5 | MTP 품질 원인 수정 (3(a)가 확인될 때만) | 결선, 양자화, 헤드 원인에 맞춰 고친다(예: BF16 사이드카 사용) | 결과에 따라 다름 | 결과에 따라 다름 | — | 동일 |
| — | 트리 verify, 8행 초과 | 하지 않는다. MoE 행당 약 9 ms라 Kyojin, vcruz, gufo 모두 기각했다 | — | — | — | — |
| — | Flash-Next DFlash2 | 하지 않는다. 공개 드래프터가 없고, gufo 실험 기록에 감사가 필요하다고 적혀 있다(EXPERIMENTS "Flash-Next DFlash2") | — | — | — | — |

**27B:**

- I1 아이디어를 DFlash2에 적용할 수 있다. 블록 logits에서 토큰별 p를 얻어 verify 창을 자른다. 다만 C1 Q4 비용이 평평해서(7개 1.29×) [짐작] 이득은 1–3 %다. 드래프터 블록 자체는 이미 계산했으므로 줄일 수 있는 것은 verify 행뿐이다.
- 27B MTP(nextn)는 HTTP로 노출되지 않고 길이도 고정이다. DFlash2가 2.28×이므로 우선순위가 낮다.
- 27B는 M0 범주 분석 결과가 특정 범주(예: prose)의 약점을 보일 때만 다시 본다.

**출력 동일성 검증(2단계, 구현마다):**

- 기존 `tests/functional/`(`long-context`, `cache` 포함)과 Flash-Next session/replay 테스트를 돌린다.
- M0 세트 A, K에서 후보와 main의 completion sha256이 64/64 같아야 한다. `serving_bench`가 `completion_sha256`을 낸다.
- 컨트롤러를 바꾸면 snapshot payload 버전(`engine.hpp:240`)을 올리고, 복원 테스트를 한다.

## 6. 제안 순서와 필요한 결정

1. **(CPU, 허락 불필요한 가벼운 일)** 27B R0 수락 보고서를 범주별로 재집계한다. Kyojin 프롬프트로 suite JSON을 만든다(세트 K, D). 결과는 이 문서에 적는다.
2. **(감독 결정)**
   - 하네스 보강 방식: `serving_bench.py` 필드 추가, 또는 rq_serving 원응답 저장.
   - 모델 배치: 4.3의 (a) 또는 (b).
3. **M0 창:** Flash-Next A/K/D × mtp/ar. **판단점:** HEAD의 배율이 이미 Kyojin(종류별 1.42–2.01×)에 가까우면, W9는 I2/I3만 하고 마친다.
4. **M1:** trace 도구 + 시뮬레이터(새 파일, Flash-Next 테스트 디렉터리와 `tools/qwen-flash/`). 빌드 1회, GPU 창 1회(약 30분). 현재 정책 재현 검증, a_d 표, 후보 정책 표를 낸다. **판단점:** a_d가 Kyojin/vcruz보다 확실히 낮으면 I5(원인 진단)를 먼저 한다.
5. **I1**(시뮬레이터 예상 +3 % 이상일 때) → **I2** → **I3** 순서로, 각각 구현 + 동일성 테스트 + main 대조 ABBA를 한다.
6. **I4**는 I1–I3 뒤에 남은 격차가 드래프트 스텝 비용으로 설명될 때만 한다.

**2단계 파일 경계(제안):**

- `src/models/qwen38_flash_next/{mtp_policy.hpp,mtp_costs.hpp,engine.cpp,engine.hpp}`, 필요하면 `kernels/rocm/executor.cpp`(드래프트 p 출력)
- `tests/models/qwen38_flash_next/`(trace 모드, 컨트롤러 테스트)
- `tools/qwen-flash/spec_sim.py`(새 파일)
- `plan/rotated-quant/SPEC-DECODE.md`, `plan/rotated-quant/W9-RESULTS.md`
- 공유 파일(`src/core/speculative/prompt_lookup.hpp`, `tools/gufo/serving_bench.py`, `tools/qwen27b/rq_quality/accept.py`)은 따로 승인을 받는다.

## 7. 남은 불확실성

- Kyojin이 gufo v0.8.0을 어떤 옵션(`--speculative mtp` 여부, `--draft-tokens`, 시간 측정 방식)으로 쟀는지 저장소에 없다 [확인: 문서와 `figures.json`만 있음].
- Flash-Next MTP 입력 결선은 Kyojin/upstream도 "semantic guess"라고 적었다. 어느 쪽이 맞는지는 수락률 비교로만 판단할 수 있다.
- gufo 드래프트 루프가 스텝마다 호스트 동기화를 하는지(`MtpForward`의 `.token` 출력), 그래서 I1의 p 읽기가 추가 비용 없이 되는지는 executor를 더 읽어야 한다(2단계 시작 때 확인).
- vcruz 레시피 저장소의 라이선스 파일은 확인하지 않았다(아이디어만 참고).
