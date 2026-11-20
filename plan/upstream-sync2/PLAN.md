# upstream v0.35 / Kyojin / exllamav3 가져오기 계획 (V-series, 2026-11-20)

검토 범위는 세 가지다.
- gufo-org/gufo `6d793814`(v0.31.0)부터 `upstream/main` `3e3708b5`(v0.35.0, 11-19)까지, 커밋 114개.
- Kyojin `656dab5..HEAD`.
- exllamav3의 최근 6~8주.

조사 보고서 원문은 `plan/upstream-sync2/survey-upstream.md`와 `survey-kyojin-exl3.md`에 있다. 이전 범위(U-series)의 결과는 `plan/upstream-sync/`에 있고, 거기서 미룬 U2-4는 그대로 미룬다.

## 원칙

- `plan/upstream-sync/PLAN.md`의 원칙을 그대로 따른다.
  - 전체 merge는 하지 않는다. 골라서 cherry-pick하거나 손으로 이식한다.
  - 커밋 제목에 upstream PR 번호를 적는다.
  - 항목마다 구현 → 두 번째 리뷰어 검토 → 수정 → 재검토를 하고, 반복 중에는 빌드와 CPU 단위 테스트만 돌린다.
  - 묶음이 끝나면 GPU 통합 패스를 한 번 한다.
- `tools/exl3_quant/hc/`와 `plan/exl3/HC-Q8-CALIBRATION.md`는 건드리지 않는다.
- 운영 반영은 사용자 확인 뒤에 한다. 백업은 `~/projects/_backup`에 둔다.
- `~/projects/_ai.tools`의 코드와 이미지(`ai-tools/*` docker 이미지 포함)는 실행하지 않는다.

## V0 비용표 재보정 (구현 완료, GPU 속도 확인은 V1 패스에서)

- `mtp_costs.hpp`를 운영 c03 EXL3 / K4V4 / YaRN 4 조건으로 다시 쟀다(`build/b1/cost-20261120/`).
- probe에 `--kv-cache-type`을 추가했고, `--batch`를 반영하게 했다.
- 결과는 KNOWHOW "MTP cycle-cost table recalibrated"에 있다.

## V1 작은 버그 수정 (1묶음)

| # | 출처 | 내용 | 방법 | 우리 쪽 위치 |
|---|---|---|---|---|
| V1-1 | 800ce328 (#1230) | MTP batch compact tail의 inject 행이 `xn`(=mtp_h)과 겹치는 GPU race | 손 이식 7줄 | `runtime/executor_batch.cpp:469-487` |
| V1-2 | 6ecfa504 (#1249) | GPU가 읽는 host 버퍼를 compaction 대상에서 빼기: anonymous pages + io_uring 장기 pin + hipHostRegister, 실패하면 hipHostMalloc | cherry-pick, 우리 쪽 raw hipHostMalloc 호출부 변환, pin 성공 바이트 로그 | `src/core/hip/host_allocations.*`, `inference_backend.cpp:1155`, `qwen/hip/arena.cpp:1275` |
| V1-2b | 같은 항목 | 컨테이너에서 pin이 실제로 성공하게 하는 설정(`--ulimit memlock=-1`, io_uring을 허용하는 seccomp 프로필) | 시험 창에서 후보 서버에만 적용하고, 운영 반영은 확인 뒤에 | 감독 스크립트 |
| V1-3 | 4e95aecb (#1146) | tool-call 숫자를 모델이 쓴 그대로 돌려주기(캐시 재사용) | cherry-pick | `src/core/json.*`, `openai_chat.cpp` |
| V1-4 | 390639b3 (#1273) | host 여유분을 CmaFree와 4 GiB 중 큰 값만 빼도록(지금은 둘 다 빼서 약 4 GiB 손해) | 손 이식 약 25줄 | `text_model_runner.cpp:108-129, 211` |
| V1-5 | 53193819 (#1168) | JSON 숫자를 16자리로 제한(llama.cpp와 같음) | cherry-pick | `src/core/json_constraint.cpp:179-184` |
| V1-6 | 72442dce (#1102) | `</parameter>`로 시작하는 값을 한 호출로 읽기 | 경로를 바꿔 cherry-pick | `src/models/qwen/chat_template.cpp:433` |
| V1-7 | 49b5df18 (#912) | batch 관측 뒤 낡은 MTP backoff 해제 | 손 이식 약 15줄 | `mtp_policy.hpp:80-112`, `engine.cpp` |
| V1-8 | fe2694bc (#1104) | 레코드 표가 가득 차 굳는 것을 막기(stale victim) | cherry-pick | `src/cache/retention.cpp:402` |
| V1-9 | 77621b27 (#1181) | capture가 끝난 prefill이 peer 앞에서 재개 | 손 이식 8줄 | `text_generation_scheduler.cpp:2631` |
| V1-10 | 3016d652 (#1117) | snapshot 복사를 host 등록 경계에서 나누기 | cherry-pick | `src/core/hip/snapshot_copy.hip` |

**V1 GPU 통합 패스**
- 후보와 운영(runtime-20261120a)을 운영 설정으로 비교한다. 시험 디스크 캐시를 쓰고, 끝나면 운영을 복구한다.
- 기능: functional `batch`, `cache`, tool-schema 묶음, `pi_agent.py` 한 번, `mtp_sampling`/session batch 테스트.
- 속도:
  - C1 sampled와 greedy, C4, 문맥 d0과 약 128K. 요청·단계별로 5%/10 ms 기준.
  - 4세션 MTP 수락률(V1-1, V0).
  - compaction_proactiveness 20과 50에서 MTP(V1-2), pin 성공 로그 확인.
  - U4-3 hang loop 한 번.

**V1 후속 (남은 경계 사례, Codex 검토 2026-11-20).** 생각 중 따옴표가 덜 닫힌 뒤 복구한 도구 호출을 검사할 때 남은 문제다(`openai_chat.cpp` recovered-call 검사, upstream도 같음).
- 정수 `1.0`과 `1e-7`이 한 호출에 같이 있으면 두 철자 어느 쪽으로도 통과하지 못한다. canonical 철자에서 지수 앞 0을 빼면 해결된다.
- 2^53을 넘는 정수는 canonical double로 바뀐 값으로 경계를 검사한다. 숫자 잎마다 십진값이 그대로일 때만 canonical로 대체하면 정확해진다.

**V1 패스에서 새로 찾은 기존 문제 (2026-11-20, V1과 관계없음).** 운영 runtime-20261120a에서도 똑같이 실패한다.
- 짧은 대화를 다시 보내면 캐시를 다시 쓰지 못한다(`cached_tokens` 0). 끊긴 대화를 이어 갈 때도 앞부분을 잃는다(functional `tools`, `structured*`, `tool-schema-edges`, `*-cancel`). 11-18 RAM 층 제거(c597eed1) 뒤부터인 것으로 보이며, 원인 조사가 필요하다.
- `qwen38_flash_next_session_test` `CheckServingEos`는 캐시 없는 backend에서 prefill 0 재생을 기대한다. 시험을 디스크 캐시 기준으로 고치거나, 위 문제와 함께 다룬다.

## V2 속도 개선 (2묶음, 중간 크기)

| # | 출처 | 내용 | 비고 |
|---|---|---|---|
| V2-1 | 850b5271 (#1158) | 다시 토큰화한 답이 다르게 쪼개져도 live frontier를 그대로 잇기(에이전트 대화의 재-prefill 줄이기) | 손 이식 약 250줄, `continue_final_message`와 함께 시험 |
| V2-2 | bd6b0fdd (#1128) | 중첩 tool 객체의 기본값을 llama.cpp에 맞추기 | 손 이식 약 120줄 |
| V2-3 | 0a473ebf (#1288), cc405337 (#1290) | 짧은 새 대화에 보존 유예, 편집 기록 유지 | 손 이식 약 55줄, retention replay로 확인 |
| V2-4 | a33323f2 (#1209) | 초안 head prefix의 4비트 사본(upstream C1 sampled +2.6%) | 300~400줄, EXL3 K8에서 바로 변환 |
| V2-5 | 11a51d2f (#1285) | 디스크 복원 전에 backing을 미리 확보(서버 스레드 GiB당 약 57 ms 멈춤 제거) | 약 100줄 |
| V2-6 | 70bd0fda (#1205) | 디스크 payload fsync를 동시에 | 약 130줄 |
| V2-7 | exllamav3 47d42fe | 빈도 순 초안 어휘, 크기는 miss율 기준 | 추정 C1 MTP +4~6%(측정 아님), 한국어·코드 수락률 확인 필요 |
| V2-8 | fdbc654e (#1270) | 공유 branch point를 192토큰에서 학습 | 약 10줄, GDN 상태 크기 때문에 효과 불확실 |
| V2-9 | franzmoca e2ecbf12, 41d7d621 | prompt-lookup 복사가 다른 일치와 어긋나는 곳에서 멈추기 | 약 50줄, 숫자 없음 |

## V3 큰 항목 (3묶음, 각각 따로 시험)

| # | 출처 | 내용 | 비고 |
|---|---|---|---|
| V3-1 | 2e50748f (#1265), 425e3513 (#1274), #1251의 arena 부분 | 혼자 들어온 긴 입력을 14K 행 pass로, 넓은 scratch는 그 동안만 매핑 | upstream 28K~85K 입력 prefill +2~7%. 우리 16K 실험은 +2.8%였지만 +6.38 GiB 때문에 기각했는데, 이 설계가 그 문제를 없앤다 |
| V3-2 | Kyojin 9f1b368, 2484e31 (+ 우리 da021c03) | typical acceptance 재도입. 복사 초안은 exact, 12-gram 반복 거부, 루프 신호 뒤 exact 창, `</think>` 뒤에만 완화하는 선택 | U5에서 sampled +2.5~27.6%. 11-10 반복 재현(숙소 후기)을 통과해야 하며 opt-in으로 시작 |
| V3-3 | exllamav3 805caee, Kyojin loopguard | 반복 감지: 멈추거나 열린 think를 닫기 | 두 모델이 함께 쓰는 scheduler에 넣는다. 코드·JSON 오탐 때문에 opt-in |
| V3-4 | exllamav3 0244652, Kyojin fce7501 | GDN 상태를 FP16으로 저장하고 FP32로 계산 | Kyojin 27B 8×16K decode +7.7%, KL 0.0004. 예전 BF16(#138)은 기각했으므로 27B에서 먼저 품질 확인. 캐시 형식이 바뀐다 |
| V3-5 | 27B in-pass capture (a59df291, #1224) | 27B prefill 중 checkpoint | 600~1000줄, 지금은 효과 대비 비용이 낮음 |

## 하지 않는 것

조사 보고서의 "Rejected" 목록을 따른다. 요약하면 다음과 같다.
- Rust 변경. 포크에 Rust 도구가 없다.
- GGUF 전용 항목: #1112, #958, #969.
- FP16 KV 전용 #1185.
- RAM 캐시를 끈 모드에서만 쓰는 항목: #934, #1272.
- 수요 매핑 세션 인프라 #1251, #1258, #1275(약 7K줄).
- Kyojin GDN HF chunk kernel: U4-2에서 기각한 그대로.
- exllamav3 CUDA 전용 커널.
