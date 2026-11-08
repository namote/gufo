# upstream / Kyojin 가져오기 계획 (U-series, 2026-11-08)

`upstream/main`(gufo-org/gufo `6d793814`, v0.31.0)과 Kyojin main(`656dab5`,
v1.8.14 이후)을 검토한 결과로 세운 계획이다. 검토 근거는 이 문서 끝
"검토 요약"에 있다.

## 원칙

- **전체 merge는 하지 않는다.** `git merge-tree` 결과 105개 파일, 228개 hunk가
  충돌한다. 게다가 upstream의 Flash-Next 수정 약 15건은 우리가 B9에서 쪼개
  지운 `kernels/rocm/executor.cpp`·`kernels.hip.cpp`에 들어 있어 merge하면
  조용히 빠진다. 커밋을 골라 cherry-pick하거나 손으로 이식한다.
- **출처를 남긴다.** 커밋 제목에 upstream PR 번호를 적는다(예:
  `fix(json): … (upstream #1012)`). cherry-pick은 `-x`로 원 커밋 해시를 남긴다.
- **작업 방식(CLAUDE.md):** 항목마다 구현 → 두 번째 리뷰어 검토 → 수정 →
  재검토를 한다. 반복 중에는 빌드와 CPU 단위 테스트만 돌린다. GPU 검증과
  성능 측정은 U5에서 한 번에 한다.
- **GPU 커널을 건드리는 항목**(U1-1, U2-1, U2-3, U4)은 바뀐 커널만 device code가
  달라지고 나머지 커널은 그대로인지 `build/b1/cmp/` 비교로 확인한다.
- **동시 작업:** 다른 세션이 `tools/exl3_quant/hc/`와
  `plan/exl3/HC-Q8-CALIBRATION.md`(HC Q8 보정)를 작업 중이다. 그 파일들은
  건드리지 않고, 커밋할 때는 이 작업의 경로만 stage한다. GPU를 쓰는 U5는 그
  세션의 GPU 수집과 시간이 겹치지 않게 맞춘다.
- push는 포크(`namote/gufo`)로만 한다. 사용자가 요청할 때만 한다.

## 시작 전 확인 (2026-11-09)

- HC down의 남은 F16 경로를 보정 Q8로 통일하는 안은 HC 보정 세션이 이미 두 가지 방식으로 시도했고, 둘 다 품질 검사에서 떨어져 기각했다(`4db5d3d8`, KNOWHOW "Remaining standalone HC prefill down"). 이 계획에 넣지 않는다.
- 현재 서비스는 c03 + all200 GPTQ HC Q8 사이드카(`runtime-hc-20261108`)다. U5의 비교 기준은 B10이 아니라 이 런타임이다.

## U1 바로 가져올 수정 (작은 것, 버그·안전)

| # | upstream | 내용 | 방법 | 우리 쪽 대상 |
|---|---|---|---|---|
| U1-1 | 81ff13c3 (#849) | draft-ID remap과 sparse block selection을 버퍼 크기 안으로 제한(OOB 읽기) | 손 이식 (~30줄) | `kernels/rocm/mtp.hip.cpp` `RemapDraftIdsKernel`, `attention_select.hip.cpp`, `attention.hip.cpp`, 호출부 `runtime/executor_mtp.cpp`·`executor_attention.cpp` |
| U1-2 | da04ba52 (#1012) → 0f251d4a (#1052) → de29d6a7 (#1059) | strict tool schema: items 없는 array, 일반 type union, `$schema` 같은 annotation 키워드 | cherry-pick, #1052의 extension 의존 hunk만 손 해결 | `src/core/json_constraint.cpp`, 테스트 |
| U1-3 | f07fd2fe (#1064) | grammar가 reasoning을 끝내면 reasoning budget도 끝냄 | cherry-pick | `src/core/sampling.cpp` |
| U1-4 | 1175eebe (#1043) | `tool_choice=required`에서 tool opener 앞 구분자 제거 | cherry-pick | serve |
| U1-5 | b911c150 (#964), 31c3b3a0 (#881) | 용량 초과 시 크기·수용 가능 세션 보고, 다른 slot이 바쁠 때 idle-spill polling 완화 | cherry-pick | serve, `src/cache/` |
| U1-6 | abbc5cb8 (#858), 118b7c63 (#1021), 1a6655fc (#740) | unbuffered read 실패 시 buffered 재시도, padding 없는 GGUF shard, qfn_mmq arena 점진 확장 | cherry-pick (테스트 충돌만 손 해결) | loader, gguf |
| U1-7 | 80786f73 (#1048) 일부 | graph capture 예외 안전 `CaptureScope` (성능 부분은 이미 있음) | 손 이식 | `runtime/executor_forward.cpp` |

완료 조건: 빌드, 관련 CPU 테스트, check-format/check-docs가 통과하고 리뷰어 검토가 깨끗해야 한다. U1-1은 바뀐 커널 외의 device code가 그대로여야 한다.

## U2 결정성 (같은 입력 → 같은 출력)

| # | upstream | 내용 | 방법 |
|---|---|---|---|
| U2-1 | 2e6aa96c (#876) | sparse budget(2048)을 넘는 MTP chain에서 budget 이전 행은 dense sweep(`dense_extent`). 지금은 배치 타이밍에 따라 출력이 갈릴 수 있음 | 손 이식: `attention.hip.cpp`, `runtime/executor_attention.cpp`, `kernels.hpp` |
| U2-2 | 18c89e3b (#877 정렬 부분) + 186a507d (#1016) | prefill pass 끝과 checkpoint를 compress_ratio(4)의 배수로 정렬해, 2K 넘는 프롬프트를 cache에서 복원해도 cold 실행과 bit가 같게 | 먼저 우리 B10에서 재현되는지 테스트로 확인하고 손 이식: `src/cli/serve/inference_backend.cpp`(step 계산), `engine.cpp`, component cache |
| U2-3 | 192b75a8 (#875) 일부 | prefill 행 결과가 pass route와 무관하도록 blaslt `OwnF16(type,k)` 일반화, HC inject 덧셈 순서 고정. GGUF 쪽 부분은 제외 | 손 이식: `runtime/blaslt.cpp`, `kernels/rocm/hc.hip.cpp` |
| U2-4 | 8ab834a5 (#903), 0d12f7b7 (#1063) | RAM-cache segment 거부 조건 완화, 감당 못 할 RAM cache는 bad_alloc 대신 해결 방법을 담아 거부 | cherry-pick + 손 해결 |

U2-2와 U2-3은 prefill bit가 바뀔 수 있다. 바뀌면 U5에서 품질 검사(KL256/top-1/NLL)를 한다.

열린 문제 하나를 함께 본다. B10에서 남은 session 테스트 `CheckBatchedSessions` / `CheckLookupBatch` 실패(snapshot 꼬리 7~23,822바이트 차이, `plan/exl3/HANDOFF.md` "Open")가 U2-1과 같은 원인인지 확인한다. 아니면 U5에서 bisect한다.

## U3 기능

| # | upstream | 내용 | 방법 |
|---|---|---|---|
| U3-1 | e590d170 (#814) | 생성 token logprobs | cherry-pick (코드 충돌 0, `session_test.cpp` hunk 3개) |
| U3-2 | 4ab04ab8 (#860), 6ab4569b (#915), 6d793814 (#1082) | OpenAI input token count, `reasoning_tokens`, `/tokenize` | 연쇄 cherry-pick (#1082는 docs만 충돌) |
| U3-3 | b7a2d616 (#811) | `cache_persist=false`: 요청 prompt를 disk cache에 남기지 않음 | cherry-pick + 손 해결 (hunk 8개) |

`tools/serving/flash_next`의 supervisor는 바꾸지 않는다. 새 API는 서버 바이너리만으로 동작한다.

## U4 성능 (B-series처럼 따로, 측정은 U5 또는 별도 창)

| # | 출처 | 내용 | 기대 | 먼저 할 일 |
|---|---|---|---|---|
| U4-1 | upstream a6c53a62 (#895) | prefill DeltaNet norm·gate·recurrence·epilogue를 GDN layer당 launch 하나로 fuse (출력 byte 동일) | PP +~1.5% | operator bench로 우리 `GdnRowSplitKernel` 대비 확인 |
| U4-2 | Kyojin `0e03454`, `656dab5` (`gdn_hf.hip`) | 16행 WY chunk F16 WMMA GDN prefill | PP 최대 ~+4%, 품질 게이트 필요(Kyojin KLD 0.006) | U4-1과 같은 커널이라 둘 중 하나를 고른다. 이전 기각 기록 `docs/models/qwen3.8-flash-next/EXPERIMENTS.md:487` 재검토 |
| U4-3 | upstream 96c9a0f5 (#961) + 805c9421 (#993) | prompt-lookup 복사를 최대 64행 폭 pass 하나로 검증 | 코드 재작성 작업에서 upstream +50%. 에이전트 작업에 가장 큼 | EXL3 9~64행 projection·routed 경로 설계. segment당 ~270 MiB 추가 |
| U4-4 | Kyojin `d8dbb75` (`spec_sample.py:typical_rows`) | sampled 요청용 typical acceptance (opt-in, 분포 비보존) | sampled MTP +~7%, greedy 0 | 옵션 이름과 품질 게이트 정의 |
| U4-5 | upstream eb1d52e6 (#894) | decode HC combine·Q8_0·SiLU·epilogue 한 launch | TG +≤1% (B3/B4 tail fusion 손해 기록 있음) | `hc_mix_bench`로 screen만 |

U4는 U1~U3과 U5가 끝난 뒤 사용자와 범위를 다시 정한다.

## U1~U3 진행 상황 (2026-11-09, GPU 실행 없음)

모든 항목은 구현 → 빌드(`build/b1`, pinned ROCm 10.1 컨테이너) → 관련 CPU
테스트 → 별도 리뷰어 검토 → 수정 → 재검토를 거쳤다. 기준 바이너리는 배포 중인
`runtime-hc-20261108/gufo`(device code는 B10과 같음)다.

| 항목 | upstream | 커밋 | 상태 | 비고 |
|---|---|---|---|---|
| U1-1 | #849 | `4f049197` | 완료 | 손 이식(attention_select/attention/mtp.hip.cpp, executor_mtp/batch). device code: AttentionKernel 6, SelectMarkKernel 2, SelectScoreKernel, CompactSelectionKernel, RemapDraftIdsKernel(시그니처)만 바뀜. 경계 안 입력의 출력은 그대로 |
| U1-2 | #1012, #1052, #1059 | `41623db8`, `0f6f7e63`, `d07fc9a0` | 완료 | #1052: `StringValue` 대신 우리 `Lexeme(JsonSchemaLexeme::String)` 유지, functional suite 목록·impact.json 손 해결 |
| U1-3 | #1064 | `518445f8` | 완료 | ext change note 제외 |
| U1-4 | #1043 | `746c40f1` | 완료 | |
| U1-5 | #964, #881 | `d992830d`, `c039686d` | 완료 | |
| U1-6 | #858, #1021, #740 | `ea847c1e`, `a33d7ec4`, `64e15f15` | 완료 | #740은 host만 바뀜(kernel 전부 동일). `hip_weight_upload_test`는 GPU 테스트라 빌드만 |
| U1-7 | #1048 일부 | `9db74cf5` | 완료 | `Run`의 CaptureScope + `Moe`의 SharedScope(예외 시 side stream join). decode shared-expert overlap은 가져오지 않음. host만 |
| U2-1 | #876 | — | 해당 없음 | 같은 수정이 이미 있다: `cef16057`의 `dense_prefix = indexer_top_k`(`pos < dense_prefix` ⇔ upstream `n_kv <= dense_extent`), EXL3 경로에만 켜짐. 운영은 EXL3라 그대로 둔다. GGUF Flash-Next는 서비스하지 않는다 |
| U2-2 | #877 정렬 부분 + #1016 | `44efce72` | 완료 | 재현 확인(코드 추적): sparse WMMA prefill이 pass 시작부터 4 query씩 묶으므로 4의 배수가 아닌 곳에서 시작한 pass(캐시 복원·용량 분할)는 cold와 bit가 다르다. 우리 tree엔 정렬이 없었다. 이식: `Session::Feed`, runner `StepTokens`, `prefill_alignment`, ComponentTextCache 복원·publish 필터, snapshot payload v19. 포크 전용: tail checkpoint도 정렬, live frontier 복사본(CaptureBeforeReset·WriteBackIdle)은 `kGenerated`로 표시해 위치와 무관하게 복원(live 재사용과 같은 상태), ShareCheckpoint slack은 정렬 위치만. disk readahead 수정·docs/models 측정·fixture는 제외 |
| U2-3 | #875 일부 | `1b22e7c5` | 완료 | HC inject partial 순서 고정 + blaslt `OwnF16(type,k)`. device code: `HcMixEpilogueF16Kernel<false/true>`만 바뀜(나머지 `__fadd_rn` 고정은 같은 코드로 컴파일). `OwnF16` 변경은 GGUF dense 경로만(EXL3는 BlasLt를 안 씀). SSM conv·wave64 W8A8·hc_route_ops 등 테스트는 제외 |
| U2-4 | #903, #1063 | — | 보류 | #903: 우리 backing 승인 코드는 request memory budgeting(`e5002432`, pending/reserve)으로 다시 쓰여 upstream 구조와 맞지 않고, 운영 RAM cache 2 GiB는 시작 때 GrowAhead(2 GiB)로 전부 commit돼 on-demand segment 거부가 생기지 않는다. #1063: 시작 시 RAM 부족이면 거부하는 변경으로, 다르게 쓰인 초기화 코드에 GPU 검증 없이 넣으면 지금 동작하는 설정을 거부할 위험이 있다. 필요해지면 따로 이식 |
| U3-1 | #814 | `5208c592` | 완료 | session_test main의 `--draft-vocab-only`·`--serving-sampling-only`와 `--logprobs-only` 병합. 리뷰: 모든 Flash-Next decode 경로(단일·batch·MTP greedy/sampled·lookup·terminal)에서 토큰마다 logprob 1개, 요청이 없으면 비용 없음 |
| U3-2 | #860, #915, #1082 | `2be31111`, `2917bfb8`, `a8d46243` | 완료 | #1082 docs는 `docs/SERVER.md`로 |
| U3-3 | #811 | `84fd6206` | 완료 | 포크의 `system_prefix_tokens` 인자 뒤에 `persist`. 포크 전용: persist=false 요청의 live frontier는 WriteBackIdle·CaptureBeforeReset이 disk에 쓰지 않음(`Entry::live_persist`), `CaptureRefused`도 반영 |

전체 device code(기준 → `84fd6206`): 위 U1-1의 12개 symbol과 U2-3의
`HcMixEpilogueF16Kernel` 2개 외에 7,045개 symbol이 그대로다.

반복 중 CPU 테스트: json_constraint, sampling, logprobs, openai_chat,
http_server, text_model_runner, text_generation_scheduler, component_text_cache,
cache_adapter(+lifecycle), cache_slot, cache_idle_spill, cache 단위 테스트,
continuation_disk_store, gguf_reader, speculative_verification, ds4.sampling,
functional_runner, cache_workloads, test_impact 통과. GPU 테스트 target
(attention/select/mtp/hc_mix/blaslt/projection ops, session/snapshot test,
decode_attn_bench, hip_weight_upload)은 빌드만 했다. 이번 작업과 무관한 기존
실패: `serve_cli_test`(help 문자열 `kvarn-k4v2` 기대 불일치), `cache_disk_faults_test`
(LD_PRELOAD gate가 컨테이너에서 동작 안 함), `cache_backing*`(GPU 필요).

### U5가 GPU에서 확인할 것

- **bit가 바뀔 수 있는 것**
  - U2-2: 캐시 복원·용량 분할 뒤 prefill이 정렬 위치에서 시작하므로 restore 결과가 cold와 같아져야 한다(기대: 바뀜 → 일치). cold 한 pass prefill logits sha(B10 `d051e822…`)는 그대로여야 한다. 2 K 넘는 prompt의 여러 pass cold prefill은 pass 경계가 이미 4의 배수(2048)라 그대로여야 한다.
  - U2-3: `HcMixEpilogueF16Kernel`이 바뀜 → 96행 미만 prefill pass(짧은 꼬리, 복원 뒤 짧은 pass)의 inject partial 마지막 bit. 넓은 pass는 그대로.
  - U1-1: 경계 안 입력은 그대로(비교로 확인).
  - snapshot payload v19: 기존 disk continuation cache는 재사용되지 않는다(배포 시 첫 요청은 cold).
- **테스트**: Flash-Next GPU CTest 전체(특히 `attention_ops`·`select_ops`·`mtp_ops`의 새 경계 검사, `hc_mix_ops`의 `CheckFusedInjectPartials`, `blaslt_ops`의 split 검사, `projection_ops`의 arena 검사), `hip_weight_upload_test`, `qwen38_flash_next_session_test`(새 `CheckUnalignedContinuationPasses`, `--logprobs-only`), `snapshot_test`(새 11행 in-pass tail 검사), functional `cache`·`long-context`·`conversation`·`batch`·`tool-type-unions`·`logprobs`·`responses-logprobs`·`openai-count-tokens`·cache_persist·`cache_alignment_qwen38.py`.
- **성능**: PP2048/8192, AR·MTP d0/d32K를 `runtime-hc-20261108` 대비 한 창. 특히 U2-2의 정렬로 생기는 추가 분할/캡처(nearby-stable warm turn의 aligned prompt tail snapshot, kv_pool 사용 시 in-pass 불가로 final pass 분할)와 tool-loop 재사용률(`pi_agent.py`)을 본다. U1-7의 SharedScope는 정상 경로 동작이 같아 decode 변화가 없어야 한다.
- **열린 문제**: `CheckBatchedSessions`/`CheckLookupBatch` snapshot 꼬리 차이는 U2-1과 원인이 같지 않다(U2-1 수정은 이미 있었다). U5에서 bisect한다.
- **남은 low 항목(리뷰어)**: 복원된 비정렬 complete prompt에서 토큰 없이 commit한 live frontier는 정렬 규칙 밖에서 이어질 수 있다(upstream과 같음, 드묾). logprobs 요청 하나가 batch decode 동료를 늦춘다(동기 row download + CPU log-softmax, opt-in).

## U5 통합 GPU 패스 (한 번)

1. 서버 정지(사용자 확인 후). HC 보정 세션의 GPU 작업과 겹치지 않게 한다.
2. Flash-Next GPU CTest 전체, `qwen38_flash_next_model_tests`(session·snapshot·adapter), `tests/functional`의 `conversation`·`batch`·`long-context`·`cache`·tool-schema 관련 suite.
3. prefill logits sha 비교(B10 `d051e822…`). 바뀌면 품질 검사.
4. B10 대비 성능 한 창(PP2048/8192, AR·MTP d0/d32K). 5%/10 ms 판정.
5. 통과하면 새 runtime 배포(B10은 `_backup`으로), `llm status`, 텍스트·이미지 smoke.
6. KNOWHOW·BENCHMARKS·HANDOFF·이 문서 갱신, 커밋.

## 하지 않는 것

- **extension 구조 이전**(약 40건), docs 재편, `src/models/qwen/hip` → `src/families/qwen/kernels` 이동: 구조 변경이고 얻는 것이 없다.
- **API-key 인증 제거** 9bd24d26 (#957): 보안 후퇴. health·alias만 필요하면 따로 본다.
- **GGUF 전용 변경**: #890, #788, #950, #884, #1041, UD 3.1 weights, 27B K-quant GEMV·GEMM 계열.
- **16K prefill pass** #933: 우리도 기각했고(KNOWHOW), upstream도 #954에서 되돌렸다.
- **#882 MTP chain 종료**: 이미 `mtp_policy.hpp` `MtpConfidenceRule`로 있다.
- **YaRN 세션 배율 #905, 262K 초과 attention #866**: 256K를 넘게 서비스하지 않는다. #905는 KVarN 양자화 key와도 충돌한다.
- **legacy continuation cache 제거 #1014/#1026/#1003**: 우리 Flash-Next·27B는 이미 component cache를 쓴다.
- **multi-checkpoint prefill #963, 레이어 분할 prefill #1039/#1042/#1037/#1095**: 동시 접속이 실제로 필요해지면 다시 본다.
- **Kyojin 변환기 pin 올리기**: v1.8.6 → main 사이에 `exllamav3/conversion`, `convert.py` 변경이 0건이다. 올리면 출처 기록, ext 재빌드, c03 provenance가 끊기는 비용만 생긴다. 변환 코드가 바뀌면 다시 본다.
- **Kyojin RDNA dense GEMV, shared-expert gate fusion, 단일 행 block graph**: 우리가 이미 더 빠르거나 이미 하고 있거나 해당하지 않는다.

## 검토 요약 (2026-11-08)

- upstream: 139개 커밋, 우리가 가져온 것은 0건(`git cherry`). 커밋별 단독 cherry-pick 충돌 목록과 merge-tree 결과는 세션 scratchpad에만 있다(재현: `git merge-tree --write-tree HEAD upstream/main`).
- 참고 기준선: upstream GGUF UD-Q4_K_XL PP8192 1,987 tok/s(#954, ROCm 10.1). 우리 EXL3 c03은 1,549 tok/s다. 형식이 달라 직접 비교는 아니다.
- Kyojin 공개 수치(`doc/benchmarks.md`, v1.8.10, 11/5): prefill 8K 1,576 tok/s, 추측 디코드 prose/chat/code 47.5/52.8/73.2 tok/s. 약 4bpw 팩을 HTTP와 실제 문서 프롬프트로 잰 값이라 우리 bench와 조건이 다르다.
