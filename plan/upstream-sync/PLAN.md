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

U4는 아래 "U4 진행 상황"대로 구현했다(측정은 U5).

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

- **우리 캐시 기능 재사용 비교 (먼저 본다)**: U2-2(정렬)와 U3-3(cache_persist)은 fork에서 바꾼 캐시 코드와 직접 맞물리고, CPU 테스트만 통과했다. 같은 workload를 `runtime-hc-20261108`과 후보에 돌려 기능별로 재사용 토큰 수, hit/miss, warm turn prefill 시간, 출력 일치를 비교한다. 재사용이 줄거나 출력이 다르면 배포하지 않고 원인을 고친다.
  - **블록 KV 공유(paged KV, copy-on-write prefix 공유)**: 같은 prefix의 두 세션이 블록을 공유하는지, 공유 블록 수가 줄지 않았는지. ShareCheckpoint slack 경로는 이제 정렬된 frontier에서만 공유하므로 공유 실패가 늘지 않는지 본다. K4V4 두 256K 문맥의 pool 사용량(4.62 GiB)도 그대로인지 본다.
  - **라이브 프런티어 저장(CaptureBeforeReset, WriteBackIdle)**: `CheckpointPurpose::kGenerated`로 발행돼 비정렬 위치에서도 복원되는지. 세션 리셋 뒤 다음 턴, 유휴 write-back 뒤 다음 턴이 전과 같은 길이를 재사용하는지 본다.
  - **RAM/디스크 continuation cache**: RAM hit, 디스크 hit(v19로 새로 쓴 것), 재시작 뒤 디스크 복원(이전 0.486 s 기록과 비교). v18 파일이 남아 있어도 오작동 없이 무시되고 LRU로 정리되는지 본다.
  - **`cache_persist=false`**: 그 요청의 prompt와 live frontier가 디스크에 안 써지는지(`Entry::live_persist`), RAM 재사용은 그대로인지 본다.
  - **에이전트 tool 루프**: `pi_agent.py` 재사용률과 턴별 prefill 토큰 수를 기준 런타임과 턴 단위로 비교한다.
- **bit가 바뀔 수 있는 것**
  - U2-2: 캐시 복원·용량 분할 뒤 prefill이 정렬 위치에서 시작하므로 restore 결과가 cold와 같아져야 한다(기대: 바뀜 → 일치). cold 한 pass prefill logits sha(B10 `d051e822…`)는 그대로여야 한다. 2 K 넘는 prompt의 여러 pass cold prefill은 pass 경계가 이미 4의 배수(2048)라 그대로여야 한다.
  - U2-3: `HcMixEpilogueF16Kernel`이 바뀜 → 96행 미만 prefill pass(짧은 꼬리, 복원 뒤 짧은 pass)의 inject partial 마지막 bit. 넓은 pass는 그대로.
  - U1-1: 경계 안 입력은 그대로(비교로 확인).
  - snapshot payload v19: 기존 disk continuation cache는 재사용되지 않는다(배포 시 첫 요청은 cold).
- **테스트**: Flash-Next GPU CTest 전체(특히 `attention_ops`·`select_ops`·`mtp_ops`의 새 경계 검사, `hc_mix_ops`의 `CheckFusedInjectPartials`, `blaslt_ops`의 split 검사, `projection_ops`의 arena 검사), `hip_weight_upload_test`, `qwen38_flash_next_session_test`(새 `CheckUnalignedContinuationPasses`, `--logprobs-only`), `snapshot_test`(새 11행 in-pass tail 검사), functional `cache`·`long-context`·`conversation`·`batch`·`tool-type-unions`·`logprobs`·`responses-logprobs`·`openai-count-tokens`·cache_persist·`cache_alignment_qwen38.py`.
- **성능**: PP2048/8192, AR·MTP d0/d32K를 `runtime-hc-20261108` 대비 한 창. 특히 U2-2의 정렬로 생기는 추가 분할/캡처(nearby-stable warm turn의 aligned prompt tail snapshot, kv_pool 사용 시 in-pass 불가로 final pass 분할)와 tool-loop 재사용률(`pi_agent.py`)을 본다. U1-7의 SharedScope는 정상 경로 동작이 같아 decode 변화가 없어야 한다.
- **열린 문제**: `CheckBatchedSessions`/`CheckLookupBatch` snapshot 꼬리 차이는 U2-1과 원인이 같지 않다(U2-1 수정은 이미 있었다). U5에서 bisect한다.
- **U4 항목 (U1~U3 확인 뒤 같은 창에서)**. 성공한 최적화는 스위치 없이 기본값으로 둔다. 실패하면 되돌린다(U4-4만 opt-in).
  - **U4-1 fused GDN prefill** (`c5657c82`)
    - 정확성(먼저): `qwen38_flash_next_gdn_ops_test`(새 one-launch 대 분리 경로 byte 비교, n=1..99, checkpoint), `qwen38_flash_next.gdn_stable_route_equality`, `exl3_gdn_prefill`, `exl3_prefill_overlap`. `qwen38_flash_next_gdn_prefill_bench`의 출력·state·checkpoint hash가 기준 binary(`runtime-hc-20261108`) 빌드와 같아야 한다. prefill logits sha가 B10 `d051e822…`와 같아야 한다. 하나라도 다르면 채택하지 않는다(무손실 항목이다).
    - 연산자: 같은 bench로 2048-token chunk의 µs/call 기준 대 후보(교대, 5회 이상). `check-kernel-resources`와 `tools/prof/isa_mix.py`로 `GdnPrefillKernel`의 VGPR·scratch를 본다(48 block / 20 WGP 불균형, epilogue z load가 window 경로에 있음 → 느리면 z를 window 앞서 load하는 수정을 먼저 시도).
    - 전체: PP2048/8192 기준 대비 한 창. 채택 기준: 출력 동일 + PP8192가 잡음 밖으로 빠르거나(upstream +1.7%) 최소한 느려지지 않을 것(5%/10 ms 규칙, 요청·phase별). 느리면 되돌린다.
  - **U4-3 wide copy verification** (`5dd394d8`)
    - 정확성: `qwen38_flash_next.exl3_wide_verification`(logits·hidden·state·continuation이 8행 연속 검증과 bit 동일). 실모델(K4V4 운영 설정과 FP16 둘 다): 같은 rewrite prompt를 greedy로 wide 켠 세션과 `LimitCopyRows(8)` 세션에서 decode해 token 동일. 64행 copy가 128-token KVarN group 끝을 segment 경계에서 지나는 경우를 반드시 포함(segment 사이 양자화는 upstream에 없는 코드). PLE·dense→sparse 전환(2048 근처)·logprobs(>8행)·seed 없는 sampled long copy도 확인. session test 전체(`CheckLookupBatch`·`CheckBatchedSessions`의 serial 대 batch 비교가 그대로인지), snapshot test.
    - 메모리: 첫 wide pass 뒤 할당 428 MiB, `DeferredScratchBytes`가 예약하는지, 두 256K K4V4 세션 설정에서 시작 시 width가 64로 남는지(`LimitCopyRows` 축소가 일어나면 로그로 확인).
    - 성능: `pi_agent.py` 코드 재작성/편집 workload와 functional rewrite(파일 다시 쓰기, rename, docstring)에서 decode tok/s와 cycle 수(`lookup_n`/`lookup_n_accepted`), 기준 대비. prose/code/chat MTP d0/d32K는 그대로여야 한다(5%/10 ms). 채택 기준: 출력 동일 + rewrite workload decode가 빨라지고 다른 workload가 느려지지 않음.
  - **U4-4 typical acceptance** (`da021c03`, opt-in이라 기본값은 바꾸지 않는다)
    - 꺼짐: greedy·seeded sampled 출력 hash와 MTP work 수가 기준과 같아야 한다(`typical_acceptance` 생략, false 둘 다).
    - 켜짐 성능: sampled MTP(T 0.7 top-k 20 thinking-off preset, T 1.0 thinking-on preset) prose/code/chat·Pi 에이전트 turn에서 decode tok/s와 수락률, 기준(꺼짐) 대비. Kyojin 보고 +~7%.
    - 켜짐 품질 게이트(분포를 바꾸므로 통과해야만 문서에서 권장): functional `tool-reasoning`·tool-schema suite가 sampled 설정에서 꺼짐과 같은 통과율(tool call 형식·JSON 유효성 실패가 늘지 않음), 작은 정답형 세트(GSM8K식 수학·코드 단위 테스트, seed 여러 개)의 정답률이 꺼짐 대비 잡음 안, 출력 길이·반복률(n-gram 반복) 비교. 하나라도 나빠지면 opt-in으로만 두고 문서에 결과를 적는다.
  - **U4-5 HC combine fusion screen** (`c776e86d`, production 아님)
    - `qwen38_flash_next_hc_mix_fused_bench`를 한 번 돌린다(기본 48 mixer, 16 copy). 결과 열 `bytes`가 DIFFER면 후보는 이 tree의 산술과 다르므로 이식하려면 kernel을 다시 맞춰야 한다. 채택 검토 조건: byte 동일이고 1/2/4/8 token 모두에서 production 대비 µs/mixer가 회전 launch 비용(launch + gap 약 2 µs, production epilogue 안 회전 비용만큼은 덜 엄격)을 빼고도 빠를 것. 아니면 KNOWHOW에 기각으로 기록하고 끝낸다(B3/B4 tail 융합 손해 기록과 같은 결론인지 적는다). 빠르더라도 production 연결은 별도 작업(EXL3 slab 회전 포함, decode AR/MTP d0/d32K 창 필요).
- **남은 low 항목(리뷰어)**: 복원된 비정렬 complete prompt에서 토큰 없이 commit한 live frontier는 정렬 규칙 밖에서 이어질 수 있다(upstream과 같음, 드묾). logprobs 요청 하나가 batch decode 동료를 늦춘다(동기 row download + CPU log-softmax, opt-in).

## U4 진행 상황 (2026-11-09, GPU 실행 없음)

U1~U3과 같은 방식(구현 → `build/b1` 빌드 → 관련 CPU 테스트 → 별도 리뷰어 검토 → 수정 → 재검토)으로 했다. 측정은 모두 U5로 미룬다. device code 비교 기준은 `build/b1/u/final-gufo`(`84fd6206`)다.

| 항목 | 출처 | 커밋 | 상태 | 비고 |
|---|---|---|---|---|
| U4-3 | #961, #993 | `5dd394d8` | 되돌림 (U5 GPU hang) | 손 이식. 혼자 decode하는 세션(occupancy 1, defer_head 아님)의 copy-only cycle이 greedy 또는 seed 없는 compact sampled일 때 최대 64행을 `Executor::ForwardWide` 한 pass로 검증. projection·routed expert·head는 decode batch 경로(EXL3 9~64행 경로는 8세션×8행 batch가 이미 쓰는 것)로 한 번, GDN·attention·PLE는 8행 segment로 나눠 연속 검증과 같은 산술. 포크 차이: GDN 검증이 deferred record 방식이라 upstream의 segment별 full-state 스냅샷 대신 executor 소유 `wide_state_`(layer당 state 하나)에서 이후 segment를 시작하고, rollback은 record를 live state에 replay(`RollbackWide`). 메모리: segment당 45.7 MiB + 108 MiB, 64행에서 428 MiB(upstream 1.9 GiB), 처음 쓸 때 할당하고 `DeferredScratchBytes`에 예약. KVarN: segment 끝이 128-token group 끝이면 다음 segment 전에 그 layer의 group을 양자화(upstream에 없는 코드). 리뷰 수정: `max_speculative != 8`이면 wide 비활성(HIGH, draft 7 미만에서 decode 실패), 실패한 pass도 `wide_owner_`가 record를 소유. host만 바뀜(7,059/7,059 동일). GPU 테스트 `qwen38_flash_next.exl3_wide_verification`(합성 EXL3: GDN+sparse attention+MTP, 9/17/20/64행, keep 1..n, next-forward commit)은 빌드만 |
| U4-1 | #895 | `c5657c82` | 되돌림 (U5, `4f49f025`) | EXL3 prefill GDN(`fused_conv`, F32 출력)을 layer당 launch 하나(`GdnPrefillKernel`)로: head당 block 하나(8 wave, 128행), q|k/v 채널 convolution, gate(GdnPrepAb 식), row-split recurrence, LDS의 raw 행으로 epilogue. 없어지는 launch: GdnPrepAb, GdnRowSplit<true>, GdnEpilogue(raw 행 F32 쓰기·읽기). 대상 출력 byte 동일(각 식을 분리 kernel과 같게 씀). 비교: 우리 `GdnRowSplitKernel<true>`는 grid (2, 48) 4 wave였다. 새 grid는 48 block이라 20 WGP에서 3 block을 받는 WGP가 생긴다(upstream도 같은 지적). device code: `GdnPrefillKernel` 추가만, 기존 7,059 symbol 동일(fallback으로 남은 `GdnRowSplitKernel<true>`도 그대로). 리뷰 수정: 피연산자 load 위치를 분리 kernel과 같게(fast-math 재결합 방지), epilogue의 z·norm load를 store 앞으로, `__restrict__`, n=0 guard, 새 경로 대 분리 경로 byte 비교 테스트(`gdn_ops_test`, n=1..99, checkpoint) |
| U4-2 | Kyojin `0e03454`, `656dab5` | — | 하지 않음 | 이전 기각 이유가 그대로다. (1) 손실: F16 Q/K/T/P WMMA의 layer 오차가 FP32 recurrence의 약 500배(2.6e-4 대 5.1e-7), 299-token tool prompt에서 greedy literal echo가 바뀌었고 chat NLL +0.0145(EXPERIMENTS "Chunked DeltaNet removed", "Chunked GDN prefill"). Kyojin 16행판도 F16 연산이고 KLD 0.006을 스스로 보고한다. (2) 구조: chunk 산술은 pass가 어디서 나뉘는지에 따라 달라진다. U2-2가 막 세운 계약(캐시 복원·용량 분할 뒤 prefill = cold와 bit 동일, 4 정렬)과 pass 안 checkpoint(임의의 4 정렬 위치의 state)를 16행 chunk는 지킬 수 없다. (2)는 품질 게이트로 풀리는 문제가 아니라 bench 후보도 만들지 않았다. U4-1(무손실)만 U5에서 잰다 |
| U4-4 | Kyojin `d8dbb75` | `da021c03` | opt-in 유지 (U5) | opt-in, 이름 하나: 요청 `typical_acceptance`(bool, null은 기본값 상속, 그 외 400), 서버 기본값 `--typical-acceptance`(공유 sampling 옵션, model preset이 유지). random sampling 요청의 MTP draft·lookup copy가 정확한 p/q 검사에서 거절돼도 target 분포(필터·penalty·temperature 뒤)의 p가 p > 0이고 p ≥ min(0.09, 0.3·exp(−H))면 유지(Medusa 규칙, Kyojin 상수). 끄면 token·RNG 소비가 bit 단위로 같고, greedy는 이 경로를 지나지 않는다. trace·시작 로그에 기록. CPU 테스트: mtp_sampling(임계값 양쪽, RNG), openai_chat(요청·기본값·preset·400). host만 |
| U4-5 | #894 | `c776e86d` | 기각 (U5 screen) | production에 넣지 않음. `qwen38_flash_next_hc_mix_fused_bench`(`tools/qwen38_flash_next/bench/experiments/hc_mix_fused_bench.hip`): upstream `HcMixQ8Fused`(work-list 한 launch, phase counter 대기) 복사본 대 우리 production decode mixer(HcCombineQ8_1 → HcDownSiluQ8_1 → Q8_0 up GEMV → HcMixEpilogue), graph replay, 1/2/4/8 token, byte 비교. 설계상 불리한 점: (a) B3 G1·B4에서 last-block tail 융합은 직렬 tail이 gap보다 비싸 되돌렸다(KNOWHOW B3/B4). 이 후보는 block 간 대기를 phase마다 둔다. (b) 우리 chain은 이미 B4 H3로 down+SiLU가 한 launch라 없앨 gap이 upstream(5 launch)보다 적다. (c) production EXL3 epilogue는 `mixed`를 다음 projection slab으로 회전한다(Exl3InputSlabs). 후보엔 없어서 채택하려면 그 회전 launch가 다시 생기거나 후보에 넣어야 한다. bench는 둘 다 회전을 빼서 후보 수치가 상한이다. 리뷰 수정: 후보의 1-token down 반올림을 이 tree(폭과 무관한 한 가지 반올림)에 맞춤(안 하면 n=1이 항상 DIFFER), xn/lo Q8_1도 비교, 최대 차이 출력, inject 입력을 0이 아닌 값으로 |

반복 중 CPU 테스트: prompt_lookup, flash_next_serving, text_model_runner, text_generation_scheduler, openai_chat, http_server, sampling, qwen38_flash_next.config·continuation_layout·mtp_sampling·draft_vocab·exl3_prefill_scratch 통과. GPU 테스트 target(`qwen38_flash_next_exl3_ops_test`의 `--wide-verification`, `gdn_ops_test`, `gdn_prefill_bench`, session/snapshot test, `hc_mix_fused_bench`)은 빌드만 했다. U4 전체 device code: `GdnPrefillKernel` 추가 외 7,059 symbol 동일.

## U5 통합 GPU 패스 (한 번)

1. 서버 정지(사용자 확인 후). HC 보정 세션의 GPU 작업과 겹치지 않게 한다.
2. 우리 캐시 기능 재사용 비교(위 "U5가 GPU에서 확인할 것" 첫 항목). 여기서 문제가 나오면 나머지 단계 전에 고친다.
3. Flash-Next GPU CTest 전체, `qwen38_flash_next_model_tests`(session·snapshot·adapter), `tests/functional`의 `conversation`·`batch`·`long-context`·`cache`·tool-schema 관련 suite.
4. prefill logits sha 비교(B10 `d051e822…`). 바뀌면 품질 검사.
5. `runtime-hc-20261108` 대비 성능 한 창(PP2048/8192, AR·MTP d0/d32K). 5%/10 ms 판정. U4 항목은 위 "U4 항목" 목록의 연산자·전체·품질 검사를 같은 창에서 한다.
6. 통과하면 새 runtime 배포(`runtime-hc-20261108`은 `_backup`으로), `llm status`, 텍스트·이미지 smoke.
7. KNOWHOW·BENCHMARKS·HANDOFF·이 문서 갱신, 커밋.

## U5 결과 (2026-11-09, 배포하지 않음)

후보 `0c784b00` 빌드(`cand-gufo` sha `8f700bdf…`) 대 배포 중인 `runtime-hc-20261108`(`166953f0`), c03 + all200 HC Q8, 배포 argv(FP16 KV, `--sessions 2`, MTP 7). 증거는 `build/b1/u5-20261109/`(chain4.log, chain5.log, perf.json, serve/, pisame/, func/timing-fr.json, correct/). 수치 전체는 [GFX1151-KNOWHOW](../../docs/GFX1151-KNOWHOW.md) "Upstream U5 combined GPU pass"에 있다.

**결론: 배포하지 않았다.** 서버는 `runtime-hc-20261108`(sha `166953f0`) 그대로다. 막은 것 두 가지:
1. U4-3 wide 검증이 served K4V4 rewrite에서 GPU hang을 냈다(아래). U4-3은 되돌렸다.
2. U2-2 정렬 뒤, RAM 캐시가 찬 상태에서 같은 요청을 다시 보내면 complete prompt 대신 정렬 경계(18→16, 18→8, 322→312)에서 복원해 2~10 token을 다시 prefill한다. TTFT +75~150 ms, 두 쌍에서 재현(functional `batch`). 원인: warm chat turn이 정렬된 prompt tail을 하나 더 capture해 turn당 RAM checkpoint가 두 배가 되고, retention이 rank 0인 complete-prompt(kRetry) 기록부터 내보낸다. 고치는 방법마다 비용을 다른 쪽으로 옮긴다. (A) chat turn에서는 정렬 tail을 잡지 않는다: retry 비용은 없어지지만, live frontier가 갈라진 다음 turn(thinking을 지운 history)이 최대 64 token 앞의 stable 경계에서 복원한다. (B) Acquire가 고른 checkpoint를 CaptureBeforeReset 동안 보호한다: 이미 기준에도 있는 race(`restore_failed ... unknown retained checkpoint`)를 막지만 압력 자체는 남는다. 어느 쪽으로 갈지는 사용자가 정한다.

| 항목 | 결과 |
|---|---|
| 정확성 | GPU CTest, snapshot, continuation adapter, session `--logprobs-only` 통과. session 전체는 알려진 `CheckBatchedSessions` 꼬리 차이로만 멈춘다. 그 검사를 보고로 바꾼 진단 빌드에서는 나머지 section(lookup batch, draft vocab, sampled head, EOS, frontier/fresh_load, serving sampling) 전부 통과 |
| prefill logits | sidecar `df1b3e0c…`(배포와 같음), sidecar 없음 `d051e822…`(B10과 같음). bit가 그대로라 품질 검사 불필요 |
| functional | 최종 후보에서 conversation·batch·long-context·tool-type-unions·logprobs·responses-logprobs·openai-count-tokens·tool-schema-edges 통과. cache-growth·cache-shared-prefix·cache(+persist)·`cache_alignment_qwen38.py`는 U4-1이 있던 이전 빌드에서 통과(호스트 캐시 코드는 같다). tool-reasoning은 기준과 후보가 같은 곳(`literal_protocol_raw_xml`, greedy 거부)에서 실패: 기존 모델 동작 |
| 성능 (5%/10 ms, B C 교대) | PP2048 −0.04/−1.2%, PP8192 +0.6/+0.5%, AR d0·d32K ±0.4% 안, MTP d0 +2.2/−0.3/+0.4%(work 같음), MTP d32K +5.7/+5.2/+9.7%(U4-3 덕분, 10 대 17 cycle, hash 같음). 이번 창의 MTP 절대값은 모든 binary에서 B10보다 약 30% 낮았다(같은 argv·work, AR·PP는 같음): 환경 원인, 설명 못 함 |
| U4-1 | 되돌림(`4f49f025`), byte 동일이 아님 |
| U4-3 | **되돌림.** (2026-11-09 갱신: 사용자 결정으로 다시 들였다(`21c138cc`). hang은 처음에 단편화된 메모리 창 탓으로 잠정했지만 U6에서 정리된 host에서도 재현했다. E1~E4는 미뤘다. U6 결과는 아래 "U6 통합 GPU 패스"에 있다.) 측정은 좋았다: greedy rewrite 24/24가 u13과 token 동일(FP16·K4V4 두 번씩), decode −12~−35%, Pi turn decode 최대 −22%. 그러나 두 번째 K4V4 실행에서 `docstring-admission` greedy decode가 114 token 뒤 약 6분 멈췄다(마지막 진행 05:00:03, 종료 신호 05:06:11 UTC)(GPU 100%, `ForwardBatchImpl(wide=true)`의 `hipStreamSynchronize`, `serve/hang-k4v4-gdb.txt`). 같은 요청이 첫 실행에서는 통과했다. 원인은 못 찾았다(정적 조사와 GPU 실험 계획은 아래 "U4-3 hang 조사", U4-3R). 다시 들이려면 한 세션 64행 pass의 반복 stress 실행이 먼저 필요하다. 후보 전용 cold prefill 지연(+73~150 ms, 36개 중 약 8개, u13은 2개)도 판단 보류로 남았다 |
| U4-4 | opt-in 유지, 기본값은 그대로. 꺼짐(생략·false) 36/36이 u13과 text·draft·accept·round까지 같음. 켜짐: sampled 18/18 요청이 빨라짐 +2.5~+27.6%(chat 중앙값 약 +21%, code 약 +17%, prose +8~+26%). 품질: 산수 30/30 둘 다, 평균 길이 147→162.7(중앙값 141→143.5), 4-gram 반복 0.0070→0.0084(최대 0.040→0.034), tool-schema-edges 통과, tool-reasoning은 꺼짐과 같은 greedy 검사에서 실패. 기본값은 사용자가 정한다 |
| U4-5 | 기각(현재 형태). bench 버그를 먼저 고쳤다(`c05c8e49`). byte 동일, mixer당 production/fused 45.88/41.23, 46.87/42.99, 49.27/46.35, 54.27/52.99 µs. 후보에 없는 slab 회전 launch(약 2 µs)를 빼면 +2.65/+1.88/+0.92/−0.72 µs라 8 token에서 진다. 회전을 넣어도 TG 1% 미만이 상한. B3/B4 tail 융합 손해와 같은 결론 |
| Pi (같은 경로) | 5 case 모두 통과. 첫 8 turn은 출력·재사용·prefill token이 같다. 9번째 turn(같은 비정렬 live frontier 2186에서 74-token pass)부터 greedy 출력이 갈린다: U2-3의 짧은 pass HC inject 순서 차이로 보이며 둘 다 유효한 출력이다 |
| CheckBatchedSessions | 기준에도 있는 실제 상태 차이(padding 아님). 진단은 "baseline already wrong" 참고 |

### baseline already wrong

기준 출력을 정답으로 두지 않고 독립 기준(같은 binary의 cold 실행, cache_alignment 검사, 강제 경로 비교)으로 판정했다. (a) 기준이 틀렸고 후보가 고침, (b) 후보 회귀, (c) 둘 다 틀림, (d) 계산만 다르고 둘 다 유효.
- **(a) warm 응답 ≠ cold 응답 (A2·A4 캐시 workload).** 기준은 warm 3 turn 중 1개만 같은 binary의 cold와 같았고, 후보는 3/3 같았다. 96행 미만 warm pass의 HC inject 덧셈 순서가 경로에 따라 달라서다(U2-3이 고침).
- **(a) 비정렬 위치에서 이어 쓴 prefill.** 기준은 4의 배수가 아닌 위치(D2 디스크 복원 7231, kvshare 등)에서 checkpoint를 복원하고 그 뒤를 prefill해 cold와 bit가 다르다. `cache_alignment_qwen38.py`가 기준에서 실패하고 후보에서 통과한다. 후보는 7204(정렬)에서 복원해 27 token을 더 prefill한다(+23~27 ms, 감수하는 비용). 재실행의 D2-cold 대조에서는 기준의 D2 출력(48 token, greedy)도 cold와 같았다. 계약상 비정확이지만 이 요청에서는 출력이 바뀌지 않았다. 재실행에서 후보 warm은 4/4(A2·A3·A4·D2)가 cold와 같았고, 기준은 A2·A4가 달랐다.
- **(a) 경로마다 다른 prefill bit (long-context).** `long_schema_cold`·`long_schema_cached`·`long_json_stop`의 출력이 기준과 후보 사이에 다르다(cold 포함). 긴 prompt의 짧은 마지막 pass에서 U2-3 이전 기준은 같은 행을 넓은 pass와 다르게 계산했다. 후보는 경로와 무관하다(`hc_mix_ops` `CheckFusedInjectPartials` 통과). 두 출력 모두 functional 검사를 통과한다.
- **(d) Pi 9번째 turn부터 갈린 greedy 출력.** 같은 비정렬 live frontier에서 이어지는 74-token pass. live 이어쓰기에는 cold 동일 계약이 없어 어느 쪽도 "틀렸다"고 할 수 없다.
- **(b) 아님: exact retry 재사용.** kvshare `retained-after-cold` 4867 대 4860(재실행에서는 둘 다 4867, evict에 따라 갈림)과 batch retry 18→16 등은 기준이 맞았다(complete-prompt 상태에서 prefill 없이 decode하므로 cold와 같다). 후보의 재사용 손실은 위 "막은 것 2"의 회귀다.
- **(c) CheckBatchedSessions / CheckLookupBatch (기준과 후보 같음).** 실제 상태 차이다. serial과 batch snapshot이 마지막 약 0.4%(draft K/V 행, 쓰이지 않는 draft residual)에서 2~124 B(한 번 22,102 B) 다르다. 두 snapshot을 각각 복원해 greedy MTP로 decode하면 token·logits·acceptance가 같다(9/9). 진단 빌드에서 serial과 batch 모두 전체 MTP 경로를 쓰게 하면(`GUFO_DIAG_MTP_FULL`, 커밋 안 함) byte가 같아진다. 원인은 `MtpForwardBatch`가 cohort 전체가 kv_only일 때만 kv_only 경로를 타는 것이다(`executor_batch.cpp:276`). greedy는 영향이 없고, sampled 제안은 같은 batch에 누가 있는지에 따라 달라질 수 있다. 수정 방향: 따라잡는 세션을 따로 kv_only 하위 batch로 돌린다([CLEANUP.md](CLEANUP.md) C1).
- **(c) tool-reasoning `literal_protocol_raw_xml`.** 기준·후보·typical 켜짐 모두 같은 greedy 거부 문장으로 실패한다. 모델(c03) 동작이고 코드 회귀가 아니다.
- **기준에도 있는 race.** Acquire가 checkpoint를 고른 뒤 같은 요청의 `CaptureBeforeReset` admission이 그 checkpoint를 내보내 `restore_failed ... unknown retained checkpoint`가 난다(기준 로그에도 있음). 후보에서는 정렬 tail capture 때문에 더 자주 난다.

### U5가 끝내지 못한 것
- 요청 취소 케이스 decode 지연: conversation·batch 재실행에서는 방향이 쌍마다 뒤바뀌어(scheduling) 후보 탓이 아니다. 처음 본 219→327 ms(cache suite thinking-tool-cancel)는 다시 돌리지 않아 판단 보류.
- 이번 창의 MTP 절대 속도 저하(모든 binary 약 −30%)와 두 binary 모두에서 한 번씩 본 약 30초 decode 멈춤: 환경 원인으로 보이나 확인 못 함.
- U4-3 hang의 원인과 후보 전용 prefill 지연. hang은 아래 "U4-3 hang 조사"(U4-3R)에서 정적으로 좁혔고, 원인 확정은 GPU 실험으로 넘겼다.

### U4-3 hang 조사 (U4-3R, 2026-11-09, 정적 감사만, GPU 실행 없음)

**결론: 정적으로는 원인을 찾지 못했다. U4-3은 다시 들이지 않았다.** (2026-11-09 갱신: 사용자 결정으로 다시 들였다(`21c138cc`). hang은 처음에 단편화된 메모리 창 탓으로 잠정했지만 U6에서 정리된 host에서도 재현했다. E1~E4는 미뤘다. U6 결과는 아래 "U6 통합 GPU 패스"에 있다.)  아래 GPU 실험으로 가설을 가린다.

로그에서 나온 것:
- **이 요청의 입력에 결정적으로 달린 버그가 아니다.** 첫 실행(`serve/cand-rewrite-k4v4`)과 hang 난 두 번째 실행(`serve/cand-rewrite2-k4v4`)은 binary(`cand-gufo`, `8f700bdf`), argv, 요청 순서, 메모리 예산(`deferred_scratch_budget_bytes` 866,463,744 → 첫 wide pass 뒤 354,443,264, 두 실행 같음)이 같고, 캐시 디렉터리는 실행마다 지웠으며(`stage3_serve.sh`), 모든 요청이 cache=miss라 hang 난 요청도 cold prefill(279 token)에서 시작했다. 그 요청(`docstring-admission` greedy)은 두 실행에서 57 token(51/53 accepted/proposed)과 114 token(96/130) 진행 기록까지 draft 수가 같다. 첫 실행은 같은 prompt·token·pass 폭을 통과했다(다음 기록 162 token, 끝 250). 그러므로 이 요청의 행 수, group·ring 경계, 위치 문턱, 정렬 안 된 시작만으로 정해지는 조건은 원인이 아니다. 다만 두 실행이 완전히 같지는 않다. 앞선 sampled 요청이 갈렸고(r7 1091 대 1099 token, 그래서 slot 2 frontier 2159 대 2167, rename sampled 44 대 45 round, `session_graph_count` 24 대 23), r12 admission에서 첫 실행만 `live_frontier_written_back slot=1`, retention 두 건, `published tokens=511`을 더 남겼다. 두 번째 실행은 272-token checkpoint를 멈추기 약 1초 전에 publish했고, 첫 실행은 114 token 뒤에 했다. 남은 차이는 다른 slot의 상태, graph cache, 같은 시각의 캐시 capture·publish 복사다.
- 진행 기록은 생성 token이 50의 배수를 넘을 때만 남는다(`text_generation_scheduler.cpp`). 114 뒤 150 전에 pass가 여러 번 있었을 수 있어, 멈춘 pass는 위치 약 392~428에서 시작하고 끝은 약 492 이하다. KVarN group 경계(384, 512)는 이 범위에 없다.
- 커널 로그(journalctl -k, 13:58~14:10 KST)에 amdgpu 메시지가 없다: page fault, MES timeout, reset 모두 없다.
- gdb(`serve/hang-k4v4-gdb.txt`): decode thread만 `hipStreamSynchronize`에서 기다린다. cache idle-spill·persistence·n-gram thread는 모두 condvar 대기라 다른 stream에 걸린 host 작업이 없다.
- 같은 창에서 기준 binary도 약 30초 decode 멈춤을 한 번 냈고, 모든 binary의 MTP 절대 속도가 약 30% 낮았다. 배포 서버가 같은 GPU에서 돌고 있었다.

정적 감사(K4V4에서 `ForwardWide`/`ForwardBatchImpl(wide=true)`가 실행하는 것, narrow verify와 비교):
- **대기하는 kernel이 없다.** Flash-Next·EXL3·KVarN kernel 어디에도 spin-wait나 flag 대기가 없다. 다른 block을 기다리는 곳은 모두 "마지막에 도착한 block이 epilogue" 방식(EXL3 GEMV segment counter, `RouterTopKTilesKernel` arrivals)이라 counter가 틀려도 결과만 틀린다. 데이터에 달린 loop(attention tile 순회, `next_block`, `BucketExpert`, `BuildTiles`, mmq stream-K fixup, KVarN 양자화의 8회 반복)는 모두 유한하다.
- **shape.** wide segment는 8행이라 attention·GDN·PLE kernel의 shape은 narrow verify와 같다. 64행으로 도는 것은 projection·HC mix·MoE·head(decode batch 경로)이고, 8세션×8행 batch와 같은 shape이다. 다만 `--sessions 2` 서비스에서는 이전에 16행까지만 돌았다. 17~64행 decode batch(라우터 tile 융합은 25행까지라 26행부터 `BuildTiles` 경로)는 서비스에서 이번에 처음 돌았다.
- **workspace.** `wide_controls_`(8개, segment s<8), `batch_controls_`(8개), `mtp_.target_hidden`(64행), token·n-gram 행·mapped n-gram buffer·mask(max_batch행), attention partials/lists/counts(8행, segment마다 같은 stream에서 차례로)는 모두 범위 안이다. 새 할당(`wide_state_`, `wide_segments_`)은 읽기 전에 쓴다. controls·token pinned buffer는 pass 시작에 쓰고 pass 끝 동기화 전에는 다시 쓰지 않는다.
- **KVarN.** 64행 pass는 `PlanKvarnPass`(128행 이하)로 계획하고, segment 안의 group 끝은 `KvarnQuantizePassBoundary`(segment 위치를 device에서 읽는다), segment 끝이 group 끝이면 다음 segment 전에 그 layer의 group을 양자화한다. 64행은 ring 한도(129행) 안이고, rollback 뒤 양자화된 group을 다시 여는 경로도 ring이 그 행을 갖고 있다. 출력이 24/24 token 동일한 것과 맞는다.
- **이전 pass가 남긴 것.** `RollbackWide`·`KeepHiddenRow`는 동기화 없이 같은 stream에 쌓여 순서가 지켜진다. 넓은 copy 뒤 MTP kv_only catch-up(63행 이하)은 `Run`이 동기화하므로, 그것이 멈췄다면 거기서 멈췄을 것이다.

가설(가능성 순):
1. **H1, GPU queue·scheduler 정지(kernel 논리 아님).** 64행 pass 하나가 host 동기화 없이 dispatch 수천 개, pass 중간 D2D `hipMemcpyAsync` 약 300개(segment 1~7의 linear layer conv 시작 history 252개, segment 1의 layer당 3 MiB recurrent state 36개, PLE history), MoE layer마다 side stream fork/join 48번을 넣는다. fork/join은 EXL3 decode·narrow verify에서도 매번 한다(`exl3_decode_overlap`). wide만의 새 패턴은 pass 중간 copy와 긴 무동기 구간이다. narrow verify에는 pass 중간 copy가 없다. 다른 process(배포 서버)가 같은 GPU를 쓰는 상황에서, 기준에서도 본 30초 멈춤이 길게 간 것일 수 있다. 근거: 비결정적, 대기 kernel 없음, 커널 로그 없음, 기준의 30초 멈춤.
2. **H2, 캐시 capture·publish 복사가 wide pass와 겹침.** 두 실행에서 다른 것은 이것과 다른 slot의 상태뿐이다(위). 두 번째 실행은 멈추기 약 1초 전에 같은 session의 272-token checkpoint를 publish했다. 정적으로는 capture stream이 executor stream이나 scratch를 공유하는 곳을 찾지 못했다. 실험에서 멈춤 직전의 `[cache]` 로그를 함께 본다.
3. **H3, 26~64행 decode batch 경로의 드문 순서 문제.** 서비스에서 처음 쓴 행 수(EXL3 `BuildTiles` routed MLP, side stream shared MLP와 겹침)라 후보다. 같은 경로를 쓴 FP16 두 실행은 멈추지 않았고, 정적으로는 stream 사이 scratch 충돌을 찾지 못했다(shared MLP는 projection fast slab, routed는 routed_input·tiles만 쓴다).
4. **H4, device control·count가 깨져 loop 상한이 수십억이 됨.** 깨지는 경로를 찾지 못했다. 그런 위치라면 FP16 KV와 KVarN record 읽기가 범위를 벗어나 fault가 났을 텐데 커널 로그가 없다. 낮다.
5. **H5, KVarN 경계·ring 처리 오류.** 위 감사, 멈춘 범위에 경계가 없는 점, 같은 순서를 첫 실행이 통과한 점으로 사실상 배제한다.

GPU 실험(통합 패스에서, 다른 GPU 측정 창과 겹치지 않게). 스크립트는 `build/b1/u5-20261109/tools/u43_stress.sh`(`u43_replay.py`: 같은 12 요청, prompt 원본은 `0c784b00`에 고정, 서버 로그가 90초 멈추면 hang, 요청마다 서버 로그의 가장 긴 공백을 남기고 20초를 넘으면 표시(기준의 30초 멈춤 대조), `u43_diag.sh`: gdb·`gpu_busy_percent`·커널 로그·캐시 로그 끝, `ROCGDB`가 있으면 `info dispatches`/`info waves`, sudo가 되면 KFD hqd. umr wave 정지는 배포 서버까지 멈추므로 `U43_UMR=1`일 때만). binary는 멈췄던 `bin/cand-gufo` 그대로 쓴다. 결과는 `build/b1/u5-20261109/u43r/`.
- E1 `u43_stress.sh replay-k4v4 6`: 새 K4V4 서버 6개, 각각 12 요청 전체. 재현율을 잰다(hang은 두 번째 서버의 11번째 요청이었다).
- E2 `u43_stress.sh focus-k4v4 100`: 서버 하나에서 순서 한 번, 그 뒤 `docstring-admission` greedy 100번. 64행 pass를 가장 많이 돈다.
- E3 `u43_stress.sh replay-fp16 6`: KVarN이 필요한지.
- E4 `u43_stress.sh focus-u13 100`: U4-3이 없는 u13으로 같은 부하. 멈춤이 나오면 H1의 환경 쪽이다.
- 재현되면 `VARIANT=nosdma`(`HSA_ENABLE_SDMA=0`)와 `VARIANT=serial`(`AMD_SERIALIZE_KERNEL=3`, `AMD_SERIALIZE_COPY=3`, 멈추면 gdb의 decode frame이 멈춘 launch를 가리킨다)로 다시 돌린다. wave가 상주하면 kernel 문제(H3/H4), 없으면 queue 정지(H1)다. 같은 device의 D2D copy는 보통 blit kernel이라 `nosdma`는 H2D/D2H(캐시 복사, H2)만 바꾼다. 사용자가 허락하면 배포 서버를 멈춘 창에서 E1을 한 번 더 돌린다(H1의 다중 process 조건).
- greedy 행은 모두 `serve/cand-rewrite-<kv>/rows.json`의 sha와 비교해 남긴다.
- 2026-11-09: 사용자 결정으로 다시 들였다(`21c138cc`). E1~E4는 미뤘다(정리되거나 재부팅한 host에서 나중에 돌린다). U6 결과는 아래에 있다. 원래 조건: 원인을 확정해 고치거나, E1·E2에서 수천 번의 wide pass가 무재현이고 E4 대조가 깨끗할 때. 다시 들일 때는 `b9cbf5a1`을 되돌린다(`5dd394d8`, `0c784b00`, `faefb63e`).

## U6 통합 GPU 패스 (2026-11-09, 배포 없음)

사용자가 이번에는 배포하지 않기로 했다. 그래서 게이트 결과만 남긴다. 증거는 `build/b1/u6-20261109/`(chain*.log, perf/, serve/, func/, correct/, correct2/, cache/, hang/, cached.log)에 있다. 측정 수치는 [GFX1151-KNOWHOW](../../docs/GFX1151-KNOWHOW.md) "Upstream U6 combined GPU pass"에 있다.
binary는 셋이다:
- base: 배포 중인 `runtime-hc-20261108`(`166953f0`)
- cand: `22705f7b` 빌드(`6e6574ca`, U4-3 되돌림 상태)
- cand2: `21c138cc` 빌드(`6588c56c`). U4-3을 다시 들였고 M1이 들어 있다.

조건은 c03 + all200 HC Q8, 배포 argv(FP16 KV, pool 14.4 GB, `--sessions 2`, MTP 7)이고, 후보는 `--typical-acceptance off`로 맞췄다. `6415dab8`(supervisor prepare_memory)는 binary에 들어가지 않아 이 패스에서 다루지 않는다.

**환경.** 22일 동안 떠 있던 host의 메모리 단편화가 U5에서 본 "MTP 약 30% 저하"의 원인이었다. drop_caches와 compact_memory 뒤에 base MTP d0이 39→59 tok/s, d32K가 68→94~99 tok/s로 돌아왔다. work는 같았다(KNOWHOW). 그래서 timing은 정리한 뒤의 행만 쓴다.
- base 행은 재사용했다. bench는 16:51~17:04에 잰 Br3/Br4 행, serve는 17:33~17:59에 잰 bL1·bL2·bT 행, functional은 17:59~18:07에 잰 fb 행이다.
- cand2는 18:10에 다시 compaction한 뒤에 쟀다.
- 5%/10 ms를 넘는 행만 B/C 한 쌍을 다시 쟀다: serve cohort는 b2R 대 c2L2, functional batch·cache는 rpb 대 rpc2·rpc.
- 정확성 출력은 기록해 둔 hash와 비교했다: U5·U6 base, U5 u13의 rewrite sha, U5 fnn-cand의 logprobs.

| gate | 결과 |
|---|---|
| R1 retry | cand2 functional `batch`: retry 행의 prefill·cached token이 base와 같다. complete prompt를 복원했고, 정렬 경계에서 다시 prefill하지 않았다. cache stage(재시작 포함) 로그에 `restore_untracked` 0건, `restore_failed` 0건. 통과 |
| C1 | session test 전체 통과(cand·cand2 모두 rc 0): `batch independent_state_exact=1`, lookup C2/C4/C6/C8 `tokens_logits_RNG_exact=1`, `snapshots_exact=1`, serving replay 전부 exact. 섞인 cohort의 peer A decode는 base 대비 +0.8/+2.8%로 중립이다. 따라잡는 B 행은 draft 수가 base와 달라(설계상) +19/+13/+6/+15%와 −6/−9%로 섞였다. 판단 보류 |
| typical | cand2 startup 로그에 `typical_acceptance=1`, `/props` `typical_acceptance: true`. off일 때 sampled 18/18이 base와 text·work까지 같다. on(cand): 17/18 요청이 +4.3~+28.4% 빨라지고 1건 −1.3% |
| C2 | `--cache-disk-staging-bytes 268435456`을 주면 weight를 읽기 전에 오류로 거부된다(`--cache-disk-staging-bytes applies only to DeepSeek's disk cache...`). 이 옵션 없이 띄운 서버는 모두 정상 |
| C3 | startup 로그에 `event=device_code_identity sha256=84fdeb4f…`. 같은 build로 재시작한 뒤 disk restore가 된다: A4-disk 7348 token(restore 100 ms), D2-disk 7204 token, D1 shared prefix |
| C4 | 의미 있다. 혼자일 때 A의 capture는 7.4~9.9 ms다. B가 약 21K token을 cold prefill하는 동안에는 1829~1841 ms이고, A의 TTFT가 1.9 s에서 4.1~4.4 s로 늘어난다. base도 1406~1836 ms로 같아서 기존 결함이다. 이번 패스에서는 고치지 않았다(event 대기로 바꾸는 구조 변경이다) |
| C5 | Flash-Next continuation adapter test 통과(cand) |
| C6 F1·F2 | 27B adapter의 text section(guard, boundaries, shared pool, retained blocks, prefill cancel, batch admission) 통과. image section은 테스트가 `mmproj-BF16.gguf` 이름을 요구해서 실패했다(하드링크 디렉터리로 재실행 준비함). 27B KVarN K4V4 + DFlash2(cand2): `cache`(persist·disk 포함, 10개 case)와 `long-context` 모두 통과. 27B Flash-Next식 MTP는 HTTP에서 지원하지 않아, 27B의 추측 디코드는 DFlash2다. group 끝을 넘는 거부를 일부러 만든 case는 없다. 거부는 DFlash2 draft에서 자연스럽게 생긴다 |
| C6 S1·S2 | S2: Flash-Next cache·persist·pi 로그에 `capture_failure`/`capture_failed` 0건. S1(27B payload 크기): 27B 실행 결과로 판단 |
| C8a | logprobs 값이 u13과 같다(served sha 4/4). functional `logprobs`·`responses-logprobs` 출력은 U5의 C8a 이전 후보와 17/17, 18/18 동일. logprobs B가 있을 때 peer A의 decode는 u13 9426/8884 ms, cand2 9034/8777 ms로 draft 수가 달라 판단 보류. 63행 fan-out: wide logprobs long copy에서 thread 136→198, RSS 약 +95 MiB |
| C8d | 통과. cand2로 `cache`+`batch`를 14.4 GB FP16 pool(fpool2)과 KVarN K4V4(fk42)에서 돌렸다. `capture_failure`·`restore_untracked` 0건 |
| C9 | cache suite `thinking-tool-cancel`의 cancel decode 행은 base 219.2 ms, cand2 219.4 ms로 중립이다. U5의 219→327은 재현되지 않았다 |
| M1 | 통과. cache·persist suite 동안 `Cached`가 3.8 GiB 안팎으로 평평했다(서버를 띄울 때 model을 읽느라 잠깐 5.6~6.0 GiB). cache label 26/26(`cache_page_cache_test` 포함). 같은 상태에서 matched로 다시 잰 cand(M1 없음) 대 cand2(M1)의 disk restore·prefill 행: `cache_restore_ms` −8~+9%, prefill −13~+0.2%로 나빠지지 않았다 |
| 정확성 | CTest 42/42, cache label, snapshot, Flash-Next adapter, prefill logits `df1b3e0c…`(U5와 같다). 이것들은 cand에서 쟀고, U4-3·M1이 닿지 않는 경로라 cand2에서는 재사용했다. cand2: session test 전체, `exl3_wide_verification`·`mtp_kv_executor`·`verification_sampling`, cache label 통과. `mtp_audit`은 두 후보 모두 attention-state rms 0.0070984(허용치 0.002)로 실패한다. F1부터 알려진 EXL3 oracle 차이이고 base 대조는 하지 않았다 |
| 성능 (cand2 대 base, 정리한 뒤) | PP2048 +1.4%, PP8192 +0.6%, AR d0/d32K +0.9/+1.3%, MTP d0 +2.3/+3.0%(work 같음), MTP d32K +7.1/+12.3%(U4-3: 10 대 17 cycle, hash 같음). 느려진 행 없음. sampled 18/18 동일, 4건 −7~−17% |
| align·pi | `cache_alignment_qwen38.py` 통과(cand2). Pi 같은 경로 5/5 통과. 다만 출력은 U5와 비교할 수 없다(작업 경로 이름 u5→u6이 prompt에 들어간다) |
| batch 회귀(남음) | functional `batch`의 동시 short greedy pair·reply decode가 base 대비 +20~+137%(약 +30~200 ms) 늘었다. matched 두 쌍(rpb 대 rpc2, rpb 대 rpc)에서 방향이 같고 두 후보 모두에서 나온다. C1의 kv_only 하위 batch 분리가 의심된다. `*-disk` restore 뒤 prefill 행이 두 후보 모두에서 +10~15 ms 늘었다(M1 탓은 아니다). 원인은 아직 모른다 |
| U4-3 | **cand2 FP16에서 hang을 재현했다.** 아래 "U4-3 hang, U6" 참고. 배포 후보가 아니다 |

### U4-3 hang, U6

사용자 결정으로 2026-11-09에 다시 들였다(`21c138cc`). 처음에는 단편화된 메모리 창 탓으로 잠정했지만, cand2(`6588c56c`)에서 정리된 host, FP16 KV로 재현했다. 다른 GPU 사용자는 없었다. 따라서 메모리 단편화와 KVarN은 필요 조건이 아니다. E1~E4 replay는 미뤘다. 대신 `tools/hang_loop.sh`로 같은 서버에서 `rewrite.py --sampled`를 계속 돌렸다.

| 실행 | 조건 | 결과 |
|---|---|---|
| c2L | 일반, load·typical 뒤 rewrite | 18:32, r55 rename-cache_persist greedy(1066 token), 123 token에서 멈춤. gdb 실패(ptrace) |
| c2L2 | 같은 순서 | 2 round 무재현 |
| h1 | 일반, load·typical 뒤 | round 1, r52 docstring-admission greedy(279), 57 token. decode thread는 `ForwardBatchImpl` executor_batch.cpp:1391(wide=true)의 `hipStreamSynchronize`. kfd hqds·mqds가 10초 간격 두 번 byte 동일(정상일 때는 5초 안에 바뀐다). 5분 넘게 회복하지 않음 |
| h2s | `AMD_SERIALIZE_KERNEL=3`/`COPY=3` | round 4, r88 docstring-admission. `LaunchRouted`(RoutedGemv) launch의 serialize 대기, `Exl3Experts` rows=16(fused router tile) |
| h3a | `AMD_SERIALIZE_KERNEL=2`/`COPY=2`(각 kernel 뒤 대기) | round 9, r129(1066 token rename 다음 요청). RoutedGemv<…,8> gate/up(k 2560, n 640, assignments 640, BuildTiles, rows=64) 자체가 끝나지 않는다. amdgpu tracepoint로 본 것: 마지막 EOP interrupt가 멈춘 시각이고, 그 전 약 7분 동안 BO 이동·생성·map이 없었고, fault interrupt도 없었다. **185초 뒤, 다른 GPU process가 시작되자 약 13초 만에 decode가 정상 속도로 이어졌다** |
| h5p | 일반 argv, 멈춘 뒤 진단하고 일부러 두 번째 GPU process(`cand2-gufo diagnose`, 1초 안에 끝남)를 띄움 | round 11, request 125(prompt 1068). 129 token에서 약 122초 동안 멈췄다. poke 뒤 1초 안에 decode가 다시 진행됐다(554→984 token, 70~183 tok/s). **queue 집합이 바뀌면 풀린다는 것을 확인했다** |

판정:
- 결론은 kernel이 끝나지 않는 것이 아니라 queue 수준의 정지다. 근거:
  - dispatch가 하나만 남아 있어도(serialize after) 멈춘다.
  - HQD가 움직이지 않는다.
  - fault가 없다.
  - 다른 process가 queue 집합을 바꾸면 풀린다. c2L도 SIGTERM 약 14초 뒤에 요청이 끝났다.
- 가장 그럴듯한 기전은 MES/CWSR 선점이다. 이 host에는 kwin_wayland desktop이 같은 GPU에 떠 있다. wide pass의 큰 RoutedGemv dispatch(640×2 grid)가 선점된 뒤 queue 재배치가 있을 때까지 다시 schedule되지 않는다는 설명이다.
- 배제한 것:
  - EnsureWideSegments 지연 할당: 같은 서버의 앞 round에서 같은 요청이 통과했고, trace에 BO 활동이 없다.
  - 동시 stream 조건: serialize 상태에서도 멈춘다.
  - 64행 BuildTiles 경로에만 있는 문제: 16행 fused tile 경로에서도 멈춘다.
- 환경(`build/b1/u6-20261109/hang/environment.txt`): kernel 7.0.0-38, amdgpu `cwsr_enable=1`, `queue_preemption_timeout_ms=9000`, `sched_policy=0`, MES fw 0x86, MES_KIQ 0x6f, MEC 0x22.
- 배포 후보가 아니다. 완화 설계(kick/watchdog, 긴 dispatch 분할, cwsr 설정)는 coordinator가 맡는다.

### U6 남은 것

- batch 동시 short reply decode 회귀(C1 의심)와 `*-disk` prefill +10~15 ms
- C4: capture가 다른 세션의 prefill을 기다린다(base에도 있다)
- mtp_audit의 base 대조
- 27B adapter image section: 테스트가 `mmproj-BF16.gguf` 이름을 요구한다
- U4-3 hang 완화 빌드 검증(아래 "U4-3 완화 후보")

### U4-3 완화 후보 (cand3, cand4)

사용자가 workaround를 승인했다. 다른 agent가 코드를 만들었고, 이 패스에서는 GPU 검증만 했다.

- **cand3** = `a0d951dd`(gufo `f36e6fb8`). `21c138cc`에 세 커밋을 더했다.
  - `c897aa51`: 유한하지 않은 router 행을 weight 0으로 유효한 expert에 보낸다.
  - `8a18e528`: routed decode launch를 grid y 80으로 고정한 strided grid로 돈다. 64행 gate/up이 12,800에서 1,600 block으로 준다. 출력 bit는 같다.
  - `a0d951dd`: decode·wide scratch를 load할 때 잡는다.
- **cand4** = `7c3e300b`(gufo `9ac49a5f`). cand3에 세 커밋을 더했다.
  - grid cap을 160으로 올렸다.
  - `0e0ebad8`: 취소된 세션의 rollback scratch를 slot을 놓을 때 지우지 않고 유지한다. **2026-11-10 S3에서 되돌림**: 이후 짧은 prefill을 약 7 ms 느리게 묶었다. 이제 valid 세션만 유지하고, adapter test도 취소 시 해제를 기대하도록 다시 고쳤다(KNOWHOW "Upstream S3 batch regression").
  - `6e2c504e`: 다른 요청이 admission돼 있으면 wide 검증을 하지 않는다(batch 회귀 수정).

cand3 결과:
- **GPU test:** `exl3_ops_test --wide-verification`(WideDecodeAllocations 포함) 통과. session test 전체 통과(출력이 cand2와 같다). `exl3_moe_kernel_test`의 RouterTilesSet·RouterNonFinite·RoutedGridIdentity는 통과했지만, test fixture 버그(`cae407c4`에서 고침)로 RoutedWideReplay 다음에 중단됐다.
- **hang 반복:** FP16 일반 argv, poke 없이 24 round(rewrite 요청 288개)에서 멈추지 않았다. cand2는 같은 loop가 round 1~11 안에 멈췄다. greedy 출력이 u13과 144/144 같다. 요청별 decode 중앙값은 cand2 기준 round(c2L2) 대비 −1.2~−7.1%다. K4V4 loop는 10 round(요청 120개)에서 멈추지 않았고, greedy가 u13과 60/60 같았다. decode 중앙값은 cand2 c2K 대비 −3.0~+4.0%다.
- **startup:** `decode_scratch_reserved bytes=512020480`. KV pool(14.4 GB, 2050 block)과 admission 정상.
- **ISA:** RoutedGemv VGPR가 최대 +37(117→154), scratch spill 없음, LDS 같음. cand2 ISA는 21c138cc source를 HEAD header로 컴파일한 것이다.
- **bench(정리 뒤 1회):** MTP d0/d32K 57.85/101.72(cand2 60.68/105.73, −4.7/−3.8%, 5% 안). AR 32.78/32.52, PP 1513/1548. work·hash는 같다.

cand4 결과:
- **GPU test:** `exl3_moe_kernel_test` 전체 통과. 546 CASE를 출력했고, routed wide replay(tile 113), adversarial tiles, concentrated routing 9~25행이 들어 있다. `exl3_ops_test --wide-verification` 통과. session test 전체 통과(출력이 cand2와 같다). continuation adapter test는 rc 1로 실패했다(`failed slot retained rollback scratch`, test.cpp:279). 이 test는 취소된 slot을 Invalidate하면 rollback scratch를 해제해야 한다고 요구하는데, `0e0ebad8`은 일부러 그 scratch를 남긴다. test 계약을 고칠지, 변경을 되돌릴지 결정해야 한다. 그 행 전까지의 section은 모두 PASS다. 후속: test를 새 계약에 맞게 고쳤다. 취소 뒤 Invalidate가 성공하고, 논리 상태를 비우고, `AllocatedBytes()`는 취소 전과 같으며 MTP 모드에서는 빈 상태보다 크다. 다음 Sync/Decode는 새 세션과 logits·16 token이 같다. 그 test를 다시 빌드(`8a76234d`)해 돌렸고 통과했다(rc 0, `correct4/adapter-rerun/`).
- **hang 반복:** FP16 24 round와 K4V4 10 round(요청 408개)에서 멈추지 않았다. greedy가 u13과 144/144, 60/60 같다. decode 중앙값은 cand2 대비 FP16 −5.1~+2.1%, K4V4 −3.2~+4.9%다. U4-3의 단독 wide 속도 향상은 그대로 남아 있다.
- **MTP 쌍(정리 뒤, cand2→cand4 교대 3쌍):** d0 58.58→58.40, 59.47→58.67, 58.81→58.74. d32K 102.44→105.39, 105.91→104.40, 105.74→105.47. 모두 ±3% 안이라 중립이다. work·hash는 같다. cand3 1회 측정에서 본 −4.7/−3.8%는 교대 쌍에서 재현되지 않았다.
- **functional `batch`+`cache`(rpc4, base rpb 재사용):** functional 통과(tool·cache 모두). batch 동시 short reply·pair와 cancelled_peer 행 43개의 decode 합계는 base 7,269 ms, cand2 8,718 ms, cand4 6,932 ms다. batch decode_ms flag는 57개 가운데 cand 16, cand2 22, cand4 14개다. 남은 flag는 대부분 replies 행의 +5~+25 ms이고, pairs 한 행(f64690)만 92→239 ms다. cancelled_peer_isolation 01f1bb는 731/1332/619 ms다. `session_rollback_bytes=0`은 cancel 뒤에는 나오지 않는다. 처음 admission 6건에만 있다(base 49건, cand2 51건).

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
