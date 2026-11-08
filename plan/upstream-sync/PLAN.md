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
