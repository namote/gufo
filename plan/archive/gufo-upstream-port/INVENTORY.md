# gufo 원본 이식: 기능별 대응표 (2026-10-23)

상위 문서: [이식 계획](PLAN.md). 이 표는 우리 gufo 변경(운영 빌드 `210c7633`)이 원본 main `79c2dcc5`(0.21.0 이후)의 새 캐시 구조에서 어떻게 되는지 정리한다. 줄 번호는 표기한 커밋 기준이다.

- 우리 코드: gufo 작업 트리 `~/projects/gufo`, `210c7633`(운영 빌드)
- 원본: `upstream/main` `79c2dcc5`. Qwen 27B는 #540(`da5d5a2b`)부터 component cache(`ComponentTextCache` + `models/qwen/continuation_adapter.cpp`)로 서빙한다(`inference_backend.cpp:1057` `UsesComponentCache()`가 true).
- 원본에서 사라진 것: `QwenTextRunnerSnapshot`과 Qwen 영속 snapshot 형식. 원본 문서도 "Qwen's legacy server snapshot hooks are removed"라고 적었다(`docs/models/qwen3.8-27b/CONTINUATION-ADAPTER.md`). DeepSeek만 옛 `ContinuationCache`·`ContinuationDiskStore`를 쓴다(`docs/KV-CACHE.md:40-42`).

## 분류

- **(a) 모델·커널·core 수준:** 충돌만 풀면 옮길 수 있다.
- **(b) 지워진 Qwen snapshot 경로 위:** component cache 위로 다시 구현해야 한다.
- **(c) 공통 scheduler·runner:** 원본에서도 쓰인다(DeepSeek 옛 경로 또는 component 분기).

## 대응표

|기능|우리 측정 이득|분류|원본 0.21의 대응|판정|
|---|---|---|---|---|
|P1 블록 테이블 커널, `KvBlockPool`·`KvBlockTable`(`core/hip/kv_block_*`)|출력 39/39 동일, bench pp −0.7~−2 %(손실)|(a)|없음. 커널은 base 포인터 + `max_context` stride(`decode_step.cpp:114-127`, `prefill_chunk.cpp:268-281`, `batched_decode.cpp:292-319`)|**이식(U2)**|
|P2 공유 풀·필요할 때 할당·블록 공유·CoW, 입장 예약·회수|256K×4가 40 GiB 풀, 64K×8이 24 GiB 풀에서 기동·정답. 풀이 모자라면 실패 대신 대기|모델 (a), 서빙 (b)|없음. 슬롯마다 최대 문맥 KV를 `hipMalloc`(`arena.cpp:792-798`). RFC: "Paged live KV … separately evaluated extensions"(`RFC.md:97`), "Bring paging forward if measurements show it is necessary"(`:1246-1249`), 실행 슬롯 메모리가 "largest unresolved resource limitation"(`:962-966`)|**이식·제안 핵심(U3)**|
|(신규) 슬롯 간 복원을 블록 매핑으로|—(우리도 아직 복원은 복사)|(b)|다른 슬롯 복원은 prefix 전체를 복사(`RFC.md:458-460`, 비교표 빨간 칸 `:954`)|**설계 후 측정(U4)**|
|P3 디스크 행 블록(내용 hash 체인)|SSD 쓰기 −51~−72 %, 256K 재시작 복원 23.6→15.3 s|(b)|chunk 파일(256행)을 **계보(lineage)로만** 공유. "equal tokens are insufficient"(`RFC.md:336-350`). 이미 가진 chunk는 다시 쓰지 않음(`cache/disk.hpp:94-96`)|**원본 방식으로 대체, 차이만 측정(U5)**|
|Q1 상주 블록 색인(내용 키)|공유 prefix 재시작 복원 −38~−40 %|(b)|내용 키 GPU 색인 없음. 원본은 계보만 공유하는 원칙|P3과 함께 판단(U5)|
|R1 디스크 선행 읽기(병렬 읽기·SHA 검사)|재시작 복원 32K 3.26→1.13 s, 256K 23.5→6.8 s|(b)|`Prefetcher`: 스레드 4개가 32 MiB 창 안에서 `POSIX_FADV_WILLNEED`(`cache/streaming.cpp:96-156`). 읽기·CRC 검사는 순서대로|측정 후 이식 여부(U5)|
|Q2 공유 prompt pass scratch|64K×8 host 여유 +14.5 GiB, 256K×4 +6.3 GiB, 출력 동일|(a)|없음. #688은 DFlash2 prompt feature용 별도 버퍼(세션당 약 400 MiB pinned)라 겹치지 않지만 같은 파일(`prefill*.cpp`, `executor.*`)을 고침|**이식(U1)**|
|Q3 공유 prefix 기준 512→128|팔로워 하나에 약 0.5 s|(c)|아직 512(`text_model_runner.hpp:591`). 원본 리더·팔로워 대기는 이미 있음(`text_generation_scheduler.cpp:1006-1076`)|작은 PR 후보(U5)|
|Q4 끝 checkpoint(1,024 간격 3개)|늦은 팔로워 재계산 4,098→1,031, TTFT −53~−62 %, 리더 +0.5~1.5 %|(b)|없음. grid는 prompt 끝 128 안을 건너뜀(`cache/retention.cpp:27-93`)|`PlanCaptures` 정책으로 다시 구현 후 측정(U5)|
|P4 system 메시지 경계 checkpoint|공유 system 재시작 첫 대화 TTFT 14.36→2.95 s|(b)/(c)|"stable boundary"를 필수 지점으로 잡음(`retention.cpp`). system 끝과 같은지 미확인|확인 후 판단(U5)|
|B1–B2 장치→디스크 bounce 스트리밍|디스크 쓰기 8K 0.9→0.75 s|(b)|1 MiB 고정 staging, FP16 행은 복사 없이 빌려 씀(#540)|**버림**(원본이 해결)|
|B3 축출 시 디스크 write-back, idle write|재시작 T3 TTFT 2.2–2.4→1.2–1.3 s|(b)|idle spill은 RAM으로만(`cache/slot.cpp:77-300`). 축출 시 디스크 쓰기는 없음|원본 위에서 필요성 측정(U5)|
|S4 예산 하한 검사|하한 충족 시 1개 초과로 인한 건너뜀 0|(b)|staging 고정 1 MiB라 staging 하한은 의미 없음. #677이 RAM 초과 checkpoint를 실행 중 디스크로|**버림**|
|P5 수정, Q0 수정, 리뷰 수정|—|(b)|해당 경로가 없음|이식 기능을 따라 다시 판단|

## 원본 구조에서 paged KV가 걸리는 곳 (`79c2dcc5`)

1. **할당:** `QwenGpuArena::QwenGpuArena`의 KV `hipMalloc`(`arena.cpp:702-798`)과 `GetAttentionKvPlaneElements`(`executor.hpp:339-342`)
2. **슬롯 생성:** `ContinuationAdapter::CreateSlot`(`continuation_adapter.cpp:398-417`). 공유 풀을 executor에 넘기기에 맞는 자리다. `LeasedSlot`이 이 함수로 슬롯을 만든다(`cache/slot.cpp:109`, `:833`).
3. **커널:** KV 쓰기·attention 커널이 블록 테이블을 받아야 한다. prefill attention과 검증(verification) 경로는 아직 읽지 않았다.
4. **어댑터 주소 계산:** `BorrowRowsView`(`continuation_adapter.cpp:786-811`)와 `TransferRows`(`:812-850`)가 `(plane+layer)*context_ + first`로 연속 배치를 가정한다.
   - 원본 chunk가 256행이고 우리 블록이 64행이므로, chunk 하나가 블록 4개에 걸친다.
   - 블록 크기를 256행으로 맞출지, borrow view가 여러 조각을 갖게 할지 정해야 한다(PLAN 결정 D5).
5. **보존 규칙:** `State::BeforeWrite`·`GuardAll`·`BeforeRelease`(`:133-223`)와 `slot.cpp`의 `Preserve`(`:207-240`). 블록을 공유하면 "덮기 전에 복사"가 "참조 수 확인 후 CoW"로 바뀐다.
6. **복원:** `ComponentTextCache::Acquire`가 `BeginRestore`·`CopyRowsIn`을 부른다(`component_text_cache.cpp:1099-1130`). 블록 매핑 복원(U4)이 들어갈 자리다.
7. **입장:** 원본 슬롯 선택은 슬롯이 없으면 `SlotBusy`를 던지고 scheduler가 대기열에 넣는다(`component_text_cache.cpp:968-1058`). 풀 블록 예약은 여기에 더해야 한다.

## 확인하지 못한 것

- 원본 prefill attention·검증 커널의 KV 주소 방식
- 원본 계보 공유가 서로 다른 대화의 공통 앞부분을 얼마나 공유하는지(우리 내용 hash와의 실제 차이)
- 원본 "stable boundary"가 system 메시지 끝과 같은지
- DFlash2 draft ring 크기(`sliding_window`)와 원본 backing segment 크기(문서에는 128 MiB와 1 GiB가 섞여 있음)
