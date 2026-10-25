# gufo 원본 이식 U4e: PR 전 코드 리뷰와 수정 (2026-10-25)

상위 문서: [이식 계획](PLAN.md), [U4d 결과](U4D-RESULTS.md).

사용자 지시: "코드 리뷰한번하자. pr 던지기 전에" → "오케이 수정하고 테스트 하자".

## 리뷰

- 대상은 원본 main `79c2dcc5`(`port-u0`) 위의 `ai-tools/port-u4d` 22커밋(64파일, +5,125줄)이다.
- 네 영역으로 나눠 읽기만 하는 리뷰를 돌렸다: KV 메모리 배치, serve 스케줄링, cache 디스크, U4d 버퍼와 PR 준비 상태.
- 중요한 지적은 코드에서 직접 확인했다.

## 고친 것

구현: gufo `ai-tools/port-u4e` `f07aa9f1`(`05ffd340`에서 분기, 리뷰 수정 `16c41d95` + 9번 정정).

|번호|문제|수정|테스트|
|---|---|---|---|
|1|유휴 세션을 디스크에 쓰다 실패하면 그 세션의 live KV를 버린다(`WriteBackIdle`)|capture만 실패로 기록하고 slot은 commit한다(capture는 slot을 읽기만 함)|`FailedWriteBackKeepsTheLiveFrontier`|
|2|큐가 거절해도 "디스크에 씀"으로 표시해 종료 때 건너뛴다. 쓰기 작업이 실패해도 표시가 남는다|큐에 들어간 것만 표시한다. 시도 길이는 따로 둬서 유휴 재시도는 64토큰 자랄 때까지 미루고, 종료 때는 다시 쓴다. 실패한 작업은 표시를 지운다|같은 테스트|
|3|디스크 cache가 없으면 system prompt checkpoint 최소 길이가 0이다|최소값 기본을 디스크 옵션 기본(128)으로 둔다|`FirstFollowerRestoresTheSystemPromptCheckpoint`에 100토큰 경우 추가|
|4|재배정 직전 capture가 해제하려는 KV 블록을 다시 붙잡는다(장치 행 보존일 때)|장치 행 보존이면 row 회수 경로에서 capture하지 않는다|`RowReclaimEmptiesSlotsBeforeDroppingCheckpoints`|
|5|이어지는 세션의 입장 판단이 그 slot이 이미 가진 블록을 세지 않는다|`ContinuedTokens`로 Acquire가 고를 live frontier를 찾아 그 블록을 뺀다|`PagedRowsAdmitWaitAndReclaim`에 추가|
|6|RAM에서 디스크로 간 checkpoint를 lineage+토큰만으로 다른 대화가 자기 것으로 삼는다(설계 주석이 금지)|checkpoint마다 만든 실행(`branch`)을 기록하고 같은 실행만 받아들인다|`SiblingDoesNotAdoptAnotherExecutionsCheckpoint`|
|7|row 회수가 빈 slot보다 checkpoint 삭제를 먼저 한다|빈 slot → checkpoint 삭제 → live frontier 순서|4번 테스트|
|8|로그가 용량 거절을 `captured bytes=0`으로, 할당 실패를 예산 거절로 적는다|`capture_refused cause=byte_capacity`/`backing_exhausted`로 나눈다|—|
|9|copy-on-write 여유 2블록이 최대 문맥에서 잘린다|draft 행 여유 2블록은 문맥 상한 안에 그대로 두고, 다른 소유자와 공유 중인 매핑 블록 수(최대 2)만 상한 밖에서 더 예약한다(arena). admission 추정은 원래대로|GPU `inference_backend_gpu_test`(f11에서 잡힘, 아래)|
|10|공유 prompt 버퍼가 소멸할 때 자기 stream을 비우지 않는다|holder면 `hipStreamSynchronize`|GPU `TestSharedPagesDrainStreams`(실제 stream, 넘겨받기·소멸 drain, 공유 pageable 전환)|
|11|256행 chunk와 256토큰 KV 블록이 같다는 가정을 검사하지 않는다|`RetainsRows`가 `rows_per_chunk == kKvBlockTokens`를 요구|—|
|12|주석·문서가 바뀐 동작과 다르다|`WriteBackIdle` 계약, scheduler 시계 주석, U4d 주석 위치, "(measured, U5)" 제거. `--kv-pool-bytes`, `--cache-tail-checkpoints`, 유휴·종료 write-back, 장치 보존 chunk의 RAM 회계, 공유 prompt 버퍼를 `docs/KV-CACHE.md`·`docs/SERVER.md`에 적었다|—|

- 1번과 6번 테스트는 옛 코드로 되돌리면 실패하는 것을 확인했다. 6번은 옛 코드에서 B의 frontier가 아예 디스크에 쓰이지 않았다.
- CPU 127/127, clang-format 통과. 9번은 GPU에서 다시 고쳤다(아래).

## GPU 검증

- 둘 다 RAM 예산 2 GiB, 256K × 4, 풀 40 GiB. 순서는 U4d(f10)와 같다: GPU 단위 테스트 → 6개 창을 운영 `210c7633`(`fb-*`)과 번갈아 → T2 → `memprobe`. run 직전 `fstrim`.

**`final-f11-a01`(`16c41d95`): 단위 테스트 1개 실패**
- `inference_backend_gpu_test`가 "the shared KV block pool cannot hold this request: it needs 3 more blocks of 256 tokens, 1 of 1 are free"로 실패했다.
- 원인은 9번 수정이다. 여유 2블록을 항상 문맥 상한 밖에서 더하니, 세션 수 × 전체 문맥으로 잡은 기본 풀(`sized=sessions`)에서 마지막 전체 문맥 세션이 들어가지 못한다. 테스트만의 문제가 아니라 기본 풀 용량 회귀다.
- 나머지 14개 창은 COMPLETE였고 텍스트·TTFT·anon이 U4d와 같았다. 증거는 남겼고 결과는 f12로 대신한다.
- 정정: 2블록 여유는 draft 행 몫이라 원래처럼 상한 안에 두고, copy-on-write는 실제로 공유된 매핑 블록에만 붙인다(`f07aa9f1`). CPU 127/127, clang-format 통과.

**`final-f12-a01`(`f07aa9f1`): 15개 창 모두 COMPLETE, health PASS**
- GPU 단위 테스트 모두 PASS(`TestSharedPagesDrainStreams`, `inference_backend_gpu_test` 포함).
- 텍스트는 모든 창에서 U4d·U5b a06·운영(같은 run)과 같다. gold 통과, T2 21/21 같다.
- 서버 로그에 `ERROR`, `logic_error`, `length_error`, write-back 실패가 없다. 새 로그 `live_frontier_capture_refused cause=byte_capacity`는 T2에서 1번 나온다. U4d에서도 같은 자리에서 나온 거절이고 이름만 바뀌었다.
- `slot_pinned_bytes=419430400`(공유 prompt 버퍼 400 MiB 한 번)는 그대로다.

|항목(초, GiB)|운영(f12 run)|U4d(f10)|U4e(f12)|
|---|---|---|---|
|memprobe 준비 직후 anon|—|18.30|**18.30**|
|창별 최대 anon|16.2–17.0|18.37–18.55|18.37–18.55|
|shared 첫 팔로워 c1|2.29|2.30|2.30|
|shared 둘째 턴(6개)|0.87–0.88|0.38–0.46|0.39–0.46|
|shared 합(19)|50.8|46.0|45.9|
|long32k 합(8)|245.9|245.6|246.4|
|32K 재시작 뒤 T3|1.17|0.86|0.84|
|256K 첫 턴(cold prefill)|884.1|901.4|899.7|
|256K 재시작|4.46|4.54|4.54|
|regress 합(20)|26.8|21.7|21.8|
|boundary 합(11)|24.5|22.8|22.8|
|two8k 합(8)|59.8|57.9|58.1|
|T2 합(21)|—|66.3|66.3|

- 리뷰 수정으로 속도·메모리 변화는 없다. 256K cold prefill이 운영보다 약 2 % 느린 것은 U5b부터 있던 차이다.
- 증거: `storage/evaluations/gufo-upstream-port-final-20261023/final-f11-a01`, `final-f12-a01`. 두 run의 cache-disk(run마다 약 92 GiB)는 지우고 `CACHE-REMOVED.json`에 적었다.

## 고치지 않은 것

- batched pass 중간 예외 때 참가자 블록 해제가 coordinator stream을 기다리지 않는다. 정상 경로는 pass 끝 동기화로 안전하다. 예외 경로만의 문제라 PR 리뷰에서 다시 본다.
- `noexcept` 해제 함수 두 곳의 `dynamic_cast`/`RequireQwenState`는 타입이 어긋나면 terminate한다. 도달하지 않는 불변식 위반이라 그대로 둔다.
- Qwen KV plane 설명이 serve 코드와 arena에 중복된다. pool 크기 상한(2^24 블록) 검사가 느슨하다. 둘 다 27B에서는 도달하지 않는다.

## PR 준비에 남은 것

- **기본값 정책:** 원본 AGENTS.md는 "성공한 최적화는 별도 스위치 없이 기본"을 요구한다. `--kv-pool-bytes`는 opt-in이다. 기본으로 켤지 PR 전에 정한다.
- **커밋 정리:** 단일 줄 Conventional Commit 규칙. fix-up 커밋 합치기(e604d732·a4ce681a → f8c9df17, f2edd142 → 6f86d9ff, U4e 두 커밋은 각 원 커밋으로).
- **PR 나누기(약 7개):** 공유 pass scratch / paged KV pool / 디스크 복원 warm-up / 재배정 capture·유휴 write-back / checkpoint의 블록 보존 / checkpoint 위치 / 공유 prompt 버퍼.
- 디스크 helper 수(4→8)와 창(32→64 MiB)은 우리 QLC SSD에서만 쟀다.
