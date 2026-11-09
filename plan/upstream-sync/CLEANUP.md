# 남은 작은 문제 정리 계획 (C-series, 2026-11-09)

이 문서의 문제들은 배포 중인 `runtime-hc-20261108`(트리 `80b51e9c`)을 코드만 읽어 점검하다가 나왔다. 빌드나 실행 없이 정적으로 추적했다. 감사 메모는 세션 scratchpad의 `baseline-audit.md`에 있다.

U1~U3에서 이미 고친 것(#849, #1048, #875, #881, 정렬 #877/#1016 등)은 [PLAN.md](PLAN.md)에 있다. 여기에는 아직 남은 것만 적는다.

U5가 끝난 뒤 시작한다. U5 GPU 측정이 도는 동안에는 빌드하지 않는다. 작업 방식은 다른 단계와 같다. 항목마다 구현하고, 빌드와 CPU 테스트를 돌리고, 새 리뷰어가 검토한 뒤 고친다. GPU 확인은 마지막에 한 번 몰아서 한다.

| # | 문제 | 심각도 | 서비스 영향 | 처리 |
|---|---|---|---|---|
| C1 | MTP draft 캐시를 따라잡는 세션이 draft 중인 세션과 함께 batch되면, 혼자 돌 때와 다른 경로로 계산된다(`executor_batch.cpp:276-279`). 그 결과 draft K/V 행의 FP16 마지막 비트가 달라진다. `CheckBatchedSessions`/`CheckLookupBatch`가 실패하는 원인이다 | 중하 | 있다. greedy 출력은 그대로지만, sampled 출력(T 0.9)은 같은 세션에 있는 다른 요청이 batch되는지에 따라 달라질 수 있다 | U5에서 GPU 한 번으로 원인을 확인한다. 고칠 때는 따라잡는 세션만 따로 `kv_only` 하위 batch로 돌린다. 그러면 session 테스트가 다시 끝까지 돈다. 23,822 B 차이는 다음 `DraftReplay`가 다시 만드는 `mtp_.h` 잔여 상태라 무시해도 된다 |
| C2 | `--cache-disk-staging-bytes`를 Flash-Next와 Qwen27B가 무시한다. 내부 버퍼가 1 MiB로 고정돼 있고, `event=option_ignored` 로그만 남긴다. 그런데 배포 supervisor는 256 MiB를 넘긴다 | 낮음 | 없다. 버퍼가 커도 속도가 같았다(SERVER.md:322) | (1) `tools/serving/flash_next/gufo-server.py`에서 `DISK_STAGING_BYTES`와 이 인자를 지운다. 설치본에는 다음 배포 때 반영한다. (2) Flash-Next와 Qwen27B에서 이 옵션을 받으면, 무시하지 말고 시작할 때 오류로 거부한다. 옵션은 다른 모델이 쓰는 legacy continuation disk store를 위해 남긴다. (3) `tests/functional/run.py:296,347`, `cache_startup.py`, `cache_lifecycle.py`에서 Flash-Next와 27B로 실행할 때 이 인자를 넘기지 않게 정리한다. SERVER.md 문구도 "무시"에서 "거부"로 바꾼다 |
| C3 | Flash-Next 디스크 캐시의 호환 키에 계산 방식이나 커널 버전이 없다(`inference_backend.cpp:298-337`, `continuation_adapter.cpp:244-258`). 계산 결과를 바꾸는 변경에서 `kSnapshotPayloadVersion`을 올리지 않으면, 예전 계산으로 만든 디스크 상태를 그대로 복원한다 | 낮음, 잠재 | 지금은 없다. v19로 막혀 있다 | 추천하는 방법: 빌드에 들어간 device code object의 hash를 호환 키에 넣는다. 그러면 커널이 바뀔 때마다 디스크 캐시가 자동으로 한 번 비워진다. 배포가 드물어서 비용은 작다. 계산 방식은 그대로 두고 호스트 코드만 바꾸는 배포는 영향을 받지 않는다. CPU 테스트로 "hash가 바뀌면 identity가 바뀐다"를 확인한다 |
| C4 | 체크포인트를 저장할 때 `RetainRows`가 두 세션이 함께 쓰는 stream을 `hipStreamSynchronize`로 기다린다. 이때 `retained_mutex_`를 쥔 채로 기다린다(`continuation_adapter.cpp:905-912`). 그래서 세션 A의 저장이 세션 B의 prefill이나 decode가 끝날 때까지 기다릴 수 있다 | 낮음, 지연만 | 있다(`--sessions 2`) | 먼저 GPU에서 B가 긴 prefill을 하는 동안 A의 capture 시간을 잰다. 지연이 의미 있게 나오면 고친다. stream 전체 대신 자기 작업 뒤에 기록한 event만 기다리고, mutex 밖에서 기다린다 |
| C5 | `Transfer`가 `job->completion`과 `result`에 lock 없이 쓴다(`continuation_adapter.cpp:545-556`) | 낮음 | 없다. KV pool을 쓰면 in-pass checkpoint가 꺼져서 동시에 읽는 쪽이 없다 | lock이나 atomic을 추가한다. 고치는 비용이 작아서 그냥 고친다 |
| C6 | 감사가 다루지 못한 부분: KVarN k4v4/k6v6에서 되돌린 뒤 그룹 경계를 다시 양자화하는 경로. `retention.cpp`, `checkpoint.cpp`, `ledger.cpp`도 대충만 봤다 | 모름 | 지금 서비스는 FP16 KV라 낮다 | 코드만 읽는 감사를 한 번 더 한다(빌드 없이 다른 작업과 병렬로 할 수 있다) |
| C7 | 운영 정리: `~/.local/share/gufo/runtime-b9-20261108`이 backup 위치 규칙(`~/projects/_backup`)을 어기고 있다. `README.txt`의 runtime 줄도 예전 값이다 | 낮음 | 없다 | U5에서 배포할 때 HANDOFF rollback 기록과 함께 옮기고 고친다. U5가 하지 않으면 여기서 한다 |

순서: C2 → C5 → C3 → C1 → C6(병렬로) → C4(GPU 측정 뒤 결정). GPU 확인은 C1(session 테스트 전체와 sampled restore 비교)과 C4(capture 지연)를 한 번에 묶는다.
