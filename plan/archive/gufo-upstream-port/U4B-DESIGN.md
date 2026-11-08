# gufo 원본 이식 U4b: 긴 prompt 끝 앞쪽 checkpoint로 첫 팔로워 지연 고치기 (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [U4 설계](U4-DESIGN.md), 결정 D4.

사용자 결정: "B계획없애고 그냥 1번을 고치는거로 계획을 변경하자".
- 처음 U4b였던 행 바이트 hash 저장 중복 제거(B)는 하지 않는다.
- 남는 비용은 SSD 중복 쓰기다(32K 12.9 / 6.6 GB). 사용자가 받아들였다.

## 왜

- **현상:** 공유 대화의 첫 팔로워 TTFT가 이식본 7.88 s, 운영 2.30 s다(F2-a06). RAM 예산 2·1 GiB에서는 14.4 s다(f3).
- **원인은 계보 원칙이 아니다.** 첫 팔로워가 쓸 수 있는 지점, 즉 공통 prefix 안쪽에 checkpoint가 없다.
  - 앞 대화 A의 checkpoint는 A의 prompt 끝(약 8.2K)에만 있다. 이것은 B의 prefix가 아니다.
  - 원본 grid는 앞쪽(1K, 2K, …)에만 잡힌다(`GridLimit`, 최대 4개).
  - 원본은 첫 팔로워가 처음부터 prefill하는 동안 분기점(7,174)을 배워 잡는다. 그래서 두 번째 팔로워부터 빠르다.
- **정정(결과 뒤):** 아래 "운영이 빠른 이유"는 틀렸다. 운영의 첫 팔로워는 system prompt 끝 checkpoint(`system_prefix_tokens`)를 복원한다. tail은 분기점 밖(7,177)이라 쓰이지 않는다([U4b 결과](U4B-RESULTS.md) "운영이 빠른 진짜 이유").
- **운영이 빠른 이유(처음 분석):** `--cache-tail-checkpoints K`(기본 3). 긴 prompt는 끝 앞쪽에 1024토큰마다 recurrent 상태까지 담은 RAM checkpoint를 K개 남긴다(운영 `210c7633` `serve.cpp:776`, paged KV일 때만). 첫 팔로워는 토큰 prefix로 그중 하나를 찾아 복원한다.
  - 다른 대화의 checkpoint를 통째로 복원하는 것은 원본에서도 평범한 prefix cache다. 계보 원칙(저장 chunk를 합치지 않는다)과 충돌하지 않는다.
- U4 뒤에는 KV 행이 장치 블록 참조라 checkpoint 하나의 RAM은 private 상태(약 245 MB)뿐이다. 예전처럼 행 사본 GB 단위가 들지 않는다.

## 무엇을

구현: gufo `ai-tools/port-u4b` `9a5cacee`(`17ef4326`에서 분기). 운영 `PlanTailCheckpoints`의 규칙을 원본 `PlanCaptures`로 옮겼다.

1. **계획(`src/cache/retention`):**
   - `CapturePlanRequest.tail`(K)과 `CaptureCandidate.tail`을 더했다. tail은 선택 후보이고 purpose는 grid다.
   - 기준점: prompt 끝 128토큰 안에 다른 멈춤(stable 등)이 있으면 그것, 없으면 prompt 끝이다.
   - 기준점에서 1024(`kTailStep`)씩 거꾸로 센다. 예: 끝이 8,201이고 pass가 4096이면 7,177, 6,153, 5,129이다.
   - 그 지점이 pass 조각을 1024보다 짧게 남기면 그 조각의 시작으로 옮긴다. 예: pass 1024이면 7,168, 6,144, 5,120이고, pass를 나누지 않는다.
   - 이전 멈춤(재사용 frontier, grid 등) 뒤나 다음 멈춤 앞에 1024보다 적게 남기는 지점은 넣지 않는다. 이미 있는 checkpoint 지점도 K에 센다.
2. **capture:**
   - Qwen 27B는 pass 안 checkpoint가 없다. 그래서 prefill이 tail 지점에서 멈춰 capture한다(grid와 같은 경로, `SnapshotPurpose::kHistory`).
   - RAM만 쓴다(grid는 디스크에 쓰지 않는다). retention 우선순위도 grid와 같다. RAM이 모자라면 거절되고 그대로 진행한다.
   - 공유 checkpoint(`ShareCheckpoint`)를 잡으면, 그 1024 안의 tail 지점은 뺀다(운영 `DropTailCheckpointsNear`).
3. **옵션:**
   - `--cache-tail-checkpoints K`. 이름·기본값(3)·뜻은 운영과 같다. 0이면 끈다.
   - 적용 조건은 capture가 KV 행을 장치 참조로 갖는 것이다(`ComponentTextCache::CapturesShareRows`, U4). 모델 이름이 아니라 연산 계약이다.
   - 그렇지 않은 구성은 시작 때 `event=tail_checkpoints status=not_applied`를 남긴다(component cache: `checkpoints_copy_rows`, 기존 cache: `snapshots_copy_rows`).

## 실패 처리

- tail capture는 선택이다. 거절·실패는 이벤트로 남기고 요청은 그대로 진행한다(원본 grid와 같다).
- 다른 경로로 자동 전환하거나 재시도하지 않는다.

## 검증

- **단위(T0):** CPU 127/127 통과. 새 검사는 수정 없이 실패하는 것을 확인했다.
  - `cache_retention_test` Planning: pass 4096·1024, 끝 근처 stable, 재사용 frontier, 이미 있는 지점, grid가 막는 경우, K=0.
  - `component_text_cache_test` FirstFollowerRestoresATailCheckpoint:
    - 공통 1,500토큰 뒤 다른 대화의 첫 팔로워가 앞 대화의 tail(1,076)을 복원한다. 토큰은 처음부터 계산한 것과 같다.
    - K=0이거나 행을 복사하는 capture면 0부터 시작한다.
- **GPU 창:** U4와 같은 창.
  - `final-f6-a01`: 운영과 번갈아, T2 포함.
  - `final-f7-a01`: RAM 예산 2 GiB, 이식본만.
  - 판정: 텍스트·gold 동일, T2 21/21.
  - shared 첫 팔로워 TTFT가 운영 수준(약 2.3 s)인지 본다.
  - 다른 창의 TTFT 합, 긴 prompt 첫 prefill 시간(pass 분할 비용), anon을 기록한다.
  - GPU 코드는 U4 이후 바뀌지 않았다. 그래서 모델 GPU 테스트는 다시 돌리지 않는다(`final-f4-b01` PASS).
