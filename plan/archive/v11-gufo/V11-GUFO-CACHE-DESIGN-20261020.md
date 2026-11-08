# v11 gufo 캐시 개선 설계 (2026-10-20)

상태: S4 구현·검증 완료(2026-10-20, gufo `ai-tools/cache-s4` `aa89d2a1`, 결과는 [S4 결과](V11-GUFO-S4-RESULTS-20261020.md)). S1–S3은 구현 전. 사용자 결정: 밀어내기 정책(1)과 대화별 최신 경계(2)에 동의. 예산은 자동으로 정하지 않고 명시 설정으로 두되, 하한은 "컨텍스트 최대 길이의 checkpoint 1개"로 자동 검사한다(3). 실험 기본값·Compose 불변. 측정 근거는 [gufo 실측](V11-GUFO-EVAL-20261020.md) 2차·5차.

## 문제 (측정)

- **순환 접근에서 재사용 0** (5차 `qwen27b-lru`): disk 2.5 GiB(8K checkpoint 3개분)에 대화 6개를 차례로 돌리면 T2 6/6 miss, 재시작 후 T3 5/6 miss. 저장 19건 중 16건을 LRU가 제거했다. 전역 LRU는 순환 접근에서 매번 다음에 쓸 대화를 지운다. 여러 프로젝트를 번갈아 쓰는 사용 방식이 정확히 이 패턴이다.
- **짧은 턴의 경계 미저장** (2차): 턴 증가분이 `min_step` 2,048토큰보다 작으면 disk 저장을 건너뛴다. 재시작 후 마지막 경계가 아니라 첫 프롬프트 경계부터 복원해 약 500토큰을 다시 prefill했다. 긴 턴(5차 `qwen27b-boundary`, +2,600토큰)은 경계가 저장·복원된다(11,509토큰, TTFT 20.2→2.0초).
- **예산 미달은 조용히 실패**: checkpoint가 staging이나 disk 예산보다 크면 저장을 건너뛰고 로그만 남는다(`docs/SERVER.md`의 "Snapshots that exceed either limit are skipped").

## 현재 구조 (gufo `a8e40fd8`, `~/projects/gufo`)

|부분|위치|현재 동작|
|---|---|---|
|옛 continuation disk store (Qwen 27B, DFlash2 포함)|`src/cli/serve/continuation_disk_store.cpp`|압축 token prefix 트리 `PrefixNode`(469), 항목 `Entry`(455, `last_access`만 보유), 전역 LRU `LeastRecentlyUsed`/`EvictToCapacity`/`MakeCapacity`(995–1030), `WithinCheckpointStep`(1442, `min_checkpoint_step_tokens`)으로 짧은 증가분을 건너뜀(1186)|
|새 component cache (Flash-Next, 포크 PR #3의 Gemma4)|`src/cache/` (`prefix_index`, `retention`, `disk`, `disk_catalog`)|RAM 보존에 카드 22 정책 일부 반영(`retention.cpp:224` 재사용 turnover). 카드 13(RAM·disk 통합 조회·참조 기반 축출) 완료|
|upstream 계획|`docs/cache-redesign/cards/15`,`16`|Qwen 27B AR·DFlash2를 component cache로 이전: proposed|

대화 prefix 트리(사용자 아이디어)는 두 저장소 모두에 이미 있다. 빠진 것은 트리 위의 **보존 정책**이다.

## 방향

정책은 모델별로 복제하지 않고 **component cache 하나에 공통으로** 넣는다(AGENTS 공통 최적화 원칙). Qwen 27B는 카드 15·16으로 component cache에 옮겨 같은 정책을 받게 한다. 옛 disk store에는 새 정책을 넣지 않는다. 이전 전 비교 측정용 기준선으로만 둔다.

순서:

1. **S1 Qwen 27B adapter 이전** (카드 15 AR → 16 DFlash2): capture/restore가 정확한 상태를 재현함을 모델 단위 시험으로 증명한다. 정책 변경은 없다. 옛 store와 같은 시나리오 결과(5차)를 기준으로 동등성을 확인한다.
2. **S2 disk tier 보존 정책** (결정 1).
3. **S3 대화별 최신 경계** (결정 2).
4. **S4 예산 하한 검사** (결정 3). 독립적이므로 S1 전에 옛 store에도 먼저 넣을 수 있다. 실패를 드러내는 검사일 뿐 정책이 아니기 때문이다.

## 결정 1: disk 밀어내기를 대화 단위로

카드 22의 RAM byte-pressure 순서를 disk tier에 같은 의미로 확장한다.

- **대화**는 prefix 트리에서 한 leaf까지의 경로다. 클라이언트 세션 ID는 쓰지 않는다(현재 계약 유지). 대화의 최근성은 자손 checkpoint의 최근 사용까지 포함한다.
- **재사용 표시**: restore나 continuation에 쓰인 checkpoint와 그 턴이 만든 checkpoint에 표시한다. 표시 이후 새로 저장된 disk bytes가 disk 예산의 1회전을 넘으면 표시가 만료된다(카드 22의 turnover를 disk 예산 기준으로).
- **제거 순서** (disk 예산 초과 시):
  1. retry 사본
  2. 덮인(covered) 오래된 경계 (같은 대화의 더 깊은 경계가 있는 것)
  3. 재사용 표시가 없는 대화 (최근성 순, 오래된 것부터)
  4. 재사용 표시가 있는 대화
  - 각 대화 안에서는 덮인 것 → leaf → 공유 분기점 순서로 제거한다.
  - 다른 대화가 함께 쓰는 공유 prefix(시스템 프롬프트 등)는 마지막 참조가 사라질 때까지 남긴다.
- **순환 접근의 기대 동작**: 예산이 대화 k개분이고 대화가 N > k개면 어떤 정책도 전부를 지킬 수 없다. 목표는 LRU의 재사용 0을 피하는 것이다. 이미 남아 있는 대화가 계속 적중하도록 새 대화의 저장이 기존 대화를 밀어내지 않게 한다. 이렇게 하면 k/N 정도가 적중한다(5차 조건에서 3/6). 새 대화의 저장을 거부할지는 재사용 표시의 만료 규칙으로 정해진다.
- **저장 거부도 드러낸다**: 정책 때문에 저장하지 않은 경우 `event=disk_cache action=skipped reason=retention`과 대상 대화를 기록한다. 조용한 생략은 없다.

## 결정 2: 대화별 최신 경계 1개

- `min_step` 생략의 예외: 후보가 같은 대화의 기존 disk 경계를 2,048토큰 미만으로 연장하면, 건너뛰지 않고 **그 경계를 교체**한다. 새 파일이 완전히 기록(durable)된 뒤 이전 파일을 제거한다. 이는 현재의 "교체 전 가시 항목을 지우지 않는다" 규칙(1237)과 같다.
- 쓰기량 제약: 8K Qwen checkpoint는 약 0.8 GB다. 매 턴 교체하면 대화당 턴마다 0.8 GB를 쓰게 된다. 그래서 교체는 **쓰기 지연(write-back)** 으로 한다. 경계는 RAM에 두고, 그 대화의 RAM 사본이 축출되거나 대화가 일정 시간 idle이 되거나 서버가 종료될 때만 disk에 쓴다. 종료 경로는 이미 SIGTERM에서 accepted disk write를 drain한다. 이로써 활발한 대화는 턴마다 쓰지 않고, 놓아둔 대화만 한 번 쓴다.
- 공유 prefix 예외(현재 `shared_prefix` 면제)는 유지한다.

## 결정 3: 예산 하한 = 최대 컨텍스트 checkpoint 1개

- 시작 시 model adapter가 `--context` 길이의 checkpoint 크기를 보고한다. 지금의 `PersistentSnapshotPayloadBytes`를 길이 함수로 노출한다. Qwen 27B 실측 추정은 약 243 MB + 65.5 KB/token이므로, `--context 34816`에서 약 2.5 GB다. adapter 값으로 확인해야 한다.
- `--cache-disk-bytes`, `--cache-disk-staging-bytes`가 이 하한보다 작으면 **시작을 실패시키고 필요한 크기를 출력**한다. 자동으로 올리거나 내리지 않는다.
- `--cache-ram-bytes`(GPU 보관 snapshot, 통합 메모리라 RAM으로 불림, 명시값)가 하한보다 작으면 `status=snapshot_below route=disk`로 보고만 한다. 긴 checkpoint는 보관 snapshot을 건너뛰고 disk로 가므로 손실이 아니다(사용자 결정 2026-10-20, 초안의 거부에서 변경).
- 예산 자체는 사용자가 명시한다(프로젝트 수·평균 깊이는 서버가 알 수 없다). 시작 로그에 "최대 길이 checkpoint 몇 개분인지"를 함께 출력해 판단을 돕는다.
- 하한을 통과해도 실제 대화가 하한보다 길어지는 일은 `--context`가 막는다. 따라서 checkpoint 하나가 예산을 넘는 `staging_capacity`·`byte_capacity` 생략은 정상 설정에서 나오지 않게 된다. 예산이 이미 찬 상태의 생략은 보존 정책(S2) 몫으로 남는다.
- 구현 범위: 옛 continuation store 경로(Qwen 27B AR·DFlash2). component cache(Flash-Next, Gemma4)는 단일 checkpoint 크기가 없어 `status=unsupported`로 기록한다. S1 이후 component 경로의 하한(최대 길이 row 집합)을 따로 정한다.

## 실패 처리

- adapter capture/restore 불일치: 해당 checkpoint를 쓰지 않고 오류 이벤트로 남긴다. 다른 형식이나 다른 모델로 재시도하지 않는다.
- 예산 하한 미달: 시작 실패(결정 3).
- disk 쓰기 실패·공간 부족: 그 checkpoint만 생략하고 사유를 기록한다. 기존 durable 항목은 지우지 않는다.
- 비정상 종료 후: 기존 publication·crash 복구 규약(카드 11)을 따른다. 미완성 파일은 다음 시작 때 제거한다.

## 검증 기준

같은 장비·같은 모델·같은 요청(`test/experiments/gufo_eval_20261020`, `PLAN-R2.json`/`PLAN-R5.json` 확장)으로, 옛 store 기준선과 순차 단일 GPU 창에서 비교한다.

|항목|기준|
|---|---|
|정확성|캐시 경유 출력이 cold 재계산과 텍스트 동일(T2·T3, 모든 대화), 정답 유지|
|S1 동등성|5차 두 시나리오에서 복원 토큰 수가 기준선과 같거나 많음, TTFT 회귀 ≤5 %|
|S2 순환 접근|대화 6개·예산 3개분에서 T2·T3 적중 ≥3/6(기준선 0/6, 1/6)|
|S2 회귀|예산이 충분한 2차 캐시 시나리오에서 적중 6/6 유지|
|S3 짧은 턴|재시작 후 T3가 T2 경계에서 복원(재계산 ≤ 마지막 턴 증가분 + 꼬리), 활발한 대화의 턴당 disk 쓰기 0|
|S4 하한|disk·staging 미달은 시작 실패와 필요 크기 출력, 보관 snapshot 미달은 보고 후 시작. 하한 이상에서 disk·staging의 단일 checkpoint 초과 생략 0(예산이 찬 상태의 보존 압박 생략은 정상, S4 실측으로 정정)|
|쓰기량|시나리오별 disk 총 기록 bytes를 기록하고, 기준선 대비 증가는 사유와 함께 보고|

## 위치와 범위

- 코드 변경은 외부 소스(gufo 포크)에서 한다. 실험 빌드·실행·비교는 `test/experiments/` 아래에 둔다. `tools/*/engine`·`tools/common`·`infra`에는 넣지 않는다.
- upstream과 겹치는 부분: 카드 15·16(Qwen 이전)과 카드 22(RAM 정책)는 upstream 진행 중이다. 착수 전 upstream 최신 상태를 다시 확인해 중복 구현을 피한다. 결정 1의 disk 확장과 결정 2·3은 upstream 계획에 없는 부분이다.
- 미검증: Gemma4 disk tier(포크 PR #3의 component cache 경로)에 같은 정책이 그대로 맞는지는 S2 이후 Gemma 시나리오로 따로 확인한다.
