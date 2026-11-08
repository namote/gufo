# v11 gufo 캐시 S4: 예산 하한 검사 결과 (2026-10-20)

상태: 완료. 상위 문서 [gufo 실측](V11-GUFO-EVAL-20261020.md), 설계 [캐시 개선](V11-GUFO-CACHE-DESIGN-20261020.md). 실험 기본값·Compose 불변.

설계 결정 3을 구현했다. gufo 작업 트리 `~/projects/gufo`, 브랜치 `ai-tools/cache-s4`:
- `fa41fcff`: 엔진
- `0c8520b4`: 단위 테스트
- `aa89d2a1`: GPU 보관 snapshot은 거부하지 않고 보고만 하도록 변경

패치는 `test/experiments/gufo_eval_20261020/patches/`에 있다. 이미지 `ai-tools/gufo-eval:aa89d2a1928c`(같은 Dockerfile·기반), `PLAN-S4.json`, `startup.py`. 보존본은 `storage/evaluations/gufo-eval-20261020/s4-a03/`(최종)와 `s4-a02/`(`fa41fcff`, RAM 거부판)이다. a01은 하네스 KeyError로 docker 실행 전에 중단됐다.

- **캐시 단계(코드 확인):**
  1. live KV: GPU 메모리, session arena
  2. 보관 snapshot: GPU 메모리(`hipMalloc`, 장치 간 복사), 예산 `--cache-ram-bytes`
  3. disk: staging을 거쳐 기록

  별도의 시스템 RAM 단계는 없다. 이 장비는 VRAM carve-out이 4 GiB이고 `hipMalloc`은 GTT 112 GiB(같은 LPDDR5X)에서 잡힌다. 그래서 gufo는 2의 예산을 host 여유분으로 제한하고 RAM이라 부른다.
- **구현:**
  - 각 계층(Qwen arena compact layout, DFlash executor·draft, verifier)에 max_context checkpoint 크기를 상태 생성 전에 계산하는 정적 함수를 두었다. runner는 `MaxContextCheckpointBytes()`로 보관 snapshot 크기와 disk payload 크기를 보고한다.
  - disk가 켜져 있으면 `--cache-disk-bytes`와 staging(자동 유도값 포함)이 하한 미만일 때 `status=below`를 남기고 옵션 이름과 필요 크기를 담아 **시작을 실패**시킨다. 긴 대화가 어디에도 저장되지 않기 때문이다.
  - 명시한 보관 snapshot 예산이 하한 미만이면 `status=snapshot_below route=disk`로 **보고만** 한다. 사용자 결정(2026-10-20): 긴 checkpoint는 2를 건너뛰고 disk로 바로 가므로 손실이 아니다.
  - 통과하면 tier별 "checkpoint 몇 개분"을 출력한다.
  - 크기를 보고하지 않는 runner와 component cache(Gemma4 경로)는 `status=unsupported`로 기록하고 검사하지 않는다.
  - pending target feature는 verify 1회(block_size 행) 이하로 상한을 잡았다. 이미지가 있는 snapshot은 이미지당 12 B가 더해지며, 이 몫은 bound에 없다.
- **CPU:** `text_model_runner_test`와 `continuation_disk_store_test`가 통과했다. 새 테스트는 보관 snapshot 미달 보고, disk·staging 미달 거부, 하한 일치 통과, 자동 snapshot 미검사를 확인한다. 형식 검사(`tools/ci/check-format.py`)는 호스트·이미지에 clang-format이 없어 **미실행**이다.
- **GPU `qwen27b-floor`** (s4-a03, Qwen 27B DFlash2/AR, `--context 34816`): COMPLETE·health PASS, 6/6 기대대로.

|케이스|결과|
|---|---|
|보관 snapshot 1 GiB, disk 32 GiB|시작됨. `snapshot_below route=disk`, 0개분|
|disk 2 GiB|시작 거부. `--cache-disk-bytes gives 2147483648 bytes, below one checkpoint at --context 34816 (2526400230 bytes)`|
|staging 2 GiB|시작 거부. 같은 필요 크기|
|DFlash2 snapshot 3 GiB·disk 32 GiB|통과. snapshot 1개분, disk 13개분, staging(자동 11.3 GB) 4개분|
|AR 같은 예산|통과. snapshot 2,441,553,956 B, disk 14개분|
|예산 명시 없음|검사 이벤트 없음|

  - 32K DFlash2 checkpoint는 snapshot 2,526,259,344 B, disk 2,526,400,230 B이다. 구성은 KV fp16(16층 × 1024 × K·V × 2 B = 토큰당 65,536 B, 34,816토큰에 약 2.28 GB)에 길이와 무관한 상태(약 0.24 GB)를 더한 것이다.
  - bound를 2차 실측(8,194토큰 780,740,656 B, 8,187토큰 disk payload 780,282,288 B)을 토큰당 65,536 B로 늘린 값과 대조했다. bound가 각각 약 0.82 MB(0.03 %) 크다. 이 차이가 pending feature 상한 몫이다.
- **GPU 회귀 2건:** 요청 전부 COMPLETE, 텍스트 동일, 정답, 적중 토큰 수 동일. 저장·생략 이벤트 수도 기준선과 같다.

|창|기준선|결과|
|---|---|---|
|`qwen27b-regress` (s4-a02, 2차 `qwen27b-cache` 설정, `fa41fcff`)|r2-a01|20/20. T2 6/6, 재시작 후 T3 6/6(각 8,187). TTFT T1 13.7–13.9, T2 1.67–1.70, T3 2.22–2.30. T3 c0만 3.02(이전 2.39)로 재시작 직후 첫 복원. 이벤트 stored 6, min_step 13, byte_capacity 24|
|`qwen27b-boundary` (s4-a03, 5차 설정 그대로, snapshot 1 GiB, `aa89d2a1`)|r5-a01|11/11. T2 8,187, 재시작 후 T3 11,509 복원. TTFT T2 7.22–7.29(이전 7.39–7.43), T3 1.93–1.96(이전 1.96–2.23). 이벤트 stored 6, min_step 6, byte_capacity 21|

  - **health FAIL 2건:** 종료 3초 뒤 busy 4 %, KFD 0, 새 GPU 오류 0, fence 정상. 기존 캐시 창과 같은 양상이다. r5-a01 boundary는 PASS였다.
- **기준 정정:** 보관 snapshot의 `byte_capacity` 생략(regress 24·9건, boundary 21건)은 예산이 이미 찼거나(보유분 + 새 snapshot > 예산), 보고만 하는 snapshot 미달 때문에 생긴 것이다. disk·staging에서 checkpoint 1개가 예산을 넘어 생긴 생략은 0이다. 설계의 S4 기준을 이에 맞게 정정했다.

