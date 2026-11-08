# gufo 원본 이식 U2: 블록 테이블 커널 + plane 일반 KV 블록 풀 (2026-10-23)

상위 문서: [이식 계획](PLAN.md), [API 검토](API-REVIEW.md), [U1 결과](U1-RESULTS.md). 사용자 지시: "넹"(U2 시작).

## 구현 (gufo `ai-tools/port-u2`, U1 `14bcd820` 위 커밋 3개, 최종 `7daf08d4`)

|커밋|내용|
|---|---|
|`ff5aeb7a`|P1 블록 테이블 커널을 원본 위로 옮기고, 풀을 [API 검토](API-REVIEW.md)대로 일반화|
|`11e583ff`|테스트 수정: 저장 계산 테스트가 FP16 KV를 256 블록 단위로 센다(P1 `c1a6933c`)|
|`7daf08d4`|수정: 옛 snapshot payload 계산이 빈 행을 블록 올림 기준으로 뺀다(T1 `kv_storage` 테스트가 찾음)|

- **풀 일반화:**
  - `src/core/hip/kv_block_layout.hpp`는 HIP 없이 CPU에서 검사할 수 있다.
    - `KvBlockPlane{layers, row_bytes, rows_per_block}`
    - `KvBlockGeometry`: plane 목록, layer-major 배치, 64비트 오프셋, 크기 overflow 검사
    - `KvBlockAllocator`: 낮은 번호부터 할당, 참조 수, Retain/Release
  - `KvBlockPool`은 기하·할당 위에 장치 메모리만 더한다.
  - 블록 = 256토큰(`kKvBlockTokens`, 결정 D5)이다. 블록 id 하나가 모든 plane을 가리킨다.
- **커널:** P1의 KV 쓰기·decode·graph decode·batched·검증(wave64)·tile·WMMA prefill·fused RoPE 쓰기를 `KvBlockTable`로 옮겼다.
  - P1 커널 파일은 충돌 없이 합쳐졌다.
  - P1 이후 원본이 넣은 16K 초과 online batched 경로(`attention_batched.hip`)에도 테이블을 넘긴다.
- **arena:**
  - FP16 KV는 세션 전용 풀(K·V plane 2개)에서 받는다. 블록 0..n−1이 연속이다.
  - 그래서 동작은 원본과 같고, 층 간격만 256행 단위로 올림된다.
  - 옛 snapshot 저장·복원·compact 복원은 층 간격과 plane base를 arena에서 받는다.
  - FP32 참조 KV는 원본 그대로다(테이블 없음).
- **어댑터:**
  - `BorrowRowsView`는 행이 연속 블록 하나에 있을 때만 span을 준다. 아니면 빈 view를 돌려 `TransferRows`로 가게 한다. 빈 view는 원본 FP32 경로가 이미 쓰는 계약이다.
  - `TransferRows`는 블록 run마다 복사한다.
- **P1에서 가져오지 않은 것:** 우리 B 단계(bounce stream·compact segment) 문맥과 외부 공유 풀을 받는 arena 생성자. 둘 다 원본에 없거나 U3 범위다.
- **테스트:**
  - CPU `kv_block_layout_test`(T0):
    - plane 2·3·4개, 행 비율 1·4·128
    - K/V 폭이 다른 경우, 층 수가 다른 plane
    - 256K 산수(16 GiB, 마지막 행 64비트), 잘못된 입력 거부, 할당자 참조 수
  - GPU `kv_block_table_ops_test`(T1):
    - 4,096 문맥을 뒤섞인 블록 16개에 둔 기존 검사
    - 새 검사: 262,144 문맥 한 층, 위치 200,000·204,799·204,800·262,143에서 decode 읽기·쓰기
    - 새 검사: 230,140부터 검증 8행
    - 새 검사: online batched 8행(블록 경계 230,144 포함)
    - 각 경우 연속 배치와 출력 bit, 쓴 행 위치가 같아야 한다.
- **패치 보존:** `test/experiments/gufo_upstream_port_20261023/patches/u2/`(3개)

## 검증 (PLAN-U2.json)

| 계층 | 결과 |
|---|---|
| T0 | 릴리스·gpu-test 전체 빌드 성공. 포맷 PASS(clang-format 21.1.8). CPU ctest 127/127(새 `kv_block_layout_test` 포함) |
| T1 커널(`u2-kernels-b01`, 최종 빌드) | 10종 통과: block table(큰 위치 포함), kv storage, decode, long context, WMMA prefill, fusion, component, projection, graph, L2 통합 |
| 모델 단위(`u2-models-b01`) | 5종 통과: 어댑터(공유 scratch 경우 포함), 어댑터 `--draft`, target, DFlash2, `inference_backend` |
| T2(`u2-t2-b01`) | 21요청 텍스트가 U1과 전부 같고 gold 21/21. TTFT −1.0 ~ +2.3 %. 입장 4+1 대기 그대로 |
| T3(`u2-t3-a01`) | U1 대비 모든 측정점 −0.5 % 안. AR pp2048 647.4(−0.3 %), tg 12.91(−0.3 %). DFlash2 pp 595.9(−0.5 %), tg @ d8192 60.68(−0.4 %) |
| T4s(`u2-t4s-*-a01`) | 7창의 텍스트가 U0 기준과 전부 같다: density4 8/8, density8 16/16, long16k 5/5, two8k 8/8, regress 20/20, boundary 11/11, shared 19/19. gold 전부 정답 |
| 기동 메모리 | U1과 같다. 16K × 4 87,478 MiB, 64K × 8 54,951 MiB. 블록 올림 영향 없음(16K·64K는 256의 배수) |

### 기록할 것

- **u2-kernels-a01 실패(`ff5aeb7a`):** `kv_storage_ops_test`가 실패했다.
  - 첫 실패: 테스트의 계산 기대값이 64→256 블록 올림을 반영하지 않았다. `11e583ff`로 고쳤다.
  - 이어서 드러난 것: 실제 코드 결함이다. 옛 snapshot payload 계산이 빈 행을 `max_context`로 셌다. `7daf08d4`로 고쳤다.
  - 이 값은 옛 snapshot 경로에서만 쓰이고, 서빙(component cache)은 쓰지 않는다. 그래도 테스트·모델·T2를 최종 빌드로 다시 돌렸다(b01).
- **T4s shared 첫 실행의 차이:** U2 a01에서 팔로워 c2·c3가 공유 system checkpoint를 찾지 못해 TTFT가 2.3 → 14.6 s였다.
  - 확인 방법: U0와 U2를 두 번씩 번갈아 다시 돌렸다.
  - 결과: **원본(U0 a02)에서도 같은 일이 났다.** U2 a02·a03은 정상(c2 RAM 재사용)이었다. 원본 동작이며 우리 변경 탓이 아니다.
  - 원인 추정: RAM 예산 1 GiB 안에 checkpoint 715 MB가 비동기로 게시·축출되는 시점이 갈린다.
  - 이 RAM checkpoint 게시 경합은 U5 재측정 항목에 넣는다. 재현 조건이 정리되면 원본 이슈 후보다.
- **u2-t3-a01:** 창 판정이 `FAILED_no_fallback`이다. 원인은 끝난 뒤 GPU busy 4 %다. 이전과 같은 패턴이고 guard는 완화하지 않았다.
- **최종 빌드에서 다시 돌리지 않은 것:** T3·T4s·기동 메모리는 `ff5aeb7a`의 결과다. 최종 빌드와 다른 줄은 옛 snapshot 계산 한 줄뿐이다.

## 판정

- 커널을 테이블 주소로 바꾸고 256토큰 블록으로 만든 뒤에도 출력·텍스트가 같다.
- bench 손실은 0.5 % 안이다. P1의 64행 블록 때(−0.7 ~ −2 %)보다 작다.
- 큰 위치(200K 이상)와 블록 경계도 bit 동일이다.
- **U2 통과.** D5(256행)도 bench·메모리로 유지한다.
- 증거: `storage/evaluations/gufo-upstream-port-u2-20261023/`

## 다음 (U3)

- 슬롯 executor가 공유 풀에서 블록을 필요할 때 받는다.
- 원본 checkpoint의 행 빌림을 블록 참조(Retain)로 바꾸고, 덮어쓰기 보존을 참조 수 확인 뒤 CoW로 바꾼다. paged component의 블록 참조 경로는 [API 검토](API-REVIEW.md) "바꿀 것" 6이다.
- 슬롯 선택(`SlotBusy`)에 블록 예약을 더한다.
- 결정 D6은 절충안으로 정해졌다(PLAN 결정표): 값이 없으면 자동 크기, 명시하면 그 값 그대로. 우리 운영은 명시한다. 자동 산식은 U3 설계에서 정한다.
- 운영 전환 제안 직전에 최종점검 F를 돈다.
