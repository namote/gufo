# gufo paged KV 공통 API 검토 (U2 전, 2026-10-23)

상위 문서: [이식 계획](PLAN.md), [모델 확장](MODELS.md). 사용자 지시: "오케이 추천안대로 가자, 검토도~"(D5 = 256행 먼저, U2 전에 풀·테이블 API 검토).

이 문서는 설계 검토다. 코드는 바꾸지 않았다.

**검토 대상:**
- 우리 P1 API: `KvBlockPool`, `KvBlockTable`, `KvBlockReservation`(gufo `0296c09c`, `src/core/hip/kv_block_*`)

**대조한 모델 사례:**

| 사례 | 저장소·기준 |
|---|---|
| Qwen 27B | 원본 `79c2dcc5` |
| Qwen3.8 Flash-Next | 원본 `79c2dcc5` |
| Gemma 4 26B-A4B·31B | 포크 `feat/gemma` `21aad161` |
| DeepSeek V4 Flash | 원본 `79c2dcc5`, 아직 옛 snapshot 경로 |

줄 번호는 위 기준 커밋을 따른다.

## 결론

- P1 API의 **할당·참조 수·예약·CoW 부분**(`Allocate`·`Retain`·`Release`·`TryReserve`·`KvBlockReservation`)은 모델을 모르는 형태라 그대로 쓴다.
- **행 기하**(풀과 테이블이 행을 어떻게 놓는가)는 세 모델을 표현하지 못한다. U2에서 아래 "바꿀 것" 1–4를 반영해 일반화한 뒤 Qwen 27B에 붙인다.
- **D5(블록 = 256토큰)는 검토로도 맞다.**
  - DeepSeek ratio-128 층은 압축 행 하나가 128토큰을 덮는다. 블록이 64토큰이면 블록마다 행이 0.5개라 블록으로 나눌 수 없다.
  - 256토큰이면 모든 사례에서 블록마다 행 수가 정수이고 2의 거듭제곱이다.
  - 원본 chunk(256행)와도 K/V에서 1:1이다.

## 모델별 상태 계약 (조사 요약)

| 모델 | paged 후보(append·불변) | 토큰당 행 | 행 바이트 | 층 수 | 256토큰 블록당 행 | 고유로 둘 것 |
|---|---|---|---|---|---|---|
| Qwen 27B | K, V | 1 | 2,048 / 2,048(FP16) | 16 | 256 | recurrent, conv, logits |
| Flash-Next | K, V | 1 | 1,024 / 1,024 | 12 | 256 | raw keys ring(2048–4096행), recurrent, conv, PLE, MTP |
| | pooled keys | 1/4 | 256 | 12 | 64 | |
| Gemma 26B-A4B | global K(파생 키), global V | 1 | **512 / 2,048** | 5 | 256 | sliding K/V ring(3,328행), hidden, logits |
| Gemma 31B | global K, V | 1 | **1,024 / 4,096** | 10 | 256 | 같음 |
| DeepSeek V4 Flash | ratio-4 압축 KV(f32) | 1/4 | 2,048 | 21 | 64 | raw ring(128 live), compressor 상태, DSpark |
| | ratio-4 f16 거울 | 1/4 | 1,024 | 21 | 64 | (f32에서 다시 만들 수 있음) |
| | ratio-4 indexer 키 | 1/4 | 512 | 21 | 64 | |
| | ratio-128 압축 KV | 1/128 | 2,048 | 20 | 2 | |

토큰당 paged 바이트와 256토큰 블록 하나의 크기(모든 plane 합):

| 모델 | 토큰당 | 블록당 | 최대 문맥 세션 하나 |
|---|---|---|---|
| Qwen 27B | 64 KiB | 16 MiB | 256K에 16 GiB |
| Flash-Next | 25,344 B | 6.2 MiB | 256K에 6.2 GiB(MTP 별도 약 0.5 GiB) |
| Gemma 26B | 12,800 B | 3.1 MiB | 128K에 1.56 GiB |
| Gemma 31B | 51,200 B | 12.5 MiB | 128K에 6.25 GiB |
| DeepSeek | 약 19,136 B | 4.7 MiB | 256K에 약 5 GiB. 가중치 80.76 GiB와 같은 메모리를 나눠 씀 |

## 지금 P1 API가 가정한 것과 어긋나는 점

1. **K와 V의 행 크기가 같다고 가정한다.**
   - P1: `KvBlockPool(layers, row_bytes, blocks)`, plane 2개(K·V)가 같은 `row_bytes`
   - 어긋나는 사례:
     - Gemma global 층은 V가 없는 층에서 K를 파생 형태로 저장해 K 512 B, V 2,048 B다(포크 `executor.cpp:228-241`).
     - Flash-Next는 K/V 1,024 B 옆에 pooled keys 256 B가 붙는다.
     - DeepSeek은 plane 4종이다.
2. **plane이 모두 토큰당 한 행이라고 가정한다.**
   - P1: `KvRowInLayer`가 `pos >> 6`으로 블록을 찾는다.
   - 어긋나는 사례: Flash-Next pooled keys는 4토큰당 한 행, DeepSeek은 4·128토큰당 한 행이다.
3. **plane마다 같은 층 수를 가정한다.** DeepSeek은 ratio-4 층(21)과 ratio-128 층(20)이 다르다.
4. **블록 크기가 64토큰으로 상수 고정이다.** `kKvBlockTokens = 64`, `kKvBlockShift = 6`.
5. **주소 폭:**
   - Gemma `RowSplitAttentionKernel`과 WMMA global 변형은 **32비트 바이트 오프셋**을 쓰고, 캐시가 4 GiB를 넘으면 거절한다(포크 `kernels.hip.cpp:1488-1495`, `attention_wmma.hip.cpp:78`).
   - 공유 풀 하나에 모든 세션을 담으면 층 하나의 영역이 커진다.

## 바꿀 것 (U2 설계에 반영)

1. **plane 목록으로 풀을 만든다.**
   - 형태: `PagedRowPool(planes, blocks)`. 각 plane은 `{layers, row_bytes, rows_per_block}`이다.
   - 메모리 배치: plane마다 `[layer][block][rows_per_block][row_bytes]`(P1과 같은 layer-major)
   - 블록 id 하나가 모든 plane의 같은 블록 자리를 가리킨다.
   - 그래서 세션은 테이블 **하나**만 갖고, 예약·CoW·공유도 블록 id 하나로 모든 plane에 함께 적용된다.
   - 각 모델에 맞추는 방법:
     - Qwen 27B: plane 2개
     - Flash-Next: plane 3개
     - Gemma: plane 2개(K·V 행 크기가 다름)
     - DeepSeek: plane 4개
2. **블록 = 256토큰을 풀 상수로 둔다.**
   - plane의 `rows_per_block = 256 / 토큰당 행 비율`이다(256, 64, 2).
   - 생성할 때 2의 거듭제곱인지 검사하고, 아니면 시작을 거부한다.
3. **커널 주소 도우미를 plane 기준으로 바꾼다.**
   - `PagedRow(table, plane_layer_base, logical_row, rows_shift)`
   - = `plane_layer_base + (table[logical_row >> rows_shift] << rows_shift) + (logical_row & mask)`
   - 모든 계산은 64비트로 한다.
   - 32비트 오프셋을 쓰는 커널(Gemma)은 층별 base 포인터를 받고, 층 안 오프셋만 32비트로 둔다. 층 하나의 plane 영역이 4 GiB를 넘는 풀 크기는 그 모델에서 시작할 때 거부한다.
   - 거부가 실제로 걸리는 크기: Gemma V 2,048 B 행 기준으로 층당 2M토큰이 넘을 때다. 운영 크기(세션 4 × 128K = 512K토큰)에서는 걸리지 않는다.
4. **gather 경로에 테이블을 꽂는 자리:**
   - Flash-Next QSA: 4토큰 블록 선택. 블록 256토큰 안에 정렬된다.
   - DeepSeek: indexed attention의 `comp_rows`
   - 둘 다 이미 행마다 간접 주소를 쓰므로, 같은 자리에서 테이블을 한 번 더 거친다.
   - 연속으로 읽는 경로(dense scan, WMMA prefill)는 블록 경계마다 base를 바꾼다. 256행이면 경계가 드물다.
5. **ring과 private 상태는 풀 밖이다.** 대상: Gemma sliding, Flash-Next raw keys, DeepSeek raw ring·compressor 상태·DSpark, recurrent·conv. 원칙 그대로다.
6. **원본 cache 계층과의 연결(U3·U4에서 확정):**
   - 원본 `ComponentDescriptor`는 (part, layer)마다 `row_bytes`와 `rows_per_chunk = 256`을 가진다(`src/cache/types.hpp:33-40`, Qwen `continuation_layout.cpp:41`).
   - 추가 정보: descriptor에 "paged plane·층" 정보를 더한다. 그러면 cache 계층이 component를 풀 plane에 대응시킬 수 있다.
   - chunk와 블록의 대응:
     - K/V chunk = 블록 1개
     - Flash-Next pooled chunk(256행) = 블록 4개
     - DeepSeek은 카드 17이 정할 chunk 기하를 따른다.
   - 빌려주기: 원본 `BorrowRowsView`는 연속 span 하나를 돌려준다. 그래서 paged component에는 바이트를 빌려주는 대신 **블록 참조(Retain)를 넘기는 경로**를 새로 둔다.
   - 이 경로가 Gemma의 "capture는 복사"(포크 `continuation_adapter.hpp:49-54`)를 없애는 주된 이득이 된다.

## 모델별로 블록 테이블이 필요한 커널 (적용 단계 범위 추정용)

| 모델 | 커널 |
|---|---|
| Qwen 27B (U2) | P1에서 이미 바꾼 KV 쓰기·decode·batched decode·prefill·검증 경로를 원본 위로 옮긴다. 원본 prefill attention·검증 커널 주소 방식은 U2 시작 때 확인(대응표 "확인하지 못한 것") |
| Flash-Next (U7) | `StoreKvKernel`, `PrepareAttentionKernel`, `AttentionF16Gemm` epilogue, `AttentionKernel`, `WmmaCausalAttentionKernel`, `SelectScoreKernel`, `PoolBlocksKernel`(pooled 쓰기). batched decode·MTP는 같은 커널을 세션마다 부른다 |
| Gemma 4 (U8) | `QkvPostKernel`(쓰기), `AttentionKernel`(fallback), `RowSplitAttentionKernel`(decode·MTP 검증·drafter), `WmmaPrefillAttentionKernel`(`kGlobal` 변형 포함), `MoveKeyKernel`, engine snapshot 복사·adapter `RowPieces` |
| DeepSeek (U9) | 압축기 쓰기, indexed attention(`comp_rows` gather), ratio-128 dense scan, indexer scoring 전체 스캔. 지금 버퍼를 두 배로 늘려 다시 할당하는 방식(`rocm_graph.cpp:616-680`)을 풀 할당으로 바꾼다 |

## 남는 위험과 미정

- **DeepSeek ratio-4 압축 행:** 앞 4토큰을 겹쳐 읽는다(compressor 상태에 들어 있음). 그래서 블록 경계 checkpoint에서는 compressor 상태를 private로 함께 저장해야 한다. 구조상 가능해 보이지만 카드 17(원본, proposed)이 chunk 기하를 정하기 전까지는 가능성으로만 둔다(결정 D9).
- **Flash-Next 메모리 이득:** 세션당 6.2 GiB로 Qwen 27B보다 작다. 이득의 중심은 세션 밀도보다 블록 공유와 복원 쪽일 수 있다(MODELS.md "측정할 것").
- **Gemma 26B 세션 밀도 이득:** 작다(128K에 1.56 GiB). 주된 이득은 capture 복사 제거다.
- **블록 256토큰의 비용:**
  - 세션 꼬리의 빈 행: Qwen 27B에서 평균 8 MiB, 최대 16 MiB
  - 공유 꼬리 블록의 CoW 복사: 16 MiB, GPU에서 약 0.1 ms 수준
  - U2·U3에서 bench와 메모리로 확인하고, 손실이 크면 계획대로 64행 + 여러 조각 borrow view로 돌아간다(D5 보조안).
- **원본 반응:** 원본은 "같은 토큰만으로는 공유하지 않는다"는 계보 원칙을 둔다. 블록 공유도 계보 안에서만 하도록 설계한다(D4와 같음).

## U2에 넘기는 할 일

1. `PagedRowPool`·테이블·주소 도우미를 plane 목록과 256토큰 블록으로 일반화한다. P1 단위 테스트를 plane 2개·3개·4개, 행 비율 1·4·128, 서로 다른 행 크기, 층 수가 다른 plane으로 넓힌다(T0).
2. Qwen 27B는 plane 2개로 붙인다. 원본 component descriptor에 plane·층 대응을 더한다.
3. T1은 plane 기준 주소로, 위치 200K 이상과 블록 경계 앞뒤를 bit 비교한다.
