# F1 KVarN 설계: Flash-Next k4v4 (K1–K3) (2026-10-30)

상위 문서: [F1-DESIGN.md](F1-DESIGN.md) 3.2, [F1-RESULTS.md](F1-RESULTS.md) "K0-FN". 표기: **[확인]**은 코드(file:line)나 측정으로 확인한 것, **[계산]**은 공식으로 계산한 것, **[짐작]**은 추론이다.

**결정(사용자 2026-10-30):** Flash-Next KVarN은 **k4v4**다. 근거는 K0-FN에서 KL 꼬리가 27B k4v2 수준이었다는 것이다(32K 최대 1.7e-3, needle 200K 최대 1.8e-3).

이 문서는 설계다. 코드는 아직 없다.

## 1. V 배치 결정: Flash-Next는 V를 token-major로 둔다

### 27B 형식 [확인]

- 레코드는 (128토큰 그룹, KV 헤드) 하나에 하나다(`src/models/qwen/hip/kvarn_layout.hpp:47-90`).
- K 코드는 token-major다. `[token 128][128 B]` 4비트이고, 32비트 워드 하나가 채널 8개를 담는다.
- V 코드는 channel-major다. `[channel 256][128 × b bit]`이고, 4비트에서 워드 하나가 토큰 8개를 담는다.
- scale/zero는 FP16이다. K는 채널마다 scale/zero, 토큰마다 column을 둔다. V는 토큰마다 scale/zero, 채널마다 column을 둔다.
- k4v4 레코드는 35,072 B다.
- 27B decode 커널(`attention_kvarn.hip:975-`)은 그룹 128토큰 전체를 WMMA 16토큰 단위로 촘촘히 훑는다. P·V에서는 토큰이 K축이다. 그래서 채널마다 토큰이 이어지는 channel-major가 B 조각을 바로 만든다.

### Flash-Next가 읽는 방식 [확인]

- **decode**(`Attention`, `kernels.hip.cpp` `AttentionKernel`): 선택된 4토큰 블록 약 512개(2048토큰)를 키 위치 목록으로 받는다(`lists`, `keys[]`). wave의 lane마다 키 행 하나의 8차원을 16바이트로 읽는다(`load_batch`). 행 단위, 토큰 단위로 읽는 것이다.
- **prefill WMMA**(`WmmaCausalAttention`): 16키 tile을 4토큰 블록 4개에서 모은다. K는 행을 그대로 LDS에 넣고, V는 키 행을 읽어 LDS에서 전치한다(`load_v`/`stage_v`). 역시 토큰 단위로 읽는다.
- 선택된 블록은 문맥 전체에 흩어져 있다. 그래서 128토큰 그룹 안에서 4토큰만 꺼내는 일이 흔하다.

### 비교 [계산, KV 헤드 하나, k4v4, 4토큰 블록 하나]

| V 배치 | V 코드 바이트 | 메모리 접근 |
|---|---:|---|
| channel-major (27B) | 256채널 × 2 B = 512 B | 64 B 간격의 2 B 조각 256개. 32–64 B 섹터로 읽으면 16–32배 증폭된다 |
| **token-major** | 4 × 128 B = 512 B | 연속 512 B. 한 wave가 행 하나(128 B)를 lane마다 4 B씩 읽는다 |

**결정: Flash-Next 레코드의 V 코드는 K와 같은 token-major로 둔다.**
- 형식: `[token 128][128 B]`, 워드 안의 니블 순서도 K와 같다.
- 레코드 크기는 27B k4v4와 같은 35,072 B다. 부분의 순서만 바뀐다.
- K와 V가 같은 모양이므로 dequant 경로 하나를 같이 쓴다.
- 27B 형식(channel-major)은 그대로 둔다. 레코드 배치는 `KvarnValueOrder{kChannelMajor, kTokenMajor}` 인자로 갈린다.

### 메타데이터 비용 [계산]

- 흩어진 선택에서는 그룹마다 메타데이터를 읽어야 한다: K scale/zero 1 KB, V column 512 B, 토큰 column과 scale.
- 최악의 경우(선택한 블록마다 그룹이 다름) 4토큰마다 약 1.5 KB 메타 + 1 KB 코드를 읽는다. FP16은 4 KB다. 그래서 대역폭 절감은 1.6배에 그친다.
- [짐작] selection 목록은 블록 번호 오름차순이고, 이웃 블록이 같은 그룹인 경우가 많다. 그래서 마지막 그룹의 메타를 레지스터/LDS에 두면 실제 비용은 더 작다.
- Flash-Next decode 속도는 2048토큰 선택에 묶여 있어 KV 읽기 절감의 이득이 원래 작다(F1-DESIGN 3.2). 이 단계의 목표는 메모리이고, 속도는 회귀가 없으면 된다.

## 2. 무엇을 양자화하나

| 부분 | 형식 | 이유 |
|---|---|---|
| trunk attention 12층 K/V | **KVarN k4v4** 레코드, 풀 plane(`rows_per_block = 2` 그룹) | 메모리의 대부분 |
| sink(그룹 0, 128토큰), 꼬리 ring(256토큰) | FP16, 세션 전용 | 27B와 같다(`kvarn_layout.hpp:29-35`). 세션당 384 × 1 KB × 2(K,V) × 12 ≈ 9.4 MiB [계산] |
| MTP draft 층 K/V | **FP16으로 남긴다(1단계)** | 27B에서는 draft가 target KV를 읽지 않아 대조할 근거가 없다. draft 수락률에 주는 영향을 먼저 재지 않고는 바꾸지 않는다. 13층 중 1층이다 |
| pooled indexer key, raw ring | FP16 / F32 그대로 | 블록 선택이 FP16 key에 그대로 묶인다(K0-FN 전제와 같다) |

**토큰당 바이트(MTP 세션) [계산]:**

| 구성 | 계산 | 토큰당 | FP16 대비 |
|---|---|---:|---:|
| FP16 | — | 27,460 B | 1.0× |
| k4v4, MTP 층 FP16 | 12 × 548 + MTP 2,048 + pooled 13 × 64 | 9,456 B | 2.9× |
| (참고) MTP 층도 k4v4 | 13 × 548 + pooled 832 | 7,956 B | 3.45× |

- 262K MTP 세션 하나: 7.01 GiB에서 **약 2.6 GiB**로 준다(sink/tail, ring, DeltaNet 포함).
- 이 수치로 F1-RESULTS의 bpw 표를 다시 계산했다. 지금 가중치로 262K 세션이 4개에서 약 11개가 되지만, 서버 상한은 8이다.

## 3. 쓰기, 회전, 양자화

### 프레임

- K/V는 쓸 때 256점 Hadamard(정규화, 부호 없음)로 회전한다. 27B와 같다(`attention_kvarn.hip:128-163`).
- Q는 attention 커널 안에서 회전한다.
- 출력 `Σ p·v_rot`에는 합 뒤에 H를 한 번 곱해 원래 프레임으로 되돌린다. H는 대칭이고 직교다. 그다음 gate를 곱한다.
- sink와 꼬리 FP16 행도 회전된 프레임으로 둔다. 그래야 attention 안에서 프레임이 하나다.

### 쓰기 경로 (Flash-Next에는 쓰기 커널이 셋이다)

- 쓰기 커널 셋은 그대로 둔다: `PrepareAttention`, `AttentionF16Gemm` epilogue, `StoreKv`.
- 이 커널들의 행 주소를 KVarN 모드에서는 FP16 sink/꼬리 ring 행으로 돌린다. P1의 `KvRow`처럼 위치 → 행 함수를 바꾼다.
- 그 직후 **`RotateKvRows` 커널**이 방금 쓴 n행(K, V × 2헤드)을 그 자리에서 FWHT256으로 회전한다. 위치는 device control에서 읽으므로 graph capture에 안전하다. 비용은 n × 4 KB 읽기·쓰기다.
- **이유:** 융합된 prefill epilogue 안에 FWHT를 넣으면 레지스터와 LDS 압력이 커진다. 그래서 1단계는 따로 돈다. [짐작] prefill에서 비용이 보이면 epilogue 융합을 2단계 최적화로 다룬다.

### 그룹 flush

- **`KvarnQuantize`**는 27B `KvarnQuantizeKernel<VBits>`(`attention_kvarn.hip:386`)를 KV 헤드 수와 V 배치로 템플릿화한다.
- 완성된 그룹(128토큰)의 회전된 FP16 꼬리 행을 레코드로 양자화하고, 그 ring 칸을 비운다.
- 알고리즘(Hadamard, balancing, RTN, FP16 scale)은 `kvarn_fake_quant.hpp`와 비트 단위로 같아야 한다(27B K1 기준).
- **flush 시점:**
  - 투기 batch는 아직 확정되지 않은 그룹을 끝낼 수 있다. 그래서 확정 위치가 (g+1)·128 이상이 된 뒤에만 그룹 g를 양자화한다.
  - 꼬리 ring 256은 꼬리 최대 127 + verify 8 + prefill pass를 담아야 한다.
  - prefill pass가 128을 넘으면 pass 안에서 완성되는 그룹을 다음 pass 전에 flush한다. 27B의 "pass ≤ RingTokens − 127" 규칙(`kvarn_layout.hpp:28-30`, `PlanKvarnPass`)을 그대로 쓴다.
  - Flash-Next 청크는 4096토큰이다. 그래서 KVarN 모드에서는 pass를 ring 한도로 나눈다. [짐작] 27B처럼 큰 prefill은 FP16 scratch에 쓰고 pass 뒤 일괄 양자화하는 경로(`PlanKvarnPrefill`)를 옮긴다. 그 비용은 측정으로 정한다.

## 4. 읽기 (attention 커널)

**위치 p의 키:**
- p < 128이면 sink 행이다.
- 그렇지 않고 그룹 g = p >> 7이 양자화되었으면 레코드다. 레코드 위치 = `(blocks[p >> 8] · 2 + (g & 1)) · kv_heads + kv`이고, 토큰은 p & 127이다. 27B `RecordOffset`(`attention_kvarn.hip:58-65`)과 같은 식이다.
- 아니면 꼬리 ring 행이다.

**커널별 변경:**

| 커널 | 변경 |
|---|---|
| `Attention` decode/verify (8행 이하) | KVarN 변형을 만든다. Q를 LDS에서 FWHT256으로 회전한다. `load_batch`는 위의 세 경로 중 하나에서 lane마다 8채널을 만든다(레코드에서는 4 B 코드 + 그 그룹의 채널 scale/zero/column). 마지막 그룹의 메타를 캐시한다. 출력은 split merge(`AttentionMergeKernel`)나 단일 split 끝에서 H를 곱해 되돌린 뒤 gate를 곱한다 |
| `WmmaCausalAttention` prefill dense/sparse | `load_k`/`load_v`가 같은 세 경로로 FP16 tile을 LDS에 만든다. 레코드 dequant(16키 × 256)는 tile마다 한 번이다. Q 회전과 출력 역회전은 prefill에서 두 번의 FWHT256 pass다(`PrepareHalfInput` 전후). [짐작] 비용은 작다 |
| `PoolIndexerBlocks`, `SelectBlocks` | 바꾸지 않는다(indexer는 FP16) |
| MTP 경로 | 바꾸지 않는다(FP16 층) |

graph capture는 P1과 같다. 블록 표와 그룹 frontier는 device에서 읽는다. 27B는 split-K decode를 128토큰 아래에서만 캡처하지만, Flash-Next는 깊이와 무관하게 캡처하므로 frontier를 control block에 둔다.

## 5. 공통화 (`src/core/`)

| 옮길 것 | 지금 | 바꿀 점 |
|---|---|---|
| 그룹·sink·ring 상수, `KvarnFp16Row`, `PlanKvarnPass`, `PlanKvarnPrefill`, `KvarnReadGroups`, checkpoint 행 규칙 | `src/models/qwen/hip/kvarn_layout.hpp:166-316`, `kvarn_checkpoint.hpp` | KV 헤드 수를 인자로 받는다(호스트만). 27B 호출은 같은 값을 넘긴다 |
| 레코드 배치 `KvarnRecord(bits, order)` | `kvarn_layout.hpp:59-90` | V 배치 인자를 더한다. 27B는 channel-major로 비트 동일 |
| `Fwht256` device helper, 양자화 커널 | `attention_kvarn.hip:128-163`, `386-` | KV 헤드와 V 배치를 템플릿 인자로 받는다. 27B 인스턴스는 지금 코드와 같은 산술 |
| fake quant 참조 | `tests/models/qwen27b/kvarn_fake_quant.hpp` | `tests/core/`로 옮기거나 그대로 include한다(K0-FN은 include한다) |

attention 커널은 모델 전용이다. 27B는 dense 그룹 WMMA, Flash-Next는 sparse 행 gather다. 그래서 공통화하지 않는다.

**27B 회귀 방지:** 27B KVarN 테스트(`qwen_attention_kvarn_*`, `qwen27b_kvarn_k0 --real`)의 출력이 비트 단위로 같아야 한다. 공통화 커밋마다 확인한다.

## 6. 세션, 스냅샷, 캐시

- **할당:** P1 블록 span에 KVarN 레코드 plane(12층)을 두고, FP16 K/V plane은 MTP 층 하나만 둔다. pooled plane은 그대로다. sink/꼬리는 세션 전용 FP16이다.
- **CLI:** `src/cli/kv_cache_type.cpp:22-23`의 `qwen4exp`에 `kvarn-k4v4`를 허용한다. 기본은 FP16 그대로다. 사용자가 고르게 하고, 기본값 변경은 속도·품질 판정 뒤 감독/사용자가 정한다.
- **되감기:** 꼬리 ring 행만 영향을 받는다. 확정 전 그룹은 양자화하지 않으므로(3절) 레코드를 되돌릴 일이 없다.
- **스냅샷(legacy payload):**
  - KVarN 세션에는 새 영역을 둔다: 레코드(완성 그룹), sink, 꼬리 행.
  - payload 버전을 올리고(`kSnapshotPayloadVersion`), FP16 세션 형식은 그대로 둔다.
  - 헤더에 KV 형식을 기록하고, 형식이 다른 스냅샷은 거절한다.
- **continuation adapter:** 레코드 component(그룹 단위 append rows)와 sink/꼬리 private state를 추가한다(27B adapter 선례).
- **디스크 캐시 identity:** KV 형식을 포함한다.

## 7. 구현 단계, 통과 기준, 예상 크기

| 단계 | 내용 | 통과 기준 | 크기 [짐작] |
|---|---|---|---|
| **K1a** 공통화 | 호스트 planner와 레코드 배치를 `src/core/`로 옮기고, KV 헤드와 V 배치를 인자로 받게 한다. 27B 호출을 옮긴다 | 27B KVarN op 테스트와 K0 `--real` 출력이 비트 단위로 같다. 27B 서버 greedy 동일 | 300–500줄 |
| **K1b** 쓰기·양자화 | `RotateKvRows`, 템플릿 `KvarnQuantize`(2헤드, token-major V), Flash-Next 쓰기 주소 전환, flush 시점 | op 테스트: 레코드가 `kvarn_fake_quant.hpp`와 비트 단위로 같다(코드, scale), 회전 행이 FP64 FWHT와 맞는다 | 500–800줄 |
| **K2** 읽기 | decode/verify KVarN 변형, WMMA prefill dense/sparse 변형, 출력 역회전 | op 테스트: fake 복원 FP16 행(같은 회전 프레임) 대비 출력 오차가 27B K1 기준(rounding 대조의 10배 이내)이다. 블록 표 섞기 비트 동일(P1 방식) | 800–1,200줄 |
| **K3** 통합 | 세션 할당, CLI, 스냅샷/adapter/디스크 캐시, `qwen38_flash_next_kvarn_k0 --real` | real‖fake KL ≤ 측정한 rounding 대조의 10배, top-1 차이 1/100 이내(27B `KVARN.md:228,267`). K0-FN 입력(32K, 256K, needle 32K/200K) gold와 needle 정답. `long-context`, `cache` 통과. 메모리: 262K MTP 세션 약 2.6 GiB(측정으로 확인). 속도: FP16 대비 요청·단계별 5 %·10 ms(decode d0/32K, prefill pp2048 d0/32K)이고, 회귀가 있으면 원인을 보고한다 | 600–900줄 |

**순서:** K1a → K1b → K2 → K3. 단계마다 감독 승인을 받고, 창은 단계마다 1–2개다.

**P2(공유 풀)와의 관계:** K1–K3은 P1의 세션 전용 span 위에서 만든다. P2는 FP16과 KVarN plane 모두에 같은 방식으로 붙는다. 순서는 감독이 정한다. W9의 `engine.*` 작업이 끝나야 하는 P2보다 K1a/K1b를 먼저 할 수 있다.

## 8. 열린 질문 / 위험

1. MTP 층을 FP16으로 두는 것(토큰당 2 KB 더 쓴다). K3 뒤 draft 수락률 영향을 재고 나서 바꿀지 정한다.
2. [짐작] 흩어진 선택에서 그룹 메타데이터 읽기가 decode 시간에 보일 수 있다. 메타 캐시로 줄이고, ABBA로 판정한다.
3. 큰 prefill 청크(4096)와 ring 256 사이의 pass 분할 비용은 측정한다.
4. 27B와 형식이 갈라지므로(V 배치) 레코드 형식 이름을 따로 둔다. 디스크 캐시 identity에 포함한다.
