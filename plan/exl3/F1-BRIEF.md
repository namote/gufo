# F1 브리프: Flash-Next n-gram 디스크 배치(F0), 페이지 KV, KVarN (2026-10-30)

공통 규칙은 [W-COMMON.md](../rotated-quant/W-COMMON.md)를 따른다. 계획은 [PLAN.md](PLAN.md) F1이다. 형식(EXL3)과는 독립이다.

## 배경 (감독 확인)

- **페이지 KV:** 공통 코드 `src/core/hip/kv_block_pool.{hpp,cpp}`, `kv_block_table.hpp`, `kv_block_layout.hpp`가 있다. 모델에 묶여 있지 않고, 호출자가 행을 plane으로 설명하는 구조다.
  - 지금은 27B(`src/models/qwen/`)만 쓴다. executor, decode/batched decode, prefill, DFlash2, MTP, attention 커널 전부가 해당한다. serve 쪽은 `src/cli/serve/inference_backend.cpp`다.
- **Flash-Next KV:** `src/models/qwen38_flash_next/`는 세션별 연속 `ContinuationLayout`을 쓴다.
  - KV는 FP16이고, MTP를 포함해 토큰당 약 27.5 KiB다.
  - admission이 설정된 용량 전체를 선점한다.
  - layout에 압축 비율(`ratio_`)이 있다. 어떤 상태인지는 아직 확인하지 않았다.
- **KVarN:** 27B 커널 `src/models/qwen/hip/kernels/attention_kvarn.hip`이 있다. 256점 Hadamard를 쓰고 부호는 없다. 기준은 vLLM `kvarn_store.py`, 참조는 `tests/models/qwen27b/kvarn_fake_quant.hpp`다. 27B 문서의 KVarN 품질 기준(K0)도 확인한다.

## F0 (사용자 지시 2026-10-30, 가장 먼저)

- n-gram 표를 GPU 상주에서 빼고 디스크(mmap, 필요한 행만 읽기)로 둔다. Kyojin과 upstream EXL3도 `ngram_embedding`을 파일에서 읽는다. 먼저 gufo `src/models/qwen38_flash_next/ngram.{hpp,cpp}`와 weights 로드를 읽고, 지금 어디에 얼마를 두는지 확인한다.
- 그다음 메모리 점유(가중치 종류별, n-gram, KV, MTP, 작업 버퍼, 세션 수별 증가)를 표로 측정한다. 이 결과로 사용자가 Flash-Next EXL3 bpw를 정한다.
- 통과 기준: greedy 출력이 바이트 단위로 같다. decode와 prefill이 main 대비 5 %·10 ms 안이다. 페이지 캐시 cold/warm을 기록한다.

## 1단계: 설계 (읽기만, 허락 불필요)

0. F0 설계: n-gram 조회 경로(토큰마다 몇 행, 어느 시점), 디스크 읽기 방식(mmap과 prefetch, 호스트 핀 메모리 스테이징), 지연 숨기기, 측정 방법

1. **Flash-Next KV 구조:**
   - attention 종류(full / sparse / indexer 등)와 head_dim, KV 행 형식
   - 압축 상태(`ratio_`)
   - MTP 되감기, 스냅샷, 디스크 캐시와 component cache가 KV를 다루는 방식
   - attention 커널 목록(fused WMMA, 토큰별 커널 등)
2. **페이지 KV 적용 설계:**
   - 어떤 plane을 풀에 넣을지
   - 커널이 `KvBlockTable`로 행을 찾게 바꿀 곳
   - 되감기, 스냅샷, 디스크 캐시의 블록 단위 처리
   - admission 변경(쓴 만큼 할당)
   - 27B가 이미 푼 문제(분기, graph capture)를 재사용할 수 있는지
3. **KVarN 적용 설계:**
   - 27B KVarN 커널과 참조를 공통으로 옮길 수 있는지(`src/core/`)
   - Flash-Next head_dim과 attention 종류에 맞는지
   - sparse/indexer 경로와 어떻게 맞물리는지
   - 품질 기준과 측정 방법(R0 하네스, needle)
4. **충돌 확인:** W9가 `src/models/qwen38_flash_next/engine.cpp`, `mtp_policy.hpp`를 고치고 있다(I1, 판정 중). 파일 충돌과 순서를 적는다.
5. 위 내용을 `plan/exl3/F1-DESIGN.md`에 쓰고 `WAITING: 구현 승인`으로 보고한다. 구현 단계와 통과 기준, 예상 크기를 함께 적는다.

## 통과 기준 (구현 단계, 제안을 다듬어 확정)

- **페이지 KV:** greedy 출력이 지금(연속 FP16)과 바이트 단위로 같다. 기능 테스트 `long-context`, `cache`를 통과한다. 다중 세션 메모리 이득을 측정한다. main과 짝지은 타이밍에서 요청·단계별 5 %·10 ms 회귀가 없다.
- **KVarN:** FP16 KV 대비 KL, needle, 긴 문맥 품질이 27B K0 기준 방식을 따른다. 속도와 메모리를 함께 본다.

## 파일 경계 (1단계)

- 쓸 수 있는 파일은 `plan/exl3/F1-DESIGN.md` 하나다. 코드, 빌드, GPU는 쓰지 않는다.
