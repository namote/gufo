# F1 결과: Flash-Next F0 측정, 페이지 KV(P1), KVarN K0 (2026-10-30)

상위 문서: [F1-DESIGN.md](F1-DESIGN.md), [F1-BRIEF.md](F1-BRIEF.md). 표기: **[확인]**은 측정·빌드·코드로 확인한 것, **[짐작]**은 추론이다.

## 상태

- 감독 결정(2026-10-30): F0은 측정만, n-gram 표는 F2 범위에서 뺀다, 이미지 선점은 P2에 별도 커밋, MTP 층은 같은 plane, KVarN V 배치는 K0-FN 뒤에 정한다.
- **F0:** 창 계획 완료, GPU 순서(XQ → X-A 참조 덤프 → F1 F0)에서 감독의 "start"를 기다린다.
- **P1, K0-FN:** 코드와 빌드 완료. GPU 창은 아직 돌리지 않았다. 이미지를 만들 디스크 여유가 필요하다(아래 "디스크").

## 브랜치와 커밋 [확인]

다른 에이전트와 같은 작업 트리를 쓰므로(flake 빌드는 dirty tracked 파일을 포함한다) 별도 worktree에서 작업했다.

- worktree `build/f1-wt`, 브랜치 `ai-tools/f1-paged-kv`, 기준 `d2b751a9`(W9 I1 포함 HEAD)
- `b75a0051` refactor(qwen38_flash_next): address attention K/V and pooled indexer rows through a session block table
  - 커널 7개 경로(`PrepareAttention`, `AttentionF16Gemm` epilogue, `StoreKv`, `Attention`, `WmmaCausalAttention`, `PoolIndexerBlocks`, `SelectBlocks`)가 nullable `kv_blocks` 표로 행을 찾는다(`kernels.hip.cpp` `KvRow`/`PooledRow`, 공통 `KvRowInLayer`). null이면 지금과 같은 연속 배치다.
  - 세션은 K, V, pooled key(13층, MTP 포함) 를 `hip::KvBlockGeometry` 한 span에 두고 identity 표를 쓴다. 할당은 기존 `hip::AllocateDevice`(managed fallback)를 그대로 쓴다. 그래서 층마다 행이 위치 순서로 연속이고, 스냅샷·checkpoint·adapter 영역 코드는 바뀌지 않는다.
  - pooled 용량은 `max_context/4 + 1`행을 덮도록 `BlocksFor(max_context + 4)` 블록으로 잡는다(adapter guard가 그 범위를 쓴다, `continuation_adapter.cpp:159,169`).
  - `SessionBytes`는 새 할당(블록 단위 반올림 + 표)과 같은 식으로 바꿨다.
  - `attention_ops_test`에 `CheckPagedRows`: 섞은 블록 표(풀 2 b + 3 블록, 역순 홀수 보폭)로 위 커널을 돌려 연속 배치와 비트 단위로 비교한다(decode walk/list × split 0/8, WMMA dense/sparse, 쓰기 커널, pooling과 selection). 4개 형상.
- `766a593e` test(qwen38_flash_next): add the KVarN K0 fake-quantization accuracy study
  - `tests/models/qwen38_flash_next/kvarn_k0.cpp`: 공개 engine API만 쓴다. AR 세션 prefill → materialized 스냅샷 payload의 attention K/V 영역(12층, 2 헤드 × 256)을 27B `kvarn_fake_quant.hpp`로 바꾸고 복원 → teacher forcing KL/top-1, free greedy. payload 영역 계산은 `WalkSnapshot` 순서를 따르고, 합이 executor payload 크기와 정확히 같아야 한다(아니면 중단).
  - `--prefill-chunk`, `--quantize-decode`, `--context`(기본 262,144 모델, 세션은 프롬프트 + steps)
- 이 두 커밋은 아직 `ai-tools/port-u4f`에 넣지 않았다(감독 결정: cherry-pick 시점).

## 빌드와 검사 [확인]

- 포맷: `tools/ci/check-format.py` PASS(729 파일, nixos/nix, `build/f1-format.log`).
- 테스트 빌드: 운영 derivation override(`build/f1-nix-tests.nix`, ROCm 10.1 / HIP Clang 24, BUILD_TESTING=ON) → `/nix/store/r5zpiajzg87q6k0fb68sk5czvall3mnf-gufo-f1-tests-0-unstable-dirty`(attention_ops, session, snapshot, continuation_adapter, kvarn_k0). 첫 시도는 K0의 `Relative`(27B 세션 헤더에만 있음) 컴파일 오류로 실패했고, 고쳐 다시 빌드했다.
- 운영 빌드: 후보 `path:build/f1-src#default` → `/nix/store/7lcxjf2g70cfszj228xzfzic1akkgxcv-gufo-0-unstable-dirty`, 기준 `git archive d2b751a9` → `/nix/store/vvvrxigmyyf0k6fhisdk5ryv6cdzhs9k-gufo-0-unstable-dirty`.
- 빌드 소스 `build/f1-src`는 포맷 적용 전 스냅샷이다. 커밋된 파일과 공백을 뺀 sha256이 파일마다 같다(토큰 동일).
- eval 이미지(`7c2c5a728270`, ROCm 7.2.4) CMake 빌드는 HEAD가 요구하는 `sndfile`이 없어 configure에서 실패했다. 그래서 nix 경로를 썼다.

## 창 계획 (ai.tools `test/experiments/gufo_upstream_port_20261023/`, LOCKS 반영)

| 계획 | 내용 | 이미지 | 상태 |
|---|---|---|---|
| `PLAN-F1-F0.json` `f1-f0` | 메모리 점유 7개 경우 + n-gram cold/warm, 드라이버 `f1_f0.py` | `rqw3-base-rocm101`(c94572b3) | "start" 대기 |
| `PLAN-F1-P1T.json` `f1-p1-unit` / `f1-p1-greedy` / `f1-p1-suite` | 페이지 행 op 테스트와 모델 테스트, greedy 89개 sha(I1V/M0 AR 대조), 기능 `long-context`+`cache` | `f1-cand-rocm101`(만들 예정, 라벨 `766a593e…`) | 이미지와 승인 대기 |
| `PLAN-F1-K0.json` `f1-k0-32k` / `f1-k0-long` | K0-FN: 27B K0 fixture(32K, 256K 문서) + 새 needle(32K, 200K, 깊이 10/50/90 %) | 같음 | 이미지와 승인 대기 |

- 새 fixture: `fixtures/f1-needle-{32k,200k}.txt`, `f1-needle-gold.json`(생성 방법·sha256은 gold 파일 manifest).
- P1 타이밍(ABBA, 기준 이미지 `d2b751a9`)은 정확성 창 뒤에 따로 계획한다.

## 디스크 [확인]

- 빌드 세 번과 closure tar로 루트 여유가 146.8 GiB까지 내려갔다(창 시작 하한 150 GiB). 내 tar와 임시 빌드 디렉터리를 지워 149.8 GiB가 되었고, 감독에게 알렸다.
- 이미지 두 개(후보, 기준)에 약 3.2 GB가 더 든다. 공유 nix store GC나 다른 세션의 이미지는 건드리지 않았다.

## 남은 문제

- [짐작] P1의 표 간접 참조 비용(decode attention 행마다 표 한 번 읽기)은 ABBA로 확인해야 한다.
- K0 payload 영역 계산은 스냅샷 형식(`QFNSNAP6`, payload version 18)에 묶여 있다. 형식이 바뀌면 K0이 크기 불일치로 멈춘다(조용히 틀리지 않는다).
