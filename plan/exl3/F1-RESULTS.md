# F1 결과: Flash-Next F0 측정, 페이지 KV(P1), KVarN K0 (2026-10-30)

상위 문서: [F1-DESIGN.md](F1-DESIGN.md), [F1-BRIEF.md](F1-BRIEF.md). 표기: **[확인]**은 측정·빌드·코드로 확인한 것, **[짐작]**은 추론이다.

## 상태

- 감독 결정(2026-10-30): F0은 측정만, n-gram 표는 F2 범위에서 뺀다, 이미지 선점은 P2에 별도 커밋, MTP 층은 같은 plane, KVarN V 배치는 K0-FN 뒤에 정한다.
- **F0:** 완료(아래 "F0 결과").
- **P1, K0-FN:** 코드, 빌드, 이미지 완료. GPU 창은 아직 돌리지 않았다. 순서는 X-A xa-ref → F1 F0 → W9 3개 → X-A X1 → P1T/K0이다.

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
| `PLAN-F1-P1T.json` `f1-p1-unit` / `f1-p1-greedy` / `f1-p1-suite` | 페이지 행 op 테스트와 모델 테스트, greedy 89개 sha(I1V/M0 AR 대조), 기능 `long-context`+`cache` | `f1-cand-rocm101`(id `65ed13bc`) | 승인 대기(순서: X-A X1 뒤) |
| `PLAN-F1-K0.json` `f1-k0-32k` / `f1-k0-long` | K0-FN: 27B K0 fixture(32K, 256K 문서) + 새 needle(32K, 200K, 깊이 10/50/90 %) | 같음 | 승인 대기(순서: X-A X1 뒤) |

- 새 fixture: `fixtures/f1-needle-{32k,200k}.txt`, `f1-needle-gold.json`(생성 방법·sha256은 gold 파일 manifest).
- P1 타이밍(ABBA, 기준 이미지 `d2b751a9`)은 정확성 창 뒤에 따로 계획한다.

## 디스크 [확인]

- 빌드 세 번과 closure tar로 루트 여유가 146.8 GiB까지 내려갔다(창 시작 하한 150 GiB). 내 tar와 임시 빌드 디렉터리를 지워 149.8 GiB가 되었고, 감독에게 알렸다.
- 공유 nix store GC나 다른 세션의 이미지는 건드리지 않았다. 감독이 사용자 승인을 받아 docker build cache만 정리했다(루트 177 GiB).
- 이미지(2026-10-30, 감독 승인, closure tar는 빌드 직후 지움, 뒤 여유 166.1 GiB):
  - 후보 `ai-tools/gufo-eval:f1-cand-rocm101`, id `65ed13bc`, 라벨 `766a593e9834382cb7a304655acf04de1ab2034b`. 운영 gufo `7lcxjf2g…`와 F1 테스트 `r5zpiajz…`의 closure를 담는다.
  - 기준 `ai-tools/gufo-eval:f1-base-rocm101`, id `8b33a7a6`, 라벨 `d2b751a90502d5cb37cceb66d40947ea830638fd`, 운영 gufo `vvvrxigm…`.
  - 둘 다 `FROM ai-tools/gufo-eval:64849ca6ec7b`(F23/R2 방식, `build/f1-img-{cand,base}/Dockerfile`, closure 목록 `closure.txt`). 컨테이너 안에서 바이너리 경로와 revision 파일만 확인했다(GPU 사용 없음).

## F0 결과 (2026-10-30, 창 `PLAN-F1-F0.json` `f1-f0`) [확인]

- **창:** 감독 "start" 뒤 한 번 돌렸다. COMPLETE, health_PASS, 재시도 없음. 이미지 `rqw3-base-rocm101`(c94572b3), NVMe 모델 `:ro`, mmproj 외장 `:ro`(a6만).
  - 첫 실행 시도는 `--out`을 상대 경로로 넘겨 `window.py`의 경로 검사(lock과 preflight 전)에서 멈췄다. 컨테이너와 GPU 사용은 없었고, 로그는 `f1-f0.launch-error-1.log`로 남겼다.
- **결과 위치:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/f1-f0/f1-f0/receipt/run/`(`RESULT.json`, 경우별 `server-1.log`). 분석 스크립트는 F1 에이전트 scratchpad `f0_analyze.py`다.
- **이 장비의 HIP 총량은 112.00 GiB다**(설계의 124 GiB는 card 14의 다른 커널 설정 값이다). 서버 전 사용량은 4 MiB다(표본기 자신의 HIP context).
- **greedy 동일성:** 세 프롬프트의 완성이 7개 경우, 반복 3회, AR과 MTP에서 모두 같다(sha `320a650b`, `e808b7ea`, `0111b72a`).

### 메모리 (hipMemGetInfo used, GiB)

"상주"는 card 14의 측정 상주 가중치(AR 82,384,141,824 B, MTP 85,157,259,776 B)다. "세션 공식"은 c94572b3 `Executor::SessionBytes`(`executor.cpp:1473-1512`)의 S배다(이미지 선점 제외). "나머지" = 준비 − 상주 − 세션 공식이다.

| 경우 | 준비(ready + 10 s) | 상주 | 세션 공식(S × 1개) | 나머지 | 요청 중 최대 | 서버 RSS |
|---|---:|---:|---:|---:|---:|---:|
| a1 AR S1 262K | 85.97 | 76.73 | 6.34 | 2.90 | 86.35 | 2.54 |
| a2 MTP S1 262K | 89.10 | 79.31 | 7.01 | 2.78 | 89.64 | 2.52 |
| a3 MTP S2 131K | 89.32 | 79.31 | 7.32 (3.66) | 2.70 | 90.02 | 2.45 |
| a4 MTP S4 65K | 89.76 | 79.31 | 7.93 (1.98) | 2.52 | 90.76 | 2.45 |
| a5 MTP S8 32K | 90.65 | 79.31 | 9.16 (1.14) | 2.18 | 92.18 | 2.47 |
| a6 MTP S1 262K + mmproj | 90.09 | 79.31 | 7.01 | 3.77 | 90.63 | 2.91 |
| a7 MTP S8 262K | 시작 거절 | — | 56.08 (7.01) | — | — | — |

읽은 것:
- **세션 증가는 공식과 맞는다.** 나머지(executor scratch, hipBLASLt, 런타임)는 2.2–2.9 GiB다. 세션이 많을수록 세션당 약 75–85 MiB 줄어든다. [짐작] 공식에 든 MTP rollback 행(첫 행은 DeltaNet 전체 상태, 36 × 3 MiB)이 첫 투기 단계에서야 할당되기 때문이다(`EnsureRollback`, `executor.cpp:1427`). 그래서 공식은 상한이다.
- MTP와 AR의 차이(S1 262K)는 3.13 GiB다. 사이드카 2.58 GiB + 세션 차이 0.67 GiB = 3.25 GiB 예측보다 0.12 GiB 작다(위와 같은 rollback 몫).
- **mmproj:** 준비가 +0.99 GiB 늘었다(인코더 상주). 세션당 이미지 선점 2.50 GiB(262K × 10,252 B)는 실제로 할당되지 않았다. admission 청구만 9.51 GiB로 커진다.
- **요청 중 증가:** S1은 +0.4–0.5 GiB, S8은 요청마다 약 +0.15 GiB씩 늘어 +1.5 GiB에 이른다. [짐작] 첫 투기 단계의 rollback 할당과 접두 캐시 스냅샷(장치 쪽 보존)이다.
- **a7 거절:** "text runner request-state claim exceeds state capacity"로 시작이 거절되었다(8 × 7.01 GiB > 남는 약 30 GiB, `text_model_runner.cpp:264-273`).
- **호스트:** 서버 RSS는 2.45–2.54 GiB(mmproj 2.91)다. 로드 직후 `host_available_mib`는 약 20.5 GiB였다(GPU 사용이 같은 LPDDR을 쓴다).

### n-gram 디스크 조회: cold 대 warm

cold는 새 프로세스의 첫 요청(빈 행 캐시), warm은 같은 요청의 반복이다. `O_DIRECT`라 OS 페이지 캐시는 영향이 없다. 각 값은 표본 1개라 참고용이다.

| 경우 | 프롬프트 | prefill ms cold / warm1 / warm2 | decode ms(128) cold / warm | n-gram 디스크 읽기 cold / warm |
|---|---|---|---|---|
| a1 AR S1 | 7,369 | 4,730 / 4,262 / 4,224 | 4,529 / 4,554 / 4,557 | 90,520 / 6,888 |
| a1 AR S1 | 27,158 | 16,797 / 15,912 / 15,987 | 4,538 / 4,537 / 4,534 | 311,022 / 85,748 |
| a2 MTP S1 | 7,369 | 4,747 / 4,291 / 4,552 | 3,010 / 2,997 / 2,996 | 91,313 / 7,018 |
| a2 MTP S1 | 27,158 | 17,083 / 16,359 / 16,339 | 3,695 / 3,693 / 3,699 | 311,373 / 85,954 |
| a3–a5 MTP S2/4/8 | 27,158 | 17,204–17,510 / 17,210–17,491 / 17,318–17,657 | 3,701–3,741 / 3,689–3,760 | 약 311K / 87K |

- **decode:** 디스크 읽기가 정체를 만들지 않는다. 같은 길이 프롬프트의 cold와 warm decode 차이가 −25 ms에서 +13 ms 사이다(128토큰, 1 % 아래, 잡음 범위). 짧은 프롬프트의 cold MTP decode +298 ms는 서버의 첫 요청(graph capture 등 warmup)과 겹친다. 그래서 n-gram 몫으로 보지 않는다.
- **prefill, 세션 1개:** cold가 7.4K에서 0.47–0.51 s(약 10–12 %), 27K에서 0.72–0.89 s(약 4.5–5.3 %) 느리다. 27K cold는 약 31만 행(4 KiB 정렬 직접 읽기, 약 1.2 GiB)을 읽는다.
- **prefill, 세션 2개 이상:** cold와 warm 차이가 없다(±0.3 %). 대신 warm도 S1보다 약 5–7 % 느리다. [짐작] 다중 세션 admission은 pass가 더 작아(2048) 다음 청크 prefetch가 읽기를 다 숨기지만, pass가 잘게 나뉘는 비용이 있다. 확인하지 않았다.
- **warm인데도 디스크 읽기가 남는다:** 27K에서 약 8.6만 행(20 %)이다. 128 MiB 직접 사상 캐시(약 149만 슬롯)의 충돌이다.
- **결론:** n-gram 표를 디스크에 두어도 decode 속도는 해치지 않는다. 새 문서의 첫 prefill(S1)만 최대 약 10 % 손해를 본다. 개선 후보(행 캐시를 키우거나 연관도를 높이기)는 F0 범위 밖이므로 따로 제안한다.

### F2 bpw 결정용 표 (112 GiB 기준으로 다시 계산) [계산]

- 남는 메모리 = 112 − 상주 − 나머지 약 2.8 GiB다.
- 262K MTP 세션 하나는 7.01 GiB(공식, FP16 KV)다. KVarN k4v2는 약 1.6 GiB로 본다 [짐작: 설계 3.2의 6.3 KB/토큰].
- 전문가 상주는 14.06 × b GiB, 나머지 상주는 7.6 GiB로 둔다. 서버 상한은 8세션이다.

| 전문가 bpw | 상주 | 남는 메모리 | 262K MTP 세션 수 (FP16 / KVarN k4v2) | 32K MTP 세션 수 (FP16, 상한 8) |
|---:|---:|---:|---|---|
| 지금 UD-Q4_K_XL(5.10) | 79.3 | 29.9 | 4 / 8(상한) | 8 |
| 3.05 | 50.5 | 58.7 | 8 / 8 | 8 |
| 4.00 | 63.8 | 45.4 | 6 / 8 | 8 |
| 5.00 | 77.9 | 31.3 | 4 / 8 | 8 |
| 6.00 | 92.0 | 17.2 | 2 / 8 | 8 |

## 남은 문제

- [짐작] P1의 표 간접 참조 비용(decode attention 행마다 표 한 번 읽기)은 ABBA로 확인해야 한다.
- K0 payload 영역 계산은 스냅샷 형식(`QFNSNAP6`, payload version 18)에 묶여 있다. 형식이 바뀌면 K0이 크기 불일치로 멈춘다(조용히 틀리지 않는다).
