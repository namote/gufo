# F1 결과: Flash-Next F0 측정, 페이지 KV(P1), KVarN K0 (2026-10-30)

상위 문서: [F1-DESIGN.md](F1-DESIGN.md), [F1-BRIEF.md](F1-BRIEF.md). 표기: **[확인]**은 측정·빌드·코드로 확인한 것, **[짐작]**은 추론이다.

## 상태

- 감독 결정(2026-10-30): F0은 측정만, n-gram 표는 F2 범위에서 뺀다, 이미지 선점은 P2에 별도 커밋, MTP 층은 같은 plane, KVarN V 배치는 K0-FN 뒤에 정한다.
- **F0:** 완료(아래 "F0 결과").
- **P1 정확성, K0-FN, P1 ABBA:** 완료(아래). ABBA는 slower 0, inconclusive 1. KVarN K1a 커밋, K1b 코드 작성 중이다.

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

## P1T 첫 시도 (2026-10-30, `f1-p1-unit`) [확인]

- **감독 "start"(P1T 세 개 → K0 두 개) 뒤 첫 창 `f1-p1-unit`이 4초 만에 FAILED가 되었다.** health_PASS였고, 테스트는 하나도 실행되지 않았다.
  - 다섯 바이너리 모두 로드 단계에서 `libelf.so.1: cannot open shared object file`(rc 127)로 멈췄다.
  - 원인: `build/f1-nix-tests.nix`가 테스트를 감싸지 않고 그대로 설치했다(`postFixup = "true"`). 운영 `gufo`는 wrapper가 `LD_LIBRARY_PATH=/nix/store/88jjsz00…-rocm-runtime-10.1/lib`를 넣는다. W9 probe 빌드는 `wrapProgram`으로 같은 일을 했다.
- 지시("P1T 정확성 창이 실패하면 멈추고 보고")대로 그 뒤 창(greedy, suite, K0)은 돌리지 않았다.
- 결과는 `f1-p1/f1-p1-unit.attempt-1-libelf/`에 남겼다.
- **고침(재빌드 없음):** 두 계획의 모든 테스트에 같은 런타임 경로를 `env.LD_LIBRARY_PATH`로 넣었다. 이미지 안에서 `ldd`로 다섯 바이너리 모두 빠진 라이브러리가 0개임을 확인했다(GPU 사용 없음). LOCKS를 반영했다.
- 다시 돌릴지는 감독이 승인한다.

## P1T 두 번째 시도 (2026-10-30, `f1-p1-unit`) [확인]

- 감독 재승인 뒤 다시 돌렸다. 창마다 사전 검사(창 컨테이너 없음, LOCKS 일치, 디스크 150 GiB 이상)를 했다. 결과는 FAILED, health_PASS다.
  - **통과:**
    - snapshot test(exact round trips, 164 s)
    - session test(450 s)
    - continuation adapter test(214 s)
    - 페이지 행 op 검사 3개 형상(n=1/5000, 8/9000, 64/4093): decode walk/list split 0/8, WMMA dense/sparse, 쓰기 커널, pooling, selection이 모두 비트 단위로 같다.
  - **실패:** 4번째 형상(n=300, start=0, 2블록)이다. 내 테스트 표 생성기가 보폭 7로 풀 2 b + 3 = 7 블록을 돌아 같은 블록을 반복했고, 테스트 자체의 검사가 "paged test table repeats a block"를 던졌다. 그 뒤의 기존 attention_ops 검사는 이 창에서 돌지 않았다.
- 지시대로 멈췄다. greedy, suite, K0은 돌지 않았다. 결과는 `f1-p1/f1-p1-unit.attempt-2-testtable/`이다.
- **고침:**
  - `83d90aaa` test(qwen38_flash_next): keep the paged-row test table a permutation for two-block sequences. 풀이 7의 배수이면 1을 더한다. 1–199 블록에서 순열임을 확인했고, 포맷 검사도 통과했다.
  - 테스트 재빌드 결과는 `/nix/store/nr6f886qfcmwaw7mqqgxzn2h86djai55-gufo-f1-tests-0-unstable-dirty`다.
  - 후보 이미지를 다시 만들었다: `f1-cand-rocm101` id `99c908e2`, 라벨 `83d90aaa…`. 운영 gufo `7lcxjf2g`는 그대로다(운영 소스가 바뀌지 않았다). closure tar는 바로 지웠다.
  - 계획 경로와 라벨을 갱신하고 LOCKS를 반영했다(`7692cbe4…`).
- 디스크가 다시 148.3 GiB로 내려갔다(새 이미지 층 + nix store). 감독이 build cache를 정리해 154 GiB가 되었다.

## P1 정확성 (2026-10-30, 세 번째 시도, 이미지 `f1-cand-rocm101` id `99c908e2`) [확인]

창마다 사전 검사를 하고 한 번씩 돌렸다. 세 창 모두 COMPLETE, health_PASS, 재시도 없음이다. 결과는 `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/f1-p1/`에 있다.

- **`f1-p1-unit`:**
  - attention_ops(48 s) 통과.
    - 페이지 행 4개 형상(20/43, 36/75, 17/37, 2/8 블록)이 모두 비트 단위로 같다.
    - 기존 검사도 모두 통과했다: chunk 경계 동일, WMMA live extent, list 대 walk 비트 동일, structured FP64 오차 1.5e-7 이하, WMMA 2e-2 이하.
  - snapshot(163 s, exact round trips), session(445 s), continuation adapter(215 s) 통과.
  - 이 모델 테스트들은 identity 표 뒤 블록 span 저장소로 돈다.
- **`f1-p1-greedy`:** A 64, K 17, D 8 = 89개 완성 sha256이 I1V와 같고 M0 AR과도 같다(89/89).
  - draft 수는 I1V와 87/89가 다르다. 기준 `d2b751a9`에 W9 `92fcaf37`("keep the first-cycle MTP width and relax deep confidence thresholds")이 들어 있어 정책이 I1V와 다르기 때문이다.
  - 같은 정책으로 돈 기록이 없으므로, draft 수가 정확히 같은지는 ABBA의 BASE(d2b751a9) 팔과 비교한다.
  - 후보의 drafted / accepted 합은 15,850 / 11,115다(I1V는 15,332 / 10,982).
- **`f1-p1-suite`:**
  - single-session(`--sessions 1`, `long-context` + `cache`): 13개 사례가 모두 통과했다.
  - standard-cache(`--sessions 2`, `cache`): 12개 사례가 모두 통과했다.
  - 모델은 NVMe에 있고, mmproj는 같은 디렉터리로 마운트했다. 테스트는 이미지 runner(W9 I1F/I1F2와 같다)를 썼다.

## K0-FN: Flash-Next KVarN fake quant 품질 (2026-10-30) [확인]

- **창:** `f1-k0-32k`(6.5분), `f1-k0-long`(8.5분). 둘 다 COMPLETE, health_PASS다.
- **원시 기록:** `f1-p1/f1-k0-*/receipt/run/*.jsonl`
- **채점:** 27B 문서 과제는 `kvarn-gold.json`과 JSON 일치로, needle은 세 코드가 모두 들어 있는지로 본다.
- **fp16 행:** 스냅샷 왕복에서 KL이 0이고 top-1이 1.0이다. 하네스(payload 영역 계산과 복원)가 정확하다는 뜻이다.
- **모든 설정에서** teacher forcing top-1이 1.0000이고, free greedy가 FP16과 같으며(갈림 없음), gold/needle을 모두 맞혔다. 그래서 차이는 KL에만 있다.

| 입력 | 토큰 | 설정 | K 상대오차 | V 상대오차 | KL 중앙값 | KL p95 | KL 최대 |
|---|---:|---|---:|---:|---:|---:|---:|
| 27B 32K 과제 | 32,770 | rtn_k4v2 | 0.041 | 0.624 | 5.8e-6 | 2.6e-3 | 2.2e-2 |
| | | hrtn_k4v2 | 0.046 | 0.541 | 5.2e-6 | 6.9e-4 | 6.6e-3 |
| | | **kvarn_k4v2** | 0.046 | 0.541 | 4.4e-6 | 6.0e-4 | 1.0e-2 |
| | | kvarn_k4v2_g64 | 0.040 | 0.541 | 4.6e-6 | 8.3e-4 | 9.8e-3 |
| | | **kvarn_k4v4** | 0.046 | 0.108 | 1.2e-6 | 8.3e-5 | 1.7e-3 |
| | | kvarn_k8v4 | 0.003 | 0.108 | 1.3e-6 | 3.0e-5 | 6.1e-4 |
| 같은 과제, prefill 4096 청크 | 32,770 | rtn_k4v2 | | | 6.9e-6 | 1.7e-3 | 2.3e-1 |
| | | kvarn_k4v2 | | | 1.3e-5 | 1.1e-3 | 8.8e-3 |
| | | kvarn_k4v4 | | | 1.6e-6 | 1.7e-4 | 1.3e-3 |
| 27B "256K" 문서 | 254,880 | rtn_k4v2 | 0.042 | 0.619 | 1.2e-5 | 1.3e-3 | 3.0e-2 |
| | | kvarn_k4v2 | 0.047 | 0.541 | 1.0e-5 | 7.6e-4 | 1.3e-2 |
| | | kvarn_k4v4 | 0.047 | 0.108 | 1.8e-6 | 8.8e-5 | 9.6e-3 |
| | | kvarn_k8v4 | 0.003 | 0.108 | 1.9e-6 | 7.9e-5 | 3.0e-3 |
| needle 32K (docs) | 27,567 | rtn_k4v2 | 0.075 | 0.590 | 3.7e-5 | 8.5e-4 | 3.7e-3 |
| | | kvarn_k4v2 | 0.075 | 0.542 | 1.9e-5 | 2.5e-4 | 9.6e-4 |
| | | kvarn_k4v4 | 0.075 | 0.108 | 4.1e-6 | 6.3e-5 | 1.7e-4 |
| needle 200K (docs) | 195,049 | kvarn_k4v2 | 0.074 | 0.542 | **2.7e-4** | 5.4e-3 | 3.2e-2 |
| | | kvarn_k4v4 | 0.074 | 0.108 | 3.8e-5 | 4.9e-4 | 1.8e-3 |

읽은 것:
- **27B K0 대비:** 27B kvarn_k4v2의 KL 최대는 32K에서 9.5e-4, 256K에서 6.7e-3이었다(`plan/archive/gufo-upstream-port/KVARN.md:128-157`). Flash-Next k4v2는 1.0e-2 / 1.3e-2로 32K에서 약 10배 크다. [짐작] KV 헤드가 2개이고 GQA가 12라 KV 행 하나의 오차가 더 많은 query 헤드에 퍼진다. 확인하지 않았다.
- **긴 자연어 문맥(needle 200K)에서 k4v2의 KL 중앙값이 2.7e-4로 커진다**(27B 과제 문서 1.0e-5). 그래도 top-1과 답은 그대로다. k4v4는 중앙값 3.8e-5, 최대 1.8e-3이다.
- **Hadamard와 balancing:** 회전은 V 오차를 0.62에서 0.54로 줄이고, KL 꼬리(p95, 최대)를 rtn 대비 2–4배 줄인다. balancing(kvarn 대 hrtn)은 이 입력에서 차이가 작다.
- **chunked prefill:** k4v2의 중앙값이 약 3배 커지지만 최대는 비슷하다. rtn_k4v2는 청크에서 최대 0.23으로 튄다.
- **판정 기준은 사용자와 정한다**(설계 3.2). 이 결과만 보면 k4v4가 27B k4v2 수준(최대 1e-3대)이다. k4v2는 답은 유지하지만 KL 꼬리가 10배다. V 배치 결정(channel-major / token-major)에 이 결과를 쓴다.

## P1 ABBA 타이밍 (2026-10-30 23:16 – 10-31 00:42) [확인]

- **실행:** 감독의 조건부 "start" 뒤 돌렸다. 조건은 내 빌드 중지, EXL3SAME 종료, `docker ps` 비어 있음, 1분 load < 2이고, 23:16에 충족되었다. 창마다 사전 검사(빈 `docker ps`, LOCKS 일치, 153 GiB)를 했다.
- **순서:** BASE-mtp a1 → CAND-mtp b1 → CAND-mtp b2 → BASE-mtp a2 → BASE-ar a1 → CAND-ar b1 → CAND-ar b2 → BASE-ar a2. 8개 모두 COMPLETE, health_PASS, 재시도 없음.
- **이미지:** BASE `f1-base-rocm101`(d2b751a9, id `8b33a7a6`), CAND `f1-cand-rocm101`(83d90aaa 라벨, 운영 gufo `7lcxjf2g` = P1 b75a0051, id `99c908e2`).
  - CAND 계획의 `gufo_revision`을 이미지 라벨(83d90aaa)에 맞춰 고쳤다. LOCKS는 `ac9f15e4…`다.
- **결과 위치:** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/f1-abba/`
- **판정:** `tools/qwen-flash/spec/abba.py`, 요청·단계별 5 %·10 ms. 출력은 `~/projects/ai.tools/storage/evaluations/f1-abba-20261031/{mtp-0,mtp-1,mtp-2,ar-0}.json`(sha256 `3e1f5306…`, `ae9e2e9c…`, `cbd54721…`, `8afb98a4…`)이다.
- **load:** 창 시작 직전 1분 load가 9.65, 18.12로 높은 때가 있었다. 직전 창 서버의 로드·종료(n-gram reader의 I/O 대기)가 1분 평균에 남은 것이다. 창 실행 중 확인했을 때 외부 부하는 없었다(승인된 외장 HF 다운로드 `curl` 3.6 % CPU만).

| 서버 | 세트 | 프롬프트 | pass | slower | faster | inconclusive | decode tok/s BASE / CAND(참고) |
|---|---|---:|---:|---:|---:|---:|---|
| MTP | A | 64 | 127 | 0 | 0 | 1 | 53.17 / 52.85 (0.994) |
| MTP | K | 17 | 34 | 0 | 0 | 0 | 51.43 / 51.42 (1.000) |
| MTP | D (8K/32K) | 8 | 16 | 0 | 0 | 0 | 45.79 / 45.55 (0.995) |
| AR | D (8K/32K) | 8 | 16 | 0 | 0 | 0 | 28.48 / 28.18 (0.990) |

- **동일성:** 97개 프롬프트 × 4회(두 팔 × 2반복)에서 완성 sha256, draft, accepted, prompt 토큰 수가 모두 같다(97/97). P1이 draft 정책 경로를 바꾸지 않았다는 것이 BASE 팔로 확인되었다.
- **inconclusive 1건:** MTP A `repetition_sequence` decode다. BASE 1,583–2,183 ms(중앙값 1,666), CAND 1,587–2,300 ms(중앙값 1,828)이다. 두 팔 모두 1.6 s대와 2.0–2.3 s대로 갈리는 쌍봉이고, 범위가 겹친다. 규칙대로 확대하거나 평균 내지 않고 inconclusive로 둔다. 확인하려면 이 history만 번갈아 다시 돌린다(감독 결정).
- **slower 판정은 0건이다.** decode tok/s 비율 0.990–1.000은 참고값이다(판정은 요청별).

## P1 inconclusive 재측정: `repetition_sequence` (2026-10-31 01:35–01:48) [확인]

- **실행:** `PLAN-F1-REP-{BASE,CAND}.json`, 순서 B C C B B C C B, 창 8개 모두 COMPLETE, 재시도 없음.
  - 창마다 `docker ps`가 비고 1분 load < 2가 될 때까지 기다렸다(실제 0.18–1.86). 158 GiB, LOCKS 일치.
  - 요청은 ABBA와 같고, warmup 1 + 반복 5라 팔마다 20표본이다.
- **판정**(`abba.py`, 출력 `storage/evaluations/f1-abba-20261031/rep-repetition_sequence.json`, sha256 `71de0a57…`):

| 단계 | 판정 | BASE 중앙값 | CAND 중앙값 | 차이 | 기준 |
|---|---|---:|---:|---:|---:|
| prefill | pass | 145.6 ms | 146.0 ms | +0.3 ms | 10 ms |
| decode | **pass** | 1,578.8 ms | 1,585.0 ms | +6.2 ms (0.4 %) | 78.9 ms |

- 두 팔 모두 20개 중 4개가 1.67–1.88 s의 느린 표본이다(BASE 1,716–1,883, CAND 1,670–1,783). ABBA에서 보인 쌍봉은 두 팔에 공통인 잡음이다.
- 완성 sha256(`2a606f9f…`)과 draft 110 / accepted 92가 REP 8회와 ABBA 4회에서 모두 같다.
- **결론: P1 ABBA의 inconclusive가 해소되었다. P1 타이밍은 slower 0, inconclusive 0이다.**

## KVarN K1a / K1b (2026-10-31, 설계 [F1-KVARN-DESIGN.md](F1-KVARN-DESIGN.md), 감독 승인)

- **커밋(브랜치 `ai-tools/f1-paged-kv`):**
  - `db7aa9ac` refactor(core): share the KVarN record layout and tail planners across models
    - `src/core/hip/kvarn_layout.hpp`(새 파일)와 `kvarn_checkpoint.hpp`(이동)에 공통 부분을 둔다: 그룹, sink, ring, 레코드(V 배치 인자 `KvarnValueOrder`), FP16 행, read groups, pass/prefill/checkpoint planner.
    - 27B 헤더에는 모델 모양과 조정된 정책만 남긴다. 이름과 값은 그대로다.
    - core 테스트 `kvarn_layout_test`를 추가했다.
  - `b17b5868` feat(qwen38_flash_next): rotate and quantize KVarN k4v4 token-major records
    - `src/core/hip/detail/kvarn_device.hpp`: FWHT256, float-float 합, 정확히 반올림한 log/exp, `KvarnQuantizeTile<VBits, Order>`. 27B 커널은 이 tile을 같은 상수로 부른다.
    - Flash-Next: `RotateKvRows`, `KvarnQuantizeGroups`(2헤드, k4v4 token-major, P1 블록 표), writer 전용 FP16 sink/ring 행 map(`kKvarnFp16RowTag`, 읽기 경로는 그대로다).
    - 테스트 `qwen38_flash_next_kvarn_ops_test`를 추가했다.
- **검사 [확인]:**
  - 포맷 PASS.
  - eval 이미지 g++로 core와 27B layout 테스트 PASS.
  - hipcc 7.2.4로 27B `attention_kvarn.hip` 컴파일 성공.
  - nix 테스트 빌드 `/nix/store/6azsdm5z5rvxgsx1nxqxjcs629nl0gb6-gufo-f1-k1b-tests-0-unstable-dirty`.
  - 테스트 이미지 `ai-tools/gufo-eval:f1-k1-tests-rocm101` id `d07be443`, 라벨 `b17b5868…`.
- **27B 동일성 증명 방법 [확인]:**
  - ISA는 바뀌었다(스케줄과 dual-issue 묶음).
  - 운영 바이너리 바이트 비교는 증명이 되지 못한다. 같은 운영 소스(`766a593e`와 `83d90aaa`)의 두 빌드도 `.text`와 `.hip_fatbin`이 다르기 때문이다(경로와 줄 위치가 박힌다).
  - 그래서 설계대로 27B KVarN op 테스트로 증명한다. 이 테스트는 flush된 모든 tile이 CPU 참조와 비트 단위로 같아야 통과한다.
- **창 계획:** `PLAN-F1-K1T.json`(`f1-k1-unit`, sha `a53b8d29…`). 대상은 core/27B layout, 27B KVarN op, Flash-Next KVarN op, Flash-Next attention op 테스트다. LOCKS `e9bf9ac4…`. 감독 "start"를 기다린다.
### K1T 창 (2026-10-31, `PLAN-F1-K1T.json` `f1-k1-unit`, 한 번) [확인]

- **결과:** FAILED, health_PASS, 재시도 없음. 위치는 `tmp/test-results/gufo-upstream-port-20261023/f1-k1/f1-k1-unit/`.
- **통과:**
  - core `kvarn_layout_test`
  - 27B `qwen_kvarn_layout_test`
  - **27B `qwen_attention_kvarn_ops_test`(20 s):** flush tile이 CPU 참조와 비트 단위로 같다. verify==decode도 차이 0이다. **K1a/K1b 공통화 뒤 27B 동일성이 증명되었다.**
  - Flash-Next `attention_ops`(48 s, P1 페이지 행과 writer 변경 포함)
- **실패: Flash-Next `kvarn_ops`**
  - writer ring 행: 통과(2개 경우).
  - 회전 행: 통과(sink와 ring 모두 비트 동일).
  - k4v4 token-major 레코드: 32 tile 중 21개가 정확하다. 코드는 32 tile 모두 같다(code_diff 0). FP16 scale/zero/column은 11 tile에서 4,355개가 다르다.
- **[짐작] 원인 후보:**
  1. 코드가 모두 같고 scale만 다르다. 그래서 balancing이 고른 행·열 scale(최선 반복의 상태)이 참조와 약간 달라, 반올림한 코드는 같지만 저장한 곱이 다른 경우다. 27B 테스트 데이터에서는 드러나지 않은 경로(채널별로 1–5.5배 범위를 준 이 테스트 데이터)일 수 있다.
  2. 레코드 필드를 덮어쓰는 경합이나 오프셋 오류. 코드 영역은 맞으므로 가능성은 낮다고 본다.
- **확인하지 않았다.** 어떤 tile(K/V, 그룹, 층, 헤드)이 다른지, 첫 차이 값이 무엇인지 테스트가 출력하지 않는다.
### K1T 진단 재실행 (2026-10-31, `30c1833c` 테스트, 이미지 id `40174bbd`) [확인]

- **결과:** FAILED(`flash-next-kvarn-ops`만), health_PASS, 재시도 없음. 나머지 4개(core/27B layout, 27B KVarN op, Flash-Next attention op)는 다시 통과했다. 첫 시도 결과는 `f1-k1/f1-k1-unit.attempt-1/`에 둔다.
- **두 가지 불일치가 있고, 원인은 둘 다 같다.** GPU 양자화기의 표준편차 합은 float-float(약 48비트)이고, 참조(`kvarn_fake_quant.hpp`)는 double로 차례대로 더한다. 둘은 대부분 같은 float로 반올림되지만 항상은 아니다.
  1. **동률 단계 선택(늘린 데이터, 11 tile, 코드 0개 차이):**
     - balancing imbalance가 5–6단계부터 float 정밀도에서 같은 값(2.00000048)으로 수렴한다.
     - 참조는 `<=`로 마지막 동률 단계(8)를 고른다.
     - GPU는 이후 단계의 imbalance를 1 ulp 크게 계산해 6 또는 7단계를 고른다. GPU 열 scale이 참조의 6/7단계 값과 정확히 같다.
     - 그래서 코드는 같고, 저장된 FP16 scale/zero/column이 약 0.2–0.6 % 다르다(예: 0.372314 대 0.373291).
     - 두 결과 모두 같은 imbalance의 유효한 balancing이다.
  2. **경계 반올림(균일 데이터, 9 tile):**
     - 고른 단계는 같다(8).
     - tile마다 코드 1–2개(32,768개 중) 또는 FP16 값 1개만 다르다. 반올림 경계에서 한 비트가 뒤집힌 것이다.
- **27B 테스트가 정확한 이유:** 27B 테스트 데이터는 Gaussian이고 채널마다 gain/mean과 outlier가 있다(`kvarn_ops_test.cpp:99-112`). 그래서 수렴 동률과 경계 사례가 생기지 않는다. 내 Flash-Next 테스트는 균일 분포다. 균일 분포는 채널·토큰 std가 거의 같아 동률이 쉽게 난다.
- **[짐작] 실제 KV에서:** 실제 KV는 채널마다 크기가 크게 다르다(27B 데이터에 가깝다). 그래서 동률은 드물 것이다. 그래도 일어날 수 있다. 27B 운영 KVarN에도 같은 성질이 이미 있다(같은 함수다). 27B 실제 품질 기준(real‖fake KL ≤ rounding 대조의 10배)은 이것을 허용한 기준이다.

**고칠 방법 (감독/사용자 결정, 코드는 아직 바꾸지 않았다):**

| 안 | 내용 | 27B 운영 출력 | 비용 |
|---|---|---|---|
| A | 테스트만 고친다. 27B와 같은 Gaussian/채널 gain/outlier 데이터로 비트 단위 정확성을 요구한다. 균일 데이터는 별도 허용 검사로 둔다: 코드 차이 1e-4 이하, FP16 값 1 ulp 이내, 또는 imbalance 동률(float ulp 이내)인 단계의 scale과 같음 | **변화 없음** | 테스트 재빌드 1회, 창 1개 |
| B | 참조와 커널의 단계 선택을 동률에 강하게 바꾼다(예: 상대 1e-6 넘게 나아질 때만 갱신, 첫 최소값을 고름) | 동률 tile의 양자화 KV가 바뀐다. 27B KVarN 재검증(op 테스트, K0 `--real`, 서버 gold)이 필요하고, 디스크 캐시 호환 판단도 필요하다 | 중간 |
| C | GPU 합을 FP64로 한다 | 경계 사례의 양자화 KV가 바뀐다(참조에 더 가까워지지만, 합 순서가 달라 비트 동일은 보장되지 않는다). 재검증이 필요하다 | gfx1151 FP64 처리율이 낮다. flush와 prefill 비용이 늘어난다 [짐작] |

- **추천: A.** 공통 산술을 그대로 두어 27B 운영 출력이 바뀌지 않는다. Flash-Next의 실제 정확성은 K3의 real‖fake KL 기준(27B와 같다)으로 판정한다.

**결정(사용자 2026-10-31): A안. 테스트만 고친다. 커널과 공통 산술은 그대로이고, 27B 운영 출력도 그대로다.**
- 비트 동일성 검사는 27B식 데이터로 한다: key는 Gaussian에 채널별 gain/offset, 37채널마다 outlier, value는 Gaussian(27B `RandomRows`와 같은 식, 행 폭 512). 모든 tile이 비트 단위로 같아야 한다.
- 균일 데이터(늘린 것과 늘리지 않은 것)는 별도 허용 검사로 둔다. tile이 다음 중 하나를 만족하면 통과다.
  - 정확히 같다.
  - 참조가 고른 단계, 또는 그 단계와 imbalance가 float 1 ulp 안에서 동률인 단계를 같은 RTN으로 다시 계산한 기록과 비교해 코드 차이가 3개 이하(32,768개 중 약 1e-4)이고, 모든 FP16 scale/zero/column이 1 ulp 이내다.
  - 통과 tile은 exact, tie, boundary로 나누어 센다. 단계별 재계산이 참조와 같아야 한다는 자체 검사(`mirror_mismatch` 0)도 함께 본다.
- **구현:** `f1333e24` test(qwen38_flash_next): check KVarN records bitwise on 27B-style rows and within a tie tolerance on uniform rows. 테스트 파일만 바꿨다. 포맷 PASS, hipcc syntax 검사 PASS.
- **빌드:** 테스트 `/nix/store/hc66ick8ppp2avw5q0nz1wk1z7m5xqvy-gufo-f1-k1b-tests-0-unstable-dirty`. 이미지 `f1-k1-tests-rocm101` id `3eba25ca`, 라벨 `f1333e24…`(이전 이미지는 지웠다).
- **창 계획:** `PLAN-F1-K1T.json`(sha `0e214322…`), LOCKS `3771bbec…`. 감독 "start"를 기다린다. 이전 결과는 `f1-k1/f1-k1-unit.attempt-2-diagnosis/`에 있다.
- **P1 inconclusive 재측정 계획:** `PLAN-F1-REP-{BASE,CAND}.json`. `repetition_sequence`만 대상으로 ABBA × 2, 반복 5, 팔마다 20표본이다. 감독 "start"를 기다린다.

### K1T option-A 창 (2026-10-31 09:55, `f1333e24` 테스트, 이미지 id `3eba25ca`) [확인]

- **결과:** FAILED(`flash-next-kvarn-ops`만), health_PASS, 재시도 없음. 나머지 4개는 다시 통과했다(27B KVarN op 비트 동일 포함). 위치 `f1-k1/f1-k1-unit/`.
- **실패:** 27B식 Gaussian 데이터에서 32 tile 중 21개만 비트 동일했다(11개 불일치).
- **원인 [확인]:** 진단 때와 같은 float ulp 동률이다. imbalance가 5–8단계에서 2.00000048로 같고, GPU 열 scale은 5/6/7단계와 같으며 참조는 8단계를 고른다. 11개 중 10개는 코드 차이가 0이다. **k4v4 token-major V는 Gaussian 데이터에서도 동률이 난다.** 그래서 "27B식 데이터는 동률이 없다"는 option-A 가정은 k4v4 token-major V에서 틀렸다. 27B k4v2 테스트는 그대로 비트 동일이다.
- **결정(감독 2026-10-31, option A의 범위 안):** 테스트만 고친다. 27B와 커널은 그대로다. Gaussian 데이터에도 같은 tie/boundary 허용(코드 3개 이하, FP16 1 ulp)을 적용하고, 허용한 tile마다 종류와 코드 차이를 출력한다. **실제 정확성은 최종 통합 품질 관문(real‖fake KL)에서 판정한다.**
- **테스트 수정(`fa5ba8af`에 포함):** Gaussian도 허용 검사로 바꿨다. `Near`의 ulp 검사가 부호가 다른 첫 값에서 나머지를 보지 않고 끝내던 문제를 고쳤다(더 엄격해졌다).

## K2/K3: Flash-Next KVarN k4v4 실행기 연결 (2026-10-31, `fa5ba8af`, 브랜치 `ai-tools/f1-paged-kv`)

- **커밋:** `fa5ba8af` feat(qwen38_flash_next): store trunk attention K/V as KVarN k4v4 records.
- **읽기 경로(kernels):** `KvarnReadView`(staging → sink → 레코드 → ring), per-token `Attention`과 WMMA `WmmaCausalAttention`의 KVarN 변형(템플릿 인자, FP16 경로는 그대로 인스턴스화된다), `RotateHeadRows`(q를 회전 좌표로 넣고 ctx를 되돌린다), 장치 조건부 in-pass 경계 flush `KvarnQuantizePassBoundary`(graph replay에서도 위치로 판정), staging 행을 읽는 `RotateKvRows`/`KvarnQuantizeGroups`.
- **실행기:**
  - `SessionKvPlanes`: KVarN이면 trunk 12층은 레코드 plane 하나(블록당 두 그룹), FP16 K/V plane은 MTP 층 하나만 둔다. pooled indexer plane은 그대로다. 세션마다 FP16 sink+ring(384행 × 12층 × K/V)을 따로 할당한다. `SessionBytes`도 같은 식이다.
  - 실행기마다 staging K/V(`KvarnStagingRows(max_batch)` 행)를 둔다.
  - `PlanKvarn`(본문 전): pass(≤128행)는 `PlanKvarnPass`, 그보다 긴 prompt chunk는 `PlanKvarnPrefill`로 계획하고, 필요한 그룹을 ring에서 먼저 양자화한다. `FinishKvarn`(성공 뒤): tail을 갱신하고, pass가 그룹 끝에서 끝나면 그 그룹을 바로 양자화한다.
  - `Attention`: writer 포인터만 바꾼다(pass는 sink/ring 태그 행, chunk는 staging). 그다음 `KvarnAttention`이 attention, chunk 뒤처리(완성 그룹을 staging에서 양자화, sink 행과 rewind 가능한 마지막 행을 ring에 복사), 출력 gate와 projection을 맡는다. pass는 질의마다 `KvarnReadGroups`로 읽으므로 per-token kernel을 쓴다. chunk는 고정 `quant_end`로 WMMA를 쓴다.
  - `ForwardBatch`: 세션마다 같은 plan/finish를 부른다.
  - 스냅샷: magic `QFNSNAPK`. sink, 그룹 바닥 전까지의 레코드, 바닥 뒤 ring 행을 private 영역으로 복사한다. 복원 뒤 바닥보다 앞으로의 rewind는 거절한다(그 그룹 FP16 행이 payload에 없다).
  - in-pass checkpoint와 continuation adapter는 KVarN을 거절한다.
- **서버/CLI:** `--kv-cache-type kvarn-k4v4`(기존 canonical 옵션, alias 없음)를 Flash-Next에 허용한다. k4v2는 이름을 대며 거절한다. 기본값은 fp16 그대로다. KVarN 서버는 state ABI `qwen38-flash-next-rocm-session-v1+kvarn-k4v4`를 쓰고, component cache, prefix reuse, snapshot/fork, in-pass checkpoint, persistence를 끈다. 레코드 형식의 디스크 캐시 identity는 component가 생길 때 정한다(지금은 디스크 캐시 없음).
- **K0 `--real`:** `qwen38_flash_next_kvarn_k0 --dump-reference FILE`(FP16 실행)과 `--real FILE`(KVarN 실행)의 두 프로세스로, 기준 FP16 logits에 대해 teacher-forced KL, top-1, 자유 생성 일치를 잰다.
- **빌드 [확인]:** 포맷 PASS. nix 테스트 빌드 `/nix/store/v72kjw6jn11l323jm2rzj5wy4a0smgms-gufo-f1-k2-tests-0-unstable-dirty`(gufo, CLI 테스트, op 테스트, K0, session/snapshot/adapter 테스트, 모두 컴파일). 이미지 `ai-tools/gufo-eval:f1-k2-tests-rocm101` id `62616023`, 라벨 `fa5ba8af…`.

### K2T 창 (2026-10-31 10:07, `PLAN-F1-K2T.json` `f1-k2-unit`, 한 번) [확인]

- **결과:** COMPLETE, 3개 모두 PASS, health_PASS. 위치 `f1-k2/f1-k2-unit/`.
  - `kv_cache_type_test` PASS.
  - Flash-Next `kvarn_ops` PASS:
    - writer, 회전 비트 동일.
    - 레코드: Gaussian exact 21 / tie 9 / boundary 2, uniform 23 / 0 / 9, stretched 21 / 8 / 3. 불일치 0, mirror 0. 코드가 다른 tile은 최대 2개 차이다.
    - `RotateHeadRows` 오차 0.
    - **읽기 경로:** chunk 2개(sink에서 시작, ring 행 staging), pass 9종(in-pass 경계, rewind 뒤 재양자화, 그룹 끝 flush)의 14회 실행이 모두 연속 FP16 참조와 **비트 단위로 같다**(max abs 0).
  - Flash-Next `attention_ops` PASS(49 s, FP16 kernel이 템플릿 분리 뒤에도 그대로임을 확인).

### Codex 리뷰 수정 (2026-10-31, `cdbe0563`) [확인]

- **리뷰(감독 전달, `fa5ba8af`):** rewind, 재양자화, graph replay, batch 계획, 262K 산술에는 결함이 없다고 했다. 고친 세 가지는 다음과 같다.
  1. **(P1) KVarN HTTP 서버가 뜨지 않았다:** component cache를 끄면 일반 session pool이 `CreateState()`를 부르는데, Flash-Next는 거기서 무조건 throw했다. 이제 세션을 소유하는 상태를 만든다. adapter가 없을 때 무효화는 세션 `Reset()`이다.
  2. **(P2) 그룹 끝 flush와 snapshot capture 경합:** `FinishKvarn`이 forward의 마지막 synchronize 뒤에 양자화를 띄우고 기다리지 않았다. 이제 그 flush 뒤에 stream을 synchronize한다(128 토큰마다 한 번).
  3. **(P2) prompt/bench가 kvarn-k4v4를 조용히 무시했다:** 이제 `ModelOptions::kvarn`을 넘긴다. bench는 `KV cache:` 줄을 출력한다.
- **창 `PLAN-F1-K2S.json` `f1-k2-smoke`(이미지 id `9119e062`, 라벨 `cdbe0563`):** COMPLETE.
  - `kv_cache_type_test`와 Flash-Next `kvarn_ops`(K2T와 같은 결과, 읽기 경로 14회 비트 동일) PASS.
  - **서버 기동 smoke(`f1_smoke.py`, 품질·타이밍 검사가 아니다):** `--kv-cache-type kvarn-k4v4 --context 8192 --sessions 2`로 AR과 MTP 각각 서버가 뜬다. 340토큰 prompt(chunk > 128)에 300토큰 decode(그룹 끝 384/512/640 통과), 동시 2요청(batched decode)과 반복 1요청이 모두 일관된 답을 냈다. 반복 응답은 같다.
  - 첫 시도는 smoke 스크립트의 model 이름 오류(404)였다. 서버는 그때도 떴다(`f1-k2-smoke.attempt-1-model-name/`).

### Codex F2(fn-exl3)와 충돌 지점

- `Executor::Attention`의 fused `AttentionF16Gemm` / `Dense(attn_qkv)` + `PrepareAttention` 호출(F2-DESIGN.md:240-266의 projection 호출 지점). F1은 이 호출들에서 K/V writer 인자(`write_k`, `write_v`, `write_rows`)만 바꿨다. F2가 이 호출을 EXL3 projection으로 바꾸면 merge 때 writer 인자 세 개를 새 호출에 옮겨야 한다.
- `executor.cpp`의 `CreateSession`/`SessionBytes`(KV plane 배치)와 `executor.hpp`의 `AttentionState`는 F2가 건드리지 않는다고 본다 [짐작].

### 남은 일

- 모델 단계 검사는 최종 통합 창에서 한다: K0 `--real`(real‖fake KL, 27B와 같은 기준), session/snapshot 테스트, KVarN 서버 correctness와 ABBA.
- [짐작] pass 모드(MTP 검증 폭, 짧은 prompt)는 WMMA 대신 per-token kernel을 쓴다. prefill 단계의 ≤128행 pass가 느려질 수 있다. 최종 창에서 잰다.
- P2 공유 pool은 하지 않았다(W9 engine.* 관문).

## FINAL-FN 품질 창 (2026-10-31, `PLAN-FINAL-FN.json`, ai-tools/integ `c97d555d`) [확인]

- **이미지:** X-A의 `final-c97d555d-tests-rocm101`(a42ab0c0), 그리고 F1 추가본 `final-c97d555d-fn-tests-rocm101`(95e7bf4a). 추가본은 테스트 이미지에 `build/fn-final-tests.nix` closure를 더한 것이고, 최종 테스트 derivation이 빼는(EXCLUDE_FROM_ALL) session/snapshot/adapter 테스트, kvarn_k0, gpu_probe, reference_probe가 들어 있다.
- **source mount:** `gufo/build/fn-final-src`. c97d555d의 detached worktree이고 `.gufo-revision`이 들어 있다. 결과는 `ai.tools/tmp/test-results/gufo-upstream-port-20261023/final-fn/`에 있다.
- **아래 UD-Q4 결과는 모두 `c97d555d`에서 나왔다.** EXL3 부분은 BLOCKED-loader다. `exl3_weights.cpp:168`이 텐서가 없는 UD shard 1을 거절한다. 고친 것은 efc396ba → integ b6e3ff82이고, 새 이미지에서 다시 돌린다. FAIL로 세지 않는다.

| 창 | 결과 |
|---|---|
| q1 op 단위 16개 | PASS. kvarn_ops 읽기 경로 14회 비트 동일, exl3_ops 계약 통과 |
| q2 GGUF 경로(FP16 KV) | PASS: session, snapshot, continuation adapter, MTP audit, F32 fixture dump. 첫 시도는 plan의 mount 오류였다(mmproj가 모델 옆에 없었다) |
| q4 K0 real(UD-Q4) | 아래 참조 |
| q5 KVarN 서버 suite(UD-Q4) | **expected-fail**(아래). 통과로 세지 않는다 |
| q6a UD-Q4 대 BF16 prefill | 32,760행: KL_256 0.0392(token CI [0.0378, 0.0407]), top-1 0.909, flip 2,994 |
| q7 MTP 수락(UD-Q4) | K/A/D greedy trace 완료 |
| q3, q4b, q5b, q5c, q6/q7 EXL3 절반 | BLOCKED-loader |

### b6e3ff82에서 다시 돌린 EXL3 부분 (2026-10-31, loader 수정 efc396ba) [확인]

- **이미지:** `final-b6e3ff82-tests-rocm101`(4db07a5b), `final-b6e3ff82-fn-tests-rocm101`(ca11e0e4), q6용 `final-b6e3ff82-fn-q-rocm101`(2ed9523d). q6 수집기는 0dc0a457에서 운영 toolchain으로 다시 빌드했다(0dc0a457 = b6e3ff82 + fn-quality-bf16fix, sha256 b569ac49…). 48451349의 수집기는 loader 수정 전 코드라 sidecar를 거절했다.

| 창 | 결과 |
|---|---|
| q3 EXL3 session | **FAIL-pending-fix(Codex fn-exl3-impl 담당).** `--batch-only`: C6 row 0 cycle 0에서 batched MTP frontier가 다르다(C2/C4는 정확). `--prefill-only`: chunk 분할이 logit을 바꾼다(length 136, boundary 94). MTP audit은 scalar oracle 허용치를 넘는다(recursive-carry rms 2.1e-4, split-fusion 2.2e-4, attention-state 4.2e-4; GGUF는 1e-7 수준). sampling과 execution-32k는 PASS. engine floor는 889 토큰 prompt를 써서 끝날 수 없었다(plan 오류, 중단). 다음 통합 패스에서 짧은 prompt로 다시 돈다 |
| q4b EXL3+KVarN K0 32K | PASS. real median 1.09e-6, p99 1.9e-4, max 9.9e-4. chunk 맞춘 fake는 1.02e-6, 2.7e-4, 9.3e-4. top-1 1.0, 자유 생성 동일 |
| q5c EXL3 FP16 KV long-context + cache | COMPLETE(통과) |
| q5b EXL3+KVarN suite | q5와 같은 expected-fail. 6개 suite가 같은 cache 재사용 assertion에서 멈췄다 |
| q6 BF16 대비 prefill(32,760행) | 아래 |
| q7b EXL3 MTP trace | COMPLETE |

**q6: EXL3 5.05와 UD-Q4_K_XL을 BF16과 비교(FP16 KV, MTP 없음, a01 prefill, 3중 CI)**

| | KL_256 | ΔNLL | top-1 | flip(KL>1) | KL median |
|---|---:|---:|---:|---:|---:|
| UD-Q4_K_XL | 0.0391 [0.0377, 0.0405] | +0.0061 | 0.9098 | 2,955 (56) | 0.0135 |
| EXL3 5.05 | 0.0269 [0.0259, 0.0279] | −0.0064 | 0.9209 | 2,591 (26) | 0.0092 |
| EXL3 − UD | −0.0122 (window CI [−0.0146, −0.0102]) | −0.0125 (window CI [−0.039, +0.005]) | +0.0111 (window CI [+0.0085, +0.0137]) | | |

- 범주별 KL(UD → EXL3): chat 0.0374 → 0.0265, code 0.0290 → 0.0198, en_prose 0.0388 → 0.0269, ko_prose 0.0513 → 0.0342. 모든 범주에서 EXL3가 낫다. top-1도 모든 범주에서 높다.
- top-256 질량 평균은 0.970이다(p5 0.852). KL_256은 full KL의 하한이다.
- decode는 하지 않았다(CPU BF16 decode는 며칠이 걸린다). 27B에서는 prefill과 decode가 같은 경향이었다.
- 4.05와 Kyojin 3.05는 XQ 확인 뒤 통합 패스에서 한다.

**q7: MTP greedy chain 수락 길이(depth 7, 위치마다 앞에서부터 맞은 draft 수, 정책 없음)**

| 범주 | UD | EXL3 |
|---|---:|---:|
| chat / prose / code (suite K) | 1.97 / 1.41 / 4.58 | 2.04 / 1.45 / 4.54 |
| multi / copy | 1.99 / 5.43 | 2.11 / 5.44 |
| 8K / 32K (suite D) | 2.35 / 2.07 | 2.28 / 2.19 |

- A suite도 비슷하다(예: creative 1.37 → 1.20, d4_instruction 2.74 → 3.25). 수락은 양자화에 따라 크게 달라지지 않는다.

### q6b: 4.05와 "Kyojin ~4.0 bpw" 대 BF16 (2026-11-01, `PLAN-FINAL-FN2-Q6B.json` `fn2-q6b-ef76`) [확인]

- **수집기:** `qwen38_flash_next_rq_logits` sha256 8631d444…. 브랜치는 `ai-tools/fn-quality-ef76`(ddfd1b56 = f3-integ ef76672e + fn-quality-bf16fix)이고, docker 64849ca6ec7b(/opt/rocm 7.2.4) RelWithDebInfo로 빌드해 `storage/tools/fn-quality-ef76-ddfd1b56`에 두었다. 셋 다 같은 binary로 수집했다. 5.05는 b6e3ff82 이후 prefill 경로가 바뀌어 다시 쟀다.
- UD-Q4_K_XL은 q6a의 실행을 그대로 쓴다(c97d555d, nix 수집기). 모두 FP16 KV, MTP 없음, a01 prefill 32,760행이다. CI는 window CI다.

| | KL_256 | ΔNLL | top-1 | flip(KL>1) | KL median |
|---|---:|---:|---:|---:|---:|
| UD-Q4_K_XL | 0.0392 [0.0321, 0.0466] | +0.0067 | 0.9086 | 2,994 (59) | 0.0134 |
| EXL3 5.05 (ef76672e) | 0.0269 [0.0217, 0.0322] | −0.0041 | 0.9191 | 2,649 (28) | 0.0093 |
| turboderp 4.05 | 0.0424 [0.0344, 0.0514] | −0.0284 | 0.9056 | 3,094 (79) | 0.0145 |
| Kyojin ~4.0 bpw | 0.0598 [0.0497, 0.0700] | +0.0277 | 0.8730 | 4,162 (96) | 0.0252 |

- **UD 대비 짝 비교(window CI):**
  - 5.05: KL −0.0124 [−0.0146, −0.0102], top-1 +0.0105 [+0.0067, +0.0157]. 더 낫다.
  - 4.05: KL +0.0031 [−0.0010, +0.0085], top-1 −0.0031 [−0.0067, +0.0013]. 비슷하다(구간이 0을 포함). ΔNLL은 −0.035 [−0.085, −0.003]이라 NLL은 더 낮다.
  - Kyojin: KL +0.0206 [+0.0170, +0.0243], top-1 −0.0357 [−0.0461, −0.0252]. 모든 범주에서 나쁘다(ko_prose KL 0.0784).
- 5.05의 ef76672e 값은 b6e3ff82 값(0.0269, top-1 0.9209)과 같은 수준이다. prefill 속도를 올린 경로가 품질을 바꾸지 않았다.

### KVarN K0 기준 바로잡기 (감독 수락 2026-10-31)

- **처음 기준(fake-after):** FP16으로 prefill한 뒤 양자화한다. 이 기준으로 보면 needle의 max KL이 10× 막대를 넘었다.
- **원인 [확인]:** 운영 prefill은 4096 토큰 chunk(`kPrefillChunkTokens`)마다 chunk 시작보다 앞 그룹을 레코드로 읽는다. 그래서 K/V 자체가 FP16 prefill과 달라진다. fake-after는 이것을 재현하지 않는다.
- **진단:** `kvarn_k0 --prefill-chunk 4096 --quantize-decode`(fake-chunked: chunk마다, decode 중 그룹마다 양자화)로 다시 쟀다.

| needle-32k (42 step) | fake-after | fake-chunked | real |
|---|---:|---:|---:|
| step 0 KL | 0 | 0.0478 | 0.0279 |
| median | 4.1e-6 | 5.4e-6 | 2.7e-6 |
| max | 1.7e-4 | 0.0478 | 0.0279 |

| needle-200k (42 step) | fake-after | fake-chunked | real |
|---|---:|---:|---:|
| step 0 KL | 0 | 8.6e-3 | 6.2e-3 |
| step 9 KL | 2.8e-5 | 0.0702 | 0.1476 |
| median | 3.8e-5 | 1.2e-4 | 9.6e-5 |
| max | 1.8e-3 | 0.0702 | 0.1476 |

- **세 경우 모두 같다:** top-1 일치는 모든 step에서 1.0이다(step 0과 9 포함). needle 답은 FP16, fake-chunked, real 모두 `A=418-27-9063, B=775-04-2318, C=296-61-5547`이다. real은 FP16과 토큰 단위로 같다.
- **결정:** 막대는 그대로 두고 기준만 바꾼다. 앞으로 KVarN 기준은 운영 chunk와 맞춘 fake-chunked다. real 대 FP16의 median/p99/max가 그 기준의 10× 이하, top-1 차이 1/100 이하, needle 답 정확. 이 기준으로 needle 둘 다 통과한다.
- **k0-32k, k0-256k:** real의 max(8.0e-4, 6.1e-3)는 fake-after의 max(1.7e-3, 9.6e-3)보다도 작다.
- **관찰(실패는 아님):** real이 fake-chunked의 10×를 넘는 단일 step이 두 개 있다. needle-32k step 5(3.3e-3 대 9.1e-5)와 needle-200k step 23(2.9e-3 대 1.1e-4)이다. 절대값은 작고, 분포 막대는 통과한다.

### KVarN 서버 suite: expected-fail (cdbe0563, prefix cache 없음)

- KVarN 서버는 prefix cache, snapshot, persistence를 끈다. suite마다 첫 cache 재사용 assertion에서 멈춘다. 그래서 그 뒤의 검사는 돌지 않았다. 실패한 assertion은 다음과 같다.
  - long-context: `cache_growth.py:44` `check_unchanged_retry`의 `prefilled <= mutable_suffix_tokens`(재시도가 12,425 토큰을 다시 prefill했다)
  - state-edges: `openai_sdk.py:2002` `cached_tokens > 0`
  - batch: `openai_sdk.py:689` `replay cached_tokens > 0`
  - tools: `openai_sdk.py:1524` `check_strict_tools`의 `cached_tokens > 0`
  - structured: `openai_sdk.py:1288` `schema_cache_2_alpha cached_tokens > 0`
  - sampling-defaults: `openai_sdk.py:410` `cached cached_tokens > 0`
- **이것과 별개로 답은 맞다:** long-context는 `{"keyword": "LANTERN"}`, state-edges는 맞는 tool call이었다. MTP draft도 실제로 돌았다(11개 중 10개, 39개 중 30개 수락).
- prefix cache 추가는 이후 F3 항목이다.
- functional runner는 이미지 안의 사본이다(64849ca6ec7b의 `tests/functional`).

### q2 F32 fixture: 기존 바닥, 회귀 아님

- **기준이 계산하는 것:** `qwen38_flash_next_reference_probe`(revision 8b52b42c)는 같은 UD-Q4_K_XL shard를 쓴다. 그 shard의 sha256은 storage 사본과 4/4 같다. 양자화 가중치를 F32로 풀어 CPU F32로 계산한다. fixture는 prefill pass마다 마지막 256행의 top-256 logit과 전체 vocabulary logsumexp를 담는다. KL은 top-256과 나머지 묶음으로 구한 하한이다.
- GPU 엔진은 FP16/BF16 활성값, W8A8 MMQ 활성 양자화, FP16 KV, QSA를 쓴다. 그래서 이 값은 엔진 산술 차이를 포함한 바닥이다. 같은 가중치의 1e-4 수준 floor와는 다른 양이다.
- **값:** c97d555d는 전체 KL 0.0341(median 6.6e-4), top-1 0.951, NLL 1.2049(ref 1.1079)다. tools는 KL 0.0997, p99 2.02, max 3.53이다. EXPERIMENTS.md:139-145에 기록된 main 값(0.0341, tools 0.0997, p99 2.02)과 같다.
- **pre-merge 22425173 대조:** 이미지 `fn-base-probe-rocm101`(fn-base-rocm101 + 같은 revision의 gpu_probe, `PLAN-FINAL-FN-BASE.json` `fn-q2-base-f32`)로 dump를 다시 떴다. dump 4개가 c97d555d와 **바이트 단위로 같다**(sha256 28a454a1/c108c0a7/5206cbbc/1aecc119). 따라서 GGUF 경로는 바뀌지 않았고, 이 KL은 기존 바닥이다.

### BF16 기준 (q6)

- `PLAN-FINAL-FN-BF16.json`, 이미지 7c2c5a728270(transformers 5.19.0)이다. prefill만 했다(decode는 CPU로 며칠 걸린다는 추정; 27B에서는 prefill과 decode가 같은 경향이었다).
- 32,760행, 2.84 h, peak RSS 8.2 GB, `bf16-prefill.bin` sha256 de728d88….
- 도구 수정은 브랜치 `ai-tools/fn-quality-bf16fix`에 있다(ec2d4dc1 DiskEmbedding.weight, 55500d5a logits_io prefix, 52cb1318 창 단위 저장·재개). port-u4f 병합은 감독이 한다. 앞으로 BF16 실행은 window.py 밖에서 한다(GPU 장치 없음, nice 10).
- 수집기는 `qwen38_flash_next_rq_logits` sha256 21f7dc49…다(fn-quality 48451349, 64849ca6ec7b /opt/rocm 7.2.4 gpu-test 빌드, `storage/tools/fn-quality-48451349`).

## HC int8 실험 (2026-11-01, 사용자 승인, 브랜치 `ai-tools/fn-hc-int8`) — 결과: **채택하지 않음** [확인]

- **목표:** EXL3 decode에서 HC mixer F16 가중치(down 320×10240, up 10240×320, 토큰당 96개, 1.26 GB)를 group-int8 사본으로 읽는다. 기대치는 tg +7 % 정도였다.
- **구현:** 71fa4a63, 182cf295. base는 f3-integ c8625608이다.
  - 형식은 Kyojin EXL3_GR_GS와 같다. down은 128, up은 64 입력마다 FP32 scale 하나(absmax/127)를 두고, 코드는 nearbyint로 ±127에 clamp한다(plain rounding).
  - load 때 host에서 양자화한다(`hc_int8.hpp`). HIP TU의 -ffast-math에서는 `__fdiv_rn`과 `#pragma float_control`로도 scale이 1 ulp 낮게 나왔기 때문이다(측정).
  - kernel은 `SmallGemmStable`을 따라 썼고, 가중치는 half(code·scale)로 읽는다. 단위 테스트 44개를 통과했다: F16 kernel(dequant 가중치)과 폭 1–9, 16, 17에서 비트 동일하고, 모든 행이 폭 1과 같다.
  - trunk 48층의 attn/ffn mixer만 바꾼다. head와 MTP mixer는 F16 그대로다(prompt head, head replay, prompt chunk의 MTP catch-up). snapshot payload version은 18에서 19로 올렸다. 장치 메모리는 +659 MB다.
  - Codex 리뷰 3회의 지적(measure 계산, leak, prompt 경로, snapshot version, non-finite)을 모두 반영했다.
- **품질 관문**(teacher-forced decode, a01 2048–2559, 4,096행, BF16 prefill 기준, docker RelWithDebInfo):

| | KL_256 | ΔNLL | top-1 |
|---|---:|---:|---:|
| UD-Q4_K_XL | 0.0420 | −0.0040 | 0.9004 |
| EXL3 5.05 F16 HC | 0.0234 | −0.0161 | 0.9167 |
| EXL3 5.05 int8 HC | 0.0254 | −0.0130 | 0.9167 |

  - int8 − F16HC: KL +0.0019 [+0.0003, +0.0034](막대 ≤ +0.002, **FAIL**). top-1 0.0000 [−0.0093, +0.0071](막대 ≥ −0.005, **FAIL**). ΔNLL +0.0031 [−0.0018, +0.0077].
  - int8 − UD: KL −0.0167 [−0.0201, −0.0143](**PASS**).
  - suite K MTP greedy: 평균 수락 길이는 거의 같다. 17개 중 6개 완성이 F16과 같다.
- **속도**(probe-hc-int8-docker, EXL3 5.05 + KVarN, A B A, tg128 t/s):

| | F16 HC | int8 HC | F16 HC (재측정) | 이득 |
|---|---:|---:|---:|---:|
| d0 | 24.60 | 25.56 | 24.79 | +3.5 % |
| d32K | 24.11 | 24.69 | 24.11 | +2.4 % |

  - int8 HC load는 33.6 s다(F16 30.0 s, host 양자화 +3.6 s).
- **결정(사용자 규칙):** d0와 d32K 모두 tg128 +10 % 이상이어야 기본값으로 둔다. 실제 이득은 +3.5 %와 +2.4 %라 **버린다.** g32 변형은 같은 바이트 규모라 10 %에 이를 수 없어 재지 않았다. 브랜치는 기록으로 남기고 병합하지 않는다.

## ABBA 계획 초안 (P1 정확성 통과 뒤)

- **계획:** `PLAN-F1-ABBA-BASE.json`(이미지 `f1-base-rocm101`, d2b751a9)와 `PLAN-F1-ABBA-CAND.json`(`f1-cand-rocm101`, 766a593e)이다.
- **항목:** 팔마다 `-mtp`(W9 I1 ABBA와 같은 서버 인자, A 64 / K 17 / D 8, 반복 2)와 `-ar`(D 8만, 깊은 문맥 일반 decode)가 있다.
- **순서:** BASE-mtp → CAND-mtp → CAND-mtp → BASE-mtp, 그다음 BASE-ar → CAND-ar → CAND-ar → BASE-ar다.
- **판정:** `tools/qwen-flash/spec/abba.py`로 요청·단계별 5 %·10 ms를 보고, 완성과 draft 수가 같아야 한다.
- **예상 시간 [짐작]:** mtp 창 하나에 약 20분, ar 창 하나에 약 8분이라 합계 약 2시간이다. 창 사이에 CPU 빌드는 하지 않는다.

## 남은 문제

- [짐작] P1의 표 간접 참조 비용(decode attention 행마다 표 한 번 읽기)은 ABBA로 확인해야 한다.
- K0 payload 영역 계산은 스냅샷 형식(`QFNSNAP6`, payload version 18)에 묶여 있다. 형식이 바뀌면 K0이 크기 불일치로 멈춘다(조용히 틀리지 않는다).
