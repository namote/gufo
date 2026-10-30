# F1 설계: Flash-Next n-gram 디스크 배치(F0), 페이지 KV, KVarN (2026-10-30)

상위 문서: [PLAN.md](PLAN.md) F0·F1·F2, [F1-BRIEF.md](F1-BRIEF.md), 공통 규칙 [W-COMMON.md](../rotated-quant/W-COMMON.md).
표기: **[확인]**은 코드(file:line)나 기존 측정에서 확인한 것, **[계산]**은 코드의 크기 공식으로 계산한 것(측정 아님), **[짐작]**은 추론이다.
1단계(설계, 읽기만)의 결과다. 코드, 빌드, GPU는 쓰지 않았다. 읽은 디스크는 GGUF 헤더(샤드마다 수 MB 이하)뿐이다.

## 0. 요약

1. **F0의 전제는 이미 운영에 들어가 있다.** n-gram 표(`per_layer_token_embd`, IQ4_NL, **26.82 GiB**)는 GPU에 올리지 않는다. 지금도 GGUF에서 `O_DIRECT`로 필요한 행만 읽고, 128 MiB 압축 행 캐시와 128개 reader를 쓴다. 그래서 F0에서 남은 일은 **"옮기기"가 아니라 측정**이다. 메모리 점유 표를 만들고, 디스크 조회가 decode/prefill을 막는지 확인한다. 측정만 하면 코드 변경도 빌드도 필요 없다.
2. **bpw 결정에 쓸 핵심 수치 [확인+계산]:**
   - 상주 가중치는 AR 76.73 GiB, MTP 79.31 GiB다.
   - 그중 전문가(MoE)가 71.7 GiB이고, 평균 5.10 bpw다.
   - EXL3 전문가 b bpw의 상주 크기는 약 **14.06 × b GiB + 비전문가 7.6 GiB**다. 예: 3.05 → 50.5, 4.0 → 63.8, 5.0 → 77.9 GiB.
   - n-gram 표는 bpw와 상관없이 디스크에 남는다. 그래서 상주 메모리에는 들어가지 않는다.
3. **KV는 FP16 연속 배열이고 압축하지 않는다.** brief의 `ratio_`는 KV 압축이 아니라 QSA indexer의 4토큰 pooling 비율이다(`continuation_layout.cpp:107`, `config.cpp:254`).
   - 토큰당 크기는 AR 24.75 KiB, MTP 26.81 KiB다 [확인: card 14 측정 25,348 / 27,460 B와 일치].
   - admission은 `sessions × SessionBytes(max_context)`를 시작 때 선점한다.
   - 이미지 투영기(mmproj)가 있으면 토큰당 10,252 B(최악 경우 상한)를 더 선점한다. 실제로는 쓰지 않는 몫이다.
4. **페이지 KV:** 공통 `KvBlockPool`(256토큰 블록, plane 기술)을 그대로 쓸 수 있다. K, V, pooled indexer key(`rows_per_block = 64`)를 세 plane으로 둔다.
   - 바꿀 커널은 7개 경로다. graph capture는 27B 방식(고정 크기 device 표를 제자리에서 갱신)을 재사용한다.
   - 이득은 다중 세션과 긴 `--context`에서 "쓴 만큼만 할당"하는 데서 나온다.
5. **KVarN:** 27B 커널은 KV 헤드 4, GQA 6에 묶여 있어 그대로 옮길 수 없다. head_dim 256은 맞는다.
   - Flash-Next decode는 sparse 선택(2048토큰)만 읽으므로, KVarN의 decode 속도 이득은 거의 없다 [짐작]. 이득은 메모리(약 4.3배)다.
   - 그래서 K0(fake quant 품질)부터 하고, 결과를 보고 이어갈지 정한다.
6. **W9와의 충돌:** F0은 코드 변경이 없어 충돌이 없다. P1(페이지 KV 커널·executor)은 W9 파일을 건드리지 않는다. `engine.cpp` 수정은 P2 하나에만 있고, W9 I1 ABBA 판정 뒤에 한다(5절).

## 1. F0: n-gram 표와 메모리 점유

### 1.1 지금 n-gram 표는 어디에 있나 [확인]

| 항목 | 내용 | 근거 |
|---|---|---|
| 텐서 | `per_layer_token_embd.weight` [160, 320,001,536], IQ4_NL, 행당 90 B, **28,800,138,240 B (26.82 GiB)**, 샤드 2 | GGUF 헤더 직접 읽음 |
| 해시 | PLE 층은 1, n-gram 3(bigram·trigram), 헤드 16개(ngram당 8), 헤드마다 약 20 M행이고 offset이 연속이다 | 메타데이터 `qwen4exp.ple.*`, `config.cpp:180-219` |
| 바인딩 | 헤더만 읽는다. 주석이 "read from disk only"다 | `weights.hpp:133`, `weights.cpp:271-283` |
| GPU 업로드 | `DeviceModel::Upload`의 `Model()`은 token_embd, output, hc_head, 층, MTP만 올린다. `ple_table`은 올리지 않는다 | `device_model.cpp:302-321` |
| 열기 | `NgramTable::Open`이 샤드 fd를 `ReopenRandomReadFile`로 `O_DIRECT`로 다시 연다 | `engine.cpp:127-135`, `ngram.cpp:79-118`, `mapped_file.cpp:485-488` |
| 읽기 | 행마다 4 KiB로 정렬된 `pread`를 한다. 행이 페이지 경계를 넘으면 8 KiB를 읽는다(확률 89/4096 ≈ 2 %) | `ngram.cpp:150-209` |
| 캐시 | 128 MiB 압축 행 캐시(직접 사상, 약 149만 행). 바쁜 슬롯은 건너뛴다 | `ngram.cpp:25`, `95-97`, `133-148` |
| 병렬 | reader 스레드 128개. 큰 gather(1024행 이상)는 행 번호로 정렬하고 8개씩 묶는다. 작은 gather는 캐시를 먼저 보고, 남은 행 수만큼만 reader를 깨운다 | `ngram.cpp:22-24`, `211-239`, `319-345` |
| 중복 제거 | gather 안에서 같은 행은 한 번만 읽는다(평탄 해시표) | `ngram.cpp:281-318` |
| 페이지 캐시 | 표는 페이지 캐시를 거치지 않는다. 가중치 업로드(`weight_upload.cpp:30`, 직접 reader)와 artifact identity 해시(`gguf_identity.cpp:121-157`, unbuffered, digest 캐시)도 그렇다. GGUF 전체 mmap은 `MAP_PRIVATE`이고, 표 영역은 건드리지 않으므로 RSS에 들어가지 않는다 [짐작: 표 영역을 읽는 호스트 코드는 CPU reference뿐이다] | `mapped_file.cpp:262-270` |
| 호스트 메모리 | 행 캐시 128 MiB, pinned mapped 버퍼 `T × 2560 × 4 B`(T = 4224 → 41.25 MiB), prefetch 버퍼(같은 크기, pageable, 첫 사용 때 할당), reader 스택(가상 메모리) | `executor.cpp:1110-1131`, `2270-2294`, `engine.cpp:30-33`, `246-249` |

이 경로의 이력은 [EXPERIMENTS.md](../../docs/models/qwen3.8-flash-next/EXPERIMENTS.md)에 남아 있다: 245(gather scratch 재사용), 291(pinned 버퍼를 GPU가 직접 읽기), 358(다음 청크 prefetch), 370(reader 128개).
card 14(`docs/cache-redesign/cards/14-flash-next-adapter.md:187-189`)는 상주 모델 바이트를 AR 82,384,141,824 B, MTP 85,157,259,776 B로 쟀다. 그리고 "PLE ngram table uses the existing bounded DirectIO path"라고 적었다.
**검산 [계산]:** trunk GGUF 103.68 GiB에서 PLE 26.82 GiB를 빼면 76.86 GiB다. 측정값 76.73 GiB와 맞는다(차이는 F32→F16 변환으로 준 몫이다, `device_model.cpp:439-445`).

### 1.2 조회 경로 (토큰마다 몇 행, 언제) [확인]

- **토큰당 16행**(헤드 16개)이다. 1층(PLE)에서 쓴다.
- **decode / verify (n ≤ 8):**
  - `Forward`가 `PleFetch`로 n×16행을 해시하고 `StartRead`를 건다(`executor.cpp:3125-3133`).
  - graph 경로는 0층을 prefix graph로 먼저 큐에 넣는다. 그다음 `WaitPle`로 기다리고 나머지 graph를 띄운다(`executor.cpp:3175-3198`).
  - 그래서 디스크 읽기를 숨길 수 있는 시간은 **0층 하나와 호스트 준비 시간**뿐이다.
  - MTP verify는 (k+1)×16행을 읽는다. 투기 경로는 토큰별 n-gram 스냅샷도 남긴다(`executor.cpp:2219-2229`).
- **prefill:** 다음 청크의 행을 지금 청크의 forward 동안 미리 읽는다(`PrefetchPle`, `executor.cpp:2270-2294`, `2307-2312`). 다만 첫 청크는 자기 gather를 기다린다(`ngram.cpp:19-21` 주석).
- **다중 세션 decode 배치:** 세션별 행을 한 gather로 모아 읽고, 0층 뒤에 기다린다(`batch.cpp:884-919`, `960`).
- **텔레메트리:** 요청마다 `ngram_rows`, `ngram_dedup_rows`, `ngram_cache_hits`, `ngram_disk_reads`가 서버 timing에 나온다(`generation_metrics.hpp:191-195`, `text_generation_scheduler.cpp:958-962`). cold/warm은 이 값으로 구분한다.
- **cold/warm의 뜻:** `O_DIRECT`라 OS 페이지 캐시는 영향이 없다 [짐작: 파일 시스템이 direct I/O를 지원한다는 전제. 지원하지 않으면 buffered duplicate로 바뀐다, `ngram.cpp:104-107`]. 그래서 "cold"는 새 프로세스(행 캐시가 빈 상태), "warm"은 같은 요청을 반복한 상태로 정의한다. 디스크 쪽 상태로는 dm-crypt 위 NVMe임을 함께 적는다(`PROMPT-CHECKPOINTS.md:213-217`, ksoftirqd 관찰).

### 1.3 F0에서 할 일

**결론:** n-gram 표를 디스크로 옮기는 구현은 필요 없다. 운영 경로가 이미 Kyojin·upstream EXL3와 같은 방식(파일에서 필요한 행만 읽기)이다.
upstream은 `ngram_embedding.safetensors` 32.6 GB를 따로 둔다. gufo는 GGUF 안의 IQ4_NL(26.8 GiB)을 그대로 읽는다. [짐작] upstream 파일은 행당 약 102 B로, 다른 4비트 계열 형식으로 보인다. 확인하지 않았다.

F0은 다음 두 가지 측정으로 정의한다.

1. **F0-M 메모리 점유 표**(1.4)
2. **F0-S 디스크 조회 정체 측정:** decode 토큰마다 `WaitPle`에서 GPU가 쉬는 시간과, prefill 첫 청크의 대기 시간을 잰다(1.5).

F0-S에서 정체가 decode 시간의 1 % 이상 나오면 개선 후보를 낸다. 후보는 다음과 같다.
- (a) MTP verify에서는 드래프트 토큰이 verify 전에 정해진다. 그 행을 드래프트 중에 미리 읽는다 [짐작: 숨길 시간이 0층에서 드래프트 시간 전체로 늘어난다].
- (b) 행 캐시를 키운다.
- (c) 정체가 dm-crypt softirq 때문이면 측정 노트로만 남긴다.

후보를 구현하면 그때 PLAN의 F0 통과 기준을 적용한다: greedy 바이트 동일, decode·prefill이 main 대비 5 %·10 ms 안. 측정만 하면 운영 바이너리가 그대로이므로 동일성은 당연히 성립한다.

### 1.4 메모리 점유 표 설계

**공식(코드에서 읽음) [확인]:**

- **상주 가중치:** `DeviceModel::resident_bytes()`(`engine.cpp:270-272`, ResourceClaim `inference_backend.cpp:2631`)
- **세션 상태:** `Executor::SessionBytes`(`executor.cpp:1473-1512`). 항목은 다음과 같다.
  - KV: 층마다 `2 × ctx × 512 × 2 B`. attention 12층, MTP면 +1층
  - pooled key: 층마다 `(ctx/4 + 1) × 128 × 2 B`
  - raw indexer ring: 층마다 `bit_ceil(min(ctx, 2048 + 4224)) × 128 × 4 B`. 긴 문맥에서는 8192행 = 4 MiB
  - DeltaNet 36층: 상태 3 MiB, conv 120 KiB
  - PLE conv 기록
  - rollback: depth 6. `GdnRollbackRowFloats`(`kernels.hpp:28-34`)로 첫 행만 전체 상태이고, 나머지는 작다
  - MTP: target hidden, draft 표 1024행
- **이미지:** `ctx × (2560 × 4 + 12) B`(`engine.cpp:274-286`). **선점만 하는 최악 경우 상한이다.** 실제 할당은 위치 12 B/토큰과 이미지 행의 embedding뿐이다(`device_input.hpp:43-51`).
- **공유 scratch:** `DeferredScratchBytes`(`executor.cpp:1515-1537`)와 executor 생성 때의 scratch(`max_batch` = 4224행 기준)
- **admission:** `sessions × SessionBytes`가 `hipMemGetInfo` free − deferred 안에 들어가야 서버가 시작한다(`text_model_runner.cpp:264-273`). 이것은 할당이 아니라 시작 때의 거절 검사다. 실제 세션 할당은 CreateSession에서 full context로 한다(`executor.cpp:1257-1340`).

**계산값 (MTP, 세션 하나) [계산]:**

| context | KV(13층) | pooled+ring | DeltaNet+rollback | 이미지 선점 | 세션 합계(이미지 없음 / 있음) |
|---:|---:|---:|---:|---:|---:|
| 32,768 | 832 MiB | 78 MiB | 약 0.23 GiB | 320 MiB | 약 1.12 / 1.43 GiB |
| 131,072 | 3.25 GiB | 156 MiB | 약 0.23 GiB | 1.25 GiB | 약 3.64 / 4.89 GiB |
| 262,144 | 6.50 GiB | 260 MiB | 약 0.23 GiB | 2.50 GiB | 약 6.98 / 9.48 GiB |

**측정할 표 (F0-M 산출물):**

| 행 | 값 | 출처 |
|---|---|---|
| 상주 가중치, 종류별 | trunk dense(Q8_0 등), 전문가(Q4_K/Q5_K/Q5_1/Q8_0), 헤드·임베딩, MTP 사이드카, 비전 인코더 | GGUF 헤더 집계(아래) + `phase=target_weights/draft_weights` 로그의 hipMemGetInfo 차이 |
| n-gram | 디스크 26.82 GiB, 호스트 상주(행 캐시 + pinned + prefetch) | `rss_mib` 차이와 코드 상수 |
| 세션 수별 증가 | S = 1, 2, 4, 8에서 `phase=sessions` 전후와 `load_completed gpu_device_used_mib` | 서버 로그(`inference_backend.cpp:3562`, `serve.cpp:84-88`) |
| 요청 중 최대값 | pp2048+tg128, 32K prefix + pp4096 + tg128 | `tools/gufo/model_bench/llm.py:718-775` 표본기(250 ms) |
| 스냅샷/RAM 캐시 | `HostSnapshotBudgetBytes` 설정값, 장치 쪽 스냅샷 slack 512 MiB | `inference_backend.cpp:2638-2639`, `executor.cpp:162-164` |

**지금 GGUF 헤더 집계 (UD-Q4_K_XL, trunk 103.68 GiB) [확인]:**

| 종류 | 크기 |
|---|---:|
| PLE 표 (IQ4_NL, 디스크) | 26.82 GiB |
| 전문가 down Q5_1 / Q8_0 | 25.20 / 4.15 GiB |
| 전문가 gate·up Q4_K / Q5_K | 41.31 / 1.07 GiB |
| attention·DeltaNet dense (Q8_0) | 2.63 GiB |
| 헤드 + 임베딩 (Q8_0) | 1.26 GiB |
| HC mixer (Q8_0) | 0.63 GiB |
| 공유 전문가 (Q8_0) | 0.23 GiB |
| 라우터, α/β, indexer 등 (F32/BF16) | 약 0.32 GiB |
| MTP 사이드카 (Q8_0 위주) | 2.58 GiB |

**F2 bpw 결정용 표 [계산, 측정 뒤 갱신]:**
- 전문가 매개변수는 512 × 3 × 2560 × 640 × 48층 = 120.8 B개다.
- 비전문가 상주(MTP 포함)는 약 7.6 GiB로 둔다. dense도 EXL3로 바꾸면 더 줄어든다.
- HIP에 보이는 메모리는 약 124 GiB다(card 14:147).

| 전문가 bpw | 전문가 GiB | 상주 합계 GiB | 남는 GiB (124 − 합계 − scratch 약 2 [짐작]) | 262K MTP 세션 수(FP16 KV / KVarN k4v2 [계산]) |
|---:|---:|---:|---:|---|
| 지금 5.10 | 71.7 | 79.3 | 약 42 | 6 / 약 15 |
| 3.05 | 42.9 | 50.5 | 약 71 | 10 / 약 26 |
| 4.00 | 56.3 | 63.8 | 약 58 | 8 / 약 21 |
| 5.00 | 70.3 | 77.9 | 약 44 | 6 / 약 16 |
| 6.00 | 84.4 | 92.0 | 약 30 | 4 / 약 11 |

- 세션 수는 배치 폭 상한 8(`inference_backend.cpp:2645-2653`, decode concurrency 1–8 `inference_backend.cpp:3325-3326`)과 무관하게 메모리만으로 계산했다. 이미지 선점은 뺐다.
- KVarN 열은 3.2절의 토큰당 6.3 KB를 쓴다 [짐작].

### 1.5 F0 측정 방법 (GPU 창 1개, 코드 변경 없음)

- **이미지:** `ai-tools/gufo-eval:rqw3-base-rocm101`(gufo `c94572b3`, ROCm 10.1 / Clang 24 release, W9 M0 기준선과 같다).
  - 메모리 공식은 c94572b3와 HEAD 사이에 바뀌지 않았다 [확인: 그 사이 Flash-Next 변경은 `engine.cpp` I1(`PrepareDecode`)과 `mtp_policy.hpp`뿐이다. `git log` 기준].
  - 그래서 빌드가 필요 없다.
- **모델:** NVMe `~/projects/ai.tools/storage/models/llm/qwen-3.8-flash-next-ud-q4-k-xl-gguf/`를 `:ro`로 마운트한다. mmproj는 외장을 `:ro`로 쓴다(행 A6 하나).
- **F0-M 구성** (서버 실행마다 로그를 남기고, 요청 2개에 표본기를 붙인다):
  - A1: AR S1, ctx 262,144
  - A2: MTP S1, 262,144
  - A3: MTP S2, 131,072
  - A4: MTP S4, 65,536
  - A5: MTP S8, 32,768
  - A6: A2 + mmproj(이미지 선점 확인)
  - A7: MTP S8, 262,144(거절 경계 확인. 계산상 8 × 6.98 GiB가 남는 약 42 GiB를 넘으므로 시작 거절을 기대한다. 거절 메시지를 기록한다)
  - 각 실행에서 `cache_prompt: false`로 같은 요청을 두 번 보내 cold/warm `ngram_*` 수치를 기록한다.
- **F0-S:** A1, A2 서버에서 `tools/prof/prof.py`(pipeline/wall-time)로 tg128 d0과 d32K, pp4096 첫 청크를 잡는다.
  - decode에서 0층 prefix graph 끝부터 suffix graph 시작까지의 GPU 공백을 잰다. 이 공백이 `WaitPle` 대기다.
  - prefill은 첫 청크의 커널 시작이 `StartRead` 뒤로 얼마나 밀리는지 잰다.
  - trace 커널 이름과 단계 구분은 첫 결과를 보고 맞춘다.
- **예상 시간 [짐작]:** NVMe 로드 약 71 s(W9 I1F2) × 7회와 요청·profile을 합쳐 약 30분이다. seconds는 3600으로 잡는다.
- **창 계획:** `PLAN-F1-F0.json`(ai.tools `test/experiments/gufo_upstream_port_20261023/`). 감독 승인 뒤에 쓰고, LOCKS 반영은 감독이 한다.
- **산출물:** `plan/exl3/F1-RESULTS.md`에 두 표(1.4 측정 표와 bpw 표 갱신)를 쓴다. 원시 로그는 ai.tools `tmp/test-results/…/f1-f0/`에 둔다.
- **창 안의 순서:** 메모리는 타이밍과 무관하다. 다만 F0-S profile은 타이밍이므로 다른 W의 CPU 빌드와 겹치지 않게 한다(메모리 규칙 "GPU timing 중 빌드 없음").

## 2. Flash-Next KV 구조 [확인]

### 2.1 attention 종류와 행 형식

| 항목 | 값 | 근거 |
|---|---|---|
| 층 | 48층. 36층은 Gated DeltaNet(KV 없음, 고정 크기 상태), 12층은 full attention(4층마다) | `config.cpp:254-262`, 메타데이터 `full_attention_interval 4` |
| attention | 24 query 헤드, **2 KV 헤드**, head_dim 256, rotary 64, q에 출력 gate | `config.cpp:256-258` |
| KV 행 | K, V가 각각 `[max_context][kv_heads × 256]` FP16, 행 1024 B, **층마다 연속, 세션 생성 때 max_context 전체를 할당** | `executor.hpp:153-158`, `executor.cpp:1288-1305` |
| QSA indexer | 4 헤드 × 128. raw key는 F32 ring(`bit_ceil(min(ctx, 2048 + max_batch))`행), pooled block key는 F16 `[ctx/4 + 1][128]`(4토큰 평균 + RMSNorm + rope) | `executor.cpp:145-149`, `1299-1305`, `kernels.hpp:440-455` |
| sparse 선택 | 위치가 `indexer_top_k` = 2048을 넘으면 `SelectBlocks`가 블록 점수 상위 512개(= 2048토큰)를 고르고 마스크를 만든다. 그 전에는 dense다 | `executor.cpp:2552-2605`, `kernels.hpp:457-472` |
| MTP | draft block 하나(full attention). trunk와 같은 행 형식의 자기 KV, indexer, ring | `executor.cpp:1315-1335` |
| `ratio_` | `ContinuationLayout::ratio_` = `compress_ratio` = 4. **indexer pooling 비율이다. KV 압축은 없다.** KV 타입은 FP16 하나뿐이다 | `continuation_layout.cpp:107`, `src/cli/kv_cache_type.cpp:22-23` |

**토큰당 바이트 [확인]:** 12 × (1024 + 1024) + 12 × 256/4 + 4(토큰) = 25,348 B(AR). MTP는 2,112 B를 더해 27,460 B다. card 14 표(`14-flash-next-adapter.md:173-178`)의 Append bytes / Tokens와 정확히 같다. README의 27.5 KiB는 여기에 rollback 같은 몫을 더한 측정값이다.

### 2.2 KV를 읽고 쓰는 커널 (전부)

| # | 커널(launcher) | 하는 일 | 호출 위치 |
|---|---|---|---|
| 1 | `PrepareAttention` | Q/K norm + rope + **K/V 캐시 쓰기**(decode, 좁은 배치, MTP) | `kernels.hpp:418-428`, `executor.cpp:2508-2514`, `batch.cpp:315-318`, `executor.cpp:4525` |
| 2 | `AttentionF16Gemm` | 넓은 prefill의 QKV 투영과 norm·rope·**K/V 쓰기** 융합(1024토큰 이상) | `kernels.hpp:205-218`, `executor.cpp:2487-2504` |
| 3 | `StoreKv` | 분리 경로의 K/V 쓰기 | `kernels.hpp:436-439`, `executor.cpp:2542-2543` |
| 4 | `Attention` | 토큰별 attention(decode, 8행 이하). split 8, sparse list 압축 | `kernels.hpp:483-497`, `executor.cpp:2660-2665` |
| 5 | `WmmaCausalAttention` | 넓은 배치 fused WMMA(점수, softmax, PV, gate). dense와 sparse 창 | `kernels.hpp:505-520`, `executor.cpp:2622-2641` |
| 6 | `PoolIndexerBlocks` | raw ring을 읽어 **pooled key를 쓴다** | `kernels.hpp:440-455`, `executor.cpp:2566-2569` |
| 7 | `SelectBlocks` | pooled key를 읽어 점수를 매기고 마스크를 만든다 | `kernels.hpp:457-472`, `executor.cpp:2590-2597` |

- 다중 세션 decode 배치(`batch.cpp:1043`)와 MTP 드래프트/catch-up(`batch.cpp:364-373`, `executor.cpp:4489-4525`)도 같은 `Executor::Attention`으로 위 커널을 부른다. 그래서 바꿀 곳은 이 7개와 그 호출자로 닫혀 있다.
- raw ring(`StoreRows`)은 세션 전용이고 크기가 정해져 있다(최대 8192행). 그래서 풀에 넣지 않는다.

### 2.3 되감기, 스냅샷, 디스크 캐시

- **MTP 되감기:** KV는 `position_`만 줄인다. 거절된 행은 남아 있다가 덮어쓰인다. pooled 블록 수는 `min(blocks_, position/4)`로 줄인다. 상태가 있는 것(DeltaNet, conv, PLE 기록, n-gram 기록)만 스냅샷 행에서 되돌린다(`executor.cpp:3402-3470`).
- **prefill checkpoint:** 소스 세션의 KV 포인터를 그대로 공유하는 view다. append-only 전제다(`executor.cpp:746-790`, 특히 771).
- **스냅샷(서버가 지금 쓰는 legacy):** `SaveSnapshot`/`RestoreSnapshot`이 `WalkSnapshot`의 영역 목록을 쓴다(`executor.cpp:3564-3681`).
  - 영역 순서: DeltaNet 상태 → PLE → 층마다 K[0, pos), V, indexer 꼬리(ring wrap을 시간순으로 풂), pooled → MTP.
  - 지연 보존: 스냅샷 뒤 세션이 스냅샷 위치 아래를 덮어쓰려 하면, `PreserveSnapshots`가 그 영역을 장치 저장소로 먼저 복사한다(copy-on-overwrite, `executor.cpp:560-655`).
  - 디스크 캐시는 이 payload(`kSnapshotMagic "QFNSNAP6"`, `executor.cpp:3472-3473`)를 쓴다.
- **continuation adapter(card 14, 서버에는 아직 붙지 않음, `14-flash-next-adapter.md:145`):** component마다 연속 포인터를 돌려준다(`continuation_adapter.cpp:357-378`). `BeforeWrite` hook은 `Forward`와 `Rollback`에서 부른다(`executor.cpp:3112-3118`, `3410-3415`).

### 2.4 admission [확인]

- `ResourceClaim.per_request_state_bytes = SessionBytes(mode, max_context)`(`inference_backend.cpp:2619-2643`)
- 시작 때 `sessions × 요청당 바이트 ≤ hipMemGetInfo free − deferred`를 검사하고, 넘으면 시작을 거절한다(`text_model_runner.cpp:264-273`).
- 실제 세션 객체는 full `max_context`로 할당한다. 그래서 짧은 요청만 오는 서버도 `--sessions × --context`만큼 메모리를 쓴다.

## 3. 설계

### 3.1 페이지 KV

**공통 코드 [확인, 27B 조사]:**
- 블록은 256토큰이다(`kv_block_layout.hpp:17-23`).
- `KvBlockPlane{layers, row_bytes, rows_per_block}`이고, 배치는 `[layer][block][rows][bytes]`다(`:28-58`).
- 풀은 한 번 할당하고 크기를 바꾸지 않는다(`kv_block_pool.cpp:11-20`).
- 할당기는 블록별 참조 수를 세고, 예약(RAII)이 있고, 부족하면 `length_error`를 던진다(`kv_block_layout.hpp:134-283`).
- 커널에는 `KvBlockTable{const uint32_t* blocks; layer_tokens}`를 값으로 넘긴다. 행 위치는 `KvTokenRow`로 구한다(`kv_block_table.hpp:20-53`). 블록 번호는 uint32이고, null이면 연속 배치다.
- 모델에 묶인 가정은 없다.

**plane 구성 (제안):**

| plane | layers | row_bytes | rows_per_block | 블록당 바이트 |
|---|---:|---:|---:|---:|
| K | 12 (+1 MTP) | 1024 | 256 | 256 KiB × 층 |
| V | 12 (+1 MTP) | 1024 | 256 | 256 KiB × 층 |
| pooled indexer key | 12 (+1 MTP) | 256 | 64 | 16 KiB × 층 |

- 블록 하나(MTP 13층)는 6.71 MB = 6.4 MiB다 [계산].
- MTP 층을 trunk와 같은 블록 표에 넣는다. draft 행도 같은 토큰 위치를 쓰고(frontier만 한 칸 늦다), 표가 하나면 스냅샷과 되감기가 단순해진다 [짐작: 27B MTP는 자기 연속 KV를 쓴다(조사 B2). Flash-Next에서는 draft KV도 문맥 길이만큼 자라므로 풀에 넣어야 이득이 있다].
- MTP를 로드한 서버의 AR 세션은 MTP 층 몫(1/13)을 낭비한다. 측정 뒤 plane 세트를 둘로 나눌지 정한다.
- raw indexer ring, DeltaNet 상태, PLE 기록, rollback은 세션 전용으로 둔다. 크기가 정해져 있고 문맥에 비례하지 않는다.

**커널 변경 (2.2의 7개):**
- 쓰기(1, 2, 3, 6): `cache + KvTokenRow(table, layer, …, pos) × row` 형태로 바꾼다. 층 포인터 대신 plane 기준 주소 + 층 번호를 넘긴다.
- 읽기(4): 토큰별 행 조회다. split-K tile이 블록 경계를 넘을 수 있으므로 `KvBlockEnd`로 tile을 나눈다(27B verify wave64 방식, 조사 B2 `attention_decode_verify_wave64.hip:80-93`).
- 읽기(5): WMMA tile이 256을 나누어야 한다. 그래야 tile 하나가 블록 하나에 들어간다. `kernels.hip.cpp`의 attention tile 상수를 1단계 구현 첫 작업으로 확인한다. sparse 4토큰 블록은 256토큰 블록 안에 늘 들어간다.
- 읽기(7): pooled 행은 `(blocks[b >> 6] << 6) + (b & 63)`로 찾는다. 공통 helper는 토큰당 한 행만 다루므로, 27B KVarN처럼 Flash-Next 쪽에 작은 helper를 둔다(조사 C2 `attention_kvarn.hip:58-65`).
- **graph capture:** 세션마다 `BlocksFor(max_context)`개짜리 device 표를 생성 때 한 번 할당한다. 커지면 pinned 호스트 사본에서 `hipMemcpyAsync`로 제자리 갱신한다. 그러면 표 포인터가 바뀌지 않아, 캡처한 decode graph가 어느 깊이에서도 다시 돈다. 27B와 같은 방식이다(조사 A2 `arena.cpp:1210-1221`, `1340-1361`).
  - Flash-Next는 깊이와 무관하게 n ≤ 8 decode를 캡처한다(`executor.cpp:3157-3160`). 그래서 이 조건은 필수다.
  - 아직 매핑되지 않은 항목은 0번 블록을 가리켜, 마스크된 읽기도 풀 안에 머물게 한다(27B 방식).

**되감기, 스냅샷, 디스크 캐시:**
- **되감기:** 위치만 줄이고 블록은 매핑된 채로 둔다. 바꿀 것이 없다.
- **legacy 스냅샷:** `WalkSnapshot`의 K/V/pooled 영역을 블록 단위 조각의 연속으로 바꾼다. payload 바이트 순서(시간순 행)는 그대로 둔다. 그러면 `QFNSNAP6` 형식과 디스크 캐시 호환이 유지되고, 1단계에서 형식을 바꿀 필요가 없다. 지연 보존(`PreserveSnapshots`)도 영역 단위에서 블록 조각 단위로 바뀌기만 한다.
- **2단계(공유):**
  - 다 찬 블록은 참조 수로 공유한다(checkpoint, 접두 재사용). 덮어쓰기 전에 COW를 한다(27B `UnshareKvRows`, `RetainRows`, 조사 B3).
  - Flash-Next의 append-only checkpoint view(`executor.cpp:771`)는 같은 블록 표 사본으로 바뀐다.
  - 마지막의 일부만 찬 블록은 복사한다(27B와 같다).
- **continuation adapter:** `RowData`의 연속 포인터를 블록 매핑으로 바꾼다(27B `continuation_adapter.cpp:993-1112`가 선례다).

**admission (2단계):**
- 27B처럼 서버 전체가 공유 풀 하나를 쓴다.
- 크기는 `--kv-pool-bytes`(기존 플래그, `serve.cpp:720`)나 `sessions × BlocksFor(max_context)`로 정한다.
- 요청마다 `BlocksFor(prompt + max_new) + 2`를 예약하고, 쓸 때 할당한다(27B `ReserveKvRows`/`EnsureKvRows`, 조사 B1).
- 블록이 부족하면 요청은 큐에서 기다린다. 선점은 없다(27B 정책을 그대로 쓴다).
- `SessionBytes`에서 KV·pooled 몫을 빼고, 풀 바이트를 따로 청구한다.
- 이미지 선점(10,252 B/토큰)도 실제 사용량 기준으로 바꾼다. 다만 이것은 별도 커밋으로 하고, 범위는 감독이 정한다.

**재사용 판단:** 블록 크기, 할당기, 표 helper, 그래프 방식, 서버 예약 흐름은 그대로 쓸 수 있다. attention 커널은 Flash-Next 전용이다(24/2/256, sparse 마스크). 그래서 주소 계산만 바꾼다.

### 3.2 KVarN

**27B 구현 [확인, 조사 C]:**
- 형식은 `kvarn-k4v2`(K 4비트 token-major, V 2비트 channel-major)다.
  - 그룹은 128토큰이고, (그룹, KV 헤드)마다 레코드가 하나다.
  - scale/zero는 FP16이다. 앞 128토큰(sink)과 256토큰 꼬리 ring은 FP16으로 남긴다.
  - 256점 Walsh-Hadamard를 K와 V 양쪽에 쓰기 때문에 적용한다. 근거는 `kvarn_layout.hpp:22-90`, `attention_kvarn.hip:128-163`, `891-960`, `1018`이다.
- 상수가 27B에 묶여 있다: `kKvarnKvHeads = 4`, `kKvarnQueryHeads = 24`, GQA 6(`kvarn_layout.hpp:36-39`, `attention_kvarn.hip:24-26`). 쓰기 커널의 스레드 배치도 4헤드 × K/V × 32다.
- 호스트 planner(`PlanKvarnPass`, `PlanKvarnPrefill`, `KvarnReadGroups`, checkpoint 행 규칙)는 그룹, ring, sink 상수에만 의존한다(`kvarn_layout.hpp:166-316`, `kvarn_checkpoint.hpp:16-51`).

**Flash-Next 적합성:**
- **맞는 것:** head_dim 256(Hadamard 256점 그대로), 24 query 헤드, KVarN 레코드 plane을 풀 블록에 넣는 방식(`rows_per_block = 2`).
- **다른 것:** KV 헤드가 2개이고 GQA가 12다.
  - 커널을 KV 헤드와 GQA로 템플릿화해야 한다.
  - 쓰기 커널은 2헤드 × K/V × 32 = 128 스레드가 되거나, 블록 하나가 그룹 둘을 맡는다.
  - WMMA prefill pack은 M 패딩이 바뀐다.
- **sparse와 맞물림 (설계 위험):**
  - Flash-Next decode는 선택된 4토큰 블록(512개, 2048토큰)만 읽는다. 그래서 128토큰 그룹 안의 임의 4토큰을 꺼내야 한다.
  - K는 token-major라 4토큰 연속 읽기가 된다.
  - V는 channel-major라 4토큰을 꺼내려면 채널마다 띄엄띄엄 읽어야 한다. 그래서 V 쪽 대역폭 효율이 떨어진다 [짐작].
  - 대안은 Flash-Next용 V token-major 레코드다. 27B와 형식이 갈라지므로, K0 뒤 커널 단계에서 정한다.
  - indexer pooled key(64 B/토큰/층)와 raw ring은 FP16/F32로 남긴다.
- **기대 이득:**
  - 메모리 [계산]: k4v2 레코드는 토큰당 층마다 420 B다(2헤드 기준, 27B 840 B의 절반). 13층 × 420 + pooled 832 + 토큰 4 ≈ **6.3 KB/토큰**이다. FP16 27.5 KB 대비 약 4.3배 작다. sink와 꼬리 FP16(세션당 384토큰 × 2 KiB × 13 ≈ 10 MiB)은 따로 든다.
  - 속도 [짐작]: decode가 읽는 KV는 깊이와 무관하게 약 2048토큰 × 2 KiB × 12 = 48 MiB/토큰이다. 그래서 KVarN의 decode 이득은 작다. 깊은 문맥 decode 비용은 오히려 `SelectBlocks`가 전체 pooled key를 훑는 데 있다(256K에서 64K 블록 × 256 B × 12 ≈ 192 MiB/토큰).
  - 결론: 이득은 메모리(세션 수, 문맥)이고, 속도는 회귀만 막으면 된다.
- **공통으로 옮기기(`src/core/`):** 호스트 planner와 레코드 배치(헤드 수를 인자로)는 옮긴다. 커널은 템플릿화한 뒤에 옮긴다. 27B 경로는 같은 템플릿 인자로 비트 단위 동일성을 유지해야 한다. 이것이 27B 회귀 위험이다.

**품질 기준과 측정 (K0 방식을 따른다):**
- **K0-FN(오프라인 fake quant):**
  - 엔진 안에서 `K' = dequant(quant(K·H))·H`로 바꿔 넣고 teacher forcing을 한다.
  - 지표: 단계별 KL(중앙값, p95, 최대), top-1 일치, free greedy가 처음 갈리는 지점, gold 답, K/V 복원 오차(`plan/gufo-upstream-port/KVARN.md:88-109`).
  - 참조는 `tests/models/qwen27b/kvarn_fake_quant.hpp`를 쓴다. 공통 헤더로 옮기거나 Flash-Next 테스트에서 include한다.
  - 입력: 27B K0와 같은 32K, 256K gold 문서. 27B K0의 한계("too easy", `KVARN.md:157`)를 보완하려고 **needle(줄 검색) 256K**를 더한다.
- **추가 확인:** R0 하네스의 KL(FP16 KV 대비)과 ΔNLL, 위치별 꼬리. 그리고 sparse 선택이 KVarN 때문에 바뀌는 비율. indexer는 FP16이라 선택은 같아야 한다. 같지 않으면 원인을 본다.
- **판정 기준:** K0-FN 결과를 보고 사용자와 정한다. 27B 선례는 K0 뒤 기준을 정했다(`KVARN.md:351-359`). 커널 단계에서는 real‖fake ≤ 측정한 rounding 대조의 10배, top-1 차이 1/100 이내, 서버 gold 일치(`KVARN.md:228`, `267`, `316-334`)를 그대로 쓴다.

## 4. 구현 단계, 통과 기준, 예상 크기

| 단계 | 내용 | 파일(예상) | 통과 기준 | 크기 [짐작] | GPU 창 |
|---|---|---|---|---|---|
| **F0** | 측정만(1.4, 1.5). 정체가 1 % 이상이면 개선 후보를 따로 제안한다 | 없음(`plan/exl3/F1-RESULTS.md`, `PLAN-F1-F0.json`) | 표 완성, 세션 수별 증가가 공식과 ±1 % 안, cold/warm `ngram_*` 기록, 정체 시간 보고 | 문서만 | 1개, 약 30분 |
| **P1** | 페이지 KV, 세션 전용 풀(공유 없음). 7개 커널을 `KvBlockTable` 주소로, 스냅샷 블록 조각화(payload 형식 유지), adapter `RowData` 블록화 | `kernels/rocm/{kernels.hpp, kernels.hip.cpp, executor.hpp, executor.cpp, batch.cpp}`, `continuation_adapter.cpp`, 테스트 `attention_ops_test`/`snapshot_test`/`session_test`/`continuation_adapter_test` 확장 | (a) 순서를 섞은 블록 표 대 연속 배치의 op 테스트 출력이 비트 단위로 같다. (b) greedy 출력이 연속 FP16과 바이트 단위로 같다(W9 세트 A/K/D 89개 sha). (c) `long-context`, `cache` 통과. (d) main과 짝지은 pp2048/tg128 d0/32K(AR, MTP)에서 요청·단계별 5 %·10 ms 회귀 없음 | 600–900줄 | 정확성 1개(약 40분) + ABBA 1개 |
| **P2** | 공유 풀, 쓴 만큼 할당하는 admission, `--kv-pool-bytes`, 다 찬 블록 공유와 COW, 이미지 선점 실사용화(별도 커밋) | P1 파일 + `engine.{hpp,cpp}`(`SessionBytes`, 풀 연결), `src/cli/serve/inference_backend.cpp`(Flash-Next runner), 테스트 | P1 (b)–(d) + 다중 세션 메모리 이득 측정(같은 `--sessions`·`--context`에서 짧은 요청의 상주 감소, 더 많은 세션 시작) + 블록 부족 때 대기 동작 기능 테스트 | 500–800줄 | 정확성 + 메모리 1개, ABBA 1개 |
| **K0-FN** | KVarN fake quant 품질(GPU, 커널 변경 없음) | `tests/models/qwen38_flash_next/kvarn_k0.cpp`(새 파일), 참조 헤더 공유 | 표 완성(32K, 256K, needle). 기준은 사용자와 정한다 | 300–500줄 | 1개, 약 40분 |
| **K1–K3** | KVarN 커널 템플릿화와 공통화, Flash-Next 쓰기·decode·prefill, sparse 접근(V 배치 결정), CLI `kvarn-*`를 `qwen4exp`에 허용 | `src/core/…/kvarn*`, `src/models/qwen/hip/kernels/attention_kvarn.hip`(27B 비트 동일 유지), Flash-Next 커널, `src/cli/kv_cache_type.cpp` | 27B: 기존 KVarN 테스트와 출력이 비트 단위로 같다. Flash-Next: real‖fake 기준, needle/gold, `long-context`/`cache`, 속도 회귀 없음, 메모리 측정 | 1,500–2,500줄 | 여러 개(단계마다 1–2개) |

순서: F0 → P1 → P2. K0-FN은 P1과 병행할 수 있다(코드 경계가 다르다). K1은 K0-FN 판정 뒤에 시작한다.

## 5. W9와의 충돌과 순서

| 파일 | W9 | F1 | 처리 |
|---|---|---|---|
| `src/models/qwen38_flash_next/engine.cpp`, `engine.hpp` | I1(`PrepareDecode`), M1 probe(`ProbeDraftChain`) | P2만(`Model::SessionBytes` `engine.cpp:274-286`, `Load` 옵션 `165-178`) | **P2는 W9 I1 ABBA 판정이 커밋된 뒤에 시작한다.** 시작 전에 W9가 `engine.*`에 남은 작업이 없는지 감독에게 확인한다 |
| `src/models/qwen38_flash_next/mtp_policy.hpp` | I1 | 건드리지 않는다 | 충돌 없음 |
| `tests/models/qwen38_flash_next/{gpu_probe,mtp_trace,mtp_sampling_test}.cpp` | M1/I1 | 건드리지 않는다 | 충돌 없음 |
| `tests/models/qwen38_flash_next/CMakeLists.txt` | M1에서 수정했다 | K0-FN과 P1 테스트 타깃 추가 | 추가만 한다. 커밋 직전에 rebase한다 |
| `kernels/rocm/executor.{hpp,cpp}`, `batch.cpp` | I1은 건드리지 않았다(`f1cf33ac` 파일 목록). M1 probe는 `Session::ProbeDraftChain`(engine 쪽) | P1의 주 대상 | W9 다음 단계(sampled, C > 1 확장)가 `batch.cpp`/`executor.cpp`를 건드리기 시작하면 감독이 순서를 정한다 |
| `tools/qwen-flash/spec/*`, W9 이미지와 창 | W9 | 읽기만 한다 | — |

- **장비:** F1의 빌드와 창은 W9 ABBA 창이 도는 동안 시작하지 않는다(GPU timing 중 CPU 빌드 금지 규칙). F0 창은 W9 ABBA 뒤 감독이 정한 자리에 둔다.
- **기준선:** P1/P2의 main 짝 비교는 W9 I1이 들어간 HEAD를 main 대조로 쓴다. 그래야 I1 효과가 F1 회귀처럼 보이지 않는다.

## 6. 열린 질문 (감독 결정)

1. F0을 "측정만"으로 정의하는 데 동의하는가(n-gram 표는 이미 디스크에 있다)? PLAN F0 문구를 고칠지는 감독이 정한다.
2. F2 범위의 "n-gram 표 EXL3화"는 상주 메모리 이득이 없다(디스크 크기만 바뀐다). 범위에서 뺄지 정해야 한다.
3. P2에서 이미지 선점을 실사용 기준으로 바꾸는 것을 F1에 포함할지 정해야 한다.
4. MTP 층을 같은 plane에 둘지(AR 세션은 1/13 낭비), plane 세트를 둘로 나눌지 정해야 한다. 지금 제안은 하나이고, P2 측정 뒤 다시 본다.
5. KVarN V 배치: 27B와 같은 channel-major를 쓸지, Flash-Next 전용 token-major를 쓸지. K0-FN 뒤 커널 단계에서 정한다.
