# EXL3(QTIP) 형식 도입 계획 초안 (2026-10-30, 감독 gufo-06)

상태: **확정 (사용자 결정 2026-10-30)**. 결정 내용은 아래 "결정" 절에 있다. 근거는 [RQ PLAN "방향 전환"](../archive/rotated-quant/PLAN.md), [EXL3-RESULTS.md](../archive/rotated-quant/EXL3-RESULTS.md)(W8), [FLASH-NEXT.md 13절](../archive/rotated-quant/FLASH-NEXT.md)(W7), [W9-RESULTS.md](../archive/rotated-quant/W9-RESULTS.md)다.

## 목표

27B와 Flash-Next에서 **gufo 품질을 유지하면서 Kyojin 속도 이상**을 낸다. llama.cpp 호환은 포기한다. 형식은 EXL3(QTIP trellis + 입출력 Hadamard)를 쓰되, **런타임과 커널은 gufo 안에 새로 짠다**. EXL3의 torch/Triton 런타임은 가져오지 않는다.

## 근거 요약 (27B, 측정)

| 항목 | 결과 |
|---|---|
| 같은 bpw의 품질 | EXL3가 4쌍 모두 KL·top-1에서 앞선다. 대략 "EXL3 b bpw ≈ UD k-quant b + 0.7 bpw"다. EXL3 4.00(4.11 bpw)의 KL이 UD-Q4_K_M(4.80 bpw)과 같다 |
| 엔진 차이 | ExLlamaV3 BF16과 gufo BF16의 KL 중앙값은 2e-5다. EXL3 수치는 사실상 양자화 오차다 |
| decode | 두 엔진 모두 대역폭 한계다(약 190–205 GB/s). 같은 KL에서 EXL3 4.00이 tg d0 +20 %, d32K +8 %다 |
| prefill | EXL3가 33 % 느리다(약 430 대 650 tok/s). 32K 깊이 prefill은 upstream Triton 커널이 LDS 한계를 넘어 실행되지 않는다 |
| 투기 | EXL3 4.00 + MTP 31.7 tok/s(2.05×), gufo UD-Q4_K_M + DFlash2 29.5(2.28×)다. gufo 투기 배율이 더 높다 |
| 주의 | 상위 256개만 비교하는 KL(KL_256)은 복사 실패를 숨길 수 있다(EXL3 3.50 chat). 그래서 ΔNLL과 위치별 꼬리도 함께 본다 |

**결론:** EXL3 바이트(품질)와 gufo 런타임(prefill, 긴 문맥, kvarn KV, DFlash2)을 합치면 두 엔진 각각보다 낫다. 짐작으로는 EXL3 4.00 크기에서 decode +20 %, 투기 포함 +20 % 이상이고, prefill과 긴 문맥은 지금 gufo 수준을 유지한다.

## 단계

### X0 형식 명세와 참조 경로 (27B)

- **형식 명세:** EXL3 형식을 정리해 `docs/`에 둔다.
  - 담을 내용: `quantization_config`, 텐서별 `trellis`/`suh`/`svh`/`mul1`(또는 `mcg`), 16×16 타일, K 1–8, 헤드 6비트, MTP 4비트, 임베딩
  - 기준은 upstream `exllamav3` v1.6.5 `cdc785db`다.
- **참조 도구:** upstream 런타임으로 텐서별 dequant 결과와 GEMV 결과를 덤프하는 도구를 만든다. W8 이미지를 그대로 쓴다.
- **공통 위치:** 모델에 의존하지 않는 부분을 공통 위치에 둔다. 경계는 RQ의 "공통과 모델별 경계" 원칙을 그대로 따른다.
  - `src/core/quant/exl3/`: 리더, 코드북, 타일 레이아웃
  - `src/core/hip/detail/`: trellis 디코드, Hadamard
- **통과 기준:** CPU 참조 dequant가 upstream과 비트 단위로 같다. 모든 K, 코드북, 반정수 K를 대상으로 한다.

### X1 27B 리더와 정확성 경로

- **로드:** gufo가 EXL3 safetensors 팩을 읽는다. 처음에는 정확성만 본다. 텐서를 FP16으로 풀어 기존 GEMM에 넘기거나, 단순 커널을 쓴다.
- **통과 기준:** R0 하네스에서 KL(gufo-EXL3 ‖ ExLlamaV3 같은 팩)이 엔진 차이 수준이다. 예를 들어 RQ2I와 같은 크기로 1e-3 아래다. BF16 기준 KL은 W8 값과 신뢰구간 안에서 같아야 한다.
- **재사용:** 회전 로더 작업(R2)의 prism 바인딩, FWHT 헤더, 활성화 융합 지점 표를 Hadamard 입력 경로에 다시 쓸 수 있는지 본다.
  - EXL3는 출력 쪽 Hadamard(`svh`)도 있다.
  - 텐서마다 부호가 달라, 그룹이 부호를 공유하는 지금 융합 방식은 그대로 쓸 수 없을 수 있다(짐작).

### X2 decode GEMV와 verify 커널

- **커널:** trellis 디코드를 융합한 GEMV(m = 1)와 verify(m = 2–8)를 만든다.
  - 입출력 Hadamard를 활성화 생성 커널과 출력 epilogue에 융합한다.
  - DFlash2 verify 경로도 포함한다.
- **참고 코드:** upstream `exllamav3_ext/rocm/*`(RDNA WMMA, `exl3_gemv_tiles_rdna`)를 기준으로 한다. vcruz305(`v_sad_u8` 디코드, LDS pad)와 Kyojin은 비교 자료로 쓴다. 규칙은 `.agents/skills/optimize-kernel/SKILL.md`를 따른다.
- **통과 기준:**
  - 실효 대역폭이 피크의 80 % 이상이다(R0b 기준 약 190 GB/s 이상).
  - tg128 d0이 ExLlamaV3 같은 팩 이상이다(4.00: 15.6 tok/s).
  - DFlash2 텍스트 수락률이 같은 출력에서 유지되고, 실효 tok/s가 31.7을 넘는다.

### X3 prefill GEMM과 긴 문맥

- **커널:** trellis 디코드 + WMMA GEMM이다. 디코드를 다시 하지 않도록 타일을 재사용한다.
- **긴 문맥:** kvarn-k4v2 KV와 32K 이상 경로가 지금처럼 동작해야 한다.
- **통과 기준 (prefill도 속도 목표에 넣는다, 사용자 2026-10-31):**
  - 같은 운영 바이너리의 UD-Q4_K_XL과 ABBA로 짝지어, pp512, pp2048, pp8192와 깊이 32K prefill이 모두 요청·단계별 5 %·10 ms 안이거나 더 빠르다(참고: 지금 gufo 약 650 tok/s, ExLlamaV3 약 430 tok/s).
  - d32K tg가 gufo보다 떨어지지 않는다.
- **순서:** X2b와 겹쳐 지금 설계를 시작한다. XB6는 prefill을 기준값으로 기록만 하고, 판정은 X3 커널이 들어간 뒤 같은 방식으로 다시 잰다.

### X4 27B 채택 판정

- **품질:** R0 하네스 전 지표(KL, ΔNLL, top-1, 꼬리, 뒤집힘)로 본다. 기준은 "UD-Q4_K_M 이상의 품질을 더 적은 바이트로" 또는 "UD-Q4_K_XL 이상의 품질을 같은 바이트로"다. 어느 쪽인지는 사용자가 고른다.
- **D4 실제 작업 게이트:** tool-call, 한국어, 긴 문맥을 확인한다.
- **기능과 속도:** `tests/functional/` 해당 suite를 돌리고, main과 짝지은 요청·단계별 5 %·10 ms 타이밍을 본다.
- **팩 출처:**
  - turboderp 27B 팩(Apache-2.0)을 그대로 쓸지, upstream `convert.py`로 직접 만들지 정한다.
  - 직접 만들려면 HF safetensors 원본(외장에 있음)과 GPU 양자화 시간이 필요하다.

### F1 Flash-Next 페이지 KV + KVarN (형식과 무관, X와 병행 가능)

- **KV 할당:** Flash-Next의 세션별 연속 `ContinuationLayout`(FP16, 용량 선점)을 공통 `src/core/hip/kv_block_pool`/`kv_block_table`로 옮긴다.
  - attention 커널이 `KvBlockTable`로 행을 간접 참조하게 한다.
  - MTP 되감기, 스냅샷, 디스크 캐시를 블록 단위로 맞춘다.
- **KV 양자화:** KVarN을 Flash-Next attention에 도입한다. 27B의 `attention_kvarn` 경로를 공통으로 옮길 수 있는지 먼저 본다.
- **통과 기준:**
  - 출력 동일성: FP16 KV 대비 KL과 needle을 본다. 기준은 27B KVarN K0 방식을 따른다.
  - 다중 세션 메모리 이득을 측정한다.
  - 기능 테스트(`long-context`, `cache`)를 통과한다.
  - main과 짝지은 타이밍을 본다.

### F0 Flash-Next n-gram 표 디스크 배치와 메모리 점유 측정 (사용자 지시 2026-10-30, F1과 함께)

- **(F1 설계 결과 2026-10-30: 이미 운영에 들어가 있다.** n-gram 표 `per_layer_token_embd`(IQ4_NL, 26.82 GiB)는 GPU에 올리지 않고 O_DIRECT `pread` + 128 MiB 행 캐시로 필요한 행만 읽는다(`device_model.cpp:302-321`, `ngram.cpp:79-209`). 그래서 F0은 **측정만** 한다.) n-gram 표를 GPU 상주 메모리에서 빼서 디스크(mmap, 필요한 행만 읽기)로 둔다. Kyojin과 upstream EXL3는 `ngram_embedding.safetensors`(32.6 GB)를 따로 두고 파일에서 읽는다.
- 그다음 전체 메모리 점유(가중치, KV, MTP, 작업 버퍼, 세션 수별)를 측정한다.
- **이 점유 결과를 보고 Flash-Next EXL3의 bpw를 정한다**(F2의 선행 조건).
- 통과 기준: greedy 출력이 바이트 단위로 같다. decode와 prefill이 main 대비 5 %·10 ms 안이다(n-gram 조회가 디스크로 가도 속도를 해치지 않아야 한다). 페이지 캐시 상태(cold/warm)를 기록한다.

**F1 KVarN 결정 (사용자 2026-10-30, K0-FN 결과 뒤):** Flash-Next KV는 **k4v4**. K0-FN에서 모든 설정이 top-1 1.0000, greedy 동일, needle 통과였지만 k4v2는 KL 꼬리가 27B k4v2의 약 10배(32K 최대 1.0e-2, 200K needle 3.2e-2)였고, k4v4는 27B k4v2 수준(1.7e-3 / 1.8e-3)이다. 근거: [F1-RESULTS.md](archive/F1-RESULTS.md)(`97315880`).

### F2 Flash-Next EXL3

- **대상:** 전문가(MoE) trellis GEMV와 그룹 GEMM, dense 트렁크, MTP, n-gram 표다.
- **bpw: 전문가 EXL3 5.0 (사용자 결정 2026-10-30, F0 측정 뒤).** 상주 약 77.9 GiB로 지금(UD-Q4_K_XL 79.3 GiB)과 비슷하다. 품질은 지금보다 높을 것으로 보고(27B 환산), decode 속도는 지금과 비슷하다. KVarN이 있으면 262K MTP 세션 8개가 가능하다. F0 결과: [F1-RESULTS.md](archive/F1-RESULTS.md)(`01cf33e0`), HIP 전체 112 GiB. 직접 양자화하려면 Flash-Next HF safetensors 원본이 필요하다(지금은 BF16 GGUF만 있음). 받을지는 F2 시작 전에 정한다. 전문가 bpw b에 대해 상주 약 14.06·b + 7.6 GiB(F1 계산, F0-M으로 확인).
- **n-gram 표는 F2 범위에서 뺀다.** 디스크에 있으므로 EXL3로 바꿔도 메모리 이득이 없다.
- **팩: 직접 양자화하지 않고 turboderp `5.05bpw_h6_ng6` 팩을 받아 쓴다 (사용자 결정 2026-10-31).** 직접 양자화는 GPU 26–33시간이 들고, 창마다 작업 디렉터리를 복사하면 외장 디스크가 모자란다. 이 팩의 비트 배분은 전문가 5, 트렁크·attention·공유 전문가 7, 헤드 6, MTP 5, n-gram K6이다. 조사 근거: [F2-PLAN.md](archive/F2-PLAN.md)(`511d7bbc`). HF safetensors 원본(외장, sha256 확인)은 필요하면 직접 양자화할 때 쓴다.
- **통과 기준:** 품질은 gufo 지금 UD-Q4_K_XL 수준 이상이고, 속도는 Kyojin 카드(52.8/47.5/73.2, 8K/32K 52.2/51.1) 이상이다.

### 병행: 투기 디코딩 (W9)

- I1(확신도 기반 드래프트 정지)은 ABBA 판정 뒤 채택한다.
- 그다음 sampled와 C > 1 확장, AR 백오프 제거 비교, 27B DFlash2에 같은 아이디어 적용이다.

## 정리하거나 멈출 것

- **멈춤:** R3 imatrix와 E1–E10 k-quant 실험을 멈춘다. E10의 부호 변이 측정 결과는 기록으로 남긴다.
- **유지:** R0 하네스(+ `kl_tail`, `flips`, `accept.py`, kld_base 도구), R2 prism 로더(이미 커밋됨), W8 EXL3 도구와 이미지를 계속 쓴다.
- **디스크:** EXL3 팩과 fake-quant 파일은 이식 중 수치 기준으로 외장 디스크에 그대로 둔다.

## 담당 제안

| 담당 | 범위 |
|---|---|
| X-A | X0·X1: 형식, 리더, 참조, 정확성. W8의 경험을 이어받는다 |
| X-B | X2·X3: 커널. W3의 융합 경험을 이어받는다 |
| F-A | F1: 페이지 KV + KVarN |
| W9 | 투기 디코딩(계속) |

GPU 창, 빌드, 디스크 규칙은 [W-COMMON.md](../archive/rotated-quant/W-COMMON.md)를 그대로 따른다.

## 결정 (사용자 2026-10-30)

1. **27B 목표: EXL3 5.0급.** 같은 바이트로 더 좋은 품질을 낸다. 비교 기준은 UD-Q4_K_XL(5.15 bpw)이고, KL 절반 수준(EXL3 5.00: 7.6×10⁻³)이 목표다. 속도는 X2·X3에서 gufo 런타임으로 UD-Q4_K_XL 이상을 낸다.
2. **직접 양자화한다.** upstream `convert.py`를 오프라인으로 쓴다(W8 이미지). 보정 코퍼스는 사용자 지시(2026-10-29)대로 c02를 쓴다. turboderp 5.00 팩은 품질 대조군으로만 쓴다.
3. **R3 imatrix와 E 실험을 멈춘다.**
4. **X0·X1(형식과 리더), XQ(양자화), F1(Flash-Next 페이지 KV + KVarN)을 동시에 시작한다.** X2·X3은 X1 뒤에 시작한다. 담당은 X-A = W8(이어서), XQ = 새 에이전트, F1 = 새 에이전트다.
5. **오래 걸리는 양자화는 하지 않고, 이미 만들어진 공개 팩을 받는다 (사용자 2026-10-31, 27B와 Flash-Next 모두).** 27B c02 5.00 팩은 이미 끝났으므로 목표 팩으로 쓴다(품질 판정: [XQ-RESULTS.md](archive/XQ-RESULTS.md) `eeaad672`). 다른 bpw나 변형이 필요하면 turboderp 등 공개 팩을 먼저 찾아 받는다. 직접 양자화는 공개 팩이 없을 때만 감독이 사용자에게 묻는다.
6. **빠르게 구축하고, 구축 중에는 단위 테스트만 돌린다. 다 만든 뒤 통합 테스트를 하고 문제를 고친다 (사용자 2026-10-31).** 단위 테스트(CPU, 짧은 GPU 커널 테스트)는 구축 중에 바로 돌린다. 모델 수준 품질·속도 창은 마지막에 한 번 돌린다. X3(27B prefill)과 F2(Flash-Next EXL3: Codex 작업자 `fn-exl3-impl`, 작업 트리 `../gufo-fn-exl3`, 브랜치 `ai-tools/fn-exl3`), F1 K2를 모두 구현한 뒤 마지막에 묶음 창(정확성·품질 창, 그다음 ABBA timing 창)으로 한 번에 비교한다. Flash-Next 팩은 5.05를 그대로 쓴다. 토큰당 가중치는 6.03 GB로 UD-Q4_K_XL의 6.19 GB와 비슷하므로(Codex 헤더 계산), 속도 이득은 커널과 MTP에서 내야 한다.
7. **목표는 Kyojin을 따라가는 것이 아니라 압도하는 것이다. 그중 prefill이 가장 중요하다 (사용자 2026-11-01).** Flash-Next EXL3 prefill은 UD-Q4_K_XL(pp8192 약 1800 tok/s)과 Kyojin cold 8K(1621 tok/s)를 모두 크게 넘어야 하고, 하드웨어 한계(roofline)까지 올린다. 수치 목표는 pp2048·pp8192 모두 2,000 tok/s 이상이고, 현실적 상한(약 2,300–2,500 tok/s, WMMA 피크 55.5 TFLOPS의 55–65 % 효율 기준)에 가깝게 간다. Kyojin을 프로파일하거나 통째로 옮기지 않고 gufo 자체 설계로 내되, Kyojin 소스는 필요한 부분만 참고할 수 있다(MIT 고지 유지). 품질은 지금처럼 UD-Q4보다 좋아야 한다(5.05: KL 0.0269 대 0.0392).

## (처음) 사용자 결정이 필요했던 것

1. **27B 목표:** "같은 품질, 더 적은 바이트(EXL3 4.0 수준)"와 "같은 바이트, 더 좋은 품질(EXL3 5.0 수준)" 중 어느 쪽인가?
2. **팩 출처:** 공개 팩(turboderp)을 쓸지, 직접 양자화할지
3. **R3/E 실험:** 멈추는 데 동의하는지
4. **시작 순서:** X0·X1과 F1을 동시에 시작할지
