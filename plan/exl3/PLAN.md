# EXL3(QTIP) 형식 도입 계획 초안 (2026-10-30, 감독 gufo-06)

상태: **초안**이며 사용자 검토를 기다린다. 근거는 [RQ PLAN "방향 전환"](../rotated-quant/PLAN.md), [EXL3-RESULTS.md](../rotated-quant/EXL3-RESULTS.md)(W8), [FLASH-NEXT.md 13절](../rotated-quant/FLASH-NEXT.md)(W7), [W9-RESULTS.md](../rotated-quant/W9-RESULTS.md)다.

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
- **통과 기준:**
  - pp2048은 지금 gufo UD-Q4_K_M(약 650 tok/s)과 비교해 5 %·10 ms 안이거나 더 빠르다.
  - d32K tg가 gufo보다 떨어지지 않는다.

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

### F2 Flash-Next EXL3

- **대상:** 전문가(MoE) trellis GEMV와 그룹 GEMM, dense 트렁크, MTP, n-gram 표다.
- **팩:** turboderp `Qwen3.8-Flash-Next-exl3`(3.05 bpw 등, Qwen Community License)를 먼저 쓴다. 직접 양자화하려면 HF safetensors 원본이 필요한데, 지금 받은 것은 BF16 GGUF다.
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

GPU 창, 빌드, 디스크 규칙은 [W-COMMON.md](../rotated-quant/W-COMMON.md)를 그대로 따른다.

## 사용자 결정이 필요한 것

1. **27B 목표:** "같은 품질, 더 적은 바이트(EXL3 4.0 수준)"와 "같은 바이트, 더 좋은 품질(EXL3 5.0 수준)" 중 어느 쪽인가?
2. **팩 출처:** 공개 팩(turboderp)을 쓸지, 직접 양자화할지
3. **R3/E 실험:** 멈추는 데 동의하는지
4. **시작 순서:** X0·X1과 F1을 동시에 시작할지
