# X-A 브리프: EXL3 형식 명세, CPU 참조, 27B 리더 (X0·X1, 2026-10-30)

공통 규칙은 [W-COMMON.md](../rotated-quant/W-COMMON.md)를 따른다. 계획은 [PLAN.md](PLAN.md)다. 담당은 W8(이어서)이다.

## X0: 형식 명세와 CPU 참조

1. **명세 문서:** `docs/EXL3-FORMAT.md`
   - 기준은 upstream `turboderp-org/exllamav3` v1.6.5 `cdc785db`다.
   - 담을 내용: `config.json`의 `quantization_config`, 텐서별 키(`trellis`, `suh`, `svh`, `mul1`/`mcg` 등), 16×16 타일과 K 비트 배치, 코드북(mul1, mcg, 3inst)의 디코드 수식, 반정수 K, 헤드(6비트)와 MTP(4비트)와 임베딩 처리, 입출력 Hadamard(128점, 부호와 스케일)의 정확한 적용 순서와 정규화
   - 근거로 upstream 파일:줄을 단다.
2. **CPU 참조:** `src/core/quant/exl3/`에 모델과 무관한 공통 코드를 둔다.
   - 포함할 것: 리더(safetensors 텐서 → 구조체), 코드북 디코드, 타일 → 행렬 dequant, Hadamard 입출력 적용
   - gufo의 기존 safetensors 리더가 있으면 그것을 쓴다(먼저 찾는다).
3. **비트 단위 대조:**
   - upstream 런타임(W8 이미지 `exl3-eval:1.6.5-rocm100`)으로 텐서 표본의 dequant 결과와 `y = W·x` 결과를 덤프한다.
   - gufo CPU 참조와 비교한다. 대상은 모든 K, 모든 코드북, 반정수 K, 헤드/MTP 텐서다. 비교 기준은 dequant FP16 비트 동일, GEMV는 FP32 누적에서 상대 오차 1e-6 수준이다.
   - 팩: 외장의 turboderp 27B 3.00/3.50/4.00/5.00. 다른 코드북은 XQ가 만들면 추가한다.
   - 테스트: `tests/core/quant/exl3_*`. 작은 합성 텐서는 upstream 도구로 생성해 fixture로 고정한다.

## X1: 27B 정확성 경로

- **로드:** gufo 27B가 EXL3 팩을 읽는다. 위치는 `src/models/qwen/` 바인딩이다.
  - 처음 경로는 정확성만 본다. 로드할 때 FP16/BF16으로 풀어 기존 GEMM/GEMV 경로에 넘기거나, 단순 GPU dequant를 쓴다.
  - 속도 커널은 X2(X-B) 범위다.
- **역할 대응:** 텐서 이름(HF 이름)과 27B 역할(GDN `in_proj_*`, attention, ffn, lm_head, MTP)을 연결한다. GGUF 경로와 공존하고, 파일 형식은 로더가 판단한다. CLI 옵션은 추가하지 않는다.
- **통과 기준:**
  - R0 하네스(W2 도구)로 KL(gufo-EXL3 ‖ ExLlamaV3 같은 팩)을 잰다. W8의 `w8q-*` 실행이 기준이다. 결과가 엔진 차이 수준이어야 한다(목표 중앙값 1e-4 이하, 평균 1e-3 이하).
  - BF16 기준 KL이 W8 값과 신뢰구간 안에서 같아야 한다. 대상은 5.00(목표 팩)과 4.00이다.
  - 기존 GGUF 경로의 fingerprint가 바이트 단위로 그대로여야 한다(RQ2H 방식).

## 파일 경계

- 쓸 수 있는 파일:
  - `src/core/quant/exl3/**`
  - `src/models/qwen/`의 로더와 바인딩 부분(커널 파일 제외)
  - `tests/core/quant/exl3*`, `tests/models/qwen27b/exl3*`
  - `docs/EXL3-FORMAT.md`
  - `plan/exl3/XA-RESULTS.md`
  - `tools/qwen27b/rq_quality/exl3/**`
- 고치기 전에 감독에게 알릴 것: `src/models/qwen/`의 다른 파일(W3의 이전 회전 코드 포함), 공유 `CMakeLists.txt`(자기 hunk만 넣는다)
- upstream 코드를 옮길 때는 파일 머리에 출처 URL, 커밋, MIT 고지를 남긴다.

## 순서와 허락

1. 명세와 CPU 코드부터 쓴다(가벼운 작업).
2. 빌드는 무거운 CPU 작업이다. timing 창이 없을 때만, 감독 허락 뒤에 한다.
3. GPU 창은 감독이 정한 순서대로 연다.
4. 단계가 끝날 때마다 `WAITING:`으로 보고한다.
