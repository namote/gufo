# XQ 브리프: EXL3 직접 양자화 (27B 5.0 bpw, 2026-10-30)

공통 규칙은 [W-COMMON.md](../../archive/rotated-quant/W-COMMON.md)를 따른다. 계획은 [PLAN.md](../PLAN.md)다. 사용자 결정: 27B 목표는 EXL3 5.0급이고, 직접 양자화한다.

## 목표

upstream `exllamav3` v1.6.5 `convert.py`로 27B를 gfx1151에서 직접 양자화한다. 보정 코퍼스는 c02다. 결과 팩은 turboderp 5.00bpw 팩과 같거나 더 나은 품질이어야 한다.

## 입력과 도구

- **원본:** 외장의 HF BF16 safetensors `/run/media/m2on/T5 EVO/AI/model/originals-20261007T122040Z-101bd114/llm/qwen-3.8-27b-bf16/`(rev `1d4bf0f2`, 52 GB). W8이 ExLlamaV3 로드를 확인했다. 읽기 전용으로 마운트한다.
- **보정:** c02 `~/projects/ai.tools/storage/evaluations/rq-r3-calib-20261029/c02/`(`chunks.bin` `40d5d653…`, 512토큰 청크 3,906개)
  - `convert.py`가 외부 보정 데이터를 받는 방식을 확인한다(인자, 형식, 행 수와 길이). 기본값은 250행 × 2048토큰이다.
  - c02를 그 형식으로 바꾸는 작은 도구를 둔다. 청크를 이어 붙이는 방식과 선택 규칙을 기록한다.
  - 평가 코퍼스 a01과 겹치지 않는다는 기록은 이미 manifest에 있다.
- **이미지:** W8 `ai-tools/exl3-eval:1.6.5-rocm100`이다. `convert.py`가 의존하는 것이 빠졌으면 추가 빌드를 제안한다.
- **출력:** 외장 `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/qwen-3.8-27b-exl3-5.00-c02/`
  - `PROVENANCE`에 원본 sha, 도구 리비전, 명령줄, 시드, 시간, 최대 메모리를 적는다.
  - 작업 디렉터리(체크포인트)도 외장에 둔다. 루트 디스크는 쓰지 않는다(지금 약 157 GiB로 하한에 가깝다).

## 할 일

1. **조사 (가벼움):**
   - `convert.py`의 인자: `--bits 5.0`, `--head_bits`, `--hq`, `--out_scales`, 코드북, 보정 옵션
   - 예상 시간과 GPU 메모리(27B, gfx1151 128 GB)
   - turboderp 5.00 팩의 `quantization_config`(같은 설정으로 맞추는 기준)
   - ROCm 경로가 검증됐는지(upstream 문서와 이슈)
   - 계획을 `WAITING:`으로 보고한다.
2. **시험 양자화:** 작은 범위(층 몇 개, 또는 `--resume` 단계 확인)로 GPU 창 하나에서 동작을 확인한다. 감독 허락 뒤에 한다.
3. **본 양자화:** 5.00 bpw, c02, turboderp와 같은 head/hq 설정. 긴 GPU 작업이라 창을 나눌 수 있으면 나누고, 체크포인트 재개를 쓴다.
4. **품질 대조:** W8 수집기(`tools/qwen27b/rq_quality/exl3/collect.py`)로 R0 하네스 전 지표를 잰다.
   - 대상: 우리 팩과 turboderp 5.00 팩
   - 지표: KL, ΔNLL, top-1, 꼬리, 뒤집힘, 카테고리
   - paired 비교를 한다. 우리 팩이 같거나 낫다는 것을 신뢰구간으로 보인다. 나쁘면 원인 후보(보정 데이터, 설정)를 보고한다.

## 파일 경계

- 쓸 수 있는 파일:
  - `tools/exl3_quant/**`(보정 변환 도구, 실행 스크립트)
  - `plan/exl3/archive/XQ-RESULTS.md`
  - ai.tools 쪽 창 계획 `PLAN-XQ*.json`(LOCKS.py를 고친 직후 다시 돌린다)
- `src/`는 고치지 않는다.

## 장비

- 양자화는 GPU와 CPU를 오래 쓴다. timing 창(속도 측정)과 겹치면 안 되고, 감독이 순서를 정한다.
- 시작 전에 루트와 외장 디스크 여유를 확인한다.
