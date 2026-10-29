# W8 브리프: EXL3 27B 측정으로 길 고르기 (2026-10-29)

이 브리프는 [W-COMMON.md](W-COMMON.md)를 따른다. 근거는 PLAN "방향 전환" 절 1번이다(사용자 지시 2026-10-29). 목표는 27B와 Flash-Next에서 gufo 품질을 유지하면서 Kyojin 속도 이상을 내는 것이고, llama.cpp 호환은 포기해도 된다.

## 질문

같은 bpw대에서 EXL3(QTIP trellis + Hadamard)가 k-quant(UD, 회전 k-quant)보다 품질이 얼마나 좋은가? 이 답으로 3번 갈래를 정한다. 하나는 "k-quant + 회전 계열 계속(GPTQ식 양자화기, 부호 최적화)"이고, 다른 하나는 "QTIP 형식 도입(gfx1151 리더·커널)"이다.

## 측정 대상

- **EXL3:** `turboderp/Qwen3.8-27B-exl3`(Apache-2.0), 브랜치 `3.00bpw`, `3.50bpw`, `4.00bpw`. W7 조사 기준으로 `4.00bpw`는 `113cf7ab`, 16.86 GB다. 다른 브랜치의 리비전과 크기는 직접 확인한다.
  - 필요하면 `5.00bpw`(`a35e75a7`, 19.9 GB)도 넣는다. UD-Q4_K_XL(5.15)과 bpw가 맞기 때문이다. 넣을지는 감독에게 제안한다.
- **k-quant 비교군:**
  - 이미 있는 것: R0 결과(`metrics-a01/`) Q8_0, Q6_K, UD-Q4_K_XL, Q4_K_M, UD-Q4_K_M. 그리고 rot-prism Q4(seed 1, imatrix 없음)의 R0 측정이 있으면 함께 쓴다.
  - 3–3.5 bpw대 k-quant가 없다. 다음을 확인하고 제안한다.
    - `unsloth/Qwen3.8-27B-GGUF`의 UD-Q3_K_XL, UD-IQ3_XXS 같은 3비트대 파일과 그 크기, 리비전
    - gufo 27B 로더가 그 타입(Q3_K, IQ3_S/XXS 등)을 받는지
- **bpw 계산:** W2 정의를 따른다(2차원 가중치, output 포함, token_embd와 nextn 제외). EXL3의 표시 bpw는 headers(`h5`/`h6`)와 무엇을 포함하는지 다를 수 있다. 그래서 실제 텐서 바이트로 다시 계산한다.

## 실행 환경

- **런타임 순서:**
  - 1순위: upstream `turboderp-org/exllamav3` v1.6.5의 `rocm10.0` wheel(torch 2.13)이다. gfx1151 커널 경로가 실제로 쓰이는지 확인한다(fallback 경로가 아닌지 로그나 설정으로).
  - 안 되면 Kyojin(`Yamz-Labs/kyojin`, MIT)이나 `vcruz305/exllamav3-amd`를 쓴다. 고른 이유를 적는다.
- **컨테이너:** ai.tools 실험 규칙을 따른다. 새 이미지를 만드는 것(torch ROCm 내려받기와 빌드)은 무거운 CPU/디스크 작업이므로 감독 허락이 필요하다. 이미지 크기와 출처를 기록한다.
- 호스트에 torch를 설치하지 않는다. 내려받은 파일은 W-COMMON과 시스템 규칙대로 별도 디렉터리에 둔다. Python은 `-I`로 돌린다.

## 품질 측정 (R0 하네스와 같은 기준)

- **코퍼스와 위치:** 코퍼스 a01의 같은 토큰(`windows.bin`, 8 창 × 4096)을 teacher forcing으로 넣는다. EXL3 모델 forward로 위치마다 logits를 낸다.
  - 토큰은 텍스트를 다시 토큰화하지 말고 a01 토큰 id를 그대로 넣는다.
  - 토크나이저 vocab 크기와 특수 토큰 id가 gufo와 같은지 먼저 확인한다.
- **저장:** W2의 `tools/qwen27b/rq_quality/logits_io.py` 형식(top-k 256 id/logit, 전체 logsumexp)으로 쓴다. 그러면 `metrics.py score`(`kl_tail` 포함)로 BF16 기준(`rq0-a01/rq0-bf16`, prefill 경로)과 바로 비교할 수 있다.
  - EXL3에는 gufo의 prefill/verify/decode 경로 구분이 없다. 그래서 "전체 시퀀스 forward" 하나를 prefill 대응으로 적는다.
  - 가능하면 decode 경로 대응(캐시 1토큰씩)으로 일부 위치를 따로 낸다. 경로에 따른 차이가 있는지 보기 위해서다.
- **지표:** KL_256, ΔNLL, top-1, 토큰·블록·창 bootstrap 95 % 구간, 카테고리별 값, 꼬리 지표(최대 1개 제외 평균, 중앙값, 뒤집힘 비율)
- **기준 확인:** BF16 기준이 gufo 실행이므로, EXL3 쪽 수치에는 "엔진 차이"도 섞인다. 이 영향을 가늠할 방법을 제안한다. 예를 들어 EXL3 런타임으로 BF16(HF safetensors 원본, 외장 디스크 `originals-…/qwen-3.8-27b-bf16`)을 돌려 gufo BF16과 비교할 수 있는지 본다. 비용이 크면 그 점을 적는다.

## 속도 측정 (참고, 같은 장비)

- EXL3 plain decode tok/s와 prefill(pp2048 정도)을 잰다. 가능하면 같은 런타임의 MTP나 투기 디코드도 잰다.
- GPU 창 규칙을 지킨다. timing이므로 CPU가 비어 있어야 하고, 5 %·10 ms 기준에 반복은 ABBA로 한다.
- 비교 대상은 gufo R0 속도 창(AR tg128 UD-Q4_K_M 12.9 tok/s 등)이다.

## 파일 경계

- 쓸 수 있는 파일:
  - `tools/qwen27b/rq_quality/exl3/` 아래(런너 스크립트, 이 디렉터리 하나로 모은다)
  - `plan/rotated-quant/EXL3-RESULTS.md`
  - 창 계획 `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-EXL3*.json`
  - 필요하면 ai.tools 쪽 이미지 정의. 위치는 감독과 정한다.
- `src/`와 다른 W의 파일은 고치지 않는다. W2 하네스는 읽기만 하고, 고칠 것이 있으면 감독에게 보고한다.
- 모델 위치는 `~/projects/ai.tools/storage/models/llm/qwen-3.8-27b-exl3-<bpw>/`로 하고, PROVENANCE(리비전, sha256)를 둔다.

## 순서와 허락

1. 조사하고 계획을 세운다. 이 단계는 허락 없이 해도 된다.
   - 파일 목록과 크기
   - 런타임 선택
   - 이미지 구성
   - 3비트대 k-quant 후보
   - 창 계획
   - 디스크(루트 여유 150 GiB 하한)
   - 계획을 `WAITING:`으로 보고한다.
2. **WAITING:** 내려받기, 이미지 빌드
3. **WAITING:** GPU 품질 창, 그다음 속도 창
4. `EXL3-RESULTS.md`에 표와 판단 근거를 쓰고 커밋한 뒤 보고한다. 결론은 "k-quant 계열 계속 / QTIP 도입 / 결론 없음"으로 내고, 결론 없음이면 무엇을 더 재야 하는지 적는다.

짐작과 확인한 것을 구분한다.
