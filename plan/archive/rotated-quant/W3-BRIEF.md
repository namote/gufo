# W3 브리프: R2 로더와 회전 융합 실행 (2026-10-29)

이 브리프는 [W-COMMON.md](W-COMMON.md)와 [R-BRIEF.md](R-BRIEF.md)를 따른다. 범위는 PLAN의 "R2 로더와 실행" 절 전체, 그리고 "GGUF 표시 규칙 (v1)"과 "공통과 모델별 경계" 절이다. **이 브리프의 W 중에서 운영 코드(`src/`)를 바꾸는 것은 W3뿐이다.**

## 목표

회전 GGUF(R1a/R1b 컨버터 출력)를 27B가 읽고, 모든 실행 경로에서 입력 회전을 적용해 원래 모델과 같은 출력을 내게 한다. 비회전 파일의 동작과 성능은 바이트 단위로 그대로 둔다.

## 구현 범위

1. **공통 검사** `src/core/quant/rotation.{hpp,cpp}`
   - 전역 KV(`gufo.rotation.*`)와 `<name>.rot`(F16 ±1, R1a 결정)를 읽는다.
   - 모델이 넘긴 그룹 표로 PLAN의 거부 조건을 모두 검사한다. 거부 메시지에는 이름을 넣는다.
   - 표시 규칙의 정식 문서를 `docs/`에 둔다(위치는 저장소 관례에 맞춘다).
2. **FWHT device 함수** `src/core/hip/detail/hadamard.hpp`
   - 헤더 전용이다. 128점 FWHT, 부호 곱, 정규화를 한다.
   - Flash-Next 커널 트리에서도 include할 수 있게 의존성을 최소로 둔다.
3. **27B 바인딩** `src/models/qwen/weights.cpp` 등
   - 그룹 표를 공통 검사에 넘기고, 그룹별 부호를 디바이스에 올린다.
   - 역할별 회전 경로가 있는지 검사한다.
4. **활성화 생성 커널에 회전 융합**("회전 후 Q8_1 양자화". Q8_1을 풀어 다시 양자화하지 않는다)
   - 대상: `LaunchBatchedFusedRMSNormQuantizeQ8_1`(attn_in, ffn_in, head_in), `LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1`(ssm_out_in), `LaunchBatchedFusedSwiGLUQuantizeQ8_1`(ffn_down_in), attention 출력과 gate 뒤의 Q8_1(attn_out_in), 그 밖의 `LaunchQuantizeActivationQ8_1*` 호출 지점. 선언은 `src/models/qwen/hip/ops/gemm.hpp`에 있다.
   - **병합 뒤 위치를 다시 찾는다.** decode GEMV, prefill batched, DFlash2 verify, graph capture 경로가 각각 어떤 활성화 생성 커널을 거치는지 표로 만들어 `R2-RESULTS.md`에 남긴다. Q8_1을 거치지 않는 경로(FP32/BF16 matmul, hipBLASLt, 텐서 타입에 따라 갈리는 경로)도 빠뜨리지 않는다.
5. **DFlash2 드래프터 헤드 경로**
   - 드래프터가 타깃 `output`을 쓰는 곳(`src/models/qwen/dflash_weights.cpp` 부근)에도 `head_in` 회전을 적용한다.
6. **BF16 경로 지원**
   - R2 ②(회전 BF16과 원본 BF16 비교)를 하려면 BF16 가중치 경로가 회전을 지원해야 한다.
7. **거부**
   - 회전을 지원하지 않는 실행 경로(`--cpu` 등)는 회전 파일이 들어오면 경로 이름을 들어 거부한다. 자동 fallback은 없다. CLI 옵션도 추가하지 않는다.

## 통과 기준 (PLAN R2)

- ① FWHT 커널 단위 테스트: CPU 참고와 비교한다. 참고 기준은 KVarN Hadamard와 `tests/core/quant/`의 CPU 참고다. 융합 커널도 "비융합 FP32 회전 + 기존 양자화"와 비교한다.
- ② 회전 BF16이 원본 BF16에 충분히 가깝다.
  - 기준: KL(회전 BF16 ‖ BF16) ≤ KL(Q8_0 ‖ BF16)의 1/10, top-1은 Q8_0 이상.
  - decode, prefill, DFlash2 verify에서 각각 확인한다.
  - W1의 회전 BF16과 W2의 하네스가 필요하다.
- ③ 같은 레시피의 회전 Q4와 비회전 Q4에서 DFlash2 수락률이 신뢰구간 안에서 같다.
  - 파일은 W1이 만든다: `qwen-3.8-27b-rot-q4-test-gguf/`
- ④ 비회전 GGUF는 비트 단위로 이전과 같다.
  - 기존 27B 테스트의 logits와 출력이 같아야 한다.
  - 성능 회귀도 없어야 한다: 영향받는 경로에서 main과 짝지은 타이밍을 5 %·10 ms 기준으로 비교한다.
- ⑤ 거부 조건 테스트: CPU 테스트로 만든다.
- 성능 목표: 회전 비용이 decode 토큰당 시간의 1 % 미만이다. 융합 뒤 추가 시간을 잰다.

## 파일 경계

- 쓸 수 있는 파일:
  - `src/core/quant/rotation.*`
  - `src/core/hip/detail/hadamard.hpp`
  - `src/models/qwen/**`
  - `tests/core/quant/**`
  - `tests/models/qwen27b/**`(`rotquant_groups_test.py`와 W2가 만드는 새 파일은 제외)
  - 관련 `CMakeLists.txt`의 등록 부분
  - `docs/`의 표시 규칙 문서와 27B 문서의 해당 절
  - `plan/archive/rotated-quant/R2-RESULTS.md`
- 빌드 디렉터리: `build/rq-cpu-test`, `build/rq-gpu-test`, 그리고 측정용 `build/rq-release`.
- `tools/rotquant/`와 W2 하네스는 읽기만 한다. 고쳐야 할 것이 보이면 감독에게 보고한다.
- 회전 파일을 만들 때 필요한 Python은 W1 컨버터를 그대로 쓴다. 테스트용 작은 합성 회전 GGUF는 테스트 안에서 만든다.
- Flash-Next 코드(`src/models/qwen38_flash_next/`)는 고치지 않는다.

## 순서와 허락

1. 경로 표와 설계를 만든다(읽기만 하므로 허락이 필요 없다).
   - 설계 요약을 감독에게 SendMessage로 보낸다. 특히 다음을 적는다.
     - 텐서 타입에 따라 회전 적용 경로가 갈리는 곳
     - graph capture에 미치는 영향
     - Q8_1 tiled 레이아웃과 회전 블록(128)이 어떻게 맞물리는지
   - 감독의 답을 기다리지 않고 구현을 시작해도 된다.
2. 코드와 CPU 테스트를 쓴다.
3. **WAITING:** 빌드할 때마다(gpu-test, release) 무거운 CPU 작업 허락을 받는다.
4. **WAITING:** GPU 창 허락을 받는다. 순서는 다음과 같다.
   - 단위 테스트(①⑤와 융합 커널 비교)
   - ④ 비회전 회귀와 성능
   - ②③: W1, W2 산출물이 준비된 뒤
5. 커밋 단위를 나눈다: 공통 검사, FWHT 헤더, 27B 바인딩과 융합, 드래프터, 문서. `R2-RESULTS.md`를 쓰고 보고한다.

## 참고

- KVarN 회전: `src/models/qwen/hip/kernels/attention_kvarn.hip`, `tests/models/qwen27b/kvarn_fake_quant.hpp`
- DeepSeek V4 Flash indexer 회전: `src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_indexer.hip.hpp`
- 커널 규칙: `.agents/skills/optimize-kernel/SKILL.md`
- R0b FWHT 벤치: `tools/bench/rq_type_bench.hip`(독립 FWHT는 AR 토큰당 1.26 ms이고, launch에 묶인다)
