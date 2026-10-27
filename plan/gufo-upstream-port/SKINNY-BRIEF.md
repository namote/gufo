# S: 작은 M weight GEMM — K-quant 9–64행 한 번 읽기 (에이전트 안내)

2026-10-27 사용자: vLLM 때 쓴 스키니 커널(M ≤ 8) 같은 최적화가 gufo에 있는지 점검 → "응~" 진행. 감독이 단계별로 검증한다.

## 배경 (읽기 전용 점검 결과, 감독 확인)

- gufo는 이미 스키니 방식이다: 한 wave가 양자화 가중치 행을 한 번 읽어 모든 토큰 행에 쓰고, K는 wave 레인끼리 나눈다. exact FP32 활성 커널이라 verify = AR 비트 일치(`kernels/small_batch_gemm.hpp`, `kernels/small_batch_wave64.hip`, 진입 `LaunchBatchedQuantGEMMFp32` `kernels/prefill_quant_gemm.hip:1706`).
- M = 1은 대역폭 약 92 %, 2–8행은 한 번 읽기. Q8은 17–36행 ragged 한 번 읽기(이전 실험: C8 TG +21–29 %).
- **빈틈**: K-quant(Q4_K 등) 9–64행. 9–16은 정해 둔 shape 목록(`small_batch_wave64.hip:491-525`)에서 12·16일 때만 한 번에, 나머지는 `8 + 나머지`(`prefill_quant_gemm.hip:1774-1830`)로 가중치를 2–4번 읽는다. 측정된 절벽: `src/models/qwen/dflash_policy.hpp:279-299`(2명 8행 넘을 때 1.10 → 1.48, 4명 16행 넘을 때 1.33 → 1.97, 8명 3.08).
- 예전 vLLM 작업: `~/projects/ai.tools/test/experiments/skinny_group_rows_20261019_r2/`(wvSplitK_int4_g, M ≤ 5 → 6–8 확장).

## 할 일

1. **시제품·측정 먼저**: `tools/qwen27b/dflash_gemm_bench.hip`(운영 템플릿, 회전하는 cold 가중치, 시간 전 비트 확인)로 Q4_K_M 모델의 실제 shape 전부(q/k/v·SSM·gate/up/down·head)에서 행 9–16·17–64의 지금 시간과 GB/s(`tools/bench/gfx1151_peak.hip` 대비)를 표로. 그다음 후보: 9–16 어떤 폭이든 한 launch(나머지를 같은 launch의 끝 그룹으로), 17–64는 Q8 ragged처럼 한 번 읽기. shape 목록 대신 기하 규칙(VGPR·LDS 예산)으로 고르기.
2. 이득 있는 것만 운영에. **수치 계약**: 결과가 지금 경로와 비트 일치(행마다 AR 단일 decode와 같은 reduction 순서), verify = AR 비트 일치. 바뀐 폭·shape마다 비트 테스트를 더한다.
3. `dflash_policy.hpp`의 비용표는 새 경로로 다시 잰 값으로만 바꾼다(정책이 바뀌면 그 영향도 같이 보고).
4. 모델 수준 확인(Q4 DFlash2): C1 그대로(바이트·시간), C2·C4·C8 처리량. AGENTS.md 규칙대로 기준(이 브랜치 HEAD)과 번갈아, 요청별 5 %·3 ms, AR 일치·private sampled replay. 관련 functional suite만, full sweep 금지.

## 규칙

- gufo 브랜치 `ai-tools/port-u4f`. push·PR·이슈 없음. 한 줄 Conventional Commits + 감독이 준 trailer. C++ 커밋 전 format check. 자기 파일만 `git add`, 관련 없는 작업 보존.
- **K5 에이전트가 동시에 작업 중**(`attention_kvarn.hip`, `prefill_chunk.cpp`, `kvarn_layout.hpp`, KVarN quantizer·테스트). 그 파일을 고치지 않는다. 겹쳐야 하면 감독에게 먼저.
- GPU: ai.tools 창 실행기로만, 한 번에 하나, 창 전 `.gufo-revision`을 HEAD로, 창 끝 `sleep 20`, 자동 재시도 없음, 실패 포함 모든 결과 보존. **감독이 GPU 시작을 알리기 전에는 GPU 창을 돌리지 않는다**(K5 창이 쓰는 중). 그 전에는 코드 읽기·설계·CPU 빌드·format까지.
- 마진은 데이터를 본 뒤 넓히지 않는다. 이득이 없으면 운영 코드는 바꾸지 않고 측정만 기록한다.
- 보고: ① 시제품 측정표 뒤(운영 변경 전 감독 확인) ② 끝에(커밋, 창 ID·경로, 전후 표: shape × 행 수 µs·GB/s, C1/C2/C4/C8 TG). 결과는 `plan/gufo-upstream-port/SKINNY.md`(새 파일)에 기록.
