# KVarN K1 작업 안내 (에이전트용, 2026-10-26)

상위 문서: [KVarN](KVARN.md)(K0 설계·결과), [이식 계획](PLAN.md) U11. 사용자 결정(2026-10-26): "`--kv-cache-type fp16|kvarn-k4v2|kvarn-k4v4` 옵션을 넣고 처리하게 하자. 기본값은 k4v2", "K1은 우선 32K만 테스트, 256K는 몇 단계 이후", 단계 제안(K1a→K1b→K1c→K2→K3)과 "기본값은 K2 gate 통과 시 k4v2, 지원하지 않는 모델은 fp16" 승인. 감독(메인 세션)이 각 단계를 에이전트에 맡기고 결과를 검증한다.

## 공통 규칙

- 저장소 규칙: `CLAUDE.md`(= `AGENTS.md`). 특히 옵션 이름 하나·별칭 없음, Windows 코드는 `src/core/platform/`, 모델 코드·테스트는 모델 옆, 한 줄 Conventional Commit.
- **빌드**(호스트에 Nix 없음, 결정 D1): eval 이미지 컨테이너에서 소스를 `/src`로 마운트한다. 이미 구성된 `build/cpu-test`, `build/gpu-test`를 쓴다.
  ```sh
  D="docker run --rm -v /home/m2on/projects/gufo:/src -w /src --user $(id -u):$(id -g)"; I=ai-tools/gufo-eval:663d50c9f9d6
  $D --entrypoint cmake $I --build build/cpu-test --target <target>
  $D --entrypoint ctest $I --test-dir build/cpu-test -R '^<test>$' --output-on-failure
  ```
  같은 빌드 디렉터리에 두 빌드를 동시에 돌리지 않는다(ninja 충돌). K1a는 `build/cpu-test`·`build/gpu-test`, K1b는 CMake 빌드를 쓰지 않고 `build/kvarn-bench/`에 hipcc로 직접 빌드한다.
- **포맷**(커밋 전 필수): `docker run --rm -v /home/m2on/projects/gufo:/src -w /src nixos/nix sh -c 'git config --global --add safe.directory /src; nix --extra-experimental-features "nix-command flakes" shell --inputs-from . nixpkgs#clang-tools nixpkgs#python3 -c sh -c "clang-format -i <files> && chown 1000:1000 <files>; python3 tools/ci/check-format.py"'`
- **GPU**: 직접 `docker run --device`로 GPU를 쓰지 않는다. ai.tools 창 실행기만 쓴다(소유 lock, quiet 검사, health). 창 계획은 `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-K1*.json`에 `driver: test.experiments.gufo_eval_20261020.unit` 형식으로 쓰고(예: `PLAN-K0.json`), 고친 뒤 `python3 test/experiments/gufo_upstream_port_20261023/LOCKS.py`를 돌린다. 실행:
  ```sh
  cd ~/projects/ai.tools && python3 -m test.experiments.gufo_upstream_port_20261023.window --plan PLAN-K1B.json --model <entry> \
    --image ai-tools/gufo-eval:663d50c9f9d6 --out /home/m2on/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/<run>/<entry> --seconds <n>
  ```
  컨테이너는 읽기 전용이다: 쓰기는 `/receipt/run` 아래만, 프로파일은 `ROCPROF_TMPDIR=/receipt/run`. 창 하나에 한 번만 돌리고 실패해도 자동 재시도하지 않는다. 실패 원인과 결과를 그대로 남긴다. 한 번에 창 하나만 돈다(`docker ps`로 확인).
- **자동 fallback 없음**: 지원하지 않는 조합은 이름을 들어 거절한다. 몰래 fp16으로 바꾸지 않는다.
- 커밋은 자기 단계의 파일만 `git add`한다(다른 에이전트가 같은 트리에서 일한다). push·PR·이슈는 하지 않는다. 커밋 끝에 이 세션의 attribution 두 줄을 붙인다(감독이 프롬프트로 준다).
- 끝에 보고: 바꾼 파일·커밋, 돌린 검사와 결과(실패 포함 원문 요약), 측정 표, 남은 문제. 짐작은 짐작이라고 쓴다.

## 참고 위치

- K0 참고 구현: `tests/models/qwen27b/kvarn_fake_quant.hpp`(양자화 순서·scale 규칙의 기준), `tools/qwen27b/kvarn.py`(NumPy 독립 구현).
- Qwen 27B KV: 16 full-attention 층, KV head 4, head_dim 256, GQA 6, FP16, 토큰당 64 KiB. 풀·블록(256토큰): `src/core/hip/kv_block_layout.hpp`, `kv_block_pool.hpp`, `kv_block_table.hpp`. arena: `src/models/qwen/hip/arena.cpp`(`AllocateKvBlocks` 975 부근, `KvPlaneBase`).
- policy: `src/models/qwen/hip/execution_policy.hpp`(`QwenKvCacheStorage`, `Fingerprint()`), continuation layout `src/models/qwen/continuation_layout.cpp`(K/V component, `layout_version`).
- KV 쓰기 커널: `FusedQKNormRoPEKvWriteKernel`(`kernels/fused.hip`), `BatchedFusedQKNormRoPEKvWriteKernel`(`kernels/prefill_rope.hip`). 읽기: `kernels/attention_decode.hip`(`QwenDecodeSplitKAttentionHalfPartialsKernel` 618, reduce 712, `LaunchCausalDecodeAttention` 910), `attention_decode_verify_wave64.hip`(DFlash2 verify), `attention_decode_graph.hip`, prefill `attention_wmma.hip`. 선언 `src/models/qwen/hip/ops/attention.hpp`. (경로는 `src/models/qwen/hip/` 기준)
- 커널 최적화 규칙: `.agents/skills/optimize-kernel/SKILL.md`. 대역폭 상한: `tools/bench/gfx1151_peak.hip`. 기존 microbench 예: `tools/bench/attn_causal_bench.hip`, `tools/qwen27b/attention_bench.hip`.
- K0 측정 기준: AR decode attention은 32K마다 토큰당 10.2 ms(약 208 GB/s, 대역폭에 묶임), DFlash2 d64K에서 decode GPU 시간의 33 %가 attention(KVARN.md "K0 결과").

## K1a: `--kv-cache-type` 배선 (CPU)

- 값 `fp16`, `kvarn-k4v2`, `kvarn-k4v4`. serve·prompt·chat·bench 등 Qwen KV를 쓰는 명령에서 같은 이름으로 받는다(어디서 policy를 만드는지 찾아 한 곳으로). 문서(`docs/CLI.md`, `docs/SERVER.md`, 모델 README)에 적는다.
- `QwenKvCacheStorage`에 두 값을 추가하고 fingerprint·continuation identity에 반영한다(형식이 다르면 checkpoint가 섞이지 않게).
- 기본값: 값을 주지 않으면 모델별 기본. **지금은 모든 모델 fp16**(K2 gate 뒤 Qwen 27B만 k4v2로 바꾼다). 바꿀 자리가 한 곳이 되게 만든다.
- Qwen 27B 외 모델(Flash-Next, Gemma, DeepSeek 등)에 kvarn 값을 명시하면 시작 때 모델 이름을 들어 거절한다.
- Qwen 27B에 kvarn 값을 명시하면 아직 커널이 없으므로 executor 생성 시 "not implemented yet" 류의 명확한 오류로 거절한다(K1b·K1c·K2가 이 거절을 단계적으로 없앤다). fp16은 지금과 비트 단위로 같은 동작이어야 한다.
- 테스트: 옵션 파싱·거절·fingerprint 차이·기본값의 CPU 테스트. 기존 CPU suite에서 관련 테스트(`tools/qwen27b/check.py` 참고, `check-pr` 대상 중 CLI·cache·layout) 통과. GPU 실행은 하지 않는다(필요하면 보고에 적는다).

## K1b: 커널 시제품과 microbench (GPU, standalone)

- `tools/bench/kvarn_decode_bench.hip` 하나에(필요하면 헤더 하나) 다음을 만든다. 운영 코드는 건드리지 않는다.
  1. 양자화 커널: 128토큰 그룹이 찼을 때 (층, KV head)마다 K tile `[256,128]`, V tile `[128,256]`을 Hadamard 회전(FP16 반올림) → 정규화 8회 → RTN → 패킹·FP16 scale로 쓴다. 저장 배치(블록 256토큰 = 그룹 2개, 패킹 K·V와 scale의 위치)를 제안하고 문서화한다.
  2. fused dequant split-K decode attention: q를 H로 회전, 첫 128토큰 sink와 미완성 tail은 FP16, 나머지는 양자화 블록을 읽어 dequant, 출력에 H. 기존 `QwenDecodeSplitKAttentionHalfPartialsKernel`+reduce의 구조·분할을 따른다. k4v2·k4v4.
  3. DFlash2 verify 변형(질의 여러 개, 기존 verify wave64 커널 모양).
- 정확도: 양자화 값(q·scale)은 `kvarn_fake_quant.hpp`와 같은 규칙. dequant 결과와 attention 출력은 CPU 참고(fake quant된 FP16 KV로 FP16 attention, 원래 좌표)와 비교해 오차를 보고한다. scale 1 ulp 차이는 K0에서처럼 허용 범위로 다룬다.
- 속도: 같은 바이너리 안에서 FP16 기준 커널(기존 커널을 복사해도 됨)과 번갈아, 깊이 4K·16K·32K·64K·128K·256K, AR(질의 1)과 verify(질의 8). 무작위 KV로 충분하다. 실효 대역폭과 `gfx1151_peak.hip` 상한을 함께. 양자화 커널의 그룹당 시간도.
- 빌드: eval 이미지의 hipcc로 `build/kvarn-bench/`에(`--offload-arch=gfx1151`, `-O3`). GPU 실행은 창 계획 `PLAN-K1B.json`(unit driver, argv에 `/src/build/kvarn-bench/...`)으로만.
- gate(감독이 판정): 깊이 32K 이상에서 FP16 대비 attention 시간이 분명히 줄고(목표는 바이트 비율 쪽), 4K에서 손해가 작을 것. 결과가 나쁘면 원인(ALU·LDS·점유율)을 `tools/prof/isa_mix.py` 등으로 보고한다.

## K1c: 운영 통합 (2026-10-26 사용자: "k1c진행")

K1a·K1b 결과와 감독 판정은 [KVarN](KVARN.md) "K1a·K1b 결과"에 있다. 시제품: `tools/bench/kvarn_decode_bench.hip`(저장 배치는 파일 머리 주석). 운영 코드를 바꾸므로 한 에이전트가 순서대로 한다.

### K1c-1: 저장·쓰기·decode·verify

- `--kv-cache-type kvarn-k4v2|kvarn-k4v4`일 때만 바뀐다. fp16 경로는 비트 단위로 그대로(코드 경로도 분기 밖에서 그대로)여야 한다.
- 저장: 블록 풀에 K1b 배치(층·256토큰 블록마다 `[그룹 2][KV head 4][record]`)와 시퀀스별 FP16 sink·tail 평면(회전된 값). 풀 바이트·예약 계산이 새 크기를 쓰게 한다(`kv_block_layout.hpp`의 plane, arena `AllocateKvBlocks`). 공유 풀·CoW·블록 매핑이 계속 맞게 한다.
- 쓰기: KV 쓰기 커널(decode·batched·prefill)이 kvarn일 때 회전해서 tail 평면에 쓰고, 그룹이 차면(투기 디코딩은 확정된 토큰만) flush 양자화 커널을 건다. 되감기(verify 거절)·truncate가 미flush tail과 맞게 한다.
- 읽기: decode split-K(그래프 캡처 경로 포함)와 DFlash2 verify(폭 1–8)를 K1b 커널로. split 수는 깊이·점유율에 따른 정책 한 곳(K1b 측정 표 기준).
- `QwenExecutionPolicyUnsupportedReason`의 kvarn 거절은 prefill(K1c-2)이 붙을 때까지 남겨 두되, 테스트·도구가 쓸 수 있는 내부 경로(예: 테스트 전용 허용)는 범위를 좁게 만든다. checkpoint·continuation layout(K2)은 계속 거절한다.
- 작은 정리: `tests/models/qwen27b/kvarn_fake_quant.hpp`와 `tools/qwen27b/kvarn.py`의 log·exp를 정확 반올림(double로 계산해 float로 한 번)으로 바꾸고, GPU 양자화와 비트 일치를 단위 테스트로 고정한다.
- 테스트: GPU 커널 단위 테스트를 `tests/models/qwen/hip/attention/`에(K1b check의 경우들: 그룹 0/1/30/254, tail, 질의 1·8, split 여러 개, 큰 위치), CPU 테스트(풀 크기·예약). GPU는 창으로만: 새 단위 테스트 + 기존 `qwen_attention_kv_block_table_ops_test`, `qwen27b_target_test`, `inference_backend_gpu_test`(fp16 동일성 확인).

### K1c-2: prefill과 32K 실제 경로

- prefill이 양자화된 앞 블록을 읽게 한다. 먼저 `PackAttentionHeads` 단계에서 dequant하는 방법을 검토하고, 속도·메모리 영향을 보고한다. 이 단계 끝에 Qwen 27B kvarn 거절을 없앤다(continuation layout·checkpoint는 K2까지 거절).
- `qwen27b_kvarn_k0`에 실제 경로 비교 모드: kvarn policy executor로 prefill + teacher forcing + free greedy를 해서 FP16 기준과 KL·top-1·gold, 그리고 K0 fake quant 결과(`KVARN.md` K0 표)와 비교한다.
- 32K 창: 위 비교(k4v2·k4v4), `gufo bench` AR·DFlash2 `-d 0,32768`을 fp16과 kvarn으로 번갈아. 256K는 하지 않는다.

## K1b-opt: decode 커널 최적화 실험 (2026-10-26 사용자, K1c와 병행)

- **파일 경계:** `tools/bench/kvarn_decode_bench.hip`만 고친다(필요하면 같은 디렉터리의 새 헤더 하나). 빌드는 `build/kvarn-opt/`(K1b의 `build/kvarn-bench/build.sh`를 복사해 경로만 바꿔 쓴다). 운영 코드·`build/cpu-test`·`build/gpu-test`·`build/kvarn-bench`는 K1c 에이전트 것이라 건드리지 않는다. 창 계획은 `PLAN-K1BOPT.json`.
- **목표:** AR k4v2를 대역폭 하한 대비 32K 66 %·256K 79 %(K1b)에서 올리기, 4K 고정 비용 줄이기.
- **후보(예시, 더 찾아도 됨):** int8 WMMA에 코드를 바로 넣기 / dequant를 LDS에서 한 번 해서 두 half-wave가 공유 / WMMA 대신 `v_dot*`나 packed FP16 / nibble 배치·scale 위치를 바꿔 unpack 줄이기 / wave당 토큰·head 수, 그룹 루프 구조 / 4K: launch 합치기, sink·tail 처리 통합, split 정책.
- **진행:** 후보마다 짧은 시제품을 같은 바이너리에 넣어 비교하고, 가망 없는 후보는 일찍 접고 이유를 남긴다. 한 후보에 매달리지 않는다.
- **판단 기준:** 정확도는 K1b check 경우에서 커널 오차 ≤ 1e-3, 양자화 결과는 K0 규칙과 비트 일치(K1b의 scale ulp 규칙). 속도는 K1b 커널·FP16 기준과 같은 바이너리에서 번갈아, 깊이 4K–256K, 질의 1·8, 대역폭 하한 대비 %와 명령 구성(`tools/prof/isa_mix.py`).
- **저장 배치 결론 필수:** 배치를 바꿔야 하는지, 바꾸면 무엇을. K2(checkpoint 형식)와 K1c 통합 전에 반영해야 한다.
- **GPU:** 창 실행기로만, 한 번에 하나. K1c 에이전트와 번갈아 쓴다: `docker ps`와 lock(`~/projects/ai.tools/storage/runtime/experiments/vllm-multi-selected-20261016/ABBA-OWNER.lock`)이 잡혀 있으면 기다렸다가 나중에 시작한다(시작 전 lock 거절은 실패한 실행이 아니다). 창 전에 `.gufo-revision`을 HEAD로, 창 끝에 `/bin/sleep 20`. 실패하면 자동 재시도하지 않고 기록을 남긴다.
- **보고:** 후보별 결과(채택·기각과 이유), 추천 조합, 배치 변경 필요 여부, 정확도 표, 깊이×형식×질의 속도 표(K1b 대비), 남은 병목, 다음 후보 제안(양자화 커널 튜닝: double 분산·spill, FP16 verify 기준 커널 개선).

### K1b-opt2: verify에도 K 접기 (2026-10-26 사용자: "verify에도 K 접기를 넣기 시도해보자")

- 목표: `opt_latev`의 K scale 접기를 verify(질의 1–8)에도 넣고, **같은 위치의 verify 행과 1행 decode가 비트 단위로 같은 산술**이 되게 한다(greedy DFlash2 = AR, K1c 미해결 B와 같은 요구). bench에 같은 위치·같은 KV에서 AR 결과와 verify 각 행 결과의 비트 비교 check를 넣는다.
- 지난 보고의 걸림돌(LDS에 q 두 벌 + 48행 보정)을 풀 후보: 보정을 WMMA로, q를 레지스터에 압축 유지, 행 묶음 크기 조정 등. 한 후보에 매달리지 않는다.
- 같은 바이너리에서 비교: K1b, `opt_latev`, 새 커널, 그리고 **공통 최적화만(K 접기 없음) 쓴 AR**. 깊이 4K–256K, 질의 1·8, 바이트 하한 대비 %, 명령 구성.
- 안 되면: 증거와 함께 보고하고, 비트 일치를 지키는 최선 조합(공통 최적화만)의 속도를 낸다.
- 경계·GPU 규칙은 K1b-opt와 같다(K1c 에이전트가 같은 트리·GPU를 쓴다).

## K1d: `common` 커널 운영 반영과 짧은 prefill 손해 (2026-10-26 사용자: "넵 진행해주세용")

결과와 규칙은 [KVarN](KVARN.md) "K1b-opt2 결과", 시제품은 `tools/bench/kvarn_decode_bench.hip` `81e3fdcd`의 `common` 커널(읽기만, 이 파일은 고치지 않는다).

- 운영 decode·verify 커널을 `common`으로 바꾼다: biased FP16 코드, 밑 2 softmax(`v_exp_f32`), AR padding 행 생략, AR 마지막 블록 combine, verify combine 커널, 늦은 V, tail PV 쌍 펼치기와 64토큰 tail 블록, 명시적 FMA, split은 그룹 수로만. K 접기는 하지 않는다. 저장 배치는 그대로.
- 지금 운영의 verify = decode 비트 일치 수정(K1c-1 B)은 `common`의 규칙으로 통일한다. 비트 일치 테스트(exact=1, max_abs 0)는 계속 통과해야 한다. 운영 FP16 verify 커널에 같은 WMMA k 쌍 위험이 있는지 확인하고 보고한다(고치지 않고 보고만, fp16 경로는 그대로).
- 짧은 prompt prefill 손해(pp512 @ d0 fp16 482 → kvarn 461, −4.4 %)의 원인을 프로파일로 찾아 고친다. 목표: d0에서 fp16 대비 손해가 측정 잡음 수준(요청별 5 %·3 ms 규칙 안이 아니라, 원인이 남지 않게).
- 확인(32K 창 하나): 커널 단위 테스트·decode 테스트·fp16 동일성 3종, `qwen27b_kvarn_k0 --real` 32K, `gufo bench` AR·DFlash2 `-d 0,32768` fp16·k4v2·k4v4 번갈아 2회. K1c-2 수치(KVARN.md)와 같은 표로 비교한다. 256K는 하지 않는다.
