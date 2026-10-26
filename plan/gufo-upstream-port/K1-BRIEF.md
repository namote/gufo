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
