# R10: ROCm 10.1 빌드의 end-to-end 개선 (2026-10-28)

사용자: "10.1과 기존 7.2.4로 비교하고.. 좋아졌으면 유지야", "엔드단이 좋아지는게 중요", "PR안할꺼야.. 좋은것만 가져올거고.. 근거 강한 순서대로 개선방안 만들어서 진행시키자."

## 목표와 채택 기준

- 같은 소스(병합본 `ai-tools/port-u4f-upstream-c92b19c5`, worktree `~/projects/gufo-merge`)를 ROCm 10.1 / HIP Clang 24로 빌드했을 때 운영 구성의 서빙 지표가 ROCm 7.2.4 빌드보다 같거나 좋아지게 한다.
- 채택: 간이 검사에서 32K cold prefill·pp2048/tg128(깊이 0·8K, AR·DFlash2)이 7.2.4 이상이고, 나머지가 노이즈 범위다. 좋아진 변경만 남기고, 효과 없는 변경은 되돌린다(스위치·환경 변수로 남기지 않는다).
- upstream PR은 하지 않는다. 7.2.4 빌드에서 느려지는 변경은 쓰지 않는다(두 툴체인 모두 확인).

## 출발점 (`um2-a01`, long32k ABBA 각 2회, 운영 인자 kvarn-k4v2·DFlash2)

|요청|7.2.4|10.1|
|---|---|---|
|32K cold prefill|540–560 tok/s|510–518 tok/s (-6~8 %)|
|짧은 다음 턴 TTFT|227–229 ms|214–221 ms|
|decode|56–70 tok/s|58–73 tok/s|

upstream 자체 측정(`docs/models/qwen3.8-27b/ROCM.md`): 10.1 prefill -3~5 %, calibration prefill +5 %, 원인 대부분 양자화 target GEMM, 일부 BF16 draft feature projection.

## 후보 (근거 강한 순서)

|#|후보|대상|근거|
|---|---|---|---|
|C1|양자화 target GEMM의 Clang 24 코드 생성|`src/models/qwen/hip/kernels/prefill_quant_gemm.{hip,hpp}`|upstream 프로파일. Clang 24 대응 없음, launch bounds는 7.2 기준|
|C2|BF16 GEMM(hipBLASLt) 선택 차이|`src/models/qwen/hip/dflash/executor.cpp`, `prefill_chunk.cpp`의 hipBLASLt 경로|upstream 언급(draft feature projection). 라이브러리 버전 차이|
|C3|포트 attention 커널의 Clang 24 미튜닝|`attention_wmma.hip` block-table 경로(2-wave), `attention_kvarn.hip`|upstream은 head-major 경로만 6-wave·sched_barrier. 32K는 attention 비중이 커서 차이가 upstream보다 큼|
|C4|런타임(patched ROCr·CLR 10.1)|nix 패키지|launch·동기화 쪽, 가능성 낮음|

## 단계

- **S0 진단(감독 세션):** 같은 32K cold prefill을 두 이미지에서 `tools/prof/prof.py run`으로 잡고 `prof.py diff`로 stage·kernel별 차이를 본다. 차이가 큰 커널부터 C1~C3에 배정한다. 지목된 커널은 두 툴체인으로 그 파일만 컴파일해 VGPR·spill·occupancy를 비교한다(`-Rpass-analysis=kernel-resource-usage`).
- **S1~S3:** S0에서 지목된 순서로 한 단계씩. 한 메커니즘만 바꾸고, 해당 operator 테스트로 정확성을 확인한 뒤 간이 검사(아래)로 판단한다.

## 규칙

- 빌드: 10.1은 `nixos/nix` 컨테이너, `/nix`는 docker volume `gufo-nix`, `nix build path:/src#default`(소스는 `git archive` 사본). 7.2.4는 `ai-tools/gufo-eval-deps:7c2c5a72-audio` 이미지에서 CMake `release` 증분 빌드. 영향 타깃만 빌드한다.
- GPU: 창 하나씩, 공유 lock(`ABBA-OWNER.lock`)을 다른 세션과 같이 쓴다. GPU 타이밍 창이 도는 동안 CPU 빌드·ISA 덤프를 하지 않는다. 시작 전에 `docker ps`로 다른 창이 없는지 본다.
- 간이 검사: T3 bench(pp2048/tg128, 깊이 0·8K, AR·DFlash2, 3회)와 32K cold prefill 1건을 두 툴체인에서 한 번씩. 결과는 좋아짐/같음/나빠짐/노이즈로 읽는다. 5 %·3 ms 판정과 ABBA는 채택 직전에만 한다.
- 정확성: 바꾼 커널의 operator 테스트(gpu-test)를 두 툴체인 모두에서 통과해야 한다. 출력 텍스트가 툴체인 사이에 다른 것 자체는 회귀가 아니다(upstream ROCM.md).
- 결과 기록: 이 파일 아래 "결과"에 단계마다 한 줄(채택/거부, 수치, 실행 경로).

## S2: DFlash2 decode -2.6 % (에이전트 작업)

**근거:** 7.2.4에서 DFlash2 decode를 프로파일하면(`tmp/test-results/gufo-merge-c92b19c5/prof/a-df-tg.show`) 시간의 55 %가 검증 GEMV `SmallBatchKQuantExactFp32GEMMKernel`(`src/models/qwen/hip/kernels/small_batch_gemm.hpp`, 인스턴스는 `small_batch_quant16_wave64.hip`·`small_batch_wave64.hip`)이다. 상위 12개 인스턴스의 Clang 24 ISA는 7.2.4와 명령 수·global load 수가 같은데, `s_waitcnt_depctr`(주로 `depctr_va_sdst`·`depctr_va_vcc`)가 커널당 22~110개 새로 생긴다(7.2.4는 0~2). 64비트 VGPR 주소 계산(`v_add_co_u32` → carry SGPR → `v_add_co_ci_u32`)마다 LLVM 24의 gfx1151 hazard 대기가 붙는다. 이 대기는 하드웨어 hazard 대응일 수 있으므로 컴파일러 옵션으로 끄지 않는다.

**할 일:** 검증 GEMV의 가중치·scale 읽기를 64비트 VALU 주소 계산 없이 하도록 바꾼다. 예: 기준 포인터를 SGPR에 두고 32비트 오프셋으로 읽기(buffer resource + `__builtin_amdgcn_raw_buffer_load_*`, 또는 컴파일러가 `saddr` 형식을 쓰게 하는 32비트 오프셋 표현). 한 메커니즘만 바꾸고 측정한다.
- 산술 순서·누적 순서·reduction은 그대로다("Exact" 계약). 주소 계산과 load 형식만 바꾼다.
- 7.2.4(Clang 22) 빌드도 같은 소스를 쓴다. 7.2.4에서 느려지면 채택하지 않는다(필요하면 `#if __clang_major__ >= 24`).
- 주소 범위: NarrowIndex가 아닌 인스턴스는 64비트 오프셋이 필요할 수 있다. 32비트로 줄일 수 없는 곳은 건드리지 않는다.

**확인 방법:**
1. ISA: 두 툴체인 code object에서 해당 커널의 `s_waitcnt_depctr` 수와 명령 수. 추출 `python3 -I /tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/isa/extract.py <gufo binary> <out dir>`(gfx1151 code object를 `co*.o`로 꺼냄), 디스어셈블 `nixos/nix` 컨테이너에서 `/nix/store/5bv9b1y1l2yr9qnp27krmgm3nnq3n0bc-llvm-binutils-24.0.0-rocm/bin/llvm-objdump -d --mcpu=gfx1151 --disassemble-symbols=<sym> <co.o>`(volume `gufo-nix`). 대상 심볼 목록: `tmp/test-results/gufo-merge-c92b19c5/isa/top-df.json`, 기존 a(7.2.4)·f(10.1+asm WMMA) 디스어셈블리: `isa/{a,f}/df*.s`.
2. 속도(간이): DFlash2 `bench ... --speculative dflash2 --dflash-model /models/qwen-draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf -p 2048 -n 128 -r 2 --temperature 0 --seed 1`와 AR `-p 2048 -n 64 -r 3`을 7.2.4·10.1(현재 f)·10.1(후보)·7.2.4(후보)에서 ABBA. 명령 예와 이미지는 `tmp/test-results/gufo-merge-c92b19c5/prof/abf-32k-dflash.log`를 만든 방식과 같다(이미지 `ai-tools/gufo-eval:64849ca6ec7b`가 7.2.4, nix 결과 경로를 `--entrypoint`로 주면 10.1).
3. 정확성: 7.2.4 gpu-test 빌드(`~/projects/gufo-merge/build/gpu-test`, 이미지 `ai-tools/gufo-eval-deps:7c2c5a72-audio`에서 `/src`로 마운트해 빌드)로 `qwen_dflash_gpu_test`, `qwen27b_target_test`, `inference_backend_gpu_test`(모델 경로는 `test/experiments/gufo_upstream_port_20261023/PLAN-P1.json` p1-units 참고)를 돌린다. 10.1은 `bench --validate-prefill 2048` 지표와 DFlash2 bench의 accepted/proposed가 수정 전 10.1과 같아야 한다.

**빌드:** 10.1은 `tmp/test-results/gufo-merge-c92b19c5/src-ls`(현재 asm WMMA가 들어간 사본)를 새 디렉터리로 복사해 바꾼 파일만 덮어쓰고 `nix build path:/src#default`(위 "규칙"). 소스 원본은 `~/projects/gufo-merge`에서 고친다. 커밋하지 않는다.

**GPU·CPU 규칙:** GPU를 쓸 때마다 `flock -n /home/m2on/projects/ai.tools/storage/runtime/experiments/vllm-multi-selected-20261016/ABBA-OWNER.lock`으로 잡고, 시작 전 `docker ps`로 다른 창이 없는지 본다. GPU 측정 중에는 CPU 빌드를 하지 않는다. 다른 세션(gufo-d7 등) 창과 겹치면 그 측정은 버리고 다시 한다. 디렉터리 삭제가 필요하면 새 이름을 쓴다(`rm -rf` 변수 경로 금지).

**보고:** 바꾼 메커니즘, ISA 수치(depctr·명령 수), ABBA 수치, 정확성 결과, 채택/거부를 이 파일 "결과"에 한 줄로 쓰고 감독 세션에 요약한다.

## S3: DFlash2 decode 검증 GEMV의 VGPR 증가 (에이전트 작업)

**근거(2026-10-28):** 10.1 코드를 7.2.4 런타임으로 돌리면 7.2.4 rocprofv3가 붙는다(`tmp/test-results/gufo-merge-c92b19c5/prof/fr-df-tg`, 명령은 아래). decode 구간 커널 합이 7.2.4 대비 +3.5 %(4322 → 4472 ms)이고, 증가분 150 ms의 절반이 인스턴스 하나다: `SmallBatchKQuantExactFp32GEMMKernel<16, 5, 4, IQ4_XS(23), 1, 4, 1, true, 1, 64, 16, true, false, false>`(wave64, 512 스레드, Batch 5) 183 → 255 ms(+39 %). 메타데이터는 VGPR 174 → 209, spill 없음, LDS 같음. 나머지 증가는 같은 커널 계열의 다른 인스턴스(+2~11 %)다. S2(depctr 제거)는 효과가 없었으니 다시 하지 않는다.

**할 일:** Clang 24에서 이 커널 계열의 VGPR을 7.2.4 수준으로 되돌려 occupancy를 회복한다. 후보: `__launch_bounds__`의 최소 블록 수, `__attribute__((amdgpu_waves_per_eu(...)))`, `amdgpu_num_vgpr`. `#if __clang_major__ >= 24`로 Clang 24에만 적용하고, 한 메커니즘씩, 먼저 느려진 인스턴스(wave64)부터 본다. spill이 생기면 거부한다. 산술은 바꾸지 않는다.

**확인:** (1) ISA 메타데이터(VGPR·spill·occupancy)를 S2의 추출 방법으로 본다. (2) decode 커널 시간은 10.1 후보 바이너리를 7.2.4 이미지에서 7.2.4 런타임으로 돌려 7.2.4 rocprofv3로 잡는다(아래 명령의 FW를 후보 바이너리의 `.gufo-wrapped`로). `fr-df-tg`와 같은 인자에서 인스턴스별 decode 합을 비교한다(감독 세션의 비교 스크립트: `/tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/` 아래가 아니라 직접 sqlite로 `kernels`·`kernel_symbols`를 kernel_id로 조인해 첫~마지막 `SmallBatchKQuantExact` 사이를 decode 구간으로 본다). (3) 채택 전 DFlash2 tg128·pp2048과 AR tg64·pp2048을 7.2.4 / 10.1 현재(f) / 10.1 후보로 ABBA(S2의 `prof/s2-abba1` 방식). (4) 정확성: DFlash2 acceptance와 `--validate-prefill 2048` 지표가 10.1 현재와 같아야 한다.

**프로파일 명령(7.2.4 이미지, 10.1 코드, 7.2.4 런타임):** `prof/fr-df-tg.log`를 만든 방식. 이미지 `ai-tools/gufo-eval:64849ca6ec7b`, volume `gufo-nix:/nix`, `env LD_LIBRARY_PATH=/nix/store/9nhrjcichkgxyb5d0sg3ikvxh7aad4ay-glibc-2.44/lib:/nix/store/7a2cjhcgzyym5zaz6m4fksw47qvm150w-gcc-16.2.0-lib/lib:/opt/rocm/lib:/usr/lib/x86_64-linux-gnu <후보>/bin/.gufo-wrapped bench -m /models/qwen/Qwen3.8-27B-UD-Q4_K_M.gguf --kv-cache-type kvarn-k4v2 --speculative dflash2 --dflash-model /models/qwen-draft/Qwen3.8-27B-DFlash2-Q4_K_M.gguf -p 512 -n 256 -r 1 --temperature 0 --seed 1`을 `python3 /tools/prof/prof.py run -o /out/<이름> --` 뒤에 둔다(tools는 `~/projects/gufo-merge/tools`를 `/tools`로 마운트).

**빌드·GPU·CPU 규칙·보고:** S2와 같다. 소스는 `~/projects/gufo-merge`에서 고치고(S1의 `prefill_fp16.hip` 변경은 그대로 둔다), 10.1 빌드는 `tmp/test-results/gufo-merge-c92b19c5/src-ls`를 새 디렉터리로 복사해 바꾼 파일만 덮어써서 `nix build`. GPU 창 전 `df -h /`(150 GiB 이상)와 `docker ps`를 본다. 다른 세션(gufo-d7)이 모델을 내려받는 중일 수 있다. bench·프로파일은 디스크를 쓰지 않으니 그대로 진행하되 결과에 적는다.

## S4: 검증 GEMV를 Clang 24에서 spill 없이 (에이전트 작업)

사용자(2026-10-28): "일단 스필 없이 만드는 작업을 하고, 테스트를 새로하자."

**근거:** S3 에이전트의 리소스 비교(`/tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/s3/base-cmp.txt`, 형식 `템플릿 인자  Clang22 VGPR/occupancy/scratch/spill -> Clang24 ...`). `small_batch_wave64.hip`의 LogicalLanes 16·wave64·IQ4_XS(T23)·IQ4_NL(T20) Batch 12/14/16 인스턴스 29개가 Clang 24에서 VGPR 256 한도에 걸려 spill 59~326, scratch 232~872 B, occupancy 3~4 → 2. Clang 22는 162~242 VGPR, spill 0. 여러 DFlash2 요청을 함께 검증하면 이 배치가 쓰인다. 또 Batch 5 IQ4_XS(`<16,5,4,T23,...,64,16,...>`)는 spill 없이 VGPR 174 → 209로 occupancy 4 → 3이 돼 단일 요청 decode가 느려졌다(S3, 192로 묶으면 spill 1).

**목표:** Clang 24에서 위 인스턴스가 spill 0이 되고, occupancy가 Clang 22보다 낮아지지 않게 한다. 출력은 비트 단위로 같아야 한다("Exact" 계약: 각 출력 행의 곱·누적·reduction 순서 그대로).

**방향(우선순위):**
1. Clang 24에서만 이 인스턴스들의 디스패치 구성을 바꾼다. 예: 행 수(`RowsPerWave`)·토큰 분할(`TokensPerStep`/`TokenGroups`)·`TilesPerStage`·LogicalLanes를 조정해 누산기 배열(`sums[kPhases][RowsPerWave][Batch]`) 크기를 줄인다. 행·토큰 분할을 바꿔도 각 출력 행의 계산 순서는 같아야 한다. 같은지 근거(코드상 이유)를 보고에 적는다.
2. 1로 안 되면 커널 안의 unroll·live range를 Clang 24에서만 조정한다(`#pragma unroll` 상한, `__builtin_amdgcn_sched_barrier` 등). 산술 순서는 그대로.
3. 컴파일러 hazard·정확성 옵션은 바꾸지 않는다.

**확인:**
- ISA/리소스: S3의 단일 파일 컴파일 스크립트(`.../scratchpad/s3/ru24.sh`, `ru22.sh`)와 파서(`ru.py`)로 두 툴체인의 모든 인스턴스 VGPR·occupancy·spill을 본다. Clang 22 쪽 결과가 바뀌면 그 이유를 적는다(가능하면 Clang 22는 그대로).
- 정확성: 7.2.4 gpu-test(`~/projects/gufo-merge/build/gpu-test`, 이미지 `ai-tools/gufo-eval-deps:7c2c5a72-audio`, `/src` 마운트)로 `qwen_dflash_gpu_test`, `qwen27b_target_test`, `inference_backend_gpu_test`를 돌린다. 이 테스트들이 Batch 12~16 검증을 실제로 지나는지 확인하고, 안 지나면 그 배치를 지나는 테스트(예: 검증 GEMV operator 테스트가 있는지 `tests/models/qwen*/` 검색)를 찾아 돌린다. 10.1은 `bench --validate-prefill 2048` 지표와 DFlash2 acceptance가 10.1 현재와 같아야 한다.
- 속도는 여기서 길게 재지 않는다. 짧은 확인만: DFlash2 `bench ... -p 512 -n 128 -r 2`를 10.1 현재(f)와 후보로 한 번씩. 동시 요청 서빙 테스트는 감독 세션이 S4 뒤에 새로 한다.

**빌드·GPU·CPU 규칙·보고:** S2·S3와 같다(소스는 `~/projects/gufo-merge`, S1의 `prefill_fp16.hip` 변경 유지, 커밋 안 함, 10.1은 `src-ls` 사본을 새 디렉터리로 복사해 `nix build`, GPU는 lock과 `docker ps`·`df -h /` 확인 뒤, 다른 세션 gufo-d7이 sha256 계산 중이면 lock이 잡혀 있을 수 있다). 보고에 바꾼 인스턴스별 전/후 VGPR·occupancy·spill(두 툴체인), 정확성 결과, diff(file:line), 채택/거부를 적는다. 채택이면 소스를 남기고, 10.1 후보 nix 결과 경로를 적는다.

## S5: R10-RESEARCH 적용과 남은 요청 깃발 (에이전트 작업)

사용자(2026-10-28): "10.1 과 7.2.4의 차이 문서를 철저하게 조사해서 최적화방안이있는지 확인히고 계확세워서 우리코드에 적용해보자", um4 뒤 진행 확인("굿").

**출발점:** um4(아래 결과)에서 10.1(S1+S4)이 7.2.4와 같거나 +1~2 %. 이미지 `ai-tools/gufo-eval:64849ca6ec7b-rocm101-s4`(nix 결과 `/nix/store/6g8ln6ljsqj57sj82xqx2j9ca0cyqask-gufo-0-unstable-dirty`, 소스 사본 `tmp/test-results/gufo-merge-c92b19c5/src-s4`). 후보 목록·근거·순서는 `R10-RESEARCH.md`.

**할 일(순서):**
1. R10-RESEARCH의 A2 1단계(빌드 없음): hipBLASLt 선택 solutionName을 두 툴체인에서 비교, 10.1 바이너리를 7.2.4 라이브러리로 돌린 DFlash2 pp2048과 비교.
2. um4 long32k에서 10.1이 두 번 다 느린 두 요청을 조사한다: T3/0 짧은 다음 턴 prefill 230 → 310~331 ms, T1/1 cache snapshot 21 → 117~130 ms(`tmp/test-results/gufo-upstream-port-20261023/um4-a01/um4{a,b}-long32k{,-r2}`). 같은 요청을 bench 또는 짧은 서버 실행으로 재현해 원인 커널·런타임 호출을 찾는다(10.1 코드를 7.2.4 런타임으로 돌리는 방법으로 프로파일 가능, S3 참고).
3. R10-RESEARCH의 A1(attention asm WMMA, kvarn 포함), A3(k9 spill), B1/B2(스케줄러 옵션, 파일별·Clang 24만), B4(`-fno-slp-vectorize` 필요 여부), B3(스케줄 전략) 순으로 한 메커니즘씩. A4는 하지 않는다.

**채택 기준:** 10.1에서 해당 지표가 좋아지고(간이 ABBA), 7.2.4는 그대로(Clang 22 리소스·코드 불변 또는 측정상 같음), 정확성(해당 operator 테스트, `--validate-prefill`, acceptance) 유지. 효과 없거나 노이즈면 되돌린다.

**빌드·GPU·CPU 규칙·보고:** S2~S4와 같다. 소스는 `~/projects/gufo-merge`(S1·S4 변경 유지, 커밋 안 함). 10.1 빌드는 `src-s4`를 새 디렉터리로 복사해 바꾼 파일만 덮어쓴다. GPU 창 전 `df -h /`(150 GiB)·`docker ps`·lock. 다른 세션 gufo-d7은 lock으로 조율한다. 각 항목마다 "결과"에 한 줄. 끝나면 채택된 변경 목록, 항목별 수치, 10.1 최종 후보 nix 결과 경로를 보고한다.

## 결과

- **S0 진단(2026-10-28):** pp2048 bench ABBA에서 10.1 손실은 깊이 0이 -8~9 %, 16K가 -6~7 %라 attention이 아니라 GEMM이다. 32K AR prefill(7.2.4 rocprofv3)에서 `HalfPrefillGemmKernel`(`prefill_fp16.hip`)이 GPU 시간의 약 66 %. 두 바이너리의 gfx1151 code object를 비교하니 메모리 명령·WMMA 수는 같고, Clang 24가 WMMA 두 개마다 `v_lshrrev_b32` 8 + `v_mov_b16 .h` 8(real-true16 lowering의 자기 자신 재조립)을 넣고 `s_waitcnt`가 커널당 15~17개 늘었다. 소스의 피연산자 구성(union → bit_cast, 빈 asm)으로는 그대로였다. 10.1 rocprofiler-sdk는 캐시에 없어(clang 24 빌드 필요) code object 비교로 대신했다. 결과: `tmp/test-results/gufo-merge-c92b19c5/{prof,isa}/`.
- **S1 C1 시도:** `-target-feature -real-true16` 빌드는 이동 대신 no-op `v_bfi_b32`가 생겨 pp2048 -2.6 %(거부). **Clang 24에서만 `WmmaHalf`를 `v_wmma_f32_16x16x16_f16` inline asm으로 내보냄(채택 후보)**: 불필요 명령이 모두 사라지고 명령 수가 7.2.4보다 적다. ABBA 각 2회(AR, kvarn-k4v2): pp2048 d0 7.2.4 663/640 · 10.1 611/596 · asm 651/651, d16K 573/550 · 525/519 · 561/560, pp32768 561/561 · 526/526 · 568/566, DFlash2 pp2048 597/597 · 553/558 · 601/601. `--validate-prefill 2048` cosine 7.2.4 0.94773, 10.1 0.94783, asm 0.94783(10.1과 모든 자리 같음). 7.2.4 빌드는 `#if __clang_major__ >= 24`로 기존 intrinsic 그대로.
- **남은 것:** DFlash2 tg128이 10.1에서 -2.6 %(7.2.4 44.9/45.0, 10.1 43.8/43.9, asm 43.8/43.8). AR tg는 10.1이 +0.7 %라 draft 쪽(C2 후보). k9 GEMM 인스턴스의 spill 7은 남음. GEMM operator 테스트(`q4kxl_quant_ops_test`)는 Clang 24 gpu-test 빌드 방법이 없어 아직 못 돌림.
- **S2 거부(2026-10-28):** NarrowIndex 검증 GEMV의 가중치·scale 읽기를 SGPR 기준 + 32비트 바이트 오프셋 + 상수 field offset으로 바꿈(`DecodeQuantSub16Narrow`, Q4_K·Q5_K·Q6_K·IQ4_XS, 산술 그대로). ISA(상위 8개 NarrowIndex 인스턴스, 단독 컴파일): Clang 24 depctr 326→145, `v_add_co_ci` 278→93, 명령 13631→13303, global load 28개 중 24개가 saddr 형식; Clang 22 depctr 15→15, 명령 13849→13736; 남은 depctr는 x staging·y store 주소. ABBA(a 7.2.4, f 10.1 현재, g 10.1 후보, h 7.2.4 후보, a f g h h g f a): DFlash2 tg128 a 44.96/45.02 · f 43.91/43.89 · g 43.56/43.60 · h 44.79/44.74, DFlash2 pp2048 597/602 · 601/600 · 601/602 · 602/600, AR tg64 12.94/12.95 · 13.05/13.05 · 13.04/13.04 · 12.94/12.94, AR pp2048 641/638 · 647/644 · 646/644 · 640/637. 두 툴체인 모두 DFlash2 decode가 0.4~0.8 % 느려져 거부, 소스 되돌림. `--validate-prefill 2048`·acceptance 0.683은 각 툴체인 수정 전과 같음. hazard 대기는 -2.6 %의 원인이 아니다(검증 GEMV는 메모리 대역 한계). 결과·patch: `tmp/test-results/gufo-merge-c92b19c5/prof/s2-abba1/`.
- **S2 후속 진단(2026-10-28):** 10.1 바이너리(asm WMMA)를 7.2.4 이미지의 HIP·HSA·hipBLASLt·rocBLAS로 돌려도 DFlash2 tg128(p512)이 44.0/44.1로 그대로(7.2.4 45.3/45.3, 10.1 44.4/44.1) → 런타임·라이브러리(C2 hipBLASLt, C4 CLR)가 아니라 컴파일된 코드. 7.2.4 프로파일의 decode 구간은 GPU busy 96 %(쉬는 시간 4 %)라 호스트도 아니고, 검증 GEMV(S2)도 아니므로 남은 것은 decode의 다른 GPU 커널(draft 쪽 등). 같은 날 디스크가 99 %(여유 18G)였다 → cache-disk 468 GiB 정리 후 487G 여유. bench는 디스크를 쓰지 않으므로 이 차이와는 무관하지만, 앞선 창의 디스크 복원 시간(100/700 ms 이봉)은 디스크가 찬 상태에서 잰 것이다.
- **디스크(2026-10-28):** 루트 99 % → cache-disk 468 GiB 정리 → `fstrim` 490 GiB. 같은 바이너리(병합본 7.2.4)의 regress·shared 디스크 복원이 정리 전 최대 464~766 ms(이봉), 정리 후 다른 세션 다운로드 중 최대 119~1100 ms, trim 후 쓰기 없이 최대 158~207 ms(shared 복원 TTFT 중앙값 385~837 → 319~323 ms). 앞선 UM1의 복원 시간 깃발은 디스크 상태였다(`um3-disk-a01`, `um3-trim-a01`).
- **S3 근거(2026-10-28):** 10.1 코드를 7.2.4 런타임·rocprofv3로 프로파일(`prof/fr-df-tg`). decode 커널 합 +3.5 %, 증가분의 절반이 wave64 Batch 5 IQ4_XS 검증 GEMV 하나(183 → 255 ms, VGPR 174 → 209).
- **S3 거부(2026-10-28, ISA 단계):** wave64 Batch 5 IQ4_XS 검증 GEMV(`SmallBatchKQuantExactFp32GEMMKernel<16,5,4,IQ4_XS,1,4,1,true,1,64,16,true>`)에 Clang 24에서만 occupancy 4를 요구. (a) 호출부 `MinWaves` 4→8(`__launch_bounds__` 두 번째 인자 2→4 = waves-per-eu 4): Clang 24 VGPR 209→192, occupancy 3→4, **VGPR spill 1(scratch 8 B/lane)**. (b) 탐침 `amdgpu_num_vgpr(192)`: 같은 192 VGPR·spill 1·scratch 8 B. occupancy 4의 한도(wave64 192 VGPR)에서 spill 없이 맞출 수 없어 두 방법 모두 거부, GPU 측정 없이 소스 되돌림(Clang 22는 174 VGPR·occ 4·spill 0 그대로). 참고: 기준 Clang 24에서 LogicalLanes 16 wave64 IQ4_XS·IQ4_NL Batch 12/14/16 인스턴스(`<8,12|14|16,3|4,T23|T20,...,64,16,...>`)는 VGPR 256 한도에 걸려 spill 59~326개(scratch 232~872 B, Clang 22는 162~242 VGPR·spill 0)지만 p512/n256 DFlash2 decode 프로파일에는 나오지 않는다(decode는 Batch ≤ 8). 결과: `/tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/s3/{base-cmp.txt,out/*.rem}`(cc24/cc22 단일 파일 `-Rpass-analysis=kernel-resource-usage`), decode 인스턴스 비교 `/tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/s3/dec.py`.
- **S4 채택 후보(2026-10-28):** Clang 24에서만 검증 GEMV의 토큰 단계(`TokensPerStep`)를 넓힘(`small_batch_wave64.hip`의 `kLaneStep<Clang22, Clang24>`, `#if __clang_major__ >= 24`): LL16 wave64 IQ4_XS Batch 12/14/16(down·up·packed SwiGLU, groups 1–4·ragged) 1→2, IQ4_NL 14/16 1→Batch(2에서도 spill), Batch 5 IQ4_XS 1→5, 추가로 Batch 6 Q4_K `<16,6,3,…,MinWaves 8>` 6→3(Clang 24 occupancy 6→5였음). 단계는 독립 토큰 dot을 묶을 뿐이라 각 출력의 곱·누적·reduction 순서는 그대로. 리소스(단일 파일 `-Rpass-analysis`): Clang 22 236개 인스턴스 전부 수정 전과 동일; Clang 24 대상 30개 spill 59~326 → 0, VGPR 256 → B12 171(o4)·B14 226/223(o3)·B16 242/239(o3)·NL14 223(o3)·NL16 247(o3)·B5 209(o3) → 142(o5)·Q4_K B6 122(o5) → 120(o6), Clang 22보다 낮은 occupancy 없음(남은 spill은 범위 밖 BF16 `BatchedExactBf16GEMMFp32VecKernel<16,24,2>` 1개). 정확성: Clang 24 `dflash_gemm_bench`(production dispatch, 단일 행 decode GEMV와 비트 비교) IQ4_XS 34816×5120·5120×17408·17408×5120 rows 5–64, packed SwiGLU 14–64, IQ4_NL 14/16, Q4_K B6 mismatch 0; 7.2.4 gpu-test `qwen_q4kxl_quant_ops_test`·`qwen_dflash_gpu_test`·`qwen27b_target_test`·`inference_backend_gpu_test` 통과(원래 설정, 그리고 Clang 24 단계를 강제한 임시 빌드에서도 통과·target 해시 동일; rocprofv3로 q4kxl·target·backend가 바뀐 B5/B12/B14/B16/NL/Q4_K B6 인스턴스를 실제로 지남을 확인, dflash 테스트는 지나지 않음). 커널 시간(Clang 24 수정 전 → 후, Clang 22): IQ4_XS up B16 4.31 → 0.86 ms(0.83), B12 2.07 → 0.74(0.71), B14 3.56 → 0.79(0.74), SwiGLU 16 4.34 → 0.89(0.84), NL16 2.62 → 0.46(0.43), B5 0.67 → 0.46(0.45); 여전히 Clang 22보다 1~7 % 느림. 10.1 `--validate-prefill 2048` cosine 0.94782525(수정 전과 모든 자리 같음), DFlash2 p512/n128 r2 acceptance 0.683 같음, tg128 43.92 → 44.76(1회씩, 간이). 후보 nix 결과 `/nix/store/6g8ln6ljsqj57sj82xqx2j9ca0cyqask-gufo-0-unstable-dirty`(`tmp/test-results/gufo-merge-c92b19c5/{src-s4,nix-out-s4,prof/s4-quick}`), 리소스·커널 로그 `/tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/s4/`.
- **UM4(2026-10-28, `um4-a01`, 디스크 정리·trim 뒤, 다른 작업 없음):** 7.2.4(`64849ca6ec7b`) 대 10.1+S1+S4(`64849ca6ec7b-rocm101-s4`) ABBA 각 2회. bench: AR pp +1.4~2.0 %, AR tg +1.2~1.5 %, DFlash2 pp +0.9~2.0 %, DFlash2 tg 깊이 0 -0.4 %(노이즈 안)·32K +1.8 %. 시나리오 합계: long32k·256k4·shared·two8k prefill -0.2~-1 %, decode 같음, 64k8 wall 1249/1251 → 1206/1238 s. 요청 단위로 10.1이 두 번 다 느린 것: long32k T3/0 prefill 230 → 310~331 ms, T1/1 snapshot 21 → 117~130 ms(S5에서 조사). draft 수 차이는 컴파일러 수치 차이(acceptance 같음). **판정: 10.1 유지 후보**(사용자 기준 "좋아졌으면 유지").
- **S5-1 A2 닫음(2026-10-28, 빌드 없음):** `GUFO_DISPATCH_TELEMETRY=1` DFlash2 pp2048에서 `hipblaslt` 이벤트가 두 툴체인 모두 0건이고 나머지 dispatch(attention·qwen_route·hip_graph) 기록이 같다. 운영 구성(target IQ4_XS/Q4_K mix, draft Q4_K_M)은 BF16 가중치가 없어 hipBLASLt를 타지 않는다(`dflash/executor.cpp:420`, `prefill_chunk.cpp:166`). DFlash2 pp2048(-n 32 -r 2, a 7.2.4 / b 10.1 S4 / r 10.1 S4 코드 + 7.2.4 런타임·라이브러리, a b r r b a): a 605/602, b 628/618, r 602/601 → 10.1 런타임 쪽이 오히려 +3 %이고 라이브러리 순위 문제는 없다. 결과 `tmp/test-results/gufo-merge-c92b19c5/s5-a2/`.
- **S5-2 long32k 두 요청 깃발(2026-10-28, 원인 미확정·변경 없음):** 같은 harness(`scenario.py` cache, 운영 서버 인자 그대로)를 8K·1K(q0) case로 줄여 재현. T3/0(재시작 뒤 첫 요청, 디스크 복원 + 33토큰 prefill)의 T3/1 대비 초과분: 7.2.4 +12~18 ms(10회 모두), 10.1 S4 +12~322 ms(꼬리: 8K 58·322·93, q0 30·71·53), 10.1 S4 코드 + 7.2.4 런타임 +49 ms(1회). T1/1 snapshot(8K): 7.2.4 20~23 ms(3/3), 10.1 20~58 ms(6회 중 3회 54~58), 10.1 코드 + 7.2.4 런타임 19 ms(1회). 서버 로그의 background publish 순서·stream_wait는 꼬리가 있는 실행과 없는 실행이 같다. 가설로 Clang 24의 기본 offload 번들 압축(10.1 바이너리 73개 bundle 전부 CCOB zstd, 101.9 MB → 15.3 MB, 7.2.4는 비압축; 첫 launch 때 해제)을 확인하고 `--no-offload-compress`(Clang ≥ 24, CMake) 빌드(`/nix/store/rn4xaam1d9c7fr72fgnhnf673z2kzmvn-gufo-0-unstable-dirty`)로 재현했으나 T3/0 꼬리가 그대로(q0 +15·13·110·74·16, 8K +26·15)라 원인이 아니다 → 되돌림. 남은 판단: 꼬리는 10.1 런타임(CLR/ROCr) 쪽 첫 요청·D2H 복사의 간헐 지연으로 보이며(C4), 소스로 고칠 근거는 없다. 확인하려면 10.1 rocprofiler-sdk가 필요하다. 주의: 앞 8K 실행 일부는 cache-disk가 tmpfs(/tmp)에 있었고 마지막 두 q0 실행은 그 메모리 압박으로 load 실패했다(정리 후 SSD로 옮겨 다시 잼). 결과 `tmp/test-results/gufo-merge-c92b19c5/s5-rep/`.
- **S5-3 A1 거부(ISA 단계):** `attention_wmma.hip`·`attention_kvarn.hip`의 `Wmma`를 Clang 24에서 asm으로. Clang 22 code 동일. 운영 prefill(`prefill_wmma_kvarn_k4v2` = PackHeads + `WmmaCausalAttention<32,16,true>`)은 VGPR 240→238·명령 14212→14242·`v_mov_b16 .h` 80→80으로 변화 없음(이 커널의 16비트 이동은 WMMA 피연산자 재조립이 아니다; `v_lshrrev_b32`는 두 툴체인 모두 0). kvarn decode `KvarnDecodePartialsKernel`은 Clang 24가 이미 Clang 22보다 명령·`.h` 이동이 적고, asm으로 바꾸면 `<4,1>`·`<4,3>`·`<4,4>`·`<4,5>`·`<2,2>`에서 occupancy가 8→7, 8→7, 8→6, 7→6, 9→8로 떨어진다. 비운영 block-table `<32,16,false>`만 spill 3→0. GPU 측정 없이 되돌림.
- **S5-4 A3·B1·B2 거부(ISA 단계):** k9 `HalfPrefillGemmKernel<256,256,8,4,Q4_K,kGateUp,false,IQ4_XS>` Clang 24 v192 o8 scratch 16 B spill 3(루프 안 scratch load 4)·Clang 22 v164 o9. (A3) Clang 24에서만 이 조합에 `iglp_opt(0)` → 리소스 그대로(spill 3), 되돌림. (B1) `-amdgpu-scheduler-pending-queue-limit=0`은 `prefill_fp16.hip`·`small_batch_wave64.hip`·`small_batch_quant16_wave64.hip` 모두 code object가 바이트 단위로 같다(효과 없음). (B2) `-amdgpu-vgpr-threshold-percent=95`는 k9 spill 그대로·occupancy 8개 하락, `=90`은 k9 spill 0(v178 o8)이지만 TU 82개 커널 중 occupancy 9개 하락(Q5_K/Q6_K 256 타일 9→8)·16개 상승이라 거부.
- **S5-5 B4·B3 거부(ISA 단계):** (B4) Clang 24에서 `prefill_quant_gemm.hip` SLP 켬: spill 커널 3개·occupancy 하락 2개, 명령 +0.2 % → `-fno-slp-vectorize` 유지. (B3) `small_batch_quant16_wave64.hip` Clang 24: 현행 iterative-ilp 명령 161143(Clang 22 160069, +0.7 %), 기본 스케줄러 +4.3 %, max-memory-clause +5.0 %(occupancy·spill 모두 같음) → 현행 유지. **S5 채택 변경 없음.** 소스는 S1(`prefill_fp16.hip`)+S4(`small_batch_wave64.hip`) 그대로, 10.1 최종 후보는 `/nix/store/6g8ln6ljsqj57sj82xqx2j9ca0cyqask-gufo-0-unstable-dirty`(이미지 `64849ca6ec7b-rocm101-s4`). 리소스·ISA 로그 `/tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/s5/ru/out/`.
