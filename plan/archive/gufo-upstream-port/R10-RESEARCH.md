# R10 조사: ROCm 7.2.4 → 10.1 변화와 gfx1151 적용 계획 (2026-10-28)

R10-BRIEF.md(S0~S4)와 upstream `docs/models/qwen3.8-27b/ROCM.md`의 내용은 반복하지 않는다. 이 문서는 빌드·GPU 실행 없이 소스 비교(태그 `rocm-7.2.4` vs `therock-10.1`), 변경 기록, 기존 ISA 덤프(`ai.tools/tmp/test-results/gufo-merge-c92b19c5/isa/{a,b,f}`)를 읽어서 만들었다. [미확인]은 1차 출처로 확인하지 못한 항목이다.

## 1. 출처

- ROCm 10.1.0 release notes: https://rocm.docs.amd.com/en/docs-10.1.0/about/release-notes.html (LLVM 24.0.0. 컴파일러 항목에는 버전만 있고 gfx1151 성능 항목은 없다)
- ROCm 10.0.0 release notes: https://rocm.docs.amd.com/en/docs-10.0.0/about/release-notes.html (LLVM 23 → 24, HIP 7.15)
- ROCm 7.11 preview notes: https://rocm.docs.amd.com/en/7.11.0-preview/about/release-notes.html (llama.cpp pp 회귀, 우회 `-mllvm --amdgpu-unroll-threshold-local=600`)
- CLR/HIP 변경 기록(태그 therock-10.1): https://github.com/ROCm/rocm-systems/blob/therock-10.1/projects/clr/CHANGELOG.md
- hipBLASLt 변경 기록(태그 therock-10.1, **10.1의 hipBLASLt는 1.4.1**): https://github.com/ROCm/rocm-libraries/blob/therock-10.1/projects/hipblaslt/CHANGELOG.md. develop의 1.5.0(gfx11 tensor swizzle, runtime tuning cache)은 10.1에 없다.
- gfx1151 hipBLASLt logic 변경(7.2.4 이후, 10.1 포함): rocm-libraries #8398(skinny_N GridBased retune, 2026-07), #9238(gfx1151 BF16 K=1024 grid entries), #10761(gfx1151 bf16 TN Equality tuning), #11744(gfx115x KernArgsVersion 기본값). 경로 `projects/hipblaslt/library/src/amd_detail/rocblaslt/src/Tensile/Logic/asm_full/gfx1151`. 7.2.4 태그의 마지막 변경은 2025-06이다.
- ROCm LLVM 소스 비교(gh api, 태그 `rocm-7.2.4` / `therock-10.1`): `AMDGPU.td`, `GCNSubtarget.h`, `GCNHazardRecognizer.cpp`, `AMDGPUWaitSGPRHazards.cpp`, `GCNSchedStrategy.cpp`, `GCNIterativeScheduler.cpp`, `SISchedule.td`, `AMDGPUTargetTransformInfo.cpp`, `AMDGPUTargetMachine.cpp`, `clang/lib/CodeGen/Targets/AMDGPU.cpp`. 사본: `/tmp/claude-1000/-home-m2on-projects-gufo/590906a5-ff4f-4fcd-a20c-9748dbf14792/scratchpad/r10/`
- LLVM true16: gfx115x 기본 활성 https://www.phoronix.com/news/AMDGPU-LLVM-RDNA3-True16 (2025-07). packed build_vector 패턴 제거 #148715 https://llvm.googlesource.com/llvm-project/+/5138b61a25f11eb8675d0031712c1ee6b4cb8be4, #149861 https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20250825/1713305.html
- GFX11 VALU mask-write hazard 재작성 #138663 https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20250505/1626665.html, wait 병합 #169213 https://lists.llvm.org/pipermail/llvm-commits/Week-of-Mon-20251117/1783360.html
- `amdgpu_num_vgpr` 등 deprecated(LLVM 23): https://releases.llvm.org/23.1.0/docs/ReleaseNotes.html
- `amdgpu-sched-strategy` 값(main): https://github.com/llvm/llvm-project/blob/main/llvm/lib/Target/AMDGPU/AMDGPUTargetMachine.cpp. 값은 max-ilp, max-memory-clause, iterative-ilp, iterative-minreg, iterative-maxocc, coexec(gfx950/gfx1250 전용)이다.

## 2. 확인한 변화 (gfx1151 관련만)

|#|영역|7.2.4 → 10.1|우리 코드에 미치는 영향|
|---|---|---|---|
|F1|true16|gfx11 공통 feature에 `FeatureRealTrue16Insts`가 **두 버전 모두** 있다(7.2.4에도 이미 켜져 있음). 차이는 그 뒤의 true16 isel 변경이다. packed build_vector 패턴이 제거돼 16비트 레지스터 기반 build_vector + `v_mov_b16 .h`로 바뀌었다(#148715/#149861, 메커니즘은 추정)|S1의 WMMA 피연산자 재조립과 같은 형태다. **attention WMMA에도 남아 있다**: `WmmaCausalAttention<32,16,true>`의 ISA(a=7.2.4, b=10.1, 둘 다 WMMA 16개)에서 `v_lshrrev_b32` 6→78, `v_mov_b16 .h` 64→80, scratch 0→0|
|F2|feature set|`FeatureISAVersion11_5_1`이 `Feature1_5xVGPRs`에서 `Feature1536VGPRs`로 이름이 바뀌었다(내용 같음). 11_5 공통에 `Dot5`, `WMMA256bInsts`, `DX10ClampAndIEEEMode` 명시|코드 생성 차이는 없을 것으로 본다[미확인]|
|F3|hazard|`fixVALUMaskWriteHazard`가 7.2.4에서는 SALU 쓰기만 봤는데, 10.1에서는 **VALU의 SGPR 쓰기(캐리 `v_add_co_u32`)도** 본다. **wave64에서만** 적용(`if (!ST.isWave64()) return false`). gfx11에서는 `AMDGPUWaitSGPRHazards`가 wait 병합만 하고, cull 옵션은 gfx12 VALUReadSGPR용이다|S2의 depctr 증가 원인을 확정했다. 정확성 대응이고 끌 수 있는 옵션이 없다. S2에서 효과가 없었으므로 종결한다|
|F4|스케줄러|10.1에 **pending queue 검사**가 새로 생겼다(`amdgpu-scheduler-pending-queue-limit`, 기본 256, 0이면 끔. 7.2.4에는 이 로직이 없다). `amdgpu-vgpr-threshold-percent`(RP 한도를 %로 축소, 기본 0)도 새로 생겼다. stage 순서는 같고, RewriteMFMAForm은 MFMA 전용이다. GFX11 sched model은 Dummy write 추가 외에 같다. iterative/ILP scheduler는 거의 같다|S3/S4에서 본 Clang 24 VGPR 증가, 남은 1~7 % 느림, k9 spill의 후보 원인이다(가설). 파일 단위 `-mllvm`로 실험할 수 있다|
|F5|unroll|TTI unroll 기본값은 같다(threshold 300, local 1000). `amdgpu-unroll-runtime-default`만 새로 생겼다(기본 꺼짐)|7.11의 llama.cpp 우회 옵션은 7.2→10.1 차이가 아니다. 우선순위 낮음|
|F6|clang|`__launch_bounds__`/`amdgpu_waves_per_eu` lowering은 같다. `amdgpu_num_vgpr`/`num_sgpr`는 LLVM 23부터 deprecated다|S3처럼 `amdgpu_num_vgpr`를 쓰지 않는다. occupancy 지정은 `__launch_bounds__` 두 번째 인자(waves per EU)로 한다|
|F7|hipBLASLt|1.2.x → 1.4.1. StreamK 기본값 AUTO→OFF, SM count target, uniform summation order가 생겼고 **gfx1151 BF16 logic이 재튜닝됐다**(위 PR)|`hipblaslt_gemm.hip`은 heuristic 결과의 **고정 순위(4·5·마지막)**를 고른다. 목록 순서가 바뀌면 다른 kernel이 선택된다. draft feature projection(DFlash2 prefill)과 non-quant prefill이 이 경로를 탄다|
|F8|rocBLAS|gfx11 Level-3 gemm에서 m==1 또는 n==1이고 batch_count==1이면 gemv 경로를 쓴다|`LaunchHipblasGEMMBF16` fallback은 batch>1에서만 쓰여 해당 없음|
|F9|HIP 런타임|7.11~7.12: graph segment 스케줄링과 dispatch batching(`AccumulateCommand`)으로 CPU 오버헤드 감소, `hipGraphLaunch` 정리. 10.0: `hipEventDisableTiming` 이벤트 병합. 10.1: Windows APU의 hostRegister 기본값 fine-grain(Linux는 해당 없음)|AR decode는 이미 graph를 쓴다(`decode.cpp:77-99`, 10.1 AR tg +0.7 %와 맞는다). DFlash2 decode는 graph를 쓰지 않는다. 이벤트는 이미 DisableTiming(`prefill.cpp:26`, `arena.cpp:574`, `transfer_pool.cpp:169`)|
|F10|ROCr|1.21: vmem API가 CPU pool을 받는다. gfx1151 성능 항목은 없다|CWSR patch(`rocr-fixed-cwsr-rocm10.patch`)는 유지한다|

## 3. 후보 (근거 강한 순서)

### A. 소스 변경

**A1. attention WMMA를 Clang 24에서 inline asm으로 (근거: ISA, 강함)**
- 대상: `src/models/qwen/hip/kernels/attention_wmma.hip:103-105`(`Wmma`, 호출 349·492), `attention_kvarn.hip:73-75`(호출 1134·1187, 운영 kvarn-k4v2 경로). S1의 `prefill_fp16.hip:20-27`과 같은 `#if __clang_major__ >= 24` asm(`v_wmma_f32_16x16x16_f16 %0, %1, %2, %0`, `"+v"(c)`). 세 곳이 같은 helper를 쓰도록 `kernels/` 안의 공통 헤더 하나로 묶을 수 있다(Qwen 모델 안에 둔다).
- 예상 효과: head-major는 명령 14.2k 중 약 90개를 덜어 1 % 미만이다. kvarn prefill attention은 아직 덤프하지 않았다. 32K cold prefill에서 attention 비중이 크므로 거기서 확인한다. 7.2.4는 변하지 않는다.
- 위험: asm WMMA는 스케줄러가 지연을 모른다. S1에서는 이득이었다. block-table(2-wave) 경로의 Clang 24 spill 대응(`attention_wmma.hip:529-535` sched_barrier)과 상호작용할 수 있으니 VGPR·spill을 확인한다.
- 검증: (1) 두 툴체인으로 단일 파일을 컴파일해 `-Rpass-analysis=kernel-resource-usage`(S3 `ru24.sh`/`ru22.sh`/`ru.py`)로 VGPR·spill·occupancy가 같거나 나은지, (2) objdump로 커널별 `v_lshrrev_b32`·`v_mov_b16 .h`·`s_waitcnt` 수가 7.2.4 수준인지, (3) `qwen_attention_wmma_prefill_ops_test`, `qwen_attention_kvarn_ops_test`, `qwen_attention_long_context_ops_test`(7.2.4 gpu-test), 10.1은 `--validate-prefill 2048` cosine이 이전과 모든 자리까지 같은지, (4) 간이: 32K cold prefill과 pp2048 d16K.

**A2. hipBLASLt 고정 순위 선택의 10.1 재확인 (근거: 라이브러리 logic 변경, 중간~강함. 효과 크기는 미확인)**
- 대상: `hipblaslt_gemm.hip:32-36`(`kHipblasLtLargeWideRank=4`, `kHipblasLtTallRank=5`), `:77-92`(`SelectHipblasLtHeuristicRank`), `:238-283`. 호출부는 `dflash/executor.cpp:432-440`, `prefill_chunk.cpp:170-180`.
- 1단계(빌드 없음): 같은 10.1 바이너리를 `GUFO_DISPATCH_TELEMETRY`로 돌려 `hipblaslt` 이벤트의 `solutionName`을 7.2.4 바이너리와 비교한다(DFlash2 pp2048). S2 후속처럼 10.1 바이너리를 7.2.4 라이브러리로 돌려 **DFlash2 pp2048**도 잰다. decode는 이미 라이브러리와 무관함을 확인했다. 차이가 없으면 A2는 닫는다.
- 2단계: 차이가 있으면 `tools/bench/tune_hipblaslt.cpp`로 해당 shape(telemetry의 batch·m·k)의 heuristic 순위별 시간을 두 라이브러리에서 잰다. 순위 상수가 10.1에서 최선이 아니면 **런타임 hipBLASLt 버전**(`CurrentPlanDatabaseKey`가 이미 읽는 `hipblaslt_version`)으로 순위 표를 고른다. 이 경우는 clang 매크로가 아니라 라이브러리 버전으로 고른다. 환경 변수나 스위치는 만들지 않는다.
- 위험: 낮음(같은 BF16 GEMM, 출력은 FP32 누적이지만 kernel이 바뀌면 비트가 달라질 수 있다). 검증: `hipblaslt_runtime_cache_test`, `qwen_dflash_gpu_test`, DFlash2 acceptance 동일 여부, DFlash2 pp2048·32K.

**A3. k9 `HalfPrefillGemmKernel` Clang 24 spill 제거 (근거: ISA, 중간)**
- 대상: `<256,256,8,4,Q4_K(12),Epilogue 2,false,UpType IQ4_XS(23)>`(isa/f/k9: scratch 명령 7, 7.2.4는 0). 정의는 `prefill_fp16.hip:229`(`__launch_bounds__(WM*WN*32)`)이고, 루프는 `:520-545`(이 조합은 `iglp_opt(0)` 대상이 아니다).
- 시도 순서(한 번에 하나): (a) 아래 B1/B2 knob으로 원인을 진단한다(파일 단위). (b) Clang 24에서만 이 조합의 decode 결과 live range를 줄인다(Q4_K scale·min 언팩 후 `sched_barrier`, 또는 unroll 상한). 산술 순서는 바꾸지 않는다.
- 검증: 리소스(spill 0, occupancy 유지), `qwen_q4kxl_quant_ops_test`, `qwen_prefill_quant_gemm_ops_test`, `--validate-prefill 2048`, 간이 pp2048.

**A4. DFlash2 decode의 graph 캡처 (근거: 런타임 개선 + GPU busy 96 %, 낮음~중간. 7.2→10.1 차이와는 무관하고 두 툴체인 모두에 이득)**
- 10.1 graph segment·batching은 AR decode(`decode.cpp:77-99`)에만 쓰인다. DFlash2 draft·verify 단계는 stream 직접 launch다. 쉬는 4 %가 상한이다. 작업이 크고 캐시·배치 shape 키가 필요하므로 A1~A3와 B 뒤에 둔다. 검증은 DFlash2 tg128 ABBA, acceptance 동일, `qwen_graph_ops_test`.

### B. 툴체인 knob (CMake 파일 단위, `CMAKE_HIP_COMPILER_VERSION VERSION_GREATER_EQUAL 24`로만)

**B1. `-mllvm -amdgpu-scheduler-pending-queue-limit=0` (근거: 스케줄러 소스 비교, 중간. 먼저 진단용)**
- 10.1에서만 생긴 스케줄링 동작을 끄면 7.2.4 스케줄에 가까워진다. 대상 TU: `small_batch_wave64.hip`, `small_batch_quant16_wave64.hip`(S4 뒤에도 Clang 22보다 1~7 % 느림), `prefill_fp16.hip`(k9). 리소스 비교만으로 1차 판단하고, VGPR·명령 수가 7.2.4 쪽으로 가면 커널 시간(S3 방법: 10.1 바이너리, 7.2.4 런타임·rocprofv3)을 잰다. `-mllvm`은 TU 전체에 걸리므로 같은 TU의 모든 인스턴스를 비교한다.

**B2. `-mllvm -amdgpu-vgpr-threshold-percent=<90~97>` (근거: 새 옵션, 낮음~중간)**
- spill이 남은 TU(k9의 `prefill_fp16.hip`, `BatchedExactBf16GEMMFp32VecKernel<16,24,2>` spill 1이 있는 TU)에서 스케줄러 RP 한도를 낮춰 spill이 사라지는지 본다. occupancy를 낮추는 쪽으로 가면 거부한다.

**B3. `small_batch_quant16_wave64.hip`의 `iterative-ilp` 재평가 (근거: Clang 22/23 시절 튜닝, 중간)**
- `src/models/qwen/CMakeLists.txt:100-107`. Clang 24에서 기본(max-occupancy), `iterative-ilp`, `max-memory-clause`를 비교한다. 검증 GEMV는 메모리 대역 한계(S2)이므로 `max-memory-clause`가 그럴듯한 후보다. 판단은 S3 방식의 인스턴스별 decode 커널 합으로 하고, 채택 전에 DFlash2 tg128 ABBA를 한다.

**B4. `-fno-slp-vectorize` 재확인 (근거: Clang 23 spill 대응이 Clang 24에도 필요한지, 낮음)**
- `src/models/qwen/CMakeLists.txt:108-116`. Clang 24에서 SLP를 켠 단일 파일 리소스를 본다. Q3_K/Q6_K 128×128 spill이 없고 명령이 줄면 Clang 24에서만 제거한다. 아니면 그대로 둔다.

**B5. 하지 않을 것**
- `-real-true16`를 끄는 것은 S1에서 거부됐다. F3 hazard는 끌 옵션이 없고 정확성 대응이다. `amdgpu-unroll-threshold-local`은 TTI 기본값이 같아서 근거가 없다. `-mcumode`는 툴체인 차이가 아니고 근거도 없다. wave reduce builtin은 reduction 순서가 정해지지 않아 Exact 계약 커널에 쓰지 않는다. hipBLASLt gfx11 swizzle·tuning cache는 10.1에 없다(1.5.0).

## 4. 진행 순서

1. **빌드 없이 GPU 창 하나:** A2 1단계(telemetry `solutionName` 비교, 10.1 바이너리 + 7.2.4 라이브러리로 DFlash2 pp2048)를 한다. 결과로 A2를 진행할지 닫을지 정한다.
2. **CPU만(GPU 창 사이):** A1 리소스·ISA 비교와 kvarn attention 덤프, B1·B2·B4 단일 파일 리소스 비교. 근거가 생긴 것만 GPU 단계로 넘긴다.
3. A1을 정확성 확인 후 간이 검사(32K cold, pp2048 d0·16K)한다.
4. A3(B1/B2 진단 결과로 방법을 고른다)를 한다.
5. B3(+B1 결과)로 DFlash2 decode 잔여 차이를 본다.
6. A4는 위가 끝난 뒤 별도로 판단한다.

각 단계는 R10-BRIEF.md 규칙(한 메커니즘씩, 두 툴체인 모두 확인, 7.2.4가 느려지면 거부, 효과가 없으면 되돌림, "결과"에 한 줄)을 따른다.
