# R2 결과: 로더와 회전 융합 실행 (W3, 2026-10-29)

상위 문서: [계획](PLAN.md) "R2 로더와 실행", [W3 브리프](W3-BRIEF.md). 이 문서는 1·2단계(경로 표, 설계, 코드, CPU 테스트, 빌드 확인)까지의 상태다. GPU 창은 아직 돌리지 않았다. 그래서 GPU 테스트 결과와 ②③④, 성능은 비어 있다.

## 전제 수정 (감독 승인 2026-10-29)

PLAN은 "27B는 matmul 전에 활성화를 Q8_1로 양자화한다"고 적었지만, 이것은 prefill에만 맞다.

- decode GEMV(`LaunchGEMV` → `LaunchQ8KBlockGEMV`)와 verify(`LaunchBatchedQuantGEMMFp32`, `LaunchExactBf16GEMMFp32SmallBatch`)는 FP32 활성화를 그대로 읽는다.
- 그래서 decode·verify의 회전은 FP32 회전이다. "회전 후 Q8_1/FP16/BF16 인코딩" 융합은 prefill 생성 커널에만 해당한다.

## 경로 표 (그룹별 활성화 생성 지점)

표기:

- **R-norm**: 회전 norm 변형. 정규화 값은 원래 커널과 같은 산술로 만들고, 행을 LDS에 둔 뒤 128 구간마다 회전해서 쓴다.
- **단독**: `LaunchRotateRowsFp32`(FP32 제자리 회전, wave 하나가 구간 하나).
- **융합**: 생성 커널 안에서 회전한 뒤 인코딩한다.

### decode (`hip/decode_step.cpp` `ExecuteDecodeStep`, graph capture 대상)

| 그룹 | 생성 지점 | 소비 | 회전 |
|---|---|---|---|
| attn_in | `NormForward` → `LaunchRMSNorm`(FP32 `normed`) | `LaunchFusedQKVProjections` / `LaunchFusedSSMInputProjections`(FP32 x) | `LaunchRMSNormRotated`(R-norm) |
| attn_out_in | `LaunchKvarnDecodeAttention` / `LaunchAttention`(split-K 포함) → `ssm_scratch.out` | `QuantGemm` → `LaunchGEMV` | 단독(층마다 1회) |
| ssm_out_in | `LaunchSSMConvRecurrence` 에필로그 → `ssm_scratch.out` | `LaunchGEMV` | 단독 |
| ffn_in | `LaunchRMSNorm` | `FfnForward` → `LaunchFusedSwiGLUGEMV` | `LaunchRMSNormRotated` |
| ffn_down_in | `LaunchFusedSwiGLUGEMV`(GEMV 안에서 SwiGLU, 블록당 출력 8행) | `LaunchGEMV` | 단독. 회전 층은 FfnForward 모듈 대신 같은 두 launch 사이에 회전을 넣는다 |
| head_in | `LaunchRMSNorm`(output_norm) | `LaunchGEMV`(output) | `LaunchQwenOutputHead`(R-norm + GEMV) |

### 다중 세션 decode와 DFlash2 verify (`hip/batched_decode.cpp`의 두 함수)

| 그룹 | 생성 지점 | 소비(텐서 타입별 분기) | 회전 |
|---|---|---|---|
| attn_in | `LaunchBatchedRMSNorm`(FP32, bf16 없음; 5120·≤64행은 cached 커널) | `LaunchProjection`: Q8_0/Q8_K/K-quant/IQ → `LaunchBatchedQuantGEMMFp32`; BF16 → `LaunchExactBf16GEMMFp32SmallBatch`; 그 밖 → 행마다 `LaunchGEMV`. SSM 제어: `LaunchSsmControls`(Q8_0 packed alpha/beta 또는 `LaunchProjection`) | `LaunchBatchedRMSNormRotated`(같은 커널 선택) |
| attn_out_in | `LaunchKvarnDecodeAttention` / `LaunchAttention` / verify의 `LaunchCausalDecodeAttention`, `LaunchKvarnPassAttention` → `ssm.out` | `LaunchProjection` | 단독(행 전체 한 launch) |
| ssm_out_in | `LaunchSSMConvRecurrenceBatch` → `ssm.out` | `LaunchProjection` | 단독 |
| ffn_in | `LaunchBatchedRMSNorm` | `LaunchFfnActivation`: (a) packed gate/up(인접 텐서, Q4_K/Q5_K/Q6_K/IQ4_XS/Q8_0, 17408×5120, ≥2행) → `TryLaunchPackedQuantSwiGLUFp32`, 실패하면 packed `LaunchProjection` + `LaunchPackedSwiGLUActivation`; (b) 그 밖 → gate·up `LaunchProjection` + `LaunchBatchedSwiGLUActivation` | `LaunchBatchedRMSNormRotated` |
| ffn_down_in | 위 (a)/(b)의 SwiGLU → `ffn.activation` | `LaunchProjection` | 단독(분기와 무관하게 SwiGLU 뒤) |
| head_in | `LaunchBatchedRMSNorm`(output_norm) | `LaunchProjection(output)` | `LaunchBatchedRMSNormRotated` |

### prefill (`hip/prefill_chunk.cpp` `ForwardPromptChunk`)

분기 조건: `half_prefill`(`UseQwen27bFp16Prefill`: batch ≥ 1024, 27B 모양, 모든 투영이 Q8_0 또는 native K-quant/IQ이고 FFN에 저비트가 있음). 그 밖에는 `reads_q8_act`(Q8_0 또는 native K-quant/IQ: Q8_1 WMMA 경로)와 비-Q8 경로(BF16/F32 → `LaunchExactBf16GEMMFp32SmallBatch`(FP32 입력) / F32 hipBLAS; Q5_K 등 비-native 양자화는 dequant 후 hipBLASLt/hipBLAS BF16(BF16 입력))로 갈린다.

| 그룹 | half_prefill(FP16 활성화) | Q8_1 경로 | 비-Q8 경로(FP32+BF16) |
|---|---|---|---|
| attn_in | `LaunchBatchedRMSNormFp16`(HalfNorm5120) → **`LaunchBatchedRMSNormRotatedFp16`** | `norm_feeds_q8_only`: `LaunchBatchedFusedRMSNormQuantizeQ8_1` → **`…RotateQuantizeQ8_1`(LDS 회전 후 같은 양자화)**. 아니면 BF16에서 `LaunchQuantizeActivationQ8_1` | `LaunchBatchedRMSNorm`(FP32+BF16) → **`LaunchBatchedRMSNormRotated`**(두 형식 모두 회전값) |
| attn_out_in | attention(WMMA/tile/baseline/KVarN) → FP32 `d_ssm_out` → `LaunchFloatToFp16` | `LaunchQuantizeActivationQ8_1FromFp32` | `LaunchFloatToBfloat16` + `LaunchQuantizeActivationQ8_1` |
| | 모든 경로: 변환 앞에 **단독** FP32 회전(half는 `output_first_row`부터 `output_rows`행) | | |
| ssm_out_in | row-split FP16 에필로그 → **`BatchedSSMPostNormGateFp16Kernel<Rotate>`**(head 128 = 회전 구간 1개, strided 레이아웃 helper). 비 row-split → FP32 뒤 **단독** + `LaunchFloatToFp16` | row-split Q8 에필로그 `LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1` → **Rotate 변형(LDS)**. 비 row-split → FP32 뒤 **단독** + `…FromFp32` | row-split FP32 에필로그 뒤 **단독**(launcher 안), 비 row-split 뒤 **단독**, 그다음 BF16/Q8_1 변환 |
| ffn_in | **`LaunchBatchedRMSNormRotatedFp16`** | `ffn_feeds_q8_only`: residual 융합 norm → **Rotate 변형** | ResidualAdd + **`LaunchBatchedRMSNormRotated`** |
| ffn_down_in | `TryLaunchBatchedDualQuantGEMMSwiGLUFp16` / `LaunchBatchedQuantGEMMSwiGLUFp16`(GEMM 에필로그 SwiGLU) → **회전 층에서는 끔**: gate·up `LaunchBatchedQuantGEMMFp16`(FP32 출력) + **`LaunchBatchedSwiGLURotateFp16`** → `d_scratch_bf16` | `LaunchBatchedDualQuantGEMMSwiGLUQuantizeQ8_1`(GEMM 에필로그) → **회전 층에서는 끔**. `LaunchBatchedDualQuantGEMMPreQuantized` 또는 gate·up GEMM 뒤 **`LaunchBatchedFusedSwiGLURotateQuantizeQ8_1`** | `LaunchBatchedSwiGLUActivation`(FP32) → **단독** → `LaunchFloatToBfloat16` → `LaunchQuantizeActivationQ8_1` |
| head_in | 마지막 토큰 `LaunchRMSNorm` + `LaunchGEMV` → `LaunchQwenOutputHead` | 같음 | 같음 |

`CommitVerificationChunk`의 재생되지 않은 행은 `ForwardToken`(decode 경로)을 쓴다.

### DFlash2 드래프터와 그 밖의 소비자

| 경로 | 처리 |
|---|---|
| 드래프터 head(`hip/dflash/executor.cpp` 단일 요청, `hip/dflash/batch.cpp` 다중 요청) | 같은 normed 행을 `selector_hidden`(드래프터 가중치, 비회전)이 먼저 읽고, 그 뒤 `output`(타깃, 회전)을 읽는다. selector 투영 뒤 **단독** 제자리 회전(`QwenDFlashWeights::output_rotation` = 타깃 `head_in_rotation`). BF16 타깃 head는 `QuantizeDraftHead`가 회전된 가중치 그대로 Q8_0으로 다시 양자화하므로 문제가 없다 |
| 드래프터 자기 층, 임베딩 | 회전 없음(타깃 hidden·임베딩은 바뀌지 않음) |
| MTP(`hip/mtp/model.cpp`, `mtp_reference.cpp`) | head_in 회전이면 거부(`kMtpRotatedHeadError`). `d_feedback_hidden_`이 head 입력이면서 `CopyLastHidden`으로도 나가므로 v1에서는 지원하지 않는다. serve의 기본 `--speculative`는 off이다(MTP가 기본인 설정은 없음, `serve.cpp`) |
| `--cpu`(`QwenGenerator`) | 회전 파일이면 거부 |
| W2 헤드 직접 적용 | `hip/output_head.hpp` `LaunchQwenOutputHead`(회전 포함). decode와 prefill도 이 함수를 쓴다 |

### graph capture

- decode graph는 스트림 전체를 캡처한다. 회전 launch는 모델마다 고정이다.
- 부호 버퍼는 로드 때 한 번 `hipMalloc`으로 올리고(`UploadRotationSigns`), 층 포인터를 device 주소로 바꾼다. 그래서 캡처 중에는 할당이 없고 capture key도 바뀌지 않는다.
- 비회전 파일의 launch 수와 커널은 그대로다. 회전 변형은 `Rotate` 템플릿 인스턴스나 새 launcher로만 들어가고, 비회전 인스턴스의 산술은 바뀌지 않는다. 비회전 launch에는 `signs` 인자 하나(`nullptr`)만 더해졌다.

## 설계

- **공통 검사** `src/core/quant/rotation.{hpp,cpp}`: 표시 KV 4개와 `.rot`을 읽고, 모델이 넘긴 그룹 표로 검사한다. `source_*`는 표시로 보지 않는다. 검사 규칙은 컨버터 `plan.py check_rotation`과 같다. 부호는 K/32 uint32 비트 팩(비트 1 = −1)으로 돌려준다. 정식 규칙 문서는 [docs/ROTATED-WEIGHTS.md](../../docs/ROTATED-WEIGHTS.md)에 있다.
- **27B 그룹 표** `weights.cpp BuildRotationGroups`: 주 스택 층만 넣는다. 그래서 nextn 층의 `.rot`은 "role has no rotation path"로 거부되고, "일부 층만 회전" 규칙에도 걸리지 않는다. `output.weight`가 없으면(tied) `token_embd.weight`를 tied로 표시한다. 별칭 이름(wq 등)도 로더와 같은 순서로 찾는다.
- **FWHT** `src/core/hip/detail/hadamard.hpp`(hip_runtime만 의존): `RotateSegmentLanes<E>`(lane당 연속 E개), `RotateSegmentStrided<E>`(lane + L·j), `RotateRowSegments`(LDS 행, wave당 구간).
  - 산술 계약: 부호는 비트 XOR(정확)로 먼저 적용한다. butterfly는 길이 1, 2, …, 64 순서로, 쌍마다 add/sub 한 번이다. 마지막에 `kHadamard128Scale`(= float(1/√128))을 한 번 곱한다.
  - 레이아웃과 무관하게 비트 단위로 같다. CPU 참고(`tests/core/quant/rotation_reference.hpp`)도 같은 단계를 밟는다.
  - 부호를 XOR로 적용하므로 생성 산술의 곱셈이 첫 butterfly의 덧셈과 FMA로 합쳐질 수 없다.
- **Q8_1 tiled 레이아웃과 회전 블록 128:** 128 = Q8_1 블록 4개다.
  - norm 융합: 기존 LDS 행을 회전한 뒤 기존 양자화 루프(wave당 32 블록, `StoreQ8ActLane`)를 그대로 돈다.
  - SwiGLU 융합: wave가 구간 하나를 맡아 lane당 4개를 계산하고 회전한 뒤, wave별 LDS에 내려놓고 기존 32 블록 코드를 4번 돈다.
  - SSM 에필로그: head(128)를 LDS로 회전한다.
  - 어느 경우든 tiled 레이아웃과 sum sidecar 코드는 바뀌지 않는다.

## 커밋

| 커밋 | 내용 |
|---|---|
| `b6264364` feat(core): add the 128-point Hadamard input rotation device helpers | `src/core/hip/detail/hadamard.hpp` |
| `f3ac8dec` feat(core): read and check rotated GGUF markers against a model group table | `src/core/quant/rotation.*`, `tests/core/quant/*`, 루트 CMake(패치) |
| `52ac15bf` feat(qwen27b): bind rotated input groups and rotate activations on every HIP path | 27B 바인딩, 거부, 커널, 경로 연결, 27B 테스트, qwen27b CMake(패치) |
| `44e0f618` feat(qwen27b): rotate the DFlash2 draft head input for a rotated target head | 드래프터 |
| (이 커밋) docs | `docs/ROTATED-WEIGHTS.md`, 이 문서 |

후보 v3 이미지의 소스는 이 커밋들의 코드와 같다. `w3-v3.patch`에서 문서를 뺀 부분이 위 네 커밋에 해당한다.

## 바꾼 파일

| 영역 | 파일 |
|---|---|
| 공통 검사 | `src/core/quant/rotation.{hpp,cpp}`, 루트 `CMakeLists.txt`(gufo_core 소스, `quant_rotation_test` 등록) |
| FWHT 헤더 | `src/core/hip/detail/hadamard.hpp` |
| 27B 바인딩 | `src/models/qwen/state.hpp`(`QwenLayerRotation`, `head_in_rotation`), `weights.cpp`, `hip/model_loader.cpp`·`hip/executor.hpp`(device 업로드) |
| 거부 | `generator.cpp`(--cpu), `mtp_reference.{hpp,cpp}`, `hip/mtp/model.cpp` |
| 커널 | `hip/ops/hadamard.hpp`(새 선언), `hip/kernels/rotation.hip`(새, 단독), `norm.hip`, `prefill_norm.hip`, `prefill_quant_gemm.hip`, `prefill_fp16.hip`, `ssm_row_split.hip`, `hip/ops/{gemm,ssm}.hpp`, `hip/ops.hpp`, `src/models/qwen/CMakeLists.txt` |
| 경로 연결 | `hip/decode_step.cpp`, `hip/batched_decode.cpp`, `hip/prefill_chunk.cpp`, `hip/output_head.hpp`(새) |
| 드래프터 | `dflash_weights.hpp`, `hip/dflash/{model,executor,batch}.cpp` |
| 테스트 | `tests/core/quant/rotation_{test.cpp,reference.hpp,test_gguf.hpp}`, `tests/models/qwen27b/rotation_binding_test.cpp`(CPU), `tests/models/qwen27b/rotation_kernels_test.hip`(GPU), `tests/models/qwen27b/CMakeLists.txt` |
| 문서 | `docs/ROTATED-WEIGHTS.md`, 이 문서 |

## 검사 (CPU 단계 1, 이미지 `ai-tools/gufo-eval-deps:7c2c5a72-audio`)

`ai-tools/gufo-eval:7c2c5a728270`에는 `sndfile`이 없어 configure가 실패한다. 그래서 같은 7c2c5a72 기반에 audio 의존성을 더한 deps 이미지를 썼다(R10 브리프와 같은 이미지).

| 검사 | 결과 |
|---|---|
| `build/rq-cpu-test` `quant_rotation_test` | 통과. 대상: 비회전(그리고 `source_*`만 있는 파일), 수락과 비트 팩, 표시 KV 거부 10가지(키 없음 4, version 2, 문자열 version, kind, block_size 64, side, 짝 없음 2), 텐서 거부 10가지(그룹 밖, 고아 `.rot`, tied, 지원 안 하는 그룹, 혼합, 다른 부호, 일부 층, ±1 아님, 길이, F32 타입, K=96), FP32 참고 대 FP64 행렬곱(K 128–17408, 상대 1e-5), 노름 보존, H·H=I |
| `build/rq-cpu-test` `qwen27b_rotation_binding_test` | 통과. 합성 qwen35(4층 + nextn 1층)으로 그룹 순서, 층별 포인터와 부호, nextn 거부, 일부 층·혼합·tied 거부, `--cpu` 거부(head 포함/제외), MTP 거부를 확인 |
| `build/rq-gpu-test` `gufo_core`, `qwen27b_rotation_kernels_test` 빌드 | 경고·오류 없이 빌드됨(기존 경고만). 실행은 GPU 창이 필요하다 |
| 포맷 | (아래 커밋 절에 기록) |

## 측정 빌드 (ROCm 10.1, F23과 같은 방식)

| 쪽 | 소스 | nix 결과 | 이미지 (revision 라벨) |
|---|---|---|---|
| 후보 v3 | `git archive c94572b3` + `build/rq-release/w3-v3.patch`(sha256 `ff55997a5733bf717681d82032118278ad1dd26858a4585411857f9b4b9f6690`; 이 문서의 v3 이후 기록 전 상태를 포함) → `build/rq-release/src-v3` | `/nix/store/w4ww3qrqmhhnilaim6ypialfnm6r18ar-gufo-0-unstable-dirty` | `ai-tools/gufo-eval:rqw3-cand-rocm101` (`c94572b3…+w3-ff55997a`, id `dba40ef0`) |
| 후보 v2(폐기) | `git archive c94572b3` + `build/rq-release/w3-v2.patch`(W3 파일만, sha256 `7d8f53ffe9ff8bf5a16b201d3c8817c51ec62c3d93f4bf2bc49c1406acf5d68e`) → `build/rq-release/src-v2` | `/nix/store/na579jxnqsrh99lpsydxssbrz71nc4nj-gufo-0-unstable-dirty` | `ai-tools/gufo-eval:rqw3-cand-v2-rocm101` (`c94572b3…+w3-7d8f53ff`, id `578aebea`; v3의 FP32 반올림 장벽이 없음) |
| 후보 v1(폐기) | 같은 방식, `w3.patch` sha256 `d7616048…`(기존 `.hip` 5개가 통째로 재포맷되었고 SwiGLU FP16 명시 반올림이 없음) | `/nix/store/fcqjgim118vh4kzvr402qdi2dn7hnv9g-gufo-0-unstable-dirty` | `ai-tools/gufo-eval:rqw3-cand-v1-rocm101` (id `f550f259`, 예전 태그 `rqw3-cand-rocm101`) |
| 기준 | `git archive c94572b3` → `build/rq-release-base/src` | `/nix/store/c5nqyy4z36zpnz0b8wb84nkd5wz245ja-gufo-0-unstable-dirty` | `ai-tools/gufo-eval:rqw3-base-rocm101` (`c94572b3…`, id `0afd4cea`) |

- 툴체인: upstream flake `nix build path:/src#default`(nixos/nix, volume `gufo-nix`, `--cores 8`)이다. closure의 gufo 이외 경로는 F23 이미지 closure와 같다(ROCm 10.1, LLVM/Clang 24.0.0-rocm).
- 이미지는 F23처럼 `FROM ai-tools/gufo-eval:64849ca6ec7b` + `ADD closure.tar` + release 심볼릭 링크로 만들었다(`build/rq-release*/img/Dockerfile`).
- 기준 gpu-test(바이트 비교용 `qwen27b_target_test`): `build/rq-release-base/gpu-test`(ROCm 7.2.4, deps 이미지)
- 창 계획: `PLAN-RQ2A.json`(①), `PLAN-RQ2C.json`과 `PLAN-RQ2C-BASE.json`(④와 회전 비용)
  - note에 작업 트리 해시를 적는다. 값은 창 계획 note에 있다(이 문서도 해시 대상이라 여기에는 적지 않는다).

## ISA 확인: FP16 변환과 scale 곱의 합침 (2026-10-29)

디스어셈블 방법:

- gpu-test 오브젝트: `.hip_fatbin` → `clang-offload-bundler` → `llvm-objdump --mcpu=gfx1151`
- nix 바이너리: `bin/.gufo-wrapped`에서 code object를 꺼낸 뒤 Clang 24 `llvm-objdump`

| 커널 | 툴체인 | 장벽 전 (v2) | 장벽 후 (v3) |
|---|---|---|---|
| `SwiGLURotateFp16` | ROCm 7.2.4 gpu-test | `v_fma_mixlo_f16`/`v_fma_mixhi_f16 v, x, s0(=0x3db504f3), 0` 각 2개 | `v_mul_f32` 뒤 `v_cvt_f16_f32` 4개 |
| `SwiGLURotateFp16` | ROCm 10.1 / Clang 24 nix | `v_fma_mix{lo,hi}_f16` 각 2개 | `v_cvt_f16_f32` 4개 |
| `BatchedSSMPostNormGateFp16Kernel<true>` | 7.2.4 / 10.1 | `v_fma_mix{lo,hi}_f16` 각 2개 | `v_cvt_f16_f32` 4개 |
| `BatchedSSMPostNormGateFp16Kernel<false>`(비회전) | 7.2.4 | `v_cvt_f16_f32` 4개 | 같음 |

- 장벽 전에는 FWHT의 `*kHadamard128Scale`(0x3db504f3)이 FP16 변환과 하나의 mix FMA로 합쳐졌다. 정확한 곱을 FP16으로 한 번만 반올림하므로 FP32 동률에서 반올림 방향이 바뀌었다.
- a04의 불일치 패턴과 맞는다. 장벽 뒤에는 FP32 곱과 변환이 따로 나온다.

**비회전 코드 생성 확인**(기준 `c94572b3` gpu-test 대 v3 gpu-test, 고친 `.hip` 다섯 개의 모든 커널, `tmp`의 `isa/cmp.py`):

- 비교 전에 정규화한 것: 코드 배치에 따라 바뀌는 PC 상대 상수(`s_getpc` 다음 `s_add_u32`), hidden kernarg를 읽는 `s_load`의 offset, objdump의 `...` 패딩.
- 결과:
  - prefill_quant_gemm 212개, prefill_fp16 82개, ssm_row_split 10개(비회전 FP16 에필로그 포함), prefill_norm 3개, norm 2개 커널이 명령 단위로 같다.
  - 남은 차이는 `RMSNormKernel<>`(dim ≠ 5120 일반판) 하나다. hidden kernarg 주소 계산 `s_add_u32 s12, s0, 40 → 48` 한 줄뿐이고, explicit 인자 `signs`(8 B)가 늘어서 생긴 차이다. 산술은 같다.
  - 27B decode는 dim 5120의 `RMSNormKernel<5120>`을 쓰며, 이것은 같다.
- 회전 helper는 `Rotate` 인스턴스와 새 커널에서만 쓰이고, 비회전 커널 본문에는 들어가지 않는다.

## GPU 창에서 돌릴 것 (아직 안 함)

1. `qwen27b_rotation_kernels_test`(①⑤ 융합 비교). 내용은 아래와 같다.
   - helper 레이아웃(lanes 4/8, strided 4/8, LDS 행) 대 CPU 참고: 비트 비교
   - `LaunchRotateRowsFp32`(K 5120/6144/17408, 1/3/8행): 비트 비교
   - `LaunchRMSNormRotated`(5120 cached, 6144 일반) = `LaunchRMSNorm` + CPU 회전: 비트 비교
   - `LaunchBatchedRMSNormRotated`(1/8/70행) FP32·BF16: 비트 비교
   - 융합 RMSNorm Q8_1(residual 있음/없음) = 회전 FP32의 `…FromFp32` 양자화와 바이트 비교
   - SwiGLU Q8_1·FP16, SSM 에필로그 Q8_1: 같은 산술의 테스트 커널 + CPU 회전 + 기존 인코더와 바이트 비교
   - FP16 norm만 reduction tree가 달라서 1 ulp 허용
2. ④ 비회전 회귀: `qwen27b_target_test`, `qwen_dflash_gpu_test`와 main 대비 짝 타이밍.
3. 회전 실행 smoke와 ②③: W1·W2 산출물이 준비된 뒤.

## GPU 창 기록

| 창 | 결과 | 기록 |
|---|---|---|
| `rq2a-a01` units | FAILED. 실패 3개 중 2개는 테스트 쪽 문제였다. ① LDS 17408 float 행이 64 KiB를 넘어 테스트 커널 launch가 실패했는데, launch 오류를 확인하지 않았다. ② SwiGLU FP16 batch 5/33 불일치는 원인 미상. 나머지와 CPU 테스트는 통과 | `.../rq2a-a01/rq2a-units/WINDOW.json` |
| `rq2a-a02` units | FAILED. SwiGLU FP16 불일치는 batch 5에서 3/87040, batch 33에서 41/574464였다. 여러 행에 흩어져 있고 모두 1 FP16 ulp다. 그래서 레이아웃이 아니라 산술 차이로 판정했다. 운영 커널과 참고 커널 모두 반올림을 명시(`__fdiv_rn`/`__fadd_rn`/`__fmul_rn`)하도록 고쳤고, 판정은 정확 일치 그대로 두었다. 나머지는 모두 통과 | `.../rq2a-a02/rq2a-units/WINDOW.json` |

| `rq2a-a03` units | FAILED. 명시 반올림을 넣은 뒤에도 불일치 개수와 위치가 a02와 같았다. 그래서 sigmoid 산술은 원인이 아니다 | `.../rq2a-a03/rq2a-units/WINDOW.json` |
| `rq2a-a04` units(진단) | FAILED. 진단 결과: 참고 FP32를 device에서 회전하든 CPU에서 회전하든 같고, 호스트 변환과 device 변환도 같다. 운영 커널만 다르다(batch 5에서 3개, 33에서 41개). 불일치는 모두 **회전 FP32 값이 정확히 FP16 반올림 동률(tie)** 인 위치이고, 커널은 반대쪽으로 반올림했다. 원인: LLVM이 FWHT의 마지막 `*kHadamard128Scale`과 뒤따르는 FP16 변환을 한 명령(v_fma_mix 계열, 정확한 곱을 한 번만 반올림)으로 합쳤다(짐작이지만 동률 패턴과 맞는다). 수정: `hadamard.hpp`에서 scale 곱 뒤에 `asm volatile` 레지스터 장벽을 두어 FP32 반올림을 고정했다(이 파일 `RoundActivation`과 같은 방법). SwiGLU 명시 반올림은 원인과 관계없으므로 원래 식으로 되돌렸다 | `.../rq2a-a04/rq2a-units/WINDOW.json` |
| `rq2a-a05` units | **COMPLETE.** `qwen27b_rotation_kernels_test`의 비트·바이트 비교가 모두 통과했다(SwiGLU FP16 batch 1/5/33 포함, kernel과 device 연쇄의 차이 0). CPU 테스트 두 개도 통과. v3 이미지 `dba40ef0` | `.../rq2a-a05/rq2a-units/WINDOW.json` |
| `rq2a-a05` smoke | **COMPLETE.** 회전 UD-copy Q4로 `qwen27b_target_test`를 돌렸다. decode, prefill(128/257/2048, depth 8192의 1025), continuation, final output tile, verify, 동시 실행의 모든 일치 검사가 통과했다("Qwen27B target checks passed", 98 s). 회전 대 plain UD-copy Q4 기준 비교(같은 텐서 타입, 27행): top1 26/27, mean KL 0.0086, mean TV 0.040, mean NLL 차이 +0.0094. 같은 바이트의 다른 양자화이므로 이것이 ②는 아니다. ②는 BF16으로 따로 잰다 | `.../rq2a-a05/rq2a-smoke/WINDOW.json` |

경로의 `...`는 `/home/m2on/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023`이다.

## ④ 비회전 회귀 (창 `rq2c-a01`, `rq2c-a02`)

경로의 `…`는 `/home/m2on/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023`이다.

- **logits 비트 일치:** UD-Q4_K_M에서 `qwen27b_target_test --fingerprints-only`의 27줄(모든 case·row의 logits sha256)을 비교했다. 후보(v3 gpu-test, `…/rq2c-a01/rq2c-units`)와 기준(c94572b3 gpu-test, `…/rq2c-a02/rq2cb-units`)이 **바이트 단위로 같다**(`cmp`).
- 후보 `qwen27b_target_test` 전체(89 s)와 `qwen_dflash_gpu_test`(7 s)도 통과했다.
- `rq2c-a01/rq2cb-units`는 창 계획 오류(테스트 timeout 900 s = 창 900 s)로 시작 전에 거부됐다. timeout을 600 s로 고쳐 a02로 돌렸다.
- **속도 ABBA** (UD-Q4_K_M, release ROCm 10.1 nix, `-p 512,2048 -n 128 -d 0 -r 3`)
  - 순서: A1 `bench-1-cand`, B1 `bench-2-base`, B2 `bench-3-base`, A2 `bench-4-cand`(`…/rq2c-a02/`)
  - 값은 `gufo bench` 3회 평균 t/s다. ms는 단계 시간(토큰 수 ÷ t/s)이고, A·B 평균은 두 창의 평균이다.

| 모드 | 단계 | A1 | B1 | B2 | A2 | A ms | B ms | 차이 ms | 차이 % | A 폭 % | B 폭 % | 판정 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| AR | pp512 | 507.78 | 507.39 | 505.55 | 494.77 | 1021.6 | 1010.9 | +10.6 | +1.05 | 2.60 | 0.36 | 같음(5 % 안) |
| AR | pp2048 | 676.36 | 676.80 | 662.90 | 653.49 | 3081.0 | 3057.7 | +23.2 | +0.76 | 3.44 | 2.08 | 같음 |
| AR | tg128 | 13.06 | 13.06 | 13.05 | 13.03 | 9812.2 | 9804.7 | +7.5 | +0.08 | 0.23 | 0.08 | 같음 |
| DFlash2 | pp512 | 471.49 | 470.46 | 468.71 | 464.76 | 1093.8 | 1090.3 | +3.5 | +0.32 | 1.44 | 0.37 | 같음 |
| DFlash2 | pp2048 | 616.80 | 610.36 | 605.14 | 606.83 | 3347.6 | 3369.9 | −22.2 | −0.66 | 1.63 | 0.86 | 같음 |
| DFlash2 | tg128 | 45.27 | 45.18 | 45.23 | 45.13 | 2831.9 | 2831.5 | +0.3 | +0.01 | 0.31 | 0.11 | 같음 |

- 어느 단계도 5 %에 닿지 않는다. 비회전 경로의 회귀는 없다.
- prefill은 창 순서를 따라 점점 느려졌다(A1 > B1 > B2 > A2). 이것은 시간에 따른 drift로 보이고, A 폭(2.6–3.4 %)도 그래서 생겼다(짐작).
- `bench-1-cand`와 `bench-2-base`는 창이 끝난 직후 busy 4 %로 health 실패였다. 창 당시 기준은 3 %였다. 2026-10-29 사용자 지시로 기준을 4 %로 바꾼 뒤 유효로 처리했다. `bench-3-base`, `bench-4-cand`는 새 기준에서 COMPLETE다.

## 회전 비용 (창 `rq2c-a02/rotcost`, 후보 v3)

- 대상: plain과 rot UD-copy Q4(같은 텐서 타입, 회전 여부만 다름)
- 순서: plain, rot, rot, plain을 AR과 DFlash2 각각
- 설정: `-p 512,2048 -n 128 -r 3`. pp512는 Q8_1 prefill, pp2048은 half_prefill 경로다. decode는 기본 graph replay다.

| 모드 | 단계 | plain1 | rot1 | rot2 | plain2 | plain ms | rot ms | rot−plain ms | % | plain 폭 % | rot 폭 % |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| AR | pp512 | 499.46 | 485.97 | 481.35 | 482.92 | 1042.7 | 1058.6 | +16.0 | +1.53 | 3.37 | 0.96 |
| AR | pp2048 | 666.41 | 632.70 | 629.61 | 652.51 | 3105.9 | 3244.9 | +138.9 | +4.47 | 2.11 | 0.49 |
| AR | tg128 | 12.44 | 12.32 | 12.31 | 12.42 | 10297.7 | 10393.8 | +96.2 | +0.93 | 0.16 | 0.08 |
| DFlash2 | pp512 | 447.59 | 457.04 | 455.94 | 459.06 | 1129.6 | 1121.6 | −8.0 | −0.71 | 2.53 | 0.24 |
| DFlash2 | pp2048 | 605.19 | 587.53 | 585.24 | 608.57 | 3374.7 | 3492.6 | +117.9 | +3.49 | 0.56 | 0.39 |
| DFlash2 | tg128 | 44.73 | 41.15 | 41.42 | 45.89 | 2825.4 | 3100.4 | +275.0 | **+9.73** | 2.56 | 0.65 |

- **AR decode 회전 비용: 토큰당 +0.75 ms, +0.93 %.** 목표 1 % 아래지만 여유가 적다. 잡음 폭(0.08–0.16 %)보다 충분히 커서 측정된 차이다. 이 값은 graph replay에서 단독 회전 128 launch와 norm 융합을 합친 비용이다.
- **prefill:**
  - pp2048 half_prefill: +4.5 %(AR), +3.5 %(DFlash2). GEMM 에필로그 SwiGLU 융합을 끈 비용과, attention 출력·SSM 쪽 FP32 회전 패스를 더한 비용으로 본다(짐작).
  - pp512 Q8_1: AR +1.5 %. plain 폭이 3.4 %라서 잡음 안이고 결론 없음이다.
- **DFlash2 decode: −8.9 % 처리량(+9.7 % 시간).** AR 회전 비용(0.9 %)으로는 설명되지 않는다.
  - verify step의 회전 launch는 step마다 AR과 비슷한 수(약 0.75 ms)이고, 토큰당으로 나누면 더 작다.
  - 그래서 수락률이 내려갔을 가능성이 크다(짐작). `gufo bench`는 수락률을 출력하지 않아 이 창에서는 확인할 수 없다.
  - 가능한 원인은 두 가지다.
    - (a) 회전 Q4가 plain Q4보다 드래프터와 덜 맞는다. 기준 비교 KL 0.0086 수준의 차이다.
    - (b) 드래프터 head 경로의 회전 오류.
  - (b)는 ③에서 수락률을 직접 재서 가린다. 회전 파일로 `qwen_dflash_gpu_test`와 수락 통계를 보는 방식이다.
  - 2단계(decode 융합)에 들어가기 전에 이것부터 확인해야 한다.

## DFlash2 감속 진단 (창 `rq2d-a01`, 후보 v3, timing 판정용 아님)

- `rq2d-units`(COMPLETE): `qwen_dflash_gpu_test`가 회전 타깃으로 통과했다.
  - 회전 UD-copy Q4: 7.5 s
  - 회전 BF16: 47.7 s
- `rq2d-bench`(COMPLETE): DFlash2, `gufo bench -v -p 512 -n 128 -r 3`. 수락률은 bench가 출력한 `acceptance`(accepted/proposed)다.

| 순서 | 타깃 | pp512 t/s | tg128-dflash2 t/s | 수락률 |
|---|---|---:|---:|---:|
| 1 | BF16 | 78.70 | 13.88 | 0.533 |
| 2 | rot-BF16 | 65.45 | 7.50 | 0.354 |
| 3 | plain UD-copy Q4 | 468.65 | 46.23 | 0.714 |
| 4 | rot UD-copy Q4 | 466.86 | 41.60 | 0.627 |
| 5 | rot-BF16 | 64.94 | 7.49 | 0.354 |
| 6 | BF16 | 78.27 | 13.90 | 0.533 |

- **판정: (b) 쪽.**
  - 회전 BF16은 원본 BF16과 BF16 반올림만큼만 다르다. 그런데 수락률이 0.533에서 0.354로 떨어졌고, 반복해도 같다(결정적).
  - 따라서 회전 실행 경로 어딘가에 오류가 있다.
- 위치는 아직 모른다. 후보는 둘이다.
  - 드래프터 head 회전
  - 타깃 쪽(특히 BF16 가중치 경로)
- `qwen27b_target_test`의 smoke는 회전 경로들끼리 서로 같다는 것만 확인했다. 원본 대비 정확성은 확인하지 않았다.
- 회전 BF16의 pp512가 −17 %, tg가 −46 %인 것도 수락률만으로는 다 설명되지 않는다. 같이 조사한다.

### 진단 2 (창 `rq2d-a02`)

- **타깃 logits(`rq2d-trace`, `qwen27b_target_test rot-BF16 BF16`):**
  - 27행 기준: top1 27/27, mean KL 5.0e-6, mean TV 9.6e-4, NLL 차이 −0.00036.
  - row 0은 prefill 경로, row 1–8은 decode 경로다. Q8_0 대 BF16 수준(약 1e-3)보다 두 자릿수 작다.
  - **회전 타깃의 prefill·decode는 정확하다.** verify는 이 비교에 들어 있지 않다. 다만 회전 Q4에서는 target_test가 verify = decode를 확인했다.
- **드래프터(`qwen_dflash_gpu_test --trace`, BF16 대 rot-BF16, 같은 24토큰 prompt):**
  - 타깃 features: relL2 2.1e-3
  - 드래프터 층 텐서: relL2 1e-4–4e-3
  - `normalized`: 2.7e-3
  - 회전된 head를 거친 `logits`: relL2 7.1e-3, 최대 절대 차이 0.11
  - 7개 draft 위치의 argmax가 모두 같고, top-2 간격도 비슷하다.
  - `probabilities`/`proposed` 차이는 trace가 temperature 0.8 샘플링이라 생긴 것이다(후보 목록은 거의 같다).
  - **드래프터 head 회전은 맞다.**
- **AR(`rq2d-arbench`, `-p 512 -n 128 -r 1`):**
  - BF16: pp512 79.90, tg128 4.14 t/s
  - rot-BF16: pp512 66.22, tg128 3.86 t/s
  - decode가 토큰당 +17.5 ms(+7.2 %)로, Q4의 회전 비용(+0.75 ms)보다 20배 넘게 크다. prefill은 +1.3 s(+21 %)다.
  - 두 파일의 텐서 타입은 같다(BF16 506, F32 360, alignment 32).
  - 다른 점은 회전 파일이 단일 51 GB 파일이고(원본은 46.6+4.7 GB 두 조각), `.rot` 텐서 때문에 tensor offset이 바뀐 것이다.
- **판단(짐작):**
  - 수락률 차이(0.533 대 0.354)는 회전 오류가 아니다. 회전 BF16 타깃이 느려지면서 시간 기반 adaptive 길이 제어기가 다른 draft 길이를 고른 결과로 보인다. Q4의 0.714 대 0.627도 같은 영향이거나 양자화 차이일 수 있다.
  - 남은 문제는 회전 BF16 실행이 크게 느린 원인이다. 회전 연산(Q4에서 0.75 ms) 때문인지, 파일 배치·가중치 placement 때문인지 가려야 한다.

### 진단 3 (창 `rq2d-a03`, 대조 파일 `qwen-3.8-27b-plain-bf16-single-gguf`)

**AR(`rq2d3-ar`, `-p 512 -n 128 -r 2`)**

| 순서 | 파일 | pp512 t/s | tg128 t/s |
|---|---|---:|---:|
| 1 | BF16 (분할) | 78.94 | 4.11 |
| 2 | plain-single (회전 없음, 단일 파일) | 66.29 | 3.73 |
| 3 | rot-BF16 | 65.67 | 3.83 |
| 4 | rot-BF16 | 65.67 | 3.85 |
| 5 | plain-single | 66.22 | 3.82 |
| 6 | BF16 (분할) | 78.74 | 4.12 |

- 회전 없는 단일 파일도 회전 파일만큼(tg는 조금 더) 느리다. **BF16의 감속은 회전 비용이 아니라 파일 배치(텐서 offset 정렬) 효과다.**
- W1 측정에 따른 텐서 offset 정렬: 원본은 128 B 정렬 739/866, 회전은 96/866, 대조는 64 B 정렬 0개.
- 회전 대 plain-single의 차이는 잡음 안이다. 회전 BF16 경로의 추가 비용은 보이지 않는다.

**DFlash2 고정 정책(`rq2d3-fixed`, `--draft-policy fixed -v`)**

| 순서 | 파일 | pp512 t/s | tg t/s | 수락률 |
|---|---:|---:|---:|---:|
| 1 | BF16 | 78.09 | 13.91 | 0.432 |
| 2 | rot-BF16 | 65.42 | 7.98 | 0.311 |
| 3 | rot-BF16 | 65.45 | 8.11 | 0.311 |
| 4 | BF16 | 78.69 | 14.16 | 0.432 |

- 길이 제어기를 고정해도 수락률이 다르다. 그래서 adaptive 제어기 효과라는 가설은 이것만으로는 확정되지 않는다.
- 아직 plain-single의 DFlash2 수락률이 없다.
  - plain-single도 0.311이면, 원인은 파일 배치(속도)에 따라 바뀌는 다른 시간 의존 동작이다.
  - plain-single이 0.432면, 회전 경로(verify 등)에 문제가 있다.
- 회전 BF16의 verify(배치) 경로는 아직 원본과 직접 비교하지 않았다.

**프로파일(`rq2d3-prof`): 실패.** 이미지의 `/opt/rocm/bin/rocprofv3`가 nix bash로 실행되면서 `libsqlite3.so.0`을 찾지 못했다(rc 127, 측정 없음).

### 진단 4 (창 `rq2d-a04`)

**DFlash2 고정 정책 수락률(`rq2d4-fixed`)**

| 순서 | 파일 | tg t/s | 수락률 | 생성 128토큰 sha256 |
|---|---|---:|---:|---|
| 1 | plain-single | 10.39 | 0.432 | `6971f2de…` |
| 2 | BF16 (분할) | 13.64 | 0.432 | `6971f2de…` |
| 3 | BF16 (분할) | 14.14 | 0.432 | `6971f2de…` |
| 4 | plain-single | 10.27 | 0.432 | `6971f2de…` |
| (a03) | rot-BF16 | 7.98–8.11 | 0.311 | `3baef7d3…` |

**verify·prefill logits 직접 비교(W2 수집기, corpus a01 window 1, 256 위치, BF16이 기준)**

| 경로 | KL | top1 | NLL 차이 |
|---|---:|---:|---:|
| prefill | 2.3e-5 [1.7e-5, 3.1e-5] | 1.000 | +0.00057 |
| verify | 2.3e-5 [1.7e-5, 3.1e-5] | 1.000 | +0.00057 |

- verify KL은 prefill과 같다. **회전 verify 경로는 정확하다.**
- 코드로도 확인했다. verify·batched 경로가 드래프터에 넘기는 features는 잔차 hidden(`scratch.decode.hidden`)에서 복사하고, 회전 커널은 이 버퍼에 쓰지 않는다.

**결론: 회전 경로 버그가 아니다.**

- 대조 파일 plain-single은 회전 없이 배치만 다르다. 생성한 128토큰이 BF16(분할)과 비트 단위로 같고, 수락률도 0.432로 같다.
- 회전 BF16은 logits가 KL 2e-5 정도 다르다. 그 결과 greedy 생성 궤적이 어느 동률 근처에서 갈렸다(출력 sha256이 다르다).
- `gufo bench`의 수락률은 합성 prompt 하나에서 나온 128토큰 궤적 하나의 값이다. 궤적이 바뀌면 수락률도 크게 바뀐다(0.432 대 0.311).
- 앞서 본 회전 Q4의 차이(0.714 대 0.627)도 같은 단일 궤적 효과로 본다.
- ③(수락률이 같은지)은 bench의 단일 궤적으로 판정할 수 없다. 많은 prompt나 위치에서 수락률을 모아 신뢰구간으로 판정해야 한다.

**BF16 단일 파일 감속(파일 배치)의 프로파일**

- 방법: ROCm 7.2.4 gpu-test `gufo bench -p 64 -n 16`의 rocprofv3 kernel trace. 첫 패스까지 들어간 전체 합계다.
- decode GEMV 커널 평균(µs):

| 커널 | BF16 분할 | plain-single | rot-BF16 |
|---|---:|---:|---:|
| Wave32FusedSwiGLUGEMVKernel_2Rows | 1746 | 2010 | 1931 |
| FastGEMVBlockKernel | 750 | 779 | 785 |
| Wave32GEMVKernel_1Row | 600 | 620 | 622 |
| Wave32FusedSSMInputProjectionsKernel_1Row | 735 | 745 | 748 |

- 단일 파일 두 개(plain, rot)는 서로 비슷하고, 분할 원본보다 3–15 % 느리다. 회전 커널(단독 회전과 회전 norm)은 run 전체에서 약 13 ms다.
- prefill의 `BatchedExactBf16GEMMFp32VecKernel` 합계는 첫 패스(page fault) 영향이 커서 파일별 비교에 쓰지 않았다.

### 텐서 offset 정렬 분포 (헤더만 읽음, 2026-10-29)

절대 offset = 파일의 data 시작 + 텐서 offset이다. 큰 텐서는 1 MiB를 넘는 410개다.

| 파일 | data 시작 | 전체 %128 = 0 | 전체 %256 = 0 | 큰 텐서 %128 = 0 | 큰 텐서 %256 = 0 |
|---|---|---:|---:|---:|---:|
| UD-Q4_K_M (운영) | 10996640 | 0/866 | 0/866 | 0/410 | 0/410 |
| UD-Q4_K_XL | 10996640 | 0/866 | 0/866 | 0/410 | 0/410 |
| rot UD-copy Q4 (llama-quantize) | 11025600 | 192/866 | 96/866 | 0/410 | 0/410 |
| plain UD-copy Q4 (llama-quantize) | 10996832 | 0/866 | 0/866 | 0/410 | 0/410 |
| BF16 분할 (unsloth) | 10994688 / 1984 | 739/866 | 362/866 | 391/410 | 191/410 |
| rot-BF16 (rotquant writer) | 11025600 | 96/866 | 48/866 | 0/410 | 0/410 |
| plain-BF16-single (rotquant writer) | 10996832 | 0/866 | 0/866 | 0/410 | 0/410 |

- 모든 파일의 `general.alignment`는 32다. 데이터 구간 안의 상대 offset은 큰 텐서 기준 410/410이 128 B 정렬이다.
- 그래서 절대 정렬은 data 시작 위치(헤더 길이)가 정한다. 헤더 뒤의 32 B 패딩만으로는 128/256 B 정렬이 보장되지 않는다.
- **운영 UD-Q4_K_M/XL도 큰 텐서가 하나도 128 B 정렬이 아니다**(data 시작 ≡ 32 mod 128).
- `llama-quantize` 출력도 같다. 정렬은 헤더 길이에 따라 우연히 정해진다.
- 정렬을 맞추면 운영 양자화 파일의 decode도 빨라질 수 있다(짐작). BF16에서는 단일 파일 GEMV가 3–15 % 느렸다. 양자화 GEMV에서 얼마나 효과가 있는지는 재 봐야 한다.

## 로더 재정렬 설계 메모 (2026-10-29, 코드는 쓰지 않음)

사용자 결정은 "컨버터 정렬 + 로더 조사"다. A/B 창 초안은 `PLAN-RQ2F.json`(원본 UD-Q4_K_M 대 W1의 align256 repack, 같은 v3 바이너리, ABBA, 5 %·10 ms, rocprofv3)이다.

### 지금 로드 경로 (Qwen27B, `src/models/qwen/hip/model_loader.cpp`)

1. `GgufReader`의 `GetMappedRegions()`는 **파일 전체**(offset 0부터)를 region 하나로 준다. 분할 GGUF는 조각마다 region이 하나다. 텐서 포인터는 이 매핑 안을 가리킨다.
2. `CreateWeightRegions`는 region마다 다음을 한다.
   - `MapRegisteredRegion`:
     - `platform::CloneMappedWeightsToAnonymousPages`로 region 전체를 페이지 정렬된 익명 메모리(MADV_HUGEPAGE)에 복사한다. 병렬 16 MiB 청크이고, 원본 PTE는 `MADV_DONTNEED`로 버린다.
     - 그다음 `hipHostRegister(Mapped|ReadOnly)`와 `hipHostGetDevicePointer`를 한다.
   - 실패하면 `AllocateDeviceRegion`(hipMalloc)으로 가고, `CopyMappedRegion`이나 `WeightUpload`로 region 전체를 offset 0부터 올린다.
3. `RemapTensor`는 텐서 포인터를 `device_data + (tensor − region.host_data)`로 바꾼다.

**따라서 device 주소의 정렬 = 파일 안 절대 offset의 정렬이다.**

- 운영 UD-Q4_K_M/XL은 data 구간이 ≡ 32 (mod 128)에서 시작한다. 그래서 큰 텐서 410개 중 128 B 정렬이 0개다.
- 상대 offset은 410/410이 128의 배수이고, 210/410이 256의 배수다.

### 후보

**A. region 이동(권장안)**

- region마다 `delta = (−data_offset) mod 256`(최대 255 B)을 정한다.
  - `size + delta`를 할당한다(익명 mmap 또는 hipMalloc).
  - 파일 바이트를 `base + delta`에 복사한다.
  - `device_data = base_device + delta`로 둔다. `RemapTensor`는 그대로 쓴다(device_data만 이동).
  - `hipHostRegister`와 해제는 `base`와 `size + delta`로 한다.
- 비용:
  - 메모리는 조각마다 ≤ 255 B 늘어난다.
  - 복사는 지금도 하는 clone 복사 한 번 그대로다(추가 복사 없음). 로드 시간도 같다.
  - huge page, 병렬 clone, 등록 방식도 바뀌지 않는다.
- 효과:
  - 상대 offset이 128의 배수인 텐서(조사한 모든 파일의 큰 텐서 전부)가 128 B 정렬이 된다.
  - 256 B 정렬은 상대 offset이 256의 배수인 텐서(약 절반)만 된다. 256을 보장하려면 W1이 컨버터를 alignment 256으로 바꾼 파일이 필요하다.
- 필요한 변경:
  - `core::GgufMappedRegion`에 data 구간 시작 offset 필드를 더한다. 공유 파일 `src/core/gguf_reader.*`이고, 필드를 더하기만 한다.
  - `model_loader.cpp`의 세 경로(registered clone, device copy, WeightUpload)에 delta를 넣는다.
  - `GetResidentBytes`에 delta를 반영한다.

**B. 텐서별 재배치**

- 텐서마다 정렬된 위치로 복사하고, 텐서별 주소 표를 둔다. Flash-Next uploader와 같은 방식이다.
- 정렬은 완전히 보장된다(256 이상도 가능).
- 단점:
  - region이 연속이라는 가정에 기대는 코드(DFlash2 드래프터가 타깃 `token_embd`/`output`을 참조하는 것, `RemapTensor`, 진단·스냅샷 경로)를 고쳐야 한다.
  - 패딩만큼 메모리가 늘어난다(텐서 866개 × ≤ 255 B, 무시할 만하다).
  - 구현과 검증 범위가 A보다 훨씬 크다.

**C. 커널이 비정렬에 덜 민감하게 만들기**

- 행 시작 정렬 prologue 등을 넣는다. 형식마다 GEMV/GEMM 커널을 고쳐야 해서 범위가 가장 크다.

### 규칙과 다른 모델

- CLAUDE.md(AGENTS.md)에는 weight placement를 따로 정한 규칙이 없다. 관련 지침은 optimize-kernel 스킬의 UMA placement 항목이다.
  - 익명 huge page가 Q4 AR을 약 2 % 높였다.
  - 병렬 bounded copy로 warm readiness를 1초 안에 유지했다.
  - 매핑된 가중치와 복사된 가중치의 비용이 다르다.
- A는 이 성질을 모두 그대로 둔다. Windows는 `CloneMappedWeightsToAnonymousPages`가 없어 device-resident 경로를 쓰는데, 같은 delta로 처리한다. `pages_windows.cpp`는 바꾸지 않는다.
- 영향 범위:
  - Flash-Next(`kernels/rocm/device_model.cpp`)와 DeepSeek V4 Flash는 `WeightUpload`로 **텐서마다** 자기 할당에 복사한다. 그래서 이 문제의 영향을 받지 않고 변경도 필요 없다(할당 정렬은 확인이 필요하다).
  - DFlash2 드래프터와 MTP 전용 가중치도 자기 할당에 pack한다. 다만 드래프터가 쓰는 타깃 `output`/`token_embd`는 A의 이득을 함께 받는다.
  - `gguf_identity.cpp`는 region을 읽기만 하므로, 필드가 더해져도 영향이 없다.
- 판단 순서:
  1. `PLAN-RQ2F` A/B로 align256 파일의 이득(decode GEMV, AR, DFlash2)을 잰다.
  2. 이득이 있으면 A를 구현한다. A는 컨버터가 고친 파일과 이미 배포된 파일(운영 UD 포함)에 모두 효과가 있다.
  3. A의 정확성은 바이트 비교로 확인한다. 정렬만 바뀌므로 logits fingerprint가 같아야 한다.

## 알려진 제한과 최적화 후보

- decode 단독 회전은 토큰당 128 launch다(attn_out 16, ssm_out 48, ffn_down 64). 1 %를 넘으면 2단계로 간다(ssm_out을 decode 에필로그에 융합 등).
- 회전 층의 prefill은 GEMM 에필로그 SwiGLU 융합 두 개를 쓰지 않는다. half_prefill 경로 포함, pp2048에서 차이를 잰다.
- 비 row-split SSM, attention 출력, 비-Q8 SwiGLU의 prefill 회전은 FP32 한 패스를 더한다.
- `ssm_out_in` 회전의 융합 에필로그는 val_dim 128을 요구한다(row-split 경로는 원래 128만 지원).
