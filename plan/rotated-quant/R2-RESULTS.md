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

## 알려진 제한과 최적화 후보

- decode 단독 회전은 토큰당 128 launch다(attn_out 16, ssm_out 48, ffn_down 64). 1 %를 넘으면 2단계로 간다(ssm_out을 decode 에필로그에 융합 등).
- 회전 층의 prefill은 GEMM 에필로그 SwiGLU 융합 두 개를 쓰지 않는다. half_prefill 경로 포함, pp2048에서 차이를 잰다.
- 비 row-split SSM, attention 출력, 비-Q8 SwiGLU의 prefill 회전은 FP32 한 패스를 더한다.
- `ssm_out_in` 회전의 융합 에필로그는 val_dim 128을 요구한다(row-split 경로는 원래 128만 지원).
