# 회전 가중치 양자화 (RQ, Qwen3.8 27B, 2026-10-28)

사용자 지시(2026-10-28): "27B를 완전히 구축, 실험, 확장 순으로", "1) 컨버터 만들기 - 로더가 아닌 별도 컨버터, 다양한 옵션 필요 2) 해당 모델 로드 후 테스트 돌아갈 수 있도록", "품질과 속도 개선 등은 확인하고 그거로 확장하자". 구현과 실험은 27B만 한다. Flash-Next는 지금 단계에서 다루지 않지만(사용자: "완전 논외"), 나중에 확장할 것이므로 모델에 의존하지 않는 부분은 공통 위치에 둔다(사용자 2026-10-28: "우리 flash-next 확장할 꺼니까 공통으로 빼는 게 좋지 않나?"). 경계는 아래 "공통과 모델별 경계" 절에 있다.

이 문서는 설계와 단계만 담는다. 아직 구현과 GPU 실행은 하지 않았다.

**효율 목표(사용자, 2026-10-28):** 양자화로 메모리와 대역폭을 아껴 최고 효율의 엔진을 만든다. 그래서 가중치 바이트를 줄이는 것만으로 끝내지 않고, 바이트 절감이 실제 속도로 이어지는지(R0b), 여러 설정을 같은 기준으로 비교하는지(R4 효율 점수판), 실제 작업 품질이 유지되는지(D4)를 함께 본다.

## 배경 (조사 요약)

- halogen-flash-server(비공개 엔진)는 자체 HGN 체크포인트로 Flash-Next decode에서 gufo를 앞선다. 그 체크포인트는 128점 Hadamard 회전 뒤에 양자화한 가중치(`i4r`: 회전 + 4비트 정수, `ht`: 회전 + trellis 코드)다. 포맷은 클린룸 스펙 `github.com/jtsylve/hgn-spec`(`STORAGE.md`의 "The 128-point rotation")에 정리돼 있다.
- gufo #474: decode 격차는 주로 dense 트렁크의 비트 수(8비트 대 4비트)에서 나온다. 4비트로 내릴 때 `ssm_out`이 가장 민감했다. 회전 없이 재양자화한 전문가 가중치는 unsloth UD를 이기지 못했다.
- 우리 KVarN(`tests/models/qwen27b/kvarn_fake_quant.hpp`, `src/models/qwen/hip/kernels/attention_kvarn.hip`)도 같은 계열의 회전(Sylvester Hadamard)을 이미 쓰고 있다.
- 외부 비교 대상: `ISTA-DASLab/Qwen3.8-27B-GSQ-RCO-IQ3_S`(약 3.5 bpw 혼합, 11.8 GB, `storage/models/llm/qwen3_8-27b-gsq-rco-iq3_s-d562806-a01`).

## 방식

입력 쪽에만 회전을 적용한다. `D`는 ±1 부호 대각 행렬이고, `H`는 128점 정규화 Sylvester Hadamard를 블록 대각으로 놓은 행렬이다. 회전 블록은 `K / 128`개다.

```text
y = W·x = (W·D·H)·(H·D·x)          # H·H = I, D·D = I
오프라인: W' = W·D·H  → 기존 GGUF 타입으로 양자화
실행 시:  x' = H·D·x  → 기존 matmul 커널에 그대로 넣는다
```

- **행렬 출력은 바뀌지 않는다**(정확한 산술 기준). 따라서 residual stream, KV, DeltaNet 상태, DFlash2가 받는 타깃 hidden이 모두 그대로다. 회전은 matmul 안에 갇힌 국소 변환이다.
- **예외: 가중치를 공유하는 다른 소비자.** DFlash2 드래프터는 타깃의 임베딩과 `output` 헤드를 그대로 쓴다(`src/models/qwen/dflash_weights.cpp:299`). `output`을 회전하면 드래프터의 헤드 경로에서도 입력을 회전해야 한다. 빠뜨리면 품질 테스트는 통과하고 드래프트 수락률만 떨어진다. 임베딩은 회전하지 않으므로 영향이 없다.
- **새 matmul 커널이 필요 없다.** 바뀌는 곳은 활성화를 만드는 커널뿐이다. 27B는 활성화를 matmul 전에 Q8_1로 양자화한다(`LaunchBatchedFusedRMSNormQuantizeQ8_1`, `ops/gemm.hpp:238`). 회전은 이 양자화 앞, 같은 패스 안에 넣는다. 회전은 활성화 이상치도 펴 주므로 Q8_1 오차도 줄어들 수 있다(측정으로 확인).
- **입력을 공유하는 행렬은 같은 `D`를 쓴다.** 그래야 입력을 한 번만 회전한다. 회전 여부는 텐서 단위가 아니라 **입력 그룹 단위**로 정한다.

### GGUF 블록 구조와 회전 블록

- **2단 구조:** K-quant(Q3_K, Q4_K, Q5_K, Q6_K)와 IQ4_XS는 256개 값의 슈퍼블록 안에 서브블록 스케일을 둔다. 서브블록 크기는 Q4_K, Q5_K, IQ4_XS가 32개(8개 서브블록), Q3_K와 Q6_K가 16개(16개 서브블록)다.
- **1단 구조:** Q8_0, Q4_0, IQ4_NL은 32개 블록 하나에 스케일 하나다.
- **경계가 맞물린다.** 회전 블록(128)과 양자화 블록(16/32/256)은 모두 같은 입력 방향(K)을 따라 행 안에서 묶인다. 16과 32는 128을 나누고, 슈퍼블록 하나(256)는 회전 블록 두 개를 정확히 덮는다. 그래서 양자화 블록이 회전 블록 경계에 걸치는 일이 없다.
- 27B의 입력 차원 5120(= 20 × 256), 6144(= 24 × 256), 17408(= 68 × 256)은 모두 256의 배수다.
- **순서:** 행마다 128씩 회전한 다음, 기존 포맷 그대로 양자화한다. 포맷 자체는 바뀌지 않는다.

### 27B 입력 그룹 (UD-Q4_K_XL 헤더 기준, hidden 5120)

| 그룹 | 입력을 만드는 곳 | K | 소비 텐서 |
|---|---|---:|---|
| `attn_in` | `attn_norm` 뒤 | 5120 | DeltaNet 층: `attn_qkv`, `attn_gate`, `ssm_alpha`, `ssm_beta` / full-attention 층: `attn_q`, `attn_k`, `attn_v` |
| `ssm_out_in` | SSM per-head norm + SiLU gate 뒤 | 6144 | `ssm_out` |
| `attn_out_in` | attention 출력 + gate 뒤 | 6144 | `attn_output` |
| `ffn_in` | `post_attention_norm` 뒤 | 5120 | `ffn_gate`, `ffn_up` |
| `ffn_down_in` | SwiGLU 뒤 | 17408 | `ffn_down` |
| `head_in` | `output_norm` 뒤 | 5120 | `output` |

K는 모두 128의 배수다. `token_embd`는 행 조회라 회전 대상이 아니다. `ssm_alpha`와 `ssm_beta`처럼 작은 텐서도 그룹에 속하면 같이 회전한다(오프라인이라 비용이 없다).

## GGUF 표시 규칙 (v1)

- 전역 KV:
  - `gufo.rotation.version = 1`
  - `gufo.rotation.kind = "hadamard"`
  - `gufo.rotation.block_size = 128`
  - `gufo.rotation.side = "input"`
- 짝 텐서: 회전된 가중치 `<name>`마다 `<name>.rot`(`[K]`, 값 ±1)를 둔다. 짝 텐서가 있으면 회전된 텐서다. 부호는 시드로 재생성하지 않고 파일에 저장한다.
  - 저장 타입: gufo `GgmlType`(`src/core/gguf_reader.hpp:23`)에는 I8이 없다(F32, F16, I32는 있다). R1에서 둘 중 하나로 정한다: reader에 I8을 추가하거나, ±1을 F16으로 저장한다(텐서당 수 KB라 크기 차이는 무시할 만하다).
- `general.architecture`(`qwen35`)와 기존 하이퍼파라미터 키는 그대로 둔다. 짝 텐서 때문에 텐서 수가 맞지 않아 llama.cpp는 로드를 거부한다. R1에서 실제로 거부하는지 확인한다.
- 로더 거부 조건(자동 fallback 없음, 이름을 들어 거절):
  - 모르는 `version`, 지원하지 않는 `block_size`/`kind`/`side`
  - 같은 입력 그룹 안에서 회전 여부나 부호 벡터가 다름
  - 회전 경로가 아직 없는 역할의 텐서에 `.rot`이 있음
  - `.rot`은 있는데 전역 KV가 없음, 또는 그 반대
  - 임베딩과 출력 헤드가 묶인(tied) 모델에서 `output`(`head_in`)이 회전됨. 회전하면 임베딩 조회도 바뀌기 때문이다. 27B는 묶여 있지 않지만 공통 규칙으로 둔다
- 회전 GGUF인지는 로더가 파일에서 판단한다. CLI 옵션은 추가하지 않는다.

## 공통과 모델별 경계

`CLAUDE.md`의 "모델 코드·도구는 모델 옆에" 규칙에 따라, 모델마다 다른 것(입력 그룹 표, 바인딩, 회전을 융합할 커널 위치)은 모델 쪽에 두고, 모델과 무관한 것만 공통으로 뺀다.

| 구성 요소 | 공통 위치 | 모델별 위치 (27B / 나중의 Flash-Next) |
|---|---|---|
| 컨버터 본체: 회전 수학, 스트리밍 rotate, 레시피 엔진, `llama-quantize` 호출, `--dry-run` 표 | `tools/rotquant/` (Python, 기존 `tools/gufo/gguf.py`와 `tools/gufo/safetensors.py`를 재사용) | 입력 그룹 정의 파일: `tools/qwen27b/rotquant_groups.py` / 나중에 `tools/qwen-flash/rotquant_groups.py` |
| GGUF 표시 규칙 v1 (KV, `.rot` 짝 텐서) | 이 문서의 "GGUF 표시 규칙" 절. 정식 문서는 R1에서 `docs/`에 둔다 | — |
| 메타데이터 읽기와 공통 거부 검사 (버전, kind, block_size, 짝 텐서와 KV의 짝 맞음, 그룹 안 부호 일치) | `src/core/quant/rotation.{hpp,cpp}` (`src/core/gguf_reader.hpp` 위). 모델이 그룹 표를 넘기면 검사한다 | 그룹 표와 텐서 바인딩: `src/models/qwen/weights.cpp` / `src/models/qwen38_flash_next/weights.cpp` |
| FWHT 디바이스 함수 (128점, 부호 곱, 정규화) | `src/core/hip/detail/hadamard.hpp` (헤더 전용 device 함수). Flash-Next는 별도 커널 트리(`kernels/rocm/`)를 쓰므로 공유 단위는 헤더가 맞다 | 융합 위치: 27B `kernels/norm.hip` 등 / Flash-Next `kernels/rocm/kernels.hip` |
| 회전 기저 imatrix 누적과 파일 쓰기 | 누적 버퍼와 llama.cpp imatrix 형식 쓰기: `src/core/quant/` | forward 중 어느 활성화를 넘길지 고르는 훅: 모델 쪽 |
| CPU 참고 구현과 테스트 | `tests/core/quant/` (회전 수학, 메타데이터, 거부 조건) | `tests/models/qwen27b/` (그룹 표, logits 동일성) |

### 공통 설계가 Flash-Next에서 막히지 않도록 미리 확인한 것 (구현은 하지 않음)

- **3차원 fused 전문가 텐서**(`[512, N, K]`): 회전은 K 방향이고 모든 전문가가 같은 토큰 입력을 받으므로(gate/up), 전문가 전체에 `.rot` 하나(`[K]`)를 공유한다. 규칙 v1은 텐서 차원 수를 제한하지 않는다.
- **`down_exps`의 K = 640**: 640 = 5 × 128이라 회전은 가능하다. 하지만 256의 배수가 아니어서 2단 K-quant(Q4_K 등)는 쓸 수 없고, 1단 포맷(Q8_0, IQ4_NL, Q4_0, Q5_x)만 가능하다(#474에서도 확인). E6 결과가 이 텐서의 타입 선택에 바로 쓰인다.
- **hyper-connection, gated residual, n-gram 테이블, MTP 헤드**는 27B에 없다. 이것들의 입력 그룹 정의는 Flash-Next 단계에서 따로 만든다.

## 단계 (각 단계 끝에 사용자 확인)

| 단계 | 내용 | 통과 기준 |
|---|---|---|
| R0 기준선 | BF16 GGUF를 준비한다(아래 D2: unsloth BF16 GGUF를 받는다). 품질 하네스를 만든다. 기존 `docs/models/qwen3.8-27b/QUALITY.md`의 27행 비교는 Q4_K_XL과 회전 Q4를 구분하기에 작으므로, 약 32K 토큰(영문·한글 산문, 코드, 대화)에 대해 BF16 대비 paired NLL, top-k 절단 KL, top-1 일치율을 낸다. BF16 logits는 전체를 저장하지 않고 top-k와 logsumexp만 저장한다. Q8_0, Q6_K, UD-Q4_K_XL, Q4_K_M, GSQ-RCO-IQ3_S를 측정한다. 같은 날 decode/prefill(pp2048/tg128, d0와 d32K)과 DFlash2 수락률을 측정한다. 평가 코퍼스는 R3 imatrix 보정 코퍼스와 겹치지 않게 만들고 해시를 기록한다 | 모델별 품질·속도 표. 품질 지표는 토큰 단위 paired bootstrap 95% 신뢰구간으로 낸다(GPU 실행은 결정적이라 같은 파일을 두 번 돌리면 차이가 0이므로 반복 실행으로는 잡음을 잴 수 없다). 신뢰구간 폭이 Q4_K_XL과 Q4_K_M의 차이보다 충분히 좁지 않으면 코퍼스를 늘린다. 속도는 같은 파일을 번갈아 돌려 잡음 폭을 기록한다 |
| R0b 타입별 실효 대역폭 (**완료 2026-10-28**, [결과](R0B-RESULTS.md)) | 아래 "R0b" 절. 무작위 가중치로 충분하므로 모델 파일과 imatrix가 필요 없다. 디스크 확보 전에 R0, R1과 병행할 수 있다 | 타입별 bpw, 실효 GB/s, 토큰당 시간 표 |
| R1 컨버터 (R1a 합성 텐서 부분 **완료 2026-10-28**, [결과](R1A-RESULTS.md). R1b는 디스크 확보 뒤) | 아래 "R1 컨버터" 절 | 단위 테스트 통과. 회전만 한 BF16 출력에 대해 CPU 참고(NumPy)로 `W'·(H·D·x) ≈ W·x`를 확인. llama.cpp 로드 거부 확인 |
| R2 로더와 실행 | 아래 "R2 로더와 실행" 절 | ① FWHT 커널 단위 테스트(CPU 참고와 비교) ② **회전 BF16 GGUF가 원본 BF16에 충분히 가까움.** 회전 BF16은 `W·D·H`를 BF16으로 반올림한 값이라 비트 단위로 같을 수는 없다. 기준: KL(회전 BF16 ‖ BF16)이 KL(Q8_0 ‖ BF16)의 1/10 이하, top-1 일치율은 Q8_0 이상. decode, prefill, DFlash2 verify 각각에서 확인한다 ③ 같은 파일(회전 Q4)에서 DFlash2 수락률이 회전 전 파일과 신뢰구간 안에서 같음(드래프터 헤드 경로 확인) ④ 비회전 GGUF는 비트 단위로 이전과 같음 ⑤ 거부 조건 테스트 |
| R3 imatrix | 회전 기저에서 imatrix를 수집한다. gufo forward 중 각 입력 그룹의 `x' = H·D·x`의 열별 제곱합을 쌓아 llama.cpp imatrix 형식으로 쓴다. 회전 가중치가 필요 없으므로 BF16이나 Q8_0 비회전 모델에 같은 `D`를 적용해 수집할 수 있다. 원래 기저의 imatrix 대각값은 회전 후에 쓸 수 없다(공분산 전체가 필요하다). 같은 수집기로 원래 기저 imatrix도 함께 낸다(E1의 비회전 대조군용). 보정 코퍼스는 R0 평가 코퍼스와 겹치지 않게 하고 해시를 기록한다 | 수집 결과를 CPU 참고와 비교. 같은 레시피에서 imatrix 있음/없음의 R0 품질 차이 |
| R4 실험 | 아래 "R4 실험" 절 | 결과표. 채택 기준은 R0 결과를 보고 사용자와 정한다 |
| R5 채택·확장 | 이득이 확인된 설정을 기본 배포 파일로 정한다. 그다음 다른 모델로의 확장을 계획한다(아래 "Flash-Next 확장 참고") | 별도 문서 |

### R0b 타입별 실효 대역폭

- **목적:** 비트를 내려도 커널이 대역폭을 다 쓰지 못하면 속도로 이어지지 않는다. 특히 IQ 계열(IQ3_S, IQ2_*)은 코드북 조회 디코드가 무거워 gfx1151에서 연산에 묶일 수 있다. 어떤 타입이 "싸게 읽히는지"를 R4 레시피 선택 전에 안다.
- **대상 타입:** Q8_0, Q6_K, Q5_K, Q4_K, IQ4_XS, IQ4_NL, Q3_K, IQ3_S, 그리고 E7 후보 Q2_K, IQ2_XXS, IQ2_XS, IQ2_S.
- **크기:** 27B의 실제 행렬 크기(K = 5120, 6144, 17408와 각 그룹의 N). 세 가지 배치를 잰다.
  - AR: 질의 1개(GEMV)
  - DFlash2 verify: 실제 verify 배치 크기
  - prefill: 큰 배치 GEMM. IQ 계열은 디코드 연산 때문에 prefill에서도 느려질 수 있다
- **측정:** 실효 GB/s와 토큰당 시간을 재고 `tools/bench/gfx1151_peak.hip`의 대역폭 상한과 비교한다. 빌드는 `tools/bench/build.sh`를 쓴다.
- **어느 커널로 재는가:**
  - 27B가 이미 지원하는 타입은 **운영 경로의 커널**(`src/models/qwen/hip/kernels/gemv_quant.hip`, `small_batch_*`, `prefill_quant_gemm.hip` 등)로 잰다. 독립 커널로 재면 R4의 실제 속도와 맞지 않는다.
  - 27B가 지원하지 않는 2비트대 타입(Q2_K, IQ2_*)은 독립 벤치 커널로 잰다. Flash-Next에 vendored된 mmvq(`src/models/qwen38_flash_next/kernels/rocm/mmq/mmvq.hip.cpp`)는 Q5_1, Q8_0, Q4_K, Q5_K만 지원하므로 재사용할 수 없다. upstream ggml의 HIP vec-dot 구현을 가져와 벤치 커널을 만든다.
- **회전 비용:** 둘로 나눈다.
  - 독립 FWHT 커널의 비용(K별, 배치별): R0b에서 잰다.
  - 활성화 생성 커널에 융합했을 때의 추가 시간: R2에서 융합 커널을 만든 뒤 잰다.
- **산출물:** "타입별 bpw 대 실효 대역폭 대 토큰당 시간" 표. R4 레시피, E8의 비용 모델, E7b 진행 판단에 쓴다.

#### R0b 결과가 계획에 주는 것 (2026-10-28, [R0b 결과](R0B-RESULTS.md), 창 `rq0b-a03`)

| 발견 | 계획 반영 |
|---|---|
| IQ4_XS가 4비트 중 가장 빠르다. AR 27B 전체 토큰당 64.0 ms(상한의 89%), Q4_K 73.4 ms, IQ4_NL 70.1 ms | 4비트 레시피의 기본 후보는 IQ4_XS. E6에서도 속도 기준으로 쓴다 |
| Q3_K와 IQ3_S는 연산에 묶인다. AR 상한의 54%, 바이트가 24% 적은데도 Q4_K보다 느리다(84.8, 85.0 ms) | **E3는 지금 커널로는 속도 이득이 없다.** E3의 품질 결과는 그대로 재되, 속도 이득은 E9(3비트 커널 최적화) 뒤에 판단한다 |
| Q2_K는 AR에서 가장 빠르다(42.5 ms). 하지만 DFlash2 verify 8행에서는 186 ms로 Q8_0(128 ms)보다 느리다. IQ2_*는 AR부터 상한의 54–59% | DFlash2 운영에서는 2비트 이득이 verify에서 사라진다. E7b는 verify용 2비트 커널을 포함해야 하고, 진행 여부는 E7a 품질 결과를 보고 정한다(사용자 결정 2026-10-28) |
| prefill은 모든 타입이 WMMA 연산에 묶인다 | 비트를 내려도 prefill은 빨라지지 않는다. 효율 점수판의 속도 이득은 decode에서만 기대한다 |
| 독립 FWHT는 AR 토큰당 1.26 ms(2%). 회전 257회 × launch 약 5 µs로 launch에 묶인다 | 활성화 생성 커널에 융합한다는 R2 방침의 근거. 융합 뒤 추가 시간을 R2에서 다시 잰다 |

### R1 컨버터 (로더와 분리된 독립 도구)

- 위치: 공통 본체 `tools/rotquant/`, 27B 입력 그룹 정의 `tools/qwen27b/rotquant_groups.py`("공통과 모델별 경계" 절). 언어는 Python(NumPy, 기존 `tools/gufo/gguf.py`와 `tools/gufo/safetensors.py`, 필요하면 GGUF 쓰기를 추가)이다. 컨버터는 `--groups <모델 정의>`로 모델을 받는다. 정의가 없는 모델이면 거부한다.
- **양자화는 직접 구현하지 않는다.** 두 단계로 나눈다.
  1. `rotate`: 입력(BF16 GGUF. 분할 GGUF 포함. HF BF16 safetensors는 선택)을 받아 그룹별로 `W' = W·D·H`를 계산하고, **회전된 BF16 GGUF**와 짝 텐서, 전역 KV를 쓴다. 텐서는 하나씩 스트리밍해 메모리 사용을 제한한다.
  2. `quantize`: 기준 llama.cpp(`.#llama-cpp-reference`)의 `llama-quantize`를 레시피에서 만든 `--tensor-type` 규칙과 imatrix로 호출한다. ggml 참고 양자화기를 그대로 쓰므로 블록 형식이 gufo 커널이 기대하는 것과 정확히 같다. `.rot` 텐서(1차원 I8)가 그대로 복사되는지 R1에서 확인한다. 복사되지 않으면 `quantize` 뒤에 붙이는 단계를 둔다.
- 옵션:
  - `--recipe <yaml>`: 텐서 이름 패턴별 양자화 타입, 그룹별 회전 on/off
  - `--rotate-groups attn_in,ffn_in,...|all|none`: 레시피 위에 덮어쓰는 빠른 선택
  - `--seed`: 부호 생성용. 결과는 파일에 저장된다
  - `--imatrix <file>`: R3 산출물
  - `--rotate-only`: BF16 출력만 하고 양자화하지 않음(R2 ②의 검증용)
  - `--copy-recipe-from <gguf>`: 기존 GGUF(예: UD-Q4_K_XL)의 텐서별 타입을 그대로 가져온다. 같은 바이트에서 회전 효과만 비교할 때 쓴다
  - `--dry-run`: 텐서별 타입, 회전 여부, 예상 크기와 bpw 표만 출력
- 지원 타입 확인: 27B 로더가 받는 타입(`src/models/qwen/weights.cpp:59` `SupportsTensorType`: F32, BF16, Q8_0, Q6_K, Q5_K, Q4_K, Q3_K, IQ4_NL, IQ4_XS, IQ3_S, 그리고 역할에 따라 Q8_K) 밖의 타입을 레시피가 지정하면 컨버터가 거부한다.
- 출력 GGUF에 컨버터 버전, 레시피 해시, 원본 파일 해시를 KV로 남긴다(`gufo.rotation.source_*`).
- E7a용 dequant 경로 확인: 2비트대 타입을 BF16으로 다시 풀 수 있어야 한다. 지금 `tools/gufo/gguf.py`는 F16, BF16, F32, Q4_0, Q8_0, Q4_K, Q5_K, Q6_K만 풀고, C++ `src/core/quant/ggml_dequant.cpp`는 Q2_K, IQ2_XXS까지는 되지만 IQ2_XS, IQ2_S, IQ1_M은 없다. 기준 llama.cpp의 gguf-py나 ggml dequant를 쓰고, 어느 쪽을 쓸지 R1에서 확인해 기록한다.
- 테스트(`tests/` CPU, GPU 불필요): Hadamard 직교성, 그룹 부호 일치, 작은 합성 텐서로 `W'·(H·D·x) == W·x`(BF16 허용 오차), 레시피 파싱과 거부, 짝 텐서와 KV 쓰기/읽기 왕복.

### R2 로더와 실행

- 공통 검사(`src/core/quant/rotation.{hpp,cpp}`): 전역 KV와 `.rot`을 읽고, 모델이 넘긴 그룹 표로 위의 공통 거부 조건을 검사한다.
- 가중치 바인딩(`src/models/qwen/weights.cpp`): 27B 그룹 표를 공통 검사에 넘기고, 그룹별 부호 벡터를 디바이스에 올린다. 역할별로 회전 경로가 있는지는 여기서 검사한다.
- 커널: FWHT device 함수는 공통 헤더 `src/core/hip/detail/hadamard.hpp`. 27B 독립 회전 커널과 융합 커널은 `src/models/qwen/hip/kernels/`(선언 `ops/hadamard.hpp`).
  - 독립 커널 `x' = H·D·x`(행마다 128점 FWHT 7단계, wave 내 shuffle 또는 LDS)는 FP32 활성화 경로와 테스트 기준으로만 쓴다.
  - **활성화 생성 커널은 처음부터 "회전 후 Q8_1 양자화"로 고친다.** 27B의 활성화 생성 커널은 결과를 곧바로 Q8_1로 쓰기 때문에, 회전을 끼워 넣을 FP32 중간값이 없다. Q8_1을 풀어 회전하고 다시 양자화하면 오차가 더해지므로 그렇게 하지 않는다. 대상:
    - `attn_in`, `ffn_in`, `head_in`: `LaunchBatchedFusedRMSNormQuantizeQ8_1`(`ops/gemm.hpp:244`)
    - `ssm_out_in`: `LaunchBatchedFusedSSMPostNormGateQuantizeQ8_1`(`ops/gemm.hpp:228`)
    - `ffn_down_in`: `LaunchBatchedFusedSwiGLUQuantizeQ8_1`(`ops/gemm.hpp:260`)
    - `attn_out_in`: attention 출력 + gate 뒤의 Q8_1 양자화(R2 시작 때 위치를 확인한다)
    - 그 밖의 `LaunchQuantizeActivationQ8_1`, `LaunchQuantizeActivationQ8_1FromFp32` 호출 지점
  - decode(GEMV), prefill(batched), DFlash2 verify, graph capture 경로가 각각 어떤 활성화 생성 커널을 거치는지 목록으로 만들고, 빠짐없이 바꾼다.
  - **DFlash2 드래프터의 헤드 경로:** 드래프터가 타깃 `output`을 쓰는 곳에서도 `head_in` 회전을 적용한다.
  - **회전을 아직 지원하지 않는 실행 경로는 거부한다.** Qwen `--cpu` 경로, Q8_1을 거치지 않는 BF16/F16 matmul 경로(hipBLASLt 등)는 회전을 적용하거나, 회전 파일이 들어오면 경로 이름을 들어 거부한다. R2 ②의 회전 BF16 검증은 BF16 경로를 쓰므로, 적어도 그 경로는 회전을 지원해야 한다.
  - KVarN Hadamard 구현과 DeepSeek V4 Flash indexer 변환(`src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_indexer.hip.hpp`)을 테스트 기준으로 쓴다.
- 성능 목표: 회전 비용이 decode 토큰당 시간의 1% 미만(활성화가 벡터 하나라 작아야 한다). R2에서 측정해 기록한다.

### R4 실험

같은 R0 하네스와 같은 날의 속도 측정으로 비교한다.

| 실험 | 목적 |
|---|---|
| E1: UD-Q4_K_XL 레시피 그대로 ± 회전 | 같은 바이트에서 회전이 품질을 얼마나 올리는가. 회전 효과만 보도록 두 조건 모두 같은 보정 코퍼스로 만든다: 회전 없음은 원래 기저 imatrix, 회전 있음은 회전 기저 imatrix(R3). unsloth의 UD-Q4_K_XL 파일 자체는 보정 데이터가 달라 회전 효과와 섞이므로 외부 참고로만 둔다 |
| E2: dense 전체 Q4_K ± 회전, `ssm_out`은 Q8_0과 Q4_K 각각 | 민감 텐서(`ssm_out`, `attn_output`, `ffn_down`, `output`)가 회전 후 4비트를 견디는가 |
| E3: 약 3.0–3.5 bpw 혼합(Q3_K, IQ3_S, IQ4_XS) + 회전 | UD-Q4_K_XL 품질을 유지하면서 바이트를 줄여 decode 속도를 올릴 수 있는가. GSQ-RCO-IQ3_S와도 비교 |
| E4: imatrix 있음/없음 | R3의 효과 |
| E5: 채택 후보의 DFlash2 수락률과 서빙 속도 | 품질 변화가 드래프트 수락률을 통해 실제 속도에 어떻게 반영되는가 |
| E6: 회전 뒤 1단 블록과 2단 블록 비교 | 아래 E6 절 |
| E7: 회전 + 2비트대 혼합(평균 약 2.5–3.0 bpw) | 회전 이득이 가장 크다고 알려진 3비트 미만 구간에서, 실용 품질을 유지하며 바이트를 얼마나 줄일 수 있는가. 아래 E7 절 |
| E8: 텐서별 비트 배분 자동화 | 총 바이트 예산 안에서 품질 손실이 최소인 배분을 사람이 아니라 도구로 찾는다. 아래 E8 절 |
| E9: 3비트 커널 최적화(Q3_K, IQ3_S) | R0b에서 3비트가 연산에 묶여 Q4보다 느렸다. 회전의 품질 이득이 바이트 절감으로 이어질 가장 현실적인 구간이 3비트이므로, AR과 verify 커널을 대역폭에 묶이게 만들 수 있는지 본다. 아래 E9 절 |

#### E6: 회전 뒤 1단 블록과 2단 블록

- **가설:** 회전하면 값이 고르게 퍼진다. 그러면 K-quant 서브블록 스케일(국소적인 크기 차이를 따라가는 장치)의 이점이 줄어든다. 그래서 Q4_K와 IQ4_NL 같은 1단 포맷과의 품질 차이가 좁아질 수 있다.
- **비교:** 같은 텐서 그룹을 Q4_K, IQ4_XS, IQ4_NL로 각각 양자화하고, 회전 있음과 없음으로 나눈다(3 × 2). R0 하네스로 NLL, KL, top-1을 측정하고, 같은 날 decode 속도도 측정한다. 나머지 텐서는 모든 조합에서 같은 타입으로 고정한다.
- **판정:** 1단 포맷이 품질에서 비기고 커널이 더 단순하거나 빠르면 채택 후보로 둔다. 비기는지는 R0의 paired bootstrap 신뢰구간으로 판정한다.
- **대상 타입:** 27B 로더가 지원하는 타입으로 한정한다(`SupportsTensorType`). 1단 4비트는 지금 IQ4_NL뿐이다. Q4_0이나 Q4_1이 필요하면 로더 허용, decode GEMV, prefill GEMM, DFlash2 verify 경로의 커널 추가 범위를 따로 적고 사용자와 정한다.

#### E7: 회전 + 2비트대 혼합 (평균 약 2.5–3.0 bpw)

- **목적:** 회전의 이득이 가장 크다고 알려진 3비트 미만 구간에서, 27B가 실용 품질을 유지하면서 바이트를 얼마나 줄일 수 있는지 본다.
- **후보 타입:** Q2_K, IQ2_XXS, IQ2_XS, IQ2_S. 참고로 IQ1_M도 넣는다.
- **배분:** 민감한 텐서(`ssm_out`, `attn_output`, `output`, `ffn_down` 등)는 E2 결과에 따라 높은 비트로 두고, 나머지만 2비트대로 내린다. 모든 텐서를 2비트로 내리는 설정은 대조군으로만 둔다.
- **비교 대상:** UD-Q4_K_XL, E3의 최선 설정, ISTA GSQ-RCO-IQ3_S. GSQ 파일은 IQ2_S, IQ2_XS, IQ2_XXS, Q2_K, IQ1_M 텐서를 섞어 쓰므로(`source.json`의 `actual_storage_tensor_counts`) 레시피 참고로도 쓴다.
- **측정:** R0 하네스로 NLL, KL, top-1을 측정하고, 같은 날 decode 속도와 DFlash2 수락률도 측정한다. 2비트에서는 드래프트 수락률이 떨어져 속도 이득을 깎을 수 있으므로 반드시 함께 본다.
- **imatrix 필수:** 2비트대 IQ 타입은 imatrix 없이 품질이 크게 무너진다. `llama-quantize`도 IQ2_XXS, IQ2_XS, IQ1_M은 imatrix 없이 만들기를 거부한다. 그래서 E7은 R3(회전 기저 imatrix) 이후에 진행한다.
- **커널 범위:** 지금 27B 로더(`SupportsTensorType`)의 하한은 Q3_K와 IQ3_S이고, Q2_K, IQ2_*, IQ1_M은 지원하지 않는다. 그래서 둘로 나눈다.
  - **E7a:** 커널을 추가하지 않고 품질만 먼저 본다. 측정 경로는 따로 정한다. 회전 GGUF는 llama.cpp가 로드를 거부하므로 llama.cpp로는 잴 수 없다.
    - 후보(제안, 미결정): 2비트대로 양자화한 텐서를 다시 BF16으로 풀어(fake quant) 회전 BF16 GGUF로 쓰고, R2에서 검증한 회전 BF16 경로로 gufo에서 잰다. 품질은 정확히 같지만 파일이 BF16 크기(약 54 GB)라 D2 디스크 계획에 넣어야 하고, 속도는 잴 수 없다.
  - **E7b:** E7a가 유망할 때만 로더 허용, decode GEMV, prefill GEMM, DFlash2 verify 커널을 추가하고 속도를 측정한다. R0b에서 upstream mmvq는 verify 8행에서 활성화 행마다 블록을 다시 디코드해 상한의 19–27%였다. 그래서 E7b는 행 사이에 디코드를 공유하는 2비트 small-batch 커널을 포함한다. 진행 여부는 E7a 결과를 보고 정한다(사용자 결정 2026-10-28).

#### E8: 텐서별 비트 배분 자동화

- **목적:** 레시피를 사람이 정하지 않고, 총 바이트 예산 안에서 품질 손실이 최소인 배분을 찾는다.
- **비용 때문에 범위를 줄인다.** 층 단위로 실제 KL을 재면 64층 × 약 8계열 = 약 500조합이고, 조합마다 GGUF(각 11–18 GB)를 만들고 32K 토큰을 평가해야 한다. 그래서 두 단계로 나눈다.
  1. **오프라인 대리 지표로 순위를 정한다.** 텐서마다 타입별로 "imatrix로 가중한 양자화 오차"를 CPU에서 계산한다(텐서당 수 초). 이 값으로 계열과 층의 민감도 순위를 만든다.
  2. **층을 4–8개 구간으로 묶어, 상위 후보 구간만 실제 KL로 검증한다.** 한 구간의 한 계열만 낮은 비트로 바꿨을 때의 KL 증가를 R0 하네스로 잰다. 대리 지표 순위와 실제 KL 순위가 맞는지도 기록한다.
- **배분:** R0b의 타입별 실효 시간을 비용으로 두고, 예산 안에서 품질 손실이 가장 적은 배분을 탐욕법으로 고른다.
- **검증:** 탐욕법은 민감도가 서로 더해진다고 가정한다. 그래서 고른 배분은 반드시 한 번 실제로 만들어 R0 하네스와 속도로 측정하고, 예측한 KL과 실제 KL을 함께 기록한다.
- **출력:** 컨버터 `--recipe` 형식.
- **비교 대상:** 사람이 정한 E3·E7 레시피, unsloth UD-Q4_K_XL, GSQ-RCO-IQ3_S.

#### E9: 3비트 커널 최적화 (Q3_K, IQ3_S)

- **배경:** R0b에서 Q3_K와 IQ3_S는 AR 상한의 54%, verify 8행에서 34–38%였다. 같은 커널 경로의 Q4–Q8은 80–91%다. 바이트가 24% 적은데도 Q4_K보다 느리다.
- **먼저 원인을 확인한다.** `tools/prof/isa_mix.py`와 rocprof로 Q3_K scale 풀기, IQ3_S grid 조회 중 무엇이 병목인지 본다(R0b 결과 문서는 아직 짐작이라고 적었다). 규칙은 `.agents/skills/optimize-kernel/SKILL.md`를 따른다.
- **목표:** AR에서 대역폭 상한의 80% 이상(IQ4_XS 수준). 그러면 27B 전체 토큰당 약 57 ms 이하로, IQ4_XS(64.0 ms)보다 빨라진다. verify 8행에서도 Q4 계열 이상.
- **측정:** R0b 벤치(`tools/bench/rq_type_bench.hip`)로 전후를 같은 창에서 번갈아 잰다. 운영 커널을 바꾸므로 결과가 비트 단위로 같은지(또는 기존 오차 한도 안인지) 확인한다.
- **순서:** E3 품질 결과에서 3비트가 UD-Q4_K_XL 품질에 가까울 때 진행한다. 품질이 무너지면 커널을 빠르게 만들 이유가 없다.
- 성공한 최적화는 운영 기본 경로가 된다(`CLAUDE.md`). 회전 GGUF뿐 아니라 기존 3비트 GGUF(UD 레시피 안의 Q3_K, IQ3_S 텐서)도 이득을 본다.

#### 효율 점수판

- 모든 실험 결과(R0 기준선 포함)를 한 표에 쌓는다. 위치: `plan/rotated-quant/RESULTS.md`.
- 열:
  - 품질: KL, NLL, top-1(각각 paired bootstrap 신뢰구간)
  - 크기: 평균 bpw, 파일 크기, 상주 메모리(HIP 보고 최대값)
  - 속도: decode tok/s(d0와 d32K), prefill tok/s, DFlash2 수락률과 실효 tok/s
  - 전력당 토큰(tok/J): 가능할 때만 넣는다. gufo에는 전력 측정 도구가 없으므로(`tools/bench`, `tools/prof` 확인), 전력값을 어디서 읽을지(amdgpu hwmon, 패키지 전력 등) 정해지면 넣는다
  - 실제 작업 게이트(D4) 결과
  - 측정 날짜, 리비전, 파일 해시
- 품질 대 바이트, 품질 대 실효 tok/s의 파레토 앞쪽에 있는 설정을 채택 후보로 표시한다.

속도는 `CLAUDE.md` 규칙대로 요청·단계별로 측정하고, 5% 또는 3 ms 기준으로 판정한다. 잡음이 큰 결과는 결론 없음으로 남긴다.

### R5 이후: Flash-Next 확장 참고 (지금은 구현하지 않음)

- **목표:** Flash-Next의 dense 트렁크를 8비트에서 회전 + 4비트로 전환한다. Flash-Next decode 격차의 주원인이 dense 트렁크의 비트 수이기 때문이다([gufo-org/gufo#474](https://github.com/gufo-org/gufo/issues/474)).
- **기존 근거:**
  - #474는 회전 없이 dense를 Q8_0에서 Q4_1로 내리고 전용 dense vec 커널을 추가했다. MTP 사이드카 재양자화 등 다른 변경과 합쳐 에이전트 벤치에서 46.5 → 약 54.5 tok/s를 얻었다. 같은 조건의 halogen ht43은 약 60이었다.
  - 같은 실험에서 품질 기준을 통과하려면 `ssm_out`을 Q8_0으로 남겨야 했다. 7개 텐서 계열 중 `ssm_out`이 가장 민감했다.
  - 출력 헤드를 Q8_0에서 Q4_K로 내리는 것은 속도에 도움이 됐다. 반대로 Q8_0으로 되돌리면 decode가 5.9% 느려졌다.
  - halogen v2는 dense를 회전 + 약 4비트로, mixing 층을 6비트로 둔다.
  - #474 시점(gufo 0.8.0)에 Flash-Next의 dense 행렬 로더는 BF16, F16, Q8_0만 받았다. 확장할 때는 현재 지원 범위를 다시 확인한다.
- **27B 결과 중 가져다 쓸 것:**
  - E2: 회전 후 `ssm_out`이 4비트를 견디는지. Flash-Next에서 `ssm_out`을 8비트로 남겨야 하는지 판단하는 직접 근거다.
  - E6: 1단 포맷과 2단 포맷 비교. Flash-Next `down_exps`는 K = 640으로 256의 배수가 아니어서 1단 포맷만 쓸 수 있다.
  - E7: 2비트대 결과는 Flash-Next 전문가 가중치를 더 내릴지 판단하는 참고로 쓴다. 다만 #474는 전문가 바이트가 decode 속도의 주된 제약이 아니라고 보고했으므로, Flash-Next에서는 dense 4비트 전환을 우선하고 전문가 2비트는 후순위로 둔다.
- **Flash-Next에만 있는 일:** hyper-connection, gated residual, n-gram 테이블, MTP 헤드의 입력 그룹 정의와 4비트 dense 커널이 필요하다. 이건 확장 단계에서 따로 계획한다.

## 결정이 필요한 것

- **D1 컨버터 언어와 위치 (결정 2026-10-28):** Python. 공통 본체는 `tools/rotquant/`, 모델별 그룹 정의는 각 모델 도구 디렉터리에 둔다("공통과 모델별 경계" 절).
- **D2 디스크 (결정 2026-10-28: 디스크 용량을 확보한 뒤 진행. 원본은 미리 만들어진 BF16 GGUF를 받는다):** 현재 여유 공간이 54 GB다(`/`, 97% 사용). 로컬 `qwen-3.8-27b-bf16`에는 메타데이터만 있고 safetensors가 없다.
  - **원본:** `unsloth/Qwen3.8-27B-GGUF`의 `BF16/Qwen3.8-27B-BF16-0000{1,2}-of-00002.gguf`(49.99 GB + 4.67 GB, 확인 시점 리비전 `4ca72078`). 이 파일이 R0의 BF16 기준이자 컨버터 입력이다. HF safetensors를 받아 변환하는 단계가 없어져 약 54 GB를 아낀다. gufo reader는 분할 GGUF를 읽는다(`src/core/gguf_reader.hpp:144`).
  - **받을 때 기록할 것:** 리비전, 파일별 sha256, 그리고 이 GGUF가 어느 `Qwen/Qwen3.8-27B` 리비전에서 변환됐는지(모델 카드와 GGUF 메타데이터로 확인, 확인되지 않으면 unknown으로 남긴다).
  - **받음 (2026-10-28):** `~/projects/ai.tools/storage/models/llm/qwen-3.8-27b-bf16-gguf/`(같은 폴더에 `SHA256SUMS`, `PROVENANCE.md`). 리비전 `4ca72078`. sha256은 `00001` `b9966e82…191f`, `00002` `92e3943c…fa02`이고 HF 값과 일치한다. 원본 `Qwen/Qwen3.8-27B` 리비전은 모델 카드와 GGUF 메타데이터에 없어 unknown이다. 디스크 확보 뒤 루트 여유 공간은 약 430 GB다. `window.py`는 여유 공간이 150 GiB보다 적으면 창을 시작하지 않으므로, 실험 출력과 E7a 파일은 측정이 끝나면 지운다.
  - **필요 용량:** BF16 원본(약 55 GB), 회전 BF16 GGUF(약 55 GB), 실험 출력(각 11–18 GB), E7a fake quant 파일(각 약 55 GB, 한 번에 하나만 두고 측정 후 지운다). 약 150 GB 이상을 확보한다.
- **D3 원본 소스 (결정 2026-10-28):** Q8_0에서 시험 출력을 만들지 않는다. 디스크를 확보한 뒤 BF16 원본(D2의 unsloth BF16 GGUF)에서만 만든다. 실제 모델 파일이 필요한 R0, R1의 실모델 변환, R2 이후는 디스크 확보 뒤에 시작한다. 합성 텐서만 쓰는 R1 코드와 단위 테스트를 먼저 시작할지는 사용자 확인 뒤 정한다. R0b는 무작위 가중치만 쓰므로 디스크와 무관하게 시작할 수 있다.
- **D4 채택 기준:** 수치 기준은 R0 결과를 보고 정한다(KVarN K0와 같은 방식). 다만 KL과 NLL만으로 채택하지 않는다. 저비트에서는 tool-call 형식, 한국어, 긴 문맥 능력이 먼저 깨질 수 있다. 채택 후보는 다음 실제 작업 게이트를 거친다. 비교 기준은 UD-Q4_K_XL이고, 각 suite의 통과/실패 판정 방식은 R0 결과를 보고 정한다.
  - 에이전트: `tests/functional/pi_agent.py`(리플레이), `agent_long.py`
  - tool-call 형식: `tests/functional/tool_*.py`(tool_choice, tool_streaming, tool_parameter_order 등)
  - 긴 문맥: `tests/functional/test_long_context.py`
  - 한국어: **기존 suite가 없다.** 한국어 프롬프트 세트(지시 따르기, 요약, 코드 설명, tool-call)를 새로 만든다. R0 평가 코퍼스, R3 보정 코퍼스와 겹치지 않게 한다

## 위험

- 4비트 이상에서는 회전의 품질 이득이 작을 수 있다(회전 계열 연구의 큰 이득은 주로 4비트 미만). 그 경우 E3(바이트 절감)이 주된 가치가 된다.
- 회전 후에는 K-quant 서브블록 스케일의 이점이 줄어들 수 있다. 그래서 레시피를 고를 때 기존 UD 레시피의 타입 배분이 최적이라고 가정하지 않는다(E1의 `--copy-recipe-from`은 비교 기준일 뿐이다. E2, E3, E6 결과로 다시 고른다).
- 2비트대에서는 품질 저하가 드래프트 수락률 저하로 이어져 바이트 절감만큼 속도가 오르지 않을 수 있다. 커널 추가 비용도 크므로 E7a 결과를 보고 E7b 진행 여부를 판단한다.
- `llama-quantize`가 회전 텐서에 이름 기반 휴리스틱을 적용하거나 `.rot` 텐서를 다르게 다룰 수 있다. R1에서 확인하고, 문제가 있으면 `--tensor-type`으로 모두 명시한다.
- 활성화를 만드는 경로(decode, prefill, verify, graph capture)와 가중치를 공유하는 소비자(DFlash2 드래프터 헤드)가 여럿이다. 하나라도 회전을 빠뜨리면 조용히 품질이나 수락률이 떨어진다. R2 ②를 decode, prefill, DFlash2 verify 각각에서 돌리고, R2 ③으로 드래프터 경로를 확인한다.
- 보정 코퍼스와 평가 코퍼스가 겹치면 imatrix 효과가 부풀려진다. 두 코퍼스를 분리하고 해시를 기록한다(R0, R3).
- 원본 gufo에는 회전 가중치 개념이 없다. 포크(`namote/gufo`)에서 독자적으로 유지한다는 기존 방침([PR 계획](../gufo-upstream-port/PR-PLAN.md))을 따른다.

## 다음 계획 후보 (이 계획 범위 밖, 기록만)

- **Flash-Next KV 양자화:** 지금 Flash-Next는 `fp16` KV만 받는다. upstream에 q8_0 KV PR [gufo-org/gufo#855](https://github.com/gufo-org/gufo/pull/855)가 열려 있다. KVarN 적용은 구조 차이부터 검토한다. Flash-Next attention은 pooled key와 indexer 구조라 27B와 다르다([모델 확장](../gufo-upstream-port/MODELS.md)).
- **verify 단계 MoE 커널 효율:** [gufo-org/gufo#474](https://github.com/gufo-org/gufo/issues/474)는 halogen과의 남은 격차가 드래프트가 아니라 verify 1회 비용(42 ms 대 57.5 ms)에서 나온다고 보고했다(전체 decode는 gufo가 halogen의 0.85배). rocprof로 verify 단계를 커널별로 나눈다.
- **드래프트 개선:** 요청을 넘어 유지되는 n-gram 풀([gufo-org/gufo#239](https://github.com/gufo-org/gufo/issues/239) 후속), 저비트 타깃에서의 드래프트 정책 조정.
- **아낀 메모리의 활용:** 같은 메모리로 동시 세션 수나 문맥 길이를 얼마나 늘릴 수 있는지 측정한다.
