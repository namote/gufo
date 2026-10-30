# F2 계획: Flash-Next EXL3 5.0 bpw 직접 양자화 (조사, 2026-10-31, XQ)

상위 문서: [PLAN.md](PLAN.md) F2, [XQ-RESULTS.md](XQ-RESULTS.md)(27B에서 같은 도구로 해 본 경험), [W-COMMON](../rotated-quant/W-COMMON.md).

상태: **조사만 했다.** GPU, 컨테이너, 큰 파일 해시는 쓰지 않았다. 소스 읽기, safetensors 헤더 읽기, HF API로 작은 JSON 받기만 했다. 시험 창은 감독 허락을 기다린다.

기준 소스: upstream `exllamav3` v1.6.5 `cdc785db`(`/home/m2on/projects/gufo/build/w9-src/exllamav3/`). 파일 경로는 그 아래 `exllamav3/`를 기준으로 쓴다.

## 요약

- **지원:** 1.6.5가 이 아키텍처를 그대로 지원한다. 추가로 짤 것은 없다.
  - 대상: `Qwen4ExpForConditionalGeneration`(`architecture/qwen4_exp.py`, `qwen4_exp_mtp.py`)
  - 들어 있는 것: MoE 512 experts / top-10 + shared expert, GDN 36층 + QSA full attention 12층(indexer), gated-residual hyper-connection ×4, PLE/n-gram 표, MTP 1층, 비전
  - n-gram 표 양자화 경로: `conversion/ngram.py`, `util/convert_ngram.py`
- **설정:** turboderp `5.05bpw_h6_ng6` 팩(v1.4.4)의 설정에 맞추면 아래와 같다.

  ```
  -b 5.0 -hq -hb 6 -mb 5 -ngb 6 --out_scales always -cb mul1, 비전은 기본 6 bpw
  ```

  - experts 5, attention/GDN/shared expert 7(`-hq`로 +2), indexer 5, head 6, MTP 5, n-gram K=6
  - router(`mlp.gate`), `in_proj_a/b`, hyper-connection, PLE projection은 fp16 원본으로 둔다. 임베딩은 BF16이다.
- **보정:** c02 250 × 2048 파일(`c02-250x2048.safetensors`)을 그대로 쓸 수 있다. 토크나이저 sha가 같다(`0997f410…`).
  - expert 커버리지는 구조상 전부다. 이 아키텍처는 `calibration_all_experts = True`(`qwen4_exp.py:306`)라 Hessian을 잡을 때 모든 expert가 모든 보정 토큰을 본다.
- **시간 [짐작]:**
  - 본체(experts 121 G 가중치): **약 18–20 h**
  - n-gram 표(51.2 G 가중치, K=6, 보정 없음): **약 8–13 h**
  - 합계 **약 26–33 h**다. 3시간 창으로 나누면 9–11개다.
  - n-gram 표를 turboderp 파일로 대신하면 약 18–20 h로 준다.
- **메모리 [짐작]:** 최대 약 40–70 GB. 보정 상태만 21 GB(fp32, 4 stream)이고, 체크포인트도 한 벌에 21 GB다.
- **디스크 [짐작]:**
  - 출력 팩 약 125 GB, 작업 디렉터리 약 130 GB
  - 외장 가용이 420 GB라서, 창 사이에 작업 디렉터리를 복사하는 27B 방식(창마다 qtensors 전체 복사)을 그대로 쓰면 빠듯하다.
  - 아래 "창 사이 작업 디렉터리"에서 선택이 필요하다.

## 1. 아키텍처 지원 (`architecture/qwen4_exp.py`)

| 구성 | 원본 텐서(BF16) | 1.6.5 처리 | 근거 |
|---|---|---|---|
| routed experts 512개, top-10, 48층 | `mlp.experts.gate_up_proj` [512, 1280, 2560] (150 GiB), `mlp.experts.down_proj` [512, 2560, 640] (75 GiB) | `BlockSparseMLP`가 fused 키 `experts.gate_up_proj`/`experts.down_proj`를 expert별 `up/gate/down`으로 나눠 양자화한다 | `qwen4_exp.py:185-203` (`key_gate_up_split`, `key_down_split`, `transpose_fused_weights = False`) |
| shared expert | `mlp.shared_expert.{gate,up,down}_proj` (640 × 2560) | `GatedMLP`, `select_hq_bits = 2` | `:204-217` |
| router | `mlp.gate` [512, 2560], `shared_expert_gate` [1, 2560] | 양자화하지 않음(fp16) | turboderp 팩 `quantization_config.json`에서 raw fp16 |
| GDN 36층 | `linear_attn.in_proj_qkv/z/out_proj`, `in_proj_a/b`, conv1d, A_log, dt_bias | `GatedDeltaNet`, `output_gate_type = sigmoid`, `select_hq_bits = 2`. `in_proj_a/b`는 fp16 | `:109-136` |
| full attention 12층 + QSA indexer | `self_attn.q/k/v/o_proj`, `indexer.index_qk_proj` | `Attention`(`interleaved_gate`, `select_hq_bits = 2`) + `QSAIndexer`(`select_hq_bits` 기본값 2이지만 블록에서 주지 않음. turboderp 팩에서는 5 bpw) | `:139-182`, `modules/qsa_indexer.py:58` |
| hyper-connection ×4 | `{attn,mlp}_hyper_connection.*`, `hyper_connection_mixer` | `GatedResidual`, fp16으로 저장 | `:91-104`, `:283-291` |
| PLE + n-gram 표(층 1 앞) | `layers.1.ple.ple_embedding.ngram_embedding.shard_{0..127}.weight` [2,500,012, 160] × 128 (95.4 GiB), `key_proj`, `value_proj` | 변환 첫 단계에서 표를 K비트 trellis로 양자화하거나 `-ngf` 파일을 복사한다. 이후 보정 forward는 양자화된 표를 디스크에서 읽어 쓴다. projection은 fp16 | `:249-266`, `conversion/ngram.py:600-648` |
| MTP 1층(experts 512개 포함) | `mtp.*` (약 4.9 GiB) | `Qwen4ExpMTPModel`, `qbits_key = "mtp_bits"`, 보정 없이 양자화 | `qwen4_exp_mtp.py:78,90`, `doc/convert.md` |
| 비전 | `model.visual.*` (0.84 GiB) | `Qwen3VLVisionModel`, `default_vision_bits = 6` | `architecture/qwen3_vl.py:172` |
| lm_head | [248320, 2560] | `head_bits` | `:292-300` |

- 원본: `/run/media/m2on/T5 EVO/AI/model/originals-flash-next-hf-20261030/Qwen-Qwen3.8-Flash-Next/`
  - rev `de4b8e4d`, 131개 샤드, 335.3 GiB(텐서 합계). PROVENANCE.md에 HF LFS oid 대조 결과가 있다.
  - 헤더를 읽어 위 모양을 확인했다. 모든 텐서가 BF16이고, 버퍼 3개만 I64다.
- **추가로 짤 것: 없다.** 1.6.5의 upstream 경로만 쓴다. 같은 아키텍처로 turboderp가 v1.4.4에서 팩을 만들었다.
- **위험[짐작]:** gfx1151에서 이 아키텍처의 Triton 커널(QSA indexer, fla chunk)과 MoE 양자화 경로를 돌려 본 적이 없다. 27B와 달리 MoE 분기, PLE, indexer 경로를 시험 창에서 확인해야 한다.

## 2. 텐서별 비트 할당

- **turboderp 팩**(`turboderp/Qwen3.8-Flash-Next-exl3` 브랜치 `5.05bpw_h6_ng6`, commit `7cef615f`, `quantization_config.json` 96.7 MB를 받아 집계):
  - 설정: `{"version": "1.4.4", "bits": 5.05, "head_bits": 6, "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always", "codebook": "mul1", "vision_bits": 6, "mtp_bits": 5}`
  - 저장 크기: 본체 샤드 11개 83.6 GB, `ngram_embedding.safetensors` 39.0 GB, 합계 약 123 GB

| 텐서 | turboderp 5.05 | 저장 크기(turboderp) |
|---|---|---:|
| `experts.N.{gate,up,down}_proj`(층마다 512 × 3, 73,728개) | **5** | 70.75 GiB |
| GDN `in_proj_qkv`, `in_proj_z`, `out_proj` | 7 | 1.69 GiB |
| attention `q/k/v/o_proj` | 7 | 0.49 GiB |
| shared expert `gate/up/down` | 7 | 0.19 GiB |
| `indexer.index_qk_proj` | 5 | 0.01 GiB |
| `lm_head` | 6 | 0.44 GiB |
| router `mlp.gate`, `in_proj_a/b`, hyper-connection, PLE `key/value_proj` | fp16 원본 | 1.4 GiB |
| `embed_tokens` | BF16 | 1.18 GiB |
| n-gram 표 | K=6(행당 61 × int16 = 160 × 6 + 16비트 스케일) | 36.4 GiB |
| MTP | 5 | (별도) |
| 비전 | 6 | (별도) |

- **1.6.5로 같게 맞추는 인자:** `-b 5.0 -hq -hb 6 -mb 5 -ngb 6 --out_scales always -cb mul1`
  - `-hq`: `select_hq_bits = 2`인 GDN, attention, shared expert가 5 + 2 = 7이 된다. indexer는 블록에서 `select_hq_bits`를 넘기지 않아 5로 남는다[짐작: 1.6.5 할당이 1.4.4와 같은 결과를 낼지는 시험 창의 전략 출력으로 확인].
  - `-mb 5`: turboderp MTP는 5다. 1.6.5 기본값은 4라서 명시해야 한다.
  - `-vb`: 1.6.5 기본값이 6이다(turboderp와 같음).
  - `-ngb 6`: 1.6.5 기본값은 `round(bits)` = 5이므로 명시해야 한다.
  - `-eb`: 기본값 16(BF16)을 둔다.
- **사용자 목표(5.0급 experts)와 맞다.** 재개할 때마다 `--out_scales always -cpi …`를 다시 준다(XQ 8절의 재개 함정).
- **out_scales:** `always`면 gate/up을 포함해 모두 스케일을 쓴다(turboderp와 같음). 1.6.5 기본값 `auto`는 쓰지 않는다.

## 3. 보정 데이터와 expert 커버리지

- **보정 파일:** c02 250 × 2048 파일 `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/cal/c02-250x2048.safetensors`(sha256 `46bed24c…`)를 그대로 쓴다.
  - 토크나이저: Flash-Next `tokenizer.json`의 sha256이 `0997f410…`로 27B와 같다. 어휘도 248,320으로 같다.
  - c02는 처음부터 MoE용 확장(`rq-r3-calibration 2 (c01 + MoE extension)`)이다. 행 선택 규칙은 XQ와 같다.
- **expert 커버리지:** `Qwen4ExpModel.calibration_all_experts = True`(`qwen4_exp.py:306`)다.
  - Hessian을 잡는 동안에는 모든 expert를 켠다(`convert_model.py` `capture_module`, `activate_all_experts`). 그래서 **512개 expert 모두 보정 토큰 512K개 전체로 입력 Hessian을 만든다.** 라우팅 빈도와 관계없이 커버리지는 100 %다.
  - 상태를 다음 모듈로 넘기는 단계와 오차 측정은 실제 라우팅(top-10)을 쓴다(`convert_model.py:975-998`).
  - [짐작] 그 대신 드물게 라우팅되는 expert는 자기가 실제로 받는 토큰 분포가 아니라 모든 토큰의 분포에 맞춰 양자화된다. upstream의 설계 선택이다. turboderp 팩도 같은 방식이다.
  - 라우팅 빈도 자체(어느 expert가 c02에서 몇 번 선택되는지)는 이 경로가 기록하지 않는다. 필요하면 gufo Flash-Next imatrix 수집기의 기록(R3에서 c02를 만든 이유)을 다시 보거나, 시험 창에서 router 출력 히스토그램을 따로 잰다(선택).
- **gate/up Hessian 공유:** 모든 expert가 같은 입력을 보므로 gate/up은 Hessian 하나를 공유하고, down은 expert마다 하나씩이다. `capture_passes`(`convert_model.py:912`)는 Hessian이 GPU 여유를 넘으면 여러 번에 나눠 잡는다.

## 4. 시간, 메모리, 디스크 [모두 짐작, 시험 창에서 잼]

### 시간

- **환산 기준:** 27B 측정(XQ 10절)에서 층 하나(0.39 G 가중치, K=5)의 시간은 약 235 s였다. 그중 양자화 부분이 약 175 s였다(로드에서 첫 텐서까지 약 70 s, 상태 진행 약 20 s). 그래서 **K=5 trellis는 약 450 s / G 가중치**로 잡는다.

| 단계 | 가중치 | 추정 |
|---|---:|---:|
| n-gram 표 K=6(보정 없음, `quantize_ngram_table`, 재개 가능) | 51.2 G | 8–13 h (K=5보다 1.3–2배 느리다고 봄) |
| experts K=5, 48층 | 120.8 G | 약 15 h |
| 층마다 capture forward(모든 expert 켬, 512K 토큰 × 2 × 2.52 G ≈ 2.6 PFLOP, 약 15 TFLOPS) + Hessian + 상태 진행 | — | 층당 약 200–300 s, 합계 약 3–4 h |
| trunk K=7(GDN, attention, shared expert) | 약 2 G | 약 0.5 h |
| lm_head K=6 | 0.64 G | 약 5–10분 |
| MTP(experts 512개, 보정 없음, K=5) | 약 2.5 G | 약 20–30분 |
| 비전 K=6 | 약 0.4 G | 약 5분 |
| compile(외장 exFAT에 약 125 GB 쓰기, 약 130 MB/s) | — | 약 20분 |
| **합계** | | **약 26–33 h**(n-gram을 다시 쓰면 약 18–20 h) |

### 메모리

- 보정 상태: 250 × 2048 × 4 stream × 2560 × fp32 = **21 GB**. `ExpandStreams.forward`가 fp32 (B, T, 4, 2560)를 만든다(`modules/hyperconnections.py:56`).
- 층 하나의 BF16 가중치 약 5 GB, 원본 사본, fp32 작업 버퍼, down Hessian 512개(640² packed, 약 0.4 GB), gate/up Hessian(2560² × 4 B = 26 MB)
- 전부 합쳐 최대 **약 40–70 GB**(RSS + GTT)로 본다. 호스트 가용이 111 GB라서, 다른 모델이 올라간 창과 같이 돌리면 안 된다. `MALLOC_MMAP_THRESHOLD_=1048576`은 그대로 쓴다.

### 디스크(외장 가용 420 GB, 루트 156 GB는 쓰지 않음)

| 항목 | 크기 |
|---|---:|
| 출력 팩(본체 약 84 + n-gram 39 + 기타) | 약 125 GB |
| 작업 디렉터리 qtensors | 약 86 GB |
| 체크포인트(`ckpt` + `ckpt_old`) | 42 GB |
| `-ngf`로 받은 n-gram 파일을 출력에 복사(B안) | 39 GB |

### 창 사이 작업 디렉터리 (감독 결정 필요)

- 27B 방식은 재개할 창마다 앞 창의 `ckpt/`와 `qtensors/`를 통째로 복사한다(`run_convert.py --copy-work`). Flash-Next에서는 qtensors가 86 GB까지 커지고, 창 10개 정도면 복사량 합계가 약 500 GB, 창마다 최대 10분이다. 동시에 들고 있는 양도 외장 420 GB를 넘을 수 있다.
- 방법 세 가지:
  1. **(권장) `window.py`에 쓰기 가능한 마운트(예: 계획의 `mounts_rw`)를 하나 더한다.** 작업 디렉터리 하나를 모든 창이 이어 쓴다. 복사가 없고, 디스크는 약 250 GB면 된다. 공유 파일 변경이라 감독 결정이다.
  2. `ckpt`만 복사하고(창마다 21 GB, 약 3분) qtensors는 각 창의 receipt에 둔다. 마지막 compile 창에서 한 번만 모든 qtensors(86 GB, 약 11분)를 복사한다. `run_convert.py`에 `--copy-work` 변형(qtensors 제외)을 더해야 한다. 최대 사용은 약 330–370 GB로 빠듯하다.
  3. 아주 긴 창 몇 개로 줄인다. 예: n-gram 표 창 하나(약 10 h), 본체 창 두 개(각 약 10 h). 데스크톱 영향은 사용자가 이미 받아들였다.
- **n-gram 표:**
  - A안: 우리가 직접 양자화한다. `util/convert_ngram.py -ngb 6 --resume`, 또는 변환 첫 단계에서.
    - 출처가 깨끗하고 1.6.5 코덱(CS_HEURISTIC 등)과 맞는다.
    - 약 8–13 h를 쓴다. 보정 없는 양자화라 c02와 관계없다.
  - B안: turboderp `5.05bpw_h6_ng6`의 `ngram_embedding.safetensors`(39.0 GB, Qwen Community License)를 받아 `-ngf`로 준다.
    - 약 10 h를 아낀다. 1.6.5는 키 `…ngram_embedding.trellis`를 받는다(`ngram.py:624-626`).
    - 1.4.4 코덱과 1.6.5 코덱이 같은 표를 만드는지는 모른다. 네트워크로 39 GB를 받아야 한다.
  - **권장:** A안. 시험 창에서 `--limit_rows`로 속도를 먼저 재고, 길면 B안으로 바꿀지 다시 묻는다.

## 5. 시험 창 계획 (감독 허락 필요, 약 1.5–2 h)

1. **n-gram 속도:** `util/convert_ngram.py -i <원본> -o /receipt/ng-trial.safetensors -ngb 6 --limit_rows 2000000 --verify_rows 16384`
   - 대상: 32만 행 중 2백만 행(0.6 %)
   - 행/s를 재서 전체 시간을 다시 계산하고, verify 오차를 기록한다.
2. **본체 경로:** 실제 변환을 시작하려면 n-gram 표 전체가 먼저 있어야 한다(`prepare_ngram_table_for_conversion`이 첫 단계). 그래서 시험은 둘 중 하나로 한다.
   - (a) B안 파일을 받아 `-ngf`로 주고, `--max_module 6`(embed, hc_expand, 층 0, PLE, 층 1–3: GDN 세 개 + QSA attention 하나)까지 새 작업으로 돌린다. 재개도 한 번 확인한다.
   - (b) A안이면 n-gram 표 창을 먼저 끝낸 뒤 같은 방법으로 시험한다.
   - 확인할 것:
     - 전략 출력(experts 5, trunk 7, indexer 5, head 6, MTP 5)
     - capture pass 수, 층당 시간, RSS/GTT 최대
     - 비유한 행 0개, 층별 rfn/sqnr
     - qtensors 헤더가 turboderp 5.05 같은 층과 같은지(`check_pack.py --partial`. out_scales 판정도 함께)
3. 결과로 본 양자화 창 수와 디스크를 다시 계산한다.

## 결정이 필요한 것

1. 창 사이 작업 디렉터리: `window.py`에 쓰기 가능한 마운트를 더할지(권장), ckpt만 복사할지, 긴 창으로 갈지
2. n-gram 표: A안(직접 양자화, 약 10 h 추가)과 B안(turboderp 파일 39 GB 받기) 중 어느 것인지
3. 비트: turboderp 5.05와 같게 갈지(`-hq`로 trunk 7, MTP 5, NG6). 사용자 결정 "experts 5.0"은 이 설정에서 만족된다.
4. 시험 창 순서와 시각
