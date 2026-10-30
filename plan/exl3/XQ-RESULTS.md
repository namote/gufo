# XQ 결과: EXL3 직접 양자화 (27B 5.0 bpw, 2026-10-30)

상위 문서: [XQ 브리프](XQ-BRIEF.md), [PLAN](PLAN.md), [W-COMMON](../rotated-quant/W-COMMON.md), [W8 결과](../rotated-quant/EXL3-RESULTS.md). 이 문서는 XQ가 쓴다.

상태: **1단계(조사)만 끝났다.** 컨테이너, 빌드, 큰 파일 해시, 복사는 하지 않았다(W9 I1 ABBA 창이 도는 중). 2단계(시험 양자화)는 감독 허락을 기다린다.

기준 소스: upstream `turboderp-org/exllamav3` v1.6.5 `cdc785db89e263e65e5849cb53e1ada156411414`. 얕은 클론 `/home/m2on/projects/gufo/build/w9-src/exllamav3/`을 읽었다. 아래 `convert_model.py:N`은 `exllamav3/conversion/convert_model.py`의 줄 번호다.

## 요약

- **보정 데이터 입력:** `-cd/--cal_data <file.safetensors>`를 쓴다.
  - 형식은 `input_ids` int 텐서 (rows, cols)이고, 선택으로 `lengths` (rows,)를 둔다.
  - 앞에서부터 `cal_rows`개 행을 쓰고, 행마다 `cal_cols`로 자른다(`calibration_data.py:102-130`).
  - c02를 2048토큰 250행으로 바꾸면 turboderp 팩과 같은 크기(250 × 2048)가 된다. 그러면 비교에서 보정 *내용*만 달라진다.
- **turboderp 5.00 팩은 v1.4.2로 만들었다.** 설정은 아래와 같다(팩 `config.json`의 `quantization_config`).

  ```json
  {"quant_method": "exl3", "version": "1.4.2", "bits": 5.0, "head_bits": 6,
   "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always",
   "codebook": "mul1", "mtp_bits": 4}
  ```

  - 본체 64층의 선형 400개가 모두 K=5이고, lm_head는 K=6, MTP 8개는 K=4다.
  - 비전 타워 333개 텐서는 BF16 그대로다. `-hq` 흔적은 없다.
  - v1.6.5의 기본값은 이 팩과 두 군데 다르다. `--out_scales`는 `auto`이고(1.4.2는 `always`), 비전은 6비트다(Qwen3-VL 비전의 `default_vision_bits`). 그래서 같게 맞추려면 `--out_scales always -vb 16`을 명시한다(아래 "명령").
- **ROCm에서 변환해 본 기록은 있지만 gfx1151 기록은 없다.**
  - 다른 사용자가 gfx1201 + ROCm 7.2.4에서 Qwen3.5-MoE 전체 변환에 성공했다(bf16 대비 PPL 0.4 % 안. issue #460).
  - 그 과정에서 나온 lm_head 손상(hipBLASLt fp32 GEMM 버그)은 PR #458에서 고쳤고, v1.6.5에 들어 있다.
  - 남은 위험은 호스트 RSS가 층마다 늘어나는 문제다(issue #460 2번). 통합 메모리 128 GB에서는 치명적일 수 있어 환경변수로 막는다.
- **예상[짐작]:**
  - 시간: 본 양자화 4–8시간. 시험 창에서 층당 시간을 재서 다시 정한다.
  - 메모리: 최대 25–35 GB(호스트와 GPU 합계)다. 다만 지금 호스트는 다른 창 때문에 106 GB를 쓰고 있다. 그래서 다른 모델이 올라간 창과 같이 돌릴 수 없다.
  - 디스크: 외장에 약 65 GB를 쓴다. 루트는 쓰지 않는다.

## 1. `convert.py` 인자 (v1.6.5)

출처: `doc/convert.md`, `convert_model.py:73-107`(argparse), `convert_model.py:264-290`(기본값과 재개 시 덮어쓰기 규칙).

| 인자 | 기본값 | 이번 작업 | 비고 |
|---|---|---|---|
| `-i/-o/-w` | — | 원본(ro), 외장 출력, 외장 작업 디렉터리 | `-w`에 체크포인트(`ckpt/`, `ckpt_old/`)와 `qtensors/`가 생긴다 |
| `-b/--bits` | — | `5.0` | 재개할 때 바꿀 수 없다 |
| `-hb/--head_bits` | 6 | `6` | lm_head |
| `-mb/--mtp_bits` | 4 | `4` | MTP는 보정 없이 양자화한다(`doc/convert.md`) |
| `-vb/--vision_bits` | 0 = 아키텍처 기본(Qwen3-VL 비전은 6, `architecture/qwen3_vl.py:172`) | `16` | turboderp 팩의 비전은 BF16이다 |
| `-hq` | 0 | 쓰지 않음 | dense 27B는 attention/DeltaNet의 `select_hq_bits = 2 if use_moe else 0`이라 본체에는 효과가 없다(`architecture/qwen3_5.py:353,385`). MTP만 올라간다(`qwen3_5_mtp.py:135`). turboderp MTP가 K=4로 균일하므로 끈다 |
| `--out_scales` | `auto` | `always` | 1.4.2 기본값이 `always`였다(v1.4.2 `5f3c537c` `convert_model.py:55`). `auto`는 gated MLP의 gate/up에 출력 스케일을 끈다 |
| `-cb/--codebook` | `mul1` | `mul1` | 팩 텐서의 `mul1_multiplier`가 모두 `2212286765`다 |
| `-cd/--cal_data` | 없음(내장 코퍼스) | c02 변환 파일 | v1.4.2에는 이 인자가 없었다 |
| `-cr/-cc` | 250 / 2048 | 250 / 2048 | 파일 행이 모자라면 경고하고 줄인다(`calibration_data.py:117-119`) |
| `-cpi` | 120 s | `1200`(제안) | 체크포인트마다 상태 전체(약 10.5 GB, 아래 참고)를 외장에 쓴다. 기본값이면 층마다 쓸 수도 있다 |
| `--max_module N` | 없음 | 시험용 | 모듈 N 다음에서 `sys.exit()`한다(`convert_model.py:1416-1418`). 재개 때 다시 줄 수 있는 일시 인자다(`convert_model.py:322`) |
| `-r/--resume` | — | 창 나누기 | `-w`만 주면 `args.json`과 `ckpt/job.json`에서 이어 간다 |
| `-d` | `0` | `0` | GPU 하나 |

- 그 밖에 `--recipe`, `--hessians`(YAQA), `-sh`, `-eb`, `-ngb`가 있다. 이번에는 쓰지 않는다.
- **양자화 시드:** 따로 인자가 없다. 텐서 시드는 모듈 번호다(`make_quant_args` `"seed": idx`, `convert_model.py:512`). 보정 파일과 설정이 같으면 같은 결과가 나와야 한다.
  - 다만 issue #460은 allocator 설정에 따라 결과가 조금 달라졌다고 보고한다(rfn 0.01244 대 0.01235). 같은 설정으로 두 번 돌리면 비트 단위로 같았다고 한다. 그래서 환경변수까지 `PROVENANCE`에 적는다.
- **실행 방법:** 저장소 루트의 `convert.py` 옆에는 `exllamav3/` 소스 트리가 있다. 그 자리에서 실행하면 빌드되지 않은 소스가 wheel보다 먼저 import될 수 있다[짐작: sys.path[0]가 스크립트 디렉터리라서]. 그래서 이미지의 wheel을 쓰도록 `python3 -I -m exllamav3.conversion.convert_model ...`로 실행한다. 이 모듈에는 `__main__`이 있다(`convert_model.py:1791`).

## 2. 보정 데이터 형식과 c02 변환 계획

- **upstream이 받는 형식**(`calibration_data.py:102-130`):
  - safetensors 파일에 `input_ids` (rows, cols) 텐서가 있어야 한다. `lengths`는 선택이다.
  - `lengths`가 있으면 행마다 `min(cols, lengths[i])`까지만 쓴다. 없으면 모든 행을 `cal_cols` 너비로 쓴다.
  - id가 `tokenizer.actual_vocab_size` 이상이면 실패한다.
  - BOS나 EOS는 더하지 않는다. 파일의 토큰을 그대로 쓴다.
- **내장 기본 코퍼스**(`calibration_data.py:61-99`):
  - 250행 × 2048토큰이다.
  - 섞는 비율: c4 20, code 20, multilingual 10, technical 10, wiki 50, tiny 5, random 20(가중치).
  - random 20 몫은 어휘 전체를 건드리는 무작위 토큰 행이다(c02에는 없다).
- **c02 구조**(`~/projects/ai.tools/storage/evaluations/rq-r3-calib-20261029/c02/`):
  - `chunks.bin`은 int32 LE 토큰 1,999,872개다(512 × 3,906. `tools/rotquant/calibration.py:36`). sha256은 `40d5d653…`이고 `SHA256SUMS`와 같다(manifest 기록값. 이번에 다시 해시하지 않았다).
  - 청크 순서는 시드로 섞여 있다. 0–832번은 c01이고, 그 뒤가 확장분이다.
  - chat 계열(`en_tool`, `en_chat`, `ko_chat`)은 문서가 청크를 넘어 이어진다(`documents_span_chunks: true`). 이어지는 순서는 스트림 안의 청크 번호(`<stream>-N`, `<stream>-xN`) 순이다.
  - 스트림별 번호는 모두 0..n−1로 빠짐이 없다(이번에 확인).
- **제안 변환 규칙**(도구 `tools/exl3_quant/c02_to_caldata.py`, 2단계에서 작성):
  1. 청크를 (스트림, c01/확장 구분, 스트림 안 번호) 순으로 다시 늘어놓는다. 여러 청크에 걸친 chat이 다시 이어진다.
  2. 같은 구간에서 이어지는 4청크를 한 행(2048토큰)으로 만든다. 4개가 안 되는 나머지 청크(0–3개)는 버린다. 그러면 스트림 하나로만 된 행이 모두 971개 나온다.
  3. 250행을 구간별 행 수에 비례해 나눈다(최대 나머지 방식). 결과: `en_tool` 89, `v6` 37, `ko_wiki` 6+23, `ko_web` 6+19, `ko_chat` 4+16, `code` 18, `en_chat` 18, `math` 8, `multi` 6.
  4. 구간 안에서는 seed 20261029(c02와 같은 시드)로 행을 고른다. 마지막에 행 순서를 섞는다.
  5. `input_ids`를 int64로 저장하고 `lengths`는 두지 않는다(모든 행이 2048이다). 고른 청크 id 목록, 입력 sha256, 출력 sha256은 별도 manifest에 적는다.
  - chat이 행 경계에서 잘릴 수는 있다. 내장 코퍼스도 글을 이어 붙여 2048에서 자르므로 같은 성질이다.
  - 쓰는 토큰은 c02의 26 %(512K)다. c02 전체(976 × 2048 = 2M)를 쓰면 상태가 4배(약 42 GB)가 되고, forward 시간과 체크포인트도 4배가 된다. 그래서 1차는 turboderp와 같은 크기로 한다. 더 많은 행은 결과를 본 뒤 따로 정한다.
- **평가 코퍼스와 겹치지 않음:** c02 manifest `evaluation_exclusion`에 기록이 있다. 출처에 이 저장소 파일이 하나도 없고, 확장분과 a01 `windows.bin` 사이에 공유하는 16-gram이 0개다.
- **라이선스:** v6 스트림은 "do not redistribute"다(manifest `sources.base.v6`). 보정에 쓰기만 하고 텍스트를 팩에 넣지 않으므로 로컬 사용에는 문제가 없다고 본다[짐작]. 팩을 공개 배포할 일이 생기면 감독에게 다시 묻는다.

## 3. ROCm 검증 상태

- **소스:** 양자화(trellis 인코딩) 커널에 ROCm 전용 분기가 있다.
  - RDNA LDS 64 KB 한계에 맞춘 costs-only 배치: `exllamav3_ext/quant/quantize_tiles_kernel.cuh:18-30`, `quantize.cu:98-138`
  - Hessian 누적은 Triton `_syrk_packed_kernel`이다(`modules/quant/exl3_lib/hessian_packed.py:83-100`). W8 이미지에서 Triton이 돈다는 것은 확인돼 있다.
- **upstream 이슈**(gh, 2026-10-30 조회):
  - [#460](https://github.com/turboderp-org/exllamav3/issues/460) (open): gfx1201, ROCm 7.2.4, torch 2.14.1에서 Qwen3.5-MoE 전체 변환이 bf16 대비 PPL 0.4 % 안에 들어왔다.
    - (1) `test_encode_ideal`이 mcg와 3inst에서 실패하고, mul1은 통과한다. 우리는 mul1을 쓴다.
    - (2) **convert.py RSS가 층마다 약 3.7 GB씩 는다.** glibc 힙이 남는 현상이고 `malloc_trim()`으로는 풀리지 않는다. `MALLOC_MMAP_THRESHOLD_=1048576`을 주면 RSS가 약 7 GB에서 평평하고 속도도 같다.
    - v1.6.5에 `mallopt` 수정은 없다. `exllamav3/util/memory.py:114`에는 `malloc_trim`만 있다.
  - [#458](https://github.com/turboderp-org/exllamav3/pull/458) (merged 2026-10-07, `151539c7`): ROCm hipBLASLt fp32 GEMM이 2^19행을 넘으면 값이 틀려서, 어휘가 큰 모델의 lm_head가 깨졌다(sqnr 0.02 dB). 수정은 `preapply_had_r`의 분할 실행이고, v1.6.5 `modules/quant/exl3_lib/quantize.py:440-452`에 들어 있다. 우리 어휘(248,320)와 같은 조건이다.
  - gfx1151에서 변환한 이슈는 없다(검색어 "rocm convert", "rocm quantize", "gfx1151"). gfx1151 이슈는 decode 속도에 관한 #543 하나뿐이다.
- **판단[짐작]:** gfx1151 + ROCm 10.0 wheel(gfx11-generic)에서도 mul1 경로가 돌 가능성이 높다. 다만 검증된 적은 없으므로 시험 창에서 층별 `rfn`/`sqnr`과 RSS를 본다. lm_head는 마지막 모듈이라 시험 창에서는 보지 못한다. 그래서 본 양자화 로그에서 lm_head `sqnr`가 10 dB 아래면 실패로 판정한다(#458 증상 기준).

## 4. 시간과 메모리 추정 [모두 짐작]

| 항목 | 추정 | 근거 |
|---|---|---|
| 보정 상태(CPU) | 약 10.5 GB | 250 × 2048 × 5120 × fp32. 블록 출력 `out_dtype = torch.float`(`architecture/qwen3_5.py:352,383,412,437`), `state[i] = rs.cpu()`(`convert_model.py:1656`). 참조 상태는 5행뿐이다(`num_ref_states = 5`, `convert_model.py:109`) |
| 층 하나(GPU) | 약 1–3 GB | bf16 층 약 0.84 GB, down Hessian 17408² fp32는 packed 약 0.6 GB이고 풀면 1.2 GB, LDLQ 작업 버퍼 |
| lm_head 단계 | 약 +5–8 GB | 248320 × 5120 bf16 2.5 GB, fp32 사본, 인코딩 |
| 호스트 RSS 증가 | env 없이 +수십 GB(#460의 MoE 기준 층당 3.7 GB) | `MALLOC_MMAP_THRESHOLD_=1048576`으로 막는다 |
| **최대 합계** | **25–35 GB** | 통합 메모리라 호스트와 GPU를 합친 값이다. 다른 모델이 올라간 창이 없어야 한다(지금 `free`: used 106 GB, swap 0) |
| 보정 forward | 약 45–60분 | 토큰당 약 54 GFLOP × 512K 토큰 × 2패스(capture + advance). 실효 약 20 TFLOPS(같은 장비 hipBLASLt 19–26 TFLOPS, `docs/models/qwen3.8-flash-next/EXPERIMENTS.md:364`) |
| Hessian 누적 | 약 10–20분 | 층당 Σd² ≈ 3.9 × 10⁸, 상삼각 half 입력 |
| trellis 인코딩(Viterbi) | 3–6시간 | upstream은 "27B dense 약 2시간, 빠른 GPU 하나"라고 한다(`doc/optimize.md:12`. trace와 측정 포함이고 대부분이 변환). gfx1151의 fp32/정수 ALU를 그 GPU의 1/3–1/5로 봤다 |
| **합계** | **4–8시간** | 시험 창의 층당 시간 × 64 + lm_head + MTP + compile로 다시 계산한다. convert가 3층 뒤부터 ETA를 출력한다(`convert_model.py:1237-1254`) |

## 5. 디스크

- **루트:** 쓰지 않는다(지금 가용 157 GiB). Triton 캐시도 외장 작업 디렉터리 아래에 둔다.
- **외장** `/run/media/m2on/T5 EVO`(가용 549 GiB):

  | 항목 | 크기 |
  |---|---|
  | 출력 팩 | 약 19 GiB(turboderp 5.00 실측 19 GiB, 샤드 3개) |
  | `qtensors/` | 약 19 GiB |
  | `ckpt/` + `ckpt_old/` | 상태 2벌, 약 21 GiB |
  | 보정 파일 | 4 MB |
  | **합계** | **약 60–65 GiB** |

  - 체크포인트가 상태 전체를 다시 쓴다. `-cpi 1200`이면 쓰기는 대략 6–24회 × 10.5 GB다.
- **원본:** `/run/media/m2on/T5 EVO/AI/model/originals-20261007T122040Z-101bd114/llm/qwen-3.8-27b-bf16/`(BF16 safetensors 18개, 52 GiB, `Qwen3_5ForConditionalGeneration`, hidden 5120, 64층, 3:1 DeltaNet/attention, MTP 1층, vocab 248,320). `:ro`로 마운트한다.

## 6. 명령(제안, 2단계 이후)

이미지는 `ai-tools/exl3-eval:1.6.5-rocm100`(`03a29041b45d`)이다. 의존성(torch, safetensors, tokenizers, numpy, pyyaml 등 `requirements.txt` 전부)은 이미 들어 있다. 추가 빌드는 필요 없다고 본다.

```sh
# 컨테이너 안. 마운트: 원본 :ro → /src_model, 외장 XQ 디렉터리 → /xq, gufo 트리 → /src
export MALLOC_MMAP_THRESHOLD_=1048576
export TRITON_CACHE_DIR=/xq/work/triton
python3 -I -m exllamav3.conversion.convert_model \
  -i /src_model -o /xq/qwen-3.8-27b-exl3-5.00-c02 -w /xq/work/qwen-3.8-27b-exl3-5.00-c02 \
  -b 5.0 -hb 6 -mb 4 -vb 16 --out_scales always -cb mul1 \
  -cd /xq/cal/c02-250x2048.safetensors -cr 250 -cc 2048 -cpi 1200 -d 0
# 재개
python3 -I -m exllamav3.conversion.convert_model -w /xq/work/qwen-3.8-27b-exl3-5.00-c02 -r
```

- 출력 위치: 외장 `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/`
  - `qwen-3.8-27b-exl3-5.00-c02/`
  - `work/`
  - `cal/`
- `PROVENANCE`에 적을 것: 원본 sha, wheel sha(`/opt/provenance`), 명령줄, 환경변수, 시드(모듈 번호), 시간, 최대 RSS(`/usr/bin/time -v` 또는 cgroup `memory.peak`)

## 7. 시험 양자화 계획 (2단계, 감독 허락 필요)

- **창:** GPU 창 하나, 약 45–60분(짐작)이다.
  - timing 창과 겹치면 안 된다. 다른 모델이 메모리에 올라가 있지 않아야 한다.
  - 창 계획은 `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-XQ-TRIAL.json`에 쓴다.
- **창 앞(CPU, 가벼움):**
  - `tools/exl3_quant/c02_to_caldata.py`로 `cal/c02-250x2048.safetensors`를 만든다. c02 `chunks.bin` sha256 검사를 포함한다(8 MB라 가볍다).
  - `df -h`로 디스크를 확인한다.
- **창 안:**
  1. 새 작업 `-cpi 0 --max_module 2`: 임베딩과 층 0–1이다(층 0은 DeltaNet).
  2. `-r --max_module 4`로 재개해 층 2–3을 돌린다(층 3은 full attention). 재개 경로를 확인하는 단계다.
  - 확인할 것:
    - `Calibration data:` 줄과 250행 로드
    - 층별 `proxy_err`, `rfn`, `cos`, `sqnr`
    - 층당 시간과 ETA
    - 최대 RSS와 GPU 메모리
    - 비유한(non-finite) 행이 0개
    - `qtensors`의 텐서 모양과 dtype이 turboderp 5.00 같은 층과 같은지(trellis `[…, 80]` int16, `suh`/`svh` fp16, `mul1` int32)
  - 대조로 turboderp 5.00 팩의 층 0–3 텐서 바이트 크기와 비교한다. 같은 원본이고 설정이 같으면 크기가 같아야 한다. 값은 보정이 달라서 다르다.
- **산출 크기:** 작업 디렉터리 약 12 GB(상태 2벌 + 층 4개)다. 시험 뒤에는 지우지 않고 감독에게 묻는다.
- **본 양자화(3단계):** 4–8시간이다. 체크포인트 재개로 창 2–3개(창당 약 2–3시간)로 나눌 수 있다. 외장에 약 65 GB를 쓴다.
- **품질 대조(4단계):** W8 수집기(`tools/qwen27b/rq_quality/exl3/collect.py`)로 우리 팩과 turboderp 5.00을 같은 하네스에서 paired로 비교한다.
- **선택 대조:** v1.6.5 + 내장 기본 코퍼스로 한 번 더 양자화하면, "보정(c02)"과 "도구 버전(1.4.2→1.6.5)"의 효과를 나눌 수 있다. 본 양자화와 같은 시간과 디스크가 든다. 우리 팩이 turboderp와 다를 때만 제안한다.
  - `--out_scales auto`(1.6.5 기본) 변형도 원인 후보로 남겨 둔다.

## 남은 문제와 위험

- gfx1151에서 변환을 검증한 기록이 없다. 시험 창에서 확인한다.
- turboderp 팩(1.4.2)과 우리 팩(1.6.5)은 도구 버전이 다르다. 품질 차이를 보정 데이터 때문이라고 단정하려면 위 선택 대조가 필요하다.
- 상태 dtype(fp32)은 소스를 보고 짐작했다. 시험 창의 RSS로 확인한다.
- 시간 추정은 다른 GPU의 수치에서 비율로 옮긴 것이라 오차가 크다.
