# XQ 결과: EXL3 직접 양자화 (27B 5.0 bpw, 2026-10-30)

상위 문서: [XQ 브리프](XQ-BRIEF.md), [PLAN](PLAN.md), [W-COMMON](../rotated-quant/W-COMMON.md), [W8 결과](../rotated-quant/EXL3-RESULTS.md). 이 문서는 XQ가 쓴다.

상태: **1단계(조사)와 2단계(시험 양자화 창 `xq-t01`)가 끝났다.** 결과는 아래 "8. 시험 양자화 결과"에 있다. **3단계(본 양자화)도 끝났다(2026-10-31, 10절).** 팩은 `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/qwen-3.8-27b-exl3-5.00-c02/`에 있고, `check_pack` 네 검사를 모두 통과했다. 4단계(품질 대조)는 감독 허락을 기다린다.

사용자 결정(2026-10-30): **본 양자화는 데스크톱을 쓰지 않는 시간에만 돌린다.** gfx 링 리셋이 데스크톱에 영향을 주기 때문이다. 시작 시각은 감독이 정해 준다.

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
# 재개: --out_scales always와 -cpi를 반드시 다시 준다(8절 "재개 함정"). 9절 창 계획은 -cpi 0을 쓴다
python3 -I -m exllamav3.conversion.convert_model -w /xq/work/qwen-3.8-27b-exl3-5.00-c02 -r \
  --out_scales always -cpi 0
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

## 8. 시험 양자화 결과 (창 `xq-t01`, 2026-10-30 13:05–13:23 KST)

- **도구:** gufo `e2517c88`의 `tools/exl3_quant/`
  - `c02_to_caldata.py`(+ `test_c02_to_caldata.py`)
  - `run_convert.py`(변환기 실행, 메모리 표본)
  - `compare_qtensors.py`(헤더 비교)
- **창 계획:** `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-XQ-TRIAL.json`(LOCKS.py 다시 실행함)
  - 이미지: `ai-tools/exl3-eval:1.6.5-rocm100` `sha256:03a29041b45d`
  - 마운트: 원본, `cal`, turboderp 5.00. 모두 ro다.
- **보정 파일:** `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/cal/c02-250x2048.safetensors`
  - sha256 `46bed24c4a379a2c07b2c675aeffb4941d729a4de848d90c42a235cf07a3b4f1`, 4,096,136 B
  - manifest `c02-250x2048.json`에 행별 청크 id, 할당, 입력 sha가 있다. c02 `chunks.bin` sha 검사를 통과했다. 최대 토큰 id는 248,069다.
- **창 출력은 외장 디스크에 있다:**
  - 실제 위치: `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/windows/xq-t01/`
  - `window.py`는 출력이 `tmp/test-results/gufo-upstream-port-20261023/` 아래여야 하고, `/receipt` 말고는 쓸 수 있는 마운트가 없다. 그래서 `…/gufo-upstream-port-20261023/xq-ext`를 외장으로 가는 symlink로 두었다.
  - 루트 디스크에는 창 로그 `xq-t01.xq-trial.window.log`만 있다.
- **창 상태: `FAILED_no_fallback`(health), `RESULT` = COMPLETE.** 세 단계는 모두 exit 0으로 끝났다.
  - health가 실패한 이유는 커널 로그에 GPU 오류가 새로 생겼기 때문이다. 13:10:02와 13:10:07에 `ring gfx_0.0.0 timeout`이 났고, 각각 `Ring gfx_0.0.0 reset succeeded`와 "device wedged, but recovered through reset"가 따랐다.
  - 두 번 모두 해당 프로세스는 **`kwin_wayland` pid 3289**(데스크톱 compositor)다. 리셋된 것은 그래픽 링뿐이고 compute 링이 아니다.
  - 시각은 층 0 양자화 도중(13:05:55–13:10:33)이다. 변환은 멈추지 않았고, 층 0의 오차가 다른 층과 같은 수준이다.
  - [짐작] trellis 인코딩 커널이 GPU를 오래 점유하는 동안 compositor의 gfx 제출이 2 s 제한을 넘은 것으로 보인다. 같은 일이 본 양자화(약 5시간) 내내 날 수 있다. 판단은 감독에게 맡긴다.
  - 원 기록: `KERNEL-DELTA-0001.txt`, `FENCE-000{1,2}.*`, `WINDOW.json`

### 층별 결과

| 모듈 | 종류 | 시간 (s) | bpw | rfn | sqnr (dB) | proxy_err 범위 | RSS (converter 표시) |
|---|---|---:|---:|---:|---:|---|---:|
| embed_tokens | 비양자화 | 15.5 | 16 | — | — | — | 13.0 GB |
| layers.0 | DeltaNet | 283.2 | 5.02 | 0.00320 | 51.71 | 6e-6–1.9e-4 | 14.4 GB |
| layers.1 | DeltaNet | 230.5 | 5.02 | 0.00234 | 54.51 | 7.6e-5–3.1e-4 | 14.4 GB |
| layers.2 (재개) | DeltaNet | 255.2 | 5.02 | 0.00347 | 51.97 | 1.0e-4–4.8e-4 | 14.2 GB |
| layers.3 (재개) | full attention | 235.8 | 5.00 | 0.00245 | 54.17 | 1.0e-5–6.4e-4 | 14.4 GB |

- 비유한(non-finite) 행은 0개다. `Calibration data: /models/cal/c02-250x2048.safetensors`로 250 × 2048이 로드됐다.
- **전략 출력이 turboderp 5.00과 같다.** 본체 400개 선형이 5 bpw, lm_head가 6 bpw, MTP 8개가 4 bpw이고, `Final bitrate (excluding head): 5.00`이다.
- 체크포인트 저장(상태 9.8 GB, exFAT 외장)은 한 번에 7–12 s다.
- **헤더 비교** `compare-td5.json`: **PASS**
  - 대상: embed와 층 0–3, 텐서 129개
  - 키, 모양, dtype이 turboderp 5.00과 모두 같다.
  - 층별 바이트 합도 같다: 층 0–2는 각 240,503,256 B, 층 3은 232,891,420 B, embed는 2,542,796,800 B.
  - trellis K는 모두 5다.

### 메모리 (`trial-a.summary.json`, `trial-b.summary.json`, 2 s 표본)

| 항목 | trial-a (새 작업) | trial-b (재개) |
|---|---:|---:|
| 변환기 프로세스 트리 RSS 최대 | 18.2 GB | 18.2 GB |
| GPU GTT 사용 최대(장치 전체) | 14.9 GB | 14.7 GB |
| GPU VRAM 사용 최대(장치 전체, 4 GiB carve-out) | 0.70 GB | 0.69 GB |
| 컨테이너 cgroup `memory.peak` | 63.3 GB | 65.0 GB |

- cgroup 값에는 원본 safetensors를 읽고 체크포인트를 쓴 페이지 캐시가 들어 있다. 실제 점유는 RSS + GTT ≈ **33 GB**다.
- `MALLOC_MMAP_THRESHOLD_=1048576`을 준 상태에서 층마다 RSS가 늘지 않았다(14.2–14.4 GB). 그래서 issue #460의 RSS 증가는 막힌 것으로 본다.
- 창 앞뒤 호스트 가용 메모리는 119.8 / 118.7 GB다. X-A의 nix 컨테이너가 함께 돌았다.

### 재개 함정 (본 양자화 명령에 반영)

- **`--out_scales`는 재개할 때 저장값으로 복원되지 않는다.** 이 값은 일시 인자다(`convert_model.py:316`, 명령줄에서 매번 다시 읽는다).
  - trial-b는 `--out_scales` 없이 재개해서 기본값 `auto`로 돌았다.
  - 그 결과 층 2의 `mlp.up_proj`/`gate_proj`는 출력 스케일 없이(`..` 표시) 양자화됐다. 층 3은 auto가 스케일을 켰다(`o.`).
  - 모양과 바이트 수는 같다. 그래서 헤더 비교로는 이 차이를 잡을 수 없다.
  - 따라서 이 작업 디렉터리의 층 2는 turboderp 설정과 다르다. 시험 결과로만 쓰고 본 양자화에는 이어 쓰지 않는다.
- `-cpi`도 argparse 기본값이 120이라, 재개하면 저장값(0)을 덮어쓴다(`convert_model.py:90`, 경고가 출력됨).
- 그래서 재개 명령에는 항상 `--out_scales always -cpi <값>`을 붙인다(6절 명령을 고쳤다).

### 본 양자화 추정 (측정 기반, 일부 짐작)

- **층당 시간:** 평균 251 s(230–283 s)다. DeltaNet과 attention 층의 차이는 작다.
- **64층:** 약 4.5시간이다.
  - 그 밖에 lm_head(K=6, 1.27 G 가중치로 층 하나의 약 3배)가 약 15–25분이다[짐작].
  - MTP(보정 없음)는 약 5분, compile(외장에 19 GB 쓰기)은 약 10분이다[짐작].
  - **합계 약 5–5.5시간**이다. 1단계 추정 4–8시간의 아래쪽이다.
- **창 나누기:** 재개 비용은 상태 로드 약 10 s다. 그래서 2시간 창 3개(층 약 28개씩, 마지막 창에 lm_head, MTP, compile) 또는 3시간 창 2개로 나눌 수 있다.
- **메모리:** 최대 약 33 GB(RSS 18 + GTT 15)다. lm_head 단계에서는 +5–8 GB를 예상한다[짐작].
- **디스크(외장):**
  - 작업 디렉터리: 상태 2벌 19.6 GB + qtensors 약 19 GB
  - 출력: 약 19 GB
  - **합계 약 60 GB**다(외장 가용 525 GB). 루트에는 창 로그만 쓴다.
- **시험 산출물:** `…/windows/xq-t01/receipt/work/`(22.9 GB: `ckpt` 9.8, `ckpt_old` 9.8, `qtensors` 3.3)이다. 지우지 않았다. 감독이 정한다.

## 9. 본 양자화 계획 (3단계, 준비 완료, 시작 대기)

- **창 계획:** `~/projects/ai.tools/test/experiments/gufo_upstream_port_20261023/PLAN-XQ-FULL.json`(초안, LOCKS.py 다시 실행함)
  - 도구: gufo `be695dd0`의 `tools/exl3_quant/`
    - `run_convert.py --copy-work`
    - `check_pack.py`
    - `provenance.py`
  - 이미지와 설정은 시험 창과 같다: `-b 5.0 -hb 6 -mb 4 -vb 16 --out_scales always -cb mul1`, c02 250 × 2048
- **작업 디렉터리:** 새로 만든다. 시험 창의 작업 디렉터리는 층 2가 `auto`로 오염됐으므로 이어 쓰지 않는다.
- **재개 규칙:** 재개할 때마다 `-i /models/hfbf16/ -w /receipt/work -r --out_scales always -cpi 0`을 준다.
  - `-i`는 `args.json`과 같은 값이다. 넣는 이유는 window.py가 argv에 나오는 `/models/<키>/` 경로만 마운트하기 때문이다. 이게 없으면 분할 창에 원본이 마운트되지 않는다.
  - `-cpi 0`은 모듈마다 체크포인트를 남긴다(모듈당 약 10 s, 64층이면 약 10분).
  - 그러면 `--max_module`로 멈출 때 잃는 작업이 없다.
- **창 사이 작업 디렉터리 전달:**
  - window.py 창이 쓸 수 있는 곳은 각자의 `/receipt`뿐이다. 그래서 재개 창은 앞 창의 `args.json`, `ckpt/`, `qtensors/`를 읽기 전용 마운트에서 자기 `/receipt/work`로 복사한 뒤 시작한다.
  - `run_convert.py --copy-work`가 크기를 검사하고, 복사한 바이트와 시간을 기록한다.
  - 공유 파일(`window.py`)은 고치지 않았다.
- **외장 출력 경로(xq-ext symlink, 감독 승인 2026-10-30):** `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/xq-ext`는 `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/windows`를 가리키는 symlink다.
  - 창의 `--out`은 `…/xq-ext/<창 이름>`으로 준다. 그러면 receipt 전체가 외장에 생기고, 루트 디스크에는 창 로그 `<창>.<model>.window.log`만 남는다.
  - `window.py`의 디스크 하한(150 GiB)은 여전히 루트 기준으로 검사된다.
  - 창 이름은 계획의 마운트 경로에 고정돼 있다(`xq-f1`, `xq-f2`, `xq-f3`). 다른 이름으로 돌리면 계획을 고쳐야 한다.

- **확정(감독 2026-10-30):** 세 창(`xq-f1` → `xq-f2` → `xq-f3`)을 이어서 돌린다. 시작 시각은 사용자가 데스크톱이 비었다고 알린 뒤 감독이 정한다.
- **health FAILED 판정 규칙(감독 2026-10-30):**
  - **통과(데스크톱 링 리셋 있음):** 세 조건을 모두 만족하면 통과로 본다. KERNEL-DELTA는 기록으로 남긴다.
    - 원인이 compute가 아닌 프로세스(kwin 등)의 gfx 링 리셋뿐이다.
    - 변환기가 exit 0으로 끝났다.
    - `check_pack.py`의 네 검사가 모두 통과했다.
  - **실패(멈추고 보고):**
    - compute 링이나 우리 프로세스가 리셋됐다.
    - 장치가 멈춘 뒤(wedge) 복구되지 않았다.
    - `check_pack.py`가 실패했다.

| 창 (`--model`, `--out …/xq-ext/<이름>`) | 범위 | 예상 시간 | 창 seconds | 쓰는 양(외장) |
|---|---|---:|---:|---:|
| `xq-f1` | 새 작업. 임베딩과 층 0–27(`--max_module 28`) | 약 2.0 h | 9,600 | 작업 디렉터리 약 29 GB |
| `xq-f2` | xq-f1 작업 복사(약 19 GB, 1–3분) → 층 28–55(`--max_module 56`) | 약 2.1 h | 10,200 | 약 36 GB |
| `xq-f3` | xq-f2 작업 복사(약 26 GB) → 층 56–63, norm, lm_head(K=6), MTP(K=4), compile → `check_pack.py` → `provenance.py`(원본 52 GB와 출력 19 GB 해시) | 약 1.5–2 h | 12,000 | 약 46 GB + 팩 19 GB |
| 대안 `xq-fall` | 한 창에서 전부 실행하고 점검과 provenance까지 | 약 5.5–6 h | 27,000 | 약 58 GB |

- 시간은 시험 창의 층당 평균 251 s를 바탕으로 했다. lm_head와 MTP 시간은 짐작이다.
- 외장 최대 사용은 세 창의 작업 디렉터리가 함께 있을 때 약 140 GB다(시험 23 GB 포함). 외장 가용은 525 GB다.
- **사후 점검** `check_pack.py`(xq-f3 안에서 실행, 결과 `receipt/run/check-pack.json`). 모두 통과해야 한다:
  1. 헤더: turboderp 5.00의 모든 텐서(2,426개: 본체, lm_head, MTP, 비전 BF16, 임베딩)가 우리 팩에 같은 dtype과 모양으로 있고, 그 반대도 성립해야 한다.
  2. out_scales: 선형 409개(본체 400, lm_head, MTP 8) 각각에서 출력 스케일이 있는지가 turboderp와 같아야 한다(turboderp는 모두 있음).
     - 판정 기준: `svh` 크기의 max/min > 1.01이면 스케일이 있다. 스케일이 없으면 `svh`는 부호뿐이라 |svh| ≈ 1이다.
     - 이 판정이 시험 작업의 층 2 `up_proj`/`gate_proj` 두 개만 정확히 잡아내는 것을 확인했다.
  3. `config.json`의 `quantization_config`: exl3, bits 5.0, head 6, mtp 4, out_scales always, mul1, calibration 250 × 2048
  4. sqnr: lm_head가 10 dB 이상이고(#458 증상 기준), sqnr가 10 dB 아래인 모듈이 하나도 없어야 한다.
  - 같은 스크립트를 turboderp 대 turboderp로 돌려 보니 헤더 2,426개, out_scales 409/409, config가 모두 통과했다(sqnr는 요약이 없어 해당 없음).
- **창 뒤(가벼움):**
  - 팩을 `…/windows/xq-f3/receipt/out`에서 `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/qwen-3.8-27b-exl3-5.00-c02/`로 옮긴다. 같은 파일 시스템 안의 rename이라 복사가 아니다.
  - 그다음 4단계(품질 대조) 창 계획을 쓴다.
- **정리:**
  - 시험 작업 디렉터리 `…/windows/xq-t01/receipt/work/`(22.9 GB)는 본 양자화가 끝난 뒤 지운다(감독 승인 2026-10-30).
  - `xq-f1`/`xq-f2`의 작업 디렉터리는 xq-f3이 통과하면 필요 없다. 그래도 지우기 전에 감독에게 묻는다.

## 10. 본 양자화 결과 (2026-10-31 01:55–06:40 KST)

- **창 세 개를 이어서 돌렸다.** 계획은 `PLAN-XQ-FULL.json`이다.
  - 출발 전에 이미지 태그 `ai-tools/exl3-eval:1.6.5-rocm100`이 바뀌어 있었다. X-A가 같은 이미지를 라벨만 바꿔 다시 빌드해서, 지금은 `sha256:10ccf4b7b405`, 라벨 `f26f800d`다.
  - `docker history`로 보면 레이어는 시험 이미지 `03a29041b45d`와 같은 2026-10-29 캐시 레이어다. 다른 것은 `GUFO_REVISION` ARG/LABEL뿐이다.
  - 그래서 계획의 `gufo_revision`을 `f26f800d`로 바꾸고 LOCKS를 다시 만들었다.
  - 첫 xq-f1 실행은 라벨 검사에서 멈췄다. 컨테이너도 출력도 없었고, 로그는 `xq-f1-preflight-label-mismatch.window.log`로 남겼다.

| 창 | 범위 | wall | 작업 복사 | 상태 | GPU 커널 로그 |
|---|---|---:|---|---|---|
| `xq-f1` | 새 작업, 임베딩과 층 0–27 | 6,742 s | — | `FAILED_no_fallback`(health), RESULT COMPLETE, rc 0 | `kwin_wayland` pid 3289의 `gfx_0.0.0` 링 timeout과 reset 두 번(epoch 1793381102, 1793385422). 둘 다 "reset succeeded / recovered through reset" |
| `xq-f2` | 층 28–55 | 6,909 s | 19.7 GB, 110 s | COMPLETE, health PASS | 오류 없음 |
| `xq-f3` | 층 56–63, norm, lm_head, MTP, compile, check_pack, provenance | 2,937 s(변환) + 9 s + 288 s | 26.4 GB, 208 s | COMPLETE, health PASS | 오류 없음 |

- **판정:** xq-f1의 health FAILED는 감독이 정한 규칙(9절)에 따라 **"통과(데스크톱 링 리셋 있음)"**으로 본다.
  - 원인은 compute가 아닌 프로세스(kwin)의 gfx 링 리셋뿐이었다. compute 링과 우리 프로세스는 리셋되지 않았고, 장치는 복구됐다.
  - 변환기는 rc 0으로 끝났고, check_pack은 4/4 통과했다.
  - 원 기록은 `…/windows/xq-f1/KERNEL-DELTA-0001.txt`와 `FENCE-*`다.
- **결정성:** xq-f1의 층 0–1 값(rfn 0.003198/0.002336, sqnr 51.709194/54.507862)이 시험 창과 소수점 끝자리까지 같다.
- **층별 품질:** sqnr는 층이 깊어질수록 내려간다(층 0 51.7 dB → 층 20 45.3 → 층 27 43.3 → 층 63 40.9). 가장 낮은 것은 층 51의 39.18 dB다.
  - lm_head(K=6): 565 s, rfn 0.006647, **sqnr 43.24 dB**. #458 증상은 없다.
  - MTP 1층(K=4, 보정 없음): 48 s
- **변환 시간 합계:** 16,587 s(4.6 h). 작업 복사 5.3분과 provenance 해시 4.8분은 별도다. 층당 평균은 약 235 s였고, 재개 직후 첫 층만 약 350 s였다(캐시가 차 있지 않은 상태로 보인다[짐작]).
- **메모리 최대:**
  - 층 단계: 변환기 RSS 18.5 GB, GTT 14.9 GB
  - lm_head 단계(xq-f3): **RSS 36.4 GB, GTT 38.2 GB**, cgroup peak 79.7 GB(페이지 캐시 포함)
  - 호스트 가용은 창마다 시작 때 111 GB였다.
- **check_pack** `…/windows/xq-f3/receipt/run/check-pack.json`: **PASS**
  - headers: 텐서 2,426개(그룹 239)가 turboderp 5.00과 dtype, 모양이 모두 같다.
  - out_scales: 409/409. turboderp와 같이 모두 출력 스케일이 있다. 재개 오염은 없다.
  - config: `{"quant_method": "exl3", "version": "1.6.5", "bits": 5.0, "head_bits": 6, "calibration": {"rows": 250, "cols": 2048}, "out_scales": "always", "codebook": "mul1", "mtp_bits": 4}`
  - sqnr: lm_head 43.24 dB. 10 dB 아래인 모듈은 없다(65개 중 최소는 층 51의 39.18 dB).
- **팩:** `/run/media/m2on/T5 EVO/AI/model/xq-exl3-20261030/qwen-3.8-27b-exl3-5.00-c02/`(19 GB, 파일 20개)
  - xq-f3 receipt/out에서 같은 디스크 안에서 rename으로 옮겼다.
  - 샤드 세 개의 바이트 크기는 turboderp 5.00과 같다(8,509,808,537 / 8,589,756,567 / 2,802,114,925).
  - sha256 앞 12자리: `334301cf5e9c`, `176a9b9e2fea`, `f4c4bf536b15`(전체 값은 PROVENANCE.json)
  - `PROVENANCE.json`에 원본 파일 33개의 sha256, 보정 파일, 세 실행의 argv/env/버전(exllamav3 1.6.5+rocm10.0.torch2.13.0, torch 2.13.0+rocm10.0.0, HIP 7.15.26333, gfx1151)/peak, 이미지 wheel 목록이 있다. `CALIBRATION.json`도 함께 있다.
  - PROVENANCE의 `output.dir`은 옮기기 전 경로(`/receipt/out`)로 적혀 있다.
- **정리:**
  - 시험 작업 디렉터리 `…/windows/xq-t01/receipt/work/`(22.9 GB)는 지웠다(감독 승인).
  - 남은 작업 디렉터리: xq-f1 29 GB, xq-f2 35 GB, xq-f3 33 GB, 합계 97 GB. 팩이 완성됐으니 필요 없지만, 지우기 전에 감독에게 묻는다.
  - 외장 가용은 326 GB다.

## 11. 품질 대조 계획 (4단계, 감독 허락 필요)

- **수집기:** W8과 같은 `tools/qwen27b/rq_quality/exl3/collect.py`를 쓴다(마지막 변경 `43425c9f`, 이후 바뀌지 않음). 이미지는 같은 레이어다(라벨 `f26f800d`).
- **코퍼스와 기준:** R0와 같다. a01 8창, prefill 32,760 위치와 decode 4,096 위치(prefix 2048, 창마다 512)를 gufo BF16 `rq0-bf16`에 대고 잰다. 같은 위치를 EXL3-BF16 대조(`w8q-bf16`)에도 대고 잰다.
- **창 하나 `PLAN-XQ-QUALITY.json`(약 25–30분):**
  1. `xq-5.00-c02`: 우리 팩의 prefill(약 210 s)과 decode(약 390 s)
  2. `td5-rerun`: turboderp 5.00을 같은 창에서 prefill과 decode로 다시 잰다. W8의 `w8q-a01/w8q-5.00` 기록과 레코드가 같은지 보는 재현 확인이고, 같은 조건의 paired 대조군이다.
- **지표**(`metrics.py score/compare`, CPU, 창 뒤):
  - KL(토큰과 블록 95 % 구간), ΔNLL, top-1, 뒤집힘(KL > 1 포함), 위치별 NLL 꼬리, 카테고리별(en_prose, ko_prose, code, chat 등)
  - 우리 팩과 turboderp는 같은 위치에서 paired 차이를 낸다. 판정은 "같거나 낫다"가 신뢰구간으로 보이는지다.
  - KL_256이 복사 실패를 숨길 수 있으므로(W8 3.50) ΔNLL과 꼬리를 반드시 같이 본다.
- **나쁘게 나오면 볼 원인 후보:**
  - 보정 데이터(c02의 en_tool 비중 36 %, 한국어 20 %)
  - 도구 버전(1.4.2 → 1.6.5). 분리하려면 1.6.5 + 내장 코퍼스로 다시 양자화하는 선택 대조가 필요하다(약 4.6 h).
  - 카테고리별 결과로 어느 쪽인지 좁힌다.

## 남은 문제와 위험

- gfx1151에서 층 0–3 변환은 동작했다(8절). 다만 같은 창에서 compositor(`kwin_wayland`)의 gfx 링이 두 번 timeout으로 리셋됐다. 본 양자화 중에 이 일이 반복될지, window.py의 health 판정을 어떻게 볼지는 감독이 정한다.
- lm_head(hipBLASLt #458 경로)는 본 양자화에서 sqnr 43.24 dB로 정상이었다(10절).
- turboderp 팩(1.4.2)과 우리 팩(1.6.5)은 도구 버전이 다르다. 품질 차이를 보정 데이터 때문이라고 단정하려면 위 선택 대조가 필요하다.
- 상태 크기는 체크포인트 9.8 GB로 확인했다(fp32 250 × 2048 × 5120 = 10.5 GB, 즉 9.8 GiB).
- `run_convert.py`의 버전 조회가 비어 있었다. `exllamav3.__version__`이 없어서다. 그래서 `importlib.metadata`로 고쳤다. 이번 창의 버전은 이미지 provenance(v1.6.5 wheel)로 확인한다.
