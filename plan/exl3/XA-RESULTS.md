# X-A 결과: EXL3 형식 명세, CPU 참조, 27B 정확성 경로 (진행 중, 2026-10-30)

상위 문서: [X-A 브리프](XA-BRIEF.md), [EXL3 도입 계획](PLAN.md). 이 문서는 X-A(W8)가 쓴다.

## 상태

- **X0 완료.** CPU 부분 `f26f800d`, upstream 비트 대조 창 `xa-ref-a01` 통과(2026-10-30).
- X1 매핑 완료(`04ca9c54`), BF16 GGUF 변환 완료(4.00, 5.00). 정확성 창 `PLAN-EXL3-X1.json` 초안(GPU 순서상 W9 뒤).

## X0: 형식 명세와 CPU 참조

- **명세:** [docs/EXL3-FORMAT.md](../../docs/EXL3-FORMAT.md). 기준은 upstream exllamav3 v1.6.5 `cdc785db`이고, 근거 파일:줄을 달았다.
- **코드:** `src/core/quant/exl3/`
  - `format.{hpp,cpp}`: trellis 상태 추출(정수 K와 반정수 K), 코드북 3종(3INST, mcg, mul1, FP16 RNE를 double로 재현), 16×16 타일 순열, 행렬 dequant(전체와 열 구간), 128점 Hadamard, FP64 참조 GEMV.
  - `checkpoint.{hpp,cpp}`: 모델과 무관한 safetensors 팩 리더. gufo `src/core`에는 공용 safetensors 리더가 없어(모델별 리더만 있음) 새로 만들었다.
- **CPU 자체 검사** `quant_exl3_format_test`(CTest, `cpu;core;quant;exl3`): 통과.
  - FP16 65,536개 왕복, 동률 처리
  - TileState 대 단순 비트 읽기(모든 K와 반정수 K, 무작위 타일)
  - 타일 순열
  - 코드북: 3 × 65,536 상태 전부의 출력 checksum을 독립 Python 재현(struct `e` 반올림)과 대조
  - Hadamard, 참조 GEMV 대 명시적 회전 가중치(상대 오차 1e-12 미만)
- **원본 대비 확인(명세 검증):** 4.00/3.00/5.00 팩의 3층 `k_proj`를 명세대로 numpy로 풀고 회전해, HF BF16 원본과 비교했다.

| 팩 | K | 상대 오차 | 상관 |
|---|---:|---:|---:|
| 3.00 | 3 | 0.143 | 0.9898 |
| 4.00 | 4 | 0.072 | 0.9974 |
| 5.00 | 5 | 0.037 | 0.9993 |

  - 비트가 하나 늘 때마다 오차가 절반이 된다. 그래서 비트 순서, 타일 순열, 코드북, suh/svh 쪽, Hadamard 순서가 맞다. 배치가 틀리면 오차가 1 근처가 된다.
- **upstream 비트 대조(창 `xa-ref-a01/xa-ref`, 2026-10-30 14:45, COMPLETE, health_PASS):**
  - 덤프: `dump_ref.py`(이미지 `exl3-eval:1.6.5-rocm100`, 라벨 `f26f800d`, gfx1151, exllamav3 1.6.5). `synth` 2 s, `pack` 11 s.
  - 대조: CPU에서 `quant_exl3_upstream_test`.
  - 원시 JSON: `…/xa-ref-a01/xa-ref/receipt/run/exl3_upstream_{synth,pack}.json`(sha256 `016611e2a609…`, `21957d0465df…`).

| 대상 | 결과 |
|---|---|
| 합성 27경우: K 1–8 × 3INST/mcg/mul1, K 1.5/2.5/3.5 mul1(256 × 128) | inner W' FP16 비트 **27/27 동일**(sha256). FP64 forward 대 upstream FP64 상대 오차 4e-16–1.1e-15. upstream FP16 런타임(`LinearEXL3.forward`) 대 FP64 오차 3.5e-4–8.3e-4(FP16 반올림 수준) |
| 실제 텐서 8개 | 비트 **8/8 동일**: 3.00 k_proj(K3), 3.50 k_proj(K4), 3.50 layers.10 down_proj(K3), 4.00 k_proj와 layers.0 down_proj(K4), 4.00 lm_head(K6, 5120 × 248320), 4.00 mtp.fc(K4), 5.00 k_proj(K5) |

  - 합성 JSON을 fixture `tests/core/quant/exl3_upstream_synth.json`으로 넣었다. 이제 CTest `quant_exl3_upstream_test`(`cpu;core;quant;exl3`)가 돈다. `quant_exl3_format_test`와 함께 2/2 통과다.
  - mcg와 3INST는 실제 팩이 없다. 합성 비트 대조로만 확인했다(XQ가 다른 코드북 팩을 만들면 추가).

## X1: 27B 정확성 경로

- **설계(감독 승인 2026-10-30):** 정확성 경로는 오프라인 BF16 GGUF다. EXL3 팩을 llama.cpp `qwen35` GGUF 배치의 BF16 GGUF로 바꾸고, 기존 gufo BF16 경로로 돌린다. 런타임 코드는 바꾸지 않는다.
  - 로드 시점에 BF16으로 풀면 호스트 54 GB와 장치 54 GB가 120 GB 통합 메모리에 겹친다. 파일 경로는 페이지 캐시 기반이라 FAKEQ에서 이미 돌았다(HIP 최대 51 GB).
  - 직접 로드(`.trellis`를 GPU에 둔 채)는 trellis를 직접 푸는 커널이 생기는 X2로 옮긴다. 토크나이저 메타데이터(HF `tokenizer.json` → GGUF 키)도 X2에서 정한다.
- **코드:** `src/models/qwen/exl3_weights.{hpp,cpp}` (llama.cpp `8ab29fae` `conversion/qwen.py`와 같은 배치)
  - EXL3 투영 → 유효 가중치 `diag(suh)·H·W'·H·diag(svh)`를 [out][in] BF16으로
  - linear attention V 헤드를 grouped에서 tiled 순서로: qkv의 V 행, z 행, alpha/beta 행, out_proj 입력 열, conv1d V 채널, A_log, dt_bias
  - norm +1(linear_attn.norm 제외)과 A_log → −exp(A_log)은 FP32로
  - `mtp.*` → blk.64 nextn(fc → eh_proj, pre_fc_norm_embedding/hidden → enorm/hnorm, norm → shared_head_norm, 층 텐서)
- **도구:** `tests/models/qwen27b/exl3_to_gguf.cpp`(`qwen27b_exl3_to_gguf`, EXCLUDE_FROM_ALL). `compare`는 BF16 GGUF와 텐서별로 대조하고, `write`는 단일 파일 BF16 GGUF를 쓴다(템플릿 GGUF의 메타데이터에서 `split.*` 제외, 텐서 순서와 타입은 템플릿과 같음).
- **전체 매핑 대조(EXL3 4.00 대 BF16 GGUF, 866 텐서, 실패 0):**

| 분류 | 개수 | 내용 |
|---|---:|---|
| EXACT | 350 | 바이트 동일: 임베딩, 모든 norm(+1), dt_bias, conv1d(V 재정렬 포함), ssm_norm |
| NEAR | 107 | ssm_alpha/beta 96개: 팩이 원본 BF16을 F16으로 저장해 작은 값의 FP16 반올림 차이(rel 약 1e-8). ssm_a 11개: −exp의 1 ULP 차이 |
| QUANT | 409 | EXL3 양자화 오차. 본문 K=4는 rel 0.067–0.086(상관 0.996–0.998), lm_head(K=6)는 0.020 |

  - V를 재정렬한 텐서(qkv, gate, ssm_out)의 오차가 나머지와 같다. 재정렬이 틀리면 오차가 1 이상이 된다.
  - **`blk.64.nextn.eh_proj`(mtp.fc)는 rel 0.224, 상관 0.976으로 다른 텐서보다 크다.** 짐작: `mtp_bits` 4를 보정 없이 양자화했기 때문이다(W7 조사의 "MTP 4비트, 보정 없음"). 절반이 뒤바뀌었다면 상관이 0 근처여야 한다. 27B 대상 forward는 이 텐서를 쓰지 않는다.
- **BF16 GGUF(완료 2026-10-30):** `w8-exl3-20261029/x1-gguf/`(T5)에 썼다. 폴더마다 `PROVENANCE-<bpw>.json`이 있다.

| 팩 | 파일 | 크기 | sha256 | 시간 |
|---|---|---:|---|---:|
| 4.00 | `Qwen3.8-27B-EXL3-4.00bpw-BF16.gguf` | 54,657,735,424 | `079cf6ab19559cfc…` | 725 s |
| 5.00 | `Qwen3.8-27B-EXL3-5.00bpw-BF16.gguf` | 54,657,735,424 | `01053e2d3b9ebde4…` | 795 s |

  - 8스레드, 이미지 64849ca6ec7b, 도구 `04ca9c54`.
  - 확인: 쓴 4.00 파일을 gufo `GgufReader`로 다시 열어 `compare`로 대조했다(blk.3 11개 텐서 EXACT).
- **정확성 창 `PLAN-EXL3-X1.json`(초안):** 고정 수집기 `b5219f1b`(이미지 64849ca6ec7b)로 팩마다 창 하나.
  - vs-exl3: `--reference` = 같은 팩의 ExLlamaV3 run(`w8q-a01`). KL(ExLlamaV3 ‖ gufo-EXL3)이 엔진 차이이고, 목표는 중앙값 1e-4 이하, 평균 1e-3 이하다. prefill과 decode 둘 다 잰다.
  - vs-bf16: prefill만 `rq0-bf16` 기준으로 잰다. W8 값과 구간 안에서 같은지 본다.
  - GGUF fingerprint(RQ2H 방식): `build/xa-gpu-test`(HEAD `c913b069`에서 빌드, `src`는 `04ca9c54`와 같음, 바이너리 sha256 `53cdd1f8949113d9…`)의 `qwen27b_target_test --fingerprints-only`를 UD-Q4_K_M으로 돌리고, 창 뒤에 `rq2h/rq2h-units`와 바이트 대조한다. `5c3422ff` 이후 `src/` 변경은 Flash-Next와 EXL3 신규 파일뿐이다.
