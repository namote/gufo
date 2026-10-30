# X-A 결과: EXL3 형식 명세, CPU 참조, 27B 정확성 경로 (진행 중, 2026-10-30)

상위 문서: [X-A 브리프](XA-BRIEF.md), [EXL3 도입 계획](PLAN.md). 이 문서는 X-A(W8)가 쓴다.

## 상태

- **X0 완료.** CPU 부분 `f26f800d`, upstream 비트 대조 창 `xa-ref-a01` 통과(2026-10-30).
- **X1 완료(통과).** 매핑 `04ca9c54`, BF16 GGUF 4.00/5.00, 정확성 창 `xa-x1-a01` 통과(2026-10-30).

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
- **정확성 창(`PLAN-EXL3-X1.json`, `xa-x1-a01/xa-x1-4.00` 17:00–17:51, `xa-x1-5.00` 17:51–18:44, 둘 다 COMPLETE, health_PASS, 한 번씩):**
  - 수집기는 고정 `b5219f1b`(이미지 64849ca6ec7b)다. R0, FAKEQ와 같다.
  - 지표는 `metrics.py score`/`compare`(bootstrap 2000, seed 20261029, 블록 256)로 냈다. 보고는 `storage/evaluations/rq-r0-20261029/metrics-w8/x1-*.json`, `compare-x1-*.json`에 있다.

**엔진 차이: KL(ExLlamaV3 같은 팩 ‖ gufo-EXL3). 기준은 중앙값 ≤ 1e-4, 평균 ≤ 1e-3.**

| 팩 | 경로 | n | KL 평균 [토큰 95 %] | 중앙값 | 최대 1개 제외 | top-1 | 뒤집힘, KL > 1 | 판정 |
|---|---|---:|---|---:|---:|---:|---|---|
| 4.00 | prefill | 32,760 | 2.38e-4 [1.40e-4, 3.79e-4] | 2.98e-5 | 1.82e-4 | 0.9937 | 208, 0 | 통과 |
| 4.00 | decode | 4,096 | 1.18e-4 [0.86e-4, 1.62e-4] | 3.16e-5 | 1.05e-4 | 0.9949 | 21, 0 | 통과 |
| 5.00 | prefill | 32,760 | 4.28e-4 [1.70e-4, 8.27e-4] | 2.94e-5 | 2.61e-4 | 0.9929 | 233, 3 | 통과 |
| 5.00 | decode | 4,096 | 0.99e-4 [0.74e-4, 1.34e-4] | 3.17e-5 | 0.84e-4 | 0.9941 | 24, 0 | 통과 |

  - 참고로, 같은 엔진끼리의 바닥(ExLlamaV3 BF16 대 gufo BF16, W8)은 평균 6.9e-4, 중앙값 2e-5였다. X1 값은 이와 같은 크기다. 엔진 차이의 원인은 gufo의 BF16 반올림 유효 가중치 대 ExLlamaV3의 FP16 활성화와 FP16 logits로 본다[짐작].
  - 5.00 prefill의 KL > 1 세 위치는 chat 창 6이다(예: 위치 405, KL 5.5, 뒤집힘). 위치에 민감한 chat 위치라는 RQ2B 관찰과 같은 유형이다[짐작].

**BF16 기준 KL(prefill, `rq0-bf16` 대비)이 W8 ExLlamaV3 값과 같은지**

| 팩 | gufo-EXL3 KL | W8 ExLlamaV3 KL | paired 차이(gufo − ExLlamaV3) [토큰 95 % / 블록 95 %] | top-1 차이 | ΔNLL 차이 |
|---|---|---|---|---|---|
| 4.00 | 18.86e-3 [17.33, 20.54] | 18.84e-3 | +0.02e-3 [−0.51, +0.65] / [−0.54, +0.65], 구분 안 됨 | −0.0006, 구분 안 됨 | +2.3e-3 [+1.6, +3.2] |
| 5.00 | 7.79e-3 [6.23, 9.85] | 7.64e-3 | +0.14e-3 [−0.08, +0.38] / [−0.07, +0.41], 구분 안 됨 | −0.0002, 구분 안 됨 | +1.4e-3 [+0.5, +2.2] |

  - KL과 top-1은 W8 값과 신뢰구간 안에서 같다. 통과다.
  - ΔNLL만 gufo 쪽이 1.4–2.3e-3 nats/token 크다(구분됨). 같은 크기의 엔진 차이(BF16 대 FP16 경로)로 보이며 양자화 차이보다 한 자릿수 작다[짐작]. X2 커널(FP16/FP32 활성화, 가중치를 BF16로 다시 반올림하지 않음)에서 다시 본다.

**GGUF fingerprint(RQ2H 방식):**
  - `build/xa-gpu-test`의 `qwen27b_target_test --fingerprints-only`를 UD-Q4_K_M으로 돌렸다(27줄).
  - 결과는 `rq2h/units`의 같은 로그와 **바이트 단위로 같다**(`cmp`). EXL3 코드가 기존 GGUF 경로 출력을 바꾸지 않았다.

**판정: X1 통과.** 4.00과 5.00 모두 엔진 차이가 기준 안이고, BF16 기준 KL이 W8과 같으며, GGUF 경로 fingerprint가 그대로다. 다음은 X2(직접 로드와 trellis 커널, X-B)와 토크나이저 메타데이터 결정이다.
