# W8 결과: EXL3 27B 품질 측정으로 길 고르기 (2026-10-30)

상위 문서: [W8 브리프](W8-BRIEF.md), [PLAN "방향 전환"](PLAN.md), [FLASH-NEXT 13절](FLASH-NEXT.md). 이 문서는 W8이 쓴다. 속도 창(`PLAN-EXL3-SPEED.json`)은 아직 돌리지 않았다. 결과가 나오면 "속도" 절을 채운다.

## 결론

- **품질(같은 bpw): EXL3가 모든 구간에서 k-quant(unsloth UD)보다 낫다.** 3.0, 3.5, 4.0, 5.0 bpw 네 쌍 모두에서 prefill KL 차이가 토큰·블록 구간 안에서 0을 벗어난다. top-1도 모두 EXL3가 높다.
  - KL 기준으로 EXL3 b bpw는 UD k-quant 약 b + 0.7 bpw와 비슷하다. 예: EXL3 4.00(4.11 bpw) KL 18.8×10⁻³ ≈ UD-Q4_K_M(4.80 bpw) 17.5×10⁻³(차이 구분 안 됨).
- **엔진 차이는 작다.** 같은 ExLlamaV3 런타임으로 돌린 BF16 원본과 gufo BF16의 KL은 6.9×10⁻⁴(중앙값 2×10⁻⁵)다. EXL3 수치는 거의 전부 양자화 오차다.
- **판단: 3번 갈래는 "QTIP 형식 도입" 쪽이다(품질 기준).** 조건이 둘 있다.
  1. 속도: 같은 KL에서 EXL3 4.00이 UD-Q4_K_M보다 decode가 빠르다(tg d0 +20 %, d32K +8 %, MTP 투기 decode 31.7 대 gufo DFlash2 29.5 tok/s, +8 %). 두 엔진 모두 decode는 약 200 GB/s로 대역폭에 묶인다. 다만 upstream 런타임은 prefill이 약 33 % 느리고, 깊이 32K prefill이 gfx1151에서 돌지 않는다. 그래서 형식은 들이고 커널은 gufo 쪽에서 구현하는 방향이다(아래 "속도 요약").
  2. 비교한 k-quant는 unsloth UD 레시피다. 회전 k-quant나 GPTQ식 오차 보정 k-quant(3번 갈래 (a)(b))는 아직 없다. 이들이 약 0.7 bpw의 격차를 얼마나 줄이는지는 R4/E3 결과로 판단한다[짐작: 회전 이득은 3비트대에서 크다고 알려져 있으나 이 하네스로는 아직 재지 않았다].
- **주의(꼬리):** truncated KL_256은 복사 실패를 숨길 수 있다. 정답 토큰이 기준 top-256 밖이면 KL에는 거의 안 잡히고 NLL에만 잡힌다. 그래서 **KL과 함께 ΔNLL, 위치별 NLL 꼬리를 반드시 같이 본다.** EXL3 3.50에서 실제로 일어났다(아래 "EXL3 3.50 ΔNLL 이상").

## 측정 대상

| 이름 | 출처 | 리비전 / sha256 | bpw (W2 정의) |
|---|---|---|---:|
| EXL3 3.00 | `turboderp/Qwen3.8-27B-exl3` `3.00bpw` | `6fe61ad6` | 3.165 |
| EXL3 3.50 | 〃 `3.50bpw` (K3 50 % / K4 50 %) | `8351c54e` | 3.639 |
| EXL3 4.00 | 〃 `4.00bpw` | `113cf7ab` | 4.115 |
| EXL3 5.00 | 〃 `5.00bpw` | `a35e75a7` | 5.064 |
| EXL3-BF16(대조) | HF 원본 `Qwen/Qwen3.8-27B` `1d4bf0f2`(외장 백업), ExLlamaV3 비양자화 로드 | — | 16 |
| UD-IQ3_XXS fake-quant | `unsloth/Qwen3.8-27B-GGUF` `4ca72078` → BF16 | 원본 `c0b7c303…`, BF16 `b1f8bf79…` | 3.170 |
| UD-IQ3_S fake-quant | 〃 | 원본 `d847e2c1…`, BF16 `314807a2…` | 3.475 |
| UD-IQ4_XS fake-quant | 〃 | 원본 `40fac405…`, BF16 `8ab19950…` | 4.165 |
| UD-Q4_K_XL fake-quant(대조) | R0와 같은 파일 → BF16 | 원본 `3f227079…`, BF16 `ceb9fbc5…` | 5.145 |
| R0 k-quant(네이티브) | [R0 결과](R0-RESULTS.md) | R0 표 | 4.80–8.50 |

- **파일 위치:** `/run/media/m2on/T5 EVO/AI/model/w8-exl3-20261029/`(외장 디스크). 폴더마다 `PROVENANCE.json`(리비전, 파일별 크기와 sha256, HF LFS oid와 대조)이 있다.
- **bpw:** W2 정의(2차원 가중치, output 포함, token_embd와 NextN/MTP 제외)를 실제 텐서 바이트로 다시 계산했다(`tools/qwen27b/rq_quality/exl3/bpw.py`).
  - EXL3: 층 0–63과 lm_head의 trellis + suh/svh/mul1 바이트, 그리고 비양자화 2차원 텐서(`in_proj_a/b`, F16)를 넣는다. embed, vision, mtp는 뺀다. lm_head는 K=6이다.
  - GGUF 스크립트는 R0 값(UD-Q4_K_XL 5.1450, UD-Q4_K_M 4.8032)을 그대로 재현한다.
- **3비트대 UD 파일은 gufo 27B 로더가 받지 않는다**(IQ3_XXS, IQ2_*, Q2_K, IQ1_* 텐서. `weights.cpp` `SupportsTensorType`). 그래서 fake-quant로 쟀다.
  - 경로: W1 빌드 `llama-quantize --allow-requantize <UD>.gguf <out> BF16`(upstream llama.cpp `8ab29fae`, CPU, sha256 `99d8b4e2…`).
  - 검사: `exl3/dequant_check.py`. 파일마다 모든 양자화 타입(최대 14종)의 텐서 표본과 행 표본을 gguf-py `quants.dequantize`로 풀고 BF16(RNE)로 반올림해 비트 단위로 비교했다. 네 파일 모두 통과했고, 텐서 이름과 모양도 같다.
  - **fake-quant 경로 오프셋(대조):** UD-Q4_K_XL fake-quant와 네이티브 R0 run의 paired 차이는 prefill KL −0.03×10⁻³ [토큰 −0.14, +0.11], decode +0.18×10⁻³ [−0.02, +0.52]로 구분되지 않는다. top-1 차이도 구분되지 않는다. 4비트 이상에서 fake-quant는 네이티브와 같은 값으로 봐도 된다.

## 하네스와 런타임

- **코퍼스와 기준:** R0와 같다. a01 32,768토큰(8창 × 4,096), BF16 기준 `rq0-a01/rq0-bf16`, top-k 256, bootstrap 2000, seed 20261029, 블록 256.
- **EXL3 쪽 수집기:** `tools/qwen27b/rq_quality/exl3/collect.py`(커밋 `43425c9f` 기준).
  - a01 토큰 id를 그대로 넣는다. 토크나이저 sha256(`0997f410…`)과 vocab(248,320)이 gufo와 같다.
  - `prefill`: 창마다 새 상태에서 4,096토큰 전체 forward(`flash_attn_nc`) 한 번, 위치 0–4,094.
  - `decode`: prefix 2,048을 fp16 캐시에 prefill한 뒤 한 토큰씩, 위치 2,048–2,559(gufo decode 기준과 같은 위치).
  - 레코드는 `logits_io` 형식이다(`records.py`: top-256 동률은 낮은 id, logsumexp float64, ref_logits는 기준 run의 id에서).
  - 정렬 검사: CPU에서 stub 엔진으로 실제 `rq0-bf16` prefill·decode run과 짝을 지어 키, target, 행 순서, 메타데이터가 같음을 확인했다. GPU smoke(`w8s-a03`)도 통과했다.
- **ExLlamaV3 런타임:** upstream v1.6.5(`cdc785db`) 릴리스 wheel `rocm10.0.torch2.13.0`. 이미지 `ai-tools/exl3-eval:1.6.5-rocm100`(Dockerfile `ai.tools/test/experiments/gufo_upstream_port_20261023/exl3/`, 8.6 GB, pip freeze와 wheel 46개 sha256을 `/opt/provenance`에 둠).
  - 장치: `gfx1151`(AMD Radeon 8060S), HIP 7.15.26333, torch 2.13.0+rocm10.0.0.
  - logits는 fp16이다. Triton 캐시는 `/receipt`에 둔다(창의 `/tmp`는 noexec).
  - 실행된 커널 목록은 아직 얻지 못했다(torch.profiler가 이 ROCm 빌드에서 GPU 이벤트를 내지 않았다). 속도 창의 `rocprofv3 --kernel-trace`로 확인한다.
- **gufo 쪽(fake-quant):** R0/RQ2B-prism과 같은 수집기 `b5219f1b`, 이미지 `64849ca6ec7b`. prefill과 decode 두 경로, 헤드 재생 검사 8/8.

## 결과: prefill (n = 32,760, gufo BF16 대비, ×10⁻³)

| 설정 | bpw | KL [토큰 95 %] | KL vs EXL3-BF16 | 중앙값 | 최대 1개 제외 | top-1 | 뒤집힘(비율), 그중 KL>1 | ΔNLL [95 %] |
|---|---:|---|---:|---:|---:|---:|---|---|
| EXL3-BF16(대조) | 16 | 0.69 [0.09, 1.82] | — | 0.02 | 0.15 | 0.9933 | 220 (0.67 %), 1 | −0.6 [−1.5, +0.9] |
| EXL3 3.00 | 3.165 | 60.1 [56.9, 63.5] | 59.7 | 23.4 | 59.4 | 0.8855 | 3750 (11.4 %), 147 | +11.9 [+4.7, +19.2] |
| UD-IQ3_XXS fq | 3.170 | 106.7 [101.3, 112.3] | — | 41.6 | 105.9 | 0.8555 | 4735 (14.5 %), 276 | +105.6 [+95.7, +115.4] |
| EXL3 3.50 | 3.639 | 49.4 [46.2, 52.8] | 49.3 | 14.6 | 49.0 | 0.9063 | 3070 (9.4 %), 162 | +68.2 [+61.8, +74.8] |
| UD-IQ3_S fq | 3.475 | 70.3 [66.6, 74.0] | — | 26.2 | 69.8 | 0.8809 | 3903 (11.9 %), 161 | +30.8 [+22.9, +38.3] |
| EXL3 4.00 | 4.115 | 18.8 [17.3, 20.6] | 18.8 | 6.05 | 18.5 | 0.9353 | 2118 (6.5 %), 50 | +11.6 [+7.6, +15.7] |
| UD-IQ4_XS fq | 4.165 | 31.7 [29.2, 34.5] | — | 10.5 | 31.1 | 0.9177 | 2697 (8.2 %), 79 | −9.4 [−15.2, −4.1] |
| UD-Q4_K_M(R0) | 4.803 | 17.5 [15.4, 20.0] | — | — | — | 0.9413 | — | +1.3 |
| EXL3 5.00 | 5.064 | 7.6 [6.1, 9.7] | 7.0 | 1.80 | 7.1 | 0.9611 | 1274 (3.9 %), 14 | −1.6 [−4.5, +1.4] |
| UD-Q4_K_XL fq | 5.145 | 15.2 [12.9, 18.1] | — | 3.50 | 14.4 | 0.9478 | 1709 (5.2 %), 42 | +10.3 [+6.0, +14.8] |
| UD-Q4_K_XL(R0 네이티브) | 5.145 | 15.2 | — | — | — | 0.9475 | — | +10.7 |
| Q6_K(R0) | 7.045 | 3.0 | — | — | — | 0.9698 | — | +0.8 |
| Q8_0(R0) | 8.502 | 2.1 | — | — | — | 0.9795 | — | +1.1 |

- EXL3 KL은 gufo BF16 기준과 EXL3-BF16 기준이 세 자리까지 같다(5.00만 7.6 대 7.0).
- 기준 top-256 질량: prefill 평균 0.970, decode 0.964(R0와 같다).

### 카테고리별 (prefill KL ×10⁻³)

| 설정 | chat | code | en_prose | ko_prose |
|---|---:|---:|---:|---:|
| EXL3-BF16 | 2.6 | 0.1 | 0.0 | 0.1 |
| EXL3 3.00 | 107 | 30 | 49 | 54 |
| UD-IQ3_XXS fq | 206 | 50 | 78 | 93 |
| EXL3 3.50 | 123 | 17 | 28 | 30 |
| UD-IQ3_S fq | 129 | 34 | 52 | 66 |
| EXL3 4.00 | 39 | 7.9 | 13.1 | 15.3 |
| UD-IQ4_XS fq | 66 | 14 | 21 | 25 |
| EXL3 5.00 | 20 | 2.3 | 3.7 | 4.6 |
| UD-Q4_K_XL fq | 38 | 5.2 | 7.2 | 10.3 |

- 모든 카테고리에서 같은 bpw 쌍의 순서가 같다(EXL3가 낮다). 예외는 EXL3 3.50 대 IQ3_S의 chat(123 대 129)로, 거의 같다.

## 결과: decode (n = 4,096, gufo BF16 대비, ×10⁻³)

| 설정 | KL [토큰 95 %] | KL vs EXL3-BF16 (창 2·6, n=1,024) | top-1 | ΔNLL [95 %] |
|---|---|---:|---:|---|
| EXL3-BF16 | 0.17 [0.09, 0.27] (창 2·6) | — | 0.9971 | −2.3 |
| EXL3 3.00 | 60.2 [48.9, 75.7] | 95.0 | 0.8879 | +23.8 [+12.1, +35.1] |
| UD-IQ3_XXS fq | 90.2 [80.6, 103.1] | — | 0.8601 | +77.3 [+59.0, +97.3] |
| EXL3 3.50 | 39.2 [30.7, 50.8] | 71.4 | 0.9136 | +35.8 [+24.5, +47.0] |
| UD-IQ3_S fq | 64.9 [54.8, 79.0] | — | 0.8735 | +32.0 [+19.2, +45.5] |
| EXL3 4.00 | 15.9 [12.9, 20.2] | 26.3 | 0.9370 | +8.4 [+2.2, +14.3] |
| UD-IQ4_XS fq | 35.3 [24.8, 50.0] | — | 0.9167 | +19.5 [+9.6, +29.6] |
| EXL3 5.00 | 10.4 [4.0, 20.5] | 23.7 | 0.9639 | +2.0 [−2.4, +5.9] |
| UD-Q4_K_XL fq | 15.2 [9.2, 24.2] | — | 0.9473 | −0.6 [−6.4, +4.8] |

- "vs EXL3-BF16" 열은 대조 decode가 창 2(ko_prose)와 6(chat)뿐이라 어려운 카테고리만 들어 있다. 다른 열과 직접 비교하지 않는다.
- prefill과 decode의 순서와 크기는 같은 경향이다(같은 64위치에서 EXL3 4.00 decode KL 13.5 = prefill 13.5, smoke `w8s-a03`).

## 같은 bpw의 paired 차이 (EXL3 − k-quant, 같은 위치, ×10⁻³)

| 경로 | 비교 | KL [토큰 / 블록 95 %] | top-1 [토큰 95 %] | ΔNLL [토큰 / 블록 95 %] |
|---|---|---|---|---|
| prefill | EXL3 3.00 − UD-IQ3_XXS | −46.6 [−52.0, −41.5] / [−57.6, −37.8] | +0.0301 [+0.0261, +0.0341] | −93.7 [−104.6, −83.3] / [−134.2, −61.9] |
| prefill | EXL3 3.50 − UD-IQ3_S | −20.9 [−24.5, −17.2] / [−26.0, −15.1] | +0.0254 [+0.0216, +0.0290] | **+37.5** [+28.8, +46.6] / [+15.0, +65.1] |
| prefill | EXL3 4.00 − UD-IQ4_XS | −12.9 [−15.5, −10.4] / [−16.3, −10.1] | +0.0177 [+0.0144, +0.0210] | **+21.1** [+15.0, +27.2] / [+9.0, +36.4] |
| prefill | EXL3 5.00 − UD-Q4_K_XL fq | −7.5 [−10.1, −5.4] / [−9.9, −5.5] | +0.0133 [+0.0105, +0.0160] | −11.9 [−16.7, −7.3] / [−20.5, −4.0] |
| prefill | EXL3 4.00 − UD-Q4_K_M(R0) | +1.3 [−1.1, +3.4](구분 안 됨) | −0.0060 [−0.0089, −0.0029] | +10.3 [+4.2, +16.1] |
| prefill | EXL3 5.00 − Q6_K(R0) | +4.7 [+3.2, +6.7] | −0.0087 [−0.0109, −0.0064] | −2.4(구분 안 됨) |
| decode | EXL3 3.00 − UD-IQ3_XXS | −30.0 [−45.6, −12.8] | +0.0278 [+0.0164, +0.0391] | −53.5 [−75.2, −33.6] |
| decode | EXL3 3.50 − UD-IQ3_S | −25.7 [−32.2, −19.4] | +0.0400 [+0.0295, +0.0510] | +3.8(구분 안 됨) |
| decode | EXL3 4.00 − UD-IQ4_XS | −19.4 [−33.9, −9.0] | +0.0203 [+0.0112, +0.0293] | −11.1 [−22.3, +0.4] / 블록 [−18.2, −4.1] |
| decode | EXL3 5.00 − UD-Q4_K_XL fq | −4.8 [−10.2, +1.7] / 블록 [−8.4, −1.5] | +0.0166 [+0.0088, +0.0237] | +2.6(구분 안 됨) |
| prefill | UD-Q4_K_XL fq − 네이티브(대조) | −0.03 [−0.14, +0.11] | +0.0003(구분 안 됨) | −0.4 [−0.7, −0.2] |
| decode | UD-Q4_K_XL fq − 네이티브(대조) | +0.18 [−0.02, +0.52] | 0.0000 | −0.1(구분 안 됨) |

- **KL과 top-1은 네 쌍 모두 EXL3가 낫다(prefill에서 모두 구분됨).**
- **ΔNLL은 3.50과 4.00 쌍에서 반대로 나온다(EXL3가 나쁨).** 원인은 아래 절의 꼬리다. 3.50은 chat 창의 복사 실패가 원인이다. 4.00 쌍은 UD-IQ4_XS fq의 ΔNLL이 음수(−9.4)인 쪽도 있다. 즉 chat 창 NLL이 양쪽으로 크게 흔들린다(R0에서도 UD-Q4_K_M 등의 chat ΔNLL은 음수였다). ΔNLL은 이 코퍼스에서 chat 꼬리에 크게 좌우되므로, 판정에는 KL·top-1과 카테고리별 ΔNLL을 함께 쓴다.

## EXL3 3.50 ΔNLL 이상 (prefill +68×10⁻³)

- **소수 위치에 몰려 있다.** 중앙값은 +0.15×10⁻³, ΔNLL > 0 비율은 54.7 %로 다른 설정과 같다. 상위 300위치(0.9 %)가 합계 2,236 nats 중 1,410 nats(63 %)를 차지하고, 그중 282개가 chat 창(6: 112, 7: 170)에 있다.
- **카테고리:** chat +0.233, code +0.013, en_prose +0.011, ko_prose +0.016이다. chat을 빼면 3.50 < 3.00 순서(3.00은 +0.023–0.040)가 예상대로다. 창 7의 위치 3,328–3,839가 가장 크다(블록 평균 +1.31, +0.93 nats).
- **무엇이 틀렸나:** 합성 대화의 복사 위치다. 반복되는 섹션 제목("08 · Committed backing pool and transfer streams"), "Answer from the project", "\nassistant\n<think>" 뒤의 "\n\n" 같은 토큰이다. gufo BF16은 이들을 거의 확실하게 맞히고(NLL ≈ 0), EXL3 3.50은 13–23 nats를 잃는다.
  - 몇 위치는 KL_256이 0.01 정도로 작다(정답이 기준 top-256 밖이라 KL에 안 잡힘).
  - 짐작: 3.50의 K3/K4 혼합에서 긴 문맥 복사(induction) 회로에 쓰이는 텐서가 낮은 비트를 받았을 수 있다. 텐서별 확인은 하지 않았다.
- **교훈(감독 지시로 기록):** truncated KL은 복사 실패를 숨길 수 있다. 품질 판정에는 KL과 함께 ΔNLL, 위치별 NLL 꼬리(상위 위치의 합계 비율, 카테고리별 값)를 같이 본다. 회전 k-quant(R4)와 D4 실제 작업 게이트에도 같은 기준을 적용하길 권한다.
- decode 경로 확인(같은 위치를 decode로 재기)은 선택 사항으로 남겼다. 지금 decode run은 위치 2,048–2,559만 덮어서 이 위치들을 포함하지 않는다.

## 실행 기록

| 창 | 내용 | 상태 | 시간 |
|---|---|---|---|
| `w8s-a01` | smoke 1차 | FAILED_no_fallback(이미지에 C 컴파일러 없음: Triton 런처 빌드 실패, decode 캐시 < 로드 chunk) | 약 1분 |
| `w8s-a02` | smoke 2차 | FAILED_no_fallback(수집기: inference_mode 밖 상태 생성, prefill 행 1개 초과) | 약 9분 |
| `w8s-a03` | smoke 3차 | COMPLETE, 5/5 | 약 8분 |
| `w8q-a01/w8q-bf16` … `w8q-5.00` | QUALITY 5창 | 모두 COMPLETE, health_PASS | 03:00–03:53 |
| `w8f-a01/w8f-ud-iq3_xxs` … `w8f-ud-q4_k_xl` | FAKEQ 4창 | 모두 COMPLETE, health_PASS | 04:24–07:11 |

- 모든 창은 한 번씩만 돌렸고 자동 재시도는 없다. 세 smoke 실패는 고친 뒤 감독 승인을 받아 다시 돌렸다.
- 결과 위치: `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/{w8s-a0*,w8q-a01,w8f-a01}/`, 지표 `~/projects/ai.tools/storage/evaluations/rq-r0-20261029/metrics-w8/`(`<run>-<path>-vs-{gufo,exl3bf16}.json`, `fq-*-vs-gufo.json`, `compare-*.json`).
- EXL3 런 사실: 로드(USB) 11–127 s(첫 로드는 Triton JIT 포함), BF16 289–356 s. torch 최대 할당은 3.00 14.9 GB, 3.50 16.5, 4.00 18.0, 5.00 21.1, BF16 55.9 GB(prefill, 4,096×248,320 logits 포함). prefill forward는 창당 약 9 s.
- gufo fake-quant 런: prefill 약 18분, decode 약 23분, HIP 최대 51.3 GB(BF16 경로).

## 속도 (깊이 0, 창 `w8sp-a01/w8sp-pptg`)

- **측정:** `exl3/bench.py pptg`(커밋 `9ea56a95`), 이미지 `ai-tools/exl3-eval:1.6.5-rocm100`. a01 토큰, fp16 캐시, chunk 2048. pp2048은 빈 캐시에서 prefill 한 번, tg128은 한 토큰씩 forward 하고 토큰마다 argmax로 호스트 동기화한다. warm-up 1회 뒤 3회 반복의 평균이다.
- **순서(ABBA):** 3.00, 3.50, 4.00, 5.00, 5.00, 4.00, 3.50, 3.00. 창 전 점검: 컨테이너 0, 1분 부하 0.43, LOCKS 일치.
- **창 상태:** FAILED_no_fallback(health_PASS true). 여덟 항목 모두 깊이 0을 잰 뒤 깊이 32K 문맥을 채우다 실패했다(아래 "깊이 32K"). 깊이 0 값은 로그(`receipt/run/pptg-*.log`)에 남아 있고 측정 자체는 유효하다(감독 확인 2026-10-30).

| 설정 | bpw | pp2048 A / B (tok/s) | tg128 A / B (tok/s) | 토큰당 읽는 바이트 [짐작] | 실효 대역폭 [짐작] |
|---|---:|---|---|---:|---:|
| EXL3 3.00 | 3.165 | 421.2 / 418.9 | 18.79 / 18.78 | 10.14 GB | 190 GB/s |
| EXL3 3.50 | 3.639 | 425.4 / 431.8 | 17.07 / 17.06 | 11.66 GB | 199 GB/s |
| EXL3 4.00 | 4.115 | 432.7 / 434.2 | 15.56 / 15.55 | 13.18 GB | 205 GB/s |
| EXL3 5.00 | 5.064 | 433.0 / 411.0 | 12.37 / 12.37 | 16.22 GB | 201 GB/s |
| gufo UD-Q4_K_M(R0, `rq0-speed-a01`) | 4.803 | 635.5 / 661.0 | 12.94 / 12.92 | 15.38 GB | 199 GB/s |
| gufo UD-Q4_K_XL(R0) | 5.145 | 639.8 / 639.5 | 12.32 / 12.32 | 16.48 GB | 203 GB/s |

- 바이트는 W2 정의의 2차원 가중치 바이트다(임베딩 한 행, MTP, KV, 활성화는 넣지 않았다). 그래서 실효 대역폭은 근사다[짐작].
- 잡음: 같은 설정의 두 표본 차이는 tg 0.1 % 미만, pp 최대 5.1 %(5.00: 433.0 대 411.0)다. **5.00의 pp 칸만 5 % 기준을 넘어 결론 없음**이고, 나머지는 기준 안이다.
- 같은 장비이지만 엔진과 이미지가 다르다. gufo 값은 R0 속도 창(`gufo bench`, 바이너리 `rqw3-base-rocm101`, kvarn-k4v2 KV)이고, EXL3 값은 fp16 KV다. 깊이 0에서 KV 차이의 영향은 작다고 본다[짐작].

**읽은 것:**
- **decode는 두 엔진 모두 대역폭에 묶여 있다.** 실효 대역폭은 둘 다 약 190–205 GB/s다. FLASH-NEXT 13.3의 "EXL3 dense GEMV는 피크의 55–65 %" 짐작은 upstream 1.6.5 + gfx1151에서는 맞지 않았다.
- **같은 품질에서 EXL3가 decode가 빠르다.** EXL3 4.00은 UD-Q4_K_M과 KL이 구분되지 않는데(18.8 대 17.5×10⁻³) tg가 15.56 대 12.93 tok/s로 약 +20 %다. EXL3 5.00은 KL이 UD-Q4_K_XL의 절반인데 tg가 같다(12.37 대 12.32).
- **prefill은 EXL3가 약 33 % 느리다**(pp2048 약 430 대 640–660 tok/s).
- 실행된 커널 목록은 이 창에서 얻지 못했다(`w8sp-kernels` 실패. 원인: `rocprofv3`의 `#!/usr/bin/env python3`에 컨테이너 PATH가 python3을 찾지 못함). 아래 "실행 커널" 절(`w8k-a01`)에 있다.

**깊이 32K:**
- upstream 1.6.5의 Triton `_paged_attn_prefill_kernel`(캐시에 과거 문맥이 있는 prefill, head_dim 256, fp16 캐시)이 gfx1151에서 LDS 131,072 B를 요구한다. 장치 한도는 65,536 B라 `OutOfResources`로 실패한다. 위치 0부터의 prefill은 된다(품질 측정, decode prefix, 깊이 0 pp).
- 그래서 **이 런타임으로는 깊이 32K의 pp를 잴 수 없다.** 실제 서빙에서 여러 chunk로 나뉘는 긴 프롬프트나 다회 대화도 같은 경로를 탈 수 있다[짐작, 확인하지 않음].
- tg 깊이 32K는 문맥을 위치 0부터 한 번에 채우는 방식(`--fill-single`)으로 SPEED2에서 잰다.

## 속도 (SPEED2, 창 `w8sp2-a01`, 2026-10-30 08:59–11:17)

- 바이너리: `exl3/bench.py` 커밋 `3b96a10e`, 이미지 `ai-tools/exl3-eval:1.6.5-rocm100`(id `03a29041b45d`). 창마다 점검: 컨테이너 0, 1분 부하 0.61–1.86, LOCKS 일치, 루트 여유 157–158 GB. 모든 창은 한 번씩만 돌렸다.
- `w8sp2-kernels`: FAILED_no_fallback. 측정 작업 자체(`kernels-exl3-4.json`)는 끝났지만, `rocprofv3`가 작업 디렉터리(`/src`, 읽기 전용)에 `.rocprofv3`를 쓰려다 abort한 뒤 timeout(1,200 s)까지 멈췄다. 커널 목록은 아래 `w8k-a01`에서 다시 쟀다.

### 실행 커널 (창 `w8k-a01/w8k-kernels`, `PLAN-EXL3-KERNELS.json`, COMPLETE)

- `rocprofv3 --kernel-trace --stats`, EXL3 4.00, `bench.py kernels`(prefill 512 + decode 16단계, 두 번). `/receipt/run`에서 시작하고 모델 경로는 env로 넘겼다. 결과: `…/w8k-a01/w8k-kernels/receipt/run/rocprof-exl3-4/trace_kernel_{trace,stats}.csv`.
- 두 번째 패스의 decode 16단계만 떼어 보았다(launch 18,829개, 커널 시간 1,035 ms, wall 1,080 ms = 토큰당 67.5 ms, 14.8 tok/s. 프로파일러 없이 잰 15.56 tok/s와 비슷하다).

| decode 커널 | 비율 | 토큰당 ms | 역할 [짐작] |
|---|---:|---:|---|
| `exl3_gemv_mr_dot_multi<4, false, 2, 4, 1>` | 40.0 % | 25.9 | 묶음 GEMV(K=4 trellis): MLP gate+up 등 |
| `exl3_gemv_mr_dot_single<4, true, 2, 8, 1>` | 21.6 % | 14.0 | 단일 GEMV(출력 Hadamard 융합) |
| `exl3_gemv_mr_dot_single<4, true, 2, 16, 1>` | 19.8 % | 12.8 | 단일 GEMV: MLP down 등 |
| `exl3_gemv_dot_kernel<6, false, 2, 4>` | 6.5 % | 4.2 | lm_head(K=6) |
| `exl3_gemv_mr_dot_single<4, false, 2, 8, 1>` | 3.5 % | 2.3 | 기타 투영 |
| `_paged_attn_decode_split_kernel`(Triton) | 1.3 % | 0.9 | full attention 16층 |
| `cuda_recurrent_gated_delta_rule_kernel_128` | 1.3 % | 0.8 | DeltaNet 48층 |
| 그 밖(rms_norm, `exl3_gemv_mr_had_in_*`, conv, gdn_ba 등) | 약 6 % | 약 4 | |

- **decode는 RDNA용 EXL3 trellis GEMV(`exl3_gemv_mr_*`)가 직접 돈다.** reconstruct(fp16로 풀기) + hipBLAS 같은 대체 경로는 decode에 거의 없다(`reconstruct_kernel` 8회, 0.4 %). 입력 Hadamard는 별도 작은 커널(`exl3_gemv_mr_had_in_*`, 0.7 %)이고 출력 Hadamard는 GEMV에 융합된다.
- 커널 단위 대역폭[짐작]: lm_head 0.95 GB / 4.2 ms ≈ 226 GB/s. MLP 세 행렬(층당 약 134 MB, 64층 8.6 GB)이 가장 큰 GEMV 세 종류(합 52.7 ms/토큰, 다른 투영 포함)에 들어 있다고 보면 약 200–220 GB/s다. 같은 장비의 gufo 4비트 GEMV(R0b: IQ4_XS 89 %, Q4_K 82 %, 약 200–215 GB/s)와 같은 수준이다.
- prefill(512토큰)은 trellis를 fp16으로 푸는 `reconstruct_kernel`/`reconstruct_had_kernel`과 hipBLASLt Tensile GEMM(`Cijk_…HHS…`), EXL3 WMMA GEMM(`exl3_wmma_gemm_kernel*`), fla Triton 커널(`chunk_*`, `recompute_w_u_fwd_kernel`)을 쓴다. 이 구간은 두 패스의 경계를 깔끔히 자르지 못해 비율은 싣지 않는다. 큰 m에서 reconstruct + GEMM은 upstream의 정상 경로다(대체 경로가 아님)[짐작: `exllamav3` ROCm 소스의 경로 선택을 읽지는 않았다].

### tg128 깊이 32K (`--fill-single`, 문맥을 위치 0부터 한 번에 채움, 3회 평균, ABBA 4.00 → 5.00 → 5.00 → 4.00)

| 설정 | tg128 d32K A / B (tok/s) | gufo 대응(R0 `rq0-speed-a01`, A / B) | 비율 |
|---|---|---|---:|
| EXL3 4.00 | 13.65 / 13.66 | UD-Q4_K_M 12.58 / 12.67 | +8 % |
| EXL3 5.00 | 11.14 / 11.14 | UD-Q4_K_XL 12.03 / 11.99 | −7 % |

- 깊이 0 대비 감소: EXL3 4.00 −12 %, 5.00 −10 %. gufo는 −2–3 %다(R0). 긴 문맥의 decode 비용(attention, fp16 KV를 읽는 양)이 EXL3 쪽에서 더 크다[짐작: gufo는 kvarn-k4v2 압축 KV라 KV 바이트가 적다].
- 두 표본 차이는 0.1 % 미만이다.

### 64프롬프트 chat: MTP 투기 디코드 대 gufo DFlash2

- 설정: R0 수락률 프롬프트 64개(`accept_prompts.json`), greedy, thinking off, 최대 256토큰, ExLlamaV3 `Generator`. 프롬프트는 팩의 chat template(한 사용자 턴)으로 만들었다. MTP는 팩 자체 MTP 헤드, 기본 draft 길이다(`-mtp`, `num_draft_tokens` 미지정). mtp는 3회, none은 1회 돌렸다. 반복 사이 출력 토큰 sha가 모두 같다.
- decode tok/s = Σ출력 토큰 / Σgenerate 시간(prefill 제외)이다. gufo 쪽은 R0 `rq0-accept-a01`(serving_bench, `gufo serve`, DFlash2 Q4_K_M adaptive)의 같은 정의 값이다.

| 설정 | 투기 decode tok/s (패스별) | AR tok/s(같은 프롬프트) | AR 대비 | 수락률 Σacc/Σdraft | 출력 토큰당 수락 토큰 |
|---|---|---:|---:|---:|---:|
| EXL3 4.00 + MTP | **31.73** (31.79, 31.71, 31.68) | 15.50 | 2.05× | 0.517 | 0.674 |
| gufo UD-Q4_K_M + DFlash2 | 29.50 [26.78, 32.81] | 12.95 | 2.28× | 0.453 | 0.663 |
| EXL3 5.00 + MTP | **27.51** (27.58, 27.49, 27.47) | 12.33 | 2.23× | 0.521 | 0.676 |
| gufo UD-Q4_K_XL + DFlash2 | 29.16 [26.38, 32.50] | 12.33 | 2.36× | 0.457 | 0.664 |

- **EXL3 4.00 + MTP가 gufo 운영(UD-Q4_K_M + DFlash2)보다 약 8 % 빠르다(31.7 대 29.5 tok/s).** 품질은 KL이 구분되지 않는다(18.8 대 17.5×10⁻³, top-1은 EXL3가 0.006 낮다). 이 차이는 거의 전부 AR 속도(바이트 감소)에서 나온다. 투기 배율은 gufo DFlash2가 더 크다(2.28× 대 2.05×).
- EXL3 5.00 + MTP는 gufo UD-Q4_K_XL + DFlash2보다 약 6 % 느리다(27.5 대 29.2). 같은 AR 속도에서 DFlash2의 배율이 MTP보다 높다.
- **비교 한계:** 두 엔진의 출력 텍스트가 다르다(엔진·양자화가 달라 greedy 경로가 갈림). 그래서 프롬프트별 paired 비교가 아니다. gufo 구간은 프롬프트 bootstrap 95 %이고, EXL3는 패스별 값만 있다(결정적이라 패스 간 차이는 0.4 % 이내). EXL3 출력 토큰 합계(4.00 11,236, 5.00 11,312)는 64 × 256의 69 %로, 33–34개 프롬프트가 256토큰 전에 끝났다.
- 투기 배율의 차이는 수락 방식이 아니라 draft 비용과 verify 비용에서 오는 것으로 보인다[짐작]. 출력 토큰당 수락 토큰은 두 엔진이 같다(0.674 대 0.663).

### 속도 요약

| 항목 | EXL3 4.00 대 UD-Q4_K_M(같은 KL) | EXL3 5.00 대 UD-Q4_K_XL(KL 절반) |
|---|---|---|
| tg128 d0 | +20 % | ±0 % |
| tg128 d32K | +8 % | −7 % |
| pp2048 d0 | −33 % | −32 % |
| pp d32K | 잴 수 없음(upstream LDS 초과) | 잴 수 없음 |
| chat 투기 decode | +8 %(MTP 대 DFlash2) | −6 % |

- **판단(속도):** 같은 품질에서 EXL3 형식은 decode를 빠르게 한다(바이트가 적고, 커널이 대역폭에 묶임). 그러나 upstream 런타임은 prefill이 느리고 긴 문맥 prefill이 gfx1151에서 돌지 않으며, 긴 문맥 decode의 감소폭이 크다. **결론은 "형식(QTIP) 도입 + 커널은 gufo 쪽 구현"이다.** EXL3 커널·런타임을 그대로 쓰는 것이 아니다. trellis 디코드를 gufo GEMV/verify/prefill 커널 구조와 kvarn KV, DFlash2 위에 얹는 것이 목표다[짐작: gufo의 prefill 효율과 KV 압축을 유지하면 EXL3 4.00급 바이트로 decode +20 % 수준을 기대].

## 남은 문제

- EXL3 MTP와 gufo DFlash2의 프롬프트별 paired 비교는 출력이 달라 불가능하다. 필요하면 같은 출력 길이를 강제하는 방식(EOS 무시)으로 다시 잰다.
- 회전/GPTQ식 k-quant가 같은 bpw에서 이 격차를 얼마나 줄이는지(R4/E3, 3번 갈래 (a)(b)).
- EXL3 3.50 복사 실패의 텐서별 원인, decode 경로 확인(선택).
- 이 결과는 turboderp 공개 차트(4.00bpw가 UD-Q4_K_XL보다 낮은 KL)와 다르다. 이 하네스에서는 4.00이 UD-Q4_K_XL보다 KL이 높다(+3.7×10⁻³, 구분됨). 코퍼스(기술 문서·코드·한국어), 기준(BF16 gufo), 지표(KL_256)가 다르기 때문으로 본다[짐작].
