# S: 작은 M K-quant GEMM 9–64행 결과

안내: [SKINNY-BRIEF.md](SKINNY-BRIEF.md). 브랜치 `ai-tools/skinny`(worktree `build/s-src`), 기준 `f0700122`.

## 범위

- 대상 형식: Q4_K, Q5_K, IQ4_XS, Q6_K. **Q3_K, IQ4_NL, IQ3_S는 지금 경로 그대로**(층 수가 적음, 감독 승인).
- 모델 shape(Q4_K_M, GGUF에서 셈): qkv 10240×5120, attn_gate 6144×5120, ssm_out·attn_output 5120×6144, attn_q 12288×5120, attn_k/v 1024×5120, FFN gate/up 17408×5120(같은 형식 짝은 packed 34816 또는 fused SwiGLU), down 5120×17408, head 248320×5120 Q6_K.

## 시제품 (`92b534fb`, `bc6b0db7`, 운영 dispatch에 연결 안 함)

- `SmallBatchKQuantExactFp32GEMMKernel`에 `Ragged` 템플릿 인자와 `valid_rows`: 마지막 group의 빈 행은 0을 stage하고 저장하지 않는다. 산 행마다 lane 누적·reduction 순서는 꽉 찬 group과 같다.
- `detail::TryLaunchKQuantGroupedWave64(type, …, rows, width, …)`: width 9–16, group 1–4를 한 launch로(인접 workgroup이 같은 가중치 행을 공유).
- 기존 커널 ISA: HEAD 대비 implicit kernarg 기준 offset 40→48 한 줄만 다르다(`small_batch_wave64` 119, `quant16` 24, `prefill_quant_gemm` 142 커널; 나머지 바이트 같음).
- bench `tools/qwen27b/dflash_gemm_bench.hip dispatch`: 운영 `LaunchBatchedQuantGEMMFp32`(FFN은 운영 packed SwiGLU 경로)와 후보(`grouped<W>`: G = ceil(n/16), W = ceil(n/G); n>16이면 W=16 ragged도)를, 회전하는 cold host-registered 가중치로, 시간 전에 한 행 decode(`LaunchQ8KBlockGEMV`/`LaunchFusedSwiGLUGEMV`)와 비트 비교 후 잰다. 4 round 교대, 중앙값.

## 창 `skinny-s-w1-a01/s-w1`

COMPLETE, health PASS, 재시도 없음. 이미지 `ai-tools/gufo-eval:4a7046b2e8ed`(컴파일러·런타임만), `/src` = S worktree, `.gufo-revision` `bc6b0db7`. 바이너리 SHA: bench `bb0cd3ea…`, 기준 bench `808b0ca3…`(f0700122 커널), peak `1a55c5ef…`. 결과 `~/projects/ai.tools/tmp/test-results/gufo-upstream-port-20261023/skinny-s-w1-a01/s-w1/receipt/run/`, 요약 `build/s-results/`(main.csv 행별, bands.txt, model.txt, ctl.csv). 모든 geometry가 비트 일치(불일치로 멈춘 시험 없음).

Peak(`gfx1151_peak`): DRAM read 240 GB/s.

### 관찰

- **8행을 넘으면 대역폭이 아니라 연산에 묶인다.** 운영 GB/s가 8행 117–216, 16행 86–176, 64행 17–46. 시간이 행 수에 거의 비례한다. 그래서 17–64행에서 "한 번 읽기"의 이득은 작고, 이득은 주로 나머지 launch(8 + 나머지)를 없애는 데서 온다.
- 9–16행: 운영이 12·16 외 폭을 `8 + 나머지`로 나누는 shape(qkv, gate, q, ssm_out, 개별 gate/up)에서 +16–38 %(Q6 12288·17408·down·34816은 +50–78 %, 운영이 Q6 일반 shape를 8행 경로로 나눔). 이미 정확 폭인 down(Q4/Q5/IQ4)·packed SwiGLU·Q6 head는 차이 없음(±1 %).
- 17–64행: 대개 +2–15 %, 폭별로 섞임. 균형 규칙의 폭 11·13·12가 지는 경우가 있다(예: IQ4 10240 n=21,22 −10–12 %, n=33 −15 %; IQ4 down 49–64 −6 %). packed SwiGLU는 이득 없음(−2–+4 %).
- 작은 1024×5120(k/v, Q5/Q6): 모든 폭 +14–71 %(운영이 8행 경로로 나눔).

### shape × 행 구간 (운영 → 두 후보 중 나은 것, 구간 평균 µs, 이득 %, (3 % 넘게 이긴 폭 수/진 폭 수))

```
type shape | GB/s@8 @16 @64 | band: prod_sum_us best_sum_us gain% (wins>3%/losses>3%)
iq4        10240x5120   | 143 93 25 | 9-16: 302->254 +18.9% (6/0) 17-32: 519->480 +8.2% (12/2) 33-48: 796->766 +3.9% (10/1) 49-64: 1068->1044 +2.3% (7/1)
iq4        12288x5120   | 137 93 25 | 9-16: 380->314 +21.0% (6/0) 17-32: 623->576 +8.1% (12/2) 33-48: 950->918 +3.5% (8/1) 49-64: 1274->1242 +2.6% (7/1)
iq4        17408x5120   | 162 102 27 | 9-16: 508->418 +21.4% (5/0) 17-32: 840->777 +8.1% (9/2) 33-48: 1262->1235 +2.2% (4/3) 49-64: 1698->1668 +1.7% (7/4)
iq4        5120x17408   | 175 93 26 | 9-16: 412->412 +0.1% (0/0) 17-32: 831->767 +8.3% (12/2) 33-48: 1219->1186 +2.8% (9/3) 49-64: 1595->1696 -6.0% (0/13)
iq4        5120x6144    | 138 87 24 | 9-16: 191->164 +16.4% (6/0) 17-32: 322->295 +9.3% (12/2) 33-48: 483->469 +3.0% (9/3) 49-64: 655->636 +2.9% (7/0)
iq4        6144x5120    | 117 86 24 | 9-16: 206->167 +23.3% (6/0) 17-32: 339->298 +14.0% (13/2) 33-48: 497->477 +4.2% (6/1) 49-64: 670->637 +5.2% (9/0)
iq4-swiglu 17408x5120   | 167 107 27 | 9-16: 813->817 -0.4% (0/0) 17-32: 1548->1546 +0.1% (6/5) 33-48: 2407->2456 -2.0% (2/4) 49-64: 3267->3306 -1.2% (2/5)
q4         10240x5120   | 157 114 31 | 9-16: 303->244 +24.2% (6/0) 17-32: 459->446 +2.9% (8/6) 33-48: 692->689 +0.4% (7/7) 49-64: 923->904 +2.1% (7/5)
q4         12288x5120   | 149 113 31 | 9-16: 379->292 +29.8% (6/0) 17-32: 577->535 +7.9% (9/6) 33-48: 860->835 +3.0% (7/5) 49-64: 1154->1105 +4.4% (9/0)
q4         17408x5120   | 176 115 31 | 9-16: 529->420 +25.9% (5/0) 17-32: 816->753 +8.3% (8/6) 33-48: 1196->1183 +1.1% (5/7) 49-64: 1614->1567 +3.0% (6/2)
q4         5120x17408   | 188 116 30 | 9-16: 417->418 -0.1% (0/0) 17-32: 767->772 -0.7% (6/6) 33-48: 1194->1189 +0.4% (5/3) 49-64: 1598->1568 +1.9% (7/0)
q4         5120x6144    | 120 107 29 | 9-16: 208->154 +35.4% (6/0) 17-32: 310->290 +6.7% (9/6) 33-48: 465->442 +5.2% (9/4) 49-64: 609->580 +5.0% (9/2)
q4         6144x5120    | 139 103 30 | 9-16: 200->160 +25.4% (6/0) 17-32: 300->277 +8.3% (9/6) 33-48: 437->427 +2.4% (7/6) 49-64: 586->564 +3.9% (8/1)
q4-swiglu  17408x5120   | 183 123 32 | 9-16: 826->829 -0.3% (0/1) 17-32: 1481->1485 -0.2% (6/6) 33-48: 2278->2334 -2.4% (3/7) 49-64: 3093->3119 -0.8% (2/6)
q5         10240x5120   | 198 130 36 | 9-16: 318->243 +31.0% (6/0) 17-32: 499->457 +9.2% (9/5) 33-48: 745->713 +4.5% (6/3) 49-64: 995->958 +3.9% (6/0)
q5         1024x5120    | 103 61 17 | 9-16: 57->47 +21.3% (8/0) 17-32: 96->62 +53.9% (16/0) 33-48: 146->103 +42.4% (16/0) 49-64: 198->120 +64.9% (16/0)
q5         12288x5120   | 200 133 36 | 9-16: 400->289 +38.3% (6/0) 17-32: 621->542 +14.5% (10/3) 33-48: 928->848 +9.4% (9/0) 49-64: 1223->1144 +6.9% (12/0)
q5         17408x5120   | 205 141 38 | 9-16: 546->395 +38.5% (5/0) 17-32: 855->740 +15.4% (10/1) 33-48: 1226->1164 +5.3% (6/1) 49-64: 1639->1570 +4.4% (5/0)
q5         5120x17408   | 212 143 37 | 9-16: 403->402 +0.2% (0/0) 17-32: 791->745 +6.3% (9/0) 33-48: 1206->1178 +2.3% (5/0) 49-64: 1588->1587 +0.0% (3/3)
q5         5120x6144    | 190 130 36 | 9-16: 187->152 +22.9% (6/0) 17-32: 295->285 +3.3% (7/6) 33-48: 446->442 +1.0% (6/5) 49-64: 590->584 +1.1% (7/6)
q5         6144x5120    | 168 119 35 | 9-16: 211->166 +27.0% (6/0) 17-32: 328->283 +16.1% (12/0) 33-48: 470->445 +5.7% (8/5) 49-64: 624->595 +4.9% (8/1)
q5-swiglu  17408x5120   | 200 151 39 | 9-16: 812->819 -0.9% (0/2) 17-32: 1540->1486 +3.6% (8/3) 33-48: 2325->2332 -0.3% (3/5) 49-64: 3115->3143 -0.9% (1/4)
q6         1024x5120    | 119 67 18 | 9-16: 58->51 +14.2% (8/0) 17-32: 100->66 +52.5% (16/0) 33-48: 156->97 +60.7% (16/0) 49-64: 212->124 +70.7% (16/0)
q6         12288x5120   | 175 88 22 | 9-16: 548->308 +77.9% (8/0) 17-32: 984->530 +85.6% (16/0) 33-48: 1567->811 +93.4% (16/0) 49-64: 2152->1091 +97.4% (16/0)
q6         17408x5120   | 200 163 46 | 9-16: 641->416 +54.1% (6/0) 17-32: 924->737 +25.4% (14/0) 33-48: 1293->1149 +12.6% (12/0) 49-64: 1704->1548 +10.1% (10/0)
q6         248320x5120  | 216 176 46 | 9-16: 5586->5592 -0.1% (0/0) 17-32: 10923->10401 +5.0% (8/0) 33-48: 16406->16271 +0.8% (2/0) 49-64: 22097->21985 +0.5% (2/0)
q6         34816x5120   | 205 170 46 | 9-16: 1260->808 +55.9% (6/0) 17-32: 1785->1447 +23.4% (14/0) 33-48: 2531->2269 +11.6% (10/0) 49-64: 3342->3068 +8.9% (9/0)
q6         5120x17408   | 205 158 45 | 9-16: 639->424 +50.8% (6/0) 17-32: 940->749 +25.6% (14/0) 33-48: 1313->1149 +14.2% (14/0) 49-64: 1708->1556 +9.8% (13/0)
q6         5120x6144    | 196 146 44 | 9-16: 208->163 +27.3% (6/0) 17-32: 312->284 +10.0% (9/2) 33-48: 445->428 +3.9% (7/5) 49-64: 587->564 +4.1% (9/4)
```

### 모델 한 verify 걸음의 projection 합(층 수 가중, 위 shape만, ms)

운영 vs (행마다 운영·후보 중 나은 것; 규칙이 아니라 상한). 실제 행 수는 C2 4–16(짝수), C4 8–32(4의 배수), C8 16–64(8의 배수).

```
 8 prod    83.5 ms  best-of    83.5 ms   +0.0%
 9 prod   116.4 ms  best-of    89.7 ms  +29.7%
10 prod   120.1 ms  best-of   101.3 ms  +18.5%
11 prod   130.8 ms  best-of   116.7 ms  +12.1%
12 prod   119.3 ms  best-of   114.1 ms   +4.5%
13 prod   144.5 ms  best-of   116.4 ms  +24.1%
14 prod   128.2 ms  best-of   117.7 ms   +8.9%
15 prod   150.2 ms  best-of   130.7 ms  +15.0%
16 prod   124.4 ms  best-of   123.6 ms   +0.6%
17 prod   191.2 ms  best-of   158.7 ms  +20.5%
20 prod   191.5 ms  best-of   180.5 ms   +6.1%
24 prod   208.2 ms  best-of   201.9 ms   +3.2%
28 prod   229.3 ms  best-of   217.4 ms   +5.5%
32 prod   236.5 ms  best-of   234.6 ms   +0.8%
36 prod   304.5 ms  best-of   299.7 ms   +1.6%
40 prod   319.5 ms  best-of   314.0 ms   +1.8%
44 prod   355.8 ms  best-of   343.7 ms   +3.5%
48 prod   348.9 ms  best-of   345.9 ms   +0.9%
52 prod   415.4 ms  best-of   411.7 ms   +0.9%
56 prod   431.4 ms  best-of   422.6 ms   +2.1%
60 prod   468.7 ms  best-of   454.5 ms   +3.1%
64 prod   463.6 ms  best-of   459.9 ms   +0.8%
```

### 1–8행 대조(kernarg offset 40→48)

기준(f0700122 커널) / 새 바이너리를 ABAB로, 운영 경로, Q4 10240·down·packed SwiGLU, Q6 5120×6144·head. 40칸 모두 ±1.1 % 안(Q4 SwiGLU 7행 −4.2 %는 새 쪽이 빠름, 범위 겹침). 퇴보 없음(`build/s-results/ctl.csv`).

## 운영 변경 (`4fc0211d`, 감독 승인 (a)·(b))

`prefill_quant_gemm.hip` `UseGroupedKQuant`: 8행 넘는 batch에서 `TryLaunchKQuantGroupedWave64`(균형 폭 W = ceil(n / ceil(n/16)))를 먼저 쓴다. 1–8행 경로는 그대로.

- (a) 9–16행, Q4_K·Q5_K·IQ4_XS·Q6_K, 6144·10240·12288·17408×5120, 5120×6144, Q6_K는 34816×5120·5120×17408도. 이미 한 launch인 Q4/Q5/IQ4 down·packed 34816과 Q6 head는 그대로. s-w1 격자의 최악 칸 +0.79 %(IQ4 17408 16행)는 운영과 같은 커널을 고르는 폭이라 잡음.
- (b) 9–64행, Q6_K 12288×5120, Q5_K·Q6_K 1024×5120: 모든 칸에서 이김(17–64 최소 이득 −37 %·−22 % 시간). Q6 5120×6144는 17–64에서 최대 +12.6 %로 져서 뺐다.
- 비트 테스트 `TestGroupedVerificationShapes`(`q4kxl_quant_ops_test.cpp`): 바뀐 폭·shape 전부를 한 행 decode GEMV와 비교, batch 뒤 guard 행 확인.

## 창 `skinny-s-w2-a01/s-w2`

창 FAILED_no_fallback(원인: `fn-new`의 타이밍 비교가 inconclusive라 runner가 0이 아닌 종료), health PASS, 재시도 없음. `.gufo-revision` `4fc0211d`, 이미지 `4a7046b2e8ed`. 바이너리 SHA: 후보 gufo `6e5600d3…`, 기준 gufo `22e4f9cf…`(f0700122 snapshot `build/s-base-src`), bench `29008fc3…`, `qwen_q4kxl_quant_ops_test` `aab4473e…`, `qwen_dflash_gpu_test` `ab4fbdca…`. 요약 `build/s-results/`(cohort.txt, rule2*.txt).

- **정확도**: `qwen_q4kxl_quant_ops_test` PASS(grouped 24줄 모두 OK), `qwen_dflash_gpu_test --concurrency-only` PASS(C2–C8 exact layers·logits·selector, private RNG/policy). 모델 cohort 4회(기준·후보·기준·후보) 모든 C·workload에서 완성 텍스트 hash가 네 번 같다. functional `batch`·`state-edges` 두 쪽 모두 passed, 출력·품질·coverage 변화 0.
- **처리량**(greedy DFlash2, speculative corpus, 128 토큰, context 4096, C마다 새 server; 요청 decode rate 합의 round 평균, 기준 2회 → 후보 2회):

```
mixed      C1: TG base 35.52 35.65 new 35.57 35.58 -> -0.0% | outputs identical across 4 runs: True | matched-work requests pass 9 flag 0, unmatched samples 0, worst +0.1% (instruction_debug 5004->5010 ms)
mixed      C2: TG base 54.24 54.53 new 55.09 55.28 -> +1.5% | outputs identical across 4 runs: True | matched-work requests pass 9 flag 0, unmatched samples 6, worst -0.4% (reasoning_train 3322->3308 ms)
mixed      C4: TG base 69.79 69.80 new 71.45 71.69 -> +2.5% | outputs identical across 4 runs: True | matched-work requests pass 9 flag 0, unmatched samples 7, worst -1.5% (instruction_debug 8734->8605 ms)
mixed      C8: TG base 83.42 81.80 new 83.73 84.02 -> +1.5% | outputs identical across 4 runs: True | matched-work requests pass 10 flag 1, unmatched samples 36, worst +5.5% (json_schedule 10311->10883 ms)
repetition C1: TG base 76.02 76.69 new 76.63 76.64 -> +0.4% | outputs identical across 4 runs: True | matched-work requests pass 1 flag 0, unmatched samples 0, worst -0.4% (repetition_word 1676->1670 ms)
repetition C2: TG base 103.75 104.46 new 108.41 108.37 -> +4.1% | outputs identical across 4 runs: True | matched-work requests pass 0 flag 0, unmatched samples 8, worst +nan% (- 0->0 ms)
repetition C4: TG base 112.63 113.15 new 113.64 113.65 -> +0.7% | outputs identical across 4 runs: True | matched-work requests pass 1 flag 0, unmatched samples 0, worst -0.7% (repetition_word 4536->4506 ms)
repetition C8: TG base 128.50 129.11 new 129.41 129.42 -> +0.5% | outputs identical across 4 runs: True | matched-work requests pass 2 flag 0, unmatched samples 0, worst -0.5% (repetition_word 8747->8707 ms)
{'pass': 41, 'flag': 1, 'inconclusive_samples': 57}
```

  C1은 그대로(−0.0 %, +0.4 %). C2 mixed +1.5 %, repetitive +4.1 %; C4 +2.5 %/+0.7 %; C8 +1.5 %/+0.5 %.
- **요청별 5 %·3 ms**: 같은 일(case, draft·accept 수, 출력 토큰이 같은 것)끼리만 비교. 41 통과, 1 표시(mixed C8 json_schedule 10311 → 10883 ms, +5.5 %). 같은 일의 기준 두 번이 10646·9977 ms로 6.7 % 달라 기준 분산이 gate(5 %)보다 크다 → **inconclusive로 남김**(퇴보로도 통과로도 판정하지 않음). 일이 다른 표본 57개(scheduler 교차로 draft 수가 달라짐, 기준끼리도 다름)는 비교하지 않음.
- **functional 타이밍**: 788 중 762 passed, 26 inconclusive, 모두 `batch_c2_greedy_short_replies_match_serial`(+ `execution_mode_probe` queue 5 ms). 같은 요청의 wall이 도착 순서에 따라 약 750/860 ms 두 값으로 갈리고(기준에서도 같은 요청이 753·858 ms), 이번 실행에서 순서가 바뀐 요청이 걸렸다. 출력 같음. 해소하려면 이 history만 기준/후보 교대 재실행 필요(미실행, 감독 판단).

### (c) 17–64행 폭 규칙 검색

규칙은 형식·shape class·행 수에만 의존(셀별 선택 없음). 후보 계열: 균형, 16, 짝수 올림, {12,14,16}, {14,16}, 고정 폭 9–16, 각각 16(·14)의 배수 행 제외 여부. 측정 격자 모든 칸에서 운영보다 느리지 않아야 채택.

- Q4_K·Q5_K·IQ4_XS 전 shape, Q6_K 5120×6144: **깨끗한 규칙 없음**(가장 덜 나쁜 것도 최악 칸 +2.7 – +24.5 %). 운영 그대로.
- Q6_K FFN(17408·34816×5120, 5120×17408): "짝수 폭(균형 폭을 짝수로 올림), 운영이 이미 14·16행 완전 group으로 한 번에 도는 행 수(14·16의 배수) 제외"가 42칸 모두 이김, 최악 칸 −0.48 %(34816 @62), 평균 −13 – −15 %. 같은 격자에서 고른 규칙이라 표본 안 결과다. 모델 효과는 작다: 이 층(down 2, packed 1, 개별 1)에서 걸음당 0.3–1.6 ms(200–450 ms 걸음의 0.1–0.5 %). **찾았지만 적용하지 않음**(감독 결정): 모델 효과가 걸음의 0.5 % 이하인데 build 1회와 확인 창 1개가 더 필요하고, 표본 밖 확인이 없다.

### dflash_policy 비용표

바꾸지 않았다. 비용표는 고정 draft 완전 cycle 측정이 필요하고, 이번 변경은 C2(10·14행 등)의 절벽을 줄여 비용표가 보수적이 된 상태다(출력 hash는 모두 같음. draft 수는 scheduler 교차에 따라 기준끼리도 달라지고, repetitive C2는 기준 110 → 후보 111로 두 번 모두 달랐다). 재측정은 별도 작업으로 제안.

## 창 `skinny-s-w3-a01/s-w3` (functional 타이밍 재확인)

창 FAILED_no_fallback, health PASS, 재시도 없음. s-w2와 같은 바이너리(후보 `6e5600d3…`, 기준 `22e4f9cf…`), `.gufo-revision` `ffdc3f6b`(4fc0211d와 문서만 다름). `batch` + `state-edges`를 `--through-case state-edges:execution_mode_probe`로 같은 history째 재생, 32768×2, DFlash2, 순서 기준(record) → 후보(기준0 대비) → 기준(기준0 대비, 대조) → 후보(기준2 대비). 마진 그대로(5 %, 3 ms).

|실행|비교|결과|
|---|---|---|
|fn3-0-base|record|passed|
|fn3-1-new|vs base0|inconclusive: 639 중 30, 거의 모두 `batch_c2_greedy_short_replies_match_serial`(wall 742 → 854 ms, inter-token 약 2배, cache_snapshot +) + queue 몇 ms; 품질·coverage 변화 0|
|fn3-2-base|vs base0 (기준끼리)|inconclusive: 639 중 48, **같은 case에서 같은 모양**(wall 742 → 857 ms, decode 105 → 214 ms)|
|fn3-3-new|vs base2|**failed**: `batch` suite가 `check_batches`의 `physical_execution_width > 1` assert에서 멈춤(16개 동시 쌍이 모두 `serial-fallback`, 한 번도 겹쳐 돌지 않음). 출력 텍스트는 serial과 같음. 비교는 만들어지지 않음. `state-edges` passed|

판정: s-w2의 `batch_c2_greedy_short_replies_match_serial` inconclusive는 기준끼리도 같은 크기로 나오므로 이 변경의 퇴보로 볼 근거가 없다. 그대로 **inconclusive**. fn3-3-new의 실패는 동시 요청이 batched로 겹치지 않은 coverage 실패로, 같은 바이너리의 fn3-1-new에서는 겹쳤다. 원인 미확인(재실행 안 함), 기록만.

## 창 `skinny-s-w4-a01/s-w4` (fn3-3 원인 확인)

창 FAILED_no_fallback(fn4-0-new의 assert), health PASS, 재시도 없음. 같은 바이너리, `.gufo-revision` `8dd3b7d2`(코드는 4fc0211d와 같음). `batch`를 `--through-case batch:batch_c2_greedy_short_replies_match_serial`까지 새 server로 8번, 순서 N B B N N B B N. 판정 규칙은 데이터 전에 감독이 정함: 모든 new 값 < 모든 base 값이거나, 두 절반(NBBN, NBBN) 모두 new 평균이 낮으면 변경 탓.

`batch_c2_greedy_short_replies_match_serial`의 16행 중 batched 행 수(`execution_plan` = `batched-w2`):

|창|순서대로|
|---|---|
|s-w2|base 6, new 2|
|s-w3|base 10, new 6, base 4, new 0(assert 실패)|
|s-w4|new 0(assert 실패), base 6, base 4, new 4, new 8, base 2, base 4, new 8|

s-w4: new {0, 4, 8, 8}, base {6, 4, 2, 4}. 모든 new < 모든 base 아님. 첫 절반 new 평균 2 < base 5, 둘째 절반 new 8 > base 3 → 규칙상 **변경 탓 아님, timing race**. base도 2/16까지 내려가고 new가 8/16까지 오른다.

결론: `check_batches`의 "16행 중 하나라도 batched" assert(`openai_sdk.py:788`)는 **이전부터 있던 flaky assert**다. 두 요청이 같은 verification에서 만나는지는 scheduler timing(prefill 교차, `decode_due`, `assemble_initial_batch`)이 정하고, 첫 공동 step 전에 도는 코드(prefill GEMM, 8행 이하 verify·draft)는 이 변경이 건드리지 않는다. RAM ledger(7.87·8.14 GB, 8 GiB 아래)와 admission 이벤트도 원인이 아니다. 테스트·마진은 바꾸지 않았다.

## 상태

(a)·(b) 채택. (c) 버림(위 Q6 FFN 규칙은 기록만).

## S2: DFlash2 Q4 비용표 재측정 (측정만, 표는 아직 그대로)

안내: [SKINNY-BRIEF.md](SKINNY-BRIEF.md) "S2". 기준 = 브랜치 HEAD `76b1660d`(코드는 `4fc0211d`와 같음, 그 뒤는 문서만).

### 방법 (원래 표의 방법, `docs/models/qwen3.8-27b/artifacts/q4-c{2,4,6,8}-focused.json`)

- 점마다 새 `gufo serve --sessions C … --draft-policy fixed --draft-tokens N`(N 1–7). speculative corpus `repetition_word`는 모두 accept되므로 매 cycle이 C·(N+1)행을 verify한다. greedy, 128 토큰, context 4096, warmup 1 + 측정 1 round, prompt cache 없음.
- C2 표: 계측 없는 wall cycle = 요청별 decode_ms / (completion − accepted − 1)의 평균(원래 C2 방법).
- C4·C6·C8 표: `rocprofv3 --kernel-trace`에서 측정 round의 포화 구간(dispatch 수가 같은 가장 긴 연속 cycle) 가운데 4 cycle의 span 중앙값. cycle 경계는 target verify의 `BatchedArgmaxKernel<true, false>`다. `dflash_selector_*`는 매 cycle 돌지 않고, 248320행 head GEMM은 행 수에 따라 template이 바뀌므로 경계로 쓰지 않는다.
- 상대 비용 = cycle(N)/cycle(1). context multiplier = C·100/cycle(1)(공식 그대로).
- 도구(추적 안 함, `build/s2-run/`): `cal.py`(창 안 driver, sha 6caa010f…), `cycles.py`(sha e7f3911b…), `analyze.py`, `choose.py`(`Choose` 재현, Python double). 요약은 `build/s2-results/`.
- 모델은 UD-Q4_K_M(S와 같음)이다. 원래 표는 **UD-Q4_K_XL**로 쟀는데, XL은 저장소에 없다.

### 창

- `skinny-s2-w1-a01/s2-w1`(PLAN-S2C.json): 결과 FAILED_no_fallback, health PASS, 재시도 없음. 이미지 `4a7046b2e8ed`, `/src` = 주 checkout, `.gufo-revision` `76b1660d`.
  - 바이너리: 기준(S 전) `build/s-src/build/s-base-release/gufo` `22e4f9cf…`, 새(HEAD) `build/s-src/build/s-release/gufo` `6e5600d3…`.
  - 순서: base plain C2–C8, new plain C2–C8, new rocprof C2·C4.
  - rocprof 아래 server는 SIGTERM에 db를 쓰고("tool finalization") 끝나지 않는다(rocprofv3의 chained signal handler). cal.py가 300 s 기다린 뒤 SIGKILL해서 점마다 약 5분이 더 들었다. 그래서 new-prof-c6 전에 deadline 검사에 걸렸고("1650 s left < 2400 s"), 마지막 `sleep 20`도 돌지 않았다. plain과 C2·C4 profile db는 모두 성한 상태다.
- `skinny-s2-w1b-a01/s2-w1b`(PLAN-S2C2.json, 감독 승인 (a)): COMPLETE, health PASS. new rocprof C6·C8만 돌렸다.
  - **cal.py 수정**: profile 모드에서 SIGTERM 뒤 5 s 안에 끝나지 않으면, server log에 `tool finalization`이 찍힌 지 2 s 뒤 SIGKILL한다. 300 s 안에 finalization이 없으면 실패로 기록한다. 그 밖의 방법은 같다.
- 모든 점: full acceptance, `batched-wC`, 완성 hash 하나(`24d4b2ba`), 요청마다 draft 63/84/95/101/105/108/111.

### cycle ms (UD-Q4_K_M, N = 1..7)

```
C2 old (Q4_K_XL)  104(N1 기준) rel 1.00 1.04 1.10 1.48 1.45 1.50 1.50
C2 base wall      87.3  92.0  98.6 137.9 136.0 144.9 147.6
C2 new  wall      87.1  92.4  99.3 122.0 134.7 137.7 148.2
C2 new  prof      86.2  91.3  98.4 121.2 134.9 138.0 147.2
C4 old            125   163   167   246   265   286   289
C4 base wall     104.3 140.5 149.0 214.4 230.7 249.6 266.2
C4 new  wall     104.6 137.5 149.0 213.3 230.5 249.3 265.5
C4 new  prof     102.5 138.8 149.0 220.5 236.4 265.4 271.7
C6 old            171   247   273   323   378   416   423
C6 base wall     143.9 210.0 234.0 281.1 324.7 353.3 372.1
C6 new  wall     140.3 208.3 234.1 280.8 323.9 351.9 372.0
C6 new  prof     141.4 216.4 238.6 298.8 347.2 393.0 395.6
C8 old            181   276   304   400   430   523   558
C8 base wall     156.0 237.5 268.4 349.1 371.6 440.2 467.7
C8 new  wall     155.3 235.4 267.2 342.1 370.0 438.7 465.2
C8 new  prof     154.8 243.2 274.3 367.0 397.3 495.3 526.3
```

### 상대 표 (old / base UD-Q4_K_M / new UD-Q4_K_M, 문서 방법: C2 wall, C4–C8 profile; base는 wall만 있음)

```
C2 old  1.00 1.04 1.10 1.48 1.45 1.50 1.50  mult 1.92
C2 base 1.00 1.05 1.13 1.58 1.56 1.66 1.69  mult 2.29 (wall)
C2 new  1.00 1.06 1.14 1.40 1.55 1.58 1.70  mult 2.30 (wall)
C4 old  1.00 1.30 1.33 1.97 2.12 2.29 2.32  mult 3.20
C4 base 1.00 1.35 1.43 2.06 2.21 2.39 2.55  mult 3.84 (wall)
C4 new  1.00 1.35 1.45 2.15 2.31 2.59 2.65  mult 3.90 (prof; wall 1.00 1.31 1.42 2.04 2.20 2.38 2.54)
C6 old  1.00 1.44 1.59 1.89 2.21 2.43 2.47  mult 3.50
C6 base 1.00 1.46 1.63 1.95 2.26 2.46 2.59  mult 4.17 (wall)
C6 new  1.00 1.53 1.69 2.11 2.46 2.78 2.80  mult 4.24 (prof; wall 1.00 1.48 1.67 2.00 2.31 2.51 2.65)
C8 old  1.00 1.52 1.68 2.21 2.38 2.89 3.08  mult 4.42
C8 base 1.00 1.52 1.72 2.24 2.38 2.82 3.00  mult 5.13 (wall)
C8 new  1.00 1.57 1.77 2.37 2.57 3.20 3.40  mult 5.17 (prof; wall 1.00 1.52 1.72 2.20 2.38 2.83 3.00)
```

관찰:
- **S의 효과는 작다.** base→new wall 차이는 C2 N4(10행) −11.5 %, C2 N6(14행) −5.0 %뿐이다. C4 N2 −2.1 %, C8 N4 −2.0 %, 그 밖은 ±1 % 안이다. S의 (a)는 12·16행이 아닌 9–16 폭만 바꿨고, Q4 shape의 17행 이상은 그대로이기 때문이다.
- **차이의 대부분은 모델이다**(Q4_K_XL → Q4_K_M). N1 cycle이 104/125/171/181 → 87/103/141/155 ms로 짧아진다. 그래서 상대 비용과 multiplier가 커지고 정책은 더 짧은 block을 고른다.
- **방법 차이**: C4–C8에서 profile이 wall보다 최대 +12 %(C8 N7 526 vs 465) 크다. wall은 동료가 prefill하는 동안의 ramp와 요청이 끝나는 꼬리(span 82–336 ms)를 평균에 넣는다. profile은 포화 cycle만 본다. C2에서는 두 방법이 ±1.5 % 안에서 같다.

### 정책이 고르는 draft 수 (old 표 → new 표, `Choose` 재현, Q4 adaptive greedy, limit 7)

acceptance 평균 mean(0.5–7, 7 = full, C5–C8은 full이면 probe 7) × position 0/8192/32768:

```
mean          0.5 1 1.5 2 2.5 3 3.5 4 4.5 5 5.5 6 6.5 7
C2 d0    old    2 3  3  3  3  3  7  7  7  7  7  7  7  7
C2 d0    new    2 2  3  3  3  3  3  3  3  6  6  6  7  7
C2 d8K   old    2 2  3  3  3  3  3  7  7  7  7  7  7  7
C2 d8K   new    2 2  3  3  3  3  3  3  3  3  6  6  6  7
C2 d32K  old    2 2  2  2  3  3  3  3  7  7  7  7  7  7
C2 d32K  new    1 2  2  2  2  3  3  3  3  3  6  6  6  7
C4 d0    old    1 1  3  3  3  3  3  3  3  3  3  3  3  7
C4 d0    new    1 1  1  1  3  3  3  3  3  3  3  3  3  7
C4 d8K   old    1 1  1  3  3  3  3  3  3  3  3  3  3  7
C4 d8K   new    1 1  1  1  3  3  3  3  3  3  3  3  3  7
C4 d32K  old    1 1  1  1  3  3  3  3  3  3  3  3  3  7
C4 d32K  new    1 1  1  1  1  1  3  3  3  3  3  3  3  7
C6 d0    old    1 1  1  1  1  1  3  3  3  3  3  3  7  7
C6 d0    new    1 1  1  1  1  1  1  1  1  3  3  3  3  7
C6 d8K   old    1 1  1  1  1  1  1  3  3  3  3  3  7  7
C6 d8K   new    1 1  1  1  1  1  1  1  1  1  3  3  3  7
C6 d32K  old    1 1  1  1  1  1  1  1  1  3  4  4  4  7
C6 d32K  new    1 1  1  1  1  1  1  1  1  1  1  1  1  7
C8 d0    old    1 1  1  1  1  1  1  1  1  3  3  3  3  7
C8 d0    new    1 1  1  1  1  1  1  1  1  1  1  1  1  7
C8 d8K   old    1 1  1  1  1  1  1  1  1  1  3  3  3  7
C8 d8K   new    1 1  1  1  1  1  1  1  1  1  1  1  1  7
C8 d32K  old    1 1  1  1  1  1  1  1  1  1  1  1  1  7
C8 d32K  new    1 1  1  1  1  1  1  1  1  1  1  1  1  7
```

새 표는 모든 C에서 같거나 더 짧은 block을 고른다(C2 mean 3.5–4.5에서 7 → 3, C6·C8 높은 acceptance에서 3 → 1). `dflash_test.cpp`의 mean 3 기대값(C2–C4 3, C5–C8 1)은 두 표 모두에서 성립한다.

### 대상 모델과 표의 key

- 문서의 운영 Q4 target은 **UD-Q4_K_XL**이다(`docs/models/qwen3.8-27b/README.md` 3·13·18행 download, `BENCHMARKS.md` 4행, 저장소 `README.md` 표). 이 checkout에는 Q4_K_M target 안내가 없다(Q4_K_M은 DFlash2 draft).
- 표는 quant 형식이 아니라 head 형식으로 key된다: `backend.cpp:195` `output.type == kQ8_0`이면 Q8 표, 아니면 Q4 표다. UD-Q4_K_XL과 UD-Q4_K_M은 둘 다 Q6_K head라서 **같은 Q4 표를 쓴다**. Q4_K_M에서 잰 값으로 바꾸면 문서 대상인 Q4_K_XL 사용자도 바뀐다.

### 상태 (Q4_K_M 측정 시점)

이 시점에는 `dflash_policy.hpp`를 바꾸지 않았다(채택은 모델 보정 결정). 사용자 결정에 따라 아래에서 UD-Q4_K_XL로 다시 쟀다.

## S2-XL: UD-Q4_K_XL 재측정, 표 변경, 확인

### 모델

`unsloth/Qwen3.8-27B-GGUF@4ca720788d1e01f1bff70c033e0d0028fd02e502` `Qwen3.8-27B-UD-Q4_K_XL.gguf`(README의 고정 revision). curl로 받았다(호스트에 Nix 없음).
- 크기 17559178144 B, sha256 `3f227079003add2511437e5b1e94812e363385225bf6a9b47b0054a72bc8b01e`. hub LFS 값과 같다.
- 위치: `/models/qwen`으로 mount되는 `storage/models/llm/qwen-3.8-27b-ud-q4-k-m-gguf/`, 옆에 `source-UD-Q4_K_XL.json`. 남은 디스크 233 GB.

### 창 `skinny-s2-w2-a01/s2-w2` (PLAN-S2X.json)

COMPLETE, health PASS, 재시도 없음. `.gufo-revision` `ea4ebf08`(코드는 HEAD와 같음). 방법은 S2와 같다(수정된 cal.py).
- base(22e4f9cf) wall C2–C8, new(6e5600d3) wall C2–C8, new rocprof C4·C6·C8.
- 모든 점이 full acceptance, `batched-wC`, hash 하나. 요약은 `build/s2-results/s2-summary-xl.json`.

```
XL cycle ms N=1..7          (C2 new = wall, C4–C8 new = profile, base = wall)
C2 base 90.7  94.3  99.5 138.9 138.4 144.3 144.3 | new 90.2  94.7 100.1 120.8 133.2 136.1 145.2
C4 base 105.1 142.3 146.2 214.7 229.0 247.1 260.9 | new 104.4 134.1 143.5 221.6 233.3 257.6 264.1
C6 base 145.7 210.8 232.7 278.2 322.3 348.3 364.8 | new 140.2 217.6 237.4 286.7 341.0 379.2 398.8
C8 base 153.2 235.8 263.7 344.4 365.3 433.3 457.7 | new 153.9 240.4 276.2 368.9 392.4 481.1 514.5
상대   old                                 | XL base (wall)                       | XL new
C2 1 1.04 1.10 1.48 1.45 1.50 1.50 m1.92 | 1 1.04 1.10 1.53 1.53 1.59 1.59 m2.21 | 1 1.05 1.11 1.34 1.48 1.51 1.61 m2.22
C4 1 1.30 1.33 1.97 2.12 2.29 2.32 m3.20 | 1 1.35 1.39 2.04 2.18 2.35 2.48 m3.80 | 1 1.28 1.38 2.12 2.24 2.47 2.53 m3.83
C6 1 1.44 1.59 1.89 2.21 2.43 2.47 m3.50 | 1 1.45 1.60 1.91 2.21 2.39 2.50 m4.12 | 1 1.55 1.69 2.05 2.43 2.70 2.85 m4.28
C8 1 1.52 1.68 2.21 2.38 2.89 3.08 m4.42 | 1 1.54 1.72 2.25 2.38 2.83 2.99 m5.22 | 1 1.56 1.79 2.40 2.55 3.13 3.34 m5.20
```

- XL에서 S의 효과(base → new wall): C2 N4 −13 %, N6 −5.7 %, C4 N2 −4 %. 그 밖은 ±1.5 % 안이다.
- XL base의 상대값은 old와 비슷하다. 다만 N1 cycle이 짧아져(C2 104 → 90, C4 125 → 105 ms) multiplier가 커진다. 원래 표를 잰 뒤 kernel이 바뀌었고, 기본 KV도 지금은 `kvarn-k4v2`다.
- C4–C8 new 표는 profile 값이라 높은 N에서 wall보다 크다(원래 C4–C8 방법과 같음).

정책 선택 replica, old → new: 대체로 같거나 짧아진다.
- C2: mean 3.5–4에서 7 → 3. mean 4.5–5에서 d0은 7 → 6, d32K는 7 → 3.
- C4: d0 mean 1.5에서 3 → 1, d32K mean 2.5–3.5에서 3 → 2.
- C6: mean 3.5–4.5에서 3 → 1, mean 6.5에서 7 → 3, d32K mean 5–6.5에서 → 1.
- C8: d0·d8K mean 5–6.5에서 3 → 1.

### 운영 변경 `eb78772f` perf(qwen): recalibrate Q4 greedy DFlash2 verification costs

- `dflash_policy.hpp`의 `kQ4Paired/FourRequest/SixRequest/EightRequestRelativeCost`, multiplier 1.92/3.20/3.50/4.42 → 2.22/3.83/4.28/5.20, 주석의 ms와 모델 이름만 바꿨다. 공식·마진·Q8 표는 그대로다.
- format check PASS(693 파일).
- 테스트 기대값은 바꾸지 않았다. replica로 본 mean 3 기대값은 그대로 성립했다. **아래 full-acceptance 기대값은 성립하지 않는다.**
- 빌드: 이미지 컨테이너에서 `build/s2-release/gufo` `94dc3f43…`, `build/s2-gpu-test` ops test `4c03bd5d…`, dflash test `cc29e04a…`. 스크립트는 `build/s2-build.sh`.

### 창 `skinny-s2-w3-a01/s2-w3` (PLAN-S2V.json, 확인)

FAILED_no_fallback, health PASS, 재시도 없음. `.gufo-revision` `eb78772f`. base = old 표(`build/s-src/build/s-release/gufo` 6e5600d3, 그 밖의 코드 같음), new = `eb78772f`. 요약은 `build/s2-results/cohort-xl.txt`, `cohort-km.txt`(`build/s2-run/cohort_an.py`).

- `qwen_q4kxl_quant_ops_test`: PASS.
- `qwen_dflash_gpu_test XL --concurrency-only`: **FAIL**. 첫 단계 `TestLengthController`(`dflash_test.cpp:66`)에서 멈췄다: "batched full acceptance retains the full block at every depth".
  - 새 표에서 C3·C4 full acceptance가 262144 위치에서 7 대신 5를 고른다(replica 확인; 131072까지는 7). old 표는 모든 깊이에서 7이다.
  - 이 test는 처음에 멈췄으므로 **C2–C8 exact layers/logits/selector와 private RNG/sampled replay는 이 창에서 확인되지 않았다.**
- functional `batch`+`state-edges` XL: fn-base passed(record). fn-new는 `batch`가 `check_batches`의 `physical_execution_width > 1` assert(`openai_sdk.py:788`)에서 멈췄다. 16 쌍이 모두 serial이었다. `state-edges`는 passed다. 이는 s-w3·s-w4에서 기준끼리도 나온 timing race assert와 같다. 비교는 만들어지지 않았으니 **inconclusive**다.
- cohort XL(base/new/base/new, greedy DFlash2, mixed 9 case + repetition, 128 토큰, context 4096, C마다 새 server). 출력 hash는 모든 C·workload에서 네 번 같다(AR 경로와 같은 greedy 출력).

```
mixed      C1: base 37.18 37.04 new 37.04 37.02 -> -0.2%
mixed      C2: base 57.03 58.08 new 56.91 57.75 -> -0.4%  drafts/acc base 1547/833 new 1433/823; per-case flag 1: reasoning_train 3141->3307 ms (+5.3%, base 3158/3124, new 3298/3317)
mixed      C4: base 74.95 74.75 new 74.58 74.24 -> -0.6%  drafts/acc base 1463/958 new 1439/949; flag 0
mixed      C8: base 90.91 89.72 new 90.87 90.44 -> +0.4%  drafts/acc base 1530/1094 new 1342/983; per-case flag 3:
             cpp_ring_buffer 10591->11360 (+7.3%, base 10561/10621, new 11465/11255), reasoning_train 9536->10302 (+8.0%), json_schedule 10883->11433 (+5.1%)
repetition C1: -0.3%  C2: base 106.88 110.85 new 107.52 108.09 -> -1.0%  C4: +0.5%  C8: +0.3% (drafts 848 -> 823)
UD-Q4_K_M 짝(base/new 1회): mixed C2 55.40 -> 54.56 -1.5% (flag reasoning_train +5.9%, summary_gpu 1350->1647 +22%), mixed C4 71.57 -> 73.35 +2.5%, repetition C2 +0.5%, C4 +0.4%
```

  - 요청별 gate(5 %·3 ms). 같은 일(matched-work) 비교는 18 통과, 0 표시다. 하지만 표가 draft 수를 바꾸므로 표본 157개가 unmatched다.
  - case별 비교(case의 모든 요청, base 평균 vs new 평균)는 36 통과, 4 표시다(XL). 표시된 C8 3 case와 C2 reasoning_train은 두 new 실행이 모두 두 base 실행보다 느리다.

### 판정 (감독에게)

- **처리량 이득 없음**(XL C2–C8 −1.0 – +0.5 %, 잡음 안).
- **요청별 퇴보 표시**가 있다: C8 mixed 3 case +5–8 %, C2 reasoning_train +5.3 %. 새 표가 C8에서 draft를 덜 제안해 accept 토큰이 1094 → 983으로 준다.
- dflash test의 full-acceptance 기대값이 깨지고, 이 때문에 정확도 test가 돌지 않았다.
- 지시대로 되돌리거나 넓히지 않았다. `eb78772f`는 브랜치에 그대로 있고, 채택 여부는 감독이 정한다.
