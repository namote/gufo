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
