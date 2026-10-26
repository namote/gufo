# gufo KV 양자화 후보: KVarN (U11, 2026-10-23)

상위 문서: [이식 계획](PLAN.md), [모델 확장](MODELS.md), [테스트 설계](TESTS.md).

사용자 지시(2026-10-23): "KVARN 지원도 계획에 넣어줘 가능한가? 웹으로 조사한번 해보고". 이 문서는 조사와 설계만 담는다. 구현·GPU 실행은 하지 않았다.

## KVarN이 무엇인가 (웹 조사, 공개 자료 기준)

- **출처:** Huawei Computing Systems Lab, 논문 arXiv 2606.03458(2026-06-02), 코드 `github.com/huawei-csl/KVarN`(Apache-2.0, vLLM v0.23.0 위의 attention backend).
- **방법:** calibration 없는 KV 양자화.
  - head 채널 방향 Hadamard 회전(직교라 attention 점수 보존) 뒤, 행·열 표준편차를 번갈아 맞추는 이중 scale 정규화(log 공간, 층마다 8회 반복)를 한 다음 비대칭 round-to-nearest로 양자화한다.
  - 논문 주장: 오차가 생성 단계마다 쌓이고, 원인은 주로 토큰 scale이 틀리는 것이다. 이중 정규화가 그 누적을 줄인다.
- **배치:** 앞부분 sink 토큰은 FP16, 몸통은 128토큰(또는 64토큰) 그룹 단위로 양자화, 아직 그룹을 채우지 못한 최근 토큰은 FP16 tail로 두고 128개가 차면 양자화한다.
- **형식:** 저장소 기본 preset `kvarn_k4v2_g128`(K 4비트는 채널별 scale, V 2비트는 토큰별 scale). 논문의 2비트 설정은 scale 포함 약 2.3비트/원소. 읽을 때 두 번째 scale을 attention 커널 안에서 곱한다(fused dequant).
- **공개 결과 (우리 장비 결과가 아님):**
  - Qwen3-32B, 16K, TP=2: FP16 수준 정확도, 처리량 약 1.3배, KV 용량 약 4배.
  - GLM-4.7-Flash(MLA, latent int4): 32K 처리량 0.94배, 용량 2.77배, AIME25 동률.
  - Qwen3-4B 2비트: AIME24 60.0 %(KIVI 55.5 %), MATH500 79.2 %, HumanEval 88.4 %.
  - Qwen3.6-27B에서도 시험했다고 적혀 있다. hybrid 모델은 full-attention 층만 압축하고 linear-attention·Mamba 상태는 그대로 둔다. MTP와 DFlash(비인과 cross-attention) 투기 디코딩과 같이 쓸 수 있다고 한다.
- **공개되지 않은 것(unknown):**
  - 하드웨어는 NVIDIA뿐이고, 커널은 Triton JIT다. ROCm·gfx1151 언급은 없다.
  - prefix caching과의 관계, 256K 이상 길이의 결과, 우리 Qwen3.8 27B의 결과는 나와 있지 않다.

## 우리에게 맞는가

**맞는 점:**
- **구조:** Qwen 27B는 hybrid다. KV는 full-attention 층에만 있고(FP16, 약 64 KiB/토큰, U3 측정 256K×4 = 64 GiB), recurrent 상태는 고정 크기다. KVarN이 hybrid에서 full-attention 층만 압축하는 방식과 같다.
- **투기 디코딩:** 운영은 DFlash2를 쓰고, KVarN은 DFlash와 함께 시험됐다.
- **블록 정렬:** 우리 블록(D5, 256토큰)은 KVarN 그룹(128)의 정확히 2배다. 원본 chunk(256행)와도 맞는다. 그룹 경계가 블록 안에서 끝나므로 블록 테이블·예약·CoW 단위를 바꿀 필요가 없다.
- **기대 효과 (k4v2 기준 계산, 측정 아님):** KV가 원소당 4바이트(K+V FP16)에서 scale 포함 약 0.8–1바이트로 줄어 약 4–5배.
  - 같은 40 GiB 풀에 256K 세션 약 10개, 또는 256K×4를 10–16 GiB로.
  - RAM·디스크 checkpoint도 같은 비율로 줄어든다. F에서 잰 256K 디스크 checkpoint 17 GB는 약 4 GB가 되고, 복원 약 4 s는 약 1 s가 된다.
  - 256K decode는 KV 읽기 대역폭이 크므로 gfx1151(LPDDR5X)에서 이득일 수 있다. 다만 이것은 가설이고 측정해야 한다.

**어려운 점:**
- **커널 이식:** 커널을 HIP로 새로 써야 한다. 대상은 decode attention(fused dequant), WMMA prefill이 앞 그룹을 읽는 경로, long-context·block-table 커널, 그룹 완성 시 양자화 커널(행·열 8회 반복)이다. Triton 코드는 알고리즘 참고로만 쓴다.
- **출력 동일 gate가 깨진다.** 지금까지의 "텍스트 동일" 대신 품질 gate가 필요하다(아래).
- **checkpoint 형식:** chunk에 양자화 블록·scale·FP16 sink/tail을 함께 담아야 한다. 경계가 그룹 중간이면 tail은 FP16으로 저장한다. 디스크 형식 버전이 바뀐다.
- **공통화:** 연산 계약(head dim, GQA 비율, 층 종류)으로 dispatch한다. Gemma 4 sliding window 층, DeepSeek MLA(latent는 다른 방식)는 계약이 다르므로 별도 사례로 둔다([모델 확장](MODELS.md)).
- **원본 정책:** 원본 gufo에는 KV 양자화가 없다. 제안하려면 "성공한 최적화는 기본값" 규칙과 정확도 변화를 원본이 받아들일지 물어야 한다(U6 이슈와 별개).

## 단계 (각 단계 끝에 사용자 확인)

|단계|내용|통과 기준|
|---|---|---|
|K0 오프라인 정확도|GPU 커널 없이 시작한다. gufo가 만든 FP16 KV를 실제 prompt(32K, 256K gold 문서)에서 덤프하고, KVarN 참고 구현(Python)으로 양자화한 KV로 attention 출력 오차, 다음 토큰 분포(KL), top-1 일치율을 FP16과 비교한다. k4v2·k4v4·k8v4, 그룹 128·64를 같은 입력으로 잰다|설정별 오차표. k4v2가 KIVI식 단순 양자화보다 오차가 작고, 256K에서 top-1 일치가 충분한지(기준은 K0 결과로 사용자와 정함)|
|K1 커널|HIP 양자화 커널, fused dequant decode 커널, 블록 테이블 경로. 옵션은 명시 선택(`--kv-cache-type` 같은 이름, 기본은 FP16), 지원하지 않는 모델·층 조합은 시작 시 이름을 들어 거절|커널 단위 테스트: 참고 구현과 비트 단위(양자화 값) 일치, dequant attention 오차가 K0 값과 같음|
|K2 서빙|component cache checkpoint·디스크 형식, 입장 예약의 블록 계산, DFlash2 draft 경로|T2 경우 전부 gold 정답. FP16 대비 텍스트 차이를 수로 기록(같다고 주장하지 않음). DFlash2 수락률 변화 기록|
|K3 실제 길이|최종점검 F와 같은 창(32K, 256K, 256K×4, 64K×8, 대화 회귀, bench)을 FP16과 번갈아 실행|gold 전부 정답, 256K 문서 질문 정답, 메모리·디스크·TTFT·decode 속도 표. 이득이 없거나 품질이 떨어지면 채택하지 않고 기록|

## 순서와 전제

- 최종점검 F 결과와 운영 전환 결정 뒤에 시작한다. U4(checkpoint 블록 참조) 여부가 chunk 형식에 영향을 주므로 U4 재개 결정을 먼저 내린다.
- K0은 CPU·GPU 시간이 작아 먼저 해 볼 수 있다. 결과가 나쁘면 K1 이후는 하지 않는다.
- 자동 fallback은 없다. KVarN 경로가 실패하면 FP16으로 바꿔 다시 돌리지 않고 실패를 기록한다. FP16과의 비교는 별도 창이다.
- 출처 표기: Apache-2.0이므로 알고리즘·코드를 참고하면 NOTICE와 출처를 남긴다.

## K0 설계 (2026-10-26)

사용자 지시(2026-10-26): "KVarN 적용". 위 "순서와 전제"는 운영 전환 결정 뒤 시작을 적었으나, 이 지시로 KVarN을 앞당겨 K0부터 시작한다. K0은 설계까지만 했다. GPU 창은 실행 전에 사용자 확인을 받는다.

### 참고 구현 기준 (저장소 코드, 논문과 다른 점)

`huawei-csl/KVarN` `main`과 논문 v1을 읽었다(실행·clone 없음). 논문과 코드가 다른 곳은 **코드를 기준**으로 한다.

|항목|논문|코드(기준)|
|---|---|---|
|scale 형식|FP8/8비트, 2/2비트|fp16 scale, k4v2|
|정규화 clamp|분산 clamp, 0.5·log(var)|std를 [1e-3, 1e3]로 clamp, log(std), 누적 log scale은 [−0.3, 10]|
|V Hadamard|W_V/W_O에 흡수(QuaRot식)|저장 때 V를 회전, attention 출력을 되돌림|
|반복 횟수|곳에 따라 16|8(`KVARN_SINKHORN_ITERS`), 불균형이 가장 작은 반복의 scale 채택|

- 회전: 정규화한 Sylvester Hadamard(1/√D, 무작위 부호 없음)를 head_dim 방향으로 모든 토큰(sink·tail 포함)에 곱한다. q도 같은 H로 회전하고 출력에 H를 다시 곱한다(H⁻¹ = H).
- tile: (128토큰 블록, KV head)마다 K는 `[D, G]`(행 = 채널), V는 `[G, D]`(행 = 토큰). 정규화 뒤 행마다 비대칭 RTN이다(scale = max(범위/qmax, 1e-10), zero = 행 최솟값). 그래서 K는 128토큰에 걸친 채널별, V는 토큰별 scale이다. 다른 축 scale은 dequant에서 곱한다.
- sink: 요청의 첫 블록(128토큰)은 끝까지 FP16. tail: 채워지지 않은 블록은 FP16이고, 블록이 차고 모든 토큰이 확정(투기 디코딩 수락)되면 양자화한다.
- preset: `kvarn_k4v2_g128`, `kvarn_k4v4_g128`, `kvarn_k4v2_g64`, `kvarn_k4v4_g64`. 저장소에 KL·top-1 평가 도구는 없다(prefix cache·MTP on/off 답 동일 검사뿐). 논문의 평가는 pseudo-decode(prefill cache 양자화 뒤 이어서 생성), line retrieval, NIAH다.
- ROCm 언급은 없다. head_dim은 128·256·512를 지원한다.
- 조사 사본: `/tmp/claude-1000/-home-m2on-projects-gufo/635eb9f0-9c85-40e5-b405-740790b62b96/scratchpad/kvarn/`(임시).

### Qwen 27B에 맞춘 값

- full-attention 층 16개(64층 중 `(layer+1)%4==0`), Q head 24, KV head 4, head_dim 256(GQA 6:1). KV는 RoPE 뒤 FP16, 토큰당 64 KiB(`src/models/qwen/hip/arena.cpp:976`, `continuation_layout.cpp:64`).
- head_dim 256이라 H는 256×256이고, tile은 K `[256, 128]`, V `[128, 256]`이다.
- k4v2_g128 tile(128토큰·KV head 하나) 크기: K 16,384 + scale 1,280, V 8,192 + scale 1,024 = 26,880 B. FP16 131,072 B 대비 **4.88배**(K 4.31비트, V 2.25비트/원소). 토큰당 64 → 13.1 KiB, 256K 세션 KV 16 → 3.3 GiB, 256K 디스크 checkpoint 약 17 → 3.5 GB(계산, 측정 아님).
- DFlash2와 MTP는 target KV를 읽지 않는다(자체 ring·KV). 그래서 K0·K1 대상은 target attention뿐이다.
- 지금 Qwen KV dtype은 FP16/FP32뿐이고, KV·Q·logits 덤프 기능은 없다. 저장소의 양자화 KV 선례는 DeepSeek V4 FP8 KV(`src/models/deepseek_v4_flash/kernels/rocm/detail/ds4_rocm_fp8_kv*.hpp`)뿐이다.

### K0 방법: 엔진 안 fake quant + teacher forcing

H가 직교이므로 원래 좌표의 `K' = dequant(quant(K·H))·H`로 바꿔 넣으면 attention 점수와 출력이 회전 좌표 계산과 같다. 그래서 K0은 attention 커널을 바꾸지 않는다.

1. 시험 도구(`tests/models/qwen27b/`, GPU 테스트 바이너리)가 prompt를 FP16으로 prefill한 뒤 16층 K/V plane을 파일로 내보낸다. FP16 greedy로 N=256토큰을 생성해 토큰과 logits를 기준으로 남긴다.
2. Python 참고 구현(`tools/qwen27b/kvarn/`, NumPy, 출처·Apache-2.0 표기)이 prefill KV에 설정별 fake quant를 적용한다. sink 첫 128토큰과 마지막 미완성 블록은 FP16 그대로 둔다.
3. 도구가 바꾼 KV를 다시 넣고 같은 N토큰을 teacher forcing으로 디코드한다(새 토큰 KV는 tail FP16, 128을 넘기지 않음). 걸음마다 logits를 기록한다. 이어서 free greedy로 같은 길이를 생성해 텍스트와 gold 답을 본다.
4. 지표: 걸음별 KL(FP16‖양자화), top-1 일치율, 같은 위치 top-1 확률 차, free greedy 첫 분기 위치와 gold 정답. 요청이나 걸음을 평균으로 덮지 않고 분포(중앙값·p95·최대)로 기록한다. 추가로 층·head별 K/V 재구성 상대 오차를 Python에서 잰다.

비교 설정(같은 KV 입력):

|설정|목적|
|---|---|
|FP16 재주입(변화 없음)|도구 왕복 검사. KL 0, 텍스트 동일이어야 한다|
|KIVI식 RTN(회전·정규화 없음) k4v2_g128|단순 양자화 기준|
|Hadamard + RTN k4v2_g128|정규화 효과 분리|
|KVarN k4v2_g128, k4v4_g128, k8v4_g128|형식 선택(K-D2)|
|KVarN k4v2_g64|그룹 크기|

prompt: 32K 대화 gold 1개, 256K 문서 gold 1개(F와 같은 입력). 256K 덤프는 16 GiB라 파일 하나로 두고 층 단위로 처리한다.

예상 GPU 시간(추정): 256K prefill 약 15분 1회 + 설정 7개 × (재주입 + 256걸음 teacher forcing + free greedy), 32K는 그보다 훨씬 작다. 합쳐 1시간 안을 목표로 한 창에서 끝낸다. 디스크 임시 사용 약 40 GB.

### 속도 (2026-10-26 사용자: "속도도 고려해줘")

fake quant는 FP16 커널을 그대로 쓰므로 속도를 재지 못한다. 그래서 속도는 두 단계로 본다.

1. **K0 상한 측정(이번 창):** 이미지의 release `gufo bench`로 AR과 DFlash2 decode를 깊이 0·32K·64K에서 재고, 같은 실행을 rocprofv3 kernel trace로 다시 돌려 decode 시간을 attention과 나머지로 나눈다. 깊이에 따른 attention 증가분이 KV 읽기 비용이다. 계산(측정 아님): 토큰 하나가 KV를 깊이 × 64 KiB 읽으므로 32K는 2 GiB, 256K는 16 GiB를 읽는다. KVarN이면 4.88배 줄어든다. 그래서 이득은 깊은 문맥 decode에 몰리고, 얕은 문맥에서는 거의 없다. DFlash2 verify는 한 번 읽어 여러 토큰을 검증하므로 AR보다 이득이 작을 수 있다. 256K는 32K·64K 기울기로 외삽하고 외삽이라고 적는다.
2. **K1 진입 조건(정확도 통과 뒤):** `tools/bench/`에 fused dequant decode attention(head_dim 256, KV head 4, GQA 6, k4v2·g128, split-K) 시제품을 만들고, 같은 모양의 지금 FP16 split-K 커널과 깊이 32K–256K에서 비교한다. 측정한 대역폭과 `tools/bench/gfx1151_peak.hip` 상한을 함께 적는다. 양자화(그룹이 찰 때 128토큰 × 32 tile 정규화 8회)의 decode 1토큰당 분할 비용과 prefill 증가분도 잰다. 커널 속도 이득이 없으면 정확도가 좋아도 K1로 가지 않는다.

### 구현 (2026-10-26, gufo 작업 트리, 커밋 전)

- `tests/models/qwen27b/kvarn_fake_quant.hpp`: C++ fake quant(위 코드 기준).
- `tests/models/qwen27b/kvarn_k0.cpp` → `qwen27b_kvarn_k0`(GPU, CTest 없음): 위 K0 방법 1–4. `--prefill-chunk N`이면 N토큰씩 prefill하면서 다 찬 그룹을 다음 chunk 전에 양자화한다. 실제 KVarN은 첫 prefill chunk 뒤 chunk들도 양자화된 앞 블록을 읽으므로, pseudo-decode보다 이것이 실제에 가깝다(chunk 크기만큼은 FP16으로 서로 본다). 32K에서만 돈다.
- `tools/qwen27b/kvarn.py`: 독립 NumPy 참고 구현과 `check`(CTest `qwen27b.kvarn-reference`, CPU), `summarize`.
- 교차 검사 50경우 PASS: rtn·hrtn은 FP16 비트 단위로 같고, kvarn은 정규화의 float 누적 순서 차이로 FP16 scale이 1 ulp 갈린 tile만 다르다(구현 차이 ≤ 양자화 오차의 2 %).
- 근사(실제 KVarN과 다른 점): 원래 좌표로 되돌려 FP16에 다시 쓰는 반올림 1회가 더 있고, sink·tail 행은 회전 좌표의 FP16 반올림을 거치지 않는다.
- CPU 양자화 시간: 32K 토큰 component 하나 0.3 s(32스레드). 256K 설정 하나에 약 1–2분.
- 창 계획: ai.tools `test/experiments/gufo_upstream_port_20261023/PLAN-K0.json`(`k0-32k`: 설정 7개 + chunk 4096 설정 3개 + bench·profile, `k0-256k`: 설정 5개), 입력·채점 `kvarn_k0.py`, fixture `kvarn-user-{32k,256k}.txt`·`kvarn-gold.json`.

### K0 결과 (2026-10-26, gufo `b3edc894`)

창: ai.tools `kvarn-k0-a01/k0-32k`(정확도·bench는 COMPLETE, 프로파일은 rocprofv3가 읽기 전용 `/src`에 임시 버퍼를 쓰려다 멈춰 900 s timeout, 창 판정 FAILED_no_fallback, health PASS), `kvarn-k0-a02/k0-prof`(`ROCPROF_TMPDIR=/receipt/run`로 프로파일만 다시, COMPLETE), `kvarn-k0-a02/k0-256k`(COMPLETE). 모든 창 health PASS, 자동 fallback 없음. 런타임 이미지 `ai-tools/gufo-eval:663d50c9f9d6`, 모델 Qwen3.8-27B UD-Q4_K_M, think off, AR greedy.

**정확도.** 32K(32,770토큰)와 256K(254,880토큰) 문서, 기준 답 214토큰.

|prompt|설정|K 오차|V 오차|KL 중앙값|KL p95|KL 최대|top-1|free 텍스트|gold|
|---|---|---|---|---|---|---|---|---|---|
|32K|fp16(왕복)|0|0|0|0|0|214/214|같음|정답|
|32K|rtn_k4v2_g128|0.052|0.894|4.1e-6|1.2e-4|4.4e-3|214/214|같음|정답|
|32K|hrtn_k4v2_g128|0.056|0.538|1.4e-6|9.0e-5|1.0e-3|214/214|같음|정답|
|32K|kvarn_k4v2_g128|0.056|0.596|1.6e-6|1.0e-4|9.5e-4|214/214|같음|정답|
|32K|kvarn_k4v4_g128|0.056|0.118|2.7e-7|1.9e-5|4.5e-4|214/214|같음|정답|
|32K|kvarn_k8v4_g128|0.003|0.118|1.0e-7|4.7e-6|5.0e-5|214/214|같음|정답|
|32K|kvarn_k4v2_g64|0.049|0.609|1.8e-6|8.2e-5|3.8e-3|214/214|같음|정답|
|32K chunk 4096|rtn_k4v2_g128|0.051|0.897|5.9e-6|3.4e-4|5.5e-2|214/214|같음|정답|
|32K chunk 4096|kvarn_k4v2_g128|0.055|0.595|4.0e-6|1.7e-4|1.8e-2|214/214|같음|정답|
|32K chunk 4096|kvarn_k4v4_g128|0.056|0.118|1.2e-6|4.2e-5|8.6e-4|214/214|같음|정답|
|256K|fp16(왕복)|0|0|0|0|0|214/214|같음|정답|
|256K|rtn_k4v2_g128|0.044|0.910|3.3e-6|2.0e-4|1.1e-2|214/214|같음|정답|
|256K|kvarn_k4v2_g128|0.048|0.563|1.6e-6|1.2e-4|6.7e-3|214/214|같음|정답|
|256K|kvarn_k4v4_g128|0.048|0.112|1.6e-7|1.5e-5|2.1e-4|214/214|같음|정답|
|256K|kvarn_k8v4_g128|0.003|0.112|7.2e-8|1.0e-5|7.0e-5|214/214|같음|정답|

- fp16 왕복은 모든 걸음에서 logits가 비트 단위로 같다. 도구의 capture·restore가 정확하다.
- 모든 설정이 두 길이 모두 top-1 214/214, free greedy 텍스트 동일, gold 정답이다. 단순 RTN도 마찬가지다.
- 차이는 KL에서만 보인다. KVarN k4v2는 RTN k4v2보다 KL 중앙값이 2–2.6배, 최대가 1.6–4.6배 작다. 같은 비트에서 Hadamard만으로도 대부분의 이득이 난다(hrtn ≈ kvarn). V를 4비트로 올리면 KL이 한 자릿수 더 줄어든다. 256K가 32K보다 나쁘지 않다.
- prefill 중에도 양자화된 앞 블록을 읽으면(chunk 4096) KL 최대가 RTN 0.055, KVarN 0.018로 늘지만 top-1은 여전히 전부 같다. KVarN이 누적에 더 강하다.
- **한계:** 과제(JSON 목록을 그대로 옮겨 쓰기)가 쉬워 정답 여부로는 설정이 갈리지 않았다. 품질 기준(K-D3)을 정하려면 더 어려운 과제(논문의 line retrieval, 256K needle 등)가 필요하다.
- 256K 설정 하나 준비 시간: RTN 15 s, KVarN 43 s(CPU 양자화 40 s).

**속도 상한.** AR bench(측정, pp512·tg64): d0 12.95, d32K 11.45, d64K 10.24 t/s. 256K K0 기준 decode는 214토큰 34.0 s = 6.3 t/s.

|decode 구간|attention 커널(토큰당)|구간 GPU 시간 중 attention|
|---|---|---|
|AR d0|0.4 ms|0.5 %|
|AR d32K|10.3 ms|11.9 %|
|AR d64K|20.3 ms|21.0 %|
|DFlash2 d0|—|0.3 %|
|DFlash2 d32K|—|21.2 %|
|DFlash2 d64K|—|33.5 %|

- AR decode attention은 32K마다 10.2 ms씩 선형으로 늘고, bench의 토큰당 시간 증가(10.1 ms)와 같다. 토큰당 KV 읽기(깊이 × 64 KiB)로 나누면 약 208 GB/s로, 대역폭에 묶여 있다. 그래서 바이트를 줄이면 그대로 시간이 준다.
- 256K AR 추정(외삽, dequant 비용 0 가정): attention 82 → 17 ms, 토큰당 159 → 94 ms, 6.3 → 10.6 t/s(+69 %). 32K는 +10 %, 0에 가까운 문맥은 이득 없음.
- 운영 구성(DFlash2)은 verify 한 번에 KV를 읽어 여러 토큰을 검증하는데도, 이미 64K에서 decode의 1/3이 attention이다. 256K로 외삽하면 2/3 이상이라 이득이 AR보다 클 수 있다(수락률이 깊이·내용에 따라 달라 수치 외삽은 하지 않는다).
- prefill: pp512가 d0 478 → d64K 243 t/s로 깊이에 따라 떨어진다(WMMA prefill attention). 양자화 KV를 읽는 prefill 커널이 이득인지 손해인지는 K1에서 잰다.
- 증거: ai.tools `tmp/test-results/gufo-upstream-port-20261023/kvarn-k0-a01/`, `kvarn-k0-a02/`(jsonl, `K0.json`, bench 로그, rocpd DB).

**판정(추천, 사용자 확인 전).** 정확도는 이 과제에서 모든 설정이 통과했고, 속도 상한은 깊은 문맥 decode에서 크다. 다음은 K1 진입 조건인 fused dequant decode attention 시제품(`tools/bench/`)과, 품질 기준을 가를 어려운 과제 하나다. 형식 후보는 k4v2(최대 압축, 4.88배)와 k4v4(KL 한 자릿수 작음, 3.74배)다.

### K1a·K1b 결과 (2026-10-26, 에이전트 작업·감독 검증, [안내](K1-BRIEF.md))

**K1a**(gufo `6e06a8d1`, `0d59a4e0`): `--kv-cache-type fp16|kvarn-k4v2|kvarn-k4v4`(serve·prompt·chat·bench), 모델별 기본값은 `src/cli/kv_cache_type.cpp` `DefaultKvCacheType` 한 곳(지금 Qwen 27B·Flash-Next fp16, 나머지는 자기 KV). 모델이 실제로 그 형식으로 KV를 저장할 때만 명시 값을 받는다: Qwen 27B GPU는 fp16과 kvarn(아직 `QwenExecutionPolicyUnsupportedReason`이 "not implemented yet"으로 거절), Flash-Next는 fp16만, DeepSeek(FP8)·Gemma 4(이 트리에 실행 코드 없음)·CPU Qwen(FP32)은 명시하면 거절. fp16·fp32 fingerprint는 이전 값 고정. 감독 재실행: CPU `check-pr` 71/71, `qwen27b.kvarn-reference`·`qwen27b.tools`·layout 3/3 PASS. fp16 GPU 실행 동일성은 아직 GPU 단위 테스트로 확인하지 않았다(다음 GPU 창).

**K1b**(gufo `2295ef2f`, `tools/bench/kvarn_decode_bench.hip`, 운영 코드 변경 없음): 그룹 flush 양자화 커널, fused dequant split-K decode(질의 1), DFlash2 verify 변형(질의 8), FP16 기준은 운영 `LaunchCausalDecodeAttention`을 그대로 링크. 창 `kvarn-k1b-a07/k1b-final` COMPLETE·health PASS(a01–a06은 기록 보존: 종료 뒤 GPU busy 4 %가 3 % 한도를 넘은 health 실패, a01은 처음 tile 완전 일치 규칙으로 검사 FAIL). 감독 확인: 커밋에서 다시 빌드한 바이너리 SHA `46d238e9…`가 a07에서 돈 것과 같다(a07 `RESULT.json`의 `source_revision`은 갱신되지 않은 `.gufo-revision` 값 `b3edc894`라 이 SHA 대조로 출처를 확인).

- 저장 배치(제안): 층·256토큰 블록마다 `[그룹 2][KV head 4][record]`, record k4v2 26,880 B·k4v4 35,072 B(FP16 131,072 B). K 코드는 토큰 우선(토큰당 128 B, nibble 교차 배치), V 코드는 채널 우선, 이어서 FP16 scale·zero·열 scale. sink·미flush tail은 시퀀스별 FP16 평면에 이미 회전된 값으로 둔다.
- 정확도(20경우: 그룹 0/1/30/254, tail 44–133, 질의 1·8, split 32·7): 커널 산술 오차(같은 코드의 CPU attention 대비) 2.6e-4–8.3e-4, K0 fake quant 기준 대비 4.5e-4–1.0e-3, 양자화 자체의 효과는 0.11–0.64라 커널 오차는 무시할 수준. FP16 기준 커널은 CPU 대비 7e-7. 양자화 커널은 같은 누적 순서 에뮬레이션과 tile 전부 비트 일치, K0 규칙과는 99.5 % tile 일치(나머지는 scale 최대 2 ulp, 코드 약 100만 개 중 1개): glibc `logf`·`expf`가 정확히 반올림되지 않는 입력(0.14 %·0.06 %) 때문이고, GPU는 double로 계산해 한 번 반올림해 host 정확 반올림과 100만 표본에서 일치.
- 속도(층 하나, µs, 최적 split, DRAM 읽기 상한 238.5 GB/s):

|깊이|질의|FP16|k4v2|k4v4|
|---|---|---|---|---|
|4K|1|83.9|42.5 (1.97×)|50.3 (1.67×)|
|4K|8|238.9|105.6 (2.26×)|100.5 (2.38×)|
|32K|1|591|179 (3.29×)|211 (2.79×)|
|32K|8|1947|457 (4.26×)|447 (4.36×)|
|64K|1|1156|324 (3.56×)|392 (2.95×)|
|64K|8|3934|845 (4.66×)|849 (4.63×)|
|256K|1|4599|1175 (3.91×)|1406 (3.27×)|
|256K|8|19456|3204 (6.07×)|3192 (6.09×)|

- FP16 verify(질의 8)는 약 69 GB/s로 ALU에 묶여 있어 KVarN 이득이 AR보다 크다. AR k4v2는 명령 발행에 묶임(그룹·wave당 2,369 명령 중 약 900이 dequant, WMMA B fragment를 두 half-wave에 중복해야 해서 dequant 반복). 대역폭 하한 대비 32K 66 %, 256K 79 %(k4v4 86 %).
- 양자화: tile 하나 142 µs(지연), 16층 그룹 flush 7.56 ms(128토큰마다 한 번, decode 토큰당 분할 약 59 µs), 64그룹 묶음 tile당 54 µs(prefill 128토큰당 약 7 ms, 256K 약 14 s 추정).
- 증거: ai.tools `tmp/test-results/gufo-upstream-port-20261023/kvarn-k1b-a0{1..7}/`, 계획 `PLAN-K1B.json`.

**판정(감독): K1b gate 통과.** 32K 이상에서 AR 3.3–3.9배, verify 4.3–6.1배로 attention 시간이 줄고, 4K에서도 손해가 아니라 이득(1.7–2.4배)이다. 층 16개로 환산하면(추정) AR 토큰당 attention이 32K 9.5 → 2.9 ms, 256K 73.6 → 18.8 ms(256K AR 약 6.3 → 9.6 t/s). DFlash2 verify 한 번은 64K에서 63 → 13.5 ms. 남은 결정: 운영 split 정책(깊이·점유율에 따라 18–128), verify 폭 1–8 전부, 회전을 KV 쓰기 커널로 옮김(tail이 이미 회전되어 있어야 함), K0 참고 구현의 log·exp를 정확 반올림으로 바꿔 GPU와 비트 일치시키기, int8 WMMA로 dequant 비용 줄이기(후속).

### K1c-1 결과 (2026-10-26, gufo `041be720`…`3436bf39`, 감독 검토)

- 통합: 배치 정의는 `src/models/qwen/hip/kvarn_layout.hpp` 한 곳(record 26,880/35,072 B, 풀 plane `[층][블록][그룹 2][KV 4][record]`, 시퀀스별 FP16 sink 128 + tail ring 256), 쓰기는 기존 norm/RoPE 뒤 `LaunchKvarnWriteRows`(회전, 층마다 launch 1개 추가), flush는 pass 전 `PrepareKvarnPass`(16층 한 launch), decode(그래프 경로 포함)·batched decode·verify 폭 1–8. 사용자 쪽 거절(`QwenExecutionPolicyUnsupportedReason`)은 남겨 두고 테스트용 좁은 경로만 열었다. K0 참고 구현의 log·exp를 정확 반올림으로 바꿔 GPU flush와 128/128 tile 비트 일치.
- fp16 동일성: `kvarn-k1c-a03`(`edadf259`, 운영 코드 마지막 커밋 뒤)에서 `qwen_attention_kv_block_table_ops_test`, `qwen27b_target_test`, `inference_backend_gpu_test` PASS(정확 logit 검사 포함).
- 창: a01 FAILED(테스트 block table 오류), a02 COMPLETE, a03 FAILED(KVarN 테스트 쪽 기준·경우 오류, fp16 테스트는 PASS), a04 FAILED(verify 기준), a05 COMPLETE. 모두 health PASS.
- 통합 경로 커널 시간은 K1b 시제품과 같은 수준(예: 32K 질의 1 k4v2 171.5 µs, 256K 질의 8 k4v2 3,134 µs).
- 모델 teacher forcing 636토큰(FP16 대비): k4v2 top-1 499/508, KL 중앙값 4.7e-5, 최대 0.35; k4v4 506/508, 최대 0.032.
- **감독 판정: 미해결 2건.** (A) 에이전트가 데이터를 본 뒤 KL 기준을 0.1 → 1.0으로 넓혔다. 기준 확대로 인정하지 않고, 같은 prompt·위치에서 fake quant와 비교해 원인을 가린다. (B) verify(8행)와 1행 decode가 비트 단위로 다르다(최대 logit 차 k4v2 0.020, k4v4 0.009, 예측은 같음). FP16은 같고 greedy DFlash2 = AR 텍스트 규칙이 걸려 있다. 가설: verify chunk 안에서 찬 그룹을 verify는 FP16으로, 1행 decode는 양자화된 값으로 읽는다. 둘 다 K1c-2 앞에서 고친다.

### K1b-opt 결과 (2026-10-26, gufo `cf7447b0`, 감독 검증)

- 창 `kvarn-k1bopt-a09/opt` COMPLETE·health PASS(a01은 계획 리비전 불일치로 시작 전 거절, a02–a07은 check FAIL 기록 보존, a08 COMPLETE는 포맷 전 바이너리). 감독 확인: `cf7447b0` 소스 + `6200e0a4` 운영 소스 snapshot으로 다시 빌드한 바이너리 SHA `20ec49f3…`가 a09에서 돈 것과 같다.
- 추천 `opt_latev`: 코드를 FP16 지수와 OR해 16+c·4+c로 바로 만드는 biased 코드(행마다 FP32로 bias 제거), **AR만** K scale을 q에 접기(그룹마다 LDS에서 A = half(q·scale), 행별 보정에 중간 코드 7.5 항 포함), 밑 2 softmax(`v_exp_f32`), AR padding 행 10개 생략, AR은 마지막 블록이 split 합침(reduce launch 없음), verify는 행당 wave 하나 combine 커널, V 코드를 자기 그룹 시작에 읽기(AR k4v4 7 → 8 waves/SIMD).
- 기각: int8 WMMA(A·P를 그룹마다 int8로 다시 양자화하거나 hi/lo로 WMMA 2배, 8비트 P는 1e-3 불가, 줄일 unpack이 이미 쌍당 1.5 명령 이하), dequant LDS 공유(쓰기·읽기·대기가 dequant보다 비쌈), AR `v_dot2`(그룹당 약 3,072 dot + lane 합산 대 wave당 WMMA 32개).
- 그룹 루프 명령 수: AR k4v2 2,248 → 1,151, verify 4,463 → 3,681.
- 정확도(최대, 커널 / K0 기준): k4v2 AR 8.1e-4 / 8.7e-4, k4v2 verify 6.4e-4 / 9.3e-4, k4v4 AR 7.2e-4 / 9.9e-4, k4v4 verify 7.1e-4 / 9.8e-4. 양자화는 그대로.
- 속도(층 하나 µs, 같은 바이너리, 바이트 하한 대비 %): AR k4v2 4K 42.6 → 35.5(51 %), 32K 186.7 → 139.4(85 %), 64K 338 → 262(89 %), 256K 1,244 → 981(94 %); AR k4v4 32K 211 → 180(85 %), 256K 1,404 → 1,290(94 %); verify 32K 457 → 389, 256K 3,238 → 2,911. FP16 32K AR 587, verify 1,933.
- **저장 배치: 바꾸지 않는다.** biased decode가 지금 nibble·crumb 순서와 채널별 FP16 K scale·zero에 기대므로 K1c·K2는 이것을 유지한다.
- 남은 병목: 4K는 블록당 그룹 2개 지연·sink/tail 블록·launch(바이트 하한 51 %), 32K는 마지막 블록 combine 약 8 µs, AR WMMA의 padding 10/16, verify는 ALU/LDS(하한 약 31 %).
- **감독 판정: 속도·정확도 gate 통과. 단 채택 전 결정 1건:** K scale 접기를 AR에만 하면 AR과 verify의 산술이 달라져, K1c에 요구한 "verify 행 = 같은 위치 1행 decode 비트 일치"(greedy DFlash2 = AR)가 구조적으로 성립하지 않는다.

### K0에서 정하지 않는 것

- 커널·서빙·checkpoint 형식(K1·K2). 추가가 필요한 것만 적어 둔다: storage enum 값과 policy fingerprint 비트, K/V component `layout_version` 3(새 row_bytes와 128토큰 scale plane), compact snapshot 검증 완화, 블록 풀의 scale plane(`KvBlockPlane::rows_per_block` 128은 이미 표현 가능하나 arena·adapter가 같은 두 plane을 가정한다).
- 품질 기준(K-D3)은 K0 오차표를 본 뒤 정한다.

## 정할 것

|번호|내용|추천|
|---|---|---|
|K-D1|시작 시점|F와 운영 전환 결정 뒤. K0만 먼저 해서 가능성을 본다|
|K-D2|목표 형식|k4v2_g128(저장소 기본)에서 시작, K0 결과로 k4v4 등과 비교해 정한다|
|K-D3|품질 기준|K0 결과를 보고 사용자와 정한다(top-1 일치율, gold, 256K 문서 정답)|
|K-D4|원본 제안 여부|우리 쪽 검증 뒤 원본 이슈로 의견을 묻는다. 그때까지 우리 이식 브랜치에만 둔다|

조사 출처: [논문 arXiv 2606.03458](https://arxiv.org/abs/2606.03458), [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN), [요약 기사](https://rits.shanghai.nyu.edu/ai/huaweis-kvarn-hits-5x-kv-cache-capacity-at-fp16-accuracy-and-throughput).
