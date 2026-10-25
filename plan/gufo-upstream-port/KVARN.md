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

## 정할 것

|번호|내용|추천|
|---|---|---|
|K-D1|시작 시점|F와 운영 전환 결정 뒤. K0만 먼저 해서 가능성을 본다|
|K-D2|목표 형식|k4v2_g128(저장소 기본)에서 시작, K0 결과로 k4v4 등과 비교해 정한다|
|K-D3|품질 기준|K0 결과를 보고 사용자와 정한다(top-1 일치율, gold, 256K 문서 정답)|
|K-D4|원본 제안 여부|우리 쪽 검증 뒤 원본 이슈로 의견을 묻는다. 그때까지 우리 이식 브랜치에만 둔다|

조사 출처: [논문 arXiv 2606.03458](https://arxiv.org/abs/2606.03458), [huawei-csl/KVarN](https://github.com/huawei-csl/KVarN), [요약 기사](https://rits.shanghai.nyu.edu/ai/huaweis-kvarn-hits-5x-kv-cache-capacity-at-fp16-accuracy-and-throughput).
