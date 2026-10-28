# P: 운영 전환 준비 결과 (2026-10-28)

상위 문서: [브리프](P-BRIEF.md), [이식 계획](PLAN.md), [F16](U4G-TURNKEY.md#전체-점검-f16-final-f16-a01-gufo-d66d5ae7), [KVarN K3](KVARN.md), [S](SKINNY.md), [T](T-RESULTS.md).

## 요약

- 후보 gufo `7c2c5a72`(`ai-tools/port-u4f` HEAD, 시작 시점)는 운영 `210c7633`과 **텍스트가 모든 요청에서 같다**(1차 95요청, 재실행 46요청). gold 전부 정답, T2 21/21, 단위 테스트 16개 PASS, CPU 131/131(10개 모델 없음 skip), functional 4묶음 passed. 서버 로그에 `ERROR`·`logic_error`·`length_error`가 없다.
- 시간은 대부분 같거나 빠르다. 다음 턴·재시작 복원이 2–3배 빠르고, DFlash2 decode가 d0 +68 %, d32K +39 %다.
- 재현된 요청별 퇴보가 두 가지 있다. 둘 다 동시 요청 배치 안의 순서 문제다.
  - 동시에 들어온 짧은 요청 묶음에서 첫 요청의 첫 토큰이 약 1.05 s 늦다(+11–19 %). F(`final-a02`)부터 있었다.
  - 64K×8에서 긴 prefill 도중 도착한 짧은 요청 4개가 0.8–5.1 s 늦다(+2–20 %). 같은 창의 긴 요청은 같거나 빠르다(8K −13–18 %).
- 메모리: 준비 직후 anon이 운영보다 +2.67 GiB(18.30 / 15.63)다. U4d와 같다. 페이지 캐시(file)는 후보가 적다.
- **전환 제안: 조건부로 전환할 수 있다.** 위 두 퇴보를 감수할지는 사용자가 정한다. 실제 전환은 하지 않았다.

## 후보와 빌드

|항목|값|
|---|---|
|후보|gufo `7c2c5a728270043c07f2aa1466151b5244369976`(작업 트리 변경 없음, untracked `.dockerignore`·`.gufo-revision`만)|
|후보 이미지|`ai-tools/gufo-eval:7c2c5a728270`(`sha256:7c440bd2…`), ai.tools `test/experiments/gufo_eval_20261020/Dockerfile`, `cmake --preset release`|
|후보 `gufo`|`27d26ba01fb310e6a63b8135277d186474217d0c0c7bd3aed6690c44471943e4`|
|운영 이미지(F16과 같은 것, 새로 빌드 안 함)|`ai-tools/gufo-eval:210c76337a5a`(`sha256:c722da6f…`), `gufo` `1457b13153dc9d0a3a5d674d3d4567f0020ffddcd00fdb281cc974c7f73ebe28`|
|단위 테스트 트리|`build/gpu-test`, `build/cpu-test`. 후보 이미지 컨테이너에서 HEAD로 빌드(`build/p-build.sh`, 로그 `build/p-logs/`). 바이너리 SHA는 `build/p-logs/unit-sha.txt`|
|계획|ai.tools `test/experiments/gufo_upstream_port_20261023/PLAN-P1.json`, `final.py --set p1`·`--set p1r`|

## 실행한 창

창 실행기는 ai.tools `final.py`(window.py)만, 한 번에 하나, 창마다 끝에 20 s 쉼. 두 run 모두 직전에 `fstrim`(29.2 GiB, 16.2 GiB). 창 사이 빌드 없음. 자동 재시도 없음.

|run|창|결과|
|---|---|---|
|`p1-a01`(07:36–09:24)|units → 시나리오 9개를 운영과 ABBA → T2 → memprobe(운영 `m-prod`와 번갈아) → functional|23창 중 21 COMPLETE. `p1-units`·`p1-bench` FAILED_no_fallback(아래)|
|`p1r-a01`(09:26–10:12)|실행되지 않은 단위 7개(`p1-units-b`) → 깃발이 난 256k4·64k8·regress와 bench를 첫 run과 반대 순서로|9창 중 7 COMPLETE. `fb-bench`·`p1-bench` FAILED_no_fallback(아래)|

- `p1-units`: 계획 실수. 창 시간이 1500 s여서 F16 단위 9개가 모두 PASS한 뒤, driver가 남은 시간(825 s)이 다음 테스트 timeout(900 s)보다 작아 멈췄다. 7개는 시작도 하지 않았다. 재시도가 아니라 실행하지 않은 테스트로 보고 `p1-units-b`(8400 s)로 한 번 돌렸다.
- bench 창 3개(`p1-a01/p1-bench`, `p1r-a01/fb-bench`, `p1r-a01/p1-bench`): 측정은 COMPLETE다. 종료 뒤 GPU busy 4 %가 health 한도 3 %를 넘어 FAILED다(새 GPU 커널 오류 없음, KFD context 없음). 운영 bench에서도 났다. KVarN K1b에서 본 것과 같은 health 실패다. 한도는 바꾸지 않았다.
- 증거: ai.tools `tmp/test-results/gufo-upstream-port-20261023/p1-a01/`, `p1r-a01/`(창별 `WINDOW.json`, `receipt/run/`), `p1-a01.log`, `p1r-a01.log`, `*-fstrim.log`. cache-disk는 지우지 않았다(남은 여유 112 GiB).

## 단위 테스트

|테스트|결과|창|
|---|---|---|
|`qwen_attention_wmma_prefill_ops_test`, `qwen_attention_kv_block_table_ops_test`, `qwen_prompt_hidden_buffer_test`|PASS|p1-units|
|`qwen27b_continuation_adapter_test`(+ draft)|PASS|p1-units|
|`qwen27b_target_test`, `qwen_dflash_gpu_test`|PASS|p1-units|
|`inference_backend_gpu_test`(동시 fork 포함)|PASS|p1-units|
|`qwen_vision_serving_test`|PASS|p1-units|
|`qwen_attention_kvarn_ops_test`, `qwen_kvarn_layout_test`|PASS|p1-units-b|
|`qwen27b_kvarn_decode_test`(전체, `--continuation-only`)|PASS (723 s, 168 s)|p1-units-b|
|`qwen_q4kxl_quant_ops_test`(skinny)|PASS|p1-units-b|
|`qwen_ssm_ops_test`, `qwen_attention_projection_ops_test`|PASS|p1-units-b|
|CPU `ctest`(`build/cpu-test`)|141개 중 131 통과, 실패 0, 10 skip(ASR·TTS·MiniMax·MTP·Gemma2 모델 없음)|호스트 빌드 컨테이너|

## 정확도

- 운영 설정: 256K×4, pool 40 GiB, disk 64 GiB, DFlash2, think off, mmproj, max_tokens 32768, reasoning 16384. 64K×8은 pool 24 GiB다.
  - 후보 서버 인수는 운영 인수에서 두 가지만 다르다. 후보에 없는 `--cache-disk-block-bytes`·`--cache-idle-write-ms`를 뺐고, RAM 예산은 U4d 결정대로 2 GiB다(운영 8 GiB).
  - `--kv-cache-type`은 주지 않았다. 후보 로그 `event=kv_pool block_bytes=3440640`이 k4v2 형식이다(운영 FP16은 4194304). bench 출력도 `KV cache: kvarn-k4v2`다.
- 텍스트: 모든 시나리오 창에서 운영과 같다. 운영은 FP16이고 후보는 k4v2라 달라질 수 있었지만 다른 요청이 없다. K3에서 k4v2가 FP16과 95/95 같았던 결과와 맞는다.

|시나리오|요청|텍스트 같음|gold|
|---|---|---|---|
|long32k|8|8|8/8|
|long256k|3|3|3/3|
|256K×4 동시(2회)|9+9|18|전부|
|64K×8 동시(2회)|17+17|34|전부|
|two8k|8|8|8/8|
|regress(2회)|20+20|40|전부|
|boundary|11|11|11/11|
|shared|19|19|19/19|

- T2 21요청: 21/21 정답(gold 19 + 이미지 2), 텍스트가 F16 `f16-t2`와 21/21 같다(T2는 F16처럼 운영 짝이 없다).
- functional(후보 한 번, `--record-baseline`): `standard`(32768×2) `batch`·`state-edges`·`cache`(취소·disk-spacing 하위 묶음 포함) passed, `single-session`(32768×1) `long-context` passed. 로그의 400 두 건은 `state-edges`의 의도된 잘못된 요청(`invalid_top_p`, `invalid_tools`)이다. `disconnected` 32건은 취소 suite와 T2 admission 검사의 의도된 취소다.
- 서버 로그: 모든 창에서 `ERROR`·`logic_error`·`length_error` 0건.

## 시간

### TTFT 합(초, 참고용. 판정은 요청별)

|시나리오|운영 P|후보 P|F16(`d66d5ae7`)|
|---|---|---|---|
|long32k(8)|250.6|243.0|244.4|
|long256k(3)|893.3|875.8|875.5|
|256K×4(9) 1차 / 2차|371.9 / 376.5|372.8 / 372.9|—|
|64K×8(17) 1차 / 2차|829.5 / 820.9|815.2 / 801.9|—|
|two8k(8)|59.6|58.0|57.2|
|regress(20) 1차 / 2차|26.5 / 26.6|21.4 / 21.2|21.5|
|boundary(11)|24.3|22.5|22.8|
|shared(19)|50.4|44.2|46.6|
|T2(21)|—|65.6|65.7|

### 요청별 판정(5 %·3 ms)

- 깃발 없는 창: long32k, long256k, two8k, boundary, shared. 다음 턴·재시작 복원은 크게 빠르다.
  - long256k 다음 턴 0.42 / 1.29 s, 재시작 복원 1.78 / 4.52 s.
  - 32K 다음 턴 0.23 / 0.47–0.88 s.
  - regress·shared 둘째 턴 0.26–0.35 / 0.80–0.87 s.
- 1차 깃발 7건: 256K×4 2건, 64K×8 4건, regress 1건. 영향받는 history(256k4·64k8·regress)만 첫 run과 반대 순서로 다시 돌렸다.
  - 동시 창에서는 같은 시각에 들어온 같은 길이 요청 중 누가 먼저 입장하느냐가 run마다 바뀐다. 그래서 같은 출발 시각 묶음 안에서는 TTFT를 정렬해 비교했다.

|깃발|1차(후보 / 운영)|2차, 순서 반대|이력|판정|
|---|---|---|---|---|
|256K×4 C4 짧은 4개 중 첫째|6.56 / 5.49 s|6.51 / 5.45 s|F `final-a02` 6.52 / 5.45, K3 6.49(FP16)·6.56(k4v2)|**재현**. 나머지 3개는 같거나 빠르다(9.8–10.4 / 9.7–10.9). 묶음 끝 시각은 같다(19.3–19.6 / 19.3–19.9 s)|
|64K×8 C8 짧은 8개 중 첫째|10.86 / 9.78 s|10.84 / 9.80 s|F 10.9 / 9.8, K3 FP16 10.9·k4v2 9.8|**재현**(+1.05 s). 나머지 7개는 같다|
|256K×4 긴·짧은 섞음 16K|89.8 / 80.4 s|89.8 / 94.7 s|운영 62.3–94.7, 이식 93.6–95.8|운영 쪽 편차가 커서 **inconclusive**|
|64K×8 t=1 s에 도착한 짧은 4개(정렬)|30.2, 43.6, 43.9, 44.2 / 25.1, 38.5, 40.1, 43.1|24.6, 38.0, 38.3, 38.5 / 21.4, 37.2, 37.5, 37.7|F에서는 이식이 빨랐다(21.5–35.5 / 34.0–45.7). K3 k4v2 21.5–35.5|이번 후보에서 두 순서 모두 **재현**(+0.8–5.1 s). 같은 창 긴 요청은 같거나 빠르다(8K 73–76 / 88–90, 16K 132–148 / 143–147, 32K 180–182 / 185–186 s)|
|regress T1 첫 요청(서버 시작 직후)|2.250 / 2.134 s|2.099 / 2.138 s|이식 첫 요청은 2.08–2.11 또는 2.25–2.39로 갈린다(f16·f17·k3f), 운영 2.13–2.16|**재현 안 됨**. 시작 직후 첫 prefill의 잡음, inconclusive|

- 동시 묶음 첫 요청(+1.05 s)의 원인(로그 확인):
  - 첫 요청은 두 빌드 모두 바로 입장한다(`queue_ms` 0.5). 자기 prefill도 같다(약 2.1 s, `prefill_tps` 490).
  - 첫 토큰은 뒤 요청들의 prefill이 끝날 때까지 나가지 않는다. 운영은 그 대기가 약 3.3 s이고 후보는 약 4.4 s다(뒤 요청 prefill 하나 분량 더).
  - KVarN·K5·S·T 이전의 F부터 같은 크기로 있었다. 이번 후보에서 생긴 변화가 아니라 이식본의 입장·prefill 끼워 넣기 순서다.
- 64K×8 t=1 s 짧은 요청: 후보는 먼저 들어온 긴 prefill을 일찍 끝내고, 그동안 도착한 짧은 요청을 조금 늦게 처리한다. K3(`be582a9f`) k4v2에서는 짧은 요청이 21.5–35.5 s로 이번(24.6–44.2)보다 빨랐다. 그 사이 바뀐 S·T 스케줄링(`f53aba73` capture resume, `28e650b4` verify 묶음)이 원인 후보지만 확정하지 않았다(이 단계는 코드 커밋 금지).
- 판정: 위 두 퇴보는 재현된 요청별 퇴보다. margin은 넓히지 않았다. 나머지 깃발은 inconclusive로 남긴다.

### bench(PLAN-R4 측정점, t/s, 2회씩)

|항목|운영|후보|차이|
|---|---|---|---|
|AR pp1024 / pp2048|655.5·653.9 / 649.9·652.1|656.6·656.9 / 652.1·653.3|+0.2–0.5 %|
|AR tg128|12.89·12.92|12.95·12.97|+0.4 %|
|AR pp1024 / pp2048 @ d32768|476.8·478.5 / 484.8·485.9|482.3·480.9 / 491.5·487.8|+0.4–1.4 %|
|AR tg128 @ d32768|11.42·11.43|12.57·12.58|+10 %|
|DFlash2 pp1024 / pp2048|597.3·594.7 / 594.6·593.6|601.0·599.1 / 597.2·593.6|0–0.7 %|
|DFlash2 tg128|26.88·26.83|45.05·45.01|+68 %|
|DFlash2 pp1024 / pp2048 @ d32768|454.8·452.9 / 460.2·459.9|459.6·459.7 / 463.8·463.9|+0.8–1.5 %|
|DFlash2 tg128 @ d32768|51.36·51.24|71.45·71.51|+39 %|

- 모든 칸에서 후보가 같거나 빠르다. 같은 빌드 두 회차 차이는 1 % 안이다.
- 시나리오 decode도 같은 방향이다. 예: long256k 첫 턴 decode 44.2 / 19.1 t/s.

## 메모리(GiB)

|창|후보 anon|후보 file|후보 최대(current)|운영 anon|운영 file|운영 최대|
|---|---|---|---|---|---|---|
|memprobe 준비 직후|18.30|0.00|—|15.63|0.00|—|
|memprobe 3턴 뒤|18.38|0.00|—|16.44|0.00|—|
|long32k|18.40|0.87|19.20|17.11|2.79|19.51|
|long256k|18.44|2.45|20.84|16.24|5.83|22.27|
|256K×4|18.38|0.95|19.34|17.16|2.23|19.38|
|64K×8|18.46|0.06|18.57|17.01|2.45|19.52|
|two8k|18.39|0.00|18.44|16.63|2.84|19.29|
|regress|18.39|0.06|18.50|16.29|2.42|18.68|
|boundary|18.39|0.00|18.44|16.42|0.23|16.46|
|shared|18.40|0.07|18.51|16.43|3.36|19.60|
|bench|18.94|0.00|18.99|18.74|0.00|18.78|

- 창 최대는 2 s 간격 cgroup 표본(final.py)이고, memprobe는 프로세스 cgroup이다.
- 후보 anon은 18.3–18.5로 일정하다. RAM 예산 2 GiB와 pinned 영역을 준비할 때 잡기 때문이다. 운영은 16.2–17.2에서 요청에 따라 늘어난다.
- anon 차이 +2.67 GiB(준비 직후)는 U4d와 같다. 내역은 [anon 바닥](ANON-FLOOR-RESULTS.md)(DFlash2 pinned, backing 미리 커밋 등)이다.
- 페이지 캐시와 디스크 cache 크기:
  - 후보는 file이 적어 컨테이너 최대가 운영과 같거나 작다(long256k 20.8 / 22.3).
  - 대신 디스크 cache는 대화 창에서 더 크다(regress 6.6 / 2.1, shared 6.7 / 2.9 GiB). 256K 창은 반대다(3.9 / 15.8).

## 남은 차이와 위험

1. 동시 짧은 요청 묶음의 첫 요청 TTFT +1.05 s(+11–19 %). F부터 있었다. 묶음 전체 끝 시각과 다른 요청은 같거나 빠르다.
2. 64K×8에서 긴 prefill 도중 도착한 짧은 요청 +0.8–5.1 s. 이번 후보의 두 순서에서 재현됐고, K3보다도 느리다. 원인은 확정하지 않았다. 다음 턴·재시작 복원과 decode가 빨라진 것과는 별개로, 혼잡할 때 짧은 대화형 요청의 응답 시작이 늦어질 수 있다.
3. anon +2.67 GiB(예산 2 GiB 기준). 같은 100 GiB 컨테이너 한도 안이다.
4. 디스크 cache 형식이 운영과 다르다(KVarN 행·블록 파일). 운영 cache 디렉터리를 그대로 쓰면 안 된다. 새 디렉터리로 시작하므로 전환 직후 첫 재시작 복원 이점은 없다.
5. bench 창 health 실패(종료 뒤 GPU busy 4 %)는 운영 빌드에서도 났다. 서빙 위험은 아니지만 측정 한도와 맞지 않는다.
6. 후보 커밋 `7c2c5a72`은 아직 원격(`fork/ai-tools/port-u4f`는 `0341dd88`)에 없다. 운영 이미지를 재현 가능하게 만들려면 push나 patch 고정이 필요하다(이번에는 push 금지).
7. 이번에 돌지 않은 것: 원본 functional 전체 sweep(브리프 범위 밖), `pi_agent.py` 재현(보고된 coding-agent loop가 없음), 8 GiB RAM 예산 후보(U4d 결정으로 2 GiB만).

## 전환 제안

**권장: 조건부 전환.**

- 근거:
  - 정확도는 운영과 같다(텍스트 141/141 + T2 21/21, gold 전부, 단위·functional 통과).
  - 복원·decode 속도는 크게 낫다. 다음 턴 2–3배, DFlash2 decode +39–68 %, 256K 재시작 복원 4.5 → 1.8 s다.
- 조건: 위험 1·2(동시 혼잡 때 짧은 요청의 첫 토큰 지연)를 사용자가 받아들일 때 전환한다.
  - 받아들이지 않으면 위험 2의 원인(K3 → 후보 사이 S·T 스케줄링 변경)을 먼저 A/B로 가른다.
  - 그때는 64k8·256k4만 K3 이미지 `be582a9f9275`와 후보로 번갈아 돌린다.

### 전환 절차 초안(실행하지 않음)

1. **소스 고정**: `7c2c5a72`을 `fork/ai-tools/port-u4f`에 push하거나, 지금처럼 원본 기준 patch + tree hash로 고정한다(`infra/services/llm/Dockerfile.gufo` 방식).
2. **이미지 빌드**: `Dockerfile.gufo`의 `GUFO_BASE_REVISION`·`GUFO_TREE`·patch를 새 값으로 바꾼다. 이번 검증 이미지와 같은 base `local/ai-tools-sglang-prep:rocm724-gfx1151-20261008`과 `cmake --preset release`를 쓴다. 새 태그(예: `local/ai-tools-gufo:7c2c5a72`)로 빌드한다. 빌드한 `gufo` SHA가 `27d26ba0…`와 같은지 확인한다(같은 base·같은 경로면 같아야 한다).
3. **compose 플래그 차이**(`infra/services/llm/compose.qwen.yaml`):

|플래그|운영 `210c7633`|후보|
|---|---|---|
|`image`|`local/ai-tools-gufo:210c7633`|새 태그|
|`--cache-disk-block-bytes 68719476736`|있음|**삭제**(후보에 없는 옵션, `Unknown option`으로 시작 실패, exit 2 확인)|
|`--cache-idle-write-ms 2000`|있음|**삭제**(같음)|
|`--cache-ram-bytes`|8589934592|2147483648(U4d 결정, 이번에 검증한 값)|
|`--kv-pool-bytes`|42949672960|그대로(후보 기본값 0은 세션마다 전체 문맥)|
|`--kv-cache-type`|없음(FP16만)|기본 `kvarn-k4v2`. 이번 검증은 플래그 없이 했다. 기본값이 바뀌어도 형식이 그대로 남게 `--kv-cache-type kvarn-k4v2`를 명시할지는 결정 사항이다|
|`--cache-tail-checkpoints`|없음|기본 3, 플래그 불필요|
|disk cache 마운트|`tmp/cache/llm/qwen-gufo-disk`|**새 빈 디렉터리**(예: `qwen-gufo-disk-7c2c5a72`). 옛 디렉터리는 되돌리기용으로 지우지 않는다|
|그 밖(모델·draft·mmproj·256K×4·max_tokens·reasoning·served-model-name·healthcheck)|—|그대로|

4. **전환**: 운영은 지금 정지 상태다. 새 compose로 시작한 뒤 확인한다.
   - `/health`가 통과하는지.
   - 로그에 `event=kv_pool block_bytes=3440640 sized=explicit`가 찍히는지.
   - 짧은 대화 한 번과 256K 문서 한 번이 정상인지.
5. **되돌리기**: compose를 운영 값으로 되돌리고 다시 시작한다.
   - 이미지 태그, 두 플래그, RAM 8 GiB, 옛 disk 디렉터리를 운영 값으로 돌린다.
   - `local/ai-tools-gufo:210c7633` 이미지와 옛 cache 디렉터리를 지우지 않고 둔다(prune 금지).
   - 후보가 쓴 새 cache 디렉터리는 운영 빌드가 읽지 않는다.

### 원본(`82c62b40`)에 없는 플래그

- 후보에만 있다:
  - `--kv-pool-bytes`(paged KV 공유 풀, 운영에도 있음)
  - `--cache-tail-checkpoints`(U4b)
  - `--kv-cache-type`(KVarN, K2)
- 운영 `210c7633`에만 있고 후보에는 없다:
  - `--cache-disk-block-bytes`, `--cache-idle-write-ms`(compose에서 씀)
  - `--cache-disk-read-ahead`, `--cache-grid-tokens`, `--draft-calibration`, `--image-tokens`, `--no-cache-resident-blocks`, `--no-shared-pass-scratch`, `--shared-prefix-min-tokens`(compose에서 안 씀)
