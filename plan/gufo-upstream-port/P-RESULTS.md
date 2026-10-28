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

## P2: 64K×8 짧은 요청 지연 원인 가르기 (2026-10-28)

브리프 [P2](P-BRIEF.md#p2-64k8-짧은-요청-지연-원인-가르기-사용자-결정-2026-10-28-b). 코드 커밋 없음. 창 계획은 ai.tools `PLAN-P2.json`(후보 `p2c-64k8`)·`PLAN-P2-K3.json`(K3 `p2k-64k8`), `final.py --set p2`(ai.tools `f6df5222`)다.

### 1. 로그 분석과 가설

- 짧은 요청 q0–q3의 지연은 prefill·decode 속도가 아니라 **입장 대기(`queue_ms`)** 다.
  - `long-and-short`에서 처음 들어온 긴 요청 하나만 바로 입장한다(`queue_ms` 0.5–1). 나머지 7개는 그 요청의 첫 prefill step이 끝난 뒤 함께 입장한다.
  - 대기: 후보 p1·p1r은 짧은 요청 5.63·5.65 s, 긴 요청 6.63·6.64 s다. K3 `k3k-64k8`은 2.45 / 3.44 s, `k3f-64k8`은 2.41 / 3.41 s다.
  - 차이 +3.2 s가 짧은 요청 TTFT 차이(정렬 기준 +3.1 s, 예: 24.6 / 21.5 s)와 같다. 짧은 요청 `prefill_tps`(475 / 470, 412 / 411)와 decode는 같다.
- 첫 step의 길이가 다른 원인은 코드가 아니라 **RAM 예산**으로 본다. K3 창은 RAM 8 GiB, 후보 p1은 U4d 결정대로 2 GiB였다. p1 결과의 "K3보다 느리다"는 이 설정 차이가 섞인 비교였다.
  - `cache::GridLimit(ram, private) = min(4, ram / 12 / private)`(`src/cache/retention.cpp`)가 cold prompt의 grid capture 지점 수를 정한다. `TextRunnerPool::Request::Prefill`은 다음 checkpoint 지점에서 step을 끊는다.
  - Qwen3.8-27B의 private state는 capture당 약 264 MB다(`private_state` 528482304 / 2 captures). 그래서 8 GiB에서는 limit 2(첫 지점 2048), 2 GiB에서는 limit 0(grid 지점 없음)이다.
  - 혼자 도는 첫 prefill step은 budget 제한이 없다(`bounded_prefill`은 decoder나 대기 peer가 있을 때만). 그래서 2 GiB에서는 첫 step이 pass 한도(4096행, 약 6.6 s)까지 가고, 8 GiB에서는 2048(약 3.4 s)에서 끝난다. 그 사이에는 `Admit`이 돌지 않는다.
  - K3 → 후보 사이 코드 변경(K5 `4a7046b2`, S `26e0a956`·`28e650b4`, T `f53aba73`)은 커널과 capture 재개 한 줄이다. 입장·chunk 경계를 바꾸지 않는다.
- 운영 편차의 원인(로그):
  - 운영은 긴 요청 셋을 바로 입장시킨다(`queue_ms` 1–3). 짧은 요청은 chunk 경계마다 하나 또는 둘씩 들어온다.
  - 1차: 3.5 / 10.2 / 12.8 / 13.9 s, 2차: 2.4 / 3.5 / 3.5 / 10.2 s.
  - 첫 경계(2.4 s 또는 3.5 s)에 들어온 짧은 요청만 21–25 s에 첫 토큰을 받는다. 나머지는 37–43 s다. 같은 q1이 21 s나 40 s가 되는 것은 도착 순서와 chunk 경계 시각 때문이다.

### 2. 창 `p2`(64k8만, K3 `be582a9f` / 후보 `7c2c5a72` ABBA)

- 두 이미지 모두 같은 서버 인수를 쓴다. 후보 p1-64k8 인수(RAM 2 GiB)에 `--kv-cache-type kvarn-k4v2`를 명시했다. 다른 것은 코드뿐이다.
- 실행 직전 `fstrim`(2.6 GiB, `p2-a01-fstrim.log`). 창마다 끝에 20 s 쉬었다. 창 중 빌드 없음, 재시도 없음.

|창|이미지|시각|결과|
|---|---|---|---|
|`p2k-64k8`|K3 `be582a9f9275`|10:24–10:28|COMPLETE, health PASS|
|`p2c-64k8`|후보 `7c2c5a728270`|10:28–10:32|COMPLETE, health PASS|
|`p2c-64k8-r2`|후보|10:33–10:37|COMPLETE, health PASS|
|`p2k-64k8-r2`|K3|10:37–10:41|COMPLETE, health PASS|

- 정확도: 4창 모두 17/17 gold다. 요청별 텍스트가 4창 모두 같다. 서버 로그에 `ERROR`·`logic_error`·`length_error`가 0건이다. 로그 `block_bytes=3440640`(k4v2), RAM `capacity_bytes=2147483648`.
- 입장 대기: K3 짧은 요청 5.56·5.66 s, 긴 요청 6.56·6.66 s. 후보 5.64·5.66 s, 6.64·6.66 s. 같다.
- 메모리(창 최대): anon 18.46 GiB, current 18.52–18.55 GiB로 4창이 같다.

TTFT(초). 같은 출발 시각 묶음 안에서 정렬한 순위별 비교다.

|단계|요청(순위)|K3 1·2|후보 1·2|판정|
|---|---|---|---|---|
|warm|q0|2.10, 2.12|2.11, 2.11|같음|
|C8-short|#0|10.64, 10.87|10.91, 10.81|같음|
|C8-short|#1–#7|19.59–21.57, 19.94–22.00|19.97–22.01, 19.88–21.93|같음(범위 겹침)|
|long-and-short|32K|179.74, 180.36|179.24, 180.20|같음|
|long-and-short|16K|131.33, 132.14|131.28, 132.05|같음|
|long-and-short|8K(t=0)|75.84, 76.31|76.05, 76.45|같음|
|long-and-short|짧은 #0|29.81, 30.25|30.16, 30.26|같음|
|long-and-short|짧은 #1|43.18, 43.73|43.57, 43.73|같음|
|long-and-short|짧은 #2|43.43, 43.98|43.81, 43.97|같음|
|long-and-short|짧은 #3|43.72, 44.27|44.10, 44.25|같음|
|long-and-short|8K(t=2 s)|106.60, 107.30|106.62, 107.24|같음|

- 판정: 5 %·3 ms를 넘는 요청이 없다. 순위마다 후보 두 값이 K3 두 값의 범위 안이나 0.4 s 안에 있다. **같은 설정에서 후보는 K3보다 느리지 않다.**
- 같은 설정의 K3도 짧은 요청이 29.8–44.3 s다. p1에서 본 후보 값(24.6–44.2 s)과 같은 범위이고, 8 GiB K3(21.5–35.5 s)보다 늦다. P1 위험 2의 지연은 K5·S·T 코드가 아니라 RAM 예산 2 GiB에서 나온다.

### 3. 이분 탐색과 수정

- 이분 탐색: 하지 않았다. 브리프 조건(후보가 K3보다 재현되게 느림)이 성립하지 않는다. 새 이미지도 빌드하지 않았다.
- 수정: 하지 않았다. 원인은 정해졌지만 고치는 방법이 trade-off라 사용자 결정이 필요하다.
  1. RAM 예산을 8 GiB로 되돌린다. 코드 변경은 없다. 대신 U4d에서 줄인 anon·RAM 사용이 다시 늘어난다(K3 64k8 ledger 8.16 GB / 2.36 GB).
  2. 입장 경계를 capture grid와 분리하는 scheduler 수정. 예: 큐에 요청이 있으면 혼자 도는 prefill도 2048 같은 bounded step으로 끊는다. 2 GiB를 유지할 수 있다. 다만 공유 scheduler 동작이 바뀌므로 64k8·256k4 ABBA와 `batch`·`cache`·`long-context` suite가 필요하고, 혼자 도는 긴 prefill 처리량에 미치는 영향도 재야 한다.
  3. 감수한다. 지연은 혼잡할 때 긴 prefill이 먼저 들어온 경우의 짧은 요청에서만 생긴다(+약 3.2 s, 첫 step 하나 분량). 운영도 도착 순서에 따라 21–43 s로 갈린다.
- 256K×4 C4 첫 짧은 요청 +1.05 s: 브리프대로 로그 분석만 했고, 위 [시간](#시간) 절의 분석이 그대로다. F(RAM 8 GiB)부터 있었으므로 이 RAM 예산 효과와는 다른 문제다.

### 증거

- ai.tools `tmp/test-results/gufo-upstream-port-20261023/p2-a01/`(`FINAL.json`, 창별 `WINDOW.json`·`receipt/run/{RESULT.json,server-1.log}`), `p2-a01.log`, `p2-a01-fstrim.log`. cache-disk는 지우지 않았다(남은 여유 87 GiB).
- 비교에 쓴 기존 창: `p1-a01/{p1,fb}-64k8`, `p1r-a01/{p1,fb}-64k8`, `k3-a01/{k3k,k3f}-64k8`.

## P3: 입장 경계를 capture grid와 분리 (2026-10-28)

브리프 [P3](P-BRIEF.md#p3-입장-경계를-capture-grid와-분리-사용자-결정-2026-10-28-오케이-그렇게-가자-일단-수정만-해). 수정과 CPU 검증만 했다. GPU 창·이미지 빌드는 하지 않았다.

### 설계

- 원인(코드): `Impl::Run`의 마지막 분기(decoder 없음)는 `StepPrefill(request, false, peer_waiting)`을 부른다. 짧은 peer나 capture가 없으면 budget이 `prompt_tokens()`라 step이 model pass 한도(Qwen `kMaxPromptBatch` 4096)나 다음 checkpoint까지 간다. `Admit`은 step 사이에만 돈다. checkpoint는 `PlannedCaptures` → `cache::PlanCaptures(grid_limit = GridLimit(ram, private))`가 정하므로 혼자 도는 prefill의 입장 경계가 RAM 예산(2 GiB → grid 0개, 8 GiB → 2048·4096)에 따라 달라졌다.
- 수정(`src/cli/serve/text_generation_scheduler.{hpp,cpp}`):
  - 상수 `kAdmissionPrefillTokens = 2048`(cache grid 간격과 같음)을 추가했다. 옵션·스위치는 없다.
  - 마지막 분기에서 세션이 비어 있으면(`prefilling + capturing + 1 < capacity`) 혼자 도는 prefill의 budget을 `2048 - prefill_position % 2048`로 한다. 즉 step이 절대 위치 2048의 배수에서 끝난다. 8 GiB grid 지점(2048 배수)과 같은 경계이고, cache hit로 중간 위치에서 시작해도 grid checkpoint와 어긋난 짧은 조각이 생기지 않는다.
  - 큐가 비어 있어도 적용한다. P2의 64k8처럼 요청은 step 도중에 도착하므로 "큐가 있을 때만"은 이 경우를 놓친다.
  - 세션이 꽉 찼거나 capacity 1이면 입장할 수 없으므로 기존 동작(model pass 전체)을 유지한다. decoder가 있거나 짧은 peer·capture가 기다릴 때의 `decode_active_tokens`(512) 분기는 그대로다.
- 바뀌지 않는 것:
  - capture 지점: `PlanCaptures`는 adapter의 `PlanPrefill`(4096 pass 계획)만 읽고 scheduler budget을 읽지 않는다. `Request::Prefill`은 budget과 무관하게 checkpoint·shared boundary·snapshot 위치에서 멈춘다. 지점 목록과 개수는 같다.
  - checkpoint 내용: step 크기만 달라진다(이미 decoder가 있을 때 512, 8 GiB grid에서 2048로 끊는 경로와 같은 종류). Qwen 27B prefill은 chunk 크기와 무관하게 같은 값을 내도록 만들어져 있다(`docs/models/qwen3.8-27b/EXPERIMENTS.md`: "Fixed per-row FP32 order retained for chunk/cache/continued-image equivalence", "Q8 full logits/features match at 8K across scheduling budgets"). flash-next도 2048/4096/8192에서 byte-identical이다(`47b63915`). GPU에서 출력 hash·텍스트 동일을 확인해야 한다(아래 계획).
  - `publication_prefill.Observe`는 여전히 혼자 도는 step만 관측한다(token당 속도라 step 크기와 무관).
- 원본 의도와 다른 점: 원본 `47b63915`(qwen-flash 4096 chunk)는 "Long prefills alone retain the model's efficient chunk size"로 빈 세션이 있어도 혼자 도는 prefill을 끊지 않았다. 이번 수정은 빈 세션이 있을 때 그 효율 일부를 입장 지연과 바꾼다. 아래 비용이 그 대가다.

### 처리량 비용 추정(측정 아님)

- 경계 하나당 비용 = pass 하나 추가분(64층 launch chain, weight 재스트리밍 중 compute에 가려지지 않는 부분, host 준비·`Admit` 한 번). attention 작업량은 행 수로 정해지므로 늘지 않는다.
- Qwen 27B Q4: pp2048 약 695 t/s(EXPERIMENTS.md)라 2048행 step이 약 2.9 s다. 경계당 30–80 ms로 보면:
  - 8K 혼자 prefill: step 2개 → 4개, +2경계, 약 0.06–0.16 s(0.5–1.4 %).
  - 256K 혼자 prefill(k3k-long256k 864 s, 294.9 t/s): 63 → 125 step, +62경계, 약 2–5 s(0.2–0.6 %).
- 로그 근거(통제되지 않음): P2에서 같은 첫 step이 8 GiB(2048행)에서 입장 대기 2.41–2.45 s, 2 GiB(4096행)에서 5.56–5.66 s였다. 2048행 두 번(약 4.9 s)이 4096행 한 번보다 느리다는 신호는 없다.
- 위험 상한: flash-next는 2048 → 4096 chunk로 pp4096 +5.3 %, 24K–102K 프롬프트 +6–11 %였다(`47b63915`, chunk마다 n-gram gather 겹침 포함). 이 scheduler 변경은 모든 모델에 적용되므로 flash-next 다중 세션 서버의 혼자 도는 prefill은 그 이득 일부를 잃을 수 있다. 운영 모델(27B)은 n-gram gather가 없어 이보다 작을 것으로 보지만 측정 전까지는 추정이다.
- 기대 이득: 2 GiB 64k8 `long-and-short`의 짧은 요청 입장 대기 5.6 s → 약 2.4 s(8 GiB K3 수준, P2 기준 −3.2 s). 긴 prefill 중 도착한 요청의 최악 대기가 RAM 예산과 무관하게 2048행 step 하나(깊은 위치에서는 attention 때문에 더 김)로 묶인다.

### CPU 검증

- 단위 테스트(`tests/cli/text_generation_scheduler_test.cpp`):
  - 추가 `TestLonePrefillStopsOnAdmissionInterval`: 5000행 혼자 prefill이 capacity 2에서 `{2048, 2048, 904}`로 끊기고, capacity 1에서는 한 step(5000)이다. model pass가 1000행이면 `{1000, 1000, 48, 1000, 1000, 48, 404}`로 2048 배수에서 끝난다.
  - 추가 `TestArrivalDuringLonePrefillWaitsOneInterval`: 큐가 빈 채 시작한 8192행 prefill 도중 도착한 짧은 요청이 첫 2048행 step 뒤 입장해 긴 요청의 다음 step 전에 첫 토큰을 낸다. 두 요청 출력 유지.
  - `TestMultiResidentPrefillUsesBoundedWorkUnits`는 기대값이 그대로이고 메시지만 고쳤다(경계 안에서는 model geometry).
- 빌드·실행: gpu-eval 컨테이너가 없을 때 후보 이미지 `ai-tools/gufo-eval:7c2c5a728270` 안에서 `build/cpu-test`의 `text_generation_scheduler_test` 타깃만 `-j 2`로 빌드했다(TU 2개). `All text generation scheduler tests passed`, ctest PASS. 처음 실행은 fake 기본 context 128 때문에 새 테스트가 `length_error`로 멈춰 테스트에 `max_context`를 넣고 다시 돌렸다. cpu-test 안에서 이 scheduler를 링크하는 다른 단위 테스트는 없다(남은 것은 `gufo` 바이너리뿐, 빌드하지 않음).
- `tools/ci/check-format.py`(nixos/nix 컨테이너 clang-format 21.1.8): PASS, 693 files.

### GPU 검증 계획(실행하지 않음)

규칙은 P·P2와 같다(창 실행기, 한 번에 하나, 창 전 `.gufo-revision`=HEAD, 끝에 `sleep 20`, 창 중 빌드 금지, 재시도 없음, 실행 직전 fstrim, 증거 보존).

1. 이미지: 수정 커밋으로 후보 이미지를 창 **전에** 빌드한다. 대조군은 후보 `7c2c5a728270`(같은 RAM 2 GiB 인수).
2. `p3-64k8` ABBA(4창, 약 16분): 수정본 / `7c2c5a72`, p2와 같은 인수. 판정: `long-and-short` 짧은 요청 `queue_ms` 약 2.4 s, TTFT 순위별 개선, 나머지 요청·phase는 5 %·3 ms 안. 텍스트 4창 동일, gold 17/17, ERROR·logic_error·length_error 0.
3. `p3-256k4` ABBA(4창): 256K×4 운영 인수. C4·T2 순위별 5 %·3 ms, 텍스트 동일.
4. 혼자 도는 prefill 처리량: 256K 단일 요청(`long256k`)과 8K·32K 단일 요청을 세션 4개 서버에서 ABBA. 요청별 `prefill_ms`·`prefill_chunks`(256K: 63 → 약 125)·`max_prefill_chunk_tokens`(4096 → 2048). 판정 5 %·3 ms. 위 추정(0.2–1.4 %)을 넘으면 기록하고 사용자에게 올린다(margin 넓히지 않음).
5. checkpoint 내용: cache 재사용 시나리오(`shared`·`regress`)에서 capture 지점 목록과 checkpoint/출력 hash가 `7c2c5a72`와 같은지 비교한다.
6. functional suite `batch`·`cache`·`long-context`를 수정본에서 한 번.
7. flash-next를 운영에 쓰게 되면 그 모델의 혼자 도는 prefill도 같은 방식으로 잰다(이번 범위 밖).
