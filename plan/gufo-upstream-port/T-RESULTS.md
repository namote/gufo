# T: 테스트 정리 결과

안내: [T-BRIEF.md](T-BRIEF.md). 감독 승인(2026-10-27): T2 수정 그대로, T1은 anchor 방식에 원래 짧은 쌍(unanchored)을 유지하고 그 coverage assert만 빼서 batched 수를 기록.

## 커밋

- `e7543862` test(qwen): keep recurrent-only rollback layouts on the flat FP32 reference KV (T2)
- `e9a0ca53` test(functional): batch short greedy replies beside a decoding anchor for deterministic coverage (T1)

## T2: `qwen_ssm_ops_test` 중단

- 원인: `TestRecurrentRollbackRows`(원본 #650 `95ba1eae`)가 주의 층이 없는 layout `{3,4}`·`{4,0}`을 `Production()` policy(FP16 KV)로 만든다. `f8c9df17` 뒤 FP16 KV는 block pool이라 `KvBlockGeometry`가 layers=0을 거절한다(`kv_block_layout.hpp:52`). 이전엔 0바이트 flat 할당이라 통과했다.
- 운영 경로는 이 layout을 만들지 않는다: Qwen 적재가 interval로 나누고(`weights.cpp:251`, `continuation_layout.cpp:83`), compact snapshot이 interval 0을 거절하고(`arena.cpp:308`), 모든 Qwen GGUF는 num_layers ≥ interval이다. 운영 검사는 그대로 둔다.
- 수정(테스트만): 주의 층이 없는 layout은 `kv_cache_storage = kFp32`(pool 없는 flat 참조 KV). recurrent rollback coverage는 FP32·BF16 그대로.
- 결과: `qwen_ssm_ops_test` PASS(모든 하위 검사 포함, 1.9 s), 같은 `ssm` label의 `qwen_attention_projection_ops_test` PASS(창 t-w1), CPU `qwen_ssm_test`·`qwen_l1_ssm_module_test` PASS(이미지 컨테이너 ctest).

## T1: `batch_c2_greedy_short_replies_match_serial`

- 원래 의도(`1b0a5a84`, #668/#612): batched 짧은 greedy 응답이 speculative block 안의 EOS에서 serial과 똑같이 멈춘다. width assert는 batching이 실제로 일어났다는 coverage 확인.
- 바꾼 테스트: (1) 원래 짧은 쌍은 그대로 돌리고 serial과 비교, batched 행 수만 `batch_c{W}_greedy_short_pairs_match_serial.batched_rows`로 기록. (2) 이어서 width-1개 짧은 행을 이미 decoding 중인 긴 anchor 요청 옆에서 돌리고 `all(width) > 1` + serial 일치를 요구(`..._greedy_short_replies_match_serial`).

### 창 `t-w1-a01/t-w1`

- 결과 FAILED_no_fallback, health PASS, 재시도 없음. 이미지 `ai-tools/gufo-eval:4a7046b2e8ed`(toolchain), `.gufo-revision` `e9a0ca53`. 바이너리는 HEAD에서 이미지 컨테이너로 빌드: `build/release/gufo` `4f1d0f8d…`, `qwen_ssm_ops_test` `39fd3703…`, `qwen_attention_projection_ops_test` `c0046f74…`. 계획 ai.tools `test/experiments/gufo_upstream_port_20261023/PLAN-T.json`, 증거 ai.tools `tmp/test-results/gufo-upstream-port-20261023/t-w1-a01/t-w1/`.
- 설정: Qwen3.8-27B UD-Q4_K_M + DFlash2, context 32768, think off, `batch` suite `--through-case`, 실행마다 새 server.

|실행|sessions|결과|짝 batched|anchor batched|
|---|---|---|---|---|
|fn-c2-0|2|FAIL|2/16|3/16|
|fn-c2-1|2|FAIL|6/16|9/16|
|fn-c2-2|2|FAIL|6/16|6/16|
|fn-c2-3|2|FAIL|4/16|4/16|
|fn-c2-4|2|FAIL|8/16|8/16|
|fn-c2-5|2|FAIL|2/16|13/16|
|fn-c2-6|2|FAIL|4/16|12/16|
|fn-c2-7|2|FAIL|4/16|7/16|
|fn-c2-8|2|FAIL|4/16|4/16|
|fn-c2-9|2|FAIL|4/16|12/16|
|fn-c4-0|4|PASS|14/16|12/12|
|fn-c4-1|4|PASS|14/16|12/12|

- 모든 실행에서 출력 텍스트·finish는 serial과 같았다(실패는 anchor coverage assert에서만, 그 전의 짝 비교는 통과).
- `fn-c2-full`(batch suite 전체)과 끝의 `sleep 20`은 돌지 않았다: 계획의 timeout 1800 s가 남은 시간 1785 s보다 커서 driver가 멈췄다(계획 실수).

### 원인 (anchor가 sessions 2에서 결정적이지 않은 이유)

- 짧은 요청마다 `cache_prompt: false`인데도 component cache capture가 2번 일어난다(`captures: 2`, `cache_snapshot_ms` 약 320–430 ms, 프롬프트 22토큰이 15+7 두 chunk). `cache_prompt`는 재사용만 끄고 prompt snapshot은 그대로 잡는다(`text_model_runner.cpp` "the complete prompt is offered after the first token").
- 요청은 prefill 뒤 `capturing`에 들어가고, 그동안 anchor는 혼자 decode한다(due 아님 → 전체 후보). capture가 끝나면 scheduler(`text_generation_scheduler.cpp` capture drain, "Resume it before another bounded chunk")가 **그 요청만** `decode_due = true`로 넣는다. anchor는 due가 아니므로 `due_decoder` 후보 1개 → 짧은 행이 혼자 한 step을 돌고 4–6토큰 응답이 그 block 안에서 끝난다. anchor의 step 경계와 capture 완료 시점이 맞을 때만 batched된다.
- sessions 4에서는 같은 그룹의 3행이 거의 같이 capture를 마쳐 서로 함께 due가 되므로 12/12였다(anchor 덕이 아니라 동료 행 덕일 수 있음).
- 같은 경로가 원래 짝의 flaky 원인이기도 하다: prefill 뒤 capture에서 돌아온 행이 혼자 due가 되어 혼자 끝난다.

### 판단 필요 (감독)

- 테스트만으로는 결정적으로 만들 수 없다(capture는 요청으로 끌 수 없음). 후보: scheduler capture drain에서 prefill이 끝난 요청을 넣은 뒤 `MarkAllDecodersDue(decoding)`(prefill 교차 경로와 같은 규칙)로 바꿔, 돌아온 요청이 다음 step에서 상주 decoder와 함께 batched되게 한다. 운영 코드 변경이라 승인 필요.
- 그 전까지 `e9a0ca53`은 sessions 2에서 거의 항상 실패한다(원래 assert보다 나쁨). 되돌릴지 감독 판단.

## `tests/functional/prompt_lookup.py:161`

- 같은 assert 형태(`max(physical_execution_width) == cohort_size`)지만 같은 경쟁은 아니다: 행이 warm cache 위의 128토큰 반복 응답이라 여러 step 동안 겹칠 기회가 있다. 다만 capture에서 돌아온 행이 혼자 due가 되는 같은 scheduler 경로를 지나므로 cohort가 커지면 이론상 같은 원인으로 흔들릴 수 있다. 실패 기록은 찾지 못했다. 바꾸지 않았다.

## T1b: capture 복귀 스케줄 (2026-10-28)

### 커밋

- `f53aba73` fix(serve): resume a captured request together with the resident decoders — capture drain에서 prefill이 끝난 요청을 `decoding`에 넣은 뒤 `MarkAllDecodersDue(decoding)`(그 요청만 due로 두던 `82711d8c`(#445) 규칙을 prefill 교차 경로와 같게).
- `6e0e03f2` test(functional): batch short greedy replies beside a decoding anchor for deterministic coverage — `386b4320` 되돌리기(앵커 테스트 + 원래 짝 유지).
- CPU: `text_generation_scheduler_test`, `text_model_runner_test`, `openai_chat_test` PASS. format check PASS.

### 창 `t1b-w1-a01/t1b-w1`

- 결과 FAILED_no_fallback(원인: `fn-base`·`fn-new` 아래), health PASS, 재시도 없음. 이미지 `4a7046b2e8ed`, `.gufo-revision` `6e0e03f2`. 계획 `PLAN-T1B.json`, 증거 ai.tools `tmp/test-results/gufo-upstream-port-20261023/t1b-w1-a01/t1b-w1/`. 모든 계획 항목이 돌았다(약 47분).
- 바이너리: base `build/t1b-base/gufo` `4f1d0f8d…`(src = `386b4320`), new `build/release/gufo` `e5899f9a…`, `inference_backend_gpu_test` `e16b2f87…`.

|항목|결과|
|---|---|
|`inference_backend_gpu_test`(new)|PASS (192 s)|
|functional base(`--record-baseline`)|batch FAIL(앵커 assert, 기준의 알려진 경쟁), 나머지 10 suite passed|
|functional new(`--baseline`)|11 suite 모두 passed(batch, state-edges, long-context, cache 8경우)|
|cohort base/new/base/new|모두 완료, 출력 hash 4회 같음(모든 C·workload)|
|앵커 sessions 2 ×10|모두 PASS, 앵커 행 batched 16/16 매번; 짝 batched 6,10,8,10,0,6,4,10,4,6/16|
|앵커 sessions 4 ×2|PASS, 앵커 12/12, 짝 12·13/16|

### functional 비교

- runner의 자동 비교는 기준이 실패해 만들어지지 않았다("functional checks did not pass"). 같은 harness의 `metrics.compare(fn-base, fn-new)`를 창 뒤 호스트에서 돌렸다(결과 scratch `fn-compare.json`, 코드 변경 없음).
- 측정 1739개: passed 1618, inconclusive 121, failed 0. quality/coverage 목록은 `batch_c2_greedy_short_replies_match_serial`의 case 변경(기준 실패·앵커 길이 차이)과 그 뒤 모든 요청의 `history_sha256` 변경뿐이다. 앵커가 취소되는 시점이 달라 서버 이력이 달라진 탓이라 batch 뒤 suite의 timing은 이력이 맞지 않는 비교다.
- inconclusive 113개는 batch의 짧은 응답 case(짝·앵커): 새 쪽에서 행이 batched되면서 draft 일이 달라져 요청별 decode가 길어진다(예: 앵커 행 decode 104 → 150–315 ms, 기준은 혼자 한 step). 의도한 동작 변화다.
- 그 밖 8개: `long_thinking_token_limit` decode 211 → 266 ms, `batch_c2_independent_samplers_and_grammars:0` prefill 325 → 405 ms, `cancelled_peer_isolation:3` queue/ttft +6 ms, `chat_responses_completions_replay:1` queue +4 ms, state-edges `cyclic_schema_rejected` wall 0.6 → 8.3 ms·`quoted_call_argument_stop_streamFalse` queue +7.6 ms, `disk-spacing-disk:1` snapshot +3 ms. 앞의 둘은 동료 없는 단일 요청(width 1, draft 같음)이라 이 변경이 아무것도 바꾸지 않는 경로다. 모두 한 번 실행·이력 불일치라 **inconclusive**로 둔다.

### cohort (DFlash2 greedy, UD-Q4_K_M, context 4096, C마다 새 server)

- 출력 hash는 모든 C·workload에서 네 번 같다.
- TG(base base / new new): mixed C1 35.66 35.65 / 35.69 35.63, C2 55.60 55.24 / 55.35 55.14, C4 71.25 71.45 / 71.35 70.98, C8 83.26 84.64 / 82.41 82.57(−1.7 %); repetition C1–C8 ±0.4 %.
- 요청별 gate: per-case 39 pass, 1 flag(mixed C8 `expository_pangram` 9284 → 10131 ms, base 8908/9659, new 10635/9626; base-base spread 9.9 %).
- C8 mixed 평균 decode_ms: base 11213·10978, new 11235·11270. 차이는 주로 base 두 번째 실행이 낮은 데서 온다. **inconclusive**(잡음 크기). 확정하려면 C8 mixed만 base/new 교대 재실행.

### 창 `t1b-w2-a01/t1b-w2` (재측정, 감독 승인)

- 결과 FAILED_no_fallback(`fnA-new`·`fnB-new`가 qualification inconclusive로 0이 아닌 종료), health PASS, 재시도 없음. `.gufo-revision` `686adea3`, 같은 바이너리(base `4f1d0f8d…`, new `e5899f9a…`). 계획 `PLAN-T1B2.json`. 양쪽 모두 `386b4320` harness(`build/t1b-old-src`, `git archive`)로 돌려 이력·harness hash가 맞는다. margin 그대로.
- **mixed C8 cohort B N B N B N**: 출력 hash 6번 같음. TG base 84.77 81.62 83.77, new 83.87 83.15 85.74(+1.0 %), 요청별 flag 0(matched 14 pass, per-case 9 pass, base-base spread 7.3 %). t1b-w1의 −1.7 %는 재현되지 않았다.
- **functional**(batch, state-edges, cache, long-context): fnA-base·fnB-base passed, fnA-new·fnB-new functional passed. 비교: A passed 1343·inconclusive 46·failed 0, B passed 1372·inconclusive 17·failed 0, quality/coverage 변화 0.

두 교대에서 모두 재현된 flag(같은 case·metric):

|요청|metric|A base → new|B base → new|
|---|---|---|---|
|`batch_c2_chat_responses_completions_replay:1`|queue_ms|5.2 → 9.6|5.3 → 9.9|
|`batch_c2_greedy_short_replies_match_serial`(짝 행)|queue_ms|0.1–0.3 → 5–7(대부분 행)|0.1–4 → 0.2–188(섞임)|
|같은 case|wall/server_duration·inter_token·cache_snapshot_ms|약 745 ↔ 860 ms, 136/193/240 ms 두 방향|같음(두 방향)|
|`disk-spacing-disk:1:0`(재시작 뒤 디스크 복원, 단일 요청)|cache_restore_ms(=queue_ms)|81 → 150|85 → 127|

- 짧은 쌍의 wall·snapshot 값은 행마다 두 방향으로 오가는 두세 개의 값(batched 여부와 snapshot 크기)이라 방향이 일정하지 않다. 일정한 쪽은 **queue_ms +4–7 ms**(A의 거의 모든 짝 행, 두 교대의 `replay:1`)와 **`disk-spacing-disk:1:0` 복원 +40–70 ms**다. 후자는 동료 없는 단일 요청이라 이 변경이 바꾸는 경로(상주 decoder를 due로 표시)가 직접 닿지 않는다. 원인 미조사.
- 한 교대에서만 나온 flag(13): `cancelled_peer_isolation:2/3` queue·ttft +5–8 ms, `replay:0` client_ttft +8 ms, 짝 `:1` queue +6 ms, `disk-spacing-disk:0:0` 복원 +9 ms·ttft +15 ms, `disk-spacing-disk:1:0` snapshot +7 ms, server restart_ms 2816 → 3014, `execution_mode_probe` queue +5 ms.
- 감독 지시대로 재현된 느려짐이 있어 여기서 멈춘다(수정 시도 없음).

### 재현 flag 조사 (오프라인, 기존 로그·지표만)

**queue +4–7 ms (짧은 쌍)**: 상주 decode step 뒤에서 기다린 것이 아니다. step은 약 105 ms인데 기다림은 5–7 ms이거나 약 190 ms다.
- 5–7 ms는 component cache의 `global_spill.residual_wait_ms`(누적)와 같은 크기로 늘어난다. fnA-new는 그룹마다 1.4 → 42.7 ms, fnA-base는 1.7 → 2.8 ms다. 앞 쌍의 비동기 snapshot 복사(snapshot 2개, 약 500 MB)가 아직 끝나지 않았을 때, 다음 쌍의 admission이 backing을 재할당하며 그 남은 복사를 기다린 값이다.
- 이 상태는 기준에도 있다. fnB-base는 1.8 → 29.7 ms(행 queue 2.5–5.2 ms)이고 fnB-new는 0.6 → 3.0 ms다. 같은 harness로 돈 10+10회(t-w1 fn-c2 = 기준 코드, t1b-w1 short-c2 = 새 코드)에서 짝 행 queue 중앙값은 기준 4.5 ms·새 3.0 ms, queue > 3 ms 행은 95·80/160, 실행당 residual 증가는 둘 다 0 또는 30–70 ms의 두 갈래다.
- 약 190 ms 기다림은 두 번째로 들어온 행이 첫 행의 prompt capture(약 195 ms) 동안 막힌 경우로, 기준 4/160·새 10/160이다(우연 범위, 둘 다 있음).
- 따라서 이 변경과 무관하게, 앞 복사가 끝나는 시점과 도착 시점이 맞물리는 이력 상태다. 도착 시 prefill을 먼저 넣는 규칙은 이 기다림을 줄이지 못하므로 제안하지 않는다.

**`replay:1` queue 5 → 10 ms**: 동시에 도착한 두 warm 요청(chat·responses)이 각자 약 5 ms 복원을 Admit에서 차례로 한다. 먼저 받은 쪽이 약 5 ms, 다음이 약 10 ms를 기다린다. 기준은 responses가 먼저(chat 9.7·9.6 ms), 새 쪽은 chat이 먼저(responses 9.6·9.9 ms)였고, 두 요청의 합(약 15 ms)과 최댓값은 같다. 이때 scheduler는 비어 있었다(앞 요청 r10 완료, 상주 0). 이 변경이 닿는 경로(상주 decoder를 due로 표시)는 돌지 않았으므로 HTTP 쪽 제출 순서 경쟁이다.

**`disk-spacing-disk:1:0` 복원 +40–70 ms**: 서버 로그 `event=disk_restore`의 세부를 보면 늘어난 몫은 모두 `filesystem_ms`다. 기준 41–45 ms, 새 쪽 112 ms(A)·85 ms(B), t1b-w1의 새 쪽은 41 ms다. `stream_wait_ms`(19–22), `checksum_ms`(13.3–13.5), `staging_wait_ms`(≈0), bytes(303,403,296)·pieces(350)는 같다. 재시작 직후의 동료 없는 단일 요청이 앞서 쓴 디스크 cache 파일을 읽는 시간이라 scheduler와 무관하다(page cache·SSD 상태). 이번 창에서 new는 늘 쌍의 두 번째로 돌아 앞 쌍의 cache 파일이 page cache를 더 차지한 상태였다. 순서와 binary가 섞여 있으므로 순서를 바꾼 확인(N B B N)이 필요하다.
