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
