# T: 테스트 정리 (에이전트 안내)

2026-10-27 사용자: 남은 일 순서 "불안정한 batch assertion → qwen_ssm_ops_test 중단 → 운영 전환 준비"에 "오케이 그렇게 진행하자". 이 안내는 앞의 둘(T1, T2). 감독이 단계별로 검증한다.

## T1: `batch_c2_greedy_short_replies_match_serial` 타이밍 경쟁

- 근거: SKINNY.md(s-w3·s-w4·S2-XL). `tests/functional/openai_sdk.py:788`의 `max(physical_execution_width) > 1` assert가 16행(8쌍) 중 하나도 같은 검증 단계에서 만나지 못하면 실패한다. 응답이 4–6토큰이라 많은 행이 draft 한 번에 끝나고, 겹침 여부는 스케줄러 타이밍(prefill interleave, `decode_due`, `assemble_initial_batch`)에 달린다. 기준 빌드에서도 0–10/16으로 흔들림.
- 목표: 이 검사가 **검사하려는 것**(동시 요청이 실제로 함께 실행될 수 있고, 그때 출력이 serial과 같음)을 결정적으로 확인하게 만든다. assert를 지우거나 "가끔 실패 허용"으로 약화하지 않는다. 먼저 이 검사의 의도를 git 이력과 주변 코드에서 확인한다.
- 후보(조사 후 고름): 요청 도착을 맞추는 방법, 겹침이 보장되는 길이·형태의 프롬프트를 별도 케이스로, 서버 쪽 결정적 batching을 쓰는 기존 옵션·관측값이 있는지. 서버 코드를 바꿔야 하면 먼저 감독에게.
- 기준: 같은 바이너리로 같은 케이스를 여러 번(예: 10회) 돌려 매번 통과, 그리고 serial 대조와 출력 일치 검사는 그대로. 다른 functional 케이스 결과 그대로.

## T2: `qwen_ssm_ops_test` 중단

- 근거: KVARN.md "K4 마무리". `qwen_ssm_ops_test`가 "KV block plane needs layers, row bytes and a power-of-two row …"(`src/core/hip/kv_block_layout.hpp:52`)로 중단. 이 검사는 `f8c9df17`(KVarN 이전)에서 들어왔다.
- 할 일: 원인(테스트가 만드는 layout이 잘못됐는지, 검사가 SSM 경우를 잘못 막는지)을 찾아 고친다. 운영 코드의 검사를 완화해야 하면 먼저 감독에게(운영 경로가 그 layout을 만들 수 있는지 근거와 함께).
- 기준: 테스트 PASS, 고친 쪽과 관련된 다른 테스트(같은 label) PASS.

## 규칙

- gufo 브랜치 `ai-tools/port-u4f`에 직접 커밋, push·PR·이슈 없음. 한 줄 Conventional Commits + 감독이 준 trailer. C++ 커밋 전 format check. 자기 파일만 `git add`, 관련 없는 작업 보존.
- GPU: ai.tools 창 실행기로만, 한 번에 하나, 창 전 `.gufo-revision` = HEAD, 창 끝 `sleep 20`, 자동 재시도 없음, 실패 포함 모든 결과 보존. GPU 창이 도는 동안 빌드 등 CPU 부하 작업 금지. 둘을 한 창에 묶어도 된다.
- 결과는 `plan/gufo-upstream-port/T-RESULTS.md`(새 파일)에.

## T1b: capture 복귀 스케줄 (2026-10-28 사용자: "스케줄러 수정")

- 근거: T-RESULTS.md. 짧은 요청은 `cache_prompt: false`여도 prompt snapshot 2개(약 320–430 ms)를 찍고, capture가 끝나면 그 요청만 due가 돼 다음 decode 단계를 혼자 돈다. 앵커 테스트(`e9a0ca53`)는 이 때문에 sessions 2에서 거의 매번 실패해 되돌렸다(`386b4320`).
- 수정: `src/cli/serve/text_generation_scheduler.cpp`의 capture drain에서, prefill이 끝난 요청이 capture에서 돌아오면 `MarkAllDecodersDue(decoding)`(prefill interleave 경로와 같은 규칙). 그다음 앵커 테스트를 다시 적용(`386b4320` 되돌리기).
- 이 변경은 모든 서버의 스케줄을 바꾼다. 확인: 서버 단위 테스트(scheduler·backend), functional `batch`(sessions 2 ×10, sessions 4 ×2, 전체 1회), `state-edges`, `cache`, `long-context`; DFlash2 C1/C2/C4/C8 cohort를 기준(`386b4320`)과 번갈아(base/new/base/new), 출력 같음·요청별 5 %·3 ms. 창 시간 한도는 계획 단계에서 넉넉히(t-w1의 `fn-c2-full` 누락 반복 금지).
- 결과는 T-RESULTS.md "T1b"에.
