# R4 브리프: 어시스턴트 답 로그확률 채점 (teacher forcing) API

## 목적

정치 프로파일의 층별, 위치별 abliteration 배수 144개(48층 × attention/mlp/residual)를 SPSA로 학습하려면, 채점이 연속값이고 빨라야 한다. 채점 방식은 다음과 같다. 질문과 정해진 답(사실 답 또는 선전 답)을 넣고, **답 토큰마다 모델의 로그확률**을 받는다. 생성하지 않는다. abliteration 요청 설정(현재 chat 요청의 `abliteration` 객체)이 그대로 적용돼야 한다.

## 요구사항 (Flash-Next 우선, 27B는 구조가 허락하면 같이)

- **API.** Chat Completions 요청이 마지막 메시지로 `assistant` 메시지를 가지고 확장 필드 하나로 채점을 요청한다. 필드 이름은 하나만 둔다(별칭 금지, AGENTS.md). 제안: `"score_assistant": true`, `max_tokens: 0`.
  - 응답에 그 assistant content 토큰들의 `{token, id, logprob}` 배열을 넣는다. 원시 모델 log-softmax이고, sampling, penalty, Han scale 같은 변형은 적용하지 않는다(기존 logprobs와 같은 정의).
  - thinking: template에서 assistant 턴을 어떻게 렌더할지(예: `<think></think>` 빈 블록) 명시하고 테스트로 고정한다.
- **계산.**
  1. 프롬프트(답 앞부분)는 평소처럼 프리필한다.
  2. 답 토큰은 `max_logit_rows`(8) 이하 행씩 나눠, 각 행의 logits를 받는 pass(verify 또는 decode 산술)로 지나간다.
  3. 행 r의 logits에서 다음 토큰(답 토큰 r+1)의 log-softmax 값을 뽑는다.
  - 마지막 프롬프트 행의 logits로 답의 첫 토큰을 채점하는 것도 포함한다.
  - 64토큰 답 기준 pass 8개, 수백 ms가 목표다.
- **캐시와 세션.**
  - 채점 요청은 continuation 캐시에 상태를 남기지 않는다(`cache_persist=false`와 같은 정책 또는 더 엄격하게).
  - 다른 세션과 섞이지 않게 하고, 끝나면 세션을 정상 반환한다.
  - abliteration ON/OFF identity 규칙을 그대로 따른다.
- **제한.**
  - stream은 거부한다. `n>1`, tools, 이미지 입력은 처음에는 거부해도 된다(명확한 에러로).
  - 답 길이 상한을 둔다(예: 1024 토큰).
- **문서.** `docs/SERVER.md`에 필드, 응답 형식, 제한, 비용을 적는다.
- **테스트.**
  - CPU/단위: 요청 파싱, template 렌더, 에러, 스케줄러/러너 fake.
  - GPU 테스트는 빌드만 하고 실행은 마지막 통합 패스에서 한다.
  - **정확성 기준:** 같은 프롬프트에서 greedy로 생성한 답을 채점하면, 각 토큰의 logprob이 생성할 때 돌려준 `logprobs`와 같거나 수치 오차 이내여야 한다.

## 작업 방식 (AGENTS.md, CLAUDE.md)

- 구현 → 다른 리뷰어 검토 → 수정 → 재검토 순서로 한다. GPU 실행은 하지 않는다. 빌드와 CPU 테스트만 한다.
- 빌드: `tools/docker/build.sh --build-dir build/int-20261111 gufo`. 운영 runtime과 같은 설정의 incremental obj다.
- CPU 테스트: `build/b1/build.sh <target>`, `build/b1/build.sh --run /build/obj/<test>`.
- C++ 커밋 전 format 검사: `tools/ci/check-format.py`. pinned clang-format 경로는 `build/abliteration/cpu.sh`를 참고한다.
- 하지 말 것:
  - 운영 서버(`llm`)를 건드리지 않는다.
  - `~/.local/share/gufo/`를 수정하지 않는다.
  - `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`(다른 세션 소관)를 건드리지 않는다.
  - 푸시하지 않는다(메인 세션이 한다).
- 커밋 메시지는 Conventional Commits 한 줄이고, 끝에 다음 trailer를 붙인다.

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8sw7Kn3qQfj7zrLHp1W
```

## 이후 (메인 세션)

1. 데이터셋을 만든다(질문, 사실 답, 선전 답 약 50쌍).
2. SPSA 드라이버(`tools/qwen38_flash_next/abliteration/`)를 만든다. 목표는 사실 답과 선전 답의 평균 로그확률 차이에, 일반 질문 KL 벌점을 더한 것이다.
3. GPU 통합 패스에서 정확성 기준을 확인한 뒤 학습을 돌린다.
