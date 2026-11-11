# R4 속도 브리프: 와이드 채점 pass

## 문제

`Session::ScoreTokensImpl` (`src/models/qwen38_flash_next/engine.cpp`)은 답 토큰을 `exec.max_speculative()`(8)행씩 `kVerify` pass로 흘린다.
- 답 토큰 n개에 pass가 n/8번 돈다. 매 pass마다 MoE 가중치를 다시 읽고, MTP `DraftCatchUp`과 `Rollback`도 한다.
- SPSA 한 번(80걸음, 걸음당 약 158회 채점)이 3.5시간 걸렸다(`docs/GFX1151-KNOWHOW.md`의 SPSA 절).

## 목표

채점 pass 하나가 **최대 64행**을 처리하게 해서 pass 수를 8분의 1 이하로 줄인다. 결과값은 지금과 같아야 한다.
- 수치 오차 이내면 된다. 정확히 같은 비트일 필요는 없다(`speed-over-bit-exactness`).
- 기존 GPU 정확성 기준을 그대로 쓴다: greedy로 생성할 때 받은 logprobs와 같은 답을 채점한 logprob이 일치해야 한다(`tests/functional/score_assistant.py`).

## 참고할 코드

- 와이드 검증이 이미 있다.
  - `rocm::Executor::kWideRows`(64), `kWideSegmentRows`(8), `max_copy_rows()`.
  - `RollbackWide` (`runtime/executor_gdn.cpp`).
  - engine.cpp에서 `pending->wide`를 쓰는 경로.
- 와이드 경로가 행마다 full logits를 주는지, 아니면 greedy 비교용으로 압축된 결과(`CanCompactWideVerification`)만 주는지 먼저 확인한다.
  - full logits가 없으면 아래 둘 중 하나를 고른다.
    1. 채점 전용으로 행별 full logits를 받는다. 64 × vocab float 복사와 CPU log-softmax 비용을 따져 본다.
    2. GPU에서 행마다 log-softmax를 하고 목표 토큰 하나의 값만 모은다(gather). 비용이 더 싸면 이 방법을 쓴다.
  - 간단하고 빠른 쪽을 고르고, 고른 이유를 커밋 메시지나 코드 주석에 적는다.
- MTP draft 상태 처리(`DraftCatchUp`): 채점이 끝난 뒤에도 세션을 계속 쓸 수 있어야 한다. 지금 코드가 그 조건을 지키고 있다.
  - 와이드로 바꾸면 이 조건을 어떻게 지킬지 정한다. 채점 요청은 캐시에 상태를 남기지 않으므로(`cache_persist=false`), 채점 뒤에 세션을 무효로 만들고 반환하는 것도 허용된다. 그쪽이 더 단순하면 그렇게 해도 된다. 다만 scheduler와 runner 쪽 계약을 확인한다.
- abliteration 요청 설정이 와이드 pass에도 똑같이 적용되는지 확인한다. writer, residual, 층별 배열 모두 해당된다.

## 선택 항목 (위 목표가 끝나고 시간이 남으면)

- 답 앞 N토큰만 채점하는 요청 필드. 이름 제안은 `score_max_tokens`이고, 별칭은 두지 않는다.
  - 서버가 tokenize한 뒤 답 토큰을 N개로 자른다. 응답의 token_count는 자른 수가 된다.
  - 문서와 파싱 테스트를 함께 넣는다.

## 작업 방식

- 구현 → 다른 리뷰어 검토(메인 세션이 따로 띄운다) → 수정 → 재검토 순서다. **GPU 실행은 하지 않는다.** 빌드와 CPU 단위 테스트만 한다.
- 빌드: `tools/docker/build.sh --build-dir build/int-20261111 gufo`.
- CPU 테스트: `build/b1/build.sh <target>`, `build/b1/build.sh --run /build/obj/<test>`.
- C++ 커밋 전 format 검사: `tools/ci/check-format.py`. pinned clang-format 경로는 `build/abliteration/cpu.sh`를 참고한다.
- `docs/SERVER.md`의 score_assistant 비용 설명을 갱신한다.
- 하지 말 것:
  - 운영 서버(`llm`, 포트 18740)와 `~/.local/share/gufo/`를 건드리지 않는다.
  - `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`를 건드리지 않는다.
  - 푸시하지 않는다.
  - `pgrep -f`/`pkill -f`를 쓰지 않는다.
- 커밋은 Conventional Commits 한 줄로 쓰고, 끝에 다음 trailer를 붙인다.

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```

- 끝나면 다음을 보고한다.
  - 커밋 목록.
  - 고른 설계와 그 이유.
  - 예상 pass 수의 변화.
  - GPU 통합 패스에서 확인할 항목.
