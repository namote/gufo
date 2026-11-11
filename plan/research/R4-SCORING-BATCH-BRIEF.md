# R4 묶음 채점 브리프: 여러 score_assistant 요청을 한 pass에

## 측정으로 확인된 문제 (`docs/GFX1151-KNOWHOW.md`의 "Scoring speed" 절)

- 채점 한 번에 약 0.2초(프롬프트) + 답 토큰당 약 4.5 ms가 든다.
  - 답 토큰은 세션 하나가 8행 검증 pass로 흘린다.
  - 64행 와이드 pass로 바꿔도 이득이 없다. 1M 운영 설정에서는 와이드 폭이 8로 좁혀지고, 넓혀도 batch MoE가 pass마다 고정비(약 148 ms)를 낸다.
- SPSA 걸음 하나는 채점 약 216회다(평가 3회 × 72). 지금은 그 하나하나가 따로 pass를 돈다.

## 목표

여러 채점 요청(세션)의 행을 **한 번의 forward pass**에 묶어 MoE 가중치 읽기를 나눠 낸다.
- 목표 수치: 72개 × 약 150토큰을 평가 한 번에 약 10초. 지금은 약 60초다.
- 결과값은 지금과 같아야 한다. 수치 오차 정도의 차이는 허용한다. 정확성 기준은 `tests/functional/score_assistant.py`(생성할 때 받은 logprobs와 일치)다.

## 생각할 수 있는 설계 (조사해서 고른다)

1. **다중 세션 배치 forward.** 이미 batched decode/verify가 있다.
   - 스케줄러의 `batched-w*`, `batched_multi_token_decode`, 엔진의 `ForwardBatchImpl`과 session row span이 그것이다. per-session abliteration도 여기서 이미 행 범위별로 적용된다.
   - 상주 중인 채점 요청들이 각자 다음 덩어리(최대 8행 또는 가능한 폭)를 내고, pass 하나로 같이 돈다.
   - 프롬프트 프리필 단계도 같은 방식으로 묶을지, 프롬프트는 따로 두고 답만 묶을지 정한다. 프롬프트는 20~70토큰으로 짧다.
2. **패킹.** 짧은 시퀀스 여러 개를 한 세션에 이어 붙이고 attention과 GDN 상태를 경계에서 끊는다.
   - 더 근본적이지만 GDN recurrent 상태 리셋과 마스크 때문에 침습적이다. 1이 막힐 때만 고려한다.

1이 우선이다. 가능하면 1로 끝낸다.

## 세션 수와 메모리

- 묶으려면 동시에 상주하는 채점 세션이 많아야 한다(예: 16~32). 운영 설정(1M YaRN, K4V4, `--kv-pool-bytes 0`)에서는 세션마다 전체 컨텍스트 KV를 잡아서 불가능하다.
- 학습용 서버는 **같은 1M YaRN**(RoPE가 운영과 같아야 한다)에 `--kv-pool-bytes N`(공유 풀)과 `--sessions 16..32`로 띄우는 것을 기본 가정으로 한다.
  - 이 조합이 실제로 메모리에 들어가는지, 세션마다 컨텍스트에 비례해 따로 잡는 메모리가 있는지(GDN 상태, MTP, hidden row, 와이드 scratch 등) 코드에서 확인한다.
  - 안 되면 가장 작은 변경을 제안하고 구현한다.
- 이 설정에서 일반 생성 경로는 바뀌면 안 된다. 운영 동작(단일 사용자 속도, 기존 batched decode, time-share)에 영향이 없어야 한다.

## 클라이언트

- `tools/qwen38_flash_next/abliteration/spsa.py`의 `--concurrency`를 세션 수에 맞춰 올리면 요청이 동시에 상주하게 된다.
- 필요하면 드라이버 쪽도 고친다. 예: 같은 프롬프트의 candid와 propaganda를 함께 보내기.

## 작업 방식

- 구현 → 다른 리뷰어 검토(메인 세션이 띄운다) → 수정 → 재검토 순서다. **GPU 실행은 하지 않는다.** 빌드와 CPU 단위 테스트만 한다(스케줄러 fake 테스트로 채점 요청이 묶이는지 확인한다).
- 빌드: `tools/docker/build.sh --build-dir build/int-20261111 gufo`.
- CPU 테스트: `build/b1/build.sh <target>`, `build/b1/build.sh --run /build/obj/<test>`.
- C++ 커밋 전 format 검사: `tools/ci/check-format.py`.
- `docs/SERVER.md`에 채점 묶음 동작과 학습용 서버 설정을 적는다.
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
  - 고른 설계와 이유.
  - 학습용 서버 실행 인자(세션 수, 풀 크기).
  - 예상 pass 수와 시간 계산.
  - GPU 통합 패스에서 확인할 항목.
