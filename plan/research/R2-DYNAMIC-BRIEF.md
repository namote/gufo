# R2-2 브리프: 동적 abliteration (감지했을 때만 켜기)

설계 배경은 `plan/research/IDEAS.md`의 R2와 R3이다. 측정 기록은 `docs/GFX1151-KNOWHOW.md`의 abliteration과 정치 방향 절에 있다. 먼저 읽는다.

## 목표

- 평소에는 OFF(원본 그대로, 원본 속도)로 답한다.
- 프로파일마다 감지기를 두고, 점수가 기준을 넘은 프로파일만 그 요청에서 켠다.
- 첫 대상은 정치 프로파일 하나다(`~/.local/share/gufo/runtime-20261111g/abliteration-pol-avg.json`, 방향 L19 projected).
  - 감지기는 L38 평균차 방향이다(`build/b1/ablit-pol-20261111/roc.py`).
  - 거절 프로파일(`abliteration-fnx.json`)도 같은 구조로 올릴 수 있어야 한다.
  - 프로파일끼리 합치지 않고 따로 둔다(사용자 결정).

## 범위

1. **감지기 학습 도구 (CPU, Python 표준 라이브러리와 numpy).** 이름 예: `tools/qwen38_flash_next/abliteration/fit_trigger.py`.
   - 입력:
     - 수집된 hidden(`build/b1/ablit-pol-20261111/collection`, `build/b1/ablit-20261110/collection`, 층별 2560 float, 데이터 jsonl의 group)
     - 대상 group
     - 감지 층
   - 방향: 대상 group 평균 − ordinary 평균, 정규화.
   - **검증 분리는 주제(질문 id 묶음) 단위로 한다.** roc.py는 같은 주제의 다른 문장이 학습과 검증에 섞여 낙관적이다.
   - 기준치: 검증 ordinary와 무해 질문의 오탐이 1% 이하가 되는 값. AUC, 재현율, 오탐을 출력한다.
   - 결과는 프로파일 JSON에 `trigger {layer, direction, threshold, ...}`로 넣는다. 원본 프로파일은 수정하지 않고 새 파일로 쓴다.
   - **엔진이 점수를 계산하는 위치와 수집기(engine.hpp의 collector)가 hidden을 잡는 위치가 정확히 같아야 기준치가 그대로 맞는다.** 코드에서 확인하고 문서에 적는다.
2. **엔진 (Qwen3.8-Flash-Next).**
   - **설정:** `--abliteration-profiles DIR`. 단일 옵션 원칙에 따라 기존 `--abliteration-profile FILE`을 대체하는지, 공존하는지는 코드를 보고 정한다. 별칭은 두지 않는다.
     - 디렉터리의 각 프로파일이 방향, 세기, `trigger`를 가진다.
     - `trigger`가 없는 프로파일은 지금처럼 요청이 명시할 때만 켠다.
   - **감지:**
     - 감지 층에서 프로파일별 점수 `<h, d>`를 계산하는 작은 커널을 둔다.
     - 대상 행: 프리필은 마지막 프롬프트 행, 디코드는 각 행.
     - **점수는 logits와 같은 host 전송에 실어 보낸다.** 추가 동기화를 만들지 않는다.
     - 감지기가 하나도 없거나 요청이 abliteration을 명시하면 감지 커널을 돌리지 않는다.
   - **적용 (A1, 필수):**
     - 프리필 끝에서 기준을 넘은 프로파일을 생성부터 켠다. 프롬프트 KV는 OFF로 계산된 그대로 둔다.
     - 디코드 중에 넘으면 다음 토큰부터 켜고, 그 요청이 끝날 때까지 켜 둔다(sticky).
     - 여러 프로파일이 동시에 켜지면 기존 다중 방향(k-direction) 경로로 함께 제거한다. 첫 단계에서 단일 프로파일만 지원하는 것이 훨씬 단순하면 그렇게 하고 보고한다.
   - **적용 (A2, 가능하면):** 감지된 요청만 ON으로 프롬프트를 다시 프리필한다. 비용이 크거나 침습적이면 설계만 적고 넘어간다. 품질 비교는 GPU 통합 패스에서 한다.
   - **요청 우선순위:** 요청의 `abliteration` 필드가 있으면 그것이 이긴다(명시 ON/OFF, 기존 동작 그대로). 없으면 동적 감지를 쓴다.
   - **보고:**
     - 응답에 `abliteration: {triggered: [...], scores: {...}, position}`를 넣는다.
     - `/props`에 로드된 프로파일과 기준치를 보여 준다.
   - **캐시와 continuation:**
     - 켜진 프로파일 조합마다 continuation identity를 분리한다. 지금의 ON/OFF 분리를 확장한다.
     - A1에서는 프롬프트 KV가 OFF 그대로이므로, 다음 턴이 그 접두부를 재사용해도 되는지 판단해 적는다.
   - **경로 범위:**
     - 단일 디코드, batched decode, MTP 검증, 채점(score_assistant)에서 동작이 일관되어야 한다.
     - 채점 요청은 명시된 abliteration만 쓰고 감지를 하지 않는다. 학습 재현성 때문이다.
3. **문서:** `docs/SERVER.md`에 옵션, 프로파일 형식(`trigger`), 응답 필드, 동작을 적는다.

## 성능 조건

- 동적 프로파일이 로드되지 않은 서버는 지금과 완전히 같은 경로를 탄다.
- 감지만 하고 켜지지 않은 요청(OFF)의 비용 목표는 0.5% 이하다. 감지 커널은 행 하나, 2560차원 내적 몇 개 수준이어야 한다.
- 켜진 뒤의 비용은 지금 ON과 같다. 커널 합치기(R2-3)는 이 브리프 범위가 아니다.

## 작업 방식

- 구현 → 다른 리뷰어 검토(메인 세션이 띄운다) → 수정 → 재검토 순서다.
- **GPU 실행은 하지 않는다.** 빌드와 CPU 단위 테스트만 한다. 스케줄러/엔진 fake 테스트로 트리거, sticky, 우선순위, 보고, 캐시 분리를 확인한다.
- 빌드: `tools/docker/build.sh --build-dir build/int-20261111 gufo`.
- CPU 테스트: `build/b1/build.sh <target>`, `build/b1/build.sh --run /build/obj/<test>`.
- C++ 커밋 전 format 검사: `tools/ci/check-format.py`. pinned clang-format 경로는 `build/abliteration/cpu.sh`를 참고한다.
- 하지 말 것:
  - 운영 서버(`llm`, 포트 18740)와 `~/.local/share/gufo/`를 건드리지 않는다. 읽기만 한다.
  - `tools/exl3_quant/hc/`, `plan/exl3/HC-Q8-CALIBRATION.md`를 건드리지 않는다.
  - 푸시하지 않는다.
  - `pgrep -f`/`pkill -f`를 쓰지 않는다.
- 커밋은 Conventional Commits 한 줄로, 단계마다 나눠서 쓴다. 끝에 다음 trailer를 붙인다.

```
Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_016Kh8svoeuCWVkepKUHx2XF
```

- 끝나면 다음을 보고한다.
  - 커밋 목록
  - 고른 설계와 이유(옵션 대체 여부, A2 여부, 다중 프로파일 여부)
  - 감지기 학습 결과(주제 단위 검증의 AUC, 오탐 1% 기준 재현율)
  - 만든 프로파일 파일 경로
  - GPU 통합 패스에서 확인할 항목
