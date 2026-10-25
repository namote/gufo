# gufo 원본 이식 U4c: system prompt 끝 checkpoint (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [U4b 결과](U4B-RESULTS.md).

사용자 결정: "B계획없애고 그냥 1번을 고치는거로 계획을 변경하자". U4b로 1번(공유 대화 첫 팔로워 지연)이 절반만 고쳐져서 이어서 한다.

## 왜

- U4b 뒤 첫 팔로워는 4.65 s, 운영은 2.29 s다.
- 운영은 system 메시지 끝(7,171)에 checkpoint를 둔다. 첫 팔로워는 그것을 복원하고 1,037토큰만 prefill한다.
- 이식본에는 이 경계가 없다. 그래서 tail 6,153을 복원하고 2,055토큰을 prefill한다([U4b 결과](U4B-RESULTS.md) "운영이 빠른 진짜 이유").

## 무엇을

구현: gufo `ai-tools/port-u4c` `852b6c72`(`9a5cacee`에서 분기). 운영 `e71947f5`의 해당 부분을 옮겼다.

1. **경계 찾기:** `TextModelRunner::SystemPrefixTokens(request, tokens)`(운영 코드 그대로).
   - 앞쪽 system/developer 메시지만 남기고 빈 user 턴을 붙인 probe를 렌더링한다. prompt와 처음 달라지는 토큰 위치가 경계다.
   - 모델 이름이 아니라 chat template 렌더링 결과로 정한다. system 메시지가 없거나, 그 뒤에 아무것도 없거나, 이미지가 있으면 0이다.
2. **전달:** `cache_prompt`일 때만 계산한다. scheduler 메타데이터(`system_prefix_tokens`)로 `TextRunnerPool::Acquire`까지 넘긴다.
3. **checkpoint:**
   - 경계가 `shared_prefix_min_tokens`(디스크 cache 옵션, 기본 128) 이상이면 `Request::ShareCheckpoint(경계)`를 부른다.
   - 이식본에서 배운 분기점과 같은 경로다. RAM은 continuation 우선순위이고, 디스크에도 저장한다. 1024 안의 tail은 뺀다(`DropTailCheckpointsNear`).
   - component cache 경로와 기존 cache 경로 둘 다 적용한다.

## 실패 처리

- 선택 checkpoint다. RAM 거절·capture 실패는 기존 이벤트로 남고 요청은 그대로 진행한다.
- 렌더링이 실패하면 경계는 0이다(checkpoint 없음). 다른 경로로 바꾸거나 재시도하지 않는다.

## 검증

- **단위(T0):** CPU 127/127, format 통과.
  - `component_text_cache_test` `FirstFollowerRestoresTheSystemPromptCheckpoint`: 공통 1,500토큰 뒤 다른 대화의 첫 팔로워가 정확히 1,500을 복원한다. 행 사본 capture에서도 같다. 토큰은 처음부터 계산한 것과 같다. 경계가 0이면 처음부터 시작한다.
  - 수정 없이 실패하는 것을 확인했다.
- **GPU 창:** U4b와 같은 창.
  - `final-f8-a01`: 운영과 번갈아 돌리고, T2를 포함한다.
  - `final-f9-a01`: RAM 2 GiB, 이식본만.
  - 판정: 텍스트·gold 동일, T2 21/21.
  - shared 첫 팔로워가 운영 수준(약 2.3 s)인지 본다. 재시작 뒤 T3(디스크)와 다른 창 TTFT·anon도 기록한다.
