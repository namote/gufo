# gufo paged KV 모델 확장: Qwen Flash-Next·Gemma 4·DeepSeek V4 Flash (2026-10-23)

상위 문서: [이식 계획](PLAN.md). 사용자 지시: "paged KV 모델 확장할꺼야. qwen flash next, gemma 같이 계획에다가 넣어줘." 이어서 "DeepSeek V4 Flash 도 해야겠네... 그거도". 설계만 한다.

## 원칙

- 프로젝트 공통 최적화 규칙(`AGENTS.md`)에 따라, 모델 이름이 아니라 **상태의 연산 계약**으로 paged 여부를 정한다.
- 블록 풀·블록 테이블·예약·CoW는 모델과 무관한 공통 코드(`src/core/hip/`, `src/cache/`)에 둔다. 모델은 어느 component를 paged로 둘지만 선언한다.
- 그래서 Qwen 27B 단계(U2·U3)의 풀·테이블 API부터 Qwen 전용으로 만들지 않는다. 이 문서의 세 모델이 같은 API로 붙을 수 있는지를 U2·U3 설계 검토 항목에 넣는다.

|상태 계약|paged 대상|이유|
|---|---|---|
|append rows: 앞에서부터 쌓이고, 다 찬 행은 바뀌지 않음(attention K/V)|예|블록 단위로 나눠 받고, 다 찬 블록을 checkpoint·세션 간에 공유할 수 있다|
|완료 블록 단위로 쌓이는 행(Flash-Next pooled indexer keys)|예|원본 RFC도 "completed immutable blocks"로 공유한다고 본다(`RFC.md:295`)|
|window ring: 같은 물리 자리를 다시 씀(Gemma sliding K/V, Flash-Next raw indexer keys)|아니요(1차)|블록이 계속 덮여 공유 이득이 없다. 세션 고유 버퍼로 둔다|
|private state(recurrent, convolution, logits, metadata)|아니요|경계마다 통째로 저장하는 상태라 행 단위 공유가 없다|
|draft 모델 KV(MTP·DFlash2)|1차 제외, U5에서 측정|크기가 작거나 ring이다. 측정 뒤 정한다|

## Qwen3.8 Flash-Next (원본 gufo-org/gufo `79c2dcc5`)

- **구조:** hybrid recurrent/QSA MoE 문맥·이미지 모델이다(`docs/models/qwen3.8-flash-next/README.md:3`). 기본 문맥 262,144이고, 그 이상은 YaRN이 필요하다.
- **이미 component cache를 쓴다.** 원본 RFC의 첫 이전 대상이었다(PR #543, 카드 14·21·22).
- **component**(`src/models/qwen38_flash_next/continuation_layout.hpp`):
  - private: tokens, logits, metadata, convolution, recurrent, PLE history, kept hidden
  - 행: key, value, pooled keys, raw keys
  - draft(MTP): key, value, pooled keys, raw keys, residual
- **paged 후보:** key·value(append rows)와 pooled keys(완료 블록)
- **paged 아님:** raw keys(원본 RFC가 ring으로 다시 쓴다고 적음, `RFC.md:286`), recurrent·convolution
- **측정할 것:**
  - 토큰당 KV 바이트와 세션당 최대 문맥 예약량. 문서에서 찾지 못해 U7 시작 때 코드로 계산한다.
  - 이것이 세션 수를 제한하는지. 제한하지 않으면 이득은 블록 공유와 슬롯 간 복원(U4 방식) 쪽에 있다.
- **제안 대상:** 원본 gufo-org/gufo
- **준비물:** 모델 파일이 `storage/models/`에 없다. 다운로드가 필요하다(크기 미확인, 결정 D8).

## Gemma 4 (franzmoca/gufo `feat/gemma`)

- **위치:** Gemma 4는 원본 main에 없다. 포크 franzmoca/gufo의 기본 브랜치 `feat/gemma`가 따로 관리한다.
  - 원본 PR #426은 "Gemma 4 isn't on upstream's roadmap"이라며 열린 채로 두었다(2026-10-23 확인: open, head `21aad161`).
  - 포크 head `21aad161`은 우리 기준(`a8e40fd8`)보다 67커밋 앞서 있고, 원본 `f020bd3a`까지 합쳐져 있다.
  - 그래서 **제안 대상은 포크**다(결정 D7).
- **이미 component cache를 쓴다.** 포크 `inference_backend.cpp:3526`의 `Gemma4TextRunner::UsesComponentCache()`가 true다.
- **component**(`src/models/gemma4/continuation_adapter.hpp:42-50`):
  - global 층 K/V: append rows → **paged 후보**
  - sliding 층 K/V: 세션 ring의 window rows → paged 아님(1차)
  - frontier logits, MTP drafter 입력 hidden, metadata: private
  - MTP drafter는 target 캐시를 읽으므로 추가 component가 없다.
- **특이점:** Gemma 어댑터는 capture 때 슬롯 행을 빌려주지 않고 복사한다. 빌려줘도 슬롯을 다시 배정하거나 복원하기 전에 결국 복사해야 하기 때문이다(같은 파일 주석).
  - paged 블록 공유가 들어가면 이 복사가 필요 없어질 수 있다. Gemma 쪽 주된 이득 후보다.
- **크기:** 26B-A4B는 global 층 KV head가 적어서, 131,072 문맥에 KV와 ring을 합쳐 약 2.3 GiB다(포크 `docs/models/gemma-4-26b-a4b/README.md:81-82`).
  - 그래서 세션 밀도 이득은 Qwen 27B(256K에 16 GiB)보다 작다.
  - 31B는 따로 계산한다.
- **준비물:**
  - 26B-A4B GGUF는 있다(`storage/models/llm/gemma-4-26b-a4b-it-ud-q4-k-m-gguf`).
  - 31B는 없다.
  - 운영 Gemma Tool은 vLLM 구성(정지)이다. gufo Gemma 운영 전환은 이 계획 밖의 별도 결정이다.

## DeepSeek V4 Flash (원본 gufo-org/gufo `79c2dcc5`)

- **모델:** Flash 0731 mixed IQ2/Q2/Q8 GGUF다. 가중치만 80.76 GiB이고, 원본은 128 GiB 장비에서 요청 상태용 여유를 남기라고 한다(`docs/models/deepseek-v4-flash/README.md:3-6`). target 정확도 대조(parity)는 원본에서도 아직 열려 있다.
- **아직 옛 snapshot 캐시를 쓴다.** 원본에서 유일하게 남은 옛 경로 사용 모델이다(`docs/KV-CACHE.md:40-42`).
  - component 어댑터는 카드 17로, 상태 "proposed", 순서상 Qwen 27B DFlash2 다음이다(`cards/README.md:41`, `:65`).
  - 원본 카드 17은 "DeepSeek은 연속 대화 용량 측정이 없다. 어댑터 전에 행 수와 committed backing을 재라"고 적었다.
- **상태**(카드 17이 꼽은 목록):
  - 압축 KV 풀
  - sliding window
  - indexer
  - 가변 compressor 상태
  - DSpark draft 상태

  원본 카드 17은 "압축 풀은 토큰당 한 행 구조가 아닐 수 있다. component마다 공유할지 고유로 둘지 정하라"고 적었고, chunk 구조를 RFC의 열린 질문으로 남겼다.
- **paged 후보:**
  - 압축 KV 풀: 압축 단위가 고정 크기로 쌓이는 append 구조라면 블록 단위로 paged할 수 있다. 확인이 필요하다.
  - sliding window, compressor 상태, draft: 위 원칙대로 고유 상태로 둔다.
- **우리에게 의미가 큰 이유:** 가중치가 80.76 GiB라, 128 GiB 장비에서 KV에 쓸 수 있는 메모리가 다른 모델보다 훨씬 적다. 세션을 최대 문맥만큼 미리 잡는 비용이 가장 아픈 모델이라 paged 이득이 클 수 있다. 다만 같은 이유로 다른 Tool과 동시에 띄우기 어렵다.
- **진행 방식 선택**(결정 D9):
  - (a) 원본이 카드 17을 구현할 때까지 기다렸다가 그 위에 paged를 얹는다.
  - (b) 우리가 카드 17(DeepSeek component 어댑터)을 paged 풀을 전제로 직접 구현해 원본에 제안한다. 카드 17이 아직 시작되지 않았으니 기여할 기회이지만, 규모가 크다(원본 표기 Size L).
  - 옛 snapshot 경로 위에 paged를 얹는 것은 하지 않는다. 원본이 그 경로를 카드 19에서 걷어 낼 예정이라 Qwen 27B 때와 같은 일이 반복된다.
- **제안 대상:** 원본 gufo-org/gufo
- **준비물:** 모델 파일이 없다. 다운로드가 필요하고, 가중치만 약 81 GiB다(결정 D8). 실행 중에는 GPU와 메모리를 이 모델 하나가 거의 다 쓴다.

## 단계 (PLAN.md 단계에 이어짐)

|단계|내용|통과 기준|
|---|---|---|
|U2·U3 설계 검토(추가, **완료**: [API 검토](API-REVIEW.md))|공통 풀·테이블·예약 API가 Flash-Next pooled keys(블록 단위 append), Gemma global K/V(층마다 head 수가 다름), DeepSeek 압축 KV 풀(구조 확인 전이라 가능성만)을 표현할 수 있는지 문서로 확인한다. 블록 크기는 component마다 다를 수 있게 한다|Qwen 27B 구현 전에 API 검토 기록|
|U7|Flash-Next에 paged KV를 적용한다. key·value·pooled keys만 paged로 하고, raw keys ring과 draft는 그대로 둔다. |T0~T3([테스트 설계](TESTS.md)). 출력 동일. 세션 밀도 또는 복원 비용 이득 수치. 단계 끝 T4s(16K 축소)|
|U8|Gemma 4 포크 최신(`feat/gemma`)에서 global K/V를 paged로 하고, capture 복사 대신 블록 공유를 쓴다. sliding ring은 그대로 둔다|T0~T3(26B-A4B). 출력 동일. capture 복사량·RAM 감소 수치. 31B는 단계 끝 T4s에서 1회|
|U9|DeepSeek V4 Flash. 먼저 원본 카드 17의 용량 측정(32K·100K에서 checkpoint 고유 바이트, 토큰당 공유 가능 바이트, capture·restore 시간)과 압축 풀 구조 조사를 한다. 그다음 결정 D9에 따라 원본 카드 17 위에 paged를 얹거나 카드 17을 paged 전제로 직접 구현한다|용량 표(서버 1회 기동). 압축 풀의 paged 가능 여부 판단. 구현 시 T0~T3, AR·DSpark 출력 동일과 이득 수치, 단계 끝 T4s|
|U10|U7·U9는 원본에, U8은 포크에 제안한다. U6에서 원본에 넣은 공통 API를 포크가 합친 뒤에 진행한다|각 저장소 리뷰 반영|

**순서:** U7·U9는 U3(공통 API가 원본에 들어간 상태)에 의존하고, U9는 결정 D9(카드 17을 기다릴지 직접 할지)에도 의존한다. U8은 포크가 그 API를 합쳤는지에 의존한다. 포크가 늦게 합치면 U8은 포크용 별도 브랜치에서 미리 검증만 해 둔다.

## 위험

- **Flash-Next 원본 상태:** Flash-Next 원본 component cache는 카드 21·22가 진행 중이다. 원본이 바꾸는 중인 코드에 paged를 얹으면 다시 맞추는 비용이 크다. U7 시작 전에 원본 상태를 다시 확인한다.
- **Gemma 이득 크기:** Gemma는 KV가 작아 이득이 크지 않을 수 있다. U8은 측정 뒤 채택 여부를 정하고, 이득이 없으면 기록만 남긴다.
- **두 저장소 동기화:** 포크와 원본, 두 곳에 같은 공통 API를 맞춰야 한다.
- **GPU·디스크:** Flash-Next(크기 미확인)·DeepSeek(약 81 GiB) 모델 다운로드와 모델마다 GPU 회귀 창이 필요하다.
- **DeepSeek 정확도 기준:** 원본에서도 target parity가 열려 있어, 출력 동일 비교는 같은 빌드의 paged 끈 상태와만 한다.
