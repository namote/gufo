# gufo 원본 이식 U4d: DFlash2 prompt feature pinned 버퍼를 슬롯이 나눠 쓰기 (2026-10-24)

상위 문서: [이식 계획](PLAN.md), [anon 바닥](ANON-FLOOR-RESULTS.md).

사용자 결정: "응~ 그대로 진행". 내용은 다음 셋이다.
1. 이식본 RAM 예산은 2 GiB로 한다.
2. DFlash2 버퍼 공유를 시도한다.
3. 그 빌드로 TRIM한 SSD에서 운영과 번갈아 다시 잰다(운영 전환 제안 전).

## 왜

- 이식본은 슬롯마다 DFlash2 prompt feature를 담을 pinned host 버퍼를 시작 때 잡는다(원본 #688 `b0eb5d47`).
  - 크기는 `4096 × target 층 × hidden × FP32`이고, 슬롯 4개에 400 MiB씩 모두 1.56 GiB다.
  - 운영보다 anon이 많은 이유 중 가장 큰 부분이다.
- 이 행은 prefill pass 하나에서 capture해 draft가 inject할 때까지만 필요하다.
  - inject는 동기 복사다(`InjectTargetContextChunk`가 stream을 비운 뒤 돌아온다).
  - serve에서는 scheduler worker 스레드 하나가 slot마다 차례로 `Prime`·`ExtendPrompt`를 부른다. capture와 inject가 한 호출 안에서 끝난다.
  - 그래서 동시에 살아 있는 capture는 하나뿐이다.

## 무엇을

구현: gufo `ai-tools/port-u4d` `d42b292c`(`852b6c72`에서 분기).

1. **`PromptHiddenPinned`(공유 pinned 페이지):**
   - allocation, ledger charge, 실패 플래그, 지금 쥔 버퍼(id), 그 버퍼의 stream을 가진다.
2. **`PromptHiddenBuffer`:**
   - 크기·pageable 사본·상태는 지금처럼 버퍼마다 따로 둔다. pinned 페이지만 공유한다.
   - capture를 예약하거나 늘릴 때 페이지를 넘겨받는다. 이전 holder의 stream을 먼저 비운다.
   - 페이지를 빼앗긴 버퍼의 행은 없어진 것으로 다룬다.
     - `View()`는 빈 span을 돌려준다. 호출부는 이미 "capture 불완전"으로 거절한다.
     - 행이 있는 채로 늘리려 하면 `logic_error`를 던진다.
     - 다른 슬롯의 내용을 조용히 돌려주는 일은 없다.
   - 기본 생성자는 자기 페이지를 가진다. 원본 동작과 기존 테스트는 그대로다.
3. **`ContinuationAdapter`:**
   - 공유 페이지 하나를 갖고, 슬롯을 만들 때 executor에 넘긴다(`SharePromptHiddenPages`).
   - `PinnedCaptureBytes()`는 공유 페이지의 charge를 한 번만 센다.
   - 슬롯마다 더하던 회계는 뺐다.

## 실패 처리

- pinned 할당·charge가 거절되면 원본처럼 pageable로 capture한다. 이 상태는 계속 유지되고, 이제는 공유한 모든 버퍼가 같이 pageable로 간다.
- 행을 빼앗긴 뒤 접근하면 예외나 빈 결과가 된다. 요청은 실패로 드러나고 다른 경로로 재시도하지 않는다.
- serve 경로에서는 일어나지 않아야 한다. 일어나면 버그이므로 그대로 기록한다.

## 검증

- **T0:** CPU 127/127, clang-format 통과.
  - GPU 테스트 `qwen_prompt_hidden_buffer_test`에 `TestSharedPages`를 더했다. 검사 내용:
    - 한 번만 할당·charge된다.
    - 넘겨받으면 같은 페이지를 쓴다.
    - 빼앗긴 행은 비어 보이고, 그 행을 늘리면 실패한다.
    - Clear 뒤에는 다시 capture한다.
    - 페이지가 해제되면 charge가 풀린다.
- **GPU 창:** `final-f10-a01`(`--set f10`, RAM 예산 2 GiB, run 전 `fstrim`).
  - GPU 단위 테스트를 돈다(DFlash2 연속성 테스트 포함).
  - f9와 같은 창을 운영 `210c7633`과 번갈아 돈다. T2도 돈다.
  - `memprobe`를 돈다.
- **판정:**
  - 텍스트·gold·T2 21/21이 같아야 한다.
  - 준비 직후 anon이 약 1.17 GiB 줄어야 한다(19.47 → 약 18.3 GiB).
  - TTFT 손해가 없어야 한다(prefill pass는 원래 한 번에 하나).
