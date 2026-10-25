# `--cache-tail-checkpoints` 존치 검토 (2026-10-25)

상위 문서: [이식 계획](PLAN.md), [U4b 결과](U4B-RESULTS.md), [PR 분할 계획](PR-PLAN.md).

사용자 지시: "--cache-tail-checkpoints 존치 여부는 근거를 확인해서 추천안과 함께 보고해줘."

## 무엇인가

- 긴 prompt의 끝 앞쪽에 1024토큰마다 RAM checkpoint를 K개(기본 3) 남긴다. 다음 요청이 prompt 끝 근처에서만 다르면 차이 지점에서 1024토큰 안쪽부터 복원한다.
- checkpoint가 KV를 풀 블록 참조로 갖는 경우(paged KV)에만 적용되고, 아니면 시작 로그에 `event=tail_checkpoints status=not_applied`를 남긴다.
- 운영 포크 `210c7633`에서 옮긴 기능이다(U4b, 이식 커밋 `2321f5ad`, PR ③-5). 원본에는 없다.
- U4c 이후로는 system 메시지 끝 checkpoint가 shared 창의 첫 팔로워를 맡는다. 그래서 꼬리 checkpoint가 아직 필요한지 다시 쟀다.

## 측정 (gufo `663d50c9`, 256K × 4, RAM 2 GiB, 같은 서버 인자에 `--cache-tail-checkpoints 0`만 추가)

### 기존 6개 시나리오 (`final-f18-a01`, 0과 3을 번갈아, shared·boundary·regress는 두 번)

|창|TTFT 합 3 / 0 (초)|prefill 합|재사용 합|RAM checkpoint 저장 3 / 0|최대 anon|
|---|---|---|---|---|---|
|shared|45.3, 49.6 / 44.7, 48.4|같음(13,606)|같음(86,366)|18, 16 / 16, 16|18.39 / 18.39|
|boundary|22.6, 24.6 / 22.6, 22.6|같음(8,373)|같음(3,759)|15 / 14|18.38 / 18.38|
|regress|21.5, 23.7 / 21.4, 24.7|같음(7,724)|같음(7,518)|18, 14 / 18, 14|18.38 / 18.38|
|two8k|57.3 / 56.5|같음(24,894)|같음(16,814)|15 / 8|18.38 / 18.38|
|long32k|242.1 / 242.1|같음(98,622)|같음(65,966)|15 / 8|18.40 / 18.40|
|T2(21)|66.4 / 66.3|같음(24,357)|같음(8,935)|37 / 34|18.52 / 18.50|

- 모든 창에서 텍스트가 같고, prefill과 재사용 토큰 합이 0과 3에서 정확히 같다. 이 6개 창에서는 꼬리 checkpoint가 한 번도 복원에 쓰이지 않았다.
- 비용: 긴 prompt 창에서 RAM checkpoint 저장이 7회 더(two8k·long32k 8 → 15), RAM 용량 거절 0 → 2회. TTFT·메모리 차이는 편차 안이다.

### 꼬리 checkpoint가 노리는 경우 (`final-f18-b01`, document 시나리오, 0과 3을 두 번씩 번갈아)

- 48K 토큰 문서 하나를 user 메시지에 넣고, 끝에 매번 다른 질문을 붙인다. 분기점이 메시지 경계가 아니라 system checkpoint가 덮지 못한다.

|요청|꼬리 3 (2회)|꼬리 0 (2회)|
|---|---|---|
|첫 질문(cold, 48,111 prefill)|91.2, 91.4 s|91.0, 91.1 s|
|**두 번째 질문**|**4.42, 4.42 s**: RAM 47,080 복원, 1,031 prefill|**94.0, 94.3 s**: 재사용 0, 48,111 전부 prefill|
|세 번째·네 번째|0.90 s(47,988 복원)|0.90 s(47,988 복원)|
|cache 끈 기준|94.5, 94.6 s|95.2, 95.5 s|

- 텍스트는 4개 창 모두 같고 gold도 전부 맞다. 최대 anon은 18.42 GiB로 같다.
- 꼬리 checkpoint가 없으면 첫 prompt의 checkpoint는 끝(stable 경계)에만 있어 두 번째 질문의 prefix가 아니다. 그래서 두 번째 질문은 전부 다시 계산한다(94 s). 세 번째부터는 두 번째 요청이 분기점(47,988)을 배워서 같아진다.

## 추천안

- **존치. 기본값 3을 유지한다.**
  - 이득: 같은 긴 문서에 다른 질문을 하는 경우 두 번째 요청이 94 s → 4.4 s(약 21배)이다. 문서 Q&A, 같은 파일을 붙인 에이전트 요청처럼 흔한 형태다.
  - 비용: 긴 prompt마다 RAM checkpoint 저장 몇 회(측정에서 +7회)와 드문 RAM 용량 거절이다. 측정한 어떤 창에서도 TTFT·메모리 손해는 없었다.
- 원본 PR에서는 ③-5의 `2321f5ad` 한 커밋이라 따로 뺄 수 있다. 이슈에서 원본이 꼬리 대신 다른 방식(예: prompt 안 고정 간격 checkpoint)을 원하면 그때 맞춘다.
- 운영 기본값 변경은 없다(이미 3).

## 증거

- `storage/evaluations/gufo-upstream-port-final-20261023/final-f18-a01`, `final-f18-b01`(cache-disk 126.6 / 59.8 GiB 삭제, `CACHE-REMOVED.json`)
- 계획: `test/experiments/gufo_upstream_port_20261023/PLAN-F18.json`, `final.py` set `f18`
