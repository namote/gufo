# P: 운영 전환 준비 (브리프)

사용자 지시(2026-10-28): 추천 순서 3번 "운영 전환 준비" 진행("ㅇㅋ 진행합시다").
상위 문서: [이식 계획](PLAN.md), [F 결과](F-RESULTS.md), [F16](U4G-TURNKEY.md#전체-점검-f16-final-f16-a01-gufo-d66d5ae7), [KVarN K3](KVARN.md), [S](SKINNY.md), [T](T-RESULTS.md).

## 목표

현재 HEAD(`ai-tools/port-u4f`, paged KV + KVarN k4v2 기본값 + K5 + S + T 포함)가 운영 빌드 `210c7633`을 대신해도 되는지 운영 회귀로 판단하고, 근거와 함께 전환 제안서를 쓴다. **실제 전환은 하지 않는다.** 운영 compose·Manager·이미지는 건드리지 않는다(운영은 사용자 지시로 정지 상태).

## 범위

1. **후보 고정:** 시작할 때의 HEAD SHA를 후보로 적는다. 진행 중 gufo 코드 커밋은 하지 않는다(문서만). 후보 release 빌드와 sha256을 기록한다.
2. **CPU/단위:** F16과 같은 단위 테스트 묶음(vision serving 포함) + KVarN·skinny·ssm 테스트. 실패는 원본 main 또는 기존 기록과 같은 실패인지 확인해 적는다.
3. **운영 회귀(GPU):** F16 순서를 기본으로 한 새 set(`p1`)을 `final.py`에 추가한다.
   - 운영 구성 그대로(256K×4, 풀·RAM·디스크 예산, DFlash2, think off, mmproj, max_tokens). 후보는 기본 KV 형식(k4v2)으로 돈다.
   - 시나리오 long32k·long256k·two8k·regress·boundary·shared, 256K×4·64K×8 동시, T2 21요청, memprobe(준비 직후 anon), bench(R4 측정점).
   - 시나리오마다 운영 `210c7633`과 번갈아(ABBA). 실행 직전 fstrim.
   - functional suite `batch`·`state-edges`·`cache`·`long-context`를 후보에서 한 번.
4. **판정:**
   - 정확도: 텍스트 운영과 같음(k4v2라 다를 수 있으면 gold·K3 기준과 비교해 원인 기록), gold 전부 정답, T2 21/21, 서버 로그 ERROR·logic_error·length_error 없음.
   - 시간: 요청·phase별 5 %·3 ms. 평균 금지. 깃발은 한 번 조사하고 영향받는 history만 번갈아 재실행. 순서·디스크 상태 효과는 T1b처럼 순서 균형으로 가른다. 잡음은 inconclusive로 남긴다.
   - 메모리: anon·file 분리, 운영 대비 차이 표.
5. **산출:** `plan/gufo-upstream-port/P-RESULTS.md` — 표(운영 vs 후보), 깃발 조사, 남은 차이와 위험, **전환 제안**(전환 절차 초안: 이미지 빌드·플래그 차이·되돌리기 방법, 원본에 없는 플래그 정리 포함). PLAN.md 상태 줄은 감독 세션이 고친다.

## 규칙

- push·PR·이슈 금지. 단일 행 Conventional Commit, 트레일러:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>` /
  `Claude-Session: https://claude.ai/code/session_01HuCQEe5fZ9Avb8GWLR598A`. 자기 파일만 `git add`.
- GPU는 ai.tools 창 실행기로만, 한 번에 하나. 창 전에 `.gufo-revision`을 HEAD로, 창 끝에 `sleep 20`. 실패해도 자동 재시도 없이 기록.
- GPU 창 동안 CPU 무거운 빌드 금지(공유 LPDDR). 빌드는 창 사이에.
- 자동 fallback 없음. 데이터 보고 margin 넓히지 않음. 모든 증거 보존(창 디렉터리·로그 삭제 금지, 디스크 여유가 부족하면 끝난 창의 cache-disk만 지우고 기록).
- 운영 빌드 `210c7633` 바이너리·이미지는 F16에서 쓴 것을 그대로 재사용(새로 빌드하지 않음).
- 창 계획·실행 시간 예상을 세운 뒤 시작한다. 막히거나 결정이 필요하면 멈추고 보고.

## P2: 64K×8 짧은 요청 지연 원인 가르기 (사용자 결정 2026-10-28 "B")

배경: [P 결과](P-RESULTS.md). 64K×8 `long-and-short`에서 긴 prefill 중 도착한 짧은 요청 q0–q3가 운영보다 +0.3–5 s(순위 정렬) 늦다. 운영도 run마다 편차가 크다(q1 21–40 s). 256K×4 C4 첫 토큰 +1.05 s는 F부터 있던 차이라 로그 분석만 한다(GPU 창 추가 없음).

K3 이후 코드 변경은 `4a7046b2`(K5), `26e0a956`·`28e650b4`(S), `f53aba73`(T)뿐이다(S2는 되돌림). 이미지 `be582a9f9275`(K3)·`4a7046b2e8ed`(K5)·`7c2c5a728270`(후보)이 이미 있다.

1. **먼저 로그로:** p1·p1r의 64k8 server 로그에서 짧은 요청의 대기 구간(입장, prefill chunk 순서, capture, decode step 합류)을 운영과 비교해 가설을 세운다. 운영 편차의 원인(도착 순서·chunk 경계)도 같이 적는다.
2. **창 `p2`:** 64k8 시나리오만, K3 `be582a9f`와 후보 `7c2c5a72`를 ABBA(4창). 판정은 요청·순위별 5 %·3 ms, 편차가 커서 갈리지 않으면 inconclusive로 적는다. 
3. 후보가 K3보다 재현되게 느릴 때만 이분: 필요한 이미지(`28e650b4`, `f53aba73^` 등)를 창 **사이에** 빌드하고 영향 변경을 좁힌다. 창은 최소로, 각 단계 전 계획을 P-RESULTS에 적는다.
4. 원인이 확정되고 고칠 방법이 분명하면 수정 커밋 가능(테스트·format 체크 포함). 수정 뒤 64k8 ABBA(수정본 vs K3 또는 운영)와 해당 변경에 맞는 functional suite만 돈다. 정확도는 텍스트 동일·gold.
5. 결과는 `P-RESULTS.md`에 "P2" 절로 추가한다. 위 규칙 전부 그대로(창 실행기, 한 번에 하나, sleep 20, 창 중 빌드 금지, 자동 재시도 없음, 증거 보존, push 금지).
