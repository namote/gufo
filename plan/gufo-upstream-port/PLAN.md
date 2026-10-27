# gufo 원본 이식·제안 계획 (2026-10-23)

> 위치: 2026-10-25 사용자 지시("ai tools에 계획이나 agent md gufo 에 관련된거 있으면 옮기라고")로 ai.tools `plan/`에서 이 gufo 저장소(`~/projects/gufo/plan/`)로 옮겼다. 본문의 `test/`·`storage/`·`tmp/`·`infra/`·`docs/` 경로는 ai.tools(`~/projects/ai.tools`) 기준이다. gufo 소스는 `~/projects/gufo`(이전 `~/.cache/ai-tools/gufo/s4`)다.

**운영 중단(2026-10-25 사용자: "운영은 중단.. gufo 완성전까지")**: 운영 플랫폼(qwen gufo `210c7633`, Manager, API)은 gufo 완성 전까지 멈춘다. 운영 전환 제안은 완성 뒤에 한다.

상태: **U3·U5 완료, 최종점검 F 완료(정확도·bench 통과, 재시작 TTFT·메모리는 운영보다 나쁨 → 운영 전환 제안 보류, [F 결과](F-RESULTS.md)), U5b 완료(재시작·다음 턴 TTFT 운영 수준, 메모리 +8 GiB anon 원인 확인, [U5b 결과](U5B-RESULTS.md)), U4 완료(RAM 복원 TTFT 개선, 텍스트·T2 동일, anon은 예산을 채워 그대로, [결과](U4-RESULTS.md)), U4b 완료(tail checkpoint, 첫 팔로워 7.9 → 4.7 s, [결과](U4B-RESULTS.md)), U4c 완료(system prompt 경계 checkpoint, 첫 팔로워 4.7 → 2.3 s 운영과 같음, [결과](U4C-RESULTS.md)), 디스크 복원 편차 원인 확인(SSD 쓰기 뒤 읽기 지연, 코드 아님, [결과](DISKREAD-RESULTS.md)), anon 바닥 내역 확인(DFlash2 pinned 1.56 GiB + backing 미리 커밋 ≤2 GiB + 약 0.5 GiB, [결과](ANON-FLOOR-RESULTS.md)), U4d 완료(RAM 예산 2 GiB, DFlash2 버퍼 공유로 anon −1.17 GiB, 텍스트·T2 동일, TTFT 손해 없음, [결과](U4D-RESULTS.md)), U4e 완료(PR 전 코드 리뷰 12건 수정, 텍스트·T2·TTFT·anon U4d와 같음, [리뷰·결과](U4E-REVIEW.md)), U4f 완료(원본 main 82c62b40 병합, 두 번째 리뷰 수정, 텍스트·T2·anon 같음, 이식 회귀 cache-depth·disk-spacing 수정, 남은 suite 실패는 원본 main과 같음, [결과](U4F-RESULTS.md))**(2026-10-23, [U0](U0-RESULTS.md)·[U1](U1-RESULTS.md)·[U2](U2-RESULTS.md)·[U3](U3-RESULTS.md) 결과). D5는 256행으로 결정, 풀·테이블 [API 검토](API-REVIEW.md) 완료. D6은 절충안으로 결정. U4는 측정(RAM 복원 약 50 GB/s, 256K 추정 0.35 s) 뒤 사용자 결정으로 보류하고 U5를 먼저 한다("U5 먼저, U4 보류"). D1은 추천안(우리 이미지, 호스트 Nix 없음)으로 진행했고 D2–D4·D7–D9는 미정. 사용자 지시: "일단 코드 받아서 머지 해보는게 좋지 않을까" → 머지 시도 결과 검토 → "응 정리하자". 설계만 요청받았으므로 이 문서는 구현·GPU 실행·외부 공개를 하지 않는다.

U0 원본 기준값은 [U0 결과](U0-RESULTS.md)에, 기능별 상세 대응은 [대응표](INVENTORY.md)에, Flash-Next·Gemma 4·DeepSeek V4 Flash로의 paged KV 확장은 [모델 확장](MODELS.md)에, KV 양자화 후보는 [KVarN](KVARN.md)에, 단계별 테스트는 [테스트 설계](TESTS.md)에 있다.

## 목적

1. 우리 gufo 변경 중 원본에 없는 이득을 원본(gufo-org/gufo) 최신 구조 위로 옮긴다.
2. 옮긴 것을 원본에 제안해, 원본과 계속 벌어지는 유지 부담을 줄인다.
3. paged KV를 Qwen 27B에서 끝내지 않고 Qwen Flash-Next, Gemma 4, DeepSeek V4 Flash로 넓힌다(2026-10-23 사용자 지시). 그래서 풀·테이블·예약은 처음부터 모델과 무관한 공통 코드로 설계한다.
4. 그동안 운영(qwen Tool, gufo `210c7633`, Manager 리비전 23)은 바꾸지 않는다. 이식본으로의 운영 전환은 아래 gate를 통과한 뒤의 별도 결정이다.

## 배경 (2026-10-23 확인)

- **머지 시도:** 원본 main `79c2dcc5`를 운영 빌드 `210c7633`에 합쳐 보았다. 20개 파일에서 47곳이 충돌했다. 머지는 되돌렸고, 충돌 목록은 `tmp/test-results/gufo-upstream-merge-20261023/`에 있다. gufo 작업 트리에 빈 브랜치 `ai-tools/upstream-0.21-merge`(`210c7633`과 같음)가 남아 있다.
- **충돌보다 큰 문제:** 원본 #540(0.20.0, 2026-10-21)이 Qwen 27B를 원본 캐시 재설계(`docs/cache-redesign/RFC.md`, 카드 15·16)의 component cache로 옮기고, 우리 기능이 올라가 있던 Qwen snapshot 경로를 지웠다. 그래서 충돌을 푸는 것으로는 끝나지 않고, 기능을 새 구조 위로 다시 구현해야 한다.
- **원본 새 구조:** 상태를 셋으로 나눈다.
  - 실행 슬롯: 가변 상태. 지금은 슬롯마다 최대 문맥 KV를 미리 잡는다.
  - 불변 chunk: 256행 단위. 같은 계산 계보를 물려받은 checkpoint끼리만 공유하고, RAM이나 디스크에 둔다.
  - checkpoint: 경계 하나의 고유 상태 + chunk 참조. 디스크 전송은 1 MiB 고정 버퍼로 한다.
- **원본에 이미 있는 것:**
  - 리더·팔로워 대기(같은 앞부분 요청은 먼저 온 요청이 끝내기를 기다림)
  - 디스크 prefetch(`WILLNEED` 힌트)
  - 행을 복사 없이 빌려 쓰는 checkpoint
- **원본에 없는 것:** paged live KV. RFC가 측정으로 필요가 보이면 앞당긴다고 적은 후속 과제다(`RFC.md:97`, `:1246-1249`, `:1546-1549`). 우리 P1·P2 측정(256K×4가 40 GiB, 64K×8이 24 GiB)이 그 근거가 된다.
- **이전 계획과의 관계:** 우리 [S1 component 이전 계획](/home/m2on/projects/ai.tools/archive/plan/2026-10-23-gufo-s1-component/V11-GUFO-S1-PLAN-20261020.md)(보류)은 원본 #540이 같은 일을 해서 취소·보관한다.

## 판정 요약

|구분|기능|
|---|---|
|이식·제안 핵심|P1 블록 테이블 커널, P2 공유 풀(필요할 때 할당, 블록 공유·CoW, 입장 예약)|
|이식 후보(독립 PR)|Q2 공유 pass scratch, Q3 기준 128|
|원본 위에서 다시 측정하고 정함|슬롯 간 블록 매핑 복원(신규), Q4 끝 checkpoint, R1 병렬 읽기, B3 축출 시 디스크 쓰기, P3·Q1 내용 hash 공유, P4 system 경계|
|버림(원본이 해결했거나 의미 없어짐)|B1–B2 bounce 스트리밍, S4 예산 하한|

## 단계

모든 코드는 외부 소스 gufo 작업 트리(`~/projects/gufo`)의 새 브랜치에서 고친다. 실행·비교 묶음은 ai.tools `test/experiments/gufo_upstream_port_*/`에 둔다. 단계마다 사용자 확인 뒤 진행한다.

통과 기준의 T0~T4s는 [테스트 설계](TESTS.md)의 계층이다.
- 수정 하나마다 바뀐 부분 검사만 20분 안에 돈다.
- 단계 끝에 짧은 검사 전부와 16K 축소 장문(T4s)을 1시간 안에 돈다.
- 실제 길이 전부(256K, 256K×4, 64K×8, 원본 suite 전체)는 최종점검 F로 묶어, 운영 전환 제안·PR 직전과 요청할 때만 돈다. F는 2~3시간 걸려도 줄이지 않는다.

|단계|내용|통과 기준|
|---|---|---|
|U0|원본 `79c2dcc5`에서 브랜치 `ai-tools/port-u0`를 만든다. 툴체인을 정하고(결정 D1) 빌드한다. 운영 Qwen 구성(256K, DFlash2, think off, 이미지)으로 T2 경우의 기준 출력과 T3 기준값을 한 번 기록한다|**완료**: 빌드 성공, T2 21요청·T4s 7창 전부 정답·cold/AR 텍스트 동일, bench 기록([결과](U0-RESULTS.md))|
|U1|Q2 공유 pass scratch를 옮긴다. 가장 작고 독립적이라 원본 PR 절차를 처음 밟아 보기에 맞다|**완료**(gufo `14bcd820`, 스위치 없이 기본): 출력 21/21 동일, 기동 여유 16K×4 +6.4 GiB·64K×8 +14.7 GiB, bench·TTFT 5 % 안([결과](U1-RESULTS.md))|
|U2|P1 블록 테이블 커널을 옮긴다. 풀·테이블 API는 [모델 확장](MODELS.md)의 설계 검토를 먼저 통과해야 한다. 세션마다 자기 풀을 전부 미리 받는 방식이라 동작은 원본과 같다. 어댑터의 `BorrowRowsView`·`TransferRows`가 블록 테이블을 거치게 한다(결정 D5)|**완료**(gufo `7daf08d4`): plane 일반 풀·256토큰 블록, 200K+ 위치 bit 동일, T2·T4s 텍스트 전부 동일, bench −0.5 % 안([결과](U2-RESULTS.md))|
|U3|P2: 슬롯 executor가 공유 풀에서 블록을 필요할 때 받는다. 입장 때 블록을 예약하고, 모자라면 유휴 슬롯의 행을 돌려받은 뒤 기다린다|**완료**(gufo `15f27e88`): 출력 전부 동일. 기본 크기에서 TTFT ±0.6 %, bench +1 % 안. 256K×4·40 GiB 입장, 풀 부족 시 대기, 블록 반환 확인. 64K×8을 24 GiB 풀로 +8 GiB([결과](U3-RESULTS.md)). checkpoint 블록 참조·CoW는 U4로 옮김. 운영 전환 제안 직전에 최종점검 F|
|U4|**완료**(gufo `ai-tools/port-u4` `17ef4326`, 2026-10-24 사용자 결정 "확인하고 U4 진행", [설계](U4-DESIGN.md)·[결과](U4-RESULTS.md)): checkpoint chunk가 RAM 사본 대신 공유 풀 블록 참조를 갖고, 쓰기 전 CoW, 다른 슬롯 복원은 블록 매핑. 텍스트·gold 동일, T2 21/21, GPU 테스트 PASS. RAM 다음 턴 0.50 → 0.28 s, regress·boundary 합 −1.7·−2.3 s. anon은 같은 예산의 U5b와 같다(예산을 채움, 8 GiB 22.5–24.9, 2 GiB 19.5). 256K 재시작은 이식본 디스크 읽기 편차(1.8–7 s) 범위 안. 운영 수준 메모리는 예산 1–2 GiB 결정이 필요하고, 그때 잃는 첫 팔로워는 U4b에서 확인|T0·T2(복원 경우). 복원 시간·복사량 감소 측정. 출력 동일|
|U4b|**완료**(gufo `ai-tools/port-u4b` `9a5cacee`, 2026-10-24 사용자 결정 "B계획없애고 그냥 1번을 고치는거로 계획을 변경하자", [설계](U4B-DESIGN.md)·[결과](U4B-RESULTS.md)): 긴 prompt 끝 앞쪽 1024토큰마다 RAM checkpoint K개(`--cache-tail-checkpoints`, 운영과 같은 기본 3)를 원본 `PlanCaptures` 선택 후보로 둔다. 텍스트·gold 동일, T2 21/21. shared 첫 팔로워 7.86 → 4.65 s(2 GiB 14.34 → 4.66 s), 다른 창 손해 없음. 운영(2.29 s)이 빠른 진짜 이유는 tail이 아니라 system prompt 끝 checkpoint였다(설계 정정). 처음 U4b였던 행 바이트 hash 저장 중복 제거는 취소했다(남는 비용: SSD 중복 쓰기)|T0 경계·복원. f6·f7 창|
|U4c|**완료**(gufo `ai-tools/port-u4c` `852b6c72`, 같은 사용자 결정으로 1번 마무리, [설계](U4C-DESIGN.md)·[결과](U4C-RESULTS.md)): 운영 `system_prefix_tokens` 경계(system 메시지 끝, 128토큰 이상)에서 `Request::ShareCheckpoint`(continuation 우선순위 RAM + 디스크). 텍스트·gold 동일, T2 21/21. shared 첫 팔로워 4.65 → 2.30 s(2 GiB 2.29 s), 운영 2.30 s와 같다. 다른 창 손해 없음. 남은 차이(둘째 턴·256K 재시작)는 이식본 디스크 복원 읽기 편차다|T0 계획·복원. U4b와 같은 창에서 텍스트 동일, shared 첫 팔로워 운영 수준(약 2.3 s)|
|U4d|**완료**(2026-10-24 사용자 결정 "응~ 그대로 진행": RAM 예산 2 GiB, DFlash2 버퍼 공유, TRIM한 SSD에서 재측정, [설계](U4D-DESIGN.md)·[결과](U4D-RESULTS.md)): gufo `ai-tools/port-u4d` `d42b292c`. 슬롯들이 DFlash2 prompt feature pinned 페이지 하나(400 MiB)를 나눠 쓴다(원본 #688은 슬롯마다 400 MiB). `final-f10-a01`: 텍스트·gold·T2 21/21 동일, 준비 직후 anon 19.47 → 18.30 GiB(운영 15.63), TTFT 손해 없음. 운영 대비 TTFT는 같거나 빠르고, 256K cold prefill만 약 2 % 느리다|텍스트·gold·T2 21/21 동일, 준비 직후 anon −1.17 GiB, TTFT 손해 없음|
|U4e|**완료**(2026-10-25 사용자 지시 "코드 리뷰한번하자. pr 던지기 전에" → "오케이 수정하고 테스트 하자", [리뷰·결과](U4E-REVIEW.md)): gufo `ai-tools/port-u4e` `f07aa9f1`. 유휴 write-back 실패 시 live KV 유지·큐에 들어간 것만 기록, checkpoint는 만든 실행만 채택, 이어지는 세션 입장에 가진 블록 반영, row 회수 순서, system prompt 최소 128, copy-on-write 여유는 공유 블록에만, 공유 prompt 버퍼 소멸 drain, 문서. `final-f11-a01`은 9번 수정이 기본 풀 용량을 넘겨 GPU 단위 테스트 1개 실패 → 정정 뒤 `final-f12-a01` 15창 COMPLETE: 텍스트·gold·T2 21/21 동일, 준비 직후 anon 18.30 GiB, TTFT U4d와 같음|텍스트·gold·T2 21/21 동일, anon·TTFT U4d와 같음|
|U4f|**완료**(2026-10-26 "오케이 그냥 유지해"; 2026-10-25 사용자 지시 "응 우선 한번 머지한번히고" → "코드한번더 검토해보자~" → "오케이~ 고쳐", [결과](U4F-RESULTS.md)): gufo `ai-tools/port-u4f` `663d50c9`. 원본 main `82c62b40` 병합(`12fbc7ad`, 충돌 3파일), 두 번째 리뷰와 전달받은 지적 수정(`8d50a674`, `923a4194`, `1a017a3f`). `final-f13-a01` 병합본 텍스트·T2 동일, 256K 첫 턴 875 s(운영 887). `final-f15-a01`(`923a4194`) 텍스트·T2·anon 동일, TTFT는 재실행 편차 안이나 RAM capture 거절 증가(원인 확인 남음). 최적화 4건 건별 커밋 뒤 원래 의도인 첫 렌더링 system 경계 반환·경계 probe 제거를 별도 정정 커밋. 앞선 GPU 집중 회귀 3종·68 HTTP 응답 동일, 순서 고정 C4 경고 0; 정정본 CPU 7종·GPU HTTP/vision 2종 PASS, ABBA 32응답 동일(시간은 참고값). 이후 hoist 0 guard→3경로 비교→조건부 probe 채택, 36응답 동일·prefill 1250→64(기존 probe와 같음); 첫 요청 비용/경고는 [별도 기록](HOIST-PREFIX-REVIEW.md). 이어서 이 세션이 개선사항 인수([마무리](U4G-TURNKEY.md)): 미매핑 블록 해제 정리·write-back 시간 로그·복원 실패 로그, `final-f16-a01`(`d66d5ae7`) 텍스트·T2 21/21 동일, anon 18.30 GiB, TTFT 병합본 수준. RAM 거절 원인 확인(장치 행 write-back 읽기 대기). 후속("오케이 진행해줘"): 동시 fork 간헐 실패는 원본 `ShareCheckpoint` 경계 조건 → 수정 `b4eec7d4`(GPU 10/10), 장치 행 write-back 미리 읽기 `663d50c9`(shared RAM 거절 5–7 → 2–3, 텍스트·TTFT 같음). 결정 반영(2026-10-25): `--kv-pool-bytes` 기본 = 세션 × 전체 문맥(`0341dd88`), 정리 브랜치 `ai-tools/port-tidy` `aeffca08`, [PR 계획](PR-PLAN.md), [이슈 초안](ISSUE-DRAFT-PAGED-KV.md), [꼬리 checkpoint 존치 검토](TAIL-CHECKPOINTS.md)(추천: 존치·기본 3, document 둘째 질문 94 → 4.4 s). 마무리 점검(2026-10-26): 이식 회귀 수정 `75cffde3`·`b76614a6`·`82e88bad`·`ad15b000`, 원본 suite 테스트 문구 `5e982d9c`·`c4f1b9a9`, conversation(Q4_K_M)·tool-reasoning 호출+echo 2경우는 원본 main과 같은 실패로 유지, 테스트 선택·재사용 도구 추가([정리](U4F-RESULTS.md#마무리-최종점검과-원본-suite-정리-2026-10-26))|텍스트·gold·T2 21/21 동일, anon·TTFT 병합본과 같음|
|U5|"다시 측정" 항목을 하나씩 U3 위에서 원본 방식과 비교한다. 이득이 있는 것만 옮긴다. 순서(예상 TTFT 이득 순): ① 밀려난 대화가 1,024 grid까지만 복원되는 문제(B3·Q4) ② 팔로워 공유 깊이 7,174(P4·Q3) ③ 디스크 복원 1.1 GB/s(R1) ④ 입장 예약 보수성·게시 경합·B3 디스크·P3/Q1. **완료**(gufo `3e36130a`): ① 재배정 전 frontier capture 채택(C-restore TTFT 0.81 → 0.34 s), ② 문제 아님, ③ 디스크 복원 helper가 실제로 읽기 채택(16K 재시작 복원 856 → 271 ms), ④ 기록([결과](U5-RESULTS.md))|해당 T2 경우·T3 측정점만. 항목마다 이득·비용 수치와 채택 여부 기록|
|U5b|F 뒤 재시작 경로 보완(2026-10-24 사용자 확인 "오케이 가자"): B3 idle write-back을 원본 구조 위로, 종료·재배정 시 write-back, 거절될 경계에서 prefill 멈춤 생략, 메모리 anon·file 분리 측정. **완료**(gufo `ai-tools/port-u5b` `3942939b`): 256K 재시작 4.99 / 5.12 s, 다음 턴 0.94 / 1.27 s, 텍스트·gold·T2 21/21 동일. 메모리 차이는 RAM checkpoint 예산을 행 사본으로 채우는 것(+8 GiB anon). 남은 결정: RAM 예산 또는 U4, 작은 checkpoint SSD 복원([결과](U5B-RESULTS.md))|F와 같은 재시작 창(F2) 운영 빌드와 번갈아. 텍스트 동일·gold 전부|
|U6|원본에 제안한다. 먼저 이슈로 방향을 묻고(결정 D3), 반응을 본 뒤 U1·U2·U3·U4 순서로 작은 PR을 낸다. **준비(2026-10-25 사용자 결정 "결정 사항 반영해서 진행해줘. 공개(push·PR·이슈)는 아직 하지 마.")**: 정리 브랜치 gufo `ai-tools/port-tidy` `aeffca08`(원본 main 위 33커밋, 트리 = `port-u4f` `0341dd88`), [PR 분할 계획](PR-PLAN.md)(13단계 모두 충돌 없음·CPU·GPU 빌드 통과), [이슈 초안](ISSUE-DRAFT-PAGED-KV.md). 아직 올리지 않음|원본 리뷰 반영. 머지 여부는 원본 결정|
|U7–U10|paged KV를 Flash-Next(원본), Gemma 4(franzmoca 포크), DeepSeek V4 Flash(원본)로 넓히고 각각 제안한다. 상세는 [모델 확장](MODELS.md)|모델마다 T0~T3, 단계 끝 T4s|
|U11|**K3 완료(2026-10-27, gufo `be582a9f`, [결과](KVARN.md#k3-결과-2026-10-27-gufo-be582a9f-실행-k3-a01-감독-검증))**: 요청 95개 두 형식 gold·k4v2 텍스트 FP16과 95/95 같음, suite 통과, DFlash2 tg @32K +22 %·@256K +77 %, 256K 복원 3.5 → 1.0 s. k4v2 기본값 유지, 남은 비용은 d0 prefill 약 1 %·깊은 문맥 짧은 prefill. **진행(2026-10-26 사용자: "KVarN 적용")**: 운영 전환 결정 뒤로 두었던 순서를 앞당겨 K0부터 시작한다. K0 설계(엔진 안 fake quant + teacher forcing, 설정 7개, 32K·256K gold)는 [KVarN](KVARN.md#k0-설계-2026-10-26)에 있다. GPU 창은 실행 전 확인. **K0 완료**(gufo `b3edc894`, [결과](KVARN.md#k0-결과-2026-10-26-gufo-b3edc894)): 32K·256K 모두 모든 설정이 top-1 214/214·텍스트 동일·gold 정답, KVarN k4v2 KL은 RTN의 1/2–1/2.6. AR decode attention은 대역폭에 묶여 32K마다 10.2 ms, 256K 외삽 6.3 → 10.6 t/s 상한. 다음: fused dequant decode 시제품(K1 진입 조건)과 어려운 품질 과제. 처음 추가(2026-10-23 사용자 지시): KVarN KV 양자화(Hadamard + 이중 scale 정규화, K4/V2·128토큰 그룹, sink·tail FP16). hybrid의 full-attention 층만, 명시 옵션, 자동 fallback 없음. K0 오프라인 정확도 → K1 HIP 커널 → K2 서빙·checkpoint 형식 → K3 실제 길이. 상세는 [KVarN](KVARN.md)|K0 오차표로 진행 여부 결정. 이후 gold 전부 정답·FP16 대비 차이 기록·메모리·속도 표|
|U12|**진행(2026-10-27 사용자: "응~")**: 작은 M weight GEMM. K-quant 9–64행(DFlash2 2–8명 검증)이 8·16행씩 나눠 가중치를 2–4번 읽는 빈틈을 한 번 읽기로. 비트 일치 유지. [안내](SKINNY-BRIEF.md)|shape × 행 수 µs·GB/s 표, C1 그대로, C2·C4·C8 TG|

운영 전환은 U3(필요하면 U4) 뒤에 운영 회귀를 통과하면 사용자에게 제안한다. 그때까지 운영 빌드는 `210c7633`이다.

## 원본 기여 규칙 (원본 `AGENTS.md`)

- 한 줄짜리 Conventional Commits를 쓰고, PR 제목은 릴리스 정책을 따르며 squash merge한다.
- C++는 `tools/ci/check-format.py`(clang-format)를 통과해야 한다.
- Nix로 고정한 툴체인을 쓰고, 비교할 때 컴파일러·의존성 버전을 유지한다.
- 기능 테스트는 `tests/functional/`에 둔다. 연속 대화(continuation) 변경이면 `long-context`·`cache` suite를 포함한다.
- 시간 기준은 요청·단계별로 5 %와 3 ms이고, 요청을 평균 내지 않는다. main과 PR을 번갈아 돌리고 결과를 모두 남긴다.
- "성공한 최적화는 별도 스위치 없이 기본값이 된다." 우리 `--kv-pool-bytes`처럼 켜야 동작하는 옵션은 기본 동작으로 바꾸는 설계가 필요하다. 예를 들어 풀 크기를 세션 수·문맥·가용 메모리로 자동으로 정하는 방식이다(결정 D6).

## 정할 것 (사용자 결정)

|번호|내용|추천|
|---|---|---|
|D1|빌드 툴체인. 원본은 Nix(GCC 15.3)이고 우리는 ROCm 7.2.4 이미지(GCC 13.3)로 검증했다. 호스트에 Nix를 설치하는 것은 시스템 변경이다|U0에서 우리 이미지로 개발·회귀하고, 원본 PR 직전에만 Nix 컨테이너에서 원본 규칙대로 다시 잰다. 호스트 설치는 하지 않는다|
|D2|공개 계정과 이메일. 지금 커밋 작성자가 회사 이메일이고, 이 장비에는 `gh`가 없다|사용자가 정한다. 정해질 때까지 공개 작업(이슈·PR·push)은 하지 않는다. **진행(2026-10-25 사용자: "포크하나해서 해")**: GitHub `namote` 계정으로 원본을 포크(`namote/gufo`, 공개)하고 `ai-tools/port-tidy` `aeffca08`, `ai-tools/port-u4f` `0341dd88`을 push했다(작성자 `Moonki Kang <namote@gmail.com>`). gufo 원격 이름은 `fork`. PR·이슈는 아직 열지 않음. **보류(2026-10-25 사용자: "머지말고 일단 분리한상태로 가자")**: 원본 병합 요청(PR·이슈)은 하지 않고 포크에서 독자 유지한다. PR 계획·이슈 초안은 문서로만 둔다|
|D3|제안 방식|이슈 먼저. 측정 수치와 함께 paged live KV가 RFC 후속 과제에 맞는지 묻는다. **결정(2026-10-25)**: 혼합 방식. 우리 브랜치는 독자 유지(운영 기준), 독립 수정(①ShareCheckpoint·②관측 로그·③-1 공유 scratch)만 원본에 PR. ①ShareCheckpoint는 원본에 내지 않고 우리 포크에서 유지(2026-10-25 사용자: "ShareCheckpoint 경계 수정은 우리가 그냥 진행하자"), paged KV는 이슈로 방향부터. 초안만 작성|
|D4|P3·Q1 내용 hash 공유. 원본은 "같은 토큰만으로는 공유하지 않는다"는 원칙을 뒀다. 운영 포크의 키는 토큰 hash다(바이트 비교 없음)|원본 계보 공유를 유지한다. 저장 중복 제거는 하지 않는다(2026-10-24 사용자 결정, SSD 중복 쓰기는 받아들임). 첫 팔로워 지연은 계보와 관계없이 tail checkpoint(U4b)와 system prompt 경계(U4c)로 고친다|
|D5|블록 크기. 우리는 64행, 원본 chunk는 256행이다|**결정(2026-10-23, 추천안)**: 256토큰 블록을 먼저 측정한다. bench 손실이나 메모리 낭비가 크면 64행 블록 + 여러 조각 borrow view로 간다. [API 검토](API-REVIEW.md): DeepSeek ratio-128 층 때문에도 256이 필요하다|
|D6|풀 크기 기본값|**결정(2026-10-23, 사용자: "오케이~ 추천안으로 가자")**: 절충안. 원본 PR의 기본 동작(값을 주지 않을 때)은 자동 크기(세션 × 평균 기대 문맥 + 여유)다. 값을 명시하면 그 값이 그대로 풀 크기가 되고 자동 계산은 쓰지 않는다. 우리 운영 구성은 지금처럼 값을 명시한다(2026-10-20 명시 예산 지시 유지). U3 구현: 값이 없으면 세션 수 × 전체 문맥(세션이 원래 쥐던 메모리, 원본이 받던 요청을 거절하지 않음). 기대 문맥을 줄인 자동 산식은 측정 근거와 함께 U6 이슈에서 원본과 정한다. **확정(2026-10-25 사용자)**: 값이 없으면 세션 수 × 전체 문맥, 명시하면 그 값. paged KV는 기본으로 켜져 있고 끄는 스위치는 없다. 고정 숫자(40 GiB)는 기본값에 넣지 않는다. gufo `0341dd88`에서 문서와 GPU 테스트(기본·명시 크기, 너무 작은 값 거절)를 맞췄다|
|D7|Gemma 4 제안 대상. Gemma 4는 원본 main에 없고, 포크 franzmoca/gufo `feat/gemma`가 관리한다(원본 PR #426은 열린 채 "not on upstream's roadmap")|포크에 제안한다. 공통 API는 원본에 먼저 넣고 포크가 합친 뒤 Gemma 적용|
|D8|모델 다운로드. Flash-Next(크기 미확인)와 DeepSeek V4 Flash(가중치 약 81 GiB)가 `storage/models/`에 없다|해당 단계 시작 때 크기·디스크 여유를 확인하고 승인받아 받는다|
|D9|DeepSeek 진행 방식. 원본 카드 17(DeepSeek component 어댑터)이 아직 시작 전이다|용량 측정 뒤 정한다. 이득이 크면 카드 17을 paged 전제로 직접 구현해 제안하는 쪽을 검토한다|

## 위험

- **원본이 빠르게 바뀐다.** 0.18에서 0.21까지 사흘이 걸렸다. 단계마다 원본 최신으로 다시 맞추는 비용이 든다. 단계를 작게 유지한다.
- **원본 Qwen 27B component cache가 아직 검증 중이다.** 카드 15는 속도 판정이 inconclusive이고, 카드 16(DFlash2)은 proposed이며, 원본 `todo.md`에 HTTP·deep·자원 검증과 vision 수치 수정이 남아 있다. U0 기준값에서 운영 구성(DFlash2 + 이미지)이 원본에서 지금 잘 도는지부터 확인한다.
- **커널 변경 범위:** prefill attention과 검증 커널의 KV 주소 방식은 아직 확인하지 않았다(대응표 "확인하지 못한 것").
- **GPU 시간:** 수정 하나에 20분, 단계 끝에 1시간 안으로 잡았다. 최종점검 F(2~3시간)는 운영 전환 제안·PR 직전과 요청할 때만 돈다.
- **축소판의 한계:** 16K 축소판은 절대 길이에서만 생기는 문제를 직접 보지 못한다. 그래서 큰 위치의 주소 계산은 T1 커널 검사로, 256K 블록 수 계산은 T0으로 따로 막는다.
- **운영 분리:** 이식 작업은 운영 빌드·compose·Manager를 건드리지 않는다. 운영 전환은 별도 결정이다.

## 검증 원칙

- 단계마다 greedy 출력 토큰 ID가 한 번 기록한 기준과 같아야 한다([테스트 설계](TESTS.md)).
- bench 측정점은 5 % 안이어야 한다. 시간 비교는 같은 축소 조건의 기준값과만 한다.
- host 최대 메모리와 SSD 쓰기량을 기록한다.
- 실패한 단계는 다른 경로로 자동 재시도하지 않는다. 실패 원인과 수치를 그대로 남긴다.
- 증거는 `storage/evaluations/gufo-upstream-port-*/`에 둔다.
