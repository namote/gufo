# gufo 원본 이식 U4f: 원본 main 병합과 두 번째 리뷰 (2026-10-25)

상위 문서: [이식 계획](PLAN.md), [U4e 리뷰·결과](U4E-REVIEW.md).

사용자 지시: "응 우선 한번 머지한번히고" → "코드한번더 검토해보자~" → "오케이~ 고쳐". 그 뒤 사용자가 다른 리뷰의 지적(P1 2건, P2 1건 → P2 1건 → P2 4건 → 2건)을 전달했다.

## 병합

- gufo `ai-tools/port-u4f` `12fbc7ad`: `port-u4e` `f07aa9f1`에 원본 main `82c62b40`(79c2dcc5 이후 29커밋, 0.22·0.23 릴리스, #705 DFlash2 가속, Windows platform 계층)을 합쳤다.
- 충돌 3파일을 풀었다.
  - `CMakeLists.txt`: 우리 `kv_block_pool.cpp`와 원본의 `gufo_platform` 링크를 둘 다 넣었다.
  - `src/cache/disk.cpp`: 우리 prefetch 읽기를 `platform::ReadAt` 위로 옮겼다.
  - `attention_wmma.hip`: launch에 block table과 `first_query`를 둘 다 넘긴다.
- 원본 테스트 1곳의 호출을 block table 시그니처에 맞췄다. 원본 대비 diff는 같은 66파일이다.
- `final-f13-a01`: 15창 모두 COMPLETE, GPU 단위 테스트 7개 PASS(WMMA prefill ops 추가).
  - 텍스트·gold·T2 21/21이 같다. 준비 직후 anon은 18.30 GiB다.
  - 256K 첫 턴은 875 s로 운영 887 s보다 빠르다. #705의 prefill 마지막 층 생략 덕분이다.

## 리뷰와 수정

|커밋|내용|
|---|---|
|`8d50a674`|copy-on-write 예약은 쓰기 시작 위치가 공유 블록을 가를 때만 1블록, 입장 판단은 같은 규칙(최대 문맥 +1 없음). 배치 pass 예외 시 coordinator 스트림 drain. 스트리밍 경로 system prompt 경계를 토큰 move 전에 계산. write-back은 제출 전에 '썼음' 표시, 실패하면 되돌림. 유휴 write-back은 자기 실행의 대기 checkpoint만 기다림. 장치 보존 회수에서 덮이지 않은 frontier는 `live_frontier_dropped`로 기록. capture 로그 `reason`·`cause=retention`. prompt 버퍼 소멸 drain 실패 처리. 테스트 `mkdtemp` → `TemporaryDirectory`, `kv_block_layout_test` PR 목록, `.hip` 서식(우리 줄만), Linux 전용 주석, 문서(paged KV 기본 켜짐, system prompt checkpoint, 재배정 capture, 읽기 warm-up)|
|`923a4194`|흩어진 KV를 디스크에 chunk별로 빌려 쓰기(RAM staging 없음). `RetainRows`·`UnshareKvRows` 실패 경로의 블록 참조 반환. 유휴 write-back deadline을 device probe와 분리. reader가 앞서면 미리읽기가 그 뒤부터 이어감(원본에도 있는 문제)|
|`1a017a3f`|복사 제출 일부 실패 시 미완료 복사가 쓸 수 있는 블록은 drain 뒤에만 반환. 자기 대기 저장 때문에 건너뛴 유휴 write-back은 재시도(`WriteBackDeferred`)|

- 새 테스트: 종료 시 backoff 무시(옛 코드에서 실패), 큐에 들어간 뒤 실패하면 다시 씀, 흩어진 행을 복사 없이 저장·재시작 복원, GPU 공유 앞부분 매핑 세션의 예약 수, block table + `first_query` 커널 케이스.
- CPU 138/138이다. `1a017a3f`는 CPU·GPU 빌드와 CPU 테스트만 했고 GPU 창은 돌리지 않았다.

## GPU 창

- `final-f14-a01`(`8d50a674`): GPU 단위 테스트 8개 PASS. 시나리오 도중 코드가 바뀌어 에이전트가 멈췄다.
- `final-f15-a01`(`923a4194`): 15창 COMPLETE, 단위 8개 PASS, 텍스트·T2 21/21 같음, anon 18.30 GiB.
  - TTFT 합은 regress 26.3 s, shared 49.0 s로 f13(21.3, 45.3)보다 느렸다.
- 같은 조건에서 regress·shared만 다시 쟀다(run마다 직전 fstrim).

|빌드|regress 합(20)|shared 합(19)|RAM capture 용량 거절(regress / shared)|
|---|---|---|---|
|병합 `12fbc7ad` f13-a01|21.3|45.3|2 / 5|
|병합 `12fbc7ad` f13-b01|22.4|52.6|2 / 7|
|`8d50a674` f14-b01|22.7|51.7|—|
|`923a4194` f15-a01|26.3|49.0|7 / 11|
|`923a4194` f15-b01|23.6|49.2|6 / 13|

- TTFT 차이는 같은 빌드 재실행 편차(병합본 45.3 → 52.6) 안이다. 디스크 복원 `filesystem_ms`가 병합본에서도 30 ms와 430–550 ms를 오간다(SSD 상태, [디스크 복원 결과](DISKREAD-RESULTS.md)).
- 그러나 `923a4194`는 RAM capture 용량 거절이 꾸준히 많고, shared에서 write-back이 19 → 17회로 줄며 재배정 capture 거절이 생긴다. 흩어진 KV를 chunk별로 빌리면 디스크 쓰기 작업이 슬롯 행을 관찰한 채 남고, 그 사이 슬롯이 바뀌면 관찰 행을 RAM에 보존하며 예산을 쓰는 것으로 추정한다(미확인).
- 증거: `storage/evaluations/gufo-upstream-port-final-20261023/final-f13-a01`, `final-f13-b01`, `final-f14-a01`, `final-f14-b01`, `final-f15-a01`, `final-f15-b01`. cache-disk는 지우고 `CACHE-REMOVED.json`에 적었다.

## 다음

- 최적화 4건은 아래와 같이 건별 커밋·집중 검증을 마쳤다. 코드 리뷰와 PR 분할에서 각 커밋을 독립 검토한다. 시스템 경계는 첫 렌더러 API를 바꾸는 대신 불필요한 history 복사를 제거했다.
- `923a4194`의 RAM 거절 증가 원인 확인과 최신 HEAD의 전체 F는 [마무리](U4G-TURNKEY.md)에서 했다. 원본 functional suite는 별도다.
- 커밋 정리·PR 분할·`--cache-tail-checkpoints` 존치·공개(D2)는 미정이다.

## 성능 후보 수정 (2026-10-25, 제안한 쪽 담당)

사용자 요청: "검토하고 니가 수정하고 수정건 하나씩 커밋 부탁해". 기존 gufo 작업 트리에서 건별 커밋하고 외부 공개·운영 전환은 하지 않는다.

- 기준: gufo `1a017a3f`, upstream `82c62b40`. gfx1151 단일 GPU, 기존 ROCm 7.2.4/GCC 13.3 이미지와 동일 모델/정밀도/입력/동시성으로 검증한다. 원본 Nix GCC 15.3 환경 검증과 다른 모델은 미검증이다.
- 현재 실행 중인 gufo/GPU 컨테이너가 없는 상태에서 수정·빌드를 시작했다. GPU 실행은 기존 단독 owner lock·NPU/quiet·메모리·deadline·반환 guard를 사용한다. 자동 fallback은 추가하지 않는다.

|수정|연산 계약·적용 범위|검증 gate·현재 상태|
|---|---|---|
|시스템 경계 probe|모델명과 무관한 ChatRequest의 leading system/developer 메시지만 복사, 나머지 metadata/프레이밍 유지|`ffc9ca7f`, CPU 전체 text_model_runner_test와 metadata/이미지 경계 회귀 PASS. 1 MiB history mock 준비 시간은 CPU 복사 비용만 측정, 모델 E2E 개선으로 해석하지 않음|
|RAM 회계|RamBytes()==0인 장치 chunk의 중복 비교 생략, mixed component의 prefix 의미 유지|`288161ea`, host/device/mixed 정확 회계·공유 구멍·byte-pressure 축출의 기준/수정본 CPU 회귀 PASS|
|hidden 보존|공유 pass scratch의 마지막 hidden 행만 세션 row0로 보존, offset0, tap 전환도 보존|`22544492`, CPU 모의 15shape+복사 실패 및 GPU full logits/features·AR·verification·tap 전환 PASS. 짧은 pass의 hidden 5120 기준 복사량 1.25 MiB→20 KiB; 할당 크기 감소나 TG 개선을 뜻하지 않음|
|retained KV 복원|같은 pool의 완성 chunk는 source block 직접 매핑, host chunk/tail만 새 블록 확보, 불완전 table의 실행/게시 금지|`667e91e7`, 여유 1블록으로 source 5블록+host tail 복원, bytes/logits/token·취소·해제 GPU 회귀 PASS. 불완전 table의 move/reset guard도 보존. 고정 pool 크기는 같고 필요한 추가 여유 블록을 줄임|

- C++ 전체 공통 format 검사를 실행했다. 기준 `1a017a3f`에도 있는 `prefill_quant_gemm.hpp:139`의 기존 서식 차이 1곳이 남는다. 이 작업의 변경 파일은 clang-format 21 검사 통과이며, 무관한 커널 파일은 수정하지 않았다.
- 원본 바이너리·SHA와 검증 출력은 `tmp/test-results/gufo-perf-fix-20261025-e93ik6x2/`에 보존한다. 이 경로는 임시 증거이며 선택 기본값이나 영구 실행 입력으로 승격하지 않는다.

### 집중 검증 결과

- GPU 창 `perf-review-units-a01`: continuation adapter 230.9 s, draft adapter 6.0 s, inference backend 186.5 s, 전부 PASS. AR/FP32·FP16/shared scratch/retained KV·이미지·취소·HTTP 동시성·정리 검증을 포함한다. source/binary SHA를 확인했고 창 반환 health PASS다. GPU 실행 중 source 파일을 수정하지 않았다.
- 최신 소스로 빌드한 CPU 4종(`cache_retention_test`, `component_text_cache_test`, `text_model_runner_test`, `text_generation_scheduler_test`) 모두 PASS. ai.tools infra 432 PASS. 변경 C++ 파일 포맷 PASS; 위의 기준 코드 서식 차이는 그대로다.
- production 기준/수정본을 같은 컨테이너 release preset으로 빌드했다. 입력 archive의 timestamp 때문에 재빌드가 생략되거나 일부 object가 섞인 초기 시도는 실패 증거로 보존하고, phase마다 실제 입력 mtime을 갱신한 최종 빌드만 비교에 썼다. 최종 source SHA와 `667e91e7`의 일치를 확인했다.

|pp2048/tg128, depth0, r=1 (tok/s)|기준 `1a017a3f`|수정 `667e91e7`|
|---|---:|---:|
|AR PP|665.16|661.89|
|AR TG|12.95|12.95|
|DFlash2 PP|617.22|616.71|
|DFlash2 TG|45.14|45.09|

- 위 표는 각 1표본의 속도 유지 control이며 유의한 속도 향상 근거가 아니다. 실제 양자화는 GGUF 표기 IQ4_XS mixed(15.33 GiB), 기존 target/draft 파일과 greedy seed 314159를 그대로 썼다.
- `perf-compare-a01`: RAM cache·C4 두 턴 ABBA, warm 포함 36응답 gold/텍스트 동일. per-request 5%/3ms 경고 11개는 그대로 남겼다. 동시 제출에서 요청별 순서가 바뀌고 일부 prompt 작업량이 123↔1032토큰으로 달라져 이 결과의 성능 판정은 **inconclusive**다. 합계/평균으로 통과시키지 않았다.
- 경고의 TTFT 경로만 `perf-ordered-a01`로 별도 확인했다. 대화를 먼저 C1으로 seed하고 이어쓰기를 50ms 간격의 고정 순서 C4로 제출, ABBA 4구간. 32응답(seed 포함) gold/텍스트 동일, 측정된 16요청은 모두 prompt33/predicted213으로 작업량 같음. 요청별 TTFT/PP/TG의 5%/3ms 경고 **0**. 원래 경고 자료와 기준은 바꾸지 않았다.
- 두 HTTP 비교 합계 68응답이 같다. RSS anon은 약 18.34 GiB로 같고 전체 RSS 절감은 주장하지 않는다. 이 범위에서 확정된 속도 회귀는 찾지 못했으나 전체 속도 개선·256K/C8/다른 모델·Nix 툴체인 검증 완료는 아니다.
- CPU fixture에서 device chunk SameChunk 비교는 115875→0(mixed 84651), 시스템 prefix의 1 MiB history 준비 mock은 113.148→0.05651 µs/call이다. 실제 renderer/모델 E2E 수치로 해석하지 않는다.
- 증거: `tmp/test-results/gufo-perf-fix-20261025-e93ik6x2/EVIDENCE.json`, `tmp/test-results/gufo-upstream-port-20261023/perf-{review-units,compare,ordered}-a01/`. 각 GPU 창은 quiet/owner/deadline/반환 health PASS, 자동 fallback 0이다. 바이너리는 비교용 임시 artifact이며 선택 기본값을 바꾸지 않았다.

- 실행 당시 driver·LOCKS 원문은 비교 artifact에 SHA 검증해 보존했다. 최종 runner에 server shutdown 실패 검사를 추가하고 저장된 8회 shutdown이 모두 exit0/강제 kill 없음인 것을 확인했다(GPU workload 재실행 없음).

## System 경계 구현 정정 (2026-10-25)

- 사용자 원래 의도는 첫 렌더링에서 system 경계를 함께 반환해 경계용 두 번째 렌더링을 제거하는 것이다. `ffc9ca7f`는 probe의 대화 복사만 줄였으므로 이 의도를 완료하지 못했다.
- Qwen/Flash-Next는 compiled renderer가 실제 system 턴 끝의 byte 경계를 반환하고, vision/text 공통 준비가 special-token 경계에서 토큰 수를 함께 센다. DeepSeek도 native 첫 렌더링의 안전 경계를 준비 결과에 전달한다. 일반 `SystemPrefixTokens` probe와 두 chat 호출부의 재렌더링을 제거한다. DeepSeek의 기존 mutable-continuation 경계 검사는 별도 계약으로 유지한다.
- 경계 정의는 빈 user probe와의 최대 공통 prefix에서 실제 system 턴 끝으로 바뀐다(기존보다 user framing 몇 토큰 앞). 전체 prompt tokens·문자열·이미지 위치/identity는 불변이어야 한다. 새로운 속도 배수나 전체 모델 검증 완료를 주장하지 않는다.
- 소스 수정은 gufo `17163c2f` 한 커밋이다. `SystemPrefixTokens`와 기존 probe 전용 테스트를 제거하고, renderer/vision 준비의 byte·token 경계·전체 ID 보존 및 chat/start_chat의 실제 system-only 캐시 재사용·bypass 동일성 테스트로 바꿨다.
- CPU 7종(template/vision/cache/runner/scheduler, 기존 rendered goldens 포함) PASS. 최초 추가 PNG fixture의 복사 오타로 template 테스트 1개가 실패했으며 원본 fixture 바이트와 동일하게 고쳐 해당 테스트를 다시 통과시켰다. 최초 실패 로그도 보존했다. infra는 최종 상태에서 432 PASS이다.
- `system-boundary-units-a01` COMPLETE/health PASS: inference_backend_gpu_test 189.3s(새 first-render system 경계 테스트 포함), qwen_vision_serving_test 112.4s(AR/speculation·sampling·concurrency·disk 이미지 복원) PASS. 소스/바이너리 SHA와 소유·quiet·유한 시간·자원 반환 gate를 지켰다.
- 변경 C++ 16파일 포맷 PASS. 전체 포맷 검사는 기존 미변경 `prefill_quant_gemm.hpp:139`에서만 실패한다.
- DeepSeek는 기존 goldens·native tokenizer 합성 byte/BPE 64경우 split/full ID 동일성까지 CPU 검증했다. 실제 DeepSeek 모델 vocabulary/GPU 및 Flash-Next 실제 모델 실행, 256K/전체 F는 이번 검증 범위가 아니다. 서비스 기본값·모델 선택·PR 공개 변경 없음.

- 동일 production 툴체인으로 `667e91e7`(SHA `92525e86…`)과 정정본 `17163c2f`(SHA `2fe1c852…`)을 빌드했다. `system-boundary-ordered-a02` ABBA(C1 seed 뒤 고정 순서 C4) 32응답 모두 gold·텍스트 동일, 측정 C4 16요청의 5%/3ms 경고 0, 별도 seed TTFT 경고도 0이다. 첫 팔로워의 prefill은 경계가 user framing 앞이므로 1032→1035토큰으로 예상대로 바뀌었다. 한 표본이고 ABBA 일부에 17.55s CPU infra 검사가 겹쳐 **시간은 참고값이며 전체 속도 향상/엄격한 성능 인증을 주장하지 않는다.**
- `system-boundary-ordered-a01`은 실행기의 `binaries` 입력 마운트 누락으로 첫 서버/모델 실행 전에 FileNotFoundError로 종료했고 health PASS였다. 실패 원본을 보존하고 window 입력 목록에 명시한 binaries를 포함하도록 수정한 뒤 수동으로 새 a02 창을 실행했다. 자동 재시도·다른 경로 fallback은 없다. a02도 quiet·health PASS로 반환했다.
- 상세 증거: `tmp/test-results/gufo-system-boundary-20261025-p8pacja0/EVIDENCE.json`, production source/binary manifest, CPU/포맷 로그, 창별 LOCKS 원문과 실행기 원문. GPU 결과는 `tmp/test-results/gufo-upstream-port-20261023/system-boundary-{units-a01,ordered-a01,ordered-a02}/`. Git 커밋 전 working diff의 SHA 목록과 커밋 후 실제 파일 SHA가 동일하다.

## Hoist 시 leading 공유 회귀 후속

합쳐진 system 턴 끝을 선제 경계로 삼은 `17163c2f`는 late instructions가 다른 대화 사이의 leading 공유 성능을 잃을 수 있다. hoisted 테스트도 그 잘못된 목표를 고정했다. 즉시 0 guard는 **손해 방지, 회귀 미해결**로 기록하고, 입력에 따른 probe 선택 후보를 세 빌드의 실제 복원 위치/prefill로 검증한다. [담당 검증 계획·결과](HOIST-PREFIX-REVIEW.md)를 따른다.

`8cd5c24d`로 0 guard를 먼저 커밋한 뒤 세 빌드를 비교했다. hoisted 둘째 요청의 old/zero/probe 복원은 1186/0/1186, prefill은 64/1250/64로 두 mirror 모두 같았다. 사용자 gate 충족 후에만 조건부 probe를 `c0a68ddc`로 채택했다. 36응답·EOS 일치, CPU7/GPU2/infra432 PASS. 첫/root 요청은 선제 checkpoint 작업으로 zero보다0.16–0.17s 느리고 관련 시간 경고6건을 보존한다(기존 probe 대비 경고0). [상세 결과·한계](HOIST-PREFIX-REVIEW.md)를 참조한다.

## 마무리 최종점검과 원본 suite 정리 (2026-10-26)

사용자 지시: "1번 진행해줘" → "테스트 하나가 다른테스트를 커버하지 않나? ... 정리해보자" → "구지 필요없는 테스트들이 너무 커져서.. 테스트 최적화가 필요해ㅑ" → "메인에서 실패하는 테스트는 뭔지보고 테스트를 수정하던가.. 삭제하던가.. 기능확인하고 기능을 수정하던가 하자" → "오케이 그냥 유지해".

### 테스트 축소

- ai.tools `gufo_upstream_port_20261023/`에 `COVERAGE.json`(경로 → suite·tier·창·unit 영역, 미매핑 경로는 멈춤), `choose.py`(diff → 선택), `reuse.py`(같은 이미지·같은 harness의 COMPLETE 창과 같은 바이너리 SHA unit 재사용), `make_plan.py`(PLAN 생성·LOCKS 재생성)를 추가했다. 시간 control(ABBA, `--controls`)은 시간 영향이 있는 변경에만 돌린다.
- F19는 HEAD unit 9/9와 long32k/long256k/mem만 새로 돌리고 나머지는 F16–F18 같은 이미지 증거를 재사용했다. 중간에 멈춘 실행에서 `fb-long256k`/`f19-long256k`의 FINAL.json 메모리 행이 빠졌다(원본 창 결과는 남음).

### 원본 functional suite 실패 정리

|suite|원본 main `82c62b40`|원인|처리|
|---|---|---|---|
|cache-depth|통과|이식 회귀: 행이 모자랄 때 분기 복원 checkpoint·prefix checkpoint·live frontier 회수 조건|`75cffde3`, `82e88bad`, `ad15b000`. F22 single-session cache-depth/cache-rotation/long-context 통과|
|disk-spacing|통과|이식 회귀: learned 경계가 다음 checkpoint의 디스크 간격을 차지|`b76614a6`. F20 통과|
|reasoning-budget|실패|우리 실행 설정 `--think off`|F21부터 서버 기본 thinking으로 실행, 통과|
|conversation|실패|Q4_K_M 정밀도. Q8_0에서 통과|테스트 유지, 기록만|
|tool-reasoning|실패|가중치에 따라 모델이 echo 지시를 따르지 않음|호출을 기대하지 않는 7경우에 "Do not call any tool" 추가(`5e982d9c`, `c4f1b9a9`). 호출+echo 경우(`closer_before_call`, `framing_between_calls`)는 그대로 둠|

- `framing_between_calls` trace(`final-f22-trace-a01`, `--trace`): Q8_0은 먼저 쓰라고 한 `</invoke>` 줄을 건너뛰고 바로 native 호출을 썼다. Q4_K_M은 system 프롬프트에 적힌 클라이언트 envelope 형식으로 완전한 호출을 썼고, 파서는 선언된 도구 이름의 envelope를 framing으로 지웠다(응답 텍스트 빈 문자열, 호출 없음). 원본 `openai_chat.cpp`의 의도된 설계("Client envelope syntax is removed only when it names a declared tool")이므로 바꾸지 않았다.
- F21 standard-full·cache-pressure 통과, units PASS. 남은 실패는 conversation(Q4_K_M)과 tool-reasoning 호출+echo 2경우이며 둘 다 원본 main에서도 실패한다.
- **결정(2026-10-26 사용자: "오케이 그냥 유지해")**: 원본 기준 가중치(UD-Q4_K_XL) 다운로드와 Q8_0 suite 전환은 하지 않는다. suite와 시나리오 모두 운영 가중치 Q4_K_M으로 유지하고, 위 두 실패는 원본 main과 같은 알려진 실패로 기록한다. envelope 파서 설계와 해당 테스트도 그대로 둔다.
- 증거: `tmp/test-results/gufo-upstream-port-20261023/final-f19*`, `final-f20*`, `final-f21*`, `final-f22*`. 각 창 quiet/owner/health PASS, 자동 fallback 없음.
