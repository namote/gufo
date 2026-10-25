# Hoisted instructions의 공유 경계 검증 (2026-10-25)

상위: [U4f](U4F-RESULTS.md). 사용자 합의에 따라 0 guard와 실제 복구안을 구분한다.

## 선택 이력과 계약

- 1단계 선택(`8cd5c24d`): `has_system_prefix && hoisted.empty()`일 때만 첫 render의 system 끝을 경계로 반환. hoist된 요청은 0. **손해 방지, 회귀 미해결**이며 원래 leading instructions 끝까지의 선제 공유를 복구한 것으로 집계하지 않는다. 기존 일반 공통 prefix 탐색은 유지한다.
- 기준: `ffc9ca7f` 이전 `1a017a3f`의 실제 prepared tokens와 선행 메시지+빈 user probe tokens의 LCP.
- 2단계 채택(`c0a68ddc`): leading instructions와 뒤늦은 system/developer가 공존하는 입력에서만, 실행 전 입력 형태로 probe 경로를 선택. 일반 요청은 첫 rendering 경계를 사용한다. 선택한 probe 실패 후 다른 경로나 정밀도/backend로 재시도하지 않는다. 어느 경로를 골랐는지는 구조화 debug 로그로 기록한다.
- 채택 경로는 BPE 임의 바이트 경계를 분할하지 않고 실제 full prompt token IDs와 probe의 LCP를 쓴다. tools/reasoning/framing/이미지 처리 계약과 full tokens는 보존해야 한다. cache bypass 요청에서는 probe 작업을 하지 않는다.

## 검증 창과 gate

- gfx1151 한 장, 기존 ROCm7.2.4/GCC13.3 이미지 및 같은 production preset. 기존 Qwen3.8 27B GGUF(실제 IQ4_XS mixed)와 tokenizer/model manifest를 그대로 사용. dtype/kernel/backend 변경 없이 prompt 준비와 cache 작업만 비교한다. HIP 최적화 matrix의 현재 Gufo 작업 참조이며 다른 모델·LLM v10/v7 재개·선택 변경은 아니다.
- benchmark 실행 코드/fixtures/plan/LOCKS는 `test/experiments/` 아래, 출력·비교 바이너리는 `tmp/test-results/` 아래. 비교물은 운영/실험 기본값으로 선택하지 않는다.
- CPU: hoisted 경계0·non-hoisted 경계 유지·전체 토큰 ID 보존·literal special/tools/reasoning/이미지 goldens. 후보는 old probe LCP 동일성, cache bypass·선택 경로·예외 전파 검증.
- GPU: 기존 owner lock/fresh NPU·KFD quiet/MemAvailable≥80GiB, 단독 창≤1500s 및 100GiB cap. 창 중 GPU가 읽는 파일 수정·컴파일·CPU suite·대형 I/O는 하지 않는다. 실패 원본·반환 health 보존, 자동 fallback/자동 재시도 없음.
- 요청은 같은 긴 leading system과 서로 다른 late developer를 가진 두 대화. fresh server에 첫 대화 뒤 둘째 대화를 제출하고, 각 빌드의 둘째 요청 복원 위치(usage cached)와 `timings.prompt_n`을 비교. 같은 요청의 cache=false cold oracle 및 빌드 간 text/gold 동일성을 확인한다. non-hoisted 대조 요청도 별도로 검사한다.
- 채택 gate: probe 후보가 0 guard보다 실제 prefill을 줄이고 old probe와 같은 복원 위치를 얻으며 cold/built text가 같다. timing은 요청별 5%/3ms 경고와 작업량 차이를 원문으로 기록하며 합산으로 회귀를 숨기지 않는다. 미확정 timing/미검증 모델은 그대로 명시한다.

## 검증 결과와 채택

- 0 guard는 gufo `8cd5c24d`로 먼저 별도 커밋. CPU template/vision cache 2종 PASS. **손해 방지, 회귀 미해결** 상태를 고정한 뒤 복구 후보를 검증했다.
- 복구 후보 CPU 7종 PASS, 최종 infra 432 PASS. `hoist-units-a01` GPU HTTP/vision 회귀 2종 PASS(187.8s/112.4s), quiet/health PASS 반환. readonly 독립 검토에서도 새 blocker 없음. 변경 C++ 포맷은 통과했고 전체 포맷 검사는 기존 미변경 `prefill_quant_gemm.hpp:139`에서만 실패한다.
- 세 production 빌드: baseline `1a017a3f`/SHA `33f6c106…`, zero `8cd5c24d`/SHA `f7904cd1…`, 후보 `c0a68ddc`/SHA `8b215bf3…`. 0 조건을 독립된 두 build 단계에서 재빌드한 결과도 byte-identical이다. 같은 ROCm/GCC/source 기반이며 버전·정밀도·backend 재시도는 없다.
- `hoist-compare-a02`는 baseline→zero→probe→probe→zero→baseline 순서, 케이스마다 fresh server를 사용했다. 2 cases × 6구간 × 3요청 = 36응답 모두 정확한 gold/text/cold oracle/EOS 일치. 12개 서버가 exit0/강제 kill 없이 종료했고 창의 quiet·health도 PASS다. 측정 중 CPU suite/build/대형 I/O를 겹치지 않았다.

### 둘째 대화의 복원·prefill

두 mirror의 결과가 각각 같다. 앞쪽 system은 같고 late developer만 다른 대화에서:

|빌드|복원 위치(토큰)|prefill 토큰|
|---|---:|---:|
|기존 probe (`ffc9ca7f` 이전)|1186|64|
|0 guard|0|1250|
|조건부 probe 복구안|1186|64|

- 복구안이 old probe의 정확한 복원 위치/prefill을 회복하고 0보다 prefill을 줄였으므로 사용자 채택 gate를 충족했다. 검증 후에만 `c0a68ddc`로 커밋했다. 이는 Gufo 소스 채택이며 서비스 기본값/배포/PR 공개 변경은 아니다.
- non-hoisted 대조에서는 zero와 probe 모두 복원1188/prefill40으로 같다. old probe의 1191/37과의 3토큰 차이는 `17163c2f`의 실제 system 턴 끝 정의에 따른 기존 차이로 별도 기록하고, 이번 hoist 복구 성과로 집계하지 않았다.
- 후보 로그는 두 mirror에서 hoisted의 `hoisted_probe,hoisted_probe,none`, 일반 대화의 `rendered,rendered,none` 순서가 정확히 확인됐다. 마지막 none은 cache=false cold oracle이며 probe를 실행하지 않는다.

### 첫 요청 비용과 시간 한계

요청별 5%/3ms 기준을 바꾸지 않았다. 아래는 평균이 아닌 두 개별 구간 값이다.

|빌드|첫 요청 TTFT(s), 두 구간|둘째 요청 TTFT(s), 두 구간|
|---|---|---|
|기존 probe|2.1871 / 2.2041|0.1968 / 0.1980|
|0 guard|2.0284 / 2.0385|2.0141 / 2.0202|
|조건부 probe|2.1986 / 2.1995|0.1988 / 0.1984|

- 시간 경고 **6건**은 모두 zero 대비 probe의 첫/root 요청 TTFT·wall·prompt_ms에서, 두 mirror 모두 발생했다. 삭제하거나 통과로 재분류하지 않았으며 원본 runner의 `performance=inconclusive`도 유지한다.
- 추가 진단: old/probe는 첫 요청에서 각각 prefill 3구간·capture3회·copied payload 522187232 bytes, zero는 2구간·2회·350724416 bytes다. 선제 checkpoint 재도입으로 추가 capture와 prefill 분할이 생긴다. **첫 요청은 zero보다 약0.16–0.17s 늦다는 tradeoff를 유지한다.** 이전 probe 대비 후보의 같은 작업 비교에는 시간 경고가 없다. 추가 복사만으로 지연 전체를 설명한다고 단정하지 않는다.
- RssAnon은 약17.81GiB 수준(표본 간 수MiB 차이)이다. 메모리 절감·전체 TG 속도 향상·모든 요청/모델 성능 인증은 주장하지 않는다. 실제 Flash-Next/다른 vocabulary/256K/전체 F는 이번 범위 밖이다.

### 실패 원본과 재현 자료

- `hoist-compare-a01`은 wrapper가 case마다 `server_args`를 필수라고 가정해 preflight에서 KeyError로 중단됐다. 서버/모델 실행 전이며, 관측한 예외 원인·당시 실행기/LOCKS를 보존했다. 입력 경로 수집에서 선택적인 case-level args를 허용한 뒤 수동으로 새 a02 창을 실행했다. 자동 재시도나 다른 모델/backend 전환은 없다.
- 출력: `tmp/test-results/gufo-hoist-prefix-20261025-d12s4szf/EVIDENCE.json`, 양쪽 build manifests/CPU·포맷 로그/adopted diff/창별 LOCKS·실행기 원문. GPU 원문: `tmp/test-results/gufo-upstream-port-20261023/hoist-{units-a01,compare-a02}/`.
- `PLAN-HOIST-PREFIX.json`은 222행/23.7KiB다. 완결된 두 대화의 반복 instruction 원문을 세 빌드에 동일 전달·SHA 동결하는 입력 묶음이라 20KiB 기준의 작은 초과를 유지했다. 문장이나 요청 body를 바이트 수에 맞춰 나누지 않았다.
