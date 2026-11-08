# RQ 인계: 감독 세션용 (2026-10-29)

새 세션은 이 문서를 먼저 읽고, 작업 흐름(W)마다 에이전트를 하나씩 붙여 총괄한다. 설계의 근거와 세부 사항은 [PLAN.md](PLAN.md)에 있고, 에이전트 공통 규칙은 [R-BRIEF.md](R-BRIEF.md)에 있다. 이 문서는 무엇을 누가 어떤 순서로 하고 감독이 무엇을 확인하는지만 정리한다.

## 최종 목표

1. **27B 혼합 양자화 완성.** Qwen3.8 27B dense에 입력 쪽 Hadamard 회전과 GGUF 혼합 양자화를 적용한다. 컨버터, 로더, imatrix, 실험을 끝내고, 품질과 속도의 이득이 확인된 레시피를 운영 기본 파일로 정한다(R5). 효율 목표(PLAN "효율 목표")에 따라 바이트 절감이 실제 tok/s로 이어지는지, 실제 작업 품질이 유지되는지(D4)까지 확인한다.
2. **Flash-Next 확장.** 27B에서 검증한 공통 부분(`tools/rotquant/`, `src/core/quant/rotation.*`, `src/core/hip/detail/hadamard.hpp`)을 그대로 쓴다. Flash-Next의 dense 트렁크를 8비트에서 회전 + 4비트로 내려 decode 격차를 줄인다(gufo-org/gufo#474). 전문가 가중치의 저비트화는 그다음이다.

사용자 원칙: "27B를 완전히 구축, 실험, 확장 순으로". 각 단계 끝에 사용자 확인을 받는다(PLAN "단계").

## 현재 상태 (브랜치 `ai-tools/port-u4f`, HEAD `56692c88`)

| 항목 | 상태 | 근거 |
|---|---|---|
| 계획 | 리뷰와 보강 완료 | `PLAN.md` |
| R0 기준선 | 완료 (2026-10-29) | [R0-RESULTS.md](R0-RESULTS.md), [RESULTS.md](RESULTS.md). 품질(32K 토큰, BF16 대비 prefill/verify/decode), AR 속도(ABBA), DFlash2 텍스트 수락률(64문항). GSQ와 d32K DFlash2는 미측정 |
| R1b 컨버터(실모델) | 완료 | [R1B-RESULTS.md](R1B-RESULTS.md). 회전 BF16 GGUF와 256B 정렬(D7) |
| R2 로더와 실행 | 완료 (2026-10-30) | [R2-RESULTS.md](R2-RESULTS.md). ①③④⑤ 통과. ② 로더 정확성은 llama.cpp 직접 대조(RQ2I)와 층별 진단(RQ2G)으로 확인. 남은 KL은 BF16 파일 한계와 부호 민감 경계 위치. 형식은 prism.hadamard(D8). 정렬 A/B: 차이 없음, 로더 옵션 A 불필요 |
| R0b 타입별 실효 대역폭 | 완료 | [R0B-RESULTS.md](R0B-RESULTS.md). 4비트는 IQ4_XS가 가장 좋다(27B 토큰당 64.0 ms, 상한의 89%). Q3_K와 IQ3_S는 연산에 묶여 상한의 54%로 Q4_K보다 느리다. FWHT는 AR 토큰당 2%다 |
| R1a 컨버터(합성 텐서) | 완료 | [R1A-RESULTS.md](R1A-RESULTS.md). `tools/rotquant/`, `tools/qwen27b/rotquant_groups.py`. 테스트 20/20, 병합 뒤 `05e527d2`에서도 통과. `.rot`은 F16 |
| BF16 원본 | 받음 | `~/projects/ai.tools/storage/models/llm/qwen-3.8-27b-bf16-gguf/`(리비전 `4ca72078`, sha256 검증, `PROVENANCE.md`) |
| upstream 병합 | 완료 | `05e527d2`(upstream main `c92b19c5`). CLAUDE.md도 upstream 것으로 바뀌어 timing 기준이 **5 %·10 ms**다. PLAN의 "5% 또는 3 ms" 문구는 이 기준으로 읽는다 |
| 운영 | 전환 완료 | 다른 세션이 `1d7b729e`(ROCm 10.1)로 운영을 전환했다. ai.tools 플랫폼은 지금 정지 상태다 |
| Flash-Next 모델 파일 | 외장 디스크로 받는 중 | `/run/media/m2on/T5 EVO/AI/model/originals-flash-next-20261029/`(rev `766911a6`). 계획은 [FLASH-NEXT.md](FLASH-NEXT.md) |

커밋 `d19126df`에는 R0b와 R1a 파일이 섞여 있다. 결과 문서에 기록해 두었고, 히스토리는 고쳐 쓰지 않는다(사용자 결정).

## 작업 흐름과 에이전트

| W | 범위 | 선행 | 자원 | 산출물 |
|---|---|---|---|---|
| W1 컨버터 실모델 (R1b) | 27B BF16을 회전 BF16으로 변환. llama.cpp 기준 빌드(`.#llama-cpp-reference` 또는 eval 이미지 안 빌드). `llama-quantize` 확인 11항목(R1A-RESULTS 목록: `.rot` 복사, 이름 휴리스틱, llama.cpp 로드 거부 등). CPU 참고로 `W'·(H·D·x) ≈ W·x`를 실제 텐서 표본에서 확인. CTest 등록 결정(JSON 레시피 또는 점검 환경에 numpy/yaml 추가, 감독이 사용자에게 묻는다). E7a용 2비트 dequant 경로(llama.cpp gguf-py 또는 ggml) 확인 | 없음 | CPU 무거움, 디스크 약 55 GB | 회전 BF16 GGUF, `R1B-RESULTS.md` |
| W2 품질 하네스와 기준선 (R0) | 약 32K 토큰 평가 코퍼스(영문·한글 산문, 코드, 대화, 해시 기록). BF16 top-k logits와 logsumexp 저장. paired NLL, top-k 절단 KL, top-1, 토큰 단위 paired bootstrap 95% CI. Q8_0, Q6_K, UD-Q4_K_XL, Q4_K_M, GSQ-RCO-IQ3_S 측정. 같은 날 pp2048/tg128(d0, d32K)과 DFlash2 수락률. 효율 점수판 `RESULTS.md` 시작. D4 한국어 프롬프트 세트 초안 | 없음 | GPU 창 | 하네스 도구(`tools/qwen27b/` 아래), `R0-RESULTS.md`, `RESULTS.md` |
| W3 로더와 실행 (R2) | PLAN "R2" 절 전체. `src/core/quant/rotation.*`, `src/core/hip/detail/hadamard.hpp`, 27B 바인딩, Q8_1 생산 커널에 회전 융합(RMSNorm, SSM gate, SwiGLU, attention 출력, 그 밖의 `LaunchQuantizeActivationQ8_1*`), DFlash2 헤드 경로, BF16 경로 지원, `--cpu` 등 미지원 경로 거부. 병합 뒤 코드 위치를 다시 찾는다. 통과 기준 ①–⑤ | 코드는 바로 시작. ②③은 W1 회전 BF16과 W2 하네스 필요 | 빌드(CPU), GPU 창 | C++/HIP 커밋, `R2-RESULTS.md` |
| W4 imatrix (R3) | 회전 기저와 원래 기저 imatrix 수집기(gufo forward 훅, llama.cpp imatrix 형식). 보정 코퍼스는 W2 평가 코퍼스와 분리하고 해시를 기록. CPU 참고와 비교 | W3(훅 위치), W2(코퍼스 분리) | GPU 창 | 수집기, imatrix 파일, `R3-RESULTS.md` |
| W5 레시피 실험 (R4: E1–E8) | E1(같은 보정의 UD 레시피 ± 회전), E2(dense Q4_K ± 회전, `ssm_out` Q8_0/Q4_K), E3(3.0–3.5 bpw, IQ4_XS 우선), E4(imatrix 유무), E5(수락률, 서빙 속도), E6(1단 대 2단), E7a(2비트 fake quant 품질), E8(대리 지표 → 구간 KL → 탐욕 배분). 결과는 모두 `RESULTS.md`에 쌓는다. 실험 GGUF는 측정 뒤 지운다 | W1, W2, W3. E1/E4/E7/E8은 W4도 필요 | GPU 창, 디스크 | 레시피, 결과 |
| W6 커널 (E9, E7b) | E9: Q3_K/IQ3_S 병목 원인 분석(isa_mix, rocprof) → AR 상한 80% 이상 목표. 원인 분석은 지금 시작해도 되지만, 커널을 고치는 일은 E3 품질 결과를 본 뒤에 한다(PLAN 순서). E7b: E7a가 유망할 때만, 행 사이 디코드를 공유하는 2비트 small-batch 커널 포함 | E9 원인 분석은 없음. 구현은 E3. E7b는 E7a | GPU 창 | 커널 커밋, 결과 |
| W7 Flash-Next 확장 | **준비(지금 가능, 읽기 전용):** Flash-Next 현재 dense 로더 지원 타입(`src/models/qwen38_flash_next/weights.cpp`, `kernels/rocm/device_model.cpp`는 dense에 F32와 Q8_0만 받는 것으로 보인다. 확인 필요), 입력 그룹 표 초안(dense, hyper-connection, gated residual, MTP 헤드, n-gram 테이블, 3차원 전문가 텐서와 `.rot` 공유, `down_exps` K=640), 활성화 생성 커널 목록, 모델 파일과 디스크 계획, zengjiajun0623/gufo `gufo-opt/flash-next` 포크의 4비트 dense 커널 검토. **구현:** 27B R5 뒤에 사용자 확인을 받아 시작. `tools/qwen-flash/rotquant_groups.py`, 4비트 dense 커널, 회전 융합, #474와 같은 에이전트 벤치로 측정 | 준비는 없음. 구현은 27B R5 | 준비는 CPU 가벼움 | `plan/archive/rotated-quant/FLASH-NEXT.md`(확장 계획) |

### 병렬 진행 권장

- **바로 시작:** W1, W2, W3(코드), W7(준비). 필요하면 W6 E9 원인 분석도 시작할 수 있다.
- **GPU 창은 한 번에 하나뿐이다.** W2, W3 검증, W6 측정이 창을 두고 경쟁하므로, 감독이 창 순서를 정한다.
- **CPU가 무거운 일(W1 변환, llama.cpp와 gufo 빌드)은 GPU 창과 겹치지 않게 둔다.** 이쪽 창이든 다른 세션 창이든 같다.
- 에이전트끼리 같은 파일을 만지지 않도록 W마다 파일 경계를 브리프에 적는다. W3만 운영 코드(`src/`)를 바꾼다. W6 E9는 운영 커널을 바꾸므로 W3과 겹치면 감독이 순서를 정한다.

## 감독 역할

1. **브리프:** W마다 `plan/archive/rotated-quant/<W>-BRIEF.md`를 쓴다(R-BRIEF.md 공통 규칙 + 범위, 파일 경계, 통과 기준, 산출물). 에이전트에게는 브리프 경로를 주고 읽게 한다.
2. **검증:** 에이전트 보고를 그대로 받지 않는다. 테스트를 다시 돌리고, 원시 결과(RESULT 행, WINDOW.json, 해시)를 표본으로 확인한다. 확인한 내용은 결과 문서에 "감독 확인" 절로 붙인다.
3. **GPU와 CPU 조율:** 다른 세션(`gufo-00`)과 장비를 나눠 쓴다. 아래 "장비 규칙"을 지킨다. 에이전트는 GPU 창을 시작하기 전에 감독의 허락을 받는다.
4. **계획 갱신:** 단계가 끝날 때마다 PLAN.md 상태 열과 결과 반영 절을 고친다. 결과가 계획의 전제를 바꾸면(R0b 때처럼) 사용자에게 먼저 묻는다.
5. **사용자 확인 지점:** 아래 "결정이 필요한 것"이 생길 때, 그리고 각 단계가 끝날 때.
6. **커밋:** 자기 단계 파일만 `git add`한다. 한 줄 Conventional Commit에 트레일러를 붙인다. push, PR, 이슈는 하지 않는다. 다른 세션의 변경과 untracked 파일(`.dockerignore`, `.gufo-revision`, `plan/archive/gufo-upstream-port/*`)은 건드리지 않는다. 에이전트가 동시에 커밋하면 섞이므로(d19126df 사례), 커밋은 감독이 순서를 정하거나 에이전트에게 `git add <경로>` 뒤 바로 커밋하게 한다.

## 장비 규칙

- 호스트에는 Nix, numpy, jj, `strings`가 없다. Python과 hipcc는 eval 이미지 `ai-tools/gufo-eval:7c2c5a728270`(hipcc ROCm 7.2.4, cmake, `/opt/venv/bin/python3` numpy 2.3.5) 컨테이너에서 돌린다. 사용법은 R-BRIEF.md에 있다. 운영이 ROCm 10.1로 바뀌었으므로, gufo 측정 빌드의 툴체인은 W2를 시작할 때 다른 세션의 F23 빌드 방식을 확인해 맞춘다.
- GPU는 ai.tools 창 실행기로만 쓴다. 창은 한 번에 하나, 실패해도 자동 재시도하지 않는다. 절차는 `plan/archive/gufo-upstream-port/K1-BRIEF.md`의 "공통 규칙"에 있다.
- 다른 세션 `gufo-00`과의 약속:
  - GPU 창, 큰 CPU 작업(빌드, 해시, 변환), 큰 디스크 쓰기(다운로드, GGUF 생성)를 시작하기 전에 SendMessage로 알린다.
  - GPU 창이나 CPU 측정을 할 때는 `/home/m2on/projects/ai.tools/storage/runtime/experiments/vllm-multi-selected-20261016/ABBA-OWNER.lock`을 `flock -n`으로 잡는다. 이 lock은 `window.py`가 쓰는 것과 같다.
  - 상대가 측정 중이면 CPU 작업과 디스크 쓰기를 멈춘다. 이어받기 가능한 다운로드는 멈췄다가 다시 받는다.
  - 상대가 긴 세트(예: 80분 ABBA)를 돌릴 때는 창 사이 빈틈에도 lock을 잡지 않는다.
- `window.py`는 루트 여유 공간이 150 GiB보다 적으면 창을 시작하지 않고, 창이 끝나면 receipt/cache-disk를 지운다. 지금 여유 공간은 약 434 GB다. 회전 BF16(55 GB) + 실험 GGUF + E7a 파일(각 55 GB)을 쌓지 말고, 측정이 끝나면 지운다.
- GPU 측정은 ai.tools 플랫폼이 꺼져 있을 때만 한다. 켜져 있으면 가용 메모리가 약 39 GiB로 줄고 대역폭도 섞인다.

## 결정이 필요한 것 (감독 → 사용자)

| 항목 | 시점 | 선택지 |
|---|---|---|
| CTest 등록 방식 | W1 | JSON 레시피로 바꿔 yaml 의존을 없앨지, 점검 환경에 numpy/yaml을 추가할지 |
| D4 채택 수치 기준 | W2 결과 뒤 | KL, NLL, top-1 한도와 suite별 판정 방식 |
| E7a 측정 경로 | W5 E7 전 | fake quant 회전 BF16 방식(제안) 또는 다른 방식 |
| Q4_0/Q4_1 커널 추가 | E6 결과 뒤 | 1단 4비트를 IQ4_NL 말고도 쓸지 |
| E9 구현 | E3 결과 뒤 | 3비트 품질이 UD-Q4_K_XL에 가까우면 진행 |
| E7b 구현 | E7a 결과 뒤 | 2비트가 유망하면 진행 |
| R5 채택 레시피 | R4 결과 뒤 | 효율 점수판의 파레토 후보 중에서 |
| Flash-Next 착수 | R5 뒤 | W7 준비 문서를 보고 범위를 정한다 |

## 참고

- 조사 배경: halogen-flash-server(HGN 포맷, `jtsylve/hgn-spec`), gufo-org/gufo #474(격차의 원인이 dense 비트 수와 verify 비용), #256, #239, #855. 요약은 PLAN "배경"과 "R5 이후"에 있다.
- 외부 비교 파일: UD-Q4_K_XL, Q4_K_M, Q8_0, GSQ-RCO-IQ3_S가 모두 `~/projects/ai.tools/storage/models/llm/` 아래에 있다. DFlash2 드래프터는 `qwen-3.8-27b-dflash2-gguf/`에 있다.
