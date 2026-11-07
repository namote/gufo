# Flash-Next 긴 calibration 자료 후보

**현재 상태 (2026-11-07):** c03 본체·보정 MTP·원본 BF16 비전 조립과 최종 검증이 완료됐다. 사용자 승인으로 c03를 상시 서비스 중이며 변환 감시는 종료했다. 아래는 날짜별 계획·실행 이력이다. [최종 결과](F3-MTP-CALIBRATION.md).

2026-11-05 00:47 KST. 사용자 재개 승인 후8K 소형 검사 통과, **c03 본체 변환 실행 중**.
시간별 점검·진단·수정·재개는 [감시 절차](F3-CONVERSION-WATCH.md)를 따른다.

8K/512 혼합 두 행을 embedding/HC/첫GDN/PLE/첫full-attention에 통과시켰다.
normal routed state를 전진시키고 두 대표 block에서 all-expert H를 캡처했다.
B1 packed 연산5개 모두 기존5% 한도 통과(최대4.828%); 전모델 품질 통과는 아니다.
pilot169.1초, peak GPU allocated15.829GiB, host MemAvailable 최저89.871GiB,
종료 KFD0. `long-row-pilot/{SUMMARY,STATE}.json`과 코드/data hash를 main gate로 고정했다.
본체는 별도100GiB container·host8GiB reserve 감시를 유지한다.

## 적용한 c03

- 2,309개 독립 샘플, **1,999,496 tokens**, 최대8,192. 기존 예산보다376tokens 적다.
- 기존 c02 512토큰 행을 원본 token IDs 그대로 소스 비율별로 추출해55.08% 유지했다.
  서로 다른 대화를 새로 이어 붙이거나 padding을 추가하지 않는다.
- 신규: SWE-smith 48개/260,843tokens, Open-SWE 41개/322,973tokens,
  한국어 Hermes 36개/123,717tokens, 한국어 wiki 33개/190,651tokens.
- 한국어 소스 기준32.26%다. 한국어 소스 안의 영어·코드까지 포함하므로 순수 한글 토큰 비율은 아니다.
- 코딩 자료는 성공 라벨, task 중복 제거, repo당 최대4개, 완결된 호출/결과 그룹 최소2개,
  system·도구 정의·첫 user 이후 작업 내용2,048tokens 이상이면서 전체의1/3 이상을 요구했다.
  전체 작업의 앞부분을 선택하므로 최종 해결 장면까지 포함된다고 주장하지 않는다.
- 반복 tool boilerplate 문제를 발견해 실제 전체 trajectory에서 사용한 도구의 정의만
  원문 그대로 남겼다. Open-SWE의 작업 내용 비중은 최초3.0%에서 최종35.9%로 늘었다.
  이것은 입력 구성 수치이며 모델 품질 개선 측정은 아니다.
- Open-SWE 원본에 tool-result ID가 없는 경우 원본 순서+개수를 검사했다. ID 연결이
  독립 확인된 것은 아니다. Hermes도 순서/개수를 검사하되 마지막 assistant 함수 호출은
  결과 없는 정상 target으로 별도 표시했다. 중간 결과 누락·orphan은 거부한다.
- 새 한국어 문서는 c02에 사용된 문서ID를 전부 제외했다. 공개 평가 전체와의 의미상
  중복까지 검증한 것은 아니며 이후 holdout에서 이 task/document ID를 제외해야 한다.
- pinned source revision·LFS SHA 확인, 대상 tokenizer/template SHA 일치 검사,
  입력 범위·형태·총량·기존 token IDs·가변 길이 저장/복원·오염 입력 거부 검사를 통과했다.
- 입력 SHA256: `36c8eddb18c51c3b9788d73195aa72527d9b9d48f818b79a6f0ab251acfebf3d`.
- 실행 묶음: `ai.tools/test/experiments/flash_next_quant_20261104/c03-input-ids.{json,safetensors}`.
  `convert_main.py`는 strict ragged reader를 사용하며 새 supervisor 기록은 `model-job-c03`로
  분리한다. 기존 c02 입력·연산 baseline·ngram 실행 코드/동결 신원은 보존했다.
- CPU 결과: `long-calibration-20261104/c03/CPU-CHECK.json`. 준비 영수증은
  `flash-next-quant-20261104/C03-PREPARED.json`. 이후 사용자 승인으로 시간별 진단/재개 활성화.
- 최종 독립 재리뷰 통과. 마지막 infra432개 통과(18.62초,
  `tmp/test-results/pytest/run-vpx38rb3`). 두 변환 unit inactive/MainPID0 재확인.

### 재개 전 추가 검토

사용자 요청으로 end-to-end 코드를 다시 검토했다. 가변 길이 forward/Hessian capture,
행별 checkpoint/PLE 입력, N2 bank 병렬화, 마지막 head 이후 side-model compile 경로에서
추가 shape 오류는 발견하지 못했다. runtime18개 동결 파일 hash·native stat과
원본131개 shard stat, c03 hash, tokenizer/template hash, recipe hash, ngram 보존을 확인했다.
원본 전체 SHA와 ngram 수치 검증은 기존 증거를 재사용했다.

최종 `config.json`만 가변 길이로 보정하고 `quantization_config.json`에는 예전
`2309×8192` 표시가 남는 metadata 버그1건을 수정했다. 두 파일에 실제1,999,496tokens,
길이 분포·hash를 동일하게 저장하고 `tensor_storage`를 보존한다. 완료 marker는
두 파일 일치 확인 뒤에만 쓴다. 추가 CPU 회귀 검사 통과, 입력 tokens/가중치는 불변이다.
수정 독립 재리뷰 및 infra432개 통과(18.51초, `tmp/test-results/pytest/run-dvypgcfg`).

검토 영수증: `flash-next-quant-20261104/READINESS-REVIEW-20261105.json`.
현재 판정은 **CPU 준비 완료 / 실제8K 연산·peak 메모리 검증 대기**다. 기존512토큰
pilot 통과만으로 긴 행도 통과했다고 간주하지 않는다. GPU·본체 변환은 실행하지 않았다.

## 중단과 보존

사용자 요청에 따라 `gufo-quant-model-20261104.service`를 중단했다.
ngram·본체 unit 모두 inactive/dead, MainPID=0이며 quant 컨테이너가 없다.
K8 ngram 51,840,275,568 bytes(약48.28 GiB)는 완료·연산 검증 상태다.
처음에는 본체 checkpoint `next_module_idx=2`를 보존했으나, 후속 사용자 지시로
기존 c02 `work/`(checkpoint·중간 텐서)77.493 GiB를 삭제했다.
설정과 파일 목록은 `C02-CHECKPOINT-REMOVAL.json`, 기존 실행 로그는 그대로 남겼다.
완료 ngram의 크기·mtime 보존을 확인했고 디스크 trim도 수행했다.
중단 영수증: `/home/m2on/projects/ai.tools/tmp/test-results/flash-next-quant-20261104/USER-STOP.json`.
새 calibration으로 바꾸면 본체는 별도 작업 디렉터리에서 처음부터 처리한다.
calibration과 독립인 완료 ngram은 재사용할 수 있다. 자동 재개하지 않는다.

## 실제 길이 조사

Flash-Next tokenizer로 내용·도구 정의·호출·결과를 계산했다. 대상 chat template의
역할 구분자 등은 제외했다. tool_calls는 JSON 직렬화로 세므로 ID/index/escaping을
포함할 수 있고 role/tool-call-id 연결 필드는 제외된다. 따라서 최종 입력 길이의
정확한 값이나 단순 하한이 아니라 내용 길이의 비교용 추정치다.
번역본의 영어 중복 필드와 metadata/gold patch는 제외했다.
API의 `truncated_cells`가 있으면 거부했다. 신규 자료는 **처음 8개 비무작위 표본**이므로
전체 분포·전체 품질을 대표하지 않는다. 표의 K는 약1,000 tokens이다.

| 자료 | 조사 수 | 중앙값 | 최소–최대 | 용도 |
| --- | ---: | ---: | ---: | --- |
| SWE-smith, tool | 8 | 15.9K | 11.3K–44.5K | 코드 수정·도구 호출·재시도 |
| Open-SWE-Traces v2.0, opencode | 8 | 32.3K | 14.0K–42.8K | 긴 코딩 에이전트 작업 |
| LongAlign-10k | 8 | 13.7K | 5.9K–53.2K | 영어·중국어 긴 문서 읽기 |
| 로컬 한국어 Hermes | 10,237(전체) | 620 | 132–7,468 | 한국어 도구 대화 |
| 로컬 영어 Hermes | 첫2,000 | 1,282 | 534–4,364 | 짧은 도구 대화 보존 |
| 로컬 한국어 Wikipedia | 한 shard 첫3,000 | 171 | 15–17,570 | 연속된 한국어 장문 보충 |
| smol-koreantalk | 8 | 607 | 97–1,624 | 이번 표본에서는 장문 근거 부족 |

한국어 Hermes에서 4,096 이상은24개, 8,192 이상은0개였다.
한국어 wiki 표본에서는 8,192 이상9개, 16,384 이상2개였다.
긴 문서와 여러 턴의 도구 대화는 서로 대체하지 않는다.

## 추천과 필터

1. [Open-SWE-Traces](https://huggingface.co/datasets/nvidia/Open-SWE-Traces)와
   [SWE-smith](https://huggingface.co/datasets/SWE-bench/SWE-smith-trajectories)를
   긴 코딩·도구 작업 후보로 우선한다. `resolved` 성공 라벨과 도구 호출/결과 연결을
   확인하고 repo/task 단위로 중복을 제거한다. SWE-smith tool/xml/ticks는 형식 변형을
   중복 작업으로 세지 않는다. Open-SWE의 metadata/reference patch는 입력에 넣지 않는다.
2. 성공 작업 중간의 오류·실패·수정·재검증은 그대로 유지한다. 첫8개 SWE-smith는
   모두 성공 라벨, Open-SWE는6개 성공·2개 실패였다. Open-SWE 실패 라벨 표본에도
   assistant의 테스트 통과 주장이 있었다. 마지막 답변만으로 성공 판정하지 않는다.
   라벨은 제공자의 판정이며 이번 조사에서 테스트를 독립 재실행하지 않았다.
3. [한국어 Hermes](https://huggingface.co/datasets/iknow-lab/hermes-function-calling-v1-ko)는
   한국어 도구 형식을 유지하는 데 쓰고, 로컬 한국어 wiki의 긴 원문을 보충한다.
   원문과 번역을 중복 가중하지 않는다. 단순히 한국어 비율을 늘리는 것이 품질 향상 증거는 아니다.
4. [LongAlign](https://huggingface.co/datasets/zai-org/LongAlign-10k)은 긴 문서 보조 후보다.
   대부분 문서+질문의2메시지 구조라 도구 루프 후보와 구분한다.
5. [Nemotron-SFT-Agentic-v4](https://huggingface.co/datasets/nvidia/Nemotron-SFT-Agentic-v4),
   [Nemotron-SFT-OpenCode-v1](https://huggingface.co/datasets/nvidia/Nemotron-SFT-OpenCode-v1)은
   도구/코딩 후보지만 API 표본 확보 실패로 실제 길이·품질을 아직 확인하지 못했다.
   검증된 후보와 같은 수준으로 취급하지 않는다.

평가용 Ko-LongRAG/Terminal-Bench 등은 calibration에서 제외한다. 같은 원문을 사용하는
한국어 wiki도 평가 문서와 중복 검사한다. 작업 성공 라벨은 선별 기준이지 EXL3 품질 개선의 증명은 아니다.

## 입력 설계의 제약과 남은 검증

현재 c02는 3,906×512=1,999,872 tokens이며 한국어 비율은19.65%다.
긴 원문이 있어도512씩 잘라 넣는 현재 방식에서는 긴 문맥을 유지하지 못한다.
총 토큰 예산을 우선 유지하고 일부를 연속된4K/8K 문맥으로 교체하는 후보부터 만든다.
16K는 소량 후보로만 고려한다. 서로 무관한 짧은 대화를 붙여 장문이라고 세지 않는다.
기본 파일 reader는 직사각형 입력이지만 내부 상태는 가변 길이 tensor 목록을 지원한다.
wrapper의 `get_file_calibration`만 strict ragged reader로 연결했다. padding 없는
`(1,L)`별 독립 샘플과 원본 token IDs를 유지한다. Hessian은 토큰 수로 가중된다.
최종 config도 실제 total_tokens·max_columns·variable_length·hash로 보정한다.
upstream 코드/커널 변경은 없으며 CPU 입력 생성에 HIP 최적화는 적용하지 않는다.
전체 토큰 수가 같아도8K all-expert 임시 버퍼는 증가한다. 실제 재개 전 긴 행의
유한값·연산·메모리 소형 검증이 필요하며, 기존512행 pilot을 그 증거로 재사용하지 않는다.
도구 쌍 보존, 실제 사용 template, repo/task 중복·평가 오염, 반복 로그 편중을 점검한 뒤
작은 연산/블록 비교로 후보를 추린다. 긴 입력의 품질 개선은 아직 측정하지 않았다.

## 재현 자료

- CPU 전용 감사 코드:
  `/home/m2on/projects/ai.tools/test/experiments/flash_next_quant_20261104/audit_long_sources.py`
- 표본 JSON·출처 URL·metadata revision snapshot·길이 결과:
  `/home/m2on/projects/ai.tools/tmp/test-results/long-calibration-20261104/`
- 핵심 결과: 위 디렉터리의 `CONTENT-LENGTH-AUDIT.json`.
- API 표본 요청은 revision pin 없이 조회했다. metadata snapshot revision을 표본의
  정확한 revision이라고 단정하지 않는다. 최종 수집에서는 revision과 hash를 고정한다.
- CPU Docker: network none, read-only, cpus2, memory4g, GPU 장치 미노출. 데이터의
  명령은 실행하지 않았다. 원본 모델·calibration 입력·완료 산출물은 변경하지 않았다.
- 독립 리뷰에서 표본 schema·중복 제외·요약 집계에 blocker 없음. 위 길이 해석의
  제한을 반영했다. CPU 구조/인프라 검사432개 통과(19.52초), GPU 검증은 실행하지 않았다.

## 2026-11-05 통계 수집 25% 후보 — 미적용

`select_capture_quarter.py`가 원본 c03의 source×길이(512/2048/4096/8192)
구간마다 토큰합의25%에 가장 가까운 비어 있지 않은 whole-row 부분집합을 고른다.
seed20261105, 원본 순서 유지,577행503322tokens(25.1724%). 희소 긴 한국어chat의
단일4428token행을 보존해 이상적499874보다3448 많고, ko_chat 비중은+0.6174%p다.
행을 자르거나 서로 붙이지 않는다. 원본/행/metadata/코드SHA와 인덱스를
`capture-quarter/CANDIDATE.json`에 보존한다. CPU300개 exhaustive/determinism 검사 통과.

실제 입력·체크포인트·동결코드에는 미적용이며 본체는 전체1999496tokens를 사용한다.
소스명/길이 비율만으로 의미·분야·품질 보존이 증명되지는 않는다. 대표 블록과 별도
독립holdout에서 분야별/상위오차/KLD 등을 비교하고, 전체state와 subset의 명시적
인덱스 매핑 및 적용 시작블록 이력을 검토해야 한다. 현재 c03의 미선택75%도 전체
calibration baseline에는 쓰인 데이터이므로 독립holdout으로 부르지 않는다.

25% 대표projection 검사를 `capture-profile-quarter/quarter/`로 실행했다(진행상태는
상위 STATE.json). archived idx2의 실제577행을 읽고 layer0 out/expert0 down만 검사하며,
완료된full2M packed baseline을 재사용한다. 원래forward prefix의 exact 확인이 선행된다.
미선택c03 진단평가는 독립holdout이 아니며, 두행렬 통과만으로 전체checkpoint를
25% 입력으로 바꾸지 않는다. 코드·매핑 및 추가품질 판단은 완료후 후속에서 처리한다.

## 25% 적용 매핑 후보 — 아직 미설치

`capture_subset.py`는 기존2309개 state와original_input_ids를 그대로 두고,
고정577 original_index의 discarded all-expert capture forward만 수행하는 후보다.
나머지1732행은 이 폐기되는 forward만 건너뛰며, 기존 첫5 normal reference와
전체2309행의 양자화후 state advance/저장/quant_preserves 처리는 유지한다.
입력ID 객체/인덱스·원래행순서·토큰합503322를 검증하고 기존skip-down을 명시scope로
결합한다. source SHA와 terminal-block 구조 검사를 통과한 cm.main 호출에서만 작동한다.

장점은 현재idx10 checkpoint의 재인덱싱/행삭제가 없고, 다음층에서full 수집으로 돌아갈
상태를 유지하는 것이다. 디스크state76GiB와advance 비용은 줄지 않으므로4배속도를
약속하지 않는다. 완료층full2M/적용시작idx10 이후Transformer H quarter의 혼합을
명시적 manifest로 기록해야 한다. 비Transformer/head/side 변환은 별도적용범위 판단이
필요하다. 후보는 CPU/독립리뷰 및작은 실제cm.main 비교 전 본체에 설치하지 않는다.
