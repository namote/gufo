# Flash-Next 자체 EXL3 양자화 비트 배분 계획

**현재 상태 (2026-11-07):** c03 본체·보정 MTP·원본 BF16 비전 조립과 최종 검증이 완료됐다. 사용자 승인으로 c03를 상시 서비스 중이며 변환 감시는 종료했다. 아래는 날짜별 계획·실행 이력이다. [최종 결과](F3-MTP-CALIBRATION.md).

2026-11-05 00:47 KST 갱신. 상태: **8K 검사 통과, 사용자 승인으로 c03 본체 변환 실행 중**.
K8 ngram은 완료·연산 검증해 보존했다. 사용자 지시로 기존 c02 본체 checkpoint와
중간 텐서77.49 GiB를 삭제했다(설정·로그·삭제 영수증은 보존).
[긴 calibration 입력 c03](F3-LONG-CALIBRATION.md)을 구성·CPU 검증했다(약200만tokens, 최대8K).
시간별 점검·수정·재개는 [감시 절차](F3-CONVERSION-WATCH.md)를 따른다.
아래 c02 실행 기록은 중단 이전 이력이다.
목표는 BF16/FP16 원본에서 c02 코퍼스로 양자화하여 한국어·도구 호출·코딩
품질을 보존하고, 현재 Vemloir 팩과 비슷한 메모리 예산에서 프리필과 TG/MTP
비용을 줄이는 것이다. 아래 비트 수는 출발 후보이며 측정된 최적값이 아니다.

## 기준과 예산

- 비교 팩: `Vemloir/Qwen3.8-Flash-Next-sc-exl3`, `4.05bpw_sc_h8_ng8`,
  revision `fe8ad64bebc6de4ae9cb1eee4fd37342c0a51bc7`.
- 비교 팩은 사용자 지시에 따라 내장 중복본을 삭제하고 외장
  `/run/media/m2on/T5 EVO/AI/model/flash-next-exl3-vemloir-4.05-sc-h8-ng8`에 보존했다.
- 저장된 전체 파일 합계 120,565,867,719 B. `4.05bpw`라는 이름만으로 ngram,
  비전, MTP, 비양자화 텐서까지 포함한 전체 크기를 판단하지 않는다.
- 첫 후보의 전체 파일 예산은 이 합계 +1 GiB 이내로 잡는다. 텐서별 실바이트로
  산출하고, 절약분을 무조건 다 쓰지 않는다. 모델 RAM과 파일 크기는 구분한다.
- 실행 메모리는 C2/native256K/FP16 공유512K KV/내장 비전의 같은 설정으로 비교한다.
  정상 작업에서 가용 RAM 8 GiB 이상을 목표로 하며 기존 4 GiB 보호 기준을 유지한다.
  UMA GPU 메모리가 cgroup 수치에 모두 잡힌다고 가정하지 않는다.
- 원본 checkpoint의 실제 정밀도·revision·토크나이저를 먼저 고정한다.
  기존 EXL3를 풀어서 새 팩을 만드는 재양자화는 BF16 원본 양자화의 대체가 아니다.

기존 속도 기록은 `docs/GFX1151-KNOWHOW.md`의 Vemloir 절을 재사용한다.
동일 빌드에서 Vemloir PP512/2048/AR128=1081.9/1395.4/26.21 tok/s,
MTP128=38.55 tok/s; Kyojin **팩을 Gufo에서 실행한** PP512/2048/AR128은
1093.8/1393.2/30.50이었다. 이것은 Kyojin 엔진과 Gufo 엔진의 비교가 아니다.
기록된 K6~K8 GEMV 비용을 배분 후보 선정에 쓰되 새 팩의 속도 개선율을 예측하지 않는다.

## 현재 팩과 출발 배분 B1

현재 값은 팩의 `quantization_config.json`과 trellis shape를 읽어 확인했다.
비전·MTP 등 별도 구성은 인덱스/설정도 함께 사용하며, 최종 recipe 생성 시
원본과 출력의 전체 텐서 목록을 대조한다.

| 구성 | 현재 Vemloir | B1 출발 후보 | 결정 기준 |
| --- | --- | --- | --- |
| 라우팅 전문가 gate/up | K3/K3.5/K4 | **K4**, 두 투영의 비트는 묶음 | K3 계열을 처음부터 늘리지 않음 |
| 라우팅 전문가 down | K4/K5 | **K4 + 민감한 bank만 K5** | gate/up와 별도 민감도 측정 |
| GDN `in_proj_qkv`, `in_proj_z` | K6 | **K5**, 손실 시 해당 그룹 K6 | 큰 매-token 투영의 비용 후보 |
| GDN `out_proj` | K7 | **K6**, 손실 시 K7 | 반복 상태를 포함한 블록 출력도 검사 |
| full attention q/o | K6 | **K6 유지** | 1차에서 낮추지 않음 |
| full attention k/v, indexer qk | K8 | **K8 유지** | 작은 텐서, context 선택·attention 보호 |
| shared expert gate/up/down | K8 | **K6**, 손실 시 해당 그룹 K8 | 모든 토큰에 실행; 실제 이득과 품질로 결정 |
| `lm_head` | K8 | **K8 유지** | 출력 분포와 토큰 선택 보호 |
| ngram | K8 | **K8 유지** | 최초 실험에서는 다른 변경과 분리 |
| 비전 | K8 | **원본 BF16 유지 (후속 변경 승인)** | 최종 원본 텐서 동일성·이미지 평가 |
| MTP 양자화 층 | 설정상 K4 | **K4/K6/K8 혼합 + 전체 c03 보정 (후속 변경 승인)** | F3-MTP-CALIBRATION.md; 수락률·완료-token/s 검증 |
| token embedding | BF16 | **BF16 유지** | 현재 Gufo 로더 계약 |
| router·norm·HC·decay·conv 등 원래 raw 텐서 | F16/BF16/F32 | **원본 dtype 유지** | 전역 비트 옵션으로 양자화하지 않음 |

K5/K6로 낮추는 항목은 품질 비열등성을 확인할 때만 최종 recipe에 남긴다.
사전 연산 검사에서 실패한 그룹은 최종 변환 전에 높은 비트로 명시적으로 결정한다.
실행 중 자동 정밀도 전환이나 silent fallback은 추가하지 않는다.

### 실제 byte 계산으로 확인한 배분 여지

현재 트렐리스와 같은 shape·스케일 저장 형식을 가정한 **payload 차이**다.
원본에서 새로 변환한 최종 파일 크기나 실행 RAM의 실측은 아니다.

| 변경 | 현재 팩 대비 |
| --- | ---: |
| routed gate/up/down 전체를 K4로 맞춤 | +100 MiB |
| GDN qkv/z K6→K5 | −180 MiB |
| GDN out K7→K6 | −67.5 MiB |
| shared experts K8→K6 | −56.25 MiB |
| 위 B1 출발안 합계 | **−203.75 MiB** |

한 층의 한 expert projection bank는 512×2560×640개 가중치이므로
K4→K5 비용은 100 MiB다. 위 절약분으로 down bank 두 개를 K5로 올리면
트렐리스 기준 현재 팩 대비 약 −3.75 MiB다. 이는 **비슷한 크기의 후보를
구성할 수 있다는 예시**이며, 두 bank의 층 번호는 아직 선정하지 않았다.
gate/up를 함께 K5로 올리면 한 층에 200 MiB가 든다.
민감한 bank가 더 많으면 +1 GiB 한도 안에서 다시 배분하거나 속도 후보를 포기한다.

## 배분 제약과 측정 방법

1. 현재 `BindExl3LinearBank`는 같은 층·같은 투영의 512 experts에 동일 bitrate를
   요구한다. 개별 expert마다 다른 K를 넣지 않는다. bank 단위로 승격한다.
   gate/up는 같은 K로 묶고, 가능하면 k/v도 같은 K를 유지한다.
2. 앞/뒤층을 무조건 높이거나 특정 층을 한국어·수학 담당이라고 가정하지 않는다.
   실제 한국어·도구·코드 입력에서의 활성값과 출력 오차로 순위를 정한다.
3. 비용은 저장 바이트, GPU 가중치/작업 메모리, PP2048 시간, AR/MTP 시간으로
   나누어 본다. routed expert의 사용 빈도는 실행 비용 가중치로 쓰지만,
   드문 전문가를 자료가 적다는 이유만으로 낮추지 않는다.
4. 전체 7만여 투영에 suffix-forward를 수행하는 탐색부터 시작하지 않는다.
   저장된 프로파일로 dense 후보를 고르고, 각 bank의 양자화 오차와 라우팅별
   `W·x`/블록 출력 오차를 작은 고정 입력에서 모은 뒤 상위 후보만 추가 검사한다.
   로컬 오차 순위는 proxy이며 최종 KL/작업 품질의 증명이 아니다.
5. GDN은 단일 토큰 오차만 보지 않고 연속 입력의 recurrent state 오차를 확인한다.
   전문가 평가는 평균 외에 한국어·도구·코드별 오차와 저빈도 표본도 기록한다.
6. 최종적으로 오류 감소 대비 byte·실행시간 비용이 유리한 그룹부터 비트를 준다.
   민감도와 시간 측정치를 저장해 배분표를 바꿀 때 다시 수집하지 않는다.

## 캘리브레이션과 품질 기준

- 기존 c02 전체: `ai.tools/storage/evaluations/rq-r3-calib-20261029/c02/`,
  3,906×512 = 1,999,872 tokens, 실제 한국어 토큰 약19.6%, tool-call 청크 약35.1%.
  `chunks.bin` SHA256 `40d5d6535aa455a42514a0f31ccf40ba3863ad51a864968cd289ad4dc2b4bcc4`.
- 토크나이저 일치 확인 후 원본 token IDs를 `input_ids` safetensors로 포장한다.
  텍스트 decode→encode가 동일하다고 가정하지 않는다. 기존 512-token 청크를
  무작위 연결해 실제 긴 대화라고 취급하지 않는다.
- 512-token 코퍼스로만 긴 문맥·도구 왕복·GDN 상태 누적을 대표할 수 없다.
  필요할 경우 원문 대화에서 연속 2K~8K 구간을 별도 확보하고 비율/출처를 고정한다.
  이 보충 자료 준비와 실제 변환은 다음 실행 단계다.
- recipe 선별용 데이터와 최종 평가는 문서/대화 단위로 분리한다. c02로 최종 점수를
  보고하지 않는다. 기존 회귀셋의 c02 중복 검사 기록을 재사용하되, 새 평가셋은 별도 확인한다.
- 동일 BF16 원본·동일 token history 대비 NLL, KL, top-1 agreement와 한국어/도구/코드
  slice를 기록한다. top-k 근사 KL이면 그렇게 명시한다. FP8 참조를 BF16 참조라 부르지 않는다.
- BF16 teacher 결과가 없다면 처음 한 번 수집해 저장한다. 모든 후보가 이 결과를 재사용한다.
  전체 원본을 올리지 못하면 참조 수집 환경부터 해결하며 양자화 팩을 원본 대신 쓰지 않는다.
- 품질 우선 제안 마진: paired NLL 증가 전체 ≤0.01 nat/token, 각 핵심 slice ≤0.02.
  문서/대화 단위 paired bootstrap 95% 상한으로 판정한다. 표본이 부족하면 미판정이다.
  마진은 계획값이며 결과를 보고 넓히지 않는다. KL 감소를 품질 개선의 추가 근거로 사용한다.
- tool JSON/schema, 인자 일치, 코딩 테스트, 한국어 응답의 기존 필수 회귀를 모두 통과한다.
  자유 생성이 달라지는 것은 양자화 실패 자체가 아니며 정답/규약/분포 오차로 평가한다.
- MTP는 같은 새 target 기준 greedy AR 일치·수락·rollback을 확인한다. K4→K5의
  draft 품질 향상이 draft 계산비용을 상쇄할 때만 후속 채택한다.

## 변환기·메모리의 사전 조건

로컬 Kyojin 1.8.6 소스의 `convert_model.py`/`allocation.py`는 recipe에
**정수 비트·정확한 텐서 키**를 요구한다. 현 Gufo가 K3.5를 읽는 것과
로컬 변환기가 recipe로 K3.5를 생성할 수 있는 것은 별개다.
따라서 B1은 정수 K4/K5/K6/K8로 만들고, wildcard는 문서상의 그룹 표기에만 쓴다.
원본 모듈 목록에서 완전한 recipe를 생성하여 누락·중복·예상치 못한 dtype을 검사한다.
`--hq`/`--bits`가 recipe와 중복 적용된다고 가정하지 않고 실제 선택 버전을 고정한다.

공식 self-calibration 최적화 문서는 sparse 모델에 아직 미검증이며, 측정 단계에서
비양자화 모델+Hessian을 요구한다고 명시한다. 이를 128GB Flash-Next에 그대로
실행 가능하다고 가정하지 않는다. 우선 작은 bank/레이어에서 기존 ROCm 변환기의
로드·해제, Hessian/activation/workspace peak를 확인한다. 별도 27B의 시간을
Flash-Next 소요시간으로 인용하지 않는다.

원본 + 최종 팩 + 작업 디렉터리(문서상 출력 한 벌 이상) + 보존 증거의 디스크
예산을 합산한다. 기존 모델·캐시를 임의로 지워 맞추지 않는다. 원본이 진짜 BF16인지,
ngram/vision/MTP가 다른 정밀도 원본에서 왔는지도 구성별 provenance로 남긴다.

## 실행 순서와 완료 조건

1. 원본·변환기 pin, 전체 텐서/byte inventory, c02 포맷/토크나이저 검증.
2. GPU 소유자 확인과 작업 락 인계 후 작은 bank 검사 및 peak 메모리 확인.
   현재 다른 작업의 GPU 프로파일링을 방해하거나 서버를 임의로 시작하지 않는다.
3. B1과 보수적인 높은-bit 대안을 **작은 연산 수준에서** 함께 추린다.
   탈락 후보마다 전체 모델을 만들거나 긴 정확성 검증을 돌리지 않는다.
4. 최종 recipe 한 개를 먼저 변환한다. checkpoint/resume와 중간 산출물을 보존한다.
   별도 recipe의 결과를 완료 중인 job에 덮어쓰지 않는다.
5. 전체 검증 한 묶음: 로딩/메모리, NLL·KL·핵심 회귀, 이미지, PP2048,
   AR128, MTP128, branch/cache/C2, 16K~32K 상태 누적. 256K 전체 검증은 별도 범위다.
6. 기존 baseline은 build/model/prompt/cache 조건이 맞으면 그대로 재사용한다.
   조건이 다르면 필요한 대조군만 한 번 저장한다. 3%짜리 후보마다 반복하지 않는다.
7. 프리필을 최우선 성능 gate로 두고 PP/TG/MTP 각각 5%를 넘는 명확한 회귀는
   자동 채택하지 않는다. 품질/메모리/속도가 동등하면 유지보수성 좋은 배분도 채택 가능하다.
   잡음·신뢰구간이 큰 결과는 미판정으로 남기며 해당 항목만 추가 확인한다.
8. 최종 byte·RAM·품질·속도 표를 남긴 후 이미지/기본값 적용은 별도 작업으로 처리한다.

## 근거

- 로컬 측정: `docs/GFX1151-KNOWHOW.md`의 Vemloir 절, `build/fn/vem-mtp-2/report.json`.
- 캘리브레이션: `plan/archive/rotated-quant/PLAN.md`의 R3 코퍼스 절 및 c02 manifest.
- [공식 변환 옵션](https://github.com/turboderp-org/exllamav3/blob/master/doc/convert.md).
- [공식 민감도·recipe 최적화와 sparse 제약](https://github.com/turboderp-org/exllamav3/blob/master/doc/optimize.md).
- 로컬 변환기: `ai.tools/test/experiments/gufo_upstream_port_20261023/kyojin_runtime/source/`.

## 2026-11-04 실행 준비 및 변환 속도 검사

- BF16 원본: 외장 `originals-flash-next-hf-20261030/Qwen-Qwen3.8-Flash-Next`,
  revision `de4b8e4d43b917e7706784d8bb445c9af86a3540`; 131 shards,
  BF16 1,655 tensors + I64 3 tensors. 기존 SHA 검증을 재사용하고 현재 header/stat 확인.
- c02 1,999,872 token IDs를 그대로 I32 safetensors로 포장했다. 원본 chunks SHA와
  Flash-Next tokenizer SHA 일치. 행/길이 3906×512 유지; 임의 축소/연결 없음.
- 변환기 Kyojin v1.8.6 `fbde8a3670c80727a202da3bce581da2de7c88da`,
  기존 ROCm 10.1 Torch 2.14 환경과 native extension 사용. 재설치/재빌드 없음.
- instantiated model로 B1 recipe 74,040개 정확한 키 생성: K4 73,728개,
  K5 72개, K6 204개, K8 36개. main budgeted weights 평균 4.035976 bpw.
  전 층 민감도 순위가 없으므로 down bank K5 승격은 아직 하지 않았다.
- 첫 블록 실제 c02 4행으로 14개 후보/대조 연산 검사 통과. 실제 packed EXL3
  forward까지 비교했고 finite/비정상 fallback 없음; peak GPU allocation 7.31 GiB.
  GDN qkv K5 packed 상대오차 0.467%, z K5 0.396%, out K6 0.447%; 공유전문가
  K6 약0.189–0.305%, 표본 expert0 down K4 3.842%. 전체 모델 비열등성 판정은 아님.
- K8 ngram 기존 ROCm 선택은 `base`; 이미 빌드된 `opt` L160 커널을 검사했다.
  동일 131,072행에서 base/8192 chunk 17.837초, opt/32768 chunk 2.975초.
  opt 2 streams 2.520초, 4 streams 2.560초; 최종 후보는 opt/32768/2 streams.
  비교한 packed bytes 전부 일치. zero/constant/tiny/ramp/alternating 경계 입력도 일치.
  1.71시간은 전체 행 수에 대한 **계산만의 환산값**이며 I/O/검사를 포함한 완료 ETA가 아니다.
- ngram 각 chunk는 원본/출력 finite 및 상대 Frobenius 오차 ≤0.01을 요구한다.
  완성 후 전체 파일 크기/행 수/K8/4096행 packed readback 검사. 최종 모델 품질과 분리.
- c02 HC 상태는 FP32 약76.29 GiB. 정밀도 변경 없음. `EXL3_CKPT_KEEP_OLD=0`으로
  새 체크포인트가 완전히 게시된 뒤 이전 것만 제거; 최신 상태와 qtensors는 보존한다.
  `EXL3_RWH=0`; 파일 작업+팩+체크포인트 피크 약329 GiB 추산, 시작 여유 약507 GiB.
- 실행 코드: `ai.tools/test/experiments/flash_next_quant_20261104/`.
  측정/상태: `ai.tools/tmp/test-results/flash-next-quant-20261104/`.
  `ngram_job.py`는 ngram을 처리한다. 별도 `model_job.py`는 ngram 전체 검증 완료를
  기다린 뒤 본체 53개 모듈의 체크포인트 경계, MTP/비전, 최종 pack 조립을 순서대로 실행한다.
  systemd user units: `gufo-quant-ngram-20261104`, `gufo-quant-model-20261104`.
- 본체는 기존 `EXL3_QUANT_THREADS=2`의 private-stream 경로를 쓰며, 수치가 달라지는
  `EXL3_ENABLE_GROUPING`은 끈다. main Hessian/입력/출력 비정상 값과 비보정 fallback을
  거부한다. 의도된 uncalibrated MTP/비전은 별도이며 최종 품질 검증이 필요하다.
  본체 각 모듈은 1시간 내 완료·체크포인트 전진을 요구한다. 초과/메모리/수치/디스크
  실패는 자동 재시도하지 않고 기록·중간 산출물을 보존한다. 200만 token 처리 시간이
  이 창을 넘으면 해당 병목을 조사한 뒤 이어야 하며, 현재 전체 완료 시간은 미측정이다.
- 실행/재시작 시 source/model/recipe/c02/native identity를 고정하고 재검사한다.
  GPU 단독 락, host 여유 8 GiB·디스크 여유 100 GiB 하한을 감시한다.
  독립 코드 검토/수정 후 재검토 및 CPU infra 432개 검사 통과.
  main 모델 변환·전체 품질/성능 검증은 아직 미완료이며 기본 모델/서버는 변경하지 않는다.

## 2026-11-04 양자화 코드 품질·추론 속도 검토

- 본체/헤드/MTP/비전의 EXL3 Linear는 `quantize.py:1270–1597`의
  `regularize`에서 입력·출력 Hadamard128, 부호/채널 스케일, global-scale 탐색을
  적용한다. 본체는 activation Hessian + LDLQ 보정도 사용한다. ngram만 별도
  무회전 160차원 trellis이며 raw embedding/router/norm/HC까지 회전하는 것은 아니다.
- CPU inventory 검사: qmap 대상 main 74,041(헤드 포함), MTP 1,546, vision 164개
  모두 입력/출력 128배수. B1 K4/K5/K6/K8, mul1은 Gufo GEMV/WMMA 지원 범위이며
  expert bank 내부 혼합 bitrate는 없다. 새 팩의 실제 dispatch/성능은 완료 후 확인한다.
- 품질 주의: `modules/quant/fp16.py:122–129`, `exl3.py:637`의 기본
  `EXL3_HIP_F32OUT_VIA_F16=1`은 FP32 출력 요청도 M≥33에서 FP16으로 계산한 뒤
  확장한다. c02 512행 capture/reference와 기존 bank pilot에도 적용되는 경로다.
  추가 반올림이 뒤 층 Hessian에 반영될 수 있다. 사용자 동의 후 본체 시작 전에
  변환 전용 옵션 0으로 변경했다. `bank-pilot-fp32`의 작은 검사 통과 후 본체로
  진행한다. 이전 파일럿은 보존한다. 오프라인 비용이 늘 수 있으며 실제 품질 향상은
  아직 미검증이다. Gufo 추론 설정/커널은 변경하지 않았다.
- MTP K4/vision K8은 회전은 적용하되 `convert_model.py:1572` 이후 meta-H를
  쓰는 비보정 변환이다. MTP는 수락률·완료-token/s, vision은 이미지 과제로 검증한다.
- c02 약200만 토큰도 각 행은 512토큰이다. 256K/GDN 누적 상태의 대표성이 없으며,
  첫 블록의 작은 검사만으로 전 층 K5/K6 배분의 비열등성을 확정하지 않는다.
- 검사 보완: wrapper와 작은 파일럿에 FP16로 저장한 `suh/svh/bias` finite 검사를
  추가했다(MTP/비전 포함). 최종 artifact audit도 유지한다. 본체 다음 층의
  calibration state는 실제 packed layer를 reload해 계산하므로 FP32 복원 weight를
  계속 사용하는 버그는 발견되지 않았다. MTP/vision은 변환 중 forward 검사가 없다.
- 본체 2-worker 경로의 Cholesky/inverse는 CPU FP32다. GPU 단일-worker와 byte-exact
  증거는 없으며, ngram에서 확인한 packed 동일성을 본체에 일반화하지 않는다.
  독립 리뷰는 `quantize.py:988/1030/1439/1458`의 shared-H 최초 worker만 입력 부호
  RNG를 소비하여 출력 부호의 난수 위치가 작업 순서에 의존할 가능성도 확인했다.
  실제 차이/품질 저하는 미측정이며, 재현성 개선 시 입력/출력 RNG 분리 후 작은
  1-worker/2-worker 반복 비교가 필요하다. 현재 파일/seed를 실행 중 변경하지 않는다.
- 리뷰 후 미시작 본체 queue만 정지하고 이전 코드·동결 명세를
  `model-job/pre-fp32-review/`에 보존했다. ngram frozen 코드/recipe/runtime는 불변이다.
  ngram 첫 창은 5,001 chunk(51.2%) 저장 후 정상 timeout,
  다음 창 admission 실패로 중단됐다. 실패 기록을 `resume-review-20261104/`에 보존하고
  fresh GPU/NPU quiet 확인 후 같은 frozen 설정으로 재개했다. 본체 대기도 복구했다.
- 본체 설정 보완 독립 재리뷰·CPU infra 432개 통과. `FP32-MIGRATION.json`에 이전/새
  identity를 기록하고 미시작 본체만 재고정했다. 새 FP32 파일럿/전체 검증은 실행 대기다.
