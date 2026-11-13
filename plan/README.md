# gufo 작업 계획 색인

이 폴더의 입구다. 진행 중인 계획은 위에, 끝난 계획은 `archive/`에 있다.
측정 결과와 교훈은 계획 문서가 아니라
[GFX1151 know-how](../docs/GFX1151-KNOWHOW.md)에 기록한다.

## 진행 중 (Active)

| 문서 | 내용 |
| --- | --- |
| [exl3/HANDOFF.md](exl3/HANDOFF.md) | **먼저 읽을 문서.** 지금 배포된 Flash-Next 서비스(c03, 1M YaRN, K4V4, 시간 분배 스케줄러)와 다음 작업 |
| [serving/CONCURRENCY-PLAN.md](serving/CONCURRENCY-PLAN.md) | 동시 접속 성능 계획과 결과(S0 측정, 시간 분배, 혼합 배치 기각) |
| [research/IDEAS.md](research/IDEAS.md) | 연구 아이디어: R1 EoRA·어댑터, R2 동적 다중 프로파일 abliteration, R3 정치 방향 |
| [research/STEERING-API-DESIGN.md](research/STEERING-API-DESIGN.md) | 방향 조절 API: A 실행 중 프로파일 올리기(진행 중), B 서버 안 활성값 수집, C 외부 도구(방향 계산·가중치 학습), D 관리 페이지(할 일: 데이터셋·비교군·레이어별 그래프 실시간·조절 시험) |
| [exl3/PLAN.md](exl3/PLAN.md) | EXL3 / Flash-Next 형식·커널·서비스 결정 기록 |
| [exl3/B-SERIES-SUMMARY.md](exl3/B-SERIES-SUMMARY.md) | Flash-Next B1–B10 최적화·정리·GPU 패스 묶음 색인(브리프와 GPU 패스 링크) |
| [exl3/HC-Q8-CALIBRATION.md](exl3/HC-Q8-CALIBRATION.md) | BF16 원본 기반 HC Q8 보정 및 코드·스케일 반복 보정 비교. 변환 도구로 이관, GPU 수집·실제 품질 검증 준비 중 |
| [upstream-sync/PLAN.md](upstream-sync/PLAN.md) | upstream gufo(v0.31.0)·Kyojin main에서 가져올 수정·기능·성능 항목 U1–U5. 전체 merge 대신 골라 가져오기 |
| [upstream-sync/CLEANUP.md](upstream-sync/CLEANUP.md) | 기존 runtime 정적 감사에서 나온 작은 문제 C1–C7 정리 계획 (batched MTP draft 경로, staging 옵션 제거, cache identity 등) |

## Archive (끝났거나 접은 계획)

측정 근거로 계속 인용되므로 지우지 않는다. 고치지도 않는다(당시 기록).

| 폴더 / 문서 | 내용 |
| --- | --- |
| [exl3/archive/](exl3/archive/) | EXL3 / Flash-Next 완료 문서: F1–F3, X3/XA/XB/XQ, B1–B10 브리프·GPU 패스·결과, 도구 가져오기 브리프. 예: [자체 EXL3 비트 배분](exl3/archive/F3-QUANT-BIT-ALLOCATION.md)(c03 조립·검증, [MTP 보정 결과](exl3/archive/F3-MTP-CALIBRATION.md), [변환 감시 종료](exl3/archive/F3-CONVERSION-WATCH.md)), [B9 구조 정리](exl3/archive/B9-STRUCTURE-BRIEF.md), [B10 GPU 패스](exl3/archive/B10-GPU-PASS-BRIEF.md) |
| [archive/gufo-upstream-port/](archive/gufo-upstream-port/PLAN.md) | paged KV 이식과 운영 전환(완료). [PR 계획](archive/gufo-upstream-port/PR-PLAN.md)은 보류: 포크 `namote/gufo`에서 독자 유지 |
| [archive/rotated-quant/](archive/rotated-quant/PLAN.md) | 회전 가중치 양자화 RQ(Qwen3.8 27B 입력 Hadamard + GGUF). EXL3 채택으로 대체 |
| [archive/v11-gufo/](archive/v11-gufo/V11-GUFO-EVAL-20261020.md) | v11 엔진 평가, [캐시 설계](archive/v11-gufo/V11-GUFO-CACHE-DESIGN-20261020.md), [자리→disk 직접 저장](archive/v11-gufo/V11-GUFO-SEAT-DISK-PLAN-20261020.md), [S4 결과](archive/v11-gufo/V11-GUFO-S4-RESULTS-20261020.md) |

archive 문서의 `test/`·`storage/`·`tmp/`·`infra/` 경로는 은퇴한 ai.tools
(`~/projects/_ai.tools`, 읽기 전용 기록) 기준이다. 평가 입력은
`~/projects/models/eval/flash-next/`, 모델은 `~/projects/models/`에 있다.
