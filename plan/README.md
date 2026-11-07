# gufo 작업 계획 (ai-tools)

> 위치: 2026-10-25 사용자 지시("ai tools에 계획이나 agent md gufo 에 관련된거 있으면 옮기라고")로 ai.tools `plan/`에서 이 gufo 저장소(`~/projects/gufo/plan/`)로 옮겼다. 본문의 `test/`·`storage/`·`tmp/`·`infra/`·`docs/` 경로는 ai.tools(`~/projects/ai.tools`) 기준이다. gufo 소스는 `~/projects/gufo`(이전 `~/.cache/ai-tools/gufo/s4`)다.

- [gufo 원본 이식·제안](gufo-upstream-port/PLAN.md): paged KV 이식, 정리 브랜치 `ai-tools/port-tidy`, [PR 계획](gufo-upstream-port/PR-PLAN.md)(원본 제안 보류, 포크 `namote/gufo`에서 독자 유지)
- [회전 가중치 양자화 RQ](rotated-quant/PLAN.md): Qwen3.8 27B 먼저(공통 구조로 두고 Flash-Next로 확장 예정), 입력 쪽 Hadamard 회전 + 기존 GGUF 타입, 컨버터(R1)→로더(R2)→imatrix(R3)→실험(R4)
- [Flash-Next 자체 EXL3 비트 배분](exl3/F3-QUANT-BIT-ALLOCATION.md): c03 본체·전체 보정 혼합 MTP·원본 BF16 비전 조립과 Nix10.1 정확성·품질·속도 검증 완료. [최종 결과·측정 한계](exl3/F3-MTP-CALIBRATION.md), [감시 종료·보존 경로](exl3/F3-CONVERSION-WATCH.md). 완료 K8 ngram/기존 baseline 재사용. 검증 당시에는 서비스를 전환하지 않았고, 이후 사용자 승인으로 c03 상시 서비스로 전환함.
- [v11 gufo 엔진 평가](v11-gufo/V11-GUFO-EVAL-20261020.md), [캐시 설계](v11-gufo/V11-GUFO-CACHE-DESIGN-20261020.md), [자리→disk 직접 저장](v11-gufo/V11-GUFO-SEAT-DISK-PLAN-20261020.md), [S4 결과](v11-gufo/V11-GUFO-S4-RESULTS-20261020.md)
