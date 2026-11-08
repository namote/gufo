# gufo 작업 계획 (ai-tools)

> 위치: 2026-10-25 사용자 지시("ai tools에 계획이나 agent md gufo 에 관련된거 있으면 옮기라고")로 ai.tools `plan/`에서 이 gufo 저장소(`~/projects/gufo/plan/`)로 옮겼다. 본문의 `test/`·`storage/`·`tmp/`·`infra/`·`docs/` 경로는 ai.tools(`~/projects/ai.tools`) 기준이다. gufo 소스는 `~/projects/gufo`(이전 `~/.cache/ai-tools/gufo/s4`)다.

## 진행 중 (Active)

- [EXL3 / Flash-Next 결정](exl3/PLAN.md): 형식·커널·서비스 결정 기록
- [EXL3 / Flash-Next 인계](exl3/HANDOFF.md): 현재 배포된 서비스 상태(B5 runtime)
- [Flash-Next B-series 요약](exl3/B-SERIES-SUMMARY.md): B1–B8 최적화 묶음 색인, 측정은 [GFX1151 know-how](../docs/GFX1151-KNOWHOW.md)
- [B8 정리 지시](exl3/B8-CLEANUP-BRIEF.md): 마무리 정리(진행 중)

## Archive

끝났거나 접은 계획. 측정 근거로 계속 인용되므로 지우지 않는다.

- [gufo 원본 이식·제안](archive/gufo-upstream-port/PLAN.md): paged KV 이식과 운영 전환 완료, [PR 계획](archive/gufo-upstream-port/PR-PLAN.md)(원본 제안 보류, 포크 `namote/gufo`에서 독자 유지)
- [회전 가중치 양자화 RQ](archive/rotated-quant/PLAN.md): Qwen3.8 27B 입력 쪽 Hadamard 회전 + GGUF 타입. EXL3 채택으로 대체됨
- [v11 gufo 엔진 평가](archive/v11-gufo/V11-GUFO-EVAL-20261020.md), [캐시 설계](archive/v11-gufo/V11-GUFO-CACHE-DESIGN-20261020.md), [자리→disk 직접 저장](archive/v11-gufo/V11-GUFO-SEAT-DISK-PLAN-20261020.md), [S4 결과](archive/v11-gufo/V11-GUFO-S4-RESULTS-20261020.md)
- [EXL3 / Flash-Next 완료 문서](exl3/archive/): F1–F3, X3/XA/XB/XQ, B1–B7 브리프·GPU 패스·결과. 예: [Flash-Next 자체 EXL3 비트 배분](exl3/archive/F3-QUANT-BIT-ALLOCATION.md)(c03 조립·검증 완료, [최종 결과·측정 한계](exl3/archive/F3-MTP-CALIBRATION.md), [감시 종료·보존 경로](exl3/archive/F3-CONVERSION-WATCH.md))
