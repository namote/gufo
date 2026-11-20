# 보관: 하지 않기로 했거나 해당 없는 항목 (2026-11-21)

사용자 결정으로 진행 계획에서 뺀 항목이다. 다시 필요해지면 여기서 꺼내 새 계획에 올린다.

## 지금 구조에 해당 없음

RAM checkpoint 층을 없앤 c597eed1(11-18) 뒤로는 효과가 없는 항목이다.

| 항목 | 출처 | 해당 없는 이유 |
| --- | --- | --- |
| V2-3 | upstream #1288, #1290 (0a473ebf, cc405337) | 운영 코드에서 `RetentionPolicy`를 쓰는 곳이 없다(테스트만 남음). 디스크 eviction은 단순 LRU라서 새 대화가 바로 밀려나지 않는다. |
| V2-5 | upstream #1285 (11a51d2f) | `CommittedBackingPool`을 부르는 곳이 없다. 디스크 복원 뒤의 capture도 serving thread에서 backing을 commit하지 않는다. |
| U2-4 | upstream #903, #1063 (`plan/upstream-sync/PLAN.md`) | RAM cache segment의 거부 조건과 시작 시 RAM 부족 처리다. RAM cache 자체가 없어졌다. |

## 미뤄 두었다가 보관한 운영 편의 항목

| 항목 | 내용 |
| --- | --- |
| OpenCode 생각 과정 길이 제한 | OpenCode가 보내는 요청에 reasoning budget을 걸어 생각이 너무 길어지는 것을 막는다. |
| OpenCode 설정 자동 전환 | 모델이나 서버 상태에 맞춰 OpenCode 설정을 자동으로 바꾼다. |
| OpenCode 이어 쓰기 저장 설정 | `tools/serving/opencode_resume.py`로 끊긴 OpenCode 세션을 이어 가도록 저장하고 캡처하는 설정이다(`plan/serving/CONTINUE-FINAL-MESSAGE-BRIEF.md` 5번). 도구는 이미 있고, 자동 연동만 하지 않았다. |
| 27B 문맥 1M (YaRN) | Qwen3.8 27B 서버에 YaRN을 켜서 문맥을 1M까지 늘린다. |

## upstream-sync2에서 빼기로 한 것 (2026-11-20 조사 보고서 "Rejected")

조사 보고서의 "Rejected" 목록을 따른다. 요약하면 다음과 같다.
- Rust 변경. 포크에 Rust 도구가 없다.
- GGUF 전용 항목: #1112, #958, #969.
- FP16 KV 전용 #1185.
- RAM 캐시를 끈 모드에서만 쓰는 항목: #934, #1272.
- 수요 매핑 세션 인프라 #1251, #1258, #1275(약 7K줄).
- Kyojin GDN HF chunk kernel: U4-2에서 기각한 그대로.
- exllamav3 CUDA 전용 커널.
