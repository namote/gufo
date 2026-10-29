#!/usr/bin/env python3
"""Build tools/qwen27b/rq_quality/accept_prompts.json: the 64-prompt DFlash2 text acceptance suite.

Sources, in order:
  - docs/models/qwen3.8-27b/artifacts/speculative-corpus.json: all 11 prompts (repository, MIT).
  - tools/qwen27b/rq_quality/d4_ko_prompts.json: the 15 instruction, summary and code-explanation
    prompts (W2-authored); a system message is prepended to the user text.
  - 38 prompts below, written for this suite by the W2 agent on 2026-10-29 ("W2-authored").
Acceptance prompts are not a quality metric, so overlap with the R0 corpus or the R3 calibration
corpora is recorded (accept.py check-overlap), not forbidden.

Run: python3 -I tools/qwen27b/rq_quality/accept_prompts_src.py > tools/qwen27b/rq_quality/accept_prompts.json
then accept.py check-overlap --write to record the overlap in the file's manifest.
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]

W2 = [
    # English
    ("en_explain_cache", "expository", "Explain to a new engineer what a CPU cache line is, why false sharing slows down multithreaded code, and how padding fixes it."),
    ("en_explain_tides", "expository", "Explain why most coasts see two high tides a day, and why the times shift by about fifty minutes each day."),
    ("en_explain_bloom", "expository", "Describe how a Bloom filter works, what a false positive is, and how the number of hash functions affects it."),
    ("en_summary_bakery", "summarization", "Summarize in three bullet points: A neighborhood bakery switched from paper order slips to a tablet queue. Morning wait times fell from twelve to seven minutes, but two older regulars said they found the screen confusing, so the owner kept one paper pad by the register for anyone who asks."),
    ("en_summary_meeting", "summarization", "Turn these notes into a short status update for a manager: build server disk at 91 percent; nightly job failed twice this week because of it; Ana is moving old artifacts to cold storage on Thursday; no customer impact so far."),
    ("en_code_python_lru", "code", "Write a small Python LRU cache class with get and put in O(1) time using an OrderedDict, then show a three-line usage example."),
    ("en_code_sql_window", "code", "Write a SQL query that returns, for each customer, their three most recent orders, using a window function. Assume a table orders(customer_id, order_id, created_at)."),
    ("en_code_rust_err", "code", "Show a short Rust function that reads a file to a string and returns a custom error enum covering not-found and permission-denied, using the ? operator."),
    ("en_code_review", "code", "Review this JavaScript and list the bugs: function avg(xs) { let s; for (let i = 1; i < xs.length; i++) s += xs[i]; return s / xs.length; }"),
    ("en_creative_lighthouse", "creative", "Write a short poem of eight lines about a lighthouse keeper who has never seen a ship."),
    ("en_creative_letter", "creative", "Write a friendly letter from a houseplant to the person who keeps forgetting to water it."),
    ("en_reason_ages", "reasoning", "Mia is twice as old as Leo was when Mia was as old as Leo is now. Leo is 24. How old is Mia? Show your steps."),
    ("en_reason_schedule", "reasoning", "Three meetings of 40, 25 and 50 minutes must fit between 9:00 and 11:00 with at least 5 minutes between them. Is it possible? Give one valid schedule or explain why not."),
    ("en_struct_recipe", "structured", "Return only JSON describing a pancake recipe with fields: name, servings, ingredients (array of {item, quantity}), and steps (array of strings)."),
    ("en_struct_yaml", "structured", "Write a YAML configuration for a web service with three environments (dev, staging, prod), each with a replica count, a log level and a database URL placeholder."),
    ("en_repeat_list", "repetitive", "Copy the following list exactly, then copy it again in reverse order: apple, river, candle, mirror, violin, harbor, pencil, garden."),
    ("en_repeat_table", "repetitive", "Print a multiplication table for 7 from 7 x 1 to 7 x 20, one line per entry, in the form '7 x n = result'."),
    ("en_instr_email", "instruction", "Write a polite two-paragraph email declining a meeting invitation for Friday and proposing two alternative times next week."),
    ("en_instr_steps", "instruction", "Give step-by-step instructions to replace a bicycle inner tube, in at most ten numbered steps."),
    # Korean
    ("ko_explain_inflation", "expository", "물가 상승률과 기준금리가 어떻게 연결되는지 고등학생이 이해할 수 있게 설명해 줘."),
    ("ko_explain_tls", "expository", "웹사이트에 접속할 때 HTTPS가 어떤 순서로 연결을 안전하게 만드는지 단계별로 설명해 줘."),
    ("ko_explain_fermentation", "expository", "김치가 익어 가면서 맛이 시어지는 이유를 미생물의 관점에서 설명해 줘."),
    ("ko_summary_policy", "summarization", "다음 내용을 세 문장으로 요약해 줘: 한 회사가 주 4일 근무를 여섯 달 동안 시범 운영했다. 직원 만족도 설문 점수는 크게 올랐고 이직 신청은 줄었다. 다만 고객 응대 부서는 금요일 문의를 처리하려고 교대 근무를 새로 짜야 했고, 일부 팀장은 회의 시간이 부족하다고 답했다. 회사는 내년에 고객 응대 부서만 다른 방식으로 운영하는 안을 검토하고 있다."),
    ("ko_summary_news", "summarization", "아래 기사 내용을 한 문장 제목과 두 줄 요약으로 정리해 줘: 지역 도서관이 다음 달부터 밤 10시까지 문을 연다. 시험 기간마다 열람실 자리가 부족하다는 민원이 이어졌기 때문이다. 연장 운영에 필요한 인력은 자원봉사자와 시간제 직원으로 채운다."),
    ("ko_code_python_dedup", "code", "파이썬에서 리스트의 순서를 유지하면서 중복을 제거하는 함수를 작성하고, 시간 복잡도를 한국어로 설명해 줘."),
    ("ko_code_regex", "code", "한국 휴대전화 번호(010-1234-5678 형식)를 찾는 정규식을 만들고, 각 부분이 무엇을 뜻하는지 설명해 줘."),
    ("ko_code_git", "code", "실수로 main 브랜치에 커밋한 변경을 새 브랜치로 옮기고 main은 원래대로 되돌리는 git 명령을 순서대로 알려 줘."),
    ("ko_code_explain_c", "code", "다음 C 코드가 출력하는 값을 예측하고 이유를 설명해 줘: int x = 5; int y = x++ + ++x; printf(\"%d %d\\n\", x, y);"),
    ("ko_creative_poem", "creative", "가을 저녁 버스 정류장을 배경으로 여섯 줄짜리 짧은 시를 써 줘."),
    ("ko_creative_story", "creative", "말을 할 줄 아는 고양이가 처음으로 편의점에 가는 이야기의 첫 문단을 써 줘."),
    ("ko_reason_coins", "reasoning", "500원, 100원, 50원 동전으로 1,000원을 만드는 방법은 모두 몇 가지인지 구하고 풀이를 보여 줘."),
    ("ko_reason_calendar", "reasoning", "2026년 3월 1일이 일요일이라면 같은 해 5월 5일은 무슨 요일인지 계산 과정을 보여 줘."),
    ("ko_struct_json", "structured", "한 주 식단표를 JSON으로만 출력해 줘. 각 날짜에 아침, 점심, 저녁 메뉴를 넣어."),
    ("ko_struct_table", "structured", "서울, 부산, 대구, 광주의 특징을 도시 이름, 대표 음식, 유명한 장소 열이 있는 마크다운 표로 정리해 줘."),
    ("ko_repeat_count", "repetitive", "1부터 40까지의 숫자를 한글로 쓴 다음(일, 이, 삼 ...), 같은 목록을 아라비아 숫자로 한 번 더 써 줘."),
    ("ko_repeat_copy", "repetitive", "다음 문장을 다섯 번 그대로 따라 쓰고, 마지막에 문장의 글자 수를 세어 줘: 오늘도 천천히 그러나 꾸준히 걷는다."),
    ("ko_instr_rules", "instruction", "동아리 첫 모임에서 쓸 규칙 다섯 가지를 존댓말로 써 줘. 각 규칙은 한 문장으로 끝내."),
    ("ko_instr_travel", "instruction", "부모님과 함께하는 1박 2일 경주 여행 일정을 시간대별로 짜 줘. 이동 거리가 너무 길지 않게 해 줘."),
]


def main() -> int:
    spec = json.loads((ROOT / "docs/models/qwen3.8-27b/artifacts/speculative-corpus.json").read_text())
    prompts = [{"id": p["id"], "category": p["category"], "text": p["text"],
                "source": "docs/models/qwen3.8-27b/artifacts/speculative-corpus.json"} for p in spec["prompts"]]
    d4 = json.loads((ROOT / "tools/qwen27b/rq_quality/d4_ko_prompts.json").read_text())
    for p in d4["prompts"]:
        if p["category"] == "tool_call":
            continue
        text = "\n\n".join(m["content"] for m in p["messages"])
        prompts.append({"id": "d4_" + p["id"].replace("-", "_"), "category": "d4_" + p["category"],
                        "text": text, "source": "tools/qwen27b/rq_quality/d4_ko_prompts.json (W2-authored)"})
    prompts += [{"id": i, "category": c, "text": t, "source": "W2-authored"} for i, c, t in W2]
    ids = [p["id"] for p in prompts]
    if len(prompts) != 64 or len(set(ids)) != len(ids):
        raise SystemExit(f"expected 64 unique prompts, got {len(prompts)} ({len(set(ids))} unique)")
    doc = {"schema": "gufo.speculative-corpus.v1", "version": "rq-accept-prompts 1",
           "note": "RQ DFlash2 text acceptance suite (W2 2026-10-29); built by accept_prompts_src.py; "
                   "manifest.overlap is written by accept.py check-overlap.",
           "prompts": prompts}
    json.dump(doc, sys.stdout, indent=1, ensure_ascii=False)
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
