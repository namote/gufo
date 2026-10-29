#!/usr/bin/env python3
"""imatrix calibration corpus builder: conversion, packing and exclusion.

Pure functions only; no tokenizer, model or dataset files are needed.
Run: python3 -I tests/tools/rotquant_calibration_test.py
"""
from __future__ import annotations

from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))

from rotquant import calibration  # noqa: E402

TOOLS_JSON = ('[{"type": "function", "function": {"name": "get_price", '
              '"parameters": {"type": "object"}}}]')


def row(*turns):
    return [{"role": r, "content": c} for r, c in turns]


class HermesTest(unittest.TestCase):
    def test_tool_conversation(self):
        system = ("XML 태그 <tools></tools> 안에 함수가 있습니다. 사용 가능한 도구:<tools> "
                  "[{'type': 'function', 'function': {'name': 'get_price'}}] </tools>")
        messages, tools = calibration.hermes_messages(row(
            ("system", system), ("user", "가격 알려 줘"),
            ("assistant", "<tool_call>\n{'name': 'get_price', 'arguments': {'item': '사과'}}\n"
                          "</tool_call>"),
            ("tool", "<tool_response>\n{'price': 3}\n</tool_response>"),
            ("assistant", "사과는 3원입니다.")))
        self.assertEqual(tools, [{"type": "function", "function": {"name": "get_price"}}])
        self.assertEqual(messages[0], {"role": "system", "content": calibration.TOOL_SYSTEM})
        self.assertEqual(messages[2]["content"], "")
        self.assertEqual(messages[2]["tool_calls"], [{"type": "function", "function": {
            "name": "get_price", "arguments": {"item": "사과"}}}])
        self.assertEqual(messages[3], {"role": "tool", "content": "{'price': 3}"})

    def test_escaped_newlines_and_json(self):
        messages, _ = calibration.hermes_messages(row(
            ("system", f"<tools>{TOOLS_JSON}</tools>"), ("user", "q"),
            ("assistant", '<tool_call>\\n{"arguments": {}, "name": "get_price"}\\n</tool_call>')))
        self.assertEqual(messages[-1]["tool_calls"][0]["function"]["name"], "get_price")

    def test_plain_conversation_keeps_its_system(self):
        messages, tools = calibration.hermes_messages(row(
            ("system", "JSON으로 답하세요."), ("user", "q"), ("assistant", "{}")))
        self.assertIsNone(tools)
        self.assertEqual(messages[0]["content"], "JSON으로 답하세요.")

    def test_rejections(self):
        cases = {
            "tools do not parse": row(("system", "<tools>[{broken</tools>"),
                                      ("user", "q"), ("assistant", "a")),
            "tool call without a known name and arguments": row(
                ("system", f"<tools>{TOOLS_JSON}</tools>"), ("user", "q"),
                ("assistant", "<tool_call>{'name': 'other', 'arguments': {}}</tool_call>")),
            "unknown role 도우미": row(("user", "q"), ("도우미", "a")),
            "conversation does not end with an assistant turn": row(
                ("user", "q"), ("assistant", "a"), ("user", "q2")),
            "unbalanced tool call tags": row(("user", "q"), ("assistant", "<tool_call>{")),
        }
        for reason, conversation in cases.items():
            with self.subTest(reason), self.assertRaisesRegex(ValueError, reason):
                calibration.hermes_messages(conversation)


class ExtendSourceTest(unittest.TestCase):
    def test_reasoning_tool_row(self):
        tools = '[{"name": "get_price", "parameters": {}}]'
        conversation = [
            {"from": "system", "value": "You are a function calling AI model. <tools>…</tools>"},
            {"from": "human", "value": "price?"},
            {"from": "gpt", "value": "<think>\nneed the tool\n</think>\n\n<tool_call>\n"
                                     '{"name": "get_price", "arguments": {"item": "x"}}\n</tool_call>'},
            {"from": "tool", "value": "<tool_response>\n{\"price\": 3}\n</tool_response>"},
            {"from": "gpt", "value": "<think>\ndone\n</think>\n\nIt costs 3."},
            {"from": "human", "value": "and y?"},
            {"from": "gpt", "value": '<tool_call>\n{"name": "get_price", "arguments": {}}\n'
                                     "</tool_call>"},
            {"from": "tool", "value": "<tool_response>\n{}\n</tool_response>"},
        ]
        messages, out_tools = calibration.reasoning_tool_messages(
            {"tools": tools, "conversations": conversation})
        self.assertEqual(out_tools, [{"type": "function", "function": {
            "name": "get_price", "parameters": {}}}])
        self.assertEqual([m["role"] for m in messages],
                         ["user", "assistant", "tool", "assistant", "user", "assistant"])
        self.assertEqual(messages[1]["reasoning_content"], "need the tool")
        self.assertEqual(messages[1]["content"], "")
        self.assertEqual(messages[3], {"role": "assistant", "reasoning_content": "done",
                                       "content": "It costs 3."})
        self.assertEqual(messages[5]["tool_calls"][0]["function"]["arguments"], {})

    def test_ultrachat_row(self):
        messages, tools = calibration.ultrachat_messages(
            {"messages": [{"role": "user", "content": " q "}, {"role": "assistant", "content": "a"}]})
        self.assertIsNone(tools)
        self.assertEqual(messages[0]["content"], "q")
        with self.assertRaises(ValueError):
            calibration.ultrachat_messages({"messages": [{"role": "user", "content": "q"}]})

    def test_source_lines(self):
        text = "def f():\\n\\treturn 1\n\nsecond\n"
        self.assertEqual(calibration.source_lines(text, "code"), ["def f():\n\treturn 1", "second"])
        self.assertEqual(calibration.source_lines("a\\nb\n", "math"), ["a\\nb"])


class TextTest(unittest.TestCase):
    def test_wiki_tail_is_cut(self):
        text = "본문 첫 줄\n\n\n\n둘째 줄\n\n각주\n\n외부 링크\n분류"
        self.assertEqual(calibration.clean_wiki(text), "본문 첫 줄\n\n둘째 줄")

    def test_korean_prose(self):
        self.assertTrue(calibration.korean_prose("한국어 문장입니다. " * 40, 200))
        self.assertFalse(calibration.korean_prose("한국어 " + "English words " * 200, 2))
        self.assertFalse(calibration.korean_prose("짧은 글", 200))


class PackingTest(unittest.TestCase):
    def test_plan_takes_the_share(self):
        counts = calibration.plan(583, 0.3, 0.25)
        korean = counts["ko_wiki"] + counts["ko_web"] + counts["ko_chat"]
        self.assertEqual(korean, 250)
        self.assertEqual((counts["ko_chat"], counts["ko_wiki"], counts["ko_web"]), (62, 94, 94))
        self.assertEqual(calibration.plan(10, 0.0, 0.25)["ko_chat"], 0)

    def test_split_stream_pads_the_tail(self):
        self.assertEqual(calibration.split_stream([1, 2, 3, 4, 5], 2, 0),
                         [[1, 2], [3, 4], [5, 0]])

    def test_pack_starts_chunks_at_documents(self):
        docs = [("a", [1] * 3), ("b", [2] * 10), ("c", [3] * 2), ("d", [4] * 9)]
        chunks, used = calibration.pack(iter(docs), 2, 6, 8, 0)
        # a (3 + pad) then b truncated to fill; c starts chunk 1, d fills it.
        self.assertEqual(chunks, [[1, 1, 1, 0, 2, 2], [3, 3, 0, 4, 4, 4]])
        self.assertEqual([u["id"] for u in used], ["a", "b", "c", "d"])
        self.assertEqual(used[1]["tokens_available"], 8)
        self.assertEqual(used[1]["tokens_used"], 2)

    def test_pack_fails_when_documents_run_out(self):
        with self.assertRaisesRegex(ValueError, "run out"):
            calibration.pack(iter([("a", [1, 2])]), 2, 4, 8, 0)

    def test_pack_spans_chunks(self):
        docs = [("a", [1] * 7), ("b", [2] * 3)]
        chunks, used = calibration.pack(iter(docs), 3, 4, 100, 0, span=True, sep=[9])
        self.assertEqual(chunks, [[1, 1, 1, 1], [1, 1, 1, 9], [2, 2, 2, 9]])
        self.assertEqual([u["tokens_used"] for u in used], [8, 4])

    def test_mix_counts_sum_to_total(self):
        counts = calibration.mix_counts(3073, calibration.EXTEND_MIX)
        self.assertEqual(sum(counts.values()), 3073)
        self.assertEqual(counts["en_tool"], 1383)

    def test_overlap(self):
        excluded = list(range(100))
        self.assertEqual(calibration.overlap([[500, 501, 502, 503]], excluded, 3), 0)
        self.assertEqual(calibration.overlap([[7, 10, 11, 12, 13]], excluded, 3), 2)


if __name__ == "__main__":
    unittest.main()
