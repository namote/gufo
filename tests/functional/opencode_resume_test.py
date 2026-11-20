"""Offline unit checks for the OpenCode resume helper; no servers or models."""

import copy
import importlib.util
import json
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "opencode_resume",
    Path(__file__).resolve().parents[2] / "tools/serving/opencode_resume.py",
)
resume = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(resume)


class ResumeTest(unittest.TestCase):
    def setUp(self):
        self.export = {"info": {"id": "ses_test"}, "messages": [
            {"info": {"role": "assistant"}, "parts": [
                {"type": "step-start"}, {"type": "reasoning", "text": "plan"},
                {"type": "text", "text": "안녕하세요 1"}]}]}
        self.request = {"model": "test", "tools": [{"type": "function"}],
                        "messages": [{"role": "system", "content": "exact system"},
                                     {"role": "user", "content": "Q"}],
                        "temperature": 0, "max_tokens": 17}

    def test_capture_is_preserved(self):
        original = copy.deepcopy(self.request)
        body = resume.rebuild(self.export, self.request, "ses_test")
        self.assertEqual(body["messages"][:-1], original["messages"])
        self.assertEqual(body["tools"], original["tools"])
        self.assertEqual(body["temperature"], 0)
        self.assertEqual(body["messages"][-1]["content"], "안녕하세요 1")
        self.assertEqual(body["messages"][-1]["reasoning_content"], "plan")
        self.assertTrue(body["continue_final_message"])
        self.assertFalse(body["add_generation_prompt"])
        self.assertEqual(self.request, original)

    def test_phase_is_part_presence_not_text_length(self):
        parts = self.export["messages"][-1]["parts"]
        parts[:] = [{"type": "reasoning", "text": " plan "}]
        opened = resume.rebuild(self.export, self.request, "ses_test")
        self.assertNotIn("content", opened["messages"][-1])
        parts.append({"type": "text", "text": ""})
        closed = resume.rebuild(self.export, self.request, "ses_test")
        self.assertEqual(closed["messages"][-1]["content"], "")
        parts.extend([{"type": "reasoning", "text": ""},
                      {"type": "text", "text": "later"}])
        with self.assertRaises(ValueError):
            resume.rebuild(self.export, self.request, "ses_test")

    def test_final_phase_without_content_delta(self):
        joined = {"role": "assistant", "reasoning_content": "old"}
        lines = [b'data: {"choices":[{"delta":{"reasoning_content":" more"}}]}\n',
                 b'\n', b'data: {"gufo":{"reasoning_open":true},"choices":[{"delta":{},"finish_reason":"length"}]}\n',
                 b'\n', b'data: [DONE]\n', b'\n']
        result = resume.consume(lines, joined, lambda _: None)
        self.assertTrue(result["reasoning_open"])
        self.assertIsNone(joined["content"])
        self.assertEqual(joined["reasoning_content"], "old more")
        lines = [b'data: {"gufo":{"reasoning_open":false},"choices":[{"delta":{},"finish_reason":"length"}]}\n',
                 b'\n', b'data: [DONE]\n', b'\n']
        result = resume.consume(lines, joined, lambda _: None)
        self.assertFalse(result["reasoning_open"])
        self.assertEqual(joined["content"], "")

    def test_content_after_open_reasoning_null(self):
        joined = {"role": "assistant", "reasoning_content": "old", "content": None}
        lines = [b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                 b'\n', b'data: {"gufo":{"reasoning_open":false},"choices":[{"delta":{},"finish_reason":"stop"}]}\n',
                 b'\n', b'data: [DONE]\n', b'\n']
        resume.consume(lines, joined, lambda _: None)
        self.assertEqual(joined["content"], "answer")

    def test_missing_context_and_tool_parts_rejected(self):
        with self.assertRaises(ValueError):
            resume.rebuild(self.export, {}, "ses_test")
        with self.assertRaises(ValueError):
            resume.rebuild(self.export, self.request, "wrong_session")
        self.export["messages"][-1]["parts"].append({"type": "tool"})
        with self.assertRaises(ValueError):
            resume.rebuild(self.export, self.request, "ses_test")

    def test_stream_joins_only_new_bytes(self):
        events = [
            {"choices": [{"delta": {"reasoning_content": " more"}}]},
            {"choices": [{"delta": {"content": "+1=2 입니다."}}]},
            {"choices": [{"delta": {}, "finish_reason": "stop"}]},
            {"choices": [], "usage": {"prompt_tokens": 100, "completion_tokens": 4}},
        ]
        lines = [b": keepalive\n", b"\n"]
        for event in events:
            lines.extend([("data: " + json.dumps(event) + "\n").encode(), b"\n"])
        lines.extend([b"data: [DONE]\n", b"\n"])
        joined = {"role": "assistant", "reasoning_content": "plan", "content": "안녕하세요 1"}
        emitted = []
        result = resume.consume(lines, joined, emitted.append)
        self.assertTrue(result["complete"])
        self.assertEqual(joined["content"], "안녕하세요 1+1=2 입니다.")
        self.assertEqual(joined["reasoning_content"], "plan more")
        self.assertEqual("".join(emitted), " more+1=2 입니다.")
        self.assertEqual(result["usage"]["prompt_tokens"], 100)

    def test_disconnect_retains_partial(self):
        joined = {"content": "before"}
        line = b'data: {"choices":[{"delta":{"content":" after"}}]}\n'
        with self.assertRaises(ValueError):
            resume.consume([line, b"\n"], joined, lambda _: None)
        self.assertEqual(joined["content"], "before after")


if __name__ == "__main__":
    unittest.main()
