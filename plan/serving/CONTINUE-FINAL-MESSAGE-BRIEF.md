# Continue a cut-off assistant turn in the same turn (brief, 2026-11-19)

The user tests models with long single-turn tasks (coding agents). When a
reply is cut (client `max_tokens`, disconnect, length), they want to resume
generation exactly where it stopped, inside the same assistant turn, not as a
new turn. Example: the true answer is `안녕하세요 1+1=2 입니다.`; it stopped
at `안녕하세요 1`; a resume request must continue with `+1=2 입니다.`, not
start a new reply. The same must work when the cut happened inside thinking.

Measured today on production (runtime-20261118c, Qwen3.8 27B):
- Chat Completions with a trailing `{"role":"assistant","content":"1 2 3 4"}`
  (count 1..10, thinking off) returned `1 2 3 4 5 6 7 8 9 10`: a new turn.
- `/v1/messages` with the same trailing assistant message opened a new turn
  and the model thought "I only gave 1 2 3 4, I need to provide the full
  answer". Anthropic Messages defines a trailing assistant message as a
  prefill that the reply continues, so this route is out of spec today.

No model tuning: render the conversation with the final assistant turn left
open (no `<|im_end|>`, no new generation prompt) so the token stream is the
same state the original generation was in.

## Items

1. **Chat Completions: `continue_final_message: true`** (vLLM's extension
   name; with it `add_generation_prompt` is implicitly false; accept an
   explicit `add_generation_prompt: false`, reject `continue_final_message`
   with `add_generation_prompt: true`, 400). Requires a last `assistant`
   message. Cases:
   - partial reasoning only (`reasoning_content` non-empty, `content`
     empty/absent): prompt ends `<|im_start|>assistant\n<think>\n` +
     reasoning text, thinking still open; generated tokens stream as
     `reasoning_content` until the model closes thinking, then as `content`.
     The reasoning budget (if any) counts the prior reasoning tokens too.
   - reasoning closed + partial content: prompt ends with the reasoning
     block closed exactly as the template renders a completed reasoning
     turn's prefix, then the partial content, turn open.
   - thinking disabled / no reasoning: the template's empty thinking block
     (`<think>\n\n</think>\n\n`) then the partial content.
   - partial tool call text in content: continue as text; the tool-call
     parser must see prior + new text (a `<tool_call>` split across the cut
     must still parse). If that is not bounded, document it and reject only
     the unsafe case.
   The response contains only the new tokens (as vLLM does). `usage`
   counts the prior assistant text as prompt tokens. Prefix cache: the
   continued request should hit the cached conversation (state for
   prompt + partial reply is usually already published or live); check the
   tokenization boundary issue (re-tokenizing the partial text may differ
   from the generated token ids at the cut) and say what it costs.
2. **Messages (`/v1/messages`): trailing assistant message = prefill** per
   Anthropic semantics: text blocks continue; a trailing `thinking` block
   (last block) continues thinking. Same renderer as item 1.
3. **Responses API**: only if it has an equivalent concept; otherwise leave.
4. Models: all Qwen-template adapters (Flash-Next, 27B, others sharing
   `src/models/qwen/chat_template.*`). Other families (DeepSeek, Gemma,
   MiniMax) return a clear 400 `unsupported_continue_final_message` unless
   trivially supported. Reuse the `score_assistant` split/render path
   (`PrepareQwenScoring`, `SplitScoringRequest`, docs/SERVER.md
   "Assistant-message scoring") where it fits; message text is tokenized as
   text, never as control tokens.
5. **Tool `tools/serving/opencode_resume.py`** (stdlib Python): given an
   OpenCode session id (`opencode export <id>` JSON on stdin or a file),
   rebuild the Chat Completions request OpenCode sent (system prompt, tools,
   messages; read OpenCode's export format and its request construction, do
   not guess; if the system prompt/tools are not in the export, say so and
   take them from a captured request instead, e.g. an optional server-side
   request dump if one exists) and the cut final assistant turn, send it
   with `continue_final_message: true`, stream to stdout, and save the
   joined turn. Keep it small; it is a test helper.

## Rules

- Do not touch production (`~/.local/share/gufo/`), the running server, or
  `~/projects/_ai.tools`. Never modify `tools/exl3_quant/hc/` or
  `plan/exl3/HC-Q8-CALIBRATION.md`. Never use `pgrep -f`/`pkill -f`.
- No GPU, model, session or timing runs (production is live on this GPU).
  Build with `build/b1/build.sh <targets>`; run CPU/unit tests with
  `build/b1/build.sh --run <cmd>`; format check `build/b1/checkfmt.sh`.
  Add unit tests: rendered token sequences for each case above (byte-exact
  against the template's completed-turn rendering minus the closing),
  request validation errors, streaming split of reasoning/content when
  resuming inside thinking, Messages prefill.
- Update docs/SERVER.md (Chat Completions extension + Messages prefill).
- Have a second reviewer (another Codex thread) review the diff, fix,
  re-review.
- Do not commit; leave the diff in the working tree. Report in English:
  the change, tests and results, what the GPU pass must check (exact
  continuation equivalence: a greedy reply cut at N tokens and resumed must
  equal the uncut greedy reply), open risks.
