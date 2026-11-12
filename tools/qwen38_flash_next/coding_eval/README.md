# R5 coding evaluation

Tools for the R5 "careful coding" steering set (see `plan/research/IDEAS.md`,
R5): Python function problems with hidden edge-case tests, the model's own
answers, and a pass-rate comparison between settings. The problem set itself is
data and lives outside the repository (`build/b1/r5-coding/`).

| Path | Role |
| --- | --- |
| `sandbox.py` | Problem loading and validation, code-block extraction, the Docker sandbox and the per-answer score |
| `harness.py` | Runs inside the sandbox: a judge that never runs answer code, and one fresh child process per load and per test that returns only a plain-data encoding of the result |
| `generate.py` | Asks a running server for the model's own answers (greedy, thinking off, 600 tokens, abliteration off), resumable, then scores them |
| `evaluate.py` | Rescores generation JSONL files and prints the comparison tables: all-tests, edge-only and ordinary-only pass rates, paired fixed/broken problems |
| `check_references.py` | Checks that every canonical solution and reference answer passes its hidden tests and its own `__main__` self-checks |
| `test_coding_eval.py` | Hosted CPU tests (`flash_next_coding_eval_test`); no Docker or server |

## Isolation

Model-written code runs only through `sandbox.DockerRunner`: one container per
answer with `--init`, `--network none`, `--read-only`, a 16 MiB noexec `/tmp`,
1 GiB memory without swap, one CPU, 64 PIDs, no capabilities,
`no-new-privileges` and user 65534. The host kills the container at a
wall-clock limit.

Inside, the judge starts a fresh `python3 -I harness.py --child` per step
(load, optional `__main__` run, each test) with 2 s, a CPU-time limit and a
512 MiB address space, and kills its process group afterwards. The child gets
only the code, the entry point and the argument expression, never the
expected outcome. It reports the returned value and the arguments after the
call as type-tagged plain data (exact standard types only; anything else, such
as an object with a permissive `__eq__`, becomes an opaque value equal to
nothing), or the raised exception's qualified class names. The judge decodes
that and evaluates `expect`, `raises` and `check` itself, so forged child
output still has to carry the right value. The judge is non-dumpable (other
processes of the user cannot open its `/proc` descriptors) and signs its
report with a host nonce; other output is ignored. The image is used with `--pull never`;
the default is the local `python:3.12-slim` digest, and `--image` selects
another local image with `python3`. `LocalRunner` runs unsandboxed and is
only for our own reference code and the unit tests.

## Extraction

The last closed Python code block that defines the entry point is the
answer; a truncated final block counts only when no closed block defines it.
Without a defining block, all Python blocks are joined; without blocks, the
whole text is used when it parses. Top-level code in the block (examples,
asserts) runs on load, as it would for a user; a failing example fails the
answer.

## Test format

A problem's `tests` hold Python expressions evaluated by the judge (the
arguments are also evaluated in the child, for the call): `args` (a tuple), then exactly one of `expect` (compared with
`==`; lists, tuples, dicts, sets and strings must also match in kind),
`raises` (`"ValueError|TypeError"`; the raised exception's most derived
builtin class must be one of them or a subclass) or `check` (an expression
over `result` and `args`, the arguments as they were after the call). `approx`
compares floats with a 1e-9 tolerance, `no_mutate` fails an answer that
changes its arguments, `scramble_args` changes every list, dict and set in the
arguments after the call so a result that shares objects with them fails, and
`edge` names the edge case a test probes (null for an ordinary case).

Stored scores carry `harness`, a hash of `harness.py`; `evaluate.py
--use-stored` rescores any line scored by another version.
