# Flash-Next serving supervisor

The persistent Qwen3.8 Flash-Next server on the development host runs from
`~/.local/share/gufo/`. That directory holds runtime state (server-state.json,
logs/, caches, the boot-scoped admission manifests and the GPU owner lock). The
code that drives it is tracked here:

| Path | Role |
| --- | --- |
| `gufo-server.py` | Supervisor: preflight admission (image, model, binary sha256, no KFD compute owner, idle NPU, 80 GiB free memory, disk reserve), the pinned `docker run … gufo serve llm` command, readiness and runtime-identity checks, the host-memory stop guard, and log-only GPU diagnostics. Actions: `start`, `stop`, `status`, `logs`, `config` |
| `ops/window.py` | GPU owner lock (`ops/OWNER.lock`), host and NPU audits, the quiet check |
| `ops/root_read.py` → `ops/legacy_root_read.py` → `ops/host_audit.py` | Root-run, read-only GPU FD/KFD/telemetry inventory with stability brackets |
| `ops/admission.py`, `ops/desktop.py` | Exact boot-scoped classification of desktop and display-infrastructure GPU clients; never a KFD exemption |
| `ops/npu_audit.py` | Root-run NPU (`/dev/accel`) FD audit |
| `ops/fence.py`, `ops/fence_read.py` | GFX ring fence readings from DRM debugfs (root-run reader) and their pending-interval lower bound |
| `ops/journal.py` | Kernel journal delta since the server started; GPU errors are recorded |
| `install.py` | Installs a committed revision into `~/.local/share/gufo/` |
| `test_serving.py`, `fixtures/` | Hosted CPU tests (`flash_next_serving_test`, Linux only): fence readings and admission identities recorded on 2026-11-08 |

The files were copied from the retired ai.tools tree (2026-11-08) and
reformatted here. Every file's Python AST equals the copy that was live then,
so behaviour is unchanged. They use the standard library only, and nothing
imports or runs ai.tools code. Root-run scripts are executed from the
installed `ops/` as `sudo -n /usr/bin/python3 -I <path>` (the fence reader
under `sudo -n /usr/bin/timeout … /usr/bin/python3 -I`).

## The `llm` command

`~/.local/bin/llm` runs `/usr/local/bin/gufo`, which runs
`/usr/bin/python3 -B ~/.local/share/gufo/gufo-server.py "$@"`:

```sh
llm start     # preflight, then supervise in the background
llm status    # JSON status; does not touch the server
llm stop      # stop the supervisor and its labelled container
llm logs      # tail the current server log
llm config    # print the docker command line
```

## Install

The live server must never run code from a working tree that is being edited,
so `install.py` copies a committed revision rather than linking to the repository:

```sh
python3 -I tools/serving/flash_next/install.py --dry-run    # what would change
python3 -I tools/serving/flash_next/install.py              # install HEAD
python3 -I tools/serving/flash_next/install.py --rev <sha>  # install another commit
python3 -I tools/serving/flash_next/install.py --check      # installed files still match?
```

It copies `gufo-server.py` and `ops/*.py` out of git and byte-compiles them
before writing anything. It backs up the files it replaces to
`~/projects/_backup/gufo-serving-<UTC stamp>-<sha>/`, adds a line for that
folder to `~/projects/_backup/README.md`, writes each file atomically, and
records the commit and the sha256 of each file in
`~/.local/share/gufo/INSTALLED.json`. It never touches `ops/OWNER.lock`,
`ops/DESKTOP-*.json`, state, logs or caches.

Install with the server stopped (`llm stop`, install, `llm start`). A running
supervisor keeps the Python code it loaded at start but starts the root-run
readers from disk for each reading, so a live install mixes old and new code.
The installer therefore refuses while a supervisor runs, unless
`--allow-running` is given, which is safe only when the readers behave as
before. It holds `~/.local/share/gufo/gufo-control.lock`, the lock `llm start`
and `llm stop` take, while it writes. Nothing is backed up or written when no
file changes.

## Rollback

Copy the files from the backup folder named in `INSTALLED.json` (`backup`)
back into `~/.local/share/gufo/`, keeping their relative paths, then restart:

```sh
B=~/projects/_backup/gufo-serving-<stamp>-<sha>
cp "$B/gufo-server.py" ~/.local/share/gufo/
cp "$B"/ops/*.py ~/.local/share/gufo/ops/
llm stop && llm start
```

`install.py --rev <older sha>` does the same from git. Earlier server
configurations (the pre-c03 runtime and the state before the ai.tools
migration) are under `~/projects/_backup/gufo-server-20261108/`, as described in
that folder's README.
