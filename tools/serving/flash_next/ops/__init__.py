"""Self-contained gufo serving ops (2026-11-08).

GPU owner lock, root read-only host/NPU audits, kernel journal delta and GFX
fence monitor used by ../gufo-server.py. Copied once from the retired ai.tools
project with only path constants changed; nothing here imports or runs
ai.tools code. Root-run scripts: root_read.py (-> legacy_root_read.py ->
host_audit.py), npu_audit.py, fence_read.py, each via
`sudo -n /usr/bin/python3 -I <path>`.
"""
