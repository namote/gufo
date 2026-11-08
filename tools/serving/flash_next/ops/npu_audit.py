"""Root read-only NPU (/dev/accel) FD audit. Run: sudo -n /usr/bin/python3 -I npu_audit.py --audit-npu-only"""
import json
from pathlib import Path
import sys

def npu_fds():
    rows = []
    for proc in Path("/proc").iterdir():
        if not proc.name.isdigit():
            continue
        try:
            for fd in (proc / "fd").iterdir():
                try:
                    target = str(fd.readlink())
                except FileNotFoundError:
                    continue
                if target.startswith("/dev/accel/"):
                    rows.append({"PID": int(proc.name), "comm": (proc / "comm").read_text().strip(), "fd": fd.name, "target": target})
        except (FileNotFoundError, ProcessLookupError):
            continue
    return rows

def main():
    if sys.argv[1:] == ["--audit-npu-only"]:
        print(json.dumps(npu_fds()))
        return
    raise SystemExit("usage: --audit-npu-only")

if __name__ == "__main__":
    main()
