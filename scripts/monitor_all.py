#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
统一监控：同时盯「评估链」和「PPO 基线训练」两条线，全部完成后退出。

继承 monitor_baseline.py 的教训：
  - 本地 SSH/DNS 抖动产生的 Python traceback 不能被当成远端训练崩溃
  - 用哨兵行包裹远端输出，只解析哨兵区

用法：
    python scripts/monitor_all.py
"""
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\GPX\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
REMOTE = Path(__file__).resolve().parent / "remote.py"
OUT = Path(__file__).resolve().parent.parent / "tmp" / "monitor_all.txt"

BEG, END = "@@MB@@", "@@ME@@"

CONN_ERRS = (
    "getaddrinfo failed", "AuthenticationException", "NoValidConnectionsError",
    "SSHException", "socket.timeout", "timed out", "Connection refused",
    "Network is unreachable",
)

# (名称, 日志路径, 完成标记)
TARGETS = [
    ("evals", "/data/logs/all_evals.log", "ALL EVALS DONE"),
    ("ppo_baseline", "/data/logs/train_ppo_baseline.log", "TRAIN RESUME DONE"),
]

INTERVAL = 300
MAX_CHECKS = 72          # 72 × 5min = 6h 上限


def build_cmd() -> str:
    parts = [f"echo {BEG}"]
    for name, log, _ in TARGETS:
        parts.append(f"echo '##{name}'; grep -c 'Learning iteration' {log} 2>/dev/null; "
                     f"tail -c 1500 {log} 2>/dev/null | grep -oE "
                     f"'ALL EVALS DONE|TRAIN RESUME DONE|Traceback|RuntimeError' | sort -u | tr '\\n' ' '; "
                     f"echo; ps -eo cmd | grep -c '[{log[5]}]{log[6:]}'")
    parts.append(f"echo {END}")
    return "; ".join(parts)


def check() -> tuple[str, bool]:
    r = subprocess.run(
        [PY, str(REMOTE), build_cmd(), "--timeout", "40"],
        capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
    )
    raw = (r.stdout or "") + (r.stderr or "")
    if BEG in raw and END in raw:
        return raw.split(BEG, 1)[1].split(END, 1)[0], False
    return raw, any(e in raw for e in CONN_ERRS)


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {name: False for name, _, _ in TARGETS}
    last = {}

    for i in range(MAX_CHECKS):
        time.sleep(INTERVAL)
        try:
            out, conn_err = check()
        except Exception as e:
            conn_err, out = True, str(e)
        if conn_err:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] CONN_ERROR (忽略)\n")
            continue

        with OUT.open("a", encoding="utf-8") as f:
            f.write(f"--- check#{i + 1} {time.strftime('%H:%M:%S')} ---\n{out}\n")

        for name, _, mark in TARGETS:
            if mark in out:
                done[name] = True

        last = dict(done)
        if all(done.values()):
            print("ALL TARGETS DONE: " + ", ".join(done.keys()))
            return 0

    print("MONITOR TIMEOUT, done=" + str(last))
    return 1


if __name__ == "__main__":
    sys.exit(main())
