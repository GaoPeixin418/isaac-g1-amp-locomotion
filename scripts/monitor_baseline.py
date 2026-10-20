#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
训练后台监控：每 5 分钟 SSH 查一次训练日志，直到出现完成标志或真实报错。

误报修正（2026-10-20 19:41 事故）：
    上一版把「本地 SSH 连接失败时 remote.py 自己打印的 Python traceback」
    当成了训练日志里的错误，导致 getaddrinfo 瞬时失败被误判为训练崩溃。
    修正两点：
      1. 用哨兵行包裹远端输出，只解析哨兵之间的内容；
      2. 显式识别连接类异常（gaierror / AuthenticationException / SSHException /
         socket.timeout / NoValidConnectionsError），计为 conn_err 而非 failed。

判活依据（只看远端日志内容，不看进程数）：
    - 完成：哨兵区内出现 "TRAIN RESUME DONE"
    - 失败：哨兵区内出现 "Traceback" 或 "RuntimeError"
    - 连续 2 次 iter 不增长 -> 记 stalled 警示（不直接判死，避免误杀）

用法：
    python scripts/monitor_baseline.py [LOG_PATH]
"""

import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\GPX\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
REMOTE = Path(__file__).resolve().parent / "remote.py"
OUT = Path(__file__).resolve().parent.parent / "tmp" / "baseline_monitor.txt"

DEFAULT_LOG = "/data/logs/train_baseline_ppo.log"
BEG, END = "@@MON_BEGIN@@", "@@MON_END@@"

CONN_ERRS = (
    "getaddrinfo failed",
    "AuthenticationException",
    "NoValidConnectionsError",
    "SSHException",
    "socket.timeout",
    "timed out",
    "Connection refused",
    "Network is unreachable",
)

INTERVAL = 300          # 每 5 分钟查一次
MAX_CHECKS = 60         # 60 次 = 5 小时上限


def build_cmd(log: str) -> str:
    return (
        f"echo {BEG}; "
        f"tail -c 3000 {log} | grep -E 'TRAIN RESUME DONE|Traceback|RuntimeError' | tail -3; "
        f"echo '---ITER---'; grep -c 'Learning iteration' {log}; "
        f"echo '---ALIVE---'; ps -eo cmd | grep -c '[t]rain.py'; "
        f"echo {END}"
    )


def check(log: str) -> tuple[str, bool]:
    """返回 (远端输出片段, 是否连接类错误)"""
    r = subprocess.run(
        [PY, str(REMOTE), build_cmd(log), "--timeout", "40"],
        capture_output=True, text=True, timeout=120, encoding="utf-8", errors="replace",
    )
    raw = (r.stdout or "") + (r.stderr or "")
    if BEG in raw and END in raw:
        body = raw.split(BEG, 1)[1].split(END, 1)[0]
        return body, False
    # 没拿到完整远端输出：先看是不是已知的连接类异常
    return raw, any(e in raw for e in CONN_ERRS)


def main() -> int:
    log = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_LOG
    OUT.parent.mkdir(parents=True, exist_ok=True)
    prev_iters = -1
    stalled = 0

    for i in range(MAX_CHECKS):
        time.sleep(INTERVAL)
        try:
            out, conn_err = check(log)
        except Exception as e:
            conn_err, out = True, str(e)

        if conn_err:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] check#{i + 1} CONN_ERROR (忽略)\n")
            continue

        iters, alive = -1, -1
        for line in out.splitlines():
            s = line.strip()
            if s.isdigit():
                if iters < 0:
                    iters = int(s)
                elif alive < 0:
                    alive = int(s)
        done = "TRAIN RESUME DONE" in out
        failed = ("Traceback" in out) or ("RuntimeError" in out)

        if iters == prev_iters:
            stalled += 1
        else:
            stalled = 0
        prev_iters = iters

        with OUT.open("a", encoding="utf-8") as f:
            f.write(
                f"[{time.strftime('%H:%M:%S')}] check#{i + 1} iters={iters} "
                f"proc={alive} done={done} failed={failed} stalled={stalled}\n"
            )

        if done or failed:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(f"=== FINAL ({'DONE' if done else 'FAILED'}) ===\n{out[-2000:]}\n")
            print(f"BASELINE {'DONE' if done else 'FAILED'} at iter {iters}")
            return 0

        if stalled >= 2 and alive == 0:
            with OUT.open("a", encoding="utf-8") as f:
                f.write("=== FINAL (STALLED & NO PROC) ===\n")
            print(f"BASELINE DEAD at iter {iters}")
            return 1

    print("BASELINE MONITOR TIMEOUT")
    return 1


if __name__ == "__main__":
    sys.exit(main())
