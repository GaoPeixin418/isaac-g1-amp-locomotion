#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轮询远程安装进度，成功(=== ENV OK ===)或失败(!! / 进程消失)时退出并打印结论。

用法：
    python scripts/wait_install.py                     # 默认盯 install.log
    python scripts/wait_install.py install_lab.log     # 盯 IsaacLab 安装
    python scripts/wait_install.py install_lab.log 04_install_isaaclab
"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import load_conf, connect  # noqa: E402

FAIL_MARK = "!!"
INTERVAL = 90

LOG_NAME = sys.argv[1] if len(sys.argv) > 1 else "install.log"
PROC_KEY = sys.argv[2] if len(sys.argv) > 2 else "install_env.sh"
OK_MARK = sys.argv[3] if len(sys.argv) > 3 else "=== ENV OK ==="
MAX_MINUTES = int(sys.argv[4]) if len(sys.argv) > 4 else 180


def grep_bracket(key: str) -> str:
    """把 install_isaaclab -> [i]nstall_isaaclab，避免 grep 匹配到执行它的 shell 自身"""
    return "[" + key[0] + "]" + key[1:]


def check() -> tuple[str, str]:
    cli = connect(load_conf())
    try:
        cmd = (
            f"tail -30 /data/logs/{LOG_NAME} 2>/dev/null; echo __PROC__; "
            f"ps -eo cmd | grep -c '{grep_bracket(PROC_KEY)}'; echo __MTIME__; "
            f"expr $(date +%s) - $(stat -c %Y /data/logs/{LOG_NAME} 2>/dev/null || echo $(date +%s))"
        )
        _stdin, stdout, _stderr = cli.exec_command(cmd, timeout=30)
        out = stdout.read().decode("utf-8", "replace")
    finally:
        cli.close()
    if "__PROC__" not in out:
        return "RUNNING", out
    log_part, rest = out.split("__PROC__", 1)
    if OK_MARK in log_part:
        return "OK", log_part
    if FAIL_MARK in log_part:
        return "FAIL", log_part
    # 判据 1：进程数
    alive = True
    mtime = None
    if "__MTTIME__" in rest:
        proc_part, mtime_part = rest.split("__MTIME__", 1)
    else:
        proc_part, mtime_part = rest, ""
    try:
        alive = int(proc_part.strip().splitlines()[0]) > 0
    except (ValueError, IndexError):
        alive = True
    try:
        mtime = int(mtime_part.strip().splitlines()[0])
    except (ValueError, IndexError):
        mtime = None
    # 判据 2：日志 mtime 超过 5 分钟没更新 → 进程僵死/已退出但没留标记
    stale = mtime is not None and mtime > 300
    if not alive or stale:
        return "DEAD", log_part
    return "RUNNING", log_part


def main() -> int:
    print(f"watching /data/logs/{LOG_NAME} (proc key={PROC_KEY})")
    start = time.time()
    last_tail = ""
    while time.time() - start < MAX_MINUTES * 60:
        try:
            status, tail = check()
            last_tail = tail
            if status == "OK":
                print("INSTALL_OK")
                print(tail)
                return 0
            if status in ("FAIL", "DEAD"):
                print(f"INSTALL_{status}")
                print(tail)
                return 1
            m = re.findall(r"\[([\d\- :]+)\] (Step \d/\d[^\n]*)", tail)
            step = f"{m[-1][0]} {m[-1][1]}" if m else ""
            print(f"[poll] still running... {step or (tail.strip().splitlines()[-1:] or [''])[0]}")
        except Exception as e:  # noqa: BLE001
            print(f"[poll] transient error: {e}")
        time.sleep(INTERVAL)
    print("INSTALL_TIMEOUT")
    print(last_tail)
    return 2


if __name__ == "__main__":
    sys.exit(main())
