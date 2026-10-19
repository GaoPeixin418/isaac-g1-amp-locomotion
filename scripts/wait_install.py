#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""轮询远程安装进度：每 120s 查一次 /data/logs/install.log，
成功(=== ENV OK ===)或失败(!! 标记 / 进程消失)时退出并打印结论。"""
import re
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import load_conf, connect  # noqa: E402

OK_MARK = "=== ENV OK ==="
FAIL_MARK = "!!"
MAX_MINUTES = 150
INTERVAL = 120


def check() -> tuple[str, str]:
    """返回 (状态, 日志尾部)"""
    cli = connect(load_conf())
    try:
        # 注意: 用字符类 [i]nstall_env 避免 grep/pgrep 匹配到执行这条命令的 shell 自身
        _stdin, stdout, _stderr = cli.exec_command(
            "tail -30 /data/logs/install.log; echo __PROC__; "
            "ps -eo cmd | grep -c '[i]nstall_env.sh'",
            timeout=30,
        )
        out = stdout.read().decode("utf-8", "replace")
    finally:
        cli.close()
    log_part, proc_part = out.split("__PROC__")
    if OK_MARK in log_part:
        return "OK", log_part
    if FAIL_MARK in log_part:
        return "FAIL", log_part
    # proc count == 0 表示安装进程已退出（未完成 => 异常）
    try:
        alive = int(proc_part.strip().splitlines()[0]) > 0
    except (ValueError, IndexError):
        alive = True
    if not alive:
        return "DEAD", log_part
    return "RUNNING", log_part


def main() -> int:
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
            step = ""
            m = re.findall(r"\[([\d\- :]+)\] (Step \d/\d[^\n]*)", tail)
            if m:
                step = f"{m[-1][0]} {m[-1][1]}"
            print(f"[poll] still running... {step or tail.strip().splitlines()[-1:]}")
        except Exception as e:  # noqa: BLE001
            print(f"[poll] transient error: {e}")
        time.sleep(INTERVAL)
    print("INSTALL_TIMEOUT")
    print(last_tail)
    return 2


if __name__ == "__main__":
    sys.exit(main())
