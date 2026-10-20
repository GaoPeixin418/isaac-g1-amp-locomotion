#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
基线训练后台监控：每 5 分钟 SSH 查一次训练日志，直到出现完成标志或报错。

设计要点（继承自 wait_install.py 的教训）：
- 不依赖进程数判活（会误报），只看日志内容：
  - 完成：日志尾部出现 "TRAIN RESUME DONE"
  - 失败：出现 Python "Traceback" 或显式 "RuntimeError"
- 完成/失败时立即退出并写结果文件，外层后台任务结束时宿主会收到通知

用法：
    python scripts/monitor_baseline.py   # 前台跑；配合 run_in_background 用
"""
import subprocess
import sys
import time
from pathlib import Path

PY = r"C:\Users\GPX\.workbuddy\binaries\python\envs\default\Scripts\python.exe"
REMOTE = Path(__file__).resolve().parent / "remote.py"
OUT = Path(__file__).resolve().parent.parent / "tmp" / "baseline_monitor.txt"

LOG = "/data/logs/train_baseline_ppo.log"
CMD = (
    f"tail -c 3000 {LOG} | grep -E 'TRAIN RESUME DONE|Traceback|RuntimeError' | tail -3; "
    f"echo '---ITER---'; grep -c 'Learning iteration' {LOG}"
)

INTERVAL = 300          # 每 5 分钟查一次
MAX_CHECKS = 40         # 40 次 = 3.3 小时上限


def check() -> str:
    r = subprocess.run(
        [PY, str(REMOTE), CMD, "--timeout", "30"],
        capture_output=True, text=True, timeout=90, encoding="utf-8", errors="replace",
    )
    return (r.stdout or "") + (r.stderr or "")


def main() -> int:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    for i in range(MAX_CHECKS):
        time.sleep(INTERVAL)
        try:
            out = check()
        except Exception as e:  # SSH 偶发抖动不中断监控
            with OUT.open("a", encoding="utf-8") as f:
                f.write(f"[{time.strftime('%H:%M:%S')}] check error: {e}\n")
            continue
        iters = "unknown"
        for line in out.splitlines():
            if line.strip().isdigit():
                iters = line.strip()
        done = "TRAIN RESUME DONE" in out
        failed = ("Traceback" in out) or ("RuntimeError" in out)
        with OUT.open("a", encoding="utf-8") as f:
            f.write(f"[{time.strftime('%H:%M:%S')}] check#{i + 1} iters={iters} done={done} failed={failed}\n")
        if done or failed:
            with OUT.open("a", encoding="utf-8") as f:
                f.write(f"=== FINAL ({'DONE' if done else 'FAILED'}) ===\n{out[-2000:]}\n")
            print(f"BASELINE {'DONE' if done else 'FAILED'} at iter {iters}")
            return 0
    print("BASELINE MONITOR TIMEOUT (3.3h)")
    return 1


if __name__ == "__main__":
    sys.exit(main())
