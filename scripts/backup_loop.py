#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
实时备份循环：每 N 分钟从服务器拉回项目资料 -> git commit -> (若配了 remote) git push。

目的是双保险：
  - 服务器实例被释放/关机 -> 本地 backup/ 里有日志、脚本、配置
  - 本地误删             -> git 历史里有一份

用法：
    python scripts/backup_loop.py                 # 默认 600s 一轮
    python scripts/backup_loop.py --interval 300  # 5 分钟一轮

日志：tmp/backup_loop.log
"""
from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOG = ROOT / "tmp" / "backup_loop.log"


def log(msg: str) -> None:
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def git(*args: str) -> tuple[int, str]:
    p = subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True,
                       encoding="utf-8", errors="replace")
    return p.returncode, (p.stdout or "") + (p.stderr or "")


def has_changes() -> bool:
    code, out = git("status", "--porcelain")
    return bool(out.strip())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--interval", type=int, default=600, help="备份间隔(秒)")
    ap.add_argument("--no-push", action="store_true")
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from sync_from_server import main as sync_main  # noqa: E402

    log(f"backup loop started, interval={args.interval}s")

    while True:
        try:
            # 1) 从服务器拉回资料
            sys.argv = ["sync_from_server.py"]
            sync_main()
        except Exception as e:  # noqa: BLE001
            log(f"sync failed: {e}")

        # 2) git 提交
        if has_changes():
            git("add", "-A")
            code, out = git("commit", "-m", f"chore: 自动备份 {time.strftime('%m-%d %H:%M')}")
            log(f"commit rc={code}: {out.strip()[:120]}")
            # 3) 推送（没配 remote 就跳过）
            if not args.no_push:
                code, out = git("push")
                if code == 0:
                    log("push OK")
                elif "no upstream" in out or "No remote" in out or not out.strip():
                    pass  # 还没挂 remote，静默跳过
                else:
                    log(f"push failed rc={code}: {out.strip()[:200]}")
        else:
            log("no changes")

        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
