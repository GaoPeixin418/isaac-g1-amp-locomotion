#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
把服务器上的项目资料拉回本地（防止关机/释放实例后文件丢失）。

备份对象（默认只拉小文件）：
    /data/logs           安装/训练日志
    /data/setup          服务器上的安装脚本
    /data/legged_lab/... 训练配置（环境装好后加入）

用法：
    python scripts/sync_from_server.py                  # 增量备份到 backup/server/<时间戳>
    python scripts/sync_from_server.py --all            # 连大文件(checkpoint)一起拉
    python scripts/sync_from_server.py --dirs /data/logs

设计原则：
    - 默认只拉 <50MB 的文件，大模型文件不占本地空间
    - 同名同大小的文件跳过，支持增量
"""
from __future__ import annotations

import argparse
import os
import stat
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BACKUP = ROOT / "backup" / "server"
DEFAULT_REMOTE_DIRS = ["/data/logs", "/data/setup"]
MAX_SIZE_DEFAULT = 50 * 1024 * 1024  # 50MB


def main() -> int:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from remote import load_conf, connect  # noqa: E402

    ap = argparse.ArgumentParser()
    ap.add_argument("--dirs", nargs="*", default=DEFAULT_REMOTE_DIRS)
    ap.add_argument("--all", action="store_true", help="不限制文件大小（含 checkpoint）")
    ap.add_argument("--max-size", type=int, default=MAX_SIZE_DEFAULT)
    args = ap.parse_args()

    limit = float("inf") if args.all else args.max_size
    stamp = time.strftime("%Y%m%d_%H%M%S")
    dest_root = BACKUP / stamp
    dest_root.mkdir(parents=True, exist_ok=True)

    cli = connect(load_conf())
    sftp = cli.open_sftp()
    n_dl = n_skip = n_big = 0

    def walk(remote_dir: str) -> None:
        nonlocal n_dl, n_skip, n_big
        try:
            items = sftp.listdir_attr(remote_dir)
        except (IOError, OSError):
            return
        for it in items:
            rpath = f"{remote_dir.rstrip('/')}/{it.filename}"
            if stat.S_ISDIR(it.st_mode):
                walk(rpath)
                continue
            if it.st_size is None or it.st_size > limit:
                n_big += 1
                continue
            local = dest_root / rpath.lstrip("/")
            local.parent.mkdir(parents=True, exist_ok=True)
            if local.exists() and local.stat().st_size == it.st_size:
                n_skip += 1
                continue
            try:
                sftp.get(rpath, str(local))
                n_dl += 1
            except (IOError, OSError) as e:
                print(f"  skip {rpath}: {e}")

    for d in args.dirs:
        print(f">>> sync {d}")
        walk(d)

    sftp.close()
    cli.close()
    print(f"done: {n_dl} downloaded, {n_skip} unchanged, {n_big} skipped(too large)")
    print(f"backup -> {dest_root}")
    (BACKUP / "latest.txt").write_text(stamp, encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
