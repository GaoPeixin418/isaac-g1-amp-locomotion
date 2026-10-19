#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""上传本地文件到服务器。

用法：python scripts/upload.py <本地路径> <远程路径>
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import load_conf, connect  # noqa: E402


def main() -> int:
    if len(sys.argv) < 3:
        print("usage: upload.py <local> <remote>")
        return 2
    local = sys.argv[1]
    remote = sys.argv[2]
    cli = connect(load_conf())
    try:
        sftp = cli.open_sftp()
        sftp.put(local, remote)
        sftp.close()
        print(f"uploaded {local} -> {remote}")
    finally:
        cli.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
