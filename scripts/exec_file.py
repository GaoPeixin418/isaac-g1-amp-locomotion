#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从文件读取命令，SSH 到服务器执行（避免 shell 转义问题）。

用法： python scripts/exec_file.py tmp/cmd.txt [timeout秒]
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import load_conf, connect  # noqa: E402


def main() -> int:
    cmd_file = sys.argv[1]
    timeout = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    out_file = Path(sys.argv[3]) if len(sys.argv) > 3 else None
    cmd = Path(cmd_file).read_text(encoding="utf-8").strip()

    lines: list[str] = []

    def emit(*parts: str) -> None:
        s = " ".join(parts)
        print(s)
        lines.append(s)

    cli = connect(load_conf())
    try:
        stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
        chan = stdout.channel
        out, err = [], []
        start = time.time()
        while True:
            while chan.recv_ready():
                out.append(chan.recv(65536).decode("utf-8", "replace"))
            while chan.recv_stderr_ready():
                err.append(chan.recv_stderr(65536).decode("utf-8", "replace"))
            if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
                break
            if time.time() - start > timeout:
                emit(f"!! timeout {timeout}s")
                break
            time.sleep(0.3)
        o = "".join(out)
        e = "".join(err)
        if o:
            emit(o)
        if e.strip():
            emit("[stderr]", e)
        code = chan.recv_exit_status() if chan.exit_status_ready() else -1
        emit(f"[exit={code}]")
        if out_file is not None:
            out_file.parent.mkdir(parents=True, exist_ok=True)
            out_file.write_text("\n".join(lines), encoding="utf-8")
        return code
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
