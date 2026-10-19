#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
远程执行工具：通过 paramiko 在云服务器上跑命令（密码认证，Windows 无 sshpass 的替代）。

用法：
    python scripts/remote.py "hostname"                    # 单条命令
    python scripts/remote.py -f setup/01_install_env.sh    # 上传脚本并执行
    python scripts/remote.py --status                      # 查看后台任务日志尾部

连接信息从 .secrets/ssh.json 读取（已 gitignore，不要提交）：
    {"host": "jq1.9gpu.com", "port": 12250, "user": "root", "password": "..."}
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import paramiko

ROOT = Path(__file__).resolve().parent.parent
SECRET = ROOT / ".secrets" / "ssh.json"


def load_conf() -> dict:
    if not SECRET.exists():
        print(f"!! 找不到 {SECRET}")
        sys.exit(1)
    return json.loads(SECRET.read_text(encoding="utf-8"))


def connect(conf: dict) -> paramiko.SSHClient:
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        conf["host"], port=int(conf["port"]), username=conf["user"],
        password=conf["password"], timeout=20, banner_timeout=20,
        look_for_keys=False, allow_agent=False,
    )
    return cli


def run(cli: paramiko.SSHClient, cmd: str, timeout: int = 120) -> int:
    stdin, stdout, stderr = cli.exec_command(cmd, timeout=timeout)
    chan = stdout.channel
    out_buf, err_buf = [], []
    start = time.time()
    while True:
        while chan.recv_ready():
            out_buf.append(chan.recv(65536).decode("utf-8", "replace"))
        while chan.recv_stderr_ready():
            err_buf.append(chan.recv_stderr(65536).decode("utf-8", "replace"))
        if chan.exit_status_ready() and not chan.recv_ready() and not chan.recv_stderr_ready():
            break
        if time.time() - start > timeout:
            print(f"!! 命令超时 ({timeout}s)，已返回的部分输出如下")
            break
        time.sleep(0.2)
    text = "".join(out_buf)
    err = "".join(err_buf)
    if text:
        print(text, end="" if text.endswith("\n") else "\n")
    if err:
        print("[stderr]", err, end="" if err.endswith("\n") else "\n")
    code = chan.recv_exit_status() if chan.exit_status_ready() else -1
    return code


def upload_and_run(cli: paramiko.SSHClient, local: str, remote_path: str, bash_args: str = "") -> int:
    sftp = cli.open_sftp()
    sftp.put(local, remote_path)
    sftp.close()
    print(f">>> uploaded {local} -> {remote_path}")
    return run(cli, f"bash {remote_path} {bash_args}", timeout=7200)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", nargs="?", help="要执行的命令")
    ap.add_argument("-f", "--file", help="上传本地脚本到服务器执行")
    ap.add_argument("--remote-path", default="/root/job.sh")
    ap.add_argument("--args", default="", help="传给脚本的参数")
    ap.add_argument("--timeout", type=int, default=120)
    args = ap.parse_args()

    conf = load_conf()
    cli = connect(conf)
    try:
        if args.file:
            return upload_and_run(cli, args.file, args.remote_path, args.args)
        if not args.cmd:
            ap.error("给一条命令或 -f 脚本")
        return run(cli, args.cmd, timeout=args.timeout)
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
