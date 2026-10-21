"""DeepMimic 训练后台监控。

教训（改了三版才稳）：
  1) 只看进程数会误判 —— 进程还在但日志不动的"假活"更常见
  2) 只看日志会误报 —— 本机 DNS 抖动产生的 traceback 混进输出，
     被当成训练崩溃误报过一次，白紧张
  所以这里用双判据：进程 OR 日志增长，两者同时满足才算活着。
"""
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paramiko

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONF = os.path.join(ROOT, ".secrets", "ssh.json")

REMOTE_LOG = "/data/logs/train_deepmimic.log"
DONE_MARK = "TRAIN DEEPMIMIC DONE"
CRASH_PAT = re.compile(r"Traceback \(most recent call last\)|RuntimeError:", re.I)

CHECK_SEC = 180
SILENCE_SEC = 900


def connect():
    conf = json.load(open(CONF, encoding="utf-8"))
    cli = paramiko.SSHClient()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        conf["host"], port=int(conf["port"]), username=conf["user"],
        password=conf["password"], timeout=30, banner_timeout=30,
        look_for_keys=False, allow_agent=False,
    )
    return cli


def run(cli, cmd, timeout=60):
    _, out, err = cli.exec_command(cmd, timeout=timeout)
    return out.read().decode("utf-8", "replace"), err.read().decode("utf-8", "replace")


def main():
    out_path = os.path.join(ROOT, "tmp", "monitor_deepmimic.txt")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with open(out_path, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log("监控启动")
    last_size = -1
    last_grow = time.time()

    while True:
        try:
            cli = connect()
        except Exception as exc:
            log(f"连接异常（本机网络抖动，不算失败）: {type(exc).__name__}")
            time.sleep(CHECK_SEC)
            continue

        try:
            out, err = run(cli, (
                f"grep -c 'Learning iteration' {REMOTE_LOG} 2>/dev/null; "
                f"grep -c '{DONE_MARK}' {REMOTE_LOG} 2>/dev/null; "
                f"stat -c %s {REMOTE_LOG} 2>/dev/null; "
                f"ps -eo cmd | grep -c '[t]rain.py'"
            ))
            parts = [p.strip() for p in out.strip().split("\n") if p.strip()]
            if len(parts) < 4:
                log(f"输出异常，稍后重试: {out[:200]}")
                cli.close()
                time.sleep(CHECK_SEC)
                continue

            iters = int(parts[0]) if parts[0].isdigit() else 0
            done = int(parts[1]) if parts[1].isdigit() else 0
            size = int(parts[2]) if parts[2].isdigit() else 0
            procs = int(parts[3]) if parts[3].isdigit() else 0

            # --- 只看最后一段，避免把本机连接噪声当崩溃 ---
            tail_out, _ = run(cli, f"tail -c 4000 {REMOTE_LOG}")
            crashed = bool(CRASH_PAT.search(tail_out)) and "paramiko" not in tail_out

            if size > last_size:
                last_grow = time.time()
                last_size = size

            if done > 0:
                log(f"训练完成！iter 共 {iters} 轮 -> 准备接评估")
                cli.close()
                return "DONE"
            if crashed:
                log(f"检测到崩溃，停在 iter {iters}")
                cli.close()
                return "CRASH"
            if procs == 0 and done == 0:
                log(f"进程已退出且无完成标志（停在 iter {iters}），需人工看日志")
                cli.close()
                return "EXIT_EARLY"
            if time.time() - last_grow > SILENCE_SEC:
                log(f"日志 {SILENCE_SEC//60} 分钟没增长（iter {iters}），可能卡住")
                cli.close()
                return "STALL"

            remain = max(0, 3000 - iters)
            eta = remain * 2.4 / 60
            log(f"iter {iters}/3000  reward推进中  剩余约 {eta:.0f} 分钟")
            cli.close()

        except Exception as exc:
            log(f"查询异常（继续重试）: {type(exc).__name__} {exc}")
        time.sleep(CHECK_SEC)


if __name__ == "__main__":
    r = main()
    print(f"MONITOR_RESULT={r}")
