# -*- coding: utf-8 -*-
"""
等最后两条线跑完：λ=1.0 PPO 基线训练 -> 收尾链（评估 + 两段视频）。

教训（上一版踩过）：本地 DNS 抖动会让 paramiko 抛 traceback，
旧版监控把这段本地 traceback 当成"训练崩溃"误报了一次。
所以这里显式区分「连接失败」和「远端日志里的报错」：
连接失败只重试，不当成任务失败。
"""
import re
import sys
import time

sys.path.insert(0, __file__.rsplit("\\", 1)[0])
from remote import load_conf, connect  # noqa: E402

CHECK_EVERY = 300          # 5 分钟查一次
MAX_HOURS = 4
CONN_ERR_PAT = re.compile(r"getaddrinfo|Error reading SSH|timed out|EOFError|No such file")

CMD = (
    "echo ==PPO_DONE==; grep -c 'TRAIN RESUME DONE' /data/logs/train_ppo_baseline.log 2>/dev/null; "
    "echo ==FIN_DONE==; grep -c 'FINISH PPO DONE' /data/logs/finish_ppo.log 2>/dev/null; "
    "echo ==ITER==; grep 'Learning iteration' /data/logs/train_ppo_baseline.log 2>/dev/null | tail -1; "
    "echo ==VIDEOS==; ls -la /data/videos/ 2>/dev/null | tail -6; "
    "echo ==NPZ==; ls -1 /data/eval_out/*.npz 2>/dev/null | tail -3; "
    "echo ==FIN_LOG==; tail -4 /data/logs/finish_ppo.log 2>/dev/null"
)


def main():
    conf = load_conf()
    deadline = time.time() + MAX_HOURS * 3600
    print("[monitor] 等待 PPO 基线训练 + 收尾链完成 ...", flush=True)

    while time.time() < deadline:
        try:
            cli = connect(conf)
            _, out, err = cli.exec_command(CMD, timeout=60)
            txt = out.read().decode("utf-8", "ignore")
            cli.close()
        except Exception as e:              # 连接类问题：只重试，不算失败
            print(f"[monitor] 连接异常（重试）：{type(e).__name__}", flush=True)
            time.sleep(60)
            continue

        if CONN_ERR_PAT.search(txt):
            print("[monitor] 输出疑似连接产物，忽略本轮", flush=True)
            time.sleep(CHECK_EVERY)
            continue

        ppo_done = "==PPO_DONE==\n1" in txt or re.search(r"==PPO_DONE==\s*\n\s*[1-9]", txt)
        fin_done = "==FIN_DONE==\n1" in txt or re.search(r"==FIN_DONE==\s*\n\s*[1-9]", txt)

        ts = time.strftime("%H:%M:%S")
        iter_line = ""
        m = re.search(r"Learning iteration (\d+)/(\d+)", txt)
        if m:
            iter_line = f"iter {m.group(1)}/{m.group(2)}"
        print(f"[{ts}] PPO训练完成={ppo_done} 收尾链完成={fin_done} {iter_line}", flush=True)

        if fin_done:
            print("[monitor] 全部完成！\n" + txt, flush=True)
            return 0
        if ppo_done and not fin_done:
            print("[monitor] 训练已完，收尾链进行中（评估 + 视频）", flush=True)
        time.sleep(CHECK_EVERY)

    print("[monitor] 超时退出（4h）", flush=True)
    return 1


if __name__ == "__main__":
    sys.exit(main())
