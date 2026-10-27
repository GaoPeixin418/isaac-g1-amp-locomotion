#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用【逐关节均值】口径重测任意 ckpt，输出可与 limp_curve.csv 直接比较的一行。

为什么必须重测
-----------------------------------------------------------------------------
历史基线（model_4999=36.7%、model_6400=37.1%）是用"腿合力"口径算的：
先对 6 个关节求 RMS 合成一个标量，再比左右。那个口径会被"分工式跛行"
骗过——髋往前顶、膝踝往回拉，合成后左右恰好相等，于是报成"对称"。
10-27 发现 iter1200 就是这种情况：合力 8.5%，逐关节却是髋 59%/膝 46%/踝 44%。

现行口径（watch_limp.measure）：先逐关节算 AI，再对 6 个关节取均值。
两者数值不可比。旧基线不重测，就没法判断新奖励到底有没有用。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python eval_baseline.py <ckpt1> <ckpt2> ...
    # 速度可覆盖： VELOCITIES=1.0 python eval_baseline.py ...
"""
import glob
import os
import subprocess
import sys
import time

sys.path.insert(0, "/data/setup")
from watch_limp import measure  # 口径唯一来源，禁止在这里另写一份

EVAL = "/data/setup/eval_gait.py"
EVAL_OUT = "/data/eval_out"
CSV = "/data/progress/baseline_ai.csv"
VEL = os.environ.get("VELOCITIES", "1.0")


def newest_npz(after_ts):
    cands = [p for p in glob.glob(os.path.join(EVAL_OUT, "gait_*.npz"))
             if os.path.getmtime(p) > after_ts - 5]
    return max(cands, key=os.path.getmtime) if cands else None


def main():
    cks = sys.argv[1:]
    if not cks:
        print("用法: eval_baseline.py <ckpt...>")
        return 1
    os.makedirs("/data/progress", exist_ok=True)
    if not os.path.exists(CSV):
        with open(CSV, "w", encoding="utf-8") as f:
            f.write("ckpt,run,iter,leg_ai_pct,hip_ai_pct,knee_ai_pct,ankle_ai_pct,"
                    "rms_L,rms_R,rms_mean,v_cmd,v_act\n")
    for ck in cks:
        if not os.path.exists(ck):
            print("[skip] 不存在 %s" % ck)
            continue
        run = os.path.basename(os.path.dirname(ck))
        it = os.path.basename(ck).replace("model_", "").replace(".pt", "")
        print("[eval] %s ..." % ck, flush=True)
        ts = time.time()
        p = subprocess.run(
            ["/data/envs/isaaclab/bin/python", EVAL,
             "--task", "LeggedLab-Isaac-AMP-G1-v0",
             "--checkpoint", ck,
             "--num_envs", "1", "--steps", "600",
             "--velocities", VEL, "--headless"],
            capture_output=True, text=True, cwd="/data/legged_lab",
        )
        if p.returncode != 0:
            print("[fail] rc=%d\n%s" % (p.returncode, (p.stdout or "")[-600:] + (p.stderr or "")[-600:]), flush=True)
            continue
        npz = newest_npz(ts)
        if npz is None:
            print("[fail] 无 npz")
            continue
        m = measure(npz)
        if m is None:
            print("[fail] 数据不足")
            continue
        legL, legR, legai, hipai, kneeai, ankai, vcmd, vact = m
        with open(CSV, "a", encoding="utf-8") as f:
            f.write("%s,%s,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f\n"
                    % (os.path.basename(ck), run, it, legai, hipai, kneeai, ankai,
                       legL, legR, 0.5 * (legL + legR), vcmd, vact))
        print("[done] %s legAI=%.1f%% (hip %.1f/knee %.1f/ankle %.1f) rms %.1f/%.1f v_act=%.2f"
              % (os.path.basename(ck), legai, hipai, kneeai, ankai, legL, legR, vact), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
