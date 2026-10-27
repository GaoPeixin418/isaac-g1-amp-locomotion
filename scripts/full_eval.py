#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对任意 ckpt：跑一次 eval，然后用唯一一套口径输出全部指标。

为什么要把所有指标塞进一个脚本
-----------------------------------------------------------------------------
这个项目已经四次栽在"口径不一致"上：
  1. 触地时长 vs 出力 —— 测错物理量
  2. 腿合力 vs 逐关节 —— 反向抵消被当成对称
  3. 单阈值接触判定 —— 足端抖动被算成多迈一步
  4. 掩码取帧 —— 片段被切碎，自相关失真
每次都是"两个数字看着都不错，但它们其实不是同一个东西"。
所以这里把所有指标固定在同一份 npz、同一段稳态区间、同一套算法上，
任何两个 ckpt 输出的行都可以直接横向比。

指标
-----------------------------------------------------------------------------
  v_cmd/v_act  指令速度 / 实际水平速率
  legAI        左腿出力不对称 = mean over 6 joints of 200|L-R|/(L+R)  [%]
               （逐关节，不用腿合力；合力会被髋与膝踝反向抵消）
  hipAI/kneeAI/ankleAI  对应关节的 AI
  cadence      双腿 steps/min（单脚步频 Hz × 120），迟滞接触判定
  stride       v / 单脚步频，**不乘 2**
  duty         支撑相占比
  knee*        膝关节角度（度）mean / 最伸直 / 最屈 / ROM
  hipROM       髋 pitch 活动范围（度）

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python full_eval.py <ckpt> [<ckpt2> ...]
    VELOCITIES=1.0 STEPS=600 /data/envs/isaaclab/bin/python full_eval.py ...
"""
import glob
import os
import subprocess
import sys
import time

import numpy as np

sys.path.insert(0, "/data/setup")
from gait_metrics2 import gait_from_footz, joint_stats  # 口径唯一来源

EVAL = "/data/setup/eval_gait.py"
EVAL_OUT = "/data/eval_out"
LEG_L, LEG_R = slice(0, 6), slice(6, 12)

HDR = ("ckpt", "v_cmd", "v_act", "legAI%", "hipAI%", "kneeAI%", "ankAI%",
       "rmsL", "rmsR", "cad", "stride", "duty", "knee", "kneeMin", "kneeMax", "kneeROM", "hipROM")


def newest_npz(after_ts):
    c = [p for p in glob.glob(os.path.join(EVAL_OUT, "gait_*.npz")) if os.path.getmtime(p) > after_ts - 5]
    return max(c, key=os.path.getmtime) if c else None


def ai(a, b):
    return abs(a - b) / (0.5 * (abs(a) + abs(b))) * 100 if (abs(a) + abs(b)) > 1e-9 else float("nan")


def measure(ck, speed, steps):
    ts = time.time()
    p = subprocess.run(
        ["/data/envs/isaaclab/bin/python", EVAL,
         "--task", "LeggedLab-Isaac-AMP-G1-v0",
         "--checkpoint", ck,
         "--num_envs", "1", "--steps", str(steps),
         "--velocities", str(speed), "--headless"],
        capture_output=True, text=True, cwd="/data/legged_lab",
    )
    if p.returncode != 0:
        print("[fail] %s rc=%d\n%s" % (ck, p.returncode, (p.stdout or "")[-500:] + (p.stderr or "")[-500:]), flush=True)
        return None
    npz = newest_npz(ts)
    if npz is None:
        print("[fail] 无 npz: %s" % ck, flush=True)
        return None
    d = np.load(npz, allow_pickle=True)
    T, N, D = d["torque"].shape
    n0 = int(T * 0.3)
    rms = np.sqrt(np.mean(d["torque"][n0:] ** 2, axis=(0, 1)))
    per = [ai(float(rms[k]), float(rms[k + 6])) for k in range(6)]
    vact = float(np.mean([np.mean(np.hypot(d["lin_vel_x"][n0:, j], d["lin_vel_y"][n0:, j])) for j in range(N)]))
    fps = 1.0 / float(d["dt"])
    g = gait_from_footz(d["foot_z"][n0:].reshape(T - n0, -1), fps, vact)
    r = {
        "ckpt": os.path.basename(ck),
        "v_cmd": float(np.mean(d["velocities"])),
        "v_act": vact,
        "legAI%": float(np.mean(per)),
        "hipAI%": float(per[0]),
        "kneeAI%": float(per[3]),
        "ankAI%": float(per[4]),
        "rmsL": float(np.sqrt(np.mean(rms[LEG_L] ** 2))),
        "rmsR": float(np.sqrt(np.mean(rms[LEG_R] ** 2))),
    }
    if g:
        r.update({"cad": g["cadence"], "stride": g["stride_m"], "duty": g["duty"]})
    if "joint_pos" in d.files:
        r.update(joint_stats(np.degrees(d["joint_pos"][n0:])))
    return r


def main():
    cks = sys.argv[1:]
    if not cks:
        print("用法: full_eval.py <ckpt...>")
        return 1
    speed = float(os.environ.get("VELOCITIES", "1.0"))
    steps = int(os.environ.get("STEPS", "600"))
    rows = []
    for ck in cks:
        if not os.path.exists(ck):
            print("[skip] %s" % ck)
            continue
        r = measure(ck, speed, steps)
        if r:
            rows.append(r)
    print("\n== 全指标对照 @ %.1f m/s ==" % speed)
    print(" ".join("%-9s" % h for h in HDR))
    for r in rows:
        print(" ".join("%-9s" % ("%.1f" % r[h] if isinstance(r.get(h), float) else str(r.get(h, "-")))
                       for h in HDR))
    return 0


if __name__ == "__main__":
    sys.exit(main())
