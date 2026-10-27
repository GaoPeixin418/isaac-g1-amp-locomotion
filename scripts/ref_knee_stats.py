#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考动作（LAFAN1 重定向到 G1 29-DOF）的膝关节角度统计，按速度分档。

目的：回答"机器人蹲着走，是参考数据本来就蹲，还是学偏了"。

判定逻辑
-----------------------------------------------------------------------------
  * 参考在 1.0-1.3 m/s 的 knee_mean 就 >40°  -> 参考本来蹲，改奖励没用
  * 参考在 15-30°（人类区间）                  -> 参考正常，是训练/奖励把机器人按蹲了

为什么不直接用"关节角"，而要看速度分档
-----------------------------------------------------------------------------
数据集里跑步片段的屈膝本来就大，整体平均会把跑步混进来。
按速度分档后才能和 1.0 m/s 的 eval 比。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python ref_knee_stats.py
"""
import glob
import os
import pickle

import numpy as np

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
KNEE_L, KNEE_R = 3, 9  # 与 watch_limp.py 一致

# 人类步行文献参照（1.0-1.4 m/s）
HUMAN_KNEE_MEAN = (15, 30)   # deg
HUMAN_KNEE_MIN = (2, 15)     # deg, 支撑中期近乎伸直


def root_speed(p):
    """根水平速率（m/s）。注意不能用 x 分量：clip 里有转弯/倒退，会互相抵消。"""
    rp = np.asarray(p["root_pos"], dtype=np.float64)
    fps = float(np.asarray(p["fps"]).reshape(-1)[0])
    dt = 1.0 / fps
    v = np.diff(rp[:, :2], axis=0) / dt
    sp = np.hypot(v[:, 0], v[:, 1])
    return np.concatenate([sp, sp[-1:]]), fps


def main():
    fs = sorted(glob.glob(os.path.join(REF_DIR, "*.pkl")))
    if not fs:
        print("没找到参考 clip")
        return 1
    bins = [(0.0, 0.8), (0.8, 1.3), (1.3, 2.0), (2.0, 99.0)]
    acc = {b: {"kmean": [], "kmin": [], "krom": [], "frames": 0} for b in bins}
    allk = []

    for f in fs:
        p = pickle.load(open(f, "rb"))
        sp, fps = root_speed(p)
        q = np.degrees(np.asarray(p["dof_pos"], dtype=np.float64))
        kl, kr = q[:, KNEE_L], q[:, KNEE_R]
        kmean = 0.5 * (kl + kr)
        allk.append(kmean)
        for lo, hi in bins:
            m = (sp >= lo) & (sp < hi)
            if m.sum() < 10:
                continue
            a, b = kl[m], kr[m]
            acc[(lo, hi)]["kmean"].append(0.5 * (np.mean(a) + np.mean(b)))
            acc[(lo, hi)]["kmin"].append(0.5 * (np.percentile(a, 2) + np.percentile(b, 2)))
            acc[(lo, hi)]["krom"].append(0.5 * (np.max(a) - np.min(a) + np.max(b) - np.min(b)))
            acc[(lo, hi)]["frames"] += int(m.sum())

    print("== 参考动作膝关节角度（按速度分档）==")
    print("%-14s %8s %8s %8s %8s" % ("speed(m/s)", "frames", "knee°", "min°", "ROM°"))
    for lo, hi in bins:
        r = acc[(lo, hi)]
        if r["frames"] == 0:
            continue
        lbl = "%.1f-%.1f" % (lo, hi) if hi < 90 else ">2.0"
        print("%-14s %8d %8.1f %8.1f %8.1f"
              % (lbl, r["frames"], np.mean(r["kmean"]), np.mean(r["kmin"]), np.mean(r["krom"])))

    k = np.concatenate(allk)
    print("\n整体: knee_mean=%.1f  p5=%.1f  p95=%.1f" % (k.mean(), np.percentile(k, 5), np.percentile(k, 95)))
    print("人类步行参照: knee_mean %.0f-%.0f deg, 支撑中期最伸直 %.0f-%.0f deg"
          % (HUMAN_KNEE_MEAN + HUMAN_KNEE_MIN))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
