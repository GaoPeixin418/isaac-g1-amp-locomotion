#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考动作的躯干高度 + 膝角分位数（按速度分档）。

目的：为两个新奖励项定"客观基准值"，避免拍脑袋：
  * base_height_bias  -> 目标躯干高度（治蹲，策略无法靠"把屈曲从膝挪到髋"躲避）
  * knee_bias         -> 目标膝角工作点（治蹲的姿态侧表现）

为什么用躯干高度而不是只看膝角
-----------------------------------------------------------------------------
只罚膝角，策略可以把屈曲从膝转移到髋/踝，膝角正常但人照样蹲着。
躯干高度是"蹲不蹲"的最终观测量，绕不过去。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python ref_height_stats.py
"""
import glob
import os
import pickle

import numpy as np

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
KNEE_L, KNEE_R = 3, 9


def root_speed(p):
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
    acc = {b: {"h": [], "k": [], "kp10": [], "kp90": [], "frames": 0} for b in bins}

    for f in fs:
        p = pickle.load(open(f, "rb"))
        sp, _ = root_speed(p)
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        h = rp[:, 2]
        q = np.degrees(np.asarray(p["dof_pos"], dtype=np.float64))
        km = 0.5 * (q[:, KNEE_L] + q[:, KNEE_R])
        for lo, hi in bins:
            m = (sp >= lo) & (sp < hi)
            if m.sum() < 10:
                continue
            acc[(lo, hi)]["h"].append(float(np.mean(h[m])))
            acc[(lo, hi)]["k"].append(float(np.mean(km[m])))
            acc[(lo, hi)]["kp10"].append(float(np.percentile(km[m], 10)))
            acc[(lo, hi)]["kp90"].append(float(np.percentile(km[m], 90)))
            acc[(lo, hi)]["frames"] += int(m.sum())

    print("== 参考动作：躯干高度 & 膝角（按速度分档）==")
    print("%-14s %8s %9s %9s %9s %9s" % ("speed", "frames", "height_m", "knee°", "knee_p10", "knee_p90"))
    for lo, hi in bins:
        r = acc[(lo, hi)]
        if r["frames"] == 0:
            continue
        lbl = "%.1f-%.1f" % (lo, hi) if hi < 90 else ">2.0"
        print("%-14s %8d %9.3f %9.1f %9.1f %9.1f"
              % (lbl, r["frames"], np.mean(r["h"]), np.mean(r["k"]),
                 np.mean(r["kp10"]), np.mean(r["kp90"])))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
