#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考动作的步频/步长（按速度分档），用于判定机器人是不是"碎步快走"。

为什么必须和参考比，不能只看机器人绝对步频
-----------------------------------------------------------------------------
G1 腿长约 0.6-0.75 m，人约 0.9 m。同样 1.0 m/s，腿短的本来就得多迈几步 ——
步频天生更高。所以"150 steps/min"本身不能定罪，得看参考在同样速度下是多少。

两侧口径一致（踩过坑的硬要求）：
  * 接触判定都用足端 z 的百分位阈值（参考没有接触力列）
  * 步数都用 contact 上升沿计数
  * stride_len = v / freq，不能乘 2（freq 是单脚步频，一个 cycle 左右各一次）

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python ref_gait_stats.py
"""
import glob
import os
import pickle

import numpy as np

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
FOOT_IDS = (0, 1)  # key_body_pos 里 z≈0.05 的两只脚


def contact(z):
    thr = np.percentile(z, 5) + 0.15 * (np.percentile(z, 95) - np.percentile(z, 5))
    return z <= thr


def main():
    bins = [(0.7, 0.9), (0.9, 1.1), (1.1, 1.4), (1.4, 2.0), (2.0, 99.0)]
    acc = {b: {"cad": [], "stride": [], "v": [], "frames": 0} for b in bins}

    for f in sorted(glob.glob(os.path.join(REF_DIR, "*.pkl"))):
        p = pickle.load(open(f, "rb"))
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        fps = float(np.asarray(p["fps"]).reshape(-1)[0])
        dt = 1.0 / fps
        v = np.diff(rp[:, :2], axis=0) / dt
        sp = np.hypot(v[:, 0], v[:, 1])
        sp = np.concatenate([sp, sp[-1:]])
        kb = np.asarray(p["key_body_pos"], dtype=np.float64)

        for lo, hi in bins:
            m = (sp >= lo) & (sp < hi)
            if m.sum() < 20:
                continue
            seg_v = float(sp[m].mean())
            steps = 0
            for fid in FOOT_IDS:
                c = contact(kb[m, fid, 2])
                steps += int(np.sum(np.diff(c.astype(int)) == 1))
            dur = int(m.sum()) * dt
            cad = steps / dur * 60.0
            stride = seg_v / (cad / 120.0) if cad > 0 else float("nan")
            acc[(lo, hi)]["cad"].append(cad)
            acc[(lo, hi)]["stride"].append(stride)
            acc[(lo, hi)]["v"].append(seg_v)
            acc[(lo, hi)]["frames"] += int(m.sum())

    print("== 参考动作步频/步长（按速度分档）==")
    print("%-14s %8s %8s %10s %10s" % ("speed(m/s)", "frames", "v_mean", "cadence", "stride_m"))
    for lo, hi in bins:
        r = acc[(lo, hi)]
        if r["frames"] == 0:
            continue
        lbl = "%.1f-%.1f" % (lo, hi) if hi < 90 else ">2.0"
        print("%-14s %8d %8.2f %10.1f %10.3f"
              % (lbl, r["frames"], np.mean(r["v"]), np.mean(r["cad"]), np.mean(r["stride"])))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
