#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从现有 eval npz 直接出对照表（不重跑 eval）。

背景：full_eval.py 的表头键名与 joint_stats() 返回的键名不一致
（"knee" vs "knee_mean"），导致膝角列全显示 "-"。数据其实都在 npz 里，
这里按正确的键重新取值，避免为了几列数字再烧一轮 Kit 冷启动。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python report_table.py
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, "/data/setup")
from gait_metrics2 import gait_from_footz

EVAL_OUT = "/data/eval_out"
KNEE_L, KNEE_R = 3, 9
HIP_PITCH_L, HIP_PITCH_R = 0, 6
LEG_L, LEG_R = slice(0, 6), slice(6, 12)


def ai(a, b):
    return abs(a - b) / (0.5 * (abs(a) + abs(b))) * 100 if (abs(a) + abs(b)) > 1e-9 else float("nan")


def one(pth):
    d = np.load(pth, allow_pickle=True)
    T, N, D = d["torque"].shape
    n0 = int(T * 0.3)
    rms = np.sqrt(np.mean(d["torque"][n0:] ** 2, axis=(0, 1)))
    per = [ai(float(rms[k]), float(rms[k + 6])) for k in range(6)]
    vact = float(np.mean([np.mean(np.hypot(d["lin_vel_x"][n0:, j], d["lin_vel_y"][n0:, j])) for j in range(N)]))
    fps = 1.0 / float(d["dt"])
    g = gait_from_footz(d["foot_z"][n0:].reshape(T - n0, -1), fps, vact)
    row = {
        "ckpt": os.path.basename(pth).replace("gait_", "").replace(".npz", ""),
        "v_act": vact,
        "legAI": float(np.mean(per)),
        "hipAI": float(per[0]),
        "kneeAI": float(per[3]),
        "ankAI": float(per[4]),
        "rmsL": float(np.sqrt(np.mean(rms[LEG_L] ** 2))),
        "rmsR": float(np.sqrt(np.mean(rms[LEG_R] ** 2))),
    }
    if g:
        row.update({"cad": g["cadence"], "stride": g["stride_m"], "duty": g["duty"]})
    if "joint_pos" in d.files:
        qd = np.degrees(d["joint_pos"][n0:])
        kl = qd[:, :, KNEE_L].mean(axis=1)
        kr = qd[:, :, KNEE_R].mean(axis=1)
        hl = qd[:, :, HIP_PITCH_L].mean(axis=1)
        hr = qd[:, :, HIP_PITCH_R].mean(axis=1)
        km = 0.5 * (kl + kr)
        hm = 0.5 * (hl + hr)
        row.update({
            "knee": float(np.mean(km)),
            "kneeMin": float(np.percentile(km, 2)),
            "kneeMax": float(np.percentile(km, 98)),
            "kneeROM": float(np.percentile(km, 98) - np.percentile(km, 2)),
            "hip": float(np.mean(hm)),
            "hipROM": float(np.percentile(hm, 98) - np.percentile(hm, 2)),
        })
    return row


def main():
    rows = []
    for pth in sorted(glob.glob(os.path.join(EVAL_OUT, "gait_*.npz")), key=os.path.getmtime):
        try:
            rows.append(one(pth))
        except Exception as e:
            print("ERR", pth, e)
    hdr = ["ckpt", "v_act", "legAI", "hipAI", "kneeAI", "ankAI", "rmsL", "rmsR",
           "cad", "stride", "duty", "knee", "kneeMin", "kneeMax", "kneeROM", "hip", "hipROM"]
    print(" ".join("%-11s" % h for h in hdr))
    for r in rows:
        print(" ".join("%-11s" % ("%.2f" % r[h] if isinstance(r.get(h), float) else str(r.get(h, "-")))
                       for h in hdr))
    return 0


if __name__ == "__main__":
    sys.exit(main())
