#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考动作 vs 机器人：髋/膝摆动幅度与步态时序的完整对照。

动机（2026-10-27 晚，重要更正）
-----------------------------------------------------------------------------
用户主诉"膝盖一直弯着、像快走"，我一度按"蹲姿"治（加了躯干高度 + 膝角
工作点惩罚）。但基线实测把这条推翻了：

  机器人 iter2999: 躯干高 0.762 m（参考 0.765）、膝均屈 16.7°（参考 15.0）、
                   duty 0.664（人类 0.60）—— **全都在正常范围，它没有蹲**。
  唯一显著异常:    步频 136 vs 参考 100、步长 0.89 vs 1.22。

所以"蹲"是误诊。真正可疑的是**髋摆动幅度不足**（hip ROM 20.9°），
腿摆不开 -> 每步迈得短 -> 只能靠提高步频凑够 1.0 m/s。
本脚本用来把参考侧的同一个量测出来，确认这条是否成立。

判据
-----------------------------------------------------------------------------
  参考 hip_rom >> 20.9°  -> 髋是根因，joint_deviation_hip 在压制摆动
  参考 hip_rom ≈ 20.9°   -> 髋不是根因，得另找（例如参考本身步频就高）

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python ref_hip_rom.py
"""
import glob
import os
import pickle

import numpy as np

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
HIP_PITCH_L, HIP_PITCH_R = 0, 6
KNEE_L, KNEE_R = 3, 9
FOOT_IDS = (0, 1)


def main():
    fs = sorted(glob.glob(os.path.join(REF_DIR, "*.pkl")))
    bins = [(0.8, 1.1), (1.1, 1.4)]
    acc = {b: [] for b in bins}

    for f in fs:
        p = pickle.load(open(f, "rb"))
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        fps = float(np.asarray(p["fps"]).reshape(-1)[0])
        v = np.diff(rp[:, :2], axis=0) * fps
        sp = np.hypot(v[:, 0], v[:, 1])
        sp = np.concatenate([sp, sp[-1:]])
        qd = np.degrees(np.asarray(p["dof_pos"], dtype=np.float64))
        kb = np.asarray(p["key_body_pos"], dtype=np.float64)
        for lo, hi in bins:
            m = (sp >= lo) & (sp < hi)
            # 连续片段
            idx = np.flatnonzero(m)
            if len(idx) < int(1.0 * fps):
                continue
            brk = np.flatnonzero(np.diff(idx) > 1)
            segs = np.split(idx, brk + 1)
            for s in segs:
                if len(s) < int(1.0 * fps):
                    continue
                a, b = s[0], s[-1] + 1
                hp = 0.5 * (qd[a:b, HIP_PITCH_L] + qd[a:b, HIP_PITCH_R])
                kn = 0.5 * (qd[a:b, KNEE_L] + qd[a:b, KNEE_R])
                fz = np.stack([kb[a:b, i, 2] for i in FOOT_IDS], axis=1)
                acc[(lo, hi)].append({
                    "hip_rom": float(np.percentile(hp, 98) - np.percentile(hp, 2)),
                    "hip_mean": float(np.mean(hp)),
                    "knee_rom": float(np.percentile(kn, 98) - np.percentile(kn, 2)),
                    "knee_mean": float(np.mean(kn)),
                    "knee_max": float(np.percentile(kn, 98)),
                    "z_amp": float(np.percentile(fz[:, 0], 95) - np.percentile(fz[:, 0], 5)),
                })

    print("== 参考动作：髋/膝摆动幅度（度）==")
    print("%-12s %6s %10s %10s %10s %10s %10s"
          % ("speed", "segs", "hipROM", "hipMean", "kneeROM", "kneeMean", "kneeMax"))
    for lo, hi in bins:
        rs = acc[(lo, hi)]
        if not rs:
            continue
        g = lambda k: float(np.mean([r[k] for r in rs]))
        print("%-12s %6d %10.1f %10.1f %10.1f %10.1f %10.1f"
              % ("%.1f-%.1f" % (lo, hi), len(rs), g("hip_rom"), g("hip_mean"),
                 g("knee_rom"), g("knee_mean"), g("knee_max")))

    print("\n== 机器人 v1 (iter2999, 1.0 m/s) ==")
    print("%-12s %6s %10.1f %10.1f %10.1f %10.1f %10.1f"
          % ("1.0", "-", 20.9, 2.4, 36.1, 16.7, 39.4))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
