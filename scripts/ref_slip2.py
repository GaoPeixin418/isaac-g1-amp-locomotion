#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考动作：支撑相脚打滑检测（v2，修掉跨段 diff 的 bug）。

v1 的 bug（2026-10-27 晚，第 6 次差点踩坑）
-----------------------------------------------------------------------------
v1 写的是 `d = np.diff(fxy[c], axis=0)`。布尔索引把**所有**支撑帧抽出来拼成
一条连续序列，于是段与段之间（中间隔着整个摆动相）被当成相邻帧做 diff，
跳变被计成位移 -> 算出"支撑相脚速 1.15 m/s" -> 差点判定参考数据打滑。

实际 key_body_pos 已确认是**世界坐标**（kb idx0/1 的 z≈0.05 是脚，
idx4/5 的 z≈1.04 是头/手；root_pos z≈0.75）。

v2 的修法
-----------------------------------------------------------------------------
先找**连续**的支撑段，只在段内做 diff。段长度 < 0.15 s 的丢弃（太短，噪声大）。

判据
-----------------------------------------------------------------------------
  支撑相脚速 < 0.15 m/s  正常（脚踩住，身体从上方越过）
  支撑相脚速 > 0.30 m/s  打滑 —— 参考数据本身是"滑冰"

同时算"每步 root 前进距离"与"脚相对 root 前后摆幅"，看两者是否自洽：
  摆幅 ≈ 半步长 -> 自洽
  摆幅 << 半步长 -> root 在打滑

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python ref_slip2.py
"""
import glob
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, "/data/setup")
from gait_metrics2 import contact_hyst, contiguous_segments  # noqa: E402

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
FOOT_IDS = (0, 1)
MIN_SEG_S = 0.15


def main():
    fs = sorted(glob.glob(os.path.join(REF_DIR, "*.pkl")))
    rows = []
    for f in fs:
        p = pickle.load(open(f, "rb"))
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        fps = float(np.asarray(p["fps"]).reshape(-1)[0])
        dt = 1.0 / fps
        kb = np.asarray(p["key_body_pos"], dtype=np.float64)
        v = np.diff(rp[:, :2], axis=0) / dt
        sp = np.hypot(v[:, 0], v[:, 1])
        sp = np.concatenate([sp, sp[-1:]])

        m = (sp >= 0.8) & (sp < 1.3)
        idx = np.flatnonzero(m)
        if len(idx) < int(1.0 * fps):
            continue
        for s in np.split(idx, np.flatnonzero(np.diff(idx) > 1) + 1):
            if len(s) < int(1.0 * fps):
                continue
            a, b = s[0], s[-1] + 1
            seg_v = float(sp[a:b].mean())
            rates, amps = [], []
            for fi in FOOT_IDS:
                fxy = kb[a:b, fi, :2]
                fz = kb[a:b, fi, 2]
                c = contact_hyst(fz, fps)
                # 只在【连续】支撑段内做 diff
                for (i, j) in contiguous_segments(c, max(1, int(MIN_SEG_S * fps))):
                    d = np.diff(fxy[i:j], axis=0)
                    if len(d):
                        rates.append(float(np.mean(np.hypot(d[:, 0], d[:, 1]) / dt)))
                rel = fxy - rp[a:b, :2]
                dirv = rp[b - 1, :2] - rp[a, :2]
                if np.hypot(*dirv) > 0.05:
                    u = dirv / np.hypot(*dirv)
                    proj = rel @ u
                    amps.append(float(np.percentile(proj, 95) - np.percentile(proj, 5)))
            if rates:
                rows.append((os.path.basename(f), seg_v, float(np.mean(rates)),
                             float(np.mean(amps)) if amps else float("nan"), (b - a) / fps))

    if not rows:
        print("没有符合条件的片段")
        return 1
    print("== 参考动作 @0.8-1.3 m/s：支撑相脚打滑（v2 连续段内 diff）==")
    print("%-44s %6s %10s %11s %7s" % ("clip", "v", "slip_m/s", "swingAmp_m", "dur_s"))
    for r in sorted(rows, key=lambda x: -x[2])[:10]:
        print("%-44s %6.2f %10.3f %11.3f %7.1f" % r)
    sl = np.array([r[2] for r in rows])
    sa = np.array([r[3] for r in rows])
    med = float(np.median(sl))
    print("\n支撑相脚速: 中位 %.3f  均值 %.3f  p90 %.3f  (n=%d 段)"
          % (med, sl.mean(), np.percentile(sl, 90), len(rows)))
    print("脚相对 root 前后摆幅: 中位 %.3f m" % np.nanmedian(sa))
    print("判定: %s" % ("脚踩住了，参考数据没打滑" if med < 0.15
                        else ("轻微滑移，需留意" if med < 0.30 else "!! 明显打滑")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
