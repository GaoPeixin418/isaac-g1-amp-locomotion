#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考动作步频/步长：整段 clip 测量，再按 clip 均速插到目标速度。

为什么不按帧掩码分档
-----------------------------------------------------------------------------
按帧挑速度会把连续步态切成碎片，边界处信号跳变，自相关和接触计数都会失真
（v1 实测：参考侧接触计数 131 而自相关 283，差一倍）。

这里改成：每个 clip 整体测一次步频和均速，得到 30 个 (v, cadence) 点，
再用线性插值插到目标速度。clip 内部速度变化带来的误差在 30 个 clip 上平均掉。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python ref_cadence.py
"""
import glob
import os
import pickle

import numpy as np

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
FOOT_IDS = (0, 1)
KNEE_L, KNEE_R = 3, 9
HIP_PITCH_L, HIP_PITCH_R = 0, 6


def contact_hyst(z, fps, lo_q=0.15, hi_q=0.30, min_dur=0.12):
    z = np.asarray(z, dtype=np.float64)
    if len(z) == 0 or z.ptp() < 1e-6:
        return np.zeros(len(z), bool)
    p5, p95 = np.percentile(z, 5), np.percentile(z, 95)
    lo = p5 + lo_q * (p95 - p5)
    hi = p5 + hi_q * (p95 - p5)
    state = False
    c = np.zeros(len(z), bool)
    for i, v in enumerate(z):
        if not state and v <= lo:
            state = True
        elif state and v > hi:
            state = False
        c[i] = state
    # 去掉过短段
    min_n = max(1, int(min_dur * fps))
    c = c.copy()
    n = len(c)
    i = 0
    while i < n:
        j = i
        while j < n and c[j] == c[i]:
            j += 1
        if (j - i) < min_n and 0 < i and j < n:
            c[i:j] = not c[i]
        i = j
    return c


def autocorr_period(x, fps, fmin=0.5, fmax=4.0):
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if len(x) < 8 or x.std() < 1e-9:
        return float("nan")
    n = len(x)
    f = np.fft.rfft(x * np.hanning(n))
    ac = np.fft.irfft(f * np.conj(f))[: n // 2]
    ac /= ac[0] + 1e-12
    lo = max(1, int(fps / fmax))
    hi = min(len(ac) - 1, int(fps / fmin))
    if hi <= lo:
        return float("nan")
    return fps / (int(np.argmax(ac[lo:hi])) + lo)


def main():
    rows = []
    for f in sorted(glob.glob(os.path.join(REF_DIR, "*.pkl"))):
        p = pickle.load(open(f, "rb"))
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        fps = float(np.asarray(p["fps"]).reshape(-1)[0])
        dt = 1.0 / fps
        v = np.diff(rp[:, :2], axis=0) / dt
        sp = np.hypot(v[:, 0], v[:, 1])
        vmean = float(sp.mean())
        kb = np.asarray(p["key_body_pos"], dtype=np.float64)
        qd = np.degrees(np.asarray(p["dof_pos"], dtype=np.float64))

        # 整段接触计数
        steps, contact_n = 0, 0
        for fid in FOOT_IDS:
            c = contact_hyst(kb[:, fid, 2], fps)
            steps += int(np.sum(np.diff(c.astype(np.int8)) == 1))
            contact_n += int(c.sum())
        T = kb.shape[0]
        dur = T * dt
        freq = steps / len(FOOT_IDS) / dur
        cad = freq * 120.0
        stride = vmean / freq if freq > 0 else float("nan")
        duty = contact_n / (T * len(FOOT_IDS))

        # 自相关交叉验证（整段，无掩码）
        af = autocorr_period(qd[:, HIP_PITCH_L], fps) * 120.0
        km = 0.5 * (qd[:, KNEE_L] + qd[:, KNEE_R])
        rows.append((os.path.basename(f), vmean, cad, stride, duty, af,
                     float(km.mean()), float(np.percentile(km, 2)),
                     float(np.percentile(km, 98) - np.percentile(km, 2))))
    rows.sort(key=lambda r: r[1])
    print("== 参考动作（整段测量，按均速排序）==")
    print("%-34s %6s %9s %9s %7s %9s %7s %7s %7s"
          % ("clip", "v", "cad/min", "stride_m", "duty", "acad/min", "knee", "knMin", "knROM"))
    for r in rows:
        print("%-34s %6.2f %9.1f %9.3f %7.3f %9.1f %7.1f %7.1f %7.1f" % r)

    # 插到 1.0 m/s
    vv = np.array([r[1] for r in rows]); cc = np.array([r[2] for r in rows])
    ss = np.array([r[3] for r in rows]); kk = np.array([r[6] for r in rows])
    kr = np.array([r[8] for r in rows])
    print("\n== 插值到 1.0 m/s ==")
    print("  cadence  = %6.1f steps/min" % np.interp(1.0, vv, cc))
    print("  stride   = %6.3f m" % np.interp(1.0, vv, ss))
    print("  knee     = %6.1f deg" % np.interp(1.0, vv, kk))
    print("  kneeROM  = %6.1f deg" % np.interp(1.0, vv, kr))
    print("  （近邻 clip 均速: %s）" % ", ".join("%.2f" % v for v in vv[(vv > 0.7) & (vv < 1.3)]))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
