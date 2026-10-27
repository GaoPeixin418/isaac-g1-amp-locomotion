#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""统一的步态指标口径（带迟滞 + 最小时长），并做自相关交叉验证。

为什么单阈值不行
-----------------------------------------------------------------------------
z <= thr 这种判定在足端高度恰好贴着阈值时会来回跳，每个小抖动都被算成一次
"着地"，步频被系统性高估。第一次用它测参考动作得到 198 steps/min，
比人类步行 (100-115) 高出近一倍 —— 明显是算法噪声，不是真实步频。

修法
-----------------------------------------------------------------------------
1. 迟滞（Schmitt trigger）：低于 lo 才算接触，要高于 hi 才算离地，hi > lo
2. 最小接触/腾空时长：短于 min_dur 的段直接合并掉
3. 自相关交叉验证：用髋 pitch 的主周期反推步频，与接触计数互相印证。
   两个原理完全不同的算法给出同一个数，这个数才可信。

口径硬约束（踩过坑）
-----------------------------------------------------------------------------
* 单脚步频 = 一只脚每秒着地次数；gait cycle 频率与之相同
* stride_len = v / freq，**不能乘 2**
* 文献 cadence 通常是双腿 steps/min，换算成单脚要 ÷2

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python gait_metrics.py            # 机器人 + 参考全表
"""
import glob
import os
import pickle

import numpy as np

EVAL_OUT = "/data/eval_out"
REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
KNEE_L, KNEE_R = 3, 9
HIP_PITCH_L, HIP_PITCH_R = 0, 6
FOOT_IDS = (0, 1)


def contact_hyst(z, fps, lo_q=0.15, hi_q=0.30, min_dur=0.12):
    """带迟滞的接触判定。返回 bool 数组（True=接触）。"""
    z = np.asarray(z, dtype=np.float64)
    if z.ptp() < 1e-6:
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

    # 去掉过短的段
    min_n = max(1, int(min_dur * fps))
    c = _drop_short(c, min_n)
    return c


def _drop_short(c, min_n):
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


def onset_count(c):
    cc = c.astype(np.int8)
    return int(np.sum(np.diff(cc) == 1))


def gait_from_footz(fz, fps):
    """fz: (T, K) 或 (T,) 足端高度。返回 (freq_单脚Hz, duty)。"""
    fz = np.asarray(fz, dtype=np.float64)
    if fz.ndim == 1:
        fz = fz[:, None]
    T, K = fz.shape
    tot_steps = 0
    tot_contact = 0
    tot = 0
    for k in range(K):
        c = contact_hyst(fz[:, k], fps)
        tot_steps += onset_count(c)
        tot_contact += int(c.sum())
        tot += T
    dur = T / fps
    freq = tot_steps / K / dur          # 单脚着地频率 Hz
    duty = tot_contact / (tot * 1.0)
    return freq, duty


def knee_stats_from_q(qdeg):
    """qdeg: (T, D) 度。"""
    kl, kr = qdeg[:, KNEE_L], qdeg[:, KNEE_R]
    km = 0.5 * (kl + kr)
    return {
        "knee_mean": float(km.mean()),
        "knee_min": float(np.percentile(km, 2)),
        "knee_max": float(np.percentile(km, 98)),
        "rom": float(np.percentile(km, 98) - np.percentile(km, 2)),
        "hip_mean": float(np.mean(0.5 * (qdeg[:, HIP_PITCH_L] + qdeg[:, HIP_PITCH_R]))),
    }


def autocorr_period(x, fps, fmin=0.5, fmax=4.0):
    """用自相关主峰反推步频（Hz）。与接触计数原理完全不同，用于交叉验证。"""
    x = np.asarray(x, dtype=np.float64)
    x = x - x.mean()
    if x.std() < 1e-9:
        return float("nan")
    n = len(x)
    f = np.fft.rfft(x * np.hanning(n))
    ac = np.fft.irfft(f * np.conj(f))[: n // 2]
    ac /= ac[0] + 1e-12
    lo = max(1, int(fps / fmax))
    hi = min(len(ac) - 1, int(fps / fmin))
    if hi <= lo:
        return float("nan")
    seg = ac[lo:hi]
    k = int(np.argmax(seg)) + lo
    return fps / k


def main():
    bins = [(0.7, 0.9), (0.9, 1.1), (1.1, 1.4), (1.4, 2.0), (2.0, 99.0)]
    print("== 参考动作（统一口径：迟滞接触 + 自相关校验）==")
    print("%-14s %8s %8s %10s %10s %10s" % ("speed", "frames", "v", "cad/min", "stride_m", "acad/min"))
    for lo, hi in bins:
        frames, vs, cads, strs, acs = 0, [], [], [], []
        for f in sorted(glob.glob(os.path.join(REF_DIR, "*.pkl"))):
            p = pickle.load(open(f, "rb"))
            rp = np.asarray(p["root_pos"], dtype=np.float64)
            fps = float(np.asarray(p["fps"]).reshape(-1)[0])
            v = np.diff(rp[:, :2], axis=0) * fps
            sp = np.hypot(v[:, 0], v[:, 1])
            sp = np.concatenate([sp, sp[-1:]])
            kb = np.asarray(p["key_body_pos"], dtype=np.float64)
            m = (sp >= lo) & (sp < hi)
            if m.sum() < 20:
                continue
            fz = np.stack([kb[m, i, 2] for i in FOOT_IDS], axis=1)
            f, _ = gait_from_footz(fz, fps)
            cad = f * 60.0 * 2  # 单脚 Hz -> 双腿 steps/min
            qd = np.degrees(np.asarray(p["dof_pos"], dtype=np.float64))[m]
            af = autocorr_period(qd[:, HIP_PITCH_L], fps)
            frames += int(m.sum())
            vs.append(float(sp[m].mean()))
            cads.append(cad)
            strs.append(float(sp[m].mean()) / f if f > 0 else float("nan"))
            acs.append(af * 120.0 if not np.isnan(af) else float("nan"))
        if frames:
            lbl = "%.1f-%.1f" % (lo, hi) if hi < 90 else ">2.0"
            print("%-14s %8d %8.2f %10.1f %10.3f %10.1f"
                  % (lbl, frames, np.mean(vs), np.nanmean(cads),
                     np.nanmean(strs), np.nanmean(acs)))

    print("\n== 机器人（同口径）==")
    print("%-28s %8s %10s %10s %10s" % ("npz", "v", "cad/min", "stride_m", "acad/min"))
    for pth in sorted(glob.glob(os.path.join(EVAL_OUT, "gait_*.npz")), key=os.path.getmtime):
        try:
            d = np.load(pth, allow_pickle=True)
            fz = d["foot_z"]
            fps = 1.0 / float(d["dt"])
            T, N, K = fz.shape
            n0 = int(T * 0.3)
            v = float(np.mean(np.hypot(d["lin_vel_x"][n0:], d["lin_vel_y"][n0:])))
            f, duty = gait_from_footz(fz[n0:].reshape(T - n0, -1), fps)
            cad = f * 60.0 * 2
            af = np.nan
            if "joint_pos" in d.files:
                qd = np.degrees(d["joint_pos"][n0:])
                af = autocorr_period(qd[:, 0, HIP_PITCH_L], fps) * 120.0
            print("%-28s %8.2f %10.1f %10.3f %10.1f"
                  % (os.path.basename(pth), v, cad, v / f if f > 0 else float("nan"), af))
        except Exception as e:
            print("%-28s ERR %s" % (os.path.basename(pth), e))
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
