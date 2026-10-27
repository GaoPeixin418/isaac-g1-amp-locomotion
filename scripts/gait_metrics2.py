#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步态指标（v2）：参考侧只取【连续】片段，机器人与参考用完全相同的口径。

v1 的问题
-----------------------------------------------------------------------------
v1 用布尔掩码 sp∈[lo,hi] 取帧。掩码在时间上不连续，把一段步态切成好几截，
拼起来的信号在原片段边界处跳变 -> 自相关主峰乱掉（参考侧 acad=283 而接触
计数只有 131，两者差一倍）。机器人侧全程恒速所以没暴露问题，
也就是说 v1 的两侧口径其实并不一致。

v2 的修法
-----------------------------------------------------------------------------
对每段 clip，先找连续且长度 >= min_seg_s 的速度区间，只在区间内部测量。

输出
-----------------------------------------------------------------------------
  cadence   双腿 steps/min（= 单脚步频 Hz × 120）
  stride_m  步长 = v / 单脚步频，**不乘 2**
  duty      支撑相占比
  knee_*    膝关节角度统计（度）
  hipROM    髋 pitch 活动范围（度）

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python gait_metrics2.py
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


def contact_hyst(z, fps, lo_q=0.15, hi_q=0.30, min_dur=0.12):
    """带迟滞的接触判定：低于 lo 才算接触，要高于 hi 才算离地。"""
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
    return _drop_short(c, max(1, int(min_dur * fps)))


def autocorr_period(x, fps, fmin=0.5, fmax=4.0):
    """自相关主峰反推步频（Hz）。与接触计数原理完全不同，用于交叉验证。"""
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


def contiguous_segments(mask, min_len):
    out, i, n = [], 0, len(mask)
    while i < n:
        if mask[i]:
            j = i
            while j < n and mask[j]:
                j += 1
            if j - i >= min_len:
                out.append((i, j))
            i = j
        else:
            i += 1
    return out


def gait_from_footz(fz, fps, v):
    """fz: (T,K)。返回 dict（cadence=双腿 steps/min, stride_m, duty）或 None。"""
    fz = np.asarray(fz, dtype=np.float64)
    if fz.ndim == 1:
        fz = fz[:, None]
    T, K = fz.shape
    if T / fps < 0.8:
        return None
    steps, contact_n = 0, 0
    for k in range(K):
        c = contact_hyst(fz[:, k], fps)
        steps += int(np.sum(np.diff(c.astype(np.int8)) == 1))
        contact_n += int(c.sum())
    if steps == 0:
        return None
    dur = T / fps
    freq = steps / K / dur           # 单脚 Hz
    return {"cadence": freq * 120.0, "stride_m": v / freq if freq > 0 else float("nan"),
            "duty": contact_n / (T * K)}


def joint_stats(qd):
    """qd: (T, N, D) 度。返回膝/髋统计。"""
    kl = qd[:, :, KNEE_L].mean(axis=1)
    kr = qd[:, :, KNEE_R].mean(axis=1)
    hl = qd[:, :, HIP_PITCH_L].mean(axis=1)
    hr = qd[:, :, HIP_PITCH_R].mean(axis=1)
    km = 0.5 * (kl + kr)
    hm = 0.5 * (hl + hr)
    return {"knee_mean": float(np.mean(km)),
            "knee_min": float(np.percentile(km, 2)),
            "knee_max": float(np.percentile(km, 98)),
            "knee_rom": float(np.percentile(km, 98) - np.percentile(km, 2)),
            "hip_mean": float(np.mean(hm)),
            "hip_rom": float(np.percentile(hm, 98) - np.percentile(hm, 2))}


def ref_all():
    bins = [(0.7, 0.9), (0.9, 1.1), (1.1, 1.4), (1.4, 2.0), (2.0, 99.0)]
    acc = {b: [] for b in bins}
    for f in sorted(glob.glob(os.path.join(REF_DIR, "*.pkl"))):
        p = pickle.load(open(f, "rb"))
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        fps = float(np.asarray(p["fps"]).reshape(-1)[0])
        v = np.diff(rp[:, :2], axis=0) * fps
        sp = np.hypot(v[:, 0], v[:, 1])
        sp = np.concatenate([sp, sp[-1:]])
        kb = np.asarray(p["key_body_pos"], dtype=np.float64)
        qd = np.degrees(np.asarray(p["dof_pos"], dtype=np.float64))
        for lo, hi in bins:
            for a, b in contiguous_segments((sp >= lo) & (sp < hi), int(1.0 * fps)):
                fz = np.stack([kb[a:b, i, 2] for i in FOOT_IDS], axis=1)
                kn_l = qd[a:b, KNEE_L]
                kn_r = qd[a:b, KNEE_R]
                hp_l = qd[a:b, HIP_PITCH_L]
                hp_r = qd[a:b, HIP_PITCH_R]
                r = gait_from_footz(fz, fps, float(sp[a:b].mean()))
                if not r:
                    continue
                km = 0.5 * (kn_l + kn_r)
                hm = 0.5 * (hp_l + hp_r)
                r.update({
                    "knee_mean": float(np.mean(km)),
                    "knee_min": float(np.percentile(km, 2)),
                    "knee_max": float(np.percentile(km, 98)),
                    "knee_rom": float(np.percentile(km, 98) - np.percentile(km, 2)),
                    "hip_mean": float(np.mean(hm)),
                    "hip_rom": float(np.percentile(hm, 98) - np.percentile(hm, 2)),
                })
                acc[(lo, hi)].append(r)
    print("== 参考动作（v2：仅连续片段）==")
    hdr = "%-13s %6s %6s %9s %9s %7s %7s %7s %7s %7s"
    print(hdr % ("speed", "segs", "v", "cad/min", "stride_m", "duty", "knee", "kneeMin", "kneeROM", "hipROM"))
    for lo, hi in bins:
        rs = acc[(lo, hi)]
        if not rs:
            continue
        lbl = "%.1f-%.1f" % (lo, hi) if hi < 90 else ">2.0"
        g = lambda k: float(np.nanmean([r[k] for r in rs if k in r]))
        print(hdr % (lbl, len(rs), "%.2f" % g("v"), "%.1f" % g("cadence"),
                     "%.3f" % g("stride_m"), "%.3f" % g("duty"),
                     "%.1f" % g("knee_mean"), "%.1f" % g("knee_min"),
                     "%.1f" % g("knee_rom"), "%.1f" % g("hip_rom")))


def robot_all():
    print("\n== 机器人（v2：同口径）==")
    hdr = "%-26s %6s %9s %9s %7s %7s %7s %7s %7s"
    print(hdr % ("npz", "v", "cad/min", "stride_m", "duty", "knee", "kneeMin", "kneeROM", "hipROM"))
    for pth in sorted(glob.glob(os.path.join(EVAL_OUT, "gait_*.npz")), key=os.path.getmtime):
        try:
            d = np.load(pth, allow_pickle=True)
            fz = d["foot_z"]
            fps = 1.0 / float(d["dt"])
            T, N, K = fz.shape
            n0 = int(T * 0.3)
            v = float(np.mean(np.hypot(d["lin_vel_x"][n0:], d["lin_vel_y"][n0:])))
            r = gait_from_footz(fz[n0:].reshape(T - n0, N * K), fps, v)
            if r is None:
                print("%-26s  --" % os.path.basename(pth))
                continue
            if "joint_pos" in d.files:
                r.update(joint_stats(np.degrees(d["joint_pos"][n0:])))
            g = lambda k: r.get(k, float("nan"))
            print(hdr % (os.path.basename(pth), "%.2f" % v, "%.1f" % g("cadence"),
                               "%.3f" % g("stride_m"), "%.3f" % g("duty"),
                               "%.1f" % g("knee_mean"), "%.1f" % g("knee_min"),
                               "%.1f" % g("knee_rom"), "%.1f" % g("hip_rom")))
        except Exception as e:
            print("%-26s ERR %s" % (os.path.basename(pth), e))


def main():
    ref_all()
    robot_all()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
