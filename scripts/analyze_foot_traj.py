# -*- coding: utf-8 -*-
"""足端轨迹形态分析 —— 针对"落地后脚又勾起来、再慢慢放下"的主诉

为什么原来的 duty / stance 指标测不出来：
  duty 只回答"脚接触地面多久"，是**要不要相信长度**。用户看到的病是
  **支撑相内部的形状畸变**（脚已经落地承重，却又抬起来悬一下再放下），
  这种病在 duty 上几乎不显形 —— 它改变的是"支撑期内足端高度曲线"，
  不是接触时长。

本脚本做两件事：
  A. 支撑相畸变（核心）：在每个 stance 段内部，测足端相对该段最低点的
     抬升峰值 -> "抬起再放下"事件的幅度、次数、占比。
  B. 摆动相形态：抬升高度、峰值相位、上升/下降斜率（左右对比）。

接触判定优先用真实接触力 foot_force_z；没有力数据（旧 npz 全 0）时
自动退回足端高度自适应阈值 —— 注意跨模型比较时两边口径必须一致。
"""
import sys
import os
import numpy as np


def ai(l, r):
    m = 0.5 * (abs(l) + abs(r))
    return abs(l - r) / m * 100 if m > 1e-9 else np.nan


def runs_of(mask, min_len=4):
    idx = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    rs = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]
    return [(a, b) for a, b in rs if b - a >= min_len]


import os

# 公平对比开关：环境变量 FORCE_Z=1 强制所有 npz 使用足端高度判定。
FORCE_Z = os.environ.get("FORCE_Z", "0") == "1"


def contact_mask(fz, ff):
    """返回 stance 掩码。有力数据优先用力（物理真值）。

    FORCE_Z=1 时强制退回足端高度判定 —— 用于不同 npz 之间的公平对比：
    有些 npz（model_6400 / lam050）带接触力，有些（amp_4999 / amp_4000_clean）
    接触力全 0 只能用高度判。混用两种口径比较等于拿不同的尺子量两个人。
    """
    if ff is not None and ff.max() > 1.0 and not FORCE_Z:
        return ff > max(0.05 * ff.max(), 5.0)
    z = np.convolve(fz, np.ones(3) / 3, mode="same")
    lo, hi = np.percentile(z, 5), np.percentile(z, 95)
    # 宽阈值：把"贴地 + 略微抬起"整体算作一次 stance，便于看段内畸变
    return z < lo + 0.30 * (hi - lo)


def stance_distortion(fz, ff, fps):
    """支撑相内部足端抬起事件"""
    mask = contact_mask(fz, ff)
    rs = runs_of(mask, min_len=int(0.12 * fps))
    if len(rs) < 2:
        return None
    z = np.convolve(fz, np.ones(3) / 3, mode="same")
    bumps, durs = [], []
    for a, b in rs:
        seg = z[a:b]
        base = float(np.percentile(seg, 10))
        bump = float(seg.max() - base)
        bumps.append(bump)
        durs.append((b - a) / fps)
    return dict(n=len(rs), bump_med=float(np.median(bumps)),
                bump_max=float(np.max(bumps)),
                bump_rate=float(np.mean(np.array(bumps) > 0.015)),
                dur=float(np.median(durs)))


def swing_shape(fz, ff, fps):
    """摆动相形态"""
    mask = contact_mask(fz, ff)
    rs = runs_of(mask, min_len=int(0.12 * fps))
    rows = []
    for i in range(len(rs) - 1):
        a, b = rs[i][1], rs[i + 1][0]
        if b - a < int(0.15 * fps):
            continue
        seg = np.interp(np.linspace(0, 1, 101), np.linspace(0, 1, b - a), zc(a, b, fz))
        k = int(np.argmax(seg))
        if k >= 100:
            continue
        fall = seg[k:]
        rise = seg[:k + 1]
        t_peak = k / 100.0
        rise_slope = float((rise[-1] - rise[0]) / max(t_peak, 1e-6))
        fall_slope = abs(float((seg[-1] - seg[k]) / max(1 - t_peak, 1e-6)))
        mono = np.linspace(seg[k], seg[-1], len(fall))
        late_bump = float(np.max(fall - mono))
        land_slope = float(np.mean(np.abs(np.diff(seg[-11:])))) if len(seg) > 11 else np.nan
        fdiff = np.abs(np.diff(fall))
        flat = float(np.mean(fdiff < 0.15 * (fdiff.max() + 1e-9))) if len(fdiff) else np.nan
        rows.append(dict(peak_h=float(seg.max() - seg[0]), t_peak=t_peak,
                         rise_slope=rise_slope, fall_slope=fall_slope,
                         late_bump=late_bump, land_slope=land_slope,
                         flat=flat, dur=(b - a) / fps))
    if not rows:
        return None
    out = {k: float(np.median([r[k] for r in rows])) for k in rows[0]}
    out["n"] = len(rows)
    return out


def zc(a, b, fz):
    z = np.convolve(fz, np.ones(3) / 3, mode="same")
    return z[a:b]


def med(rows, key):
    return float(np.median([r[key] for r in rows]))


def main():
    for path in sys.argv[1:]:
        d = np.load(path, allow_pickle=True)
        dt = float(d["dt"]); fps = 1.0 / dt
        fz = d["foot_z"]; ff = d.get("foot_force_z")
        vels = d["velocities"]
        T, N, _ = fz.shape
        n0 = int(T * 0.3)
        name = os.path.basename(path).replace("gait_", "").replace(".npz", "")
        have_f = ff is not None and ff[n0:].max() > 1.0 and not FORCE_Z
        print(f"\n########## {name}  (支撑判定: "
              f"{'接触力' if have_f else '足端高度'}"
              f"{'，FORCE_Z 强制' if FORCE_Z and ff is not None and ff[n0:].max() > 1.0 else ''}"
              f") ##########")
        print(f"{'cmd':>5} | {'内抬中位':>8}{'内抬最大':>8}{'异常步%':>8} | "
              f"{'脚0':>7}{'脚1':>7}{'AI%':>7} | {'抬升0':>6}{'抬升1':>6} | "
              f"{'峰相位0':>7}{'峰相位1':>7} | {'降斜率0':>7}{'降斜率1':>7} | "
              f"{'二抬0':>6}{'二抬1':>6}")
        for j in range(N):
            s0 = stance_distortion(fz[n0:, j, 0], ff[n0:, j, 0] if have_f else None, fps)
            s1 = stance_distortion(fz[n0:, j, 1], ff[n0:, j, 1] if have_f else None, fps)
            w0 = swing_shape(fz[n0:, j, 0], ff[n0:, j, 0] if have_f else None, fps)
            w1 = swing_shape(fz[n0:, j, 1], ff[n0:, j, 1] if have_f else None, fps)
            if s0 is None or s1 is None or w0 is None or w1 is None:
                print(f"{vels[j]:>5.1f} | 样本不足"); continue
            print(f"{vels[j]:>5.1f} | "
                  f"{s0['bump_med']:>8.4f}{s1['bump_med']:>8.4f}"
                  f"{100*0.5*(s0['bump_rate']+s1['bump_rate']):>8.1f} | "
                  f"{s0['n']:>7.0f}{s1['n']:>7.0f}{ai(s0['n'], s1['n']):>7.1f} | "
                  f"{w0['peak_h']:>6.3f}{w1['peak_h']:>6.3f} | "
                  f"{w0['t_peak']:>7.2f}{w1['t_peak']:>7.2f} | "
                  f"{w0['fall_slope']:>7.3f}{w1['fall_slope']:>7.3f} | "
                  f"{w0['late_bump']:>6.4f}{w1['late_bump']:>6.4f}")
        print("  内抬=支撑期内足端相对最低点的抬升(m)，二抬=下降段二次抬起(m)；")
        print("  异常步% = 内抬>1.5cm 的步伐占比。这两项高 = '落地又勾起来'。")


if __name__ == "__main__":
    main()
