# -*- coding: utf-8 -*-
"""用真实接触力（foot_force_z）做跛行量化 + 左右脚分配核证

只有 model_6400 / lam050 / sym_model_1600 这三个 npz 采到了接触力；
amp_4999 / amp_4000_clean 的 foot_force_z 全 0（旧版 eval 环境的接触
传感器没打开），所以跨模型对比时**必须两边用同一算法**（项目铁律）。
"""
import sys
import os
import numpy as np


def ai(l, r):
    m = 0.5 * (l + r)
    return abs(l - r) / m * 100 if m > 1e-9 else np.nan


def runs_of(mask, min_len=4):
    idx = np.flatnonzero(np.diff(np.r_[False, mask, False]))
    rs = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]
    return [(a, b) for a, b in rs if b - a >= min_len]


def analyze(path, warmup=0.3):
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    ff = d["foot_force_z"]
    lvx, lvy = d["lin_vel_x"], d["lin_vel_y"]
    vels = d["velocities"]
    T, N, _ = ff.shape
    n0 = int(T * warmup)
    out_rows = []
    for j in range(N):
        row = dict(cmd=float(vels[j]),
                   v=float(np.mean(np.hypot(lvx[n0:, j], lvy[n0:, j]))))
        per = []
        for k in range(2):
            f = ff[n0:, j, k]
            # 接触指认也可以与他人用 ost 建议运行时再指定，这里用最大值尺度为准
            thr = max(0.05 * f.max(), 5.0)
            on = f > thr
            rs = runs_of(on)
            duty = float(on.mean())
            if len(rs) >= 3:
                peaks = [f[a:b].max() for a, b in rs]
                durs = [(b - a) * dt for a, b in rs]
                imps = [f[a:b].sum() * dt for a, b in rs]
                per.append(dict(duty=duty, peak=float(np.median(peaks)),
                                stance=float(np.mean(durs)),
                                imp=float(np.mean(imps)),
                                imp_rate=float(f.sum() * dt / ((T - n0) * dt))))
            else:
                per.append(dict(duty=duty, peak=np.nan, stance=np.nan,
                                imp=np.nan,
                                imp_rate=float(f.sum() * dt / ((T - n0) * dt))))
        L, R = per[0], per[1]
        row.update(
            dutyL=L["duty"], dutyR=R["duty"], ai_duty=ai(L["duty"], R["duty"]),
            ds=float(np.mean((ff[n0:, j, 0] > max(0.05 * ff[n0:, j, 0].max(), 5.0)) &
                             (ff[n0:, j, 1] > max(0.05 * ff[n0:, j, 1].max(), 5.0)))),
            peakL=L["peak"], peakR=R["peak"], ai_peak=ai(L["peak"], R["peak"]),
            stcL=L["stance"], stcR=R["stance"], ai_stc=ai(L["stance"], R["stance"]),
            impL=L["imp_rate"], impR=R["imp_rate"], ai_imp=ai(L["imp_rate"], R["imp_rate"]),
        )
        out_rows.append(row)
    return out_rows


def main():
    print(f"{'model':<18}{'cmd':>5}{'v':>6} | {'dutyL':>6}{'dutyR':>6}{'AI%':>6}{'双撑%':>6} | "
          f"{'峰力L':>7}{'峰力R':>7}{'AI%':>6} | {'单撑L':>6}{'单撑R':>6}{'AI%':>6} | {'均力AI%':>8}")
    print("-" * 118)
    for p in sys.argv[1:]:
        name = os.path.basename(p).replace("gait_", "").replace(".npz", "")
        d = np.load(p, allow_pickle=True)
        if d["foot_force_z"].max() < 1.0:
            print(f"{name:<18}  -- 无接触力数据（foot_force_z 全 0），跳过")
            print("-" * 118)
            continue
        for r in analyze(p):
            print(f"{name:<18}{r['cmd']:>5.1f}{r['v']:>6.2f} | "
                  f"{r['dutyL']:>6.3f}{r['dutyR']:>6.3f}{r['ai_duty']:>6.1f}{100*r['ds']:>6.1f} | "
                  f"{r['peakL']:>7.0f}{r['peakR']:>7.0f}{r['ai_peak']:>6.1f} | "
                  f"{r['stcL']:>6.3f}{r['stcR']:>6.3f}{r['ai_stc']:>6.1f} | "
                  f"{r['ai_imp']:>8.1f}")
        print("-" * 118)


if __name__ == "__main__":
    main()
