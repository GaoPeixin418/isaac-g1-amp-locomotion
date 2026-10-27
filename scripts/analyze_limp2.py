# -*- coding: utf-8 -*-
"""相位版跛行分析（analyze_limp 的补强）

背景：v1 版只比了 duty / stance 时长的**幅值**不对称。这在数学上有个大漏洞：
左右脚 duty 完全相等时 AI=0，但两只脚可能是**同时落地、同时抬起**（同步步态 /
兔子跳），也可能是正常交替。前者视觉上就是"不稳、一直在找平衡"，
而幅值类指标完全无感。

本脚本补两个真正的对称性判据：
  1. 相位差 φ = 左脚接触中点相对右脚的 cycle 滞后（正常步行 ≈ 0.5，同步 ≈ 0 或 1）
  2. 双支撑占比 DS = 两只脚同时触地的时长比例（人类 1.0 m/s ≈ 10-15%；
     >25% 说明步态保守/踌躇；同步跳会偏高且相位≈0）
再补拖脚判据：足端最大抬起高度左右比。
"""
import sys
import numpy as np


def contact_mask(z, fps):
    """足端高度 -> 触地掩码。用分布自适应阈值，避免绝对门槛失效。"""
    z = np.convolve(z, np.ones(3) / 3, mode="same")
    lo, hi = np.percentile(z, 5), np.percentile(z, 95)
    thr = lo + 0.12 * (hi - lo)
    return z < thr


def run_starts(mask):
    """掩码 False->True 的上升沿索引"""
    on = np.where(~mask[:-1] & mask[1:])[0] + 1
    off = np.where(mask[:-1] & mask[1:])[0] + 1
    return on, off


def phase_stats(mL, mR, fps, T):
    """交叉相关求最佳滞后 —— 比"上升沿配对"更抗噪声"""
    a = mL.astype(float) - mL.mean()
    b = mR.astype(float) - mR.mean()
    best_lag, best_v = 0, -1e18
    maxlag = int(1.2 * fps)
    for lag in range(0, maxlag):
        if lag == 0:
            v = float(np.dot(a, b))
        else:
            v = float(np.dot(a[lag:], b[:-lag]))
        if v > best_v:
            best_v, best_lag = v, lag
    # 周期估计：用左脚自相关第一个峰
    ac = np.correlate(a, a, mode="full")[len(a) - 1:]
    ac = ac / ac[0]
    period = np.nan
    for i in range(int(0.25 * fps), int(2.0 * fps)):
        if i + 1 < len(ac) and ac[i] > ac[i - 1] and ac[i] >= ac[i + 1] and ac[i] > 0.2:
            period = i / fps
            break
    phase = best_lag / fps / period if period == period and period > 0 else np.nan
    ds = float(np.mean(mL & mR))
    single = float(np.mean(mL | mR))
    return dict(lag=best_lag / fps, period=period, phase=phase, ds=ds, stance_any=single)


def analyze(path, warmup=0.3, max_cols=3):
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    fps = 1.0 / dt
    fz = d.get("foot_z")
    if fz is None:
        return None
    if fz.ndim == 2:          # 兼容 (T,2) 单环境格式
        fz = fz[:, None, :]
    T, N, _ = fz.shape
    n0 = int(T * warmup)
    rows = []
    lvx, lvy = d.get("lin_vel_x"), d.get("lin_vel_y")
    vel_cmd = d.get("velocities")
    for j in range(min(N, max_cols)):
        zL = fz[n0:, j, 0]
        zR = fz[n0:, j, 1]
        mL = contact_mask(zL, fps)
        mR = contact_mask(zR, fps)
        st = phase_stats(mL, mR, fps, T - n0)
        dutyL, dutyR = float(mL.mean()), float(mR.mean())
        # 拖脚：swing 期足端最大抬起高度
        liftL = float(np.percentile(zL, 90) - np.percentile(zL, 10))
        liftR = float(np.percentile(zR, 90) - np.percentile(zR, 10))
        ai = lambda x, y: 200.0 * abs(x - y) / (x + y + 1e-9)
        rows.append(dict(
            cmd=float(vel_cmd[j]) if vel_cmd is not None else -1,
            v=float(np.mean(np.hypot(lvx[n0:, j], lvy[n0:, j]))) if lvx is not None else np.nan,
            dutyL=dutyL, dutyR=dutyR, ai_duty=ai(dutyL, dutyR),
            ds=st["ds"], any_foot=st["stance_any"],
            period=st["period"], phase=st["phase"], lag=st["lag"],
            liftL=liftL, liftR=liftR, ai_lift=ai(liftL, liftR),
        ))
    return rows


def main():
    print(f"{'model':<20}{'cmd':>5}{'v':>6} | {'dutyL':>6}{'dutyR':>6}{'AI%':>6} | "
          f"{'双支撑%':>7}{'周期s':>6}{'相位φ':>7} | {'抬升L':>6}{'抬升R':>6}{'AI%':>6}")
    print("-" * 104)
    for p in sys.argv[1:]:
        name = p.replace("\\", "/").split("/")[-1].replace("gait_", "").replace(".npz", "")
        rows = analyze(p)
        if rows is None:
            print(f"{name:<20}  缺 foot_z")
            continue
        for r in rows:
            ph = r["phase"]
            phs = f"{ph:>7.2f}" if ph == ph else f"{'nan':>7}"
            per = r["period"]
            pers = f"{per:>6.2f}" if per == per else f"{'nan':>6}"
            print(f"{name:<20}{r['cmd']:>5.1f}{r['v']:>6.2f} | "
                  f"{r['dutyL']:>6.3f}{r['dutyR']:>6.3f}{r['ai_duty']:>6.1f} | "
                  f"{100*r['ds']:>7.1f}{pers}{phs} | "
                  f"{r['liftL']:>6.3f}{r['liftR']:>6.3f}{r['ai_lift']:>6.1f}")
        print("-" * 104)
    print()
    print("判据：相位 φ≈0.50 正常交替步行；φ≈0.00/1.00 同步（双足跳/齐步）—— 后者视觉上")
    print("      就是'站不稳、努力保持平衡'，而 duty 对称指数对此完全无感。")
    print("      双支撑%：人类 1.0 m/s 约 10-15%；>25% 步态保守踌躇。")


if __name__ == "__main__":
    main()
