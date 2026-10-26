"""
analyze_limp.py - 左右脚不对称（跛行）量化
=============================================================================
背景（2026-10-26）：walk10_follow / amp4000_omni2 视频里明显"左脚更承重"
的跛行。需要回答两件事：
  1) amp_4000 到底有多跛行？（数值化，不靠眼睛）
  2) 10-22 新增的 air_time_variance 对称性惩罚是否真的治住了？
     -> 对比带对称的模型（lam050 / sym_model_*）不对称度是否显著更低

口径（沿用项目铁律：单脚周期、左右分离）：
  对每个 env（= 一个速度档）分别算：
    duty_L / duty_R      左右脚支撑占比（接触时间/总时间）
    peak_L / peak_R      单 stance 峰值接触力中位数 (N)
    imp_L / imp_R        单位时间冲量 (N·s/s)
    cycle_L / cycle_R    平均 gait cycle 时长（同脚两次着地间隔）
  不对称指数 AI = |L - R| / (0.5*(L+R))，百分比。人类文献通常 <5%。
  跛行判定：按 gait cycle 对齐看 奇数步 vs 偶数步 的 stance 时长是否交替。
"""
import sys
import numpy as np

def contact_runs(on, min_len=5):
    """on: bool (T,)。返回 [(start, end_exclusive), ...] 连续 True 段。"""
    idx = np.flatnonzero(np.diff(np.r_[False, on, False]))
    runs = [(idx[i], idx[i + 1]) for i in range(0, len(idx), 2)]
    return [(a, b) for a, b in runs if b - a >= min_len]

def foot_stats(fz, dt, thr):
    on = fz > thr
    runs = contact_runs(on)
    duty = on.mean()
    if len(runs) < 3:
        return None
    peaks = [fz[a:b].max() for a, b in runs]
    durs = [(b - a) * dt for a, b in runs]
    cycles = np.diff([a for a, _ in runs]) * dt
    impulse = fz.sum() * dt / (len(fz) * dt)  # 平均力 = 冲量/总时长
    return dict(
        duty=float(duty), n_runs=len(runs),
        peak_med=float(np.median(peaks)),
        stance_mean=float(np.mean(durs)), stance_std=float(np.std(durs)),
        cycle=float(np.mean(cycles)) if len(cycles) else np.nan,
        mean_force=float(impulse),
        stances=durs, peaks=peaks,
    )

def ai(l, r):
    m = 0.5 * (l + r)
    return abs(l - r) / m * 100 if m > 1e-9 else np.nan

def analyze(path, warmup=0.3):
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    fz = d["foot_force_z"] if "foot_force_z" in d else None
    if fz is not None and fz.max() < 1.0:
        fz = None  # 早期采集没配上 contact sensor，全 0（amp_4000_clean 实测）
    foot_z = d["foot_z"]        # (T, N, 2) 足端世界系高度，任何 npz 都有
    lvx = d["lin_vel_x"]; lvy = d["lin_vel_y"]
    vels = d["velocities"]
    T, N = foot_z.shape[:2]
    n0 = int(T * warmup)
    rows = []
    for j in range(N):
        v = float(np.mean(np.hypot(lvx[n0:, j], lvy[n0:, j])))
        out = dict(cmd=float(vels[j]), v=v)
        stats = []
        for k in range(2):  # 0/1 = 左/右（顺序不影响对称性指标）
            if fz is not None:
                sig = fz[n0:, j, k]
                thr = max(0.2 * np.percentile(sig, 90), 1.0)
                on = sig > thr
                peaks = [sig[a:b].max() for a, b in contact_runs(on)]
                peak_med = float(np.median(peaks)) if peaks else np.nan
            else:
                # 用足端高度检 stance：离地阈值自适应（G1 ankle 高约 4-6cm）
                z = foot_z[n0:, j, k]
                thr = float(z.min() + 0.3 * (np.median(z) - z.min()))
                on = z < thr
                peak_med = np.nan
            runs = contact_runs(on, min_len=4)
            if len(runs) < 3:
                stats.append(None); continue
            durs = [(b - a) * dt for a, b in runs]
            stats.append(dict(
                duty=float(on.mean()), peak_med=peak_med,
                stance_mean=float(np.mean(durs)), stance_std=float(np.std(durs)),
            ))
        if stats[0] is None or stats[1] is None:
            out["err"] = "接触段不足"
        else:
            out.update(
                duty_L=stats[0]["duty"], duty_R=stats[1]["duty"],
                ai_duty=ai(stats[0]["duty"], stats[1]["duty"]),
                stance_L=stats[0]["stance_mean"], stance_R=stats[1]["stance_mean"],
                ai_stance=ai(stats[0]["stance_mean"], stats[1]["stance_mean"]),
            )
            if not np.isnan(stats[0]["peak_med"]):
                out.update(
                    peak_L=stats[0]["peak_med"], peak_R=stats[1]["peak_med"],
                    ai_peak=ai(stats[0]["peak_med"], stats[1]["peak_med"]),
                )
        rows.append(out)
    return rows

def main():
    files = sys.argv[1:]
    print(f"{'model':<26}{'cmd':>5}{'v':>6} | {'dutyL':>6}{'dutyR':>6}{'AI%':>6} | "
          f"{'stcL(s)':>7}{'stcR(s)':>7}{'AI%':>6} | {'peakL':>7}{'peakR':>7}{'AI%':>6}")
    print("-" * 105)
    for p in files:
        name = p.split("\\")[-1].split("/")[-1].replace("gait_", "").replace(".npz", "")
        for r in analyze(p):
            if "err" in r:
                print(f"{name:<26}{r['cmd']:>5.1f}{r['v']:>6.2f} | {r['err']}")
                continue
            pk = (f"{r['peak_L']:>7.0f}{r['peak_R']:>7.0f}{r['ai_peak']:>6.1f}"
                  if "peak_L" in r else f"{'--':>7}{'--':>7}{'--':>6}")
            print(f"{name:<26}{r['cmd']:>5.1f}{r['v']:>6.2f} | "
                  f"{r['duty_L']:>6.3f}{r['duty_R']:>6.3f}{r['ai_duty']:>6.1f} | "
                  f"{r['stance_L']:>7.3f}{r['stance_R']:>7.3f}{r['ai_stance']:>6.1f} | {pk}")
        print("-" * 105)

if __name__ == "__main__":
    main()
