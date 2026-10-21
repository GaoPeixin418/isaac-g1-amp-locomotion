"""
11 - 多模型步态指标并排对比（纯 numpy，不需要 Isaac Sim）
=============================================================================
为什么要有这个脚本：
   单个模型的指标表只能回答"它好不好"，回答不了"AMP 比 PPO 强在哪"。
   面试要的是对照：同样 6 个速度档、同样 30 秒采样、同样指标定义，
   把 AMP(风格峰值) / AMP(训练终点) / PPO(无风格) 三行摆在一起。

核心判据（不依赖任何外部文献，全部来自我们自己的数据）：
   1) duty_factor 随指令速度单调下降  ->  存在 walk→run 的连续过渡
      人类就是这么做的：走得越快，脚在地上的时间占比越低。
      PPO 若做不到，就说明它学的是"一个步态拉伸"，而不是"一族步态"。
   2) Spearman 秩相关 (duty vs speed) 越接近 -1 越单调
   3) vel_err：任务完成度，风格奖励不该以牺牲跟速度为代价
   4) CoT：能耗，越低越接近人类经济性

人类参考值说明（诚实标注）：
   HUMAN_DUTY 是文献里 gait 实验室的典型量级（步行 ~0.60、跑 ~0.30），
   不是精确引用值，只作为"数量级是否合理"的参照带。
   真正硬的结论来自 1)2)3)4) —— 那是我们自己测的。
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gait import detect_strides  # noqa: E402

WARMUP_SEC = 5.0
# 启动暂态额外观察窗：实测 2.5 m/s 档前 6 秒几乎不动（分段均速 0.14），
# 之后稳定在 2.1-2.4。只用 5s warmup 会把这一档的误差算成 0.8，
# 掩盖"稳态其实是跟得上"的事实。所以额外报一列 10s 之后的稳态误差。
WARMUP_SS_SEC = 10.0

# 人类典型支撑占比（量级参照带，非精确引用）
HUMAN_DUTY = {0.5: 0.68, 1.0: 0.62, 1.5: 0.55, 2.0: 0.46, 2.5: 0.36, 3.0: 0.31}

METRICS = ["duty_factor", "stride_freq", "stride_len", "foot_clear", "vel_err", "vel_err_ss", "cot"]


def compute(npz_path: Path):
    d = np.load(npz_path, allow_pickle=True)
    dt = float(d["dt"])
    mass = float(d["mass"])
    vels = np.asarray(d["velocities"], dtype=float)
    T = d["lin_vel_x"].shape[0]
    W = int(WARMUP_SEC / dt)
    Wss = int(WARMUP_SS_SEC / dt)
    foot_z = d["foot_z"]

    rows = []
    for i in range(len(vels)):
        cmd = float(vels[i])
        vx = d["lin_vel_x"][W:, i]
        vy = d["lin_vel_y"][W:, i]
        Tw = T - W
        z = foot_z[W:, i, :]
        thr = np.percentile(z, 15, axis=0) + 0.03
        stance = z < thr

        per_foot = []
        for k in range(foot_z.shape[2]):
            cnt, duty, _, _ = detect_strides(stance[:, k], dt)
            zk = z[:, k]
            if cnt >= 2:
                sw = ~stance[:, k]
                clearance = float(zk[sw].max() - zk[stance[:, k]].min()) \
                    if sw.any() and stance[:, k].any() else float("nan")
                per_foot.append((cnt / (T * dt), duty, clearance))

        vx_ss = d["lin_vel_x"][Wss:, i] if Wss < T else vx
        vel_err_ss = float(np.abs(vx_ss - cmd).mean())

        if not per_foot:
            rows.append(dict(cmd=cmd, vel_err_ss=vel_err_ss,
                             **{m: float("nan") for m in METRICS if m != "vel_err_ss"}))
            continue

        freq = float(np.mean([p[0] for p in per_foot]))
        duty = float(np.mean([p[1] for p in per_foot]))
        clear = float(np.nanmean([p[2] for p in per_foot]))
        mean_v = float(vx.mean())
        dist = mean_v * Tw * dt
        # 步长：freq 是单脚步频 = gait cycle 频率，一个 cycle 前进 v/freq。
        # 原来写 2.0 * v / freq 是错的（见 analyze_gait.py 里的修正记录）。
        stride_len = mean_v / freq if freq > 1e-6 else float("nan")
        power = np.abs(d["torque"][W:, i, :] * d["joint_vel"][W:, i, :]).sum(axis=1)
        energy = float(power.sum() * dt)
        cot = energy / (mass * 9.81 * abs(dist)) if abs(dist) > 1e-3 else float("nan")
        vel_err = float(np.abs(vx - cmd).mean())
        drift = float(np.abs(vy).mean())

        rows.append(dict(cmd=cmd, duty_factor=duty, stride_freq=freq,
                         stride_len=stride_len, foot_clear=clear,
                         vel_err=vel_err, vel_err_ss=vel_err_ss, cot=cot,
                         lateral_drift=drift))
    return rows


def spearman(x, y):
    """秩相关，只用于判断单调性（样本 6 个，不谈显著性）。"""
    x, y = np.asarray(x, float), np.asarray(y, float)
    ok = np.isfinite(x) & np.isfinite(y)
    if ok.sum() < 3:
        return float("nan")
    rx = np.argsort(np.argsort(x[ok])).astype(float)
    ry = np.argsort(np.argsort(y[ok])).astype(float)
    if rx.std() < 1e-9 or ry.std() < 1e-9:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def parse_pairs(args):
    """label=path 形式，也允许只给 path（label 取文件名）。"""
    pairs = []
    for a in args:
        if "=" in a:
            lab, p = a.split("=", 1)
        else:
            p, lab = a, Path(a).stem.replace("gait_", "")
        pairs.append((lab, Path(p)))
    return pairs


def fmt(v, nd=3):
    return "n/a" if not np.isfinite(v) else f"{v:.{nd}f}"


def main() -> int:
    pairs = parse_pairs(sys.argv[1:])
    if not pairs:
        print("用法: compare_gait.py amp_2000=path1.npz amp_4999=path2.npz ppo_2999=path3.npz")
        return 2

    data = {}
    for lab, p in pairs:
        if not p.exists():
            print(f"!! 找不到 {p}")
            continue
        data[lab] = compute(p)

    if not data:
        return 1

    # ---------- 1. 逐模型明细 ----------
    for lab, rows in data.items():
        print(f"\n### {lab}")
        print("| 指令速度 (m/s) | 支撑占比 | 步频 (1/s) | 步长 (m) | 抬升 (m) | 速度误差 | 稳态误差 | CoT |")
        print("|" + "|".join(["---"] * 8) + "|")
        for r in rows:
            print(f"| {r['cmd']:.1f} | {fmt(r['duty_factor'])} | {fmt(r['stride_freq'],2)} | "
                  f"{fmt(r['stride_len'],2)} | {fmt(r['foot_clear'])} | {fmt(r['vel_err'])} | "
                  f"{fmt(r['vel_err_ss'])} | {fmt(r['cot'],2)} |")

    # ---------- 2. 支撑占比 vs 速度（核心对比） ----------
    speeds = sorted({r["cmd"] for rows in data.values() for r in rows})
    print("\n### 核心对比：支撑占比 vs 指令速度")
    hdr = "| 速度 | " + " | ".join(data.keys()) + " | 人类参照 |"
    print(hdr)
    print("|" + "|".join(["---"] * (len(data) + 2)) + "|")
    for s in speeds:
        cells = []
        for rows in data.values():
            v = next((r["duty_factor"] for r in rows if abs(r["cmd"] - s) < 1e-6), float("nan"))
            cells.append(fmt(v))
        href = HUMAN_DUTY.get(round(s, 1))
        print(f"| {s:.1f} | " + " | ".join(cells) + f" | {fmt(href,2) if href else 'n/a'} |")

    # ---------- 3. 单调性 / 误差汇总 ----------
    print("\n### 汇总判据")
    print("| 模型 | 支撑占比秩相关(应≈-1) | 平均速度误差 | 稳态速度误差(去10s) | 平均 CoT | 与人类参照带平均偏差 |")
    print("|" + "|".join(["---"] * 6) + "|")
    for lab, rows in data.items():
        sp = [r["cmd"] for r in rows]
        rho = spearman(sp, [r["duty_factor"] for r in rows])
        ve = np.nanmean([r["vel_err"] for r in rows])
        vess = np.nanmean([r["vel_err_ss"] for r in rows])
        cot = np.nanmean([r["cot"] for r in rows])
        dev = np.nanmean([abs(r["duty_factor"] - HUMAN_DUTY.get(round(r["cmd"], 1), np.nan))
                          for r in rows])
        print(f"| {lab} | {fmt(rho,2)} | {fmt(ve)} | {fmt(vess)} | {fmt(cot,2)} | {fmt(dev)} |")

    # ---------- 4. 落 CSV ----------
    out = Path("backup/eval_out/compare_gait.csv")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("model,cmd," + ",".join(METRICS) + "\n")
        for lab, rows in data.items():
            for r in rows:
                f.write(lab + f",{r['cmd']:.1f}," +
                        ",".join(f"{r[m]:.4f}" for m in METRICS) + "\n")
    print(f"\nCSV -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
