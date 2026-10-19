"""
09 - 步态指标分析（纯 numpy，不需要 Isaac Sim）
=============================================================================
输入：eval_gait.py 产出的 npz
输出：每个速度档一组步态指标 + markdown 表格 + CSV

先 Technical Terms 大白话：
   步频    一秒迈几步
   步长    一步迈多远
   足端抬升 脚离地多高  —— 抬太低会蹭地，抬太高像踢正步
   支撑占比 一只脚在地上的时间占比 —— 人类走路约 60%，跑步约 30%。
            判断"走"和"跑"最硬的指标就是这个，比看速度靠谱
   CoT     运输能耗 —— 走单位距离消耗的能量，越低越像人
"""
import sys
from pathlib import Path

import numpy as np

TERMS = {
    "stride_freq": "步频 (Hz)",
    "stride_len": "步长 (m)",
    "foot_clear": "足端抬升 (m)",
    "duty_factor": "支撑占比",
    "cot": "运输能耗 CoT",
    "vel_err": "速度跟踪误差 (m/s)",
    "lateral_drift": "侧向漂移 (m/s)",
}


def detect_strides(stance: np.ndarray, dt: float):
    """从 stance 布尔序列里数步数 + 算支撑占比。

    stance: (T,) bool
    返回 (n_steps, duty, contact_durations, swing_durations)
    """
    change = np.diff(stance.astype(np.int8))
    rise = np.where(change == 1)[0]      # swing → stance
    fall = np.where(change == -1)[0]     # stance → swing
    # 保证配对：只统计完整的 stance-swing 周期
    if stance[0]:
        fall = fall[fall > (rise[0] if len(rise) else -1)]
    pairs = []
    j = 0
    for i, r in enumerate(rise):
        later_falls = fall[fall > r]
        if len(later_falls) == 0:
            break
        f = later_falls[0]
        pairs.append((r, f))
        j += 1
    if len(pairs) < 2:
        return 0, float(stance.mean()), np.array([]), np.array([])
    contact = np.array([(f - r) * dt for r, f in pairs])
    swings = np.array([(pairs[i + 1][0] - pairs[i][1]) * dt for i in range(len(pairs) - 1)])
    duty = float(stance.mean())
    return len(pairs), duty, contact, swings


def main() -> int:
    npz_path = Path(sys.argv[1])
    data = np.load(npz_path, allow_pickle=True)
    dt = float(data["dt"])
    mass = float(data["mass"])
    vels = data["velocities"]
    T, N = data["lin_vel_x"].shape
    foot = data["foot_pos_b"]          # (T, N, 2, 3)
    has_force = "foot_force_z" in data.files

    print(f"数据：{npz_path.name}  T={T} ({T*dt:.1f}s)  N={N}  dt={dt}")
    print(f"足部 link 数：{foot.shape[2]}\n")

    rows = []
    for i in range(N):
        cmd_x = float(vels[i])
        vx = data["lin_vel_x"][:, i]
        vy = data["lin_vel_y"][:, i]

        # --- 接触判定：优先用接触力，退化到足部高度 ---
        if has_force:
            fz = np.abs(data["foot_force_z"][:, i])          # (T, 2)
            thr = 20.0
            stance = fz > thr
        else:
            z = foot[:, i, :, 2]
            stance = z < (np.percentile(z, 40) + 0.02)

        # 两只脚合并统计（任一只在地 = 有支撑）
        per_foot = []
        for k in range(foot.shape[2]):
            cnt, duty, contact, swings = detect_strides(stance[:, k], dt)
            z = foot[:, i, k, 2]
            if cnt >= 2:
                # 抬升 = swing 段最大高度 - stance 段最小高度
                swing_mask = ~stance[:, k]
                clearance = float(z[swing_mask].max() - z[stance[:, k]].min()) \
                    if swing_mask.any() and stance[:, k].any() else float("nan")
                per_foot.append((cnt / (T * dt), duty, clearance))
        if not per_foot:
            rows.append(dict(cmd=cmd_x, stride_freq=float("nan"), stride_len=float("nan"),
                             foot_clear=float("nan"), duty_factor=float("nan"),
                             cot=float("nan"), vel_err=float("nan"),
                             lateral_drift=float("nan")))
            continue

        freq = float(np.mean([p[0] for p in per_foot]))
        duty = float(np.mean([p[1] for p in per_foot]))
        clear = float(np.nanmean([p[2] for p in per_foot]))

        mean_v = float(vx.mean())
        dist = mean_v * T * dt
        stride_len = mean_v / freq if freq > 1e-6 else float("nan")

        # CoT = Σ|τ·ω| dt / (m g d)
        power = np.abs(data["torque"][:, i, :] * data["joint_vel"][:, i, :]).sum(axis=1)
        energy = float(power.sum() * dt)
        cot = energy / (mass * 9.81 * abs(dist)) if abs(dist) > 1e-3 else float("nan")

        vel_err = float(np.abs(vx - cmd_x).mean())
        drift = float(np.abs(vy).mean())

        rows.append(dict(cmd=cmd_x, stride_freq=freq, stride_len=stride_len,
                         foot_clear=clear, duty_factor=duty, cot=cot,
                         vel_err=vel_err, lateral_drift=drift))

    # ---- 打印表格 ----
    hdr = ["cmd"] + [TERMS[k] for k in TERMS]
    print("| " + " | ".join(hdr) + " |")
    print("|" + "|".join(["---"] * len(hdr)) + "|")
    for r in rows:
        cells = [f"{r['cmd']:.1f}"] + [f"{r[k]:.3f}" for k in TERMS]
        print("| " + " | ".join(cells) + " |")

    # ---- 落 CSV ----
    csv = npz_path.with_suffix(".csv")
    with open(csv, "w", encoding="utf-8") as fh:
        fh.write(",".join(hdr) + "\n")
        for r in rows:
            fh.write(",".join([f"{r['cmd']:.1f}"] + [f"{r[k]:.4f}" for k in TERMS]) + "\n")
    print(f"\nCSV -> {csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
