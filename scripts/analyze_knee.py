#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""诊断"蹲着走"：机器人膝关节角度 vs 参考动作 vs 人类文献。

用户主诉（2026-10-27）
-----------------------------------------------------------------------------
"f1200_walk10 看上去走得有点快，像是快走的姿态"
"行走过程中膝盖大部分时间是弯着的"

这两句大概率是同一个病的两面：如果机器人是**屈膝蹲着走**（crouched gait），
重心低、步子迈不开，为了跟上 1.0 m/s 就只能提高步频 -> 视觉上就是"小碎步快走"。
所以真正要修的不是速度，是**膝盖的伸展程度**。

它回答一个问题：机器人"蹲"是学来的，还是参考数据本来就蹲？
-----------------------------------------------------------------------------
  * 参考数据也蹲  -> 重定向/数据集问题，改机器人没用，得改参考
  * 参考不蹲机器人蹲 -> 奖励/训练问题（例如 default joint pos 或"回默认姿态"项把
    它按回半蹲），改奖励有效

指标（全部可证伪，都带人类文献参照）
-----------------------------------------------------------------------------
1. knee_mean      整周期平均屈膝角（度）。人类 1.2 m/s 步行约 20-25°
2. knee_min       周期内最伸直角度（度）。人类支撑中期接近 5-10°（近乎伸直）。
                  这个指标最敏感：蹲着走的人 knee_min 显著偏大（比如 >30°）
3. knee_rom       屈伸活动范围 = max - min（度）。人类约 55-65°
4. duty_supp      支撑相占比。人类 1.2 m/s 约 0.60-0.65
5. cadence / stride_len，用来验证"快走感"是不是步频偏高

接触判定用足端 z 高度（参考数据里没有可用的接触力列，10-26 已确认
gait_amp_4000_clean.npz 接触力全 0，所以两侧统一用同一套足端算法）。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python analyze_knee.py --robot <npz> [--ref <npz>]
"""
import argparse
import glob
import os
import sys

import numpy as np

# G1 29-DOF 关节顺序（与 watch_limp.py 一致，[0:6] 左腿 / [6:12] 右腿）
KNEE_L, KNEE_R = 3, 9
LEG_L, LEG_R = slice(0, 6), slice(6, 12)

# 人类步行文献参照带（数量级校验用，不是硬阈值）
HUMAN = {
    "knee_mean": (15, 30),      # deg, 1.0-1.4 m/s
    "knee_min": (2, 15),        # deg, 支撑中期近乎伸直
    "knee_rom": (50, 70),       # deg
    "duty_supp": (0.55, 0.68),  # 支撑相占比
    "cadence": (95, 125),       # steps/min（双腿），1.0-1.4 m/s
}


def contact_from_foot_z(foot_z, fps=50.0):
    """用足端高度判定支撑相。两侧（参考/机器人）必须用同一套算法，
    跨方法比较时换算法得到的差异全是算法差，不是行为差。"""
    thr = np.percentile(foot_z, 5) + 0.15 * (np.percentile(foot_z, 95) - np.percentile(foot_z, 5))
    return foot_z <= thr


def knee_stats(q_deg, foot_z=None):
    """q_deg: (T, N, D) 关节角（度）；foot_z: (T, N) 用于分支撑/摆动相。"""
    T, N, D = q_deg.shape
    out = {}
    kl = q_deg[:, :, KNEE_L]
    kr = q_deg[:, :, KNEE_R]

    # 逐 env 统计再平均，避免不同 env 相位不同互相平均掉
    def per_env(fn):
        return float(np.mean([fn(kl[:, j], kr[:, j]) for j in range(N)]))

    out["knee_mean"] = per_env(lambda a, b: 0.5 * (np.mean(a) + np.mean(b)))
    out["knee_min"] = per_env(lambda a, b: 0.5 * (np.percentile(a, 2) + np.percentile(b, 2)))
    out["knee_rom"] = per_env(lambda a, b: 0.5 * (np.max(a) - np.min(a) + np.max(b) - np.min(b)))

    if foot_z is not None:
        cl = contact_from_foot_z(foot_z)
        # 支撑相的屈膝角：蹲着走的话这个数会明显偏大
        vals = []
        for j in range(N):
            for side in (kl[:, j], kr[:, j]):
                m = cl[:, j] if side is kl[:, j] else cl[:, j]
                if m.sum() > 5:
                    vals.append(np.mean(side[m]))
        out["knee_stance"] = float(np.mean(vals)) if vals else float("nan")
        out["duty_supp"] = float(np.mean(cl))
    else:
        out["knee_stance"] = float("nan")
        out["duty_supp"] = float("nan")
    return out


def gait_from_foot(foot_z, dt=0.02):
    """步频/步长：足端高度极小值计数。"""
    c = contact_from_foot_z(foot_z)
    d = np.diff(c.astype(int))
    steps = int(np.sum(d == 1))
    dur = c.shape[0] * dt
    cad = steps / dur * 60.0 if dur > 0 else float("nan")
    return {"steps": steps, "cadence": cad}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robot", required=True, help="机器人 eval npz")
    ap.add_argument("--ref", default=None, help="参考动作 npz")
    ap.add_argument("--fps", type=float, default=50.0)
    args = ap.parse_args()

    dt = 1.0 / args.fps
    d = np.load(args.robot, allow_pickle=True)
    keys = list(d.keys())
    print("== ROBOT %s ==" % os.path.basename(args.robot))
    print("  keys: %s" % ", ".join(keys[:20]))

    q = d["dof_pos"] if "dof_pos" in keys else d.get("joint_pos")
    if q is None:
        print("  没有关节角列，无法诊断")
        return 1
    qd = np.degrees(q)

    foot_z = None
    for k in ("foot_pos_z", "foot_z", "feet_pos_z"):
        if k in keys:
            foot_z = d[k]
            break
    if foot_z is None and "foot_positions" in keys:
        foot_z = d["foot_positions"][:, :, 2] if d["foot_positions"].shape[-1] >= 3 else None

    st = knee_stats(qd, foot_z)
    for k in ("knee_mean", "knee_min", "knee_rom", "knee_stance", "duty_supp"):
        v = st[k]
        ref = HUMAN.get(k)
        band = ""
        if ref and not np.isnan(v):
            band = "  人类%s  OK" % ("%.0f-%.0f" % ref if ref[0] < 1 else "%.0f-%.0f deg" % ref)
            band += "" if (ref[0] <= v <= ref[1]) else "  <<< 偏离"
        print("  %-12s %7.1f%s" % (k, v, band))

    if foot_z is not None:
        g = gait_from_foot(foot_z.reshape(foot_z.shape[0], -1)[:, 0] if foot_z.ndim == 3 else foot_z[:, 0], dt)
        print("  %-12s %7.1f steps/min  人类 95-125" % ("cadence", g["cadence"]))

    rms = np.sqrt(np.mean(d["torque"] ** 2, axis=(0, 1))) if "torque" in keys else None
    if rms is not None:
        ai = lambda a, b: abs(a - b) / (0.5 * (abs(a) + abs(b))) * 100 if (abs(a) + abs(b)) > 1e-9 else float("nan")
        per = [ai(float(rms[k]), float(rms[k + 6])) for k in range(6)]
        print("  leg_ai_pct    %7.1f  (逐关节均值，与 limp_curve.csv 同口径)" % np.mean(per))

    if args.ref and os.path.exists(args.ref):
        print("\n== REF %s ==" % os.path.basename(args.ref))
        rd = np.load(args.ref, allow_pickle=True)
        rk = list(rd.keys())
        rq = rd["dof_pos"] if "dof_pos" in rk else rd.get("joint_pos")
        if rq is not None:
            rqn = np.degrees(rq)
            if rqn.ndim == 2:
                rqn = rqn[:, None, :]
            rst = knee_stats(rqn)
            print("  knee_mean %7.1f" % rst["knee_mean"])
            print("  knee_min  %7.1f" % rst["knee_min"])
            print("  knee_rom  %7.1f" % rst["knee_rom"])
            print("\n  --> 机器人 knee_min %.1f vs 参考 %.1f  (差 %.1f deg)"
                  % (st["knee_min"], rst["knee_min"], st["knee_min"] - rst["knee_min"]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
