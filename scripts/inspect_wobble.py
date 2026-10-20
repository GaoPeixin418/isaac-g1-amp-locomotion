# -*- coding: utf-8 -*-
"""
量化"颠簸/站不稳"，只用现有 npz 里已有的字段。

为什么不加采高度/姿态角：
   现有 npz 只存了 lin_vel_x/y、foot_z、foot_force_z、joint_vel、torque。
   躯干高度和姿态角没存，要补得重跑评估（2 min/组，但 PPO 基线正占着 GPU）。
   好在"颠簸"在前进方向上会直接表现为速度波动，侧向会表现为漂移，
   所以用现有四个代理指标已经能定位"哪一档最不稳"。

四个代理指标：
   vel_jitter   相邻帧前进速度差分的标准差  —— 越大越颠
   lateral      侧向速度的 RMS              —— 越大越晃
   joint_act    关节角速度的总幅值 RMS      —— 越大动作越抽搐
   force_jerk   足底支撑力差分的 RMS        —— 越大落脚冲击越猛
"""
import sys
import numpy as np

WARMUP = 250   # 5 s @ 0.02 s


def stats(tag, path):
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    # 不能取首帧：机器人在第 0 帧可能还没收到指令（实测 2.5 档首帧记录到 0.0）
    vels = np.median(d["cmd_x"][300:], axis=0)
    vx = d["lin_vel_x"][WARMUP:]
    vy = d["lin_vel_y"][WARMUP:]
    jv = d["joint_vel"][WARMUP:]
    ff = d["foot_force_z"][WARMUP:]
    T = vx.shape[0]

    print(f"\n### {tag}")
    print("| 指令 (m/s) | 速度抖动 (m/s) | 侧向漂移 (m/s) | 关节动作量 | 落脚冲击 (N/帧) |")
    print("|" + "|".join(["---"] * 5) + "|")
    for i in range(len(vels)):
        c = float(vels[i])
        vj = float(np.std(np.diff(vx[:, i])))
        lat = float(np.sqrt(np.mean(vy[:, i] ** 2)))
        ja = float(np.sqrt(np.mean(jv[:, i] ** 2)))
        fj = float(np.sqrt(np.mean(np.diff(ff[:, i], axis=0) ** 2)))
        print(f"| {c:.1f} | {vj:.3f} | {lat:.3f} | {ja:.2f} | {fj:.0f} |")

    vj_all = [float(np.std(np.diff(vx[:, i]))) for i in range(len(vels))]
    lat_all = [float(np.sqrt(np.mean(vy[:, i] ** 2))) for i in range(len(vels))]
    ja_all = [float(np.sqrt(np.mean(jv[:, i] ** 2))) for i in range(len(vels))]
    print(f"| **均值** | **{np.mean(vj_all):.3f}** | **{np.mean(lat_all):.3f}** | "
          f"**{np.mean(ja_all):.2f}** | — |")
    print(f"[dt={dt:.3f}s  T={T}帧 = {T*dt:.0f}s  已去前 {WARMUP} 帧]")


def main():
    for arg in sys.argv[1:]:
        tag, _, p = arg.partition("=")
        stats(tag, p)
    return 0


if __name__ == "__main__":
    sys.exit(main())
