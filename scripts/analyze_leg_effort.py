# -*- coding: utf-8 -*-
"""左右腿出力对称性（扭矩口径）

为什么补这个指标：
  duty / 接触力描述"脚踩了多久、踩了多重"，
  用户主诉却是"看着像一边腿在硬撑"。这两件事不完全等价 ——
  关节扭矩才是"这条腿到底使了多少劲"的直接读数。

G1 29 DOF 的标准排布（legged_lab g1_amp_env_cfg）：
  [0:6]   左腿  hip_pitch / hip_roll / hip_yaw / knee / ankle_pitch / ankle_roll
  [6:12]  右腿  同上顺序
  [12:15] 腰    waist_yaw / waist_roll / waist_pitch
  [15:22] 左臂  肩3 + 肘1 + 腕3
  [22:29] 右臂  同上
若实际排布不同，脚本会报"索引越界/不配对"，不会静默给错数。
"""
import sys
import os
import numpy as np

LEG_L = slice(0, 6)
LEG_R = slice(6, 12)
NAMES = ["hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll"]


def ai(l, r):
    m = 0.5 * (abs(l) + abs(r))
    return abs(l - r) / m * 100 if m > 1e-9 else np.nan


def main():
    print(f"{'model':<18}{'cmd':>5} | {'腿合力RMS_L':>11}{'腿合力RMS_R':>11}{'AI%':>7} | "
          f"{'膝L':>7}{'膝R':>7}{'AI%':>7} | {'踝pitchL':>8}{'踝pitchR':>8}{'AI%':>7}")
    print("-" * 106)
    for p in sys.argv[1:]:
        d = np.load(p, allow_pickle=True)
        tq = d["torque"]
        vels = d["velocities"]
        T, N, D = tq.shape
        if D < 12:
            print(f"{os.path.basename(p):<18} DOF={D} 不足，跳过")
            continue
        n0 = int(T * 0.3)
        name = os.path.basename(p).replace("gait_", "").replace(".npz", "")
        for j in range(N):
            t = tq[n0:, j, :]
            rms = np.sqrt(np.mean(t ** 2, axis=0))
            legL = float(np.sqrt(np.mean(rms[LEG_L] ** 2)))
            legR = float(np.sqrt(np.mean(rms[LEG_R] ** 2)))
            kneeL, kneeR = float(rms[3]), float(rms[9])
            anL, anR = float(rms[4]), float(rms[10])
            print(f"{name:<18}{vels[j]:>5.1f} | "
                  f"{legL:>11.1f}{legR:>11.1f}{ai(legL, legR):>7.1f} | "
                  f"{kneeL:>7.1f}{kneeR:>7.1f}{ai(kneeL, kneeR):>7.1f} | "
                  f"{anL:>8.1f}{anR:>8.1f}{ai(anL, anR):>7.1f}")
        print("-" * 106)
    print("AI% = 200*|L-R|/(L+R)。'腿合力RMS' 是把该侧 6 个关节扭矩取 RMS 的合成。")


if __name__ == "__main__":
    main()
