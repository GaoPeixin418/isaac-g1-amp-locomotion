"""顿挫感诊断：把"走得一顿一顿"翻译成可证伪量。

面试官的原话是"步幅感觉有问题，有种一顿一顿的感觉"、看着"匆忙"。
步频（analyze_cadence.py）只回答了"步子碎不碎"，没回答"顺不顺"。
顿挫的本质是**前进速度在步态周期内起伏**——人走路也有起伏，但小。

三个指标：
  CV        = 稳态段 lin_vel_x 的 std/mean          整体波动
  起伏幅度   = (P90 - P10) / mean                    一个周期内的快慢差
  周期强度   = 速度信号自相关在步频处的峰值(0~1)      起伏有多"规律"
              → 越接近 1 说明周期性起伏越明显，视觉上越"一顿一顿"

人类参照（文献量级，仅判数量级，不作为硬结论）：
  常速行走的速度起伏 CV ≈ 0.07~0.12。

用法：/data/envs/isaaclab/bin/python /data/setup/analyze_stutter.py
"""
import glob
import os

import numpy as np


def autocorr_peak(x, dt, lo=0.30, hi=2.00):
    """返回 (周期秒, 归一化自相关峰值)。x 已去均值。"""
    n = len(x)
    x = x - x.mean()
    if x.std() < 1e-9:
        return np.nan, np.nan
    f = np.fft.rfft(x, n=2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n].real
    ac /= ac[0]
    lo_i = max(1, int(lo / dt))
    hi_i = min(n - 1, int(hi / dt))
    if hi_i <= lo_i:
        return np.nan, np.nan
    seg = ac[lo_i:hi_i]
    i = int(np.argmax(seg))
    return (lo_i + i) * dt, float(seg[i])


def main():
    files = sorted(glob.glob("/data/eval_out/gait_*.npz"))
    keep = ["amp2000", "amp_2000", "amp_4000_clean", "lam025", "lam050", "lam075",
            "ppo_baseline"]
    print(f"{'模型':<16}{'指令':>6}{'实测v':>7}{'CV':>8}{'起伏幅度':>10}"
          f"{'周期s':>8}{'周期强度':>10}{'关节抖动':>10}")
    print("-" * 78)
    for p in files:
        b = os.path.basename(p)[5:-4]
        if not any(k in b for k in keep):
            continue
        try:
            d = np.load(p, allow_pickle=True)
        except Exception as e:
            print(f"{b:<16} 读取失败 {e}")
            continue
        dt = float(d["dt"])
        vels = d["velocities"]
        lvx = d["lin_vel_x"]
        jv = d["joint_vel"]
        n = int(len(lvx) * 0.3)
        for j in range(len(vels)):
            v = lvx[n:, j]
            m = float(v.mean())
            if m < 0.25:
                continue
            cv = float(v.std() / m)
            amp = float((np.percentile(v, 90) - np.percentile(v, 10)) / m)
            per, pk = autocorr_peak(v, dt)
            # 关节速度抖动：相邻步 joint_vel 差分 RMS，越低越顺
            jerk = float(np.sqrt(np.mean(np.diff(jv[n:, j, :], axis=0) ** 2)))
            flag = "  顿挫" if cv > 0.18 else ("" if cv > 0.08 else "  很平顺")
            print(f"{b:<16}{vels[j]:>6.1f}{m:>7.2f}{cv:>8.3f}{amp:>10.3f}"
                  f"{(per if per == per else 0):>8.2f}"
                  f"{(pk if pk == pk else 0):>10.2f}{jerk:>10.2f}{flag}")
    print("-" * 78)
    print("人类常速行走 CV ≈ 0.07~0.12（文献量级）。周期强度越接近 1 = 起伏越规律。")


if __name__ == "__main__":
    main()
