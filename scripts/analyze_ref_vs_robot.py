# -*- coding: utf-8 -*-
"""参考数据集 vs 机器人：同速度段的步频 / 步长对比。

回答 2026-10-26 用户猜想："顿挫感是不是因为展示速度段没有对应参考数据，
机器人没有可以模仿的对象？" —— 把机器人和参考人类放在相同速度区间里比。

口径（项目约定）：
  freq   = 单脚每秒着地次数 = gait cycle 频率 (Hz)
  stride = v / freq（一个 cycle 的距离）
  step   = stride / 2

参考侧：足端 z 极小值 = 着地（pkl 无接触力；key_body_pos 里取平均高度
        最低的两个 body 作为左右脚）。
机器人侧：接触力 foot_force_z 上升沿（与 analyze_cadence.py 同口径）。

用法（服务器）：
    /data/envs/isaaclab/bin/python /data/setup/analyze_ref_vs_robot.py
"""
import glob
import os
import pickle

import numpy as np

MOTION_DIR = ("/data/legged_lab/source/legged_lab/legged_lab/data/"
              "MotionData/g1_29dof/amp/walk_and_run")
EVAL_DIR = "/data/eval_out"

BINS = [(0.8, 1.3), (1.3, 1.8), (1.8, 2.3), (2.3, 3.0)]


def arr(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)


# ---------------- 参考数据集 ----------------

def ref_strides():
    """逐 clip 检测着地事件，返回 [(speed_at_cycle, cycle_dur), ...]。"""
    events = []  # (v, cycle_dur, clip)
    for p in sorted(glob.glob(os.path.join(MOTION_DIR, "*.pkl"))):
        d = pickle.load(open(p, "rb"))
        fps = float(np.asarray(d["fps"]).ravel()[0]) if not isinstance(d["fps"], float) \
            else d["fps"]
        pos = arr(d["root_pos"])
        kb = arr(d["key_body_pos"])  # (T, 6, 3)
        # 速度：root_pos 差分 + 滑动平均
        vel = np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) * fps
        vel = np.concatenate([[vel[0]], vel])
        k = max(3, int(0.2 * fps))
        vel = np.convolve(vel, np.ones(k) / k, mode="same")
        # 左右脚 = 平均 z 最低的两个 key body
        feet = np.argsort(kb[:, :, 2].mean(axis=0))[:2]
        for f in feet:
            z0 = kb[:, f, 2]
            z = np.convolve(z0, np.ones(3) / 3, mode="same")
            zmin, zmax = float(z.min()), float(z.max())
            if zmax - zmin < 0.02:  # 几乎不抬脚（站立/侧移片段），跳会误检
                pass
            thr_lo = np.percentile(z, 30)
            refr = int(0.3 * fps)
            mins = []
            for i in range(1, len(z) - 1):
                if z[i] <= z[i - 1] and z[i] < z[i + 1] and z[i] <= thr_lo:
                    if mins and (i - mins[-1]) < refr:
                        # 不应期内：保留更低的那个
                        if z[i] < z[mins[-1]]:
                            mins[-1] = i
                        continue
                    mins.append(i)
            if len(mins) < 3:
                continue
            mins = np.array(mins)
            durs = np.diff(mins) / fps
            for i in range(len(durs)):
                if 0.3 < durs[i] < 2.0:
                    v = vel[mins[i]:mins[i + 1]].mean()
                    events.append((v, durs[i], os.path.basename(p)))
    return events


def ref_table(events):
    print("\n=== 参考数据集（LAFAN1 重定向 G1，30 clip） ===")
    print(f"{'速度段':>12}{'cycles':>8}{'步频Hz':>8}{'步长m':>8}  备注")
    for lo, hi in BINS:
        sel = [e for e in events if lo <= e[0] < hi]
        if not sel:
            print(f"{lo:.1f}-{hi:.1f} m/s{'0':>5}   无数据")
            continue
        f = 1.0 / np.mean([e[1] for e in sel])
        v = np.mean([e[0] for e in sel])
        stride = v / f
        clips = {e[2].split("_stageii")[0] for e in sel}
        kind = "行走(转弯)" if hi <= 1.8 else "跑步/变速"
        print(f"{lo:.1f}-{hi:.1f} m/s{len(sel):>5}{f:>8.2f}{stride:>8.2f}"
              f"  {kind}，来自 {len(clips)} 个 clip")
    return


# ---------------- 机器人 ----------------

def contact_runs(force, thr):
    c = force > thr
    if c.sum() == 0 or (~c).sum() == 0:
        return np.array([]), np.array([])
    on = np.where(~c[:-1] & c[1:])[0] + 1
    off = np.where(c[:-1] & ~c[1:])[0] + 1
    pairs_on, pairs_off, j = [], [], 0
    for i in on:
        while j < len(off) and off[j] <= i:
            j += 1
        if j < len(off):
            pairs_on.append(i)
            pairs_off.append(off[j])
            j += 1
    return np.array(pairs_on), np.array(pairs_off)


def _freq_from_foot_z(z, fps):
    """足端 z 极小值 = 着地。与参考侧同算法（跨方法比较时用同一把尺）。"""
    z = np.convolve(z, np.ones(3) / 3, mode="same")
    thr_lo = np.percentile(z, 30)
    refr = int(0.3 * fps)
    mins = []
    for i in range(1, len(z) - 1):
        if z[i] <= z[i - 1] and z[i] < z[i + 1] and z[i] <= thr_lo:
            if mins and (i - mins[-1]) < refr:
                if z[i] < z[mins[-1]]:
                    mins[-1] = i
                continue
            mins.append(i)
    if len(mins) < 3:
        return np.nan
    durs = np.diff(np.array(mins)) / fps
    ok = (durs > 0.3) & (durs < 2.0)
    if ok.sum() < 2:
        return np.nan
    return 1.0 / np.mean(durs[ok])


def robot_rows(path, warmup=0.3):
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    vels = d["velocities"]
    lvx, lvy = d["lin_vel_x"], d["lin_vel_y"]
    fz = d["foot_force_z"] if "foot_force_z" in d.files else None
    fzc = d["foot_z"] if "foot_z" in d.files else None
    use_force = fz is not None and float(np.abs(fz).max()) > 1e-6
    n = int(len(lvx) * warmup)
    rows = []
    for j in range(len(vels)):
        v = float(np.mean(np.hypot(lvx[n:, j], lvy[n:, j])))
        freqs = []
        for k in range(2):
            if use_force:
                ff = fz[n:, j, k]
                thr = max(0.15 * np.percentile(ff, 90), 1e-3)
                on, off = contact_runs(ff, thr)
                if len(on) < 3:
                    continue
                dur = (on[1:] - on[:-1]) * dt
                ok = (dur > 0.15) & (dur < 2.0)
                if ok.sum() >= 2:
                    freqs.append(1.0 / np.mean(dur[ok]))
            elif fzc is not None:
                f1 = _freq_from_foot_z(fzc[n:, j, k], 1.0 / dt)
                if not np.isnan(f1):
                    freqs.append(f1)
        f = float(np.mean(freqs)) if freqs else np.nan
        rows.append((float(vels[j]), v, f, v / f if f > 0 else np.nan,
                     "力" if use_force else "足z"))
    return rows


def robot_table():
    print("\n=== 机器人（gait_*.npz；优先接触力，力信号缺失时用足端z同算法） ===")
    print(f"{'模型':<22}{'指令':>6}{'实测v':>7}{'步频Hz':>8}{'步长m':>8}{'口径':>6}")
    for tag in ("amp_4000_clean", "lam050", "lam075", "ppo_baseline"):
        p = os.path.join(EVAL_DIR, f"gait_{tag}.npz")
        if not os.path.exists(p):
            continue
        for cmd, v, f, st, m in robot_rows(p):
            if np.isnan(f):
                continue
            print(f"{tag:<22}{cmd:>6.1f}{v:>7.2f}{f:>8.2f}{st:>8.2f}{m:>6}")


def main():
    # 调试：看一个 clip 的足端 z 范围和检出的着地次数
    if os.environ.get("DEBUG_REF"):
        p = sorted(glob.glob(os.path.join(MOTION_DIR, "*.pkl")))[0]
        d = pickle.load(open(p, "rb"))
        kb = arr(d["key_body_pos"])
        fps = float(np.asarray(d["fps"]).ravel()[0]) if not isinstance(d["fps"], float) \
            else d["fps"]
        print(f"[DEBUG] {os.path.basename(p)} fps={fps} key_body z 均值={kb[:,:,2].mean(axis=0).round(3)}")
        feet = np.argsort(kb[:, :, 2].mean(axis=0))[:2]
        for f in feet:
            z = kb[:, f, 2]
            print(f"  body{f}: z min={z.min():.3f} max={z.max():.3f} p30={np.percentile(z,30):.3f}")
    events = ref_strides()
    print(f"参考侧共检出 {len(events)} 个 gait cycle")
    ref_table(events)
    robot_table()
    print("\n口径：freq=单脚 gait cycle Hz；stride=v/freq（一个 cycle 的距离）")
    print("对照方法：同一速度段内比 步频/步长 —— 机器人落在参考分布附近")
    print("即'在该速度下尽力模仿了数据集中的人类运动'。")


if __name__ == "__main__":
    main()
