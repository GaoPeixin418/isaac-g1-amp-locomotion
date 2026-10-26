# -*- coding: utf-8 -*-
"""探测 AMP 参考数据集（LAFAN1 重定向到 G1 的 pkl）：每个 clip 的速度分布与步频。

输出：
1. pkl 键名（只看第一个文件）
2. 每个 clip：时长、根速度均值/分位数、速度区间占比、步频估计
3. 全数据集速度直方图（回答 0.5-1.5 与 2.0 附近是否有数据）

步频估计：用膝关节角度（knee）过零点计数，或腿髋 pitch 振荡。
参考数据没有接触力，用关节振荡近似（摆动周期）。

用法（服务器）：
    /data/envs/isaaclab/bin/python /data/setup/probe_motion_data.py
"""
import glob
import os
import pickle

import numpy as np

DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"

files = sorted(glob.glob(os.path.join(DIR, "*.pkl")))
print(f"找到 {len(files)} 个 pkl")
if not files:
    raise SystemExit(1)

d0 = pickle.load(open(files[0], "rb"))
print(f"\n[键名] {os.path.basename(files[0])}:")
if isinstance(d0, dict):
    for k, v in d0.items():
        try:
            print(f"   {k:22s} {getattr(v, 'shape', type(v))}")
        except Exception:
            print(f"   {k:22s} ?")
else:
    print("   顶层不是 dict:", type(d0))

def arr(x):
    try:
        import torch
        if isinstance(x, torch.Tensor):
            return x.detach().cpu().numpy()
    except Exception:
        pass
    return np.asarray(x)

def get_speed_fps(d):
    """(speed_per_frame, fps)。优先 root_lin_vel，否则 root_pos 差分。"""
    ks = d.keys()
    fps = 30.0
    if "fps" in ks:
        f = d["fps"]
        fps = float(np.asarray(f).ravel()[0])
    for key in ("root_lin_vel", "root_vel_w", "lin_vel"):
        if key in ks:
            v = arr(d[key])
            return np.linalg.norm(v[:, :2], axis=1), fps
    if "root_pos" in ks:
        pos = arr(d["root_pos"])
        vel = np.diff(pos[:, :2], axis=0) * fps
        return np.linalg.norm(vel, axis=1), fps
    return None, fps

def est_cadence(d, fps):
    """用膝关节角度振荡估计单脚 gait cycle 频率 (Hz)。"""
    for key in ("dof_pos", "joint_pos"):
        if key in d.keys():
            q = arr(d[key])
            break
    else:
        return np.nan
    # G1 29dof：找名字里带 knee 的列；没名字就退回用方差最大的腿关节
    names = d.get("dof_names") or d.get("joint_names")
    if names is not None:
        names = [str(x) for x in names]
        knees = [i for i, n in enumerate(names) if "knee" in n.lower()]
    else:
        knees = []
    if not knees:
        knees = [int(np.argmax(q.std(axis=0)))]
    k = knees[0]
    x = q[:, k] - q[:, k].mean()
    # 上升过零点 = 一个 cycle
    up = np.where((x[:-1] < 0) & (x[1:] >= 0))[0]
    if len(up) < 3:
        return np.nan
    period = np.mean(np.diff(up)) / fps
    return 1.0 / period if period > 0 else np.nan

bins = [(0.0, 0.3), (0.3, 0.8), (0.8, 1.3), (1.3, 1.8), (1.8, 2.3), (2.3, 3.0), (3.0, 9.0)]
all_speeds = []
print(f"\n{'clip':<44}{'时长s':>7}{'均速':>7}{'p50':>6}{'p90':>6}{'步频Hz':>8}   区间占比% (0-0.3 / .3-.8 / .8-1.3 / 1.3-1.8 / 1.8-2.3 / 2.3-3 / 3+)")
print("-" * 130)
for p in files:
    try:
        d = pickle.load(open(p, "rb"))
        sp, fps = get_speed_fps(d)
        if sp is None:
            print(f"{os.path.basename(p):<44} 无速度键")
            continue
        dur = len(sp) / fps
        cad = est_cadence(d, fps)
        all_speeds.append(sp)
        pct = [100.0 * np.mean((sp >= lo) & (sp < hi)) for lo, hi in bins]
        pct_s = " ".join(f"{x:5.1f}" for x in pct)
        nm = os.path.basename(p).replace("_stageii.pkl", "")[:42]
        print(f"{nm:<44}{dur:>7.1f}{sp.mean():>7.2f}"
              f"{np.percentile(sp,50):>6.2f}{np.percentile(sp,90):>6.2f}"
              f"{cad:>8.2f}   {pct_s}")
    except Exception as e:
        print(f"{os.path.basename(p):<44} 解析失败: {e}")

if all_speeds:
    sp = np.concatenate(all_speeds)
    print("-" * 130)
    print(f"\n全集 {len(sp)} 帧。速度区间占比（帧加权）:")
    for lo, hi in bins:
        c = np.mean((sp >= lo) & (sp < hi))
        print(f"  [{lo:.1f},{hi:.1f}) m/s: {100*c:5.1f}% {'#' * int(c * 60)}")
    for v in (1.2, 2.0):
        c = np.mean(np.abs(sp - v) < 0.2)
        print(f"  |v-{v}|<0.2 帧占比: {100*c:.1f}%")
