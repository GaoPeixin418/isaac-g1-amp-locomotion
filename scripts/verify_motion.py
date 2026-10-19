#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Motion 数据体检 —— 开训前必做的一关。

为什么必须有这一步：
    retarget 出来的数据是"运动学上像"，不代表"物理上能做"。
    常见的坑：
      1. 关节角超出 G1 实际限位    -> 仿真里关节被打回限位，动作变形，训出来必然丑
      2. 脚穿地 / 骨盆高度不对     -> G1 站立骨盆约 0.74-0.80 m，数据里若偏低说明腿是弯的
      3. NaN / 跳变帧             -> discriminator 直接学废
      4. 速度范围不对             -> 走(1.0-1.8 m/s) 和 跑(2.0-3.5 m/s) 混在一个 clip 里会打架
    这一步花 20 分钟，能省掉后面 6 小时的无效训练。

用法（不需要 Isaac Lab，纯 numpy/pandas，本地也能跑）：
    python scripts/verify_motion.py --dir ./motions/lafan1
    python scripts/verify_motion.py --file ./motions/lafan1/LAFAN1/g1/walk1_subject1.csv
"""
from __future__ import annotations

import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

# --- G1 参考值（用于体检判断，非硬约束；超出只报警，不阻断） -------------------
G1_STAND_PELVIS_H = 0.74      # G1 站立时骨盆离地高度（m），合理区间 0.70~0.85
PELVIS_H_RANGE = (0.55, 0.95)
# G1 主要关节软限位（rad），超出太多说明 retarget 有问题
JOINT_SOFT_LIMIT_HINT = {
    "hip_pitch":    (-1.05, 1.57),
    "hip_roll":     (-0.29, 1.05),
    "hip_yaw":      (-1.57, 1.57),
    "knee":         (-0.10, 2.25),
    "ankle_pitch":  (-0.87, 0.52),
    "ankle_roll":   (-0.44, 0.44),
    "waist_yaw":    (-1.05, 1.05),
    "waist_roll":   (-0.44, 0.44),
    "waist_pitch":  (-0.52, 0.52),
    "shoulder":     (-1.57, 1.57),
    "elbow":        (-1.05, 1.57),
}

WARN, OK, BAD = "!!", "ok", "XX"


def _classify(col: str) -> str | None:
    """把列名归类到已知关节，用于限位比对。"""
    c = col.lower()
    for key in JOINT_SOFT_LIMIT_HINT:
        if key in c:
            return key
    return None


def audit_file(path: str, verbose: bool = True) -> dict:
    name = os.path.basename(path)
    res = {"file": name, "issues": [], "info": {}}

    try:
        df = pd.read_csv(path)
    except Exception as e:  # noqa: BLE001
        res["issues"].append(f"{BAD} 读取失败: {e}")
        return res

    n_frames, n_cols = df.shape
    res["info"]["frames"] = n_frames
    res["info"]["cols"] = n_cols
    res["info"]["columns"] = list(df.columns)

    # 1) NaN / 无穷
    nan_cols = df.columns[df.isna().any()].tolist()
    if nan_cols:
        res["issues"].append(f"{BAD} 存在 NaN 列: {nan_cols[:8]}")
    if np.isinf(df.select_dtypes(include=[np.number]).to_numpy()).any():
        res["issues"].append(f"{BAD} 存在 inf 值")

    # 2) 帧数 / 时长（假设 30fps，可在命令行改）
    res["info"]["duration_s@30fps"] = round(n_frames / 30.0, 2)
    if n_frames < 30:
        res["issues"].append(f"{BAD} 帧数过少 ({n_frames})，AMP 学不到周期性")

    numeric = df.select_dtypes(include=[np.number])

    # 3) 关节限位体检
    limit_hits = []
    for col in numeric.columns:
        key = _classify(col)
        if key is None:
            continue
        lo, hi = JOINT_SOFT_LIMIT_HINT[key]
        cmin, cmax = float(numeric[col].min()), float(numeric[col].max())
        # 允许 25% 余量（retarget 常有轻微越界）
        margin = 0.25 * (hi - lo) + 0.05
        if cmin < lo - margin or cmax > hi + margin:
            limit_hits.append(f"{col}: [{cmin:.2f}, {cmax:.2f}] vs 参考 [{lo:.2f}, {hi:.2f}]")
    if limit_hits:
        res["issues"].append(
            f"{WARN} {len(limit_hits)} 个关节角明显越界（retarget 质量问题）:\n      "
            + "\n      ".join(limit_hits[:10])
        )

    # 4) 骨盆高度（找可能的 root height 列）
    height_cols = [c for c in df.columns if c.lower() in
                   ("pelvis_position_z", "root_height", "base_height", "pelvis_z", "position_z")]
    if height_cols:
        h = numeric[height_cols[0]]
        res["info"]["pelvis_h_mean"] = round(float(h.mean()), 3)
        res["info"]["pelvis_h_min"] = round(float(h.min()), 3)
        if not (PELVIS_H_RANGE[0] <= h.mean() <= PELVIS_H_RANGE[1]):
            res["issues"].append(
                f"{WARN} 骨盆平均高度 {h.mean():.3f} m 不在合理区间 {PELVIS_H_RANGE} "
                f"（腿可能一直是弯的，走起来会矮一截）"
            )
    else:
        res["info"]["pelvis_h_mean"] = None
        res["issues"].append(
            f"--  没找到骨盆高度列（找过 {height_cols or '常见命名'}），跳过高度检查；"
            f"建议在 motion viewer 里目视确认"
        )

    # 5) 前进速度（用 root 的 x/y 位移估算）
    pos_cols = {c.lower(): c for c in df.columns}
    vx = pos_cols.get("pelvis_position_x") or pos_cols.get("root_position_x") or pos_cols.get("position_x")
    vy = pos_cols.get("pelvis_position_y") or pos_cols.get("root_position_y") or pos_cols.get("position_y")
    if vx and vy:
        dx = np.diff(numeric[vx].to_numpy())
        dy = np.diff(numeric[vy].to_numpy())
        speed = np.sqrt(dx**2 + dy**2) * 30.0  # m/s @30fps
        res["info"]["speed_mean_mps"] = round(float(speed.mean()), 2)
        res["info"]["speed_max_mps"] = round(float(speed.max()), 2)
        s = res["info"]["speed_mean_mps"]
        if s < 0.3:
            res["issues"].append(f"{WARN} 平均速度 {s} m/s 过低 —— 这不是走/跑片段，是原地动作")
        elif s < 1.6:
            res["info"]["gait_guess"] = "walk"
        elif s < 3.5:
            res["info"]["gait_guess"] = "run"
        else:
            res["issues"].append(f"{WARN} 平均速度 {s} m/s 过高，G1 很难跟上（跑也到不了 3.5+）")
    else:
        res["info"]["speed_mean_mps"] = None

    # 6) 跳变检测（相邻帧差过大 = 数据有断裂）
    max_jump = 0.0
    for col in numeric.columns[:40]:
        d = np.abs(np.diff(numeric[col].to_numpy()))
        if len(d):
            max_jump = max(max_jump, float(np.percentile(d, 99.9)))
    res["info"]["p99.9_frame_jump"] = round(max_jump, 4)
    if max_jump > 0.5:
        res["issues"].append(f"{WARN} 存在帧间跳变 (p99.9={max_jump:.3f})，可能有数据断裂")

    if verbose:
        print(f"\n{'=' * 70}\n{name}")
        print(f"  帧数 {n_frames} | 列数 {n_cols} | 时长 {res['info']['duration_s@30fps']}s @30fps")
        print(f"  列名: {list(df.columns)[:12]}{' ...' if n_cols > 12 else ''}")
        if res["info"].get("pelvis_h_mean"):
            print(f"  骨盆高度 mean={res['info']['pelvis_h_mean']} min={res['info']['pelvis_h_min']}")
        if res["info"].get("speed_mean_mps"):
            print(f"  速度 mean={res['info']['speed_mean_mps']} m/s "
                  f"max={res['info']['speed_max_mps']} m/s -> 判定: {res['info'].get('gait_guess', '?')}")
        if res["issues"]:
            for it in res["issues"]:
                print(f"  {it}")
        else:
            print(f"  {OK} 未发现明显问题")

    return res


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="批量体检一个目录（递归找 csv）")
    ap.add_argument("--file", help="只体检单个文件")
    ap.add_argument("--max-files", type=int, default=20, help="目录模式最多看多少个文件")
    args = ap.parse_args()

    if not args.dir and not args.file:
        ap.error("至少给 --dir 或 --file")

    files: List[str] = []
    if args.file:
        files = [args.file]
    else:
        for root, _d, fs in os.walk(args.dir):
            for f in fs:
                if f.lower().endswith(".csv"):
                    files.append(os.path.join(root, f))
        files.sort()
        if len(files) > args.max_files:
            print(f"[info] 共 {len(files)} 个 csv，只看前 {args.max_files} 个")
            files = files[: args.max_files]

    if not files:
        print("!! 没找到任何 csv 文件。确认下载目录对不对。")
        return 1

    all_res = [audit_file(f) for f in files]

    bad = sum(1 for r in all_res if any(i.startswith(BAD) for i in r["issues"]))
    warn = sum(1 for r in all_res if any(i.startswith(WARN) for i in r["issues"]))
    print(f"\n{'=' * 70}\n体检汇总: {len(all_res)} 个文件, {bad} 个有硬错误, {warn} 个有告警")
    print("有硬错误(XX)的文件不要用；告警(!!)需要人眼在 motion viewer 里再确认一次。")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
