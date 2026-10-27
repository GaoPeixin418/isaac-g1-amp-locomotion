#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""同速度下比较「机器人 vs 参考动作」的膝关节角度，判定"蹲"是谁造成的。

为什么这么设计
-----------------------------------------------------------------------------
用户主诉"膝盖大部分时间弯着 / 像快走"。要修它先得知道责任方：
  * 参考动作本来就屈膝大 -> 数据集/重定向的锅，改奖励无效
  * 参考正常而机器人屈膝大 -> RL 自己走捷径（蹲着走重心低、更稳、不容易触发
    termination_penalty=-50），改奖励/加约束有效

两条硬约束（都是踩过坑换来的）
-----------------------------------------------------------------------------
1. 速度必须匹配：屈膝角是速度的函数，跑步片段屈膝天然大。
   所以参考侧按 root 水平速率分档，只取与 eval 指令速度同档的帧。
2. 两侧用同一套角度口径：都是 dof_pos（弧度）-> 度，取同名关节 index。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python diag_crouch.py --ckpt <model.pt> [--speed 1.0]
"""
import argparse
import glob
import os
import pickle
import subprocess
import sys
import time

import numpy as np

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
EVAL = "/data/setup/eval_gait.py"
EVAL_OUT = "/data/eval_out"
KNEE_L, KNEE_R = 3, 9
HIP_PITCH_L, HIP_PITCH_R = 0, 6
ANKLE_PITCH_L, ANKLE_PITCH_R = 4, 10

HUMAN = {
    "knee_mean": (15, 30),
    "knee_min": (2, 15),
    "knee_rom": (50, 70),
}


def ref_frames_in_speed(lo, hi):
    """返回参考动作里速度落在 [lo,hi) 的膝角序列（度）。"""
    out = []
    for f in sorted(glob.glob(os.path.join(REF_DIR, "*.pkl"))):
        p = pickle.load(open(f, "rb"))
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        fps = float(np.asarray(p["fps"]).reshape(-1)[0])
        dt = 1.0 / fps
        v = np.diff(rp[:, :2], axis=0) / dt
        sp = np.hypot(v[:, 0], v[:, 1])
        sp = np.concatenate([sp, sp[-1:]])
        q = np.asarray(p["dof_pos"], dtype=np.float64)
        m = (sp >= lo) & (sp < hi)
        if m.sum() < 5:
            continue
        out.append(np.degrees(q[m]))
    return np.concatenate(out, axis=0) if out else None


def stats(q2):
    """q2: (T, D) 度。返回膝相关统计。"""
    kl, kr = q2[:, KNEE_L], q2[:, KNEE_R]
    km = 0.5 * (kl + kr)
    return {
        "knee_mean": float(km.mean()),
        "knee_min": float(np.percentile(km, 2)),
        "knee_max": float(np.percentile(km, 98)),
        "knee_rom": float(np.percentile(km, 98) - np.percentile(km, 2)),
        "knee_p25": float(np.percentile(km, 25)),
        "knee_p75": float(np.percentile(km, 75)),
        "hip_mean": float(np.mean(0.5 * (q2[:, HIP_PITCH_L] + q2[:, HIP_PITCH_R]))),
        "ankle_mean": float(np.mean(0.5 * (q2[:, ANKLE_PITCH_L] + q2[:, ANKLE_PITCH_R]))),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=600)
    args = ap.parse_args()

    tag = os.path.basename(args.ckpt).replace(".pt", "")
    ts = time.time()
    p = subprocess.run(
        ["/data/envs/isaaclab/bin/python", EVAL,
         "--task", "LeggedLab-Isaac-AMP-G1-v0",
         "--checkpoint", args.ckpt,
         "--num_envs", "1", "--steps", str(args.steps),
         "--velocities", str(args.speed), "--headless"],
        capture_output=True, text=True, cwd="/data/legged_lab",
    )
    if p.returncode != 0:
        print("[fail] rc=%d\n%s" % (p.returncode, (p.stdout or "")[-800:] + (p.stderr or "")[-800:]))
        return 1
    cands = [q for q in glob.glob(os.path.join(EVAL_OUT, "gait_*.npz")) if os.path.getmtime(q) > ts - 5]
    if not cands:
        print("[fail] 无新 npz")
        return 1
    npz = max(cands, key=os.path.getmtime)
    d = np.load(npz, allow_pickle=True)
    if "joint_pos" not in d.files:
        print("[fail] npz 里没有 joint_pos——eval_gait.py 的补丁没生效？")
        return 1
    q = np.degrees(d["joint_pos"])  # (T, N, D)
    T, N, D = q.shape
    n0 = int(T * 0.3)  # 丢掉启动暂态
    q2 = q[n0:].reshape(-1, D)
    rs = stats(q2)

    lo, hi = args.speed - 0.25, args.speed + 0.25
    rq = ref_frames_in_lo_hi = ref_frames_in_speed(lo, hi)
    print("\n== 机器人 %s @ %.1f m/s ==" % (tag, args.speed))
    print("  knee_mean %6.1f deg   (人类 %s)" % (rs["knee_mean"], "%.0f-%.0f" % HUMAN["knee_mean"]))
    print("  knee_min  %6.1f deg   (人类 %s)  <<< 蹲着走这个数会显著偏大"
          % (rs["knee_min"], "%.0f-%.0f" % HUMAN["knee_min"]))
    print("  knee_max  %6.1f deg" % rs["knee_max"])
    print("  knee_rom  %6.1f deg   (人类 %s)" % (rs["knee_rom"], "%.0f-%.0f" % HUMAN["knee_rom"]))
    print("  knee IQR  %6.1f - %.1f deg" % (rs["knee_p25"], rs["knee_p75"]))
    print("  hip_pitch_mean  %6.1f deg" % rs["hip_mean"])
    print("  ankle_pitch_mean %6.1f deg" % rs["ankle_mean"])

    if ref_frames_in_lo_hi is not None:
        rr = stats(ref_frames_in_lo_hi)
        print("\n== 参考动作 @ %.2f-%.2f m/s ==" % (lo, hi))
        print("  knee_mean %6.1f deg" % rr["knee_mean"])
        print("  knee_min  %6.1f deg" % rr["knee_min"])
        print("  knee_rom  %6.1f deg" % rr["knee_rom"])
        print("  hip_pitch_mean  %6.1f deg" % rr["hip_mean"])
        print("  ankle_pitch_mean %6.1f deg" % rr["ankle_mean"])
        print("\n== 判定 ==")
        dk = rs["knee_mean"] - rr["knee_mean"]
        print("  机器人比参考多屈 %.1f deg (%.1f vs %.1f)" % (dk, rs["knee_mean"], rr["knee_mean"]))
        if abs(dk) < 5:
            print("  --> 机器人与参考一致，蹲是参考数据自带的，改奖励无效应改参考")
        elif dk > 5:
            print("  --> 机器人比参考蹲得多，是 RL 走捷径，加伸展约束/调默认姿态有效")
        else:
            print("  --> 机器人比参考伸得直，蹲的问题不在这")
    return 0


if __name__ == "__main__":
    sys.exit(main())
