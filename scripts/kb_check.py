#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""澄清参考数据里 key_body_pos 的坐标系：世界坐标 还是 相对 root。

为什么必须查清（2026-10-27 晚）
-----------------------------------------------------------------------------
ref_slip.py 算出"支撑相脚速 1.15 m/s"，判定参考数据打滑。但这个结论依赖
key_body_pos 是**世界坐标**。如果它其实是**相对 root 的局部坐标**，那么脚在
支撑相相对身体向后移动、速率刚好约等于步行速度 —— 1.15 m/s 非但不是打滑，
反而是完全正常的相对运动。

不查清就下结论，就是这项目第 6 次指标陷阱。

判据
-----------------------------------------------------------------------------
  key_body 的 z 均值接近 0        -> 世界坐标（脚在地面）
  key_body 的 z 均值接近 -0.77    -> 相对 root（root 高 0.765，脚在其下方）
"""
import glob
import pickle

import numpy as np

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"


def main():
    fs = sorted(glob.glob(REF_DIR + "/*.pkl"))
    if not fs:
        print("没找到 clip")
        return 1
    for f in fs[:3]:
        p = pickle.load(open(f, "rb"))
        kb = np.asarray(p["key_body_pos"], dtype=np.float64)
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        print("clip:", f.split("/")[-1])
        print("  keys:", list(p.keys()))
        print("  key_body_pos shape:", kb.shape,
              " root_pos z mean=%.3f" % rp[:, 2].mean())
        for i in range(kb.shape[1]):
            z = kb[:, i, 2]
            rxy = np.hypot(kb[:, i, 0], kb[:, i, 1])
            print("    idx%d: z mean=%8.3f min=%8.3f max=%8.3f | |xy| mean=%6.3f"
                  % (i, z.mean(), z.min(), z.max(), rxy.mean()))
        print("  root xy[:3]:", np.round(rp[:3, :2], 3).tolist())
        print("  kb0   xy[:3]:", np.round(kb[:3, 0, :2], 3).tolist())
        print()
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
