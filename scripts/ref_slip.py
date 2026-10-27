#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""参考动作：支撑相脚打滑检测 + root 位移/脚位移一致性。

为什么查这个（2026-10-27 晚，第 5 次被数据打脸后学乖了）
-----------------------------------------------------------------------------
前四次假设全被实测推翻：
  1. 跛行=触地时长不对称   -> 错，是出力不对称
  2. finetune 能修         -> 错，36.7%->37.1% 纹丝不动
  3. 合成指标能代表对称    -> 错，逐关节反向抵消
  4. "蹲着走"              -> 错，躯干高 0.762 参考 0.765，膝 16.7 参考 15.0
  5. 髋摆动不足            -> 错，参考 hipROM 13.3° 比机器人 20.9° 还小

剩下的解释：LAFAN1 重定向到 G1 时只转了关节角，**root 平移保持真人尺度**。
G1 腿长 ~0.65 m，真人腿长 ~0.9 m。同样的 root 速度 1.0 m/s，腿短的必须
提高步频 —— 那"碎步"就不是策略的锅，是**参考数据的尺度不匹配**。

判据
-----------------------------------------------------------------------------
支撑相里脚（世界坐标）的水平位移速率：
  < 0.15 m/s  正常（脚踩住不动，身体从上方越过）
  > 0.30 m/s  打滑 —— 参考数据本身是"滑冰"，判别器学到的风格就是错的

同时看：一个 gait cycle 内 root 前进距离 vs 脚相对 root 的前后摆幅。
  脚摆幅 * 2 ≈ root 前进  -> 一致，机器人做得到
  脚摆幅 * 2 << root 前进 -> 数据打滑，机器人只能靠提高步频

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python ref_slip.py
"""
import glob
import os
import pickle

import numpy as np
import sys

sys.path.insert(0, "/data/setup")
from gait_metrics2 import contact_hyst  # noqa: E402

REF_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"
FOOT_IDS = (0, 1)


def main():
    fs = sorted(glob.glob(os.path.join(REF_DIR, "*.pkl")))
    rows = []
    for f in fs:
        p = pickle.load(open(f, "rb"))
        rp = np.asarray(p["root_pos"], dtype=np.float64)
        fps = float(np.asarray(p["fps"]).reshape(-1)[0])
        kb = np.asarray(p["key_body_pos"], dtype=np.float64)  # (T, B, 3) 世界坐标
        dt = 1.0 / fps
        v = np.diff(rp[:, :2], axis=0) / dt
        sp = np.hypot(v[:, 0], v[:, 1])
        sp = np.concatenate([sp, sp[-1:]])

        # 只看接近 1.0 m/s 的连续片段
        m = (sp >= 0.8) & (sp < 1.3)
        idx = np.flatnonzero(m)
        if len(idx) < int(1.0 * fps):
            continue
        brk = np.flatnonzero(np.diff(idx) > 1)
        for s in np.split(idx, brk + 1):
            if len(s) < int(1.0 * fps):
                continue
            a, b = s[0], s[-1] + 1
            seg_v = float(sp[a:b].mean())
            slip_rates, swing_amp = [], []
            for fi in FOOT_IDS:
                fxy = kb[a:b, fi, :2]
                fz = kb[a:b, fi, 2]
                c = contact_hyst(fz, fps)
                # 支撑相内脚的水平位移速率
                if c.sum() > 3:
                    d = np.diff(fxy[c], axis=0)
                    if len(d):
                        slip_rates.append(float(np.mean(np.hypot(d[:, 0], d[:, 1]) / dt)))
                # 脚相对 root 的前后摆幅（投影到运动方向）
                rel = fxy - rp[a:b, :2]
                mv = rp[a:b, :2]
                dirv = mv[-1] - mv[0]
                if np.hypot(*dirv) > 0.05:
                    u = dirv / np.hypot(*dirv)
                    proj = rel @ u
                    swing_amp.append(float(np.percentile(proj, 95) - np.percentile(proj, 5)))
            if slip_rates:
                rows.append((os.path.basename(f), seg_v, float(np.mean(slip_rates)),
                             float(np.mean(swing_amp)) if swing_amp else float("nan"),
                             (b - a) / fps))

    if not rows:
        print("没有符合条件的片段")
        return 1
    print("== 参考动作 @0.8-1.3 m/s：支撑相脚打滑 ==")
    print("%-42s %7s %10s %12s %7s" % ("clip", "v", "slip_m/s", "swingAmp_m", "dur_s"))
    for r in sorted(rows, key=lambda x: -x[2])[:12]:
        print("%-42s %7.2f %10.3f %12.3f %7.1f" % r)
    sl = np.array([r[2] for r in rows])
    sa = np.array([r[3] for r in rows])
    print("\n支撑相脚速: 中位 %.3f m/s  均值 %.3f  p90 %.3f  (n=%d 段)"
          % (np.median(sl), sl.mean(), np.percentile(sl, 90), len(rows)))
    print("脚相对 root 前后摆幅: 中位 %.3f m" % np.nanmedian(sa))
    print("  -> 若摆幅*2 远小于 v/freq(=步长 ~1.2m)，说明 root 在打滑")
    print("\n判定: %s"
          % ("脚基本踩住，数据没打滑" if np.median(sl) < 0.15
             else ("轻微打滑，需留意" if np.median(sl) < 0.30 else "!! 明显打滑，参考数据本身有问题")))
    return 0


if __name__ == "__main__":
    sys.exit(main())
