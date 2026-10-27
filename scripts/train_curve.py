#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""从训练日志里抽出关键曲线（零成本实时监控）

为什么需要它：rsl_rl 每个 iteration 都会把奖励分项打印一遍，
其中三项正好组成"跛行是否真被治好"的三角验证：

  leg_torque_asymmetry  奖励项本身（惩罚力度够不够）
  error_vel_xy          速度跟踪误差（有没有为了对称而走不动 / 走不稳）
  style                 风格分（有没有为了对称而丢掉人味）

三项要一起看。只看第一项就是 10-26 那次"指标满分但肉眼跛行"的翻版。

用法: python train_curve.py [日志文件] [输出csv]
"""
import re
import sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "/data/logs/train_fresh_sym.log"
OUT = sys.argv[2] if len(sys.argv) > 2 else "/data/progress/train_curve.csv"

KEYS = {
    "Episode_Reward/leg_torque_asymmetry": "torque_asym",
    "Episode_Reward/air_time_variance": "air_var",
    "Episode_Reward/feet_clearance": "clearance",
    "Metrics/base_velocity/error_vel_xy": "err_vel_xy",
    "Episode_Reward/style": "style",
    "Episode_Termination/bad_orientation": "fall",
    "Episode_Reward/termination_penalty": "term",
}

cur = None
rows = []
with open(LOG, encoding="utf-8", errors="ignore") as f:
    for line in f:
        m = re.search(r"Learning iteration\s+(\d+)\s*/\s*(\d+)", line)
        if m:
            cur = int(m.group(1))
            continue
        if cur is None:
            continue
        s = line.strip()
        if ":" not in s:
            continue
        k, _, v = s.rpartition(":")
        k = k.strip()
        if k in KEYS:
            rows.append((cur, KEYS[k], v.strip()))

agg = {}
for it, k, v in rows:
    try:
        fv = float(v)
    except ValueError:
        continue
    agg.setdefault(it, {}).setdefault(k, []).append(fv)

cols = ["torque_asym", "air_var", "clearance", "err_vel_xy", "style", "fall", "term"]
with open(OUT, "w", encoding="utf-8") as f:
    f.write("iter," + ",".join(cols) + "\n")
    for it in sorted(agg):
        d = agg[it]
        f.write(str(it) + "," + ",".join(
            ("%.4f" % (sum(d[c]) / len(d[c])) if c in d else "") for c in cols
        ) + "\n")
print("wrote %s  (%d iters)" % (OUT, len(agg)))
