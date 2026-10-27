#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 watch_v2 的同一套口径，给任意 npz 出一行完整指标（含综合分）。

用途：在 v2 训练开跑时，先把 v1 的最终模型（model_2999）量一遍，
得到"改进前"的四指标基线。没有基线就无法判断 v2 到底有没有治好。

用法
-----------------------------------------------------------------------------
    /data/envs/isaaclab/bin/python baseline_v2.py /data/eval_out/gait_model_2999.npz
"""
import sys

sys.path.insert(0, "/data/setup")
from watch_v2 import measure, score_of  # noqa: E402

if __name__ == "__main__":
    for p in sys.argv[1:]:
        m = measure(p)
        if m is None:
            print("%-40s 数据不足" % p)
            continue
        sc = score_of(m)
        print("%-40s legAI=%.1f%% (hip %.1f / knee %.1f)  cad=%.0f  stride=%.2f  "
              "knee=%.1fdeg  h=%.3fm  v_act=%.2f  score=%.1f"
              % (p.split("/")[-1], m["leg_ai"], m["hip_ai"], m["knee_ai"],
                 m["cadence"], m["stride"], m["knee_deg"], m["height"],
                 m["v_act"], sc))
