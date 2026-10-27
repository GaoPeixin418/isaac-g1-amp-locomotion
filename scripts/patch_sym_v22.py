#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.2 补丁（2026-10-27 深夜）：只动两个数

背景：v2 / v2.1 两轮从零训到 iter 800/1200 都没学会走路（v_act 0.05-0.07，
v1 同配方权重 -1.0 在 iter 800 已 v_act 0.97）。v2.1 回退 ang_vel_xy_l2 后
仍失败 => ang_vel 被排除，主嫌疑 = leg_work_asymmetry 权重 -2.0
（v1 的两倍，早期压制一切不对称的探索动作；站着不动时 |tau*omega|≈0 反而
零惩罚 => 最优解变成"站着别动"）。

改动：
  1. leg_work_asymmetry weight: -2.0 -> -1.0（回到 v1 实测可学走路的量级，
     tc=1.5 和 |tau*omega| 两个防 exploit 修复保留）
  2. feet_air_time threshold: 0.5 -> 0.4（回退 v2.1 的未验证改动，
     减少变量；碎步若复发再单独上这个旋钮）

用法: python patch_sym_v22.py [--check]
"""
import sys
from pathlib import Path

ENV_CFG = Path(
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
)

OLD_W = "func=mdp.leg_work_asymmetry,\n        weight=-2.0,"
NEW_W = "func=mdp.leg_work_asymmetry,\n        weight=-1.0,"


def airtime_seg(c: str) -> str:
    i = c.find("feet_air_time = RewTerm(")
    if i < 0:
        return ""
    return c[i:c.find("    )", i)]


def check():
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")
    w_ok = NEW_W in c
    seg = airtime_seg(c)
    t_ok = '"threshold": 0.4' in seg
    print("work_asym weight=-1.0 : %s" % ("在" if w_ok else "无(当前 %s)" % ("仍 -2.0" if OLD_W in c else "未知")))
    print("airtime threshold=0.4 : %s" % ("在" if t_ok else "无(seg=%r)" % seg[:200]))
    return w_ok and t_ok


def apply():
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")
    n = 0
    if OLD_W in c:
        c = c.replace(OLD_W, NEW_W, 1)
        n += 1
        print("== work_asym weight -2.0 -> -1.0")
    elif NEW_W in c:
        print("== work_asym 已是 -1.0，跳过")
    else:
        print("!! 没找到 work_asym weight 行，手工确认")
        sys.exit(1)

    seg = airtime_seg(c)
    if '"threshold": 0.5' in seg:
        new_seg = seg.replace('"threshold": 0.5', '"threshold": 0.4', 1)
        c = c.replace(seg, new_seg, 1)
        n += 1
        print("== feet_air_time threshold 0.5 -> 0.4")
    elif '"threshold": 0.4' in seg:
        print("== airtime 已是 0.4，跳过")
    else:
        print("!! airtime threshold 没匹配上（seg=%r），手工确认" % seg[:200])
        sys.exit(1)

    if n:
        ENV_CFG.write_text(c, encoding="utf-8")

    import py_compile
    py_compile.compile(str(ENV_CFG), doraise=True)
    print("编译通过")
    ok = check()
    print("结果: %s" % ("通过" if ok else "!! 未通过"))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if "--check" in sys.argv[1:]:
        sys.exit(0 if check() else 1)
    apply()
