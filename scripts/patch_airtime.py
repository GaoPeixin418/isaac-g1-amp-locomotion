#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""备用旋钮：把 feet_air_time 的 threshold 0.4 -> 0.5（治碎步）。

为什么它是步态尺度旋钮
-----------------------------------------------------------------------------
`feet_air_time_positive_biped` 的源码（velocity/mdp/rewards.py:58）：

    in_mode_time = 接触脚用 contact_time，腾空脚用 air_time
    reward = min over feet of (in_mode_time if single_stance)
    reward = clamp(reward, max=threshold)

即：单脚支撑期间，每帧奖励≈摆动脚已腾空的时长，**上限 threshold**。
所以 threshold 就是"希望一步腾空多久"——直接决定步态周期，是标准的
步态尺度旋钮。

安全边界（很重要，不然不敢调）
-----------------------------------------------------------------------------
* clamp 是 max 不是 min：腾空短时仍给部分奖励，**不会变成惩罚**，
  所以早期学习不会因为够不到 0.5 而被额外扣分。
* 只在 single_stance 给：双脚腾空（跳）时 reward=0，**不会诱发跳跃**。

当前实测（v1 iter2999）：单脚支撑期腾空 0.297 s < 0.4 —— 还没碰到上限，
所以 0.4 已经在"鼓励更长腾空"但力道不足；提到 0.5 会把上限抬高。

用法
-----------------------------------------------------------------------------
    python patch_airtime.py            # 打补丁（幂等）
    python patch_airtime.py --revert   # 还原
    python patch_airtime.py --check
"""
import shutil
import sys
from pathlib import Path

ENV_CFG = Path(
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
)
OLD = '"threshold": 0.4,'
NEW = '"threshold": 0.5,'

# 只替换 feet_air_time 段落里的那个 —— 全文替换有风险（其它项也可能有 0.4）
ANCHOR_NAME = "feet_air_time"


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".preairtime")
    if not b.exists():
        shutil.copy2(p, b)
        print("  备份 -> %s" % b)


def check():
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")
    i = c.find(ANCHOR_NAME + " = RewTerm(")
    seg = c[i:i + 500] if i >= 0 else ""
    print("threshold 0.5: %s" % ("是" if NEW in seg else "否"))
    print("threshold 0.4: %s" % ("是" if OLD in seg else "否"))


def revert():
    b = ENV_CFG.with_suffix(ENV_CFG.suffix + ".preairtime")
    if b.exists():
        shutil.copy2(b, ENV_CFG)
        print("已还原 %s" % ENV_CFG)
    else:
        print("!! 没有备份，跳过")
    check()


def apply():
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")
    i = c.find(ANCHOR_NAME + " = RewTerm(")
    if i < 0:
        print("!! 找不到 feet_air_time 段落")
        sys.exit(1)
    j = c.find("    )", i)
    if j < 0:
        print("!! 找不到段落结尾")
        sys.exit(1)
    seg = c[i:j]
    if NEW in seg:
        print("== 已是 0.5，跳过")
        check()
        return
    if OLD not in seg:
        print("!! 段落里没有 threshold: 0.4，手工确认：\n%s" % seg)
        sys.exit(1)
    backup(ENV_CFG)
    ENV_CFG.write_text(c[:i] + seg.replace(OLD, NEW, 1) + c[j:], encoding="utf-8")
    print("== feet_air_time threshold 0.4 -> 0.5")
    check()


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--revert" in a:
        revert()
    elif "--check" in a:
        check()
    else:
        apply()
