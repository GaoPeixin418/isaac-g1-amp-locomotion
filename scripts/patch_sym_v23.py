#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
v2.3 补丁（2026-11-11）：给 leg_work_asymmetry 加"早期 warmup"

背景（本轮取证结论，见 docs/v2.2结案与v2.3方案.md）：
  v2.2 从头训只跑到 iter 84/2400（5 分 15 秒，00:03 断电），**从未产生任何评估点**
  => v2.2 不是失败，是没跑。之前"排除法定罪权重 -2.0"的证据链也因此不完整：
  v2 在 iter 800 被判死刑，而 v1 的相变恰好发生在 400→800 之间
  （v1: 400 时 v_act 0.06 -> 800 时 0.97）。所以真正的问题是
  「约束在 motor primitive 形成期就把探索压死了」，不是"权重数值不对"。

本补丁做的事（只改一处）：
  在 leg_work_asymmetry 的返回值上乘一个 0→1 的线性斜坡，
  前 LEG_ASYM_WARMUP_STEPS 个控制步里该项权重从 0 爬到满额。
  默认 12000 步 = 500 iter（num_steps_per_env=24）。

为什么用"自己数调用次数"而不是 env.common_step_counter：
  reward 函数每个控制步被调用一次，自计数零依赖 Isaac Lab 内部属性，
  与同文件 `_leg_work_asym_ema` 的 setattr 写法风格一致，不会因版本改名而崩。

不改动的东西（刻意保留，保持与 v2.2 严格可比）：
  - weight 仍是 -1.0、time_constant 仍 1.5、|tau*omega| 不变
  - base_height_bias / knee_bias 保留：实测 0.762 m（死区 0.715-0.815）与
    16.7°（死区 5.3-24.7°）都在死区内，梯度为零 = no-op，不影响归因

用法:
    python patch_sym_v23.py --check     # 只看状态
    python patch_sym_v23.py             # 应用（自动 .bak）
    python patch_sym_v23.py --revert    # 回滚
"""
import shutil
import sys
from pathlib import Path

REWARDS = Path(
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"
)

MARK = "[PATCHED-SYM-V23-WARMUP]"

HEADER = '''
# [PATCHED-SYM-V23-WARMUP] 前 N 个控制步把本项权重从 0 线性爬到满额。
# N=0 关闭 warmup（退化为 v2.2 行为）。12000 步 = 500 iter @ 24 steps/iter。
import os as _os_v23
_V23_WARMUP_STEPS = int(_os_v23.environ.get("LEG_ASYM_WARMUP_STEPS", "12000"))


def _v23_warm_scale(env) -> float:
    n = _V23_WARMUP_STEPS
    if n <= 0:
        return 1.0
    c = int(getattr(env, "_v23_reward_calls", 0)) + 1
    setattr(env, "_v23_reward_calls", c)
    return min(1.0, c / float(n))
'''

ANCHOR = "def leg_work_asymmetry("
OLD_RET = "    rel = (w_l - w_r).abs() / (w_l + w_r + 1.0)\n    return rel.mean(dim=-1)"
NEW_RET = (
    "    rel = (w_l - w_r).abs() / (w_l + w_r + 1.0)\n"
    "    return rel.mean(dim=-1) * _v23_warm_scale(env)"
)

BAK = REWARDS.with_suffix(".py.v22.bak")


def check():
    c = REWARDS.read_text(encoding="utf-8", errors="ignore")
    has_mark = MARK in c
    has_call = "_v23_warm_scale(env)" in c
    print("mark 存在        : %s" % has_mark)
    print("return 已乘 scale: %s" % has_call)
    try:
        import py_compile

        py_compile.compile(str(REWARDS), doraise=True)
        print("语法编译        : 通过")
        ok_c = True
    except Exception as e:  # noqa: BLE001
        print("语法编译        : 失败 %s" % e)
        ok_c = False
    return has_mark and has_call and ok_c


def apply():
    c = REWARDS.read_text(encoding="utf-8", errors="ignore")

    if MARK in c:
        print("== 已打过 v2.3 补丁，跳过")
        print("结果: %s" % ("通过" if check() else "!! 未通过"))
        return 0

    if ANCHOR not in c:
        print("!! 找不到 %s，源码结构变了，手工确认" % ANCHOR)
        return 1
    if OLD_RET not in c:
        print("!! 找不到 return 语句（rel/return rel.mean），手工确认")
        return 1

    shutil.copy2(REWARDS, BAK)
    print("== 备份 -> %s" % BAK.name)

    c = c.replace(ANCHOR, HEADER.strip() + "\n\n\n" + ANCHOR, 1)
    c = c.replace(OLD_RET, NEW_RET, 1)
    REWARDS.write_text(c, encoding="utf-8")
    print("== 已注入 warmup 斜坡")

    import py_compile

    py_compile.compile(str(REWARDS), doraise=True)
    print("编译通过")
    print("结果: %s" % ("通过" if check() else "!! 未通过"))
    return 0 if check() else 1


def revert():
    if not BAK.exists():
        print("!! 没有备份 %s，无法回滚" % BAK.name)
        return 1
    shutil.copy2(BAK, REWARDS)
    print("== 已从 %s 回滚" % BAK.name)
    print("结果: %s" % ("通过" if not check() else "!! 仍有残留"))
    return 0


if __name__ == "__main__":
    a = sys.argv[1:]
    if "--check" in a:
        sys.exit(0 if check() else 1)
    if "--revert" in a:
        sys.exit(revert())
    sys.exit(apply())
