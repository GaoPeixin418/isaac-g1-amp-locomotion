#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""设置并验证 AMP 的 task_style_lerp（λ）。

=============================================================================
为什么要有这个脚本（2026-10-22 事故后补）
=============================================================================
事故经过：
  10-20 晚上为了录 PPO 基线，把 cfg 里的 task_style_lerp 改成 1.0，
  之后忘还原。10-22 的「对称性奖励消融」训练因此全部是
  **r = 1.0·task + 0.0·style = 纯 PPO**，风格奖励被完全关掉，
  而日志里的 Episode_Reward/style 照样在打印（它是独立记录的），
  所以从日志看不出来 —— 白烧了 4 小时 GPU，结论全部作废。

语义（rsl_rl/rsl_rl/modules/amp.py:203，已核）：
    r = task_style_lerp * r_task + (1 - task_style_lerp) * r_style
    λ=0.0 -> 纯模仿，没有任务奖励 -> 站着不动（10-20_18-02-56 已验证）
    λ=0.5 -> 标准 AMP
    λ=1.0 -> 纯任务 = 普通 PPO，风格奖励完全不参与

本脚本做的事：
  1. 把 λ 写进 cfg（幂等，先备份）
  2. **回读验证**：写进去的值必须和要的值一致，否则退出码非 0，
     让调用方的 `||` 拦住训练 —— 这是防止事故复发的关键一环

用法
=============================================================================
    python patch_lerp.py 0.5 [cfg路径]
    # 退出码 0 = 成功且已验证；非 0 = 没改/没写进去
"""
import re
import shutil
import sys
from pathlib import Path

DEFAULT = (
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/"
    "config/g1/agents/rsl_rl_ppo_cfg.py"
)
KEY = "task_style_lerp"


def main() -> int:
    if len(sys.argv) < 2:
        print("!! 用法: python patch_lerp.py <lambda> [cfg路径]")
        print("   必须显式给 λ：不给就默认是「我又忘了在训哪种奖励混合」")
        return 2
    want = sys.argv[1]
    try:
        float(want)
    except ValueError:
        print(f"!! λ 不是数字: {want}")
        return 2

    path = Path(sys.argv[2]) if len(sys.argv) > 2 else Path(DEFAULT)
    if not path.exists():
        print(f"!! 找不到 {path}")
        return 2

    src = path.read_text(encoding="utf-8")
    new_src, n = re.subn(rf"{KEY}=[0-9.]+", f"{KEY}={want}", src)
    if n == 0:
        print(f"!! cfg 里没有 {KEY}=，不敢造一个：{path}")
        return 2

    if new_src == src:
        print(f"== λ 已经是 {want}，无需改动: {path}")
    else:
        bak = path.with_suffix(path.suffix + ".prelerp")
        if not bak.exists():
            shutil.copy2(path, bak)
            print(f"== 已备份 -> {bak}")
        path.write_text(new_src, encoding="utf-8")
        print(f"== 已写入 {KEY}={want} -> {path}")

    # ---- 回读验证（关键）--------------------------------------------------
    chk = path.read_text(encoding="utf-8")
    m = re.findall(rf"{KEY}=([0-9.]+)", chk)
    got = m[0] if m else None
    if got != want:
        print(f"!! 验证失败：期望 {want}，实际 {got}")
        return 3
    print(f"== 验证通过：{KEY}={got}")
    if float(want) >= 1.0:
        print("== [警告] λ>=1.0 => 风格奖励被完全关掉，这是纯 PPO，不是 AMP")
    if float(want) <= 0.0:
        print("== [警告] λ<=0 => 任务奖励被完全关掉，机器人会站着不动")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
