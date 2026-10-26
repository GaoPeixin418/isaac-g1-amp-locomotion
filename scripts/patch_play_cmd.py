#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给服务器上的 play.py 打补丁：固定速度指令、走直线。

=============================================================================
为什么要有这个脚本（2026-10-22 踩坑后补）
=============================================================================
之前所有展示视频（amp_model_4000 / amp_4000_single / ppo_baseline /
style_only_lerp0）都是 `07_mid_eval.sh -> play.py` 录的，
而 play.py 是 Isaac Lab 原版，**没有任何固定指令的逻辑**。

于是视频里实际发生的是：
  * `resampling_time_range = [10.0, 10.0]`  -> 每 10 秒随机重采样一次速度指令
  * 指令范围 lin_vel_x ∈ [-0.5, 3.0]        -> 一会儿慢走一会儿冲刺
  * `rel_heading_envs = 1.0`，heading ∈ [-π, π]
                                            -> 100% 的 env 都在跟随机航向，一直在转弯
  * `rel_standing_envs = 0.02`              -> 有概率突然站着不动

观感后果：机器人"前几步不稳、后面走得急、路线乱"。
**这是录制方式造成的，不是策略缺陷** —— 固定指令下的稳态振荡
只有 CV 0.07~0.25，中高速档都在人类量级内。

补丁做的事：
  1. 关掉指令重采样与 standing env
  2. 每步把 base_velocity 钉成 PIN_VEL 指定的值，lin_vel_y / 航向 置 0（走直线）

=============================================================================
用法
=============================================================================
    python scripts/patch_play_cmd.py [play.py 路径]

环境变量（运行时生效，不是本脚本的）：
    PIN_VEL="0.5,1.0,1.5,2.0,2.5,3.0"   # 每个 env 一个，或用单值广播

安全性：
  * 找不到插入点就报错退出，绝不破坏原文件
  * 已打过补丁会检测到并跳过
  * 打补丁前自动备份为 <play.py>.precmd
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "[PATCHED-PIN-CMD-v2]"
OLD_MARK = "[PATCHED-PIN-CMD]"
DEFAULT = "/data/legged_lab/scripts/rsl_rl/play.py"


def patch(path: Path) -> int:
    if not path.exists():
        print(f"!! 找不到 {path}")
        return 1

    src = path.read_text(encoding="utf-8")
    if MARK in src:
        print(f"== 已打过 v2 补丁，跳过: {path}")
        return 0

    # v1 -> v2 升级：先恢复原始文件再重打，避免块内套块
    bak = path.with_suffix(path.suffix + ".precmd")
    if OLD_MARK in src:
        if not bak.exists():
            print("!! 检测到 v1 补丁但没有 .precmd 备份，不敢动，退出")
            return 3
        src = bak.read_text(encoding="utf-8")
        print("== 检测到 v1 补丁，已从 .precmd 恢复，将重打 v2")

    # ---- 定位插入点 -------------------------------------------------------
    m = re.search(r"^(\s*)while\s+simulation_app\.is_running\(\)\s*:\s*$",
                  src, flags=re.MULTILINE)
    if m is None:
        # 退而求其次：宽松匹配 while simulation_app.is_running():
        m = re.search(r"^([ \t]*)while[ \t]+simulation_app\.is_running\(\)[ \t]*:",
                      src, flags=re.MULTILINE)
    if m is None:
        print("!! 没找到 'while simulation_app.is_running():'，不敢改，退出")
        return 2

    indent = m.group(1) or "    "
    body_indent = indent + "    "

    block_lines = [
        "",
        f"        # ==== {MARK} BEGIN ====",
        "        # 速度指令控制。PIN_MODE=pin(默认) 钉死; PIN_MODE=rand 随机重采样走直线。",
        "        # PIN_VEL 用逗号分隔，每 env 一个；只给一个值则广播。",
        "        # 去掉本补丁：cp play.py.precmd play.py",
        "        try:",
        "            if \"_PIN_VEL\" not in globals():",
        "                import torch as _th",
        "                import os as _os",
        "                _v = _th.tensor([float(x) for x in",
        "                                 _os.environ.get(\"PIN_VEL\", \"1.0\").split(\",\")],",
        "                                dtype=_th.float32)",
        "                globals()[\"_PIN_VEL\"] = _v",
        "                _term = env.unwrapped.command_manager.get_term(\"base_velocity\")",
        "                _term.cfg.rel_standing_envs = 0.0",
        "                if _os.environ.get(\"PIN_MODE\", \"pin\") == \"rand\":",
        "                    # 随机模式：保留每 10s 重采样，但只允许前进、不许转弯/后退",
        "                    _term.cfg.resampling_time_range = (10.0, 10.0)",
        "                    _term.cfg.ranges.lin_vel_x = (0.5, 3.0)",
        "                    _term.cfg.ranges.lin_vel_y = (0.0, 0.0)",
        "                    _term.cfg.ranges.ang_vel_z = (0.0, 0.0)",
        "                    try:",
        "                        _term.cfg.ranges.heading = (0.0, 0.0)",
        "                    except Exception:",
        "                        pass",
        "                    globals()[\"_PIN_RAND\"] = True",
        "                    print(\"[PIN] rand 模式: 每10s随机换速, 走直线, 范围0.5-3.0\")",
        "                else:",
        "                    _term.cfg.resampling_time_range = (1e9, 1e9)",
        "                    globals()[\"_PIN_RAND\"] = False",
        "                    print(\"[PIN] pin 模式: 已关闭重采样/站立; 固定指令 =\", _v.tolist())",
        "            if not globals().get(\"_PIN_RAND\", False):",
        "                _cmd = env.unwrapped.command_manager.get_command(\"base_velocity\")",
        "                _pv = globals()[\"_PIN_VEL\"].to(_cmd.device)",
        "                if _pv.numel() == 1:",
        "                    _cmd[:, 0] = _pv.item()",
        "                else:",
        "                    _cmd[:, 0] = _pv[: _cmd.shape[0]]",
        "                _cmd[:, 1] = 0.0",
        "                _cmd[:, 2] = 0.0",
        "        except Exception:",
        "            pass",
        f"        # ==== {MARK} END ====",
    ]
    body = "\n".join(block_lines)

    # 插到 while 行之后（跳过紧随其后的注释/文档串，直接进循环体即可）
    insert_at = m.end()
    new_src = src[:insert_at] + body + src[insert_at:]

    bak = path.with_suffix(path.suffix + ".precmd")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"== 已备份 -> {bak}")
    path.write_text(new_src, encoding="utf-8")
    print(f"== 已打补丁 -> {path}")
    print(f"== 插入点: while simulation_app.is_running(): 之后")
    return 0


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT)
    return patch(p)


if __name__ == "__main__":
    raise SystemExit(main())
