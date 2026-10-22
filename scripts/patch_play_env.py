#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给服务器上的 G1AmpEnvCfg_PLAY 补齐「展示模式」应有的关闭项。

=============================================================================
为什么要有这个脚本（2026-10-22 诊断后补）
=============================================================================
对比 legged_lab 自带的 velocity 任务，G1FlatEnvCfg_PLAY 里有三行：

    self.observations.policy.enable_corruption = False   # 关观测噪声
    self.events.base_external_force_torque = None        # 关外力
    self.events.push_robot = None                        # 关推机器人

而 AMP 的 G1AmpEnvCfg_PLAY 只有 `self.events.reset_from_ref = None`，
**上面三条一条都没有**。后果：

  1. push_robot 每 5 秒把速度直接改成随机值
       velocity_range = {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}
     → 横向 ±0.5 m/s 的速度冲击，视频里表现为"突然晃一下"
     → 也是横向漂移的主要来源

  2. 观测噪声仍在（joint_vel ±1.5 rad/s、base_ang_vel ±0.2 rad/s）
     → 策略输入抖动，输出跟着抖

  3. heading_command=True 且 heading 范围 (-π, π)（基类）
     → 机器人一直在跟随机航向转弯；PLAY 里虽改成 (0,0)，
        但若初始朝向不为 0，仍会先转向再走直

本补丁把 PLAY 改成真正干净的展示模式：无扰动、无噪声、不转向、走直线。

=============================================================================
用法
=============================================================================
    python scripts/patch_play_env.py [g1_amp_env_cfg.py 路径]

安全性：
  * 幂等（已打过的检测 MARK 跳过）
  * 打补丁前备份为 <file>.preplay
  * 找不到插入点就报错退出，不改文件
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "# [PATCHED-PLAY-CLEAN]"
DEFAULT = (
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/"
    "config/g1/g1_amp_env_cfg.py"
)

BLOCK = """
        {mark}
        # 展示模式：关噪声、关外力、关推送、不转向（2026-10-22）
        # 与 velocity 任务的 G1FlatEnvCfg_PLAY 对齐
        self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
        # 不用 heading 控制：直接给 wz=0，避免开头先转向
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.resampling_time_range = (1e9, 1e9)
""".format(mark=MARK)


def patch(path: Path) -> int:
    if not path.exists():
        print(f"!! 找不到 {path}")
        return 1

    src = path.read_text(encoding="utf-8")
    if MARK in src:
        print(f"== 已打过补丁，跳过: {path}")
        return 0

    # 定位 G1AmpEnvCfg_PLAY 类体内的 reset_from_ref 行
    m = re.search(
        r"(class\s+G1AmpEnvCfg_PLAY\b.*?)(^[ \t]*self\.events\.reset_from_ref\s*=\s*None[ \t]*$)",
        src,
        flags=re.MULTILINE | re.DOTALL,
    )
    if m is None:
        print("!! 在 G1AmpEnvCfg_PLAY 里没找到 'self.events.reset_from_ref = None'，不改文件")
        return 2

    anchor_end = m.end(2)
    new_src = src[:anchor_end] + BLOCK + src[anchor_end:]

    bak = path.with_suffix(path.suffix + ".preplay")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"== 已备份 -> {bak}")

    path.write_text(new_src, encoding="utf-8")
    print(f"== 已打补丁 -> {path}")

    # 回读验证
    chk = path.read_text(encoding="utf-8")
    for key in [
        "enable_corruption = False",
        "push_robot = None",
        "base_external_force_torque = None",
        "heading_command = False",
    ]:
        ok = key in chk.split("class G1AmpEnvCfg_PLAY", 1)[-1]
        print(f"   [{'OK' if ok else 'MISS'}] {key}")
    return 0


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT)
    return patch(p)


if __name__ == "__main__":
    raise SystemExit(main())
