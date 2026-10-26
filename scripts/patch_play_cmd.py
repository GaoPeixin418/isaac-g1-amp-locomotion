#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给服务器上的 play.py 打补丁：控制速度指令（钉死 / 随机 / 斜坡 / 全向脚本）。

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

=============================================================================
v3 新增：指令斜坡限幅（2026-10-26）——治"启动前栽 + 换速顿挫"
=============================================================================
现象：所有视频开头机器人都猛地向前栽一下；随机指令版每次换速都一顿。

机制（同一个原因，两处表现）：
  训练时指令是**阶跃**的（每 10 s 重采样一次，瞬间从 0.8 跳到 2.6）。
  跟踪奖励 exp(-|v-v_cmd|^2 / 0.25) 在阶跃瞬间几乎为 0，
  策略被"尽快消除误差"压着学 —— 学到的最优解就是**全力前倾加速**。
  t=0 时 v=0 而 cmd=1.2，是同一个阶跃的最大化版本 → 发射式前栽。

修法：**给指令加变化率上限（slew rate limit）**，让它变成斜坡而不是阶跃。
  这是零重训的部署端改动，而且真机部署本来就要做指令平滑
  （Unitree 官方控制器同样对速度指令做 ramp）。

=============================================================================
用法
=============================================================================
    python scripts/patch_play_cmd.py [play.py 路径]

运行时环境变量：
    PIN_MODE=pin|rand|ramp|omni
      pin   钉死在 PIN_VEL（不做斜坡，保留旧行为）
      rand  每 10 s 随机重采样，只前进、走直线
      ramp  钉死在 PIN_VEL，但从 0 用 RAMP_RATE 斜坡升上去  ← 治启动前栽
      omni  按 OMNI_SCRIPT 的时间脚本走全向，段间自动斜坡

    PIN_VEL="1.2"                    逗号分隔，每 env 一个，单值则广播
    RAMP_RATE="1.0"                  线速度变化率上限 m/s^2（默认 1.0）
    RAMP_RATE_W="1.0"                转向变化率上限 rad/s^2（默认 1.0）
    OMNI_SCRIPT="秒, vx,vy,steer;..."  累计时长脚本，例：
        "2,0,0,0; 8,1.0,0,0; 14,1.0,0,0.6; 18,1.0,0,0; 22,0,0.4,0; 26,-0.4,0,0; 30,0,0,-0.6"
      steer 单位 rad，会换算成 heading 误差（训练用的就是 heading 模式）

安全性：
  * 找不到插入点就报错退出，绝不破坏原文件
  * 已打过 v3 补丁会跳过；检测到 v1/v2 会从 .precmd 恢复后重打（幂等升级）
  * 打补丁前自动备份为 <play.py>.precmd
"""
import re
import shutil
import sys
from pathlib import Path

MARK = "[PATCHED-PIN-CMD-v3]"
OLD_MARKS = ("[PATCHED-PIN-CMD-v2]", "[PATCHED-PIN-CMD]")
DEFAULT = "/data/legged_lab/scripts/rsl_rl/play.py"

BLOCK = '''
        # ==== {MARK} BEGIN ====
        # 速度指令控制 + 线速度变化率限幅。详见 scripts/patch_play_cmd.py。
        try:
            if "_CMD_CTL" not in globals():
                import os as _os
                import torch as _th
                _mode = _os.environ.get("PIN_MODE", "pin")
                _rate = float(_os.environ.get("RAMP_RATE", "1.0"))
                _term = env.unwrapped.command_manager.get_term("base_velocity")
                _term.cfg.rel_standing_envs = 0.0
                _term.cfg.resampling_time_range = (1e9, 1e9)
                try:
                    _dt = float(env.unwrapped.step_dt)
                except Exception:
                    _dt = 0.02
                _dev = env.unwrapped.device

                # ---- 脚本：[(t_end, vx, vy, d_heading), ...]，t_end 为累计秒数 ----
                # d_heading 是"进入该段时，在朝向基础上再转多少弧度"。
                # 转向交给 command manager 自带的 P 控制器闭环完成 ——
                # 训练时就是 heading 模式，这样指令分布与训练完全一致。
                _script = []
                if _mode == "omni":
                    for _p in _os.environ.get("OMNI_SCRIPT", "10,1.0,0,0").split(";"):
                        _p = _p.strip()
                        if not _p:
                            continue
                        _v = [float(_x) for _x in _p.split(",")]
                        _script.append((_v[0], _v[1], _v[2], _v[3] if len(_v) > 3 else 0.0))
                else:
                    _pv = [float(_x) for _x in _os.environ.get("PIN_VEL", "1.0").split(",")]
                    _script = [(1e9, _pv[0], 0.0, 0.0)]

                if _mode == "rand":
                    _term.cfg.resampling_time_range = (10.0, 10.0)
                    _term.cfg.ranges.lin_vel_x = (0.5, 3.0)
                    _term.cfg.ranges.lin_vel_y = (0.0, 0.0)
                    _term.cfg.ranges.ang_vel_z = (0.0, 0.0)
                    try:
                        _term.cfg.ranges.heading = (0.0, 0.0)
                    except Exception:
                        pass
                    print("[CMD] rand 模式: 每10s随机换速, 走直线, 0.5-3.0")
                else:
                    print("[CMD] %s 模式: script=%s ramp=%.2f m/s^2" % (_mode, _script, _rate))

                # 切到 heading 模式（训练就是这样），否则 is_heading_env 全 False
                # 时 heading_target 写了也没人用。
                _term.cfg.heading_command = True
                # ⚠️ 必须把 ang_vel_z 范围放回来：patch_play_env 为了"走直线"
                #    把它锁成 (0.0, 0.0)，而 heading 模式下 cmd[:,2] 会被
                #    clip 到这个区间 —— 不还原的话转向指令恒为 0，根本转不动。
                _term.cfg.ranges.ang_vel_z = (-1.0, 1.0)
                try:
                    _term.cfg.rel_heading_envs = 1.0
                except Exception:
                    pass
                try:
                    _term.is_heading_env[:] = True
                except Exception:
                    pass

                globals()["_CMD_CTL"] = dict(
                    mode=_mode, dt=_dt, dev=_dev, script=_script, rate=_rate,
                    term=_term, t=0.0, seg=-1,
                    cur=_th.zeros(2, dtype=_th.float32, device=_dev),
                )

            _c = globals()["_CMD_CTL"]
            _c["t"] += _c["dt"]

            # ---- 定位当前段 ----
            _si = len(_c["script"]) - 1
            for _i, _s in enumerate(_c["script"]):
                if _c["t"] <= _s[0]:
                    _si = _i
                    break
            # 进入新段：设一次朝向目标（之后的闭环交给 command manager）
            if _si != _c["seg"]:
                _c["seg"] = _si
                try:
                    _yaw = _c["term"].robot.data.heading_w
                    _c["term"].heading_target[:] = _yaw + _c["script"][_si][3]
                    print("[CMD] t=%.1fs 进入第%d段 %s" % (_c["t"], _si, _c["script"][_si]))
                except Exception as _he:
                    print("[CMD] heading 设置失败: %r" % (_he,))

            _tgt = _th.tensor(_c["script"][_si][1:3],
                              dtype=_th.float32, device=_c["dev"])

            if _c["mode"] == "rand":
                _c["cur"] = _tgt  # rand 模式不斜坡，由 command manager 自己重采样
            else:
                # 变化率限幅：把阶跃变成斜坡（治启动前栽 + 换速顿挫）
                _d = _tgt - _c["cur"]
                _mx = _c["rate"] * _c["dt"]
                _c["cur"] = _c["cur"] + _th.clamp(_d, -_mx, _mx)

            _cmd = env.unwrapped.command_manager.get_command("base_velocity")
            _cmd[:, 0] = _c["cur"][0].item()
            _cmd[:, 1] = _c["cur"][1].item()
        except Exception as _e:
            print("[CMD] 指令控制异常: %r" % (_e,))
        # ==== {MARK} END ====
'''


def patch(path: Path) -> int:
    if not path.exists():
        print(f"!! 找不到 {path}")
        return 1

    src = path.read_text(encoding="utf-8")
    if MARK in src:
        print(f"== 已打过 v3 补丁，跳过: {path}")
        return 0

    bak = path.with_suffix(path.suffix + ".precmd")
    for om in OLD_MARKS:
        if om in src:
            if not bak.exists():
                print("!! 检测到旧补丁但没有 .precmd 备份，不敢动，退出")
                return 3
            src = bak.read_text(encoding="utf-8")
            print(f"== 检测到 {om}，已从 .precmd 恢复，将重打 v3")
            break

    m = re.search(r"^(\s*)while\s+simulation_app\.is_running\(\)\s*:\s*$",
                  src, flags=re.MULTILINE)
    if m is None:
        m = re.search(r"^([ \t]*)while[ \t]+simulation_app\.is_running\(\)[ \t]*:",
                      src, flags=re.MULTILINE)
    if m is None:
        print("!! 没找到 'while simulation_app.is_running():'，不敢改，退出")
        return 2

    body = BLOCK.replace("{MARK}", MARK)
    insert_at = m.end()
    new_src = src[:insert_at] + body + src[insert_at:]

    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"== 已备份 -> {bak}")
    path.write_text(new_src, encoding="utf-8")
    print(f"== 已打补丁 -> {path}")
    return 0


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv[1:]) else Path(DEFAULT)
    return patch(p)


if __name__ == "__main__":
    raise SystemExit(main())
