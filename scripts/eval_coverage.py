# -*- coding: utf-8 -*-
"""指令可行域扫描 (command coverage probe) —— 一次 Kit 会话测策略的指令能力边界。

动机（2026 前沿对标）：HuMBLE / Driving the Style 等 2026 工作指出 AMP 类方法
"fail to generalize to commands outside the data distribution"，并用
command coverage / held-out velocity profiles 量化。本项目名义训练分布
  vx(-0.5,3.0) / vy(-0.5,0.5) / wz(-1,1), heading_command=True
覆盖了侧移/后退/转向，但实测"侧移/后退/原地转向输出 ≈ 0"。
本脚本把这个 gap 从定性升级为量化：每个 env 钉死一个指令矢量，测稳态跟踪。

设计约束（全部是本项目踩过坑才这么写）：
  * 复用 eval_stress.py 已验证骨架（AppLauncher / RslRlVecEnvWrapper /
    runner.load / get_inference_policy / policy_nn.reset），不自创加载路径
  * 每步在 env.step 之前覆写 cmd[:,0:3]，并关掉 resampling / standing /
    heading（heading P 控制器会覆盖 cmd[:,2]，直接钉 wz 更干净）
  * **记录 cmd 回读值**：验证 pin 真的生效，没被 command manager 覆盖
  * 丢启动暂态（前 100 步）由分析脚本处理，这里只忠实记录全部时序
  * 跌倒判据只用 root_pos_z（不用 dones，dones 混 episode timeout）
  * 逐 env 独立记录，绝不跨 env 求平均（合成会掩盖失效点）
  * 单 seed、单次 rollout、纯仿真 —— 结论强度：诊断性，非统计

用法：
  python eval_coverage.py --task LeggedLab-Isaac-AMP-G1-v0 --headless \
      --checkpoint <model_2999.pt> --out /data/eval_out --steps 300
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="指令可行域扫描 command coverage probe")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--out", type=str, default="/data/eval_out")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import AMPRunner, OnPolicyRunner  # noqa: E402
import legged_lab.tasks  # noqa: F401,E402

# ---- 指令网格 (label, vx, vy, wz) ------------------------------------------
# 分布内 = 训练 ranges 内；分布外 = 超出 ranges。写死，避免事后挑参数。
POINTS = []
# A 前进基线（对照：已知静态档 ρ=0.998，应全部跟得上）
for v in (0.5, 1.0, 1.5, 2.0, 2.5, 3.0):
    POINTS.append(("fwd", v, 0.0, 0.0))
# B 侧移 @vx=1.0（±0.25/±0.5 在训练 vy 范围内，±1.0 是分布外）
for vy in (-0.25, 0.25, -0.5, 0.5, -1.0, 1.0):
    POINTS.append(("lateral", 1.0, vy, 0.0))
# C 后退 / 负速度（-0.25/-0.5 在训练 vx 范围内，-1.0 分布外）
for vx in (-0.25, -0.5, -1.0):
    POINTS.append(("back", vx, 0.0, 0.0))
# D 原地转向 @vx=0（±0.5/±1.0 在训练 wz 范围内）
for wz in (-0.5, 0.5, -1.0, 1.0):
    POINTS.append(("yaw_stand", 0.0, 0.0, wz))
# E 前进中转向 @vx=1.0
for wz in (-0.5, 0.5, -1.0, 1.0):
    POINTS.append(("yaw_fwd", 1.0, 0.0, wz))
# F 组合指令（对角）
POINTS += [
    ("mix", 1.0, 0.3, 0.3),
    ("mix", 1.0, -0.3, -0.3),
    ("mix", 2.0, 0.5, 0.0),
    ("mix", 0.5, 0.0, 0.5),
]
# G 站立（零点：应 v≈0，测是否有漂移/自发运动）
POINTS.append(("stand", 0.0, 0.0, 0.0))

N = len(POINTS)


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg.scene.num_envs = N
    env_cfg.seed = 42
    print(f"[COV] START N={N} steps={args_cli.steps} t={time.strftime('%H:%M:%S')}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[COV] Loading checkpoint: {resume_path}", flush=True)
    cls = agent_cfg.class_name
    if cls == "AMPRunner":
        runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif cls == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"unsupported runner: {cls}")
    runner.load(resume_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    dt = float(base_env.step_dt)
    dev = base_env.device

    # 关掉所有会改写 cmd 的机制，让钉死的三分量指令真正生效
    try:
        term = base_env.command_manager.get_term("base_velocity")
        term.cfg.resampling_time_range = (1e9, 1e9)
        term.cfg.rel_standing_envs = 0.0
        term.cfg.heading_command = False          # 否则 P 控制器覆盖 cmd[:,2]
        try:
            term.cfg.rel_heading_envs = 0.0
        except Exception:
            pass
        try:
            term.is_heading_env[:] = False
        except Exception:
            pass
        # 放开限幅，避免 cmd 被 clip 回训练 ranges
        term.cfg.ranges.lin_vel_x = (-2.0, 4.0)
        term.cfg.ranges.lin_vel_y = (-2.0, 2.0)
        term.cfg.ranges.ang_vel_z = (-2.0, 2.0)
        print("[COV] 已关 resampling/standing/heading 并放开 ranges", flush=True)
    except Exception as exc:
        print(f"[COV][WARN] 关指令机制失败({exc})，将仅依赖每步覆写", flush=True)

    # 每个 env 一个固定指令点 —— 用 tensor 广播，形状 (N,3)
    cmd_grid = torch.tensor(
        [[p[1], p[2], p[3]] for p in POINTS], device=dev, dtype=torch.float32
    )

    def pin_command():
        try:
            cmd = base_env.command_manager.get_command("base_velocity")
            cmd[:, 0] = cmd_grid[:, 0]
            cmd[:, 1] = cmd_grid[:, 1]
            cmd[:, 2] = cmd_grid[:, 2]
            return cmd.detach().clone()
        except Exception as exc:
            print(f"[COV][WARN] pin 失败 {exc}", flush=True)
            return cmd_grid

    records = {
        "lin_vel_x": [], "lin_vel_y": [], "ang_vel_z": [],
        "root_pos_z": [], "cmd_rb": [], "dones": [],
    }
    obs = env.get_observations()
    pin_command()
    print(f"[COV] KIT_READY loop start t={time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    for step in range(args_cli.steps):
        cmd_rb = pin_command()  # 必须在 step 前（观测在 step 内部读 command）
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            records["lin_vel_x"].append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
            records["lin_vel_y"].append(robot.data.root_lin_vel_b[:, 1].cpu().numpy())
            records["ang_vel_z"].append(robot.data.root_ang_vel_b[:, 2].cpu().numpy())
            records["root_pos_z"].append(robot.data.root_pos_w[:, 2].cpu().numpy())
            records["dones"].append(dones.cpu().numpy())
        records["cmd_rb"].append(cmd_rb[:, 0:3].cpu().numpy())
        if step % 50 == 0:
            print(f"[COV] step {step}/{args_cli.steps} ({time.time()-t0:.0f}s)", flush=True)

    arr = {k: np.stack(v, axis=0) for k, v in records.items()}  # (T,N,...)
    tag = os.path.basename(resume_path).replace(".pt", "")
    labels = np.array([p[0] for p in POINTS], dtype=object)
    grid = np.array([[p[1], p[2], p[3]] for p in POINTS], dtype=np.float32)
    npz = os.path.join(args_cli.out, f"coverage_{tag}.npz")
    np.savez_compressed(npz, dt=dt, labels=labels, grid=grid, steps=args_cli.steps, **arr)
    print(f"[COV] saved -> {npz}  shape={arr['lin_vel_x'].shape}", flush=True)
    print("=== COVERAGE COLLECT DONE ===", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
