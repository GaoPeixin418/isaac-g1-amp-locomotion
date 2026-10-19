"""
08 - 量化步态评估
=============================================================================
为什么不能再手写一遍"看起来还行"：
   面试官是做运动控制的，"自然不自然"这种主观判断说服不了他。要的是：
   步频、步长、足端抬升、能耗 (CoT)、速度跟踪误差 —— 能和你 Poppy 的 PPO
   结果直接对比的数字。

设计要点：
   1) 一次 Kit 启动同时评估 6 个速度档 —— 每个 env 固定一个 lin_vel_x 指令。
      Kit 冷启动要 5 分钟、退出 4 分钟，分 6 次跑白白多烧 54 分钟。
   2) 速度指令每步覆写 —— CommandManager 会按 resampling_time_range 重新采样，
      不覆写的话会漂移回随机范围。
   3) 数据存 npz 落盘，指标计算在退出 Kit 之后做 —— 避免在 Kit 里跑复杂
      时序分析，出问题只能重启。

输出指标：
   stride_freq   步频 (Hz)         stance→swing 上升沿计数 / 时长
   stride_len    步长 (m)          平均前进速度 / 步频
   foot_clear    足端抬升 (m)      swing 期最高点 - stance 期最低点
   duty_factor   支撑占比          stance 时间 / 总时长，人类步行约 0.6
   cot           运输能耗 CoT      Σ|τ·ω|·dt / (m·g·d)，越低越省
   vel_err       速度跟踪误差      mean(|v_x - cmd_x|)
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="定量步态评估")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--velocities", type=str, default="0.5,1.0,1.5,2.0,2.5,3.0")
parser.add_argument("--out", type=str, default="/data/eval_out")
parser.add_argument("--no_push", action="store_true", default=True,
                    help="关闭 push_robot 扰动，得到干净的步态数据")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# 必须把 Hydra 不认识的参数从 sys.argv 里剔掉，否则 hydra 会报
# "unrecognized arguments"。这是 Isaac Lab 脚本的标准写法。
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- AppLauncher 之后才能 import isaaclab 相关模块 -------------------------
import gymnasium as gym  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import AMPRunner, OnPolicyRunner  # noqa: E402

import legged_lab.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    vel_list = [float(v) for v in args_cli.velocities.split(",")]
    n = len(vel_list)
    os.makedirs(args_cli.out, exist_ok=True)

    env_cfg.scene.num_envs = n
    env_cfg.seed = 42

    # 关掉定时推机器人：要的是干净的步态，不是鲁棒性测试
    if args_cli.no_push:
        try:
            env_cfg.events.push_robot.params["interval_range_s"] = (1e6, 1e6)
            print("[INFO] push_robot 已关闭")
        except Exception as exc:
            print(f"[WARN] 关 push_robot 失败({exc})，数据会带扰动噪声")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

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

    # ---- 找足部 body index ------------------------------------------------
    body_names = list(robot.data.body_names)
    foot_ids = [i for i, nm in enumerate(body_names) if "ankle_roll" in nm]
    print(f"[INFO] foot bodies: {[body_names[i] for i in foot_ids]}")

    has_contact = "contact_forces" in base_env.scene.sensors or (
        "contact_forces" in getattr(base_env.scene, "_sensors", {})
    )
    print(f"[INFO] contact sensor available: {has_contact}")

    # 机器人总质量（用于 CoT）
    try:
        mass = float(robot.root_physx_view.get_masses().sum())
    except Exception:
        mass = 35.0
    print(f"[INFO] robot mass = {mass:.2f} kg, dt = {dt}")

    # ---- 每步把速度指令钉死 -----------------------------------------------
    vel_tensor = torch.tensor(vel_list, device=base_env.device, dtype=torch.float32)

    def pin_command():
        try:
            cmd = base_env.command_manager.get_command("base_velocity")
            cmd[:, 0] = vel_tensor
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0
        except Exception:
            pass

    obs = env.get_observations()
    pin_command()

    records = {"lin_vel_x": [], "lin_vel_y": [], "foot_pos_b": [], "torque": [],
               "joint_vel": [], "cmd_x": []}
    if has_contact:
        records["foot_force_z"] = []

    t0 = time.time()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            records["lin_vel_x"].append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
            records["lin_vel_y"].append(robot.data.root_lin_vel_b[:, 1].cpu().numpy())
            records["foot_pos_b"].append(
                robot.data.body_pos_b[:, foot_ids, :].cpu().numpy().astype(np.float32)
            )
            records["torque"].append(robot.data.applied_torque.cpu().numpy().astype(np.float32))
            records["joint_vel"].append(robot.data.joint_vel.cpu().numpy().astype(np.float32))
            if has_contact:
                sensor = base_env.scene["contact_forces"]
                records["foot_force_z"].append(
                    sensor.data.net_forces_w[:, foot_ids, 2].cpu().numpy().astype(np.float32)
                )
            try:
                records["cmd_x"].append(
                    base_env.command_manager.get_command("base_velocity")[:, 0].cpu().numpy()
                )
            except Exception:
                records["cmd_x"].append(np.full(n, np.nan, dtype=np.float32))

        pin_command()
        if step % 300 == 0:
            print(f"  step {step}/{args_cli.steps}  ({time.time()-t0:.0f}s)")

    print(f"[INFO] rollout done in {time.time()-t0:.0f}s")

    arr = {k: np.stack(v, axis=0) for k, v in records.items()}  # (T, N, ...)
    tag = os.path.basename(resume_path).replace(".pt", "")
    npz = os.path.join(args_cli.out, f"gait_{tag}.npz")
    np.savez_compressed(npz, dt=dt, mass=mass, velocities=np.array(vel_list), **arr)
    print(f"[INFO] saved raw data -> {npz}")
    print("=== EVAL COLLECT DONE ===")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
