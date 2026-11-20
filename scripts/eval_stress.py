# -*- coding: utf-8 -*-
"""一次 Kit 启动，同时采集两类"补证据链"实验：

  组 RAMP (env 0): 速度指令连续 0.5→3.0→0.5 m/s 三角斜坡
      → 把"任意速度"从 11 档离散推论变成连续实测（P2#10）
      → 看滞后 / 超调 / 是否出现断点，而不是只看 11 个定速点

  组 PUSH (env 1..6): 定速 + 周期性横向速度冲量
      → 外力推鲁棒性（P2#9，面试最值钱："能不能用"）
      → 2 个速度 × 3 个冲量强度，统计恢复步数 / 是否摔倒

设计约束（都是踩过坑才这么写的）：
  * 复用 eval_gait.py 已验证骨架：AppLauncher / hydra / RslRlVecEnvWrapper /
    runner.load / get_inference_policy / policy_nn.reset(dones)，不自创加载路径
  * 每步覆写指令 + 关掉 resampling/standing（否则指令会被随机重采样污染）
  * push 用官方 API asset.write_root_velocity_to_sim，只改 root_vel_w 的 y 分量
    （侧向冲量最易失稳）；世界系 y 即机器人侧向（默认朝 +x）
  * 注入在 env.step 之前，让这一步物理从"被推后"的状态演化（同 pin_command 的时序坑）
  * 数据落 npz，分析在退出 Kit 之后做（不在 Kit 里跑复杂分析）
  * 记 dones 与 root_pos_z：摔倒 = done 或躯干高度骤降（正常 ~0.75 m）

用法：
  python eval_stress.py --task LeggedLab-Isaac-AMP-G1-v0 --headless \
      --checkpoint <model_2999.pt> --out /data/eval_out --steps 1200
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="连续斜坡 + 外力推鲁棒性采集")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--steps", type=int, default=1200)
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

# ---- 实验设计（写死，避免事后挑参数）--------------------------------------
# env 0: RAMP 连续指令
# env 1..6: PUSH，2 速度 × 3 强度
PUSH_VELS = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
PUSH_IMPULSE = [0.5, 1.0, 1.5, 0.5, 1.0, 1.5]   # 侧向 m/s
PUSH_PERIOD = 120        # 每 120 步(2.4s)推一次
RAMP_UP = 400            # 0.5→3.0 用 400 步(8s)
RAMP_DN = 400            # 3.0→0.5 用 400 步(8s)
V_LO, V_HI = 0.5, 3.0
FALL_Z = 0.40            # 躯干高度低于此视为跌倒（正常 ~0.75）


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    N = 1 + len(PUSH_VELS)
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg.scene.num_envs = N
    env_cfg.seed = 42

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}", flush=True)
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

    # 关掉指令重采样与 standing env（否则连续斜坡会被覆盖）
    try:
        term = base_env.command_manager.get_term("base_velocity")
        term.cfg.resampling_time_range = (1e9, 1e9)
        term.cfg.rel_standing_envs = 0.0
        print("[INFO] 已关闭 base_velocity 重采样与 standing env", flush=True)
    except Exception as exc:
        print(f"[WARN] 关重采样失败({exc})，将仅依赖每步覆写", flush=True)

    def ramp_cmd(step):
        if step < RAMP_UP:
            return V_LO + (V_HI - V_LO) * step / RAMP_UP
        elif step < RAMP_UP + RAMP_DN:
            return V_HI - (V_HI - V_LO) * (step - RAMP_UP) / RAMP_DN
        return V_LO

    def build_cmd(step):
        # (N,) 每 env 一个目标 vx
        c = np.zeros(N, np.float32)
        c[0] = ramp_cmd(step)
        for k, v in enumerate(PUSH_VELS):
            c[1 + k] = v
        return torch.tensor(c, device=dev, dtype=torch.float32)

    def pin_command(cmd_vec):
        try:
            cmd = base_env.command_manager.get_command("base_velocity")
            cmd[:, 0] = cmd_vec
            cmd[:, 1] = 0.0
            cmd[:, 2] = 0.0
        except Exception as exc:
            print(f"[WARN] pin 失败 {exc}", flush=True)

    records = {"lin_vel_x": [], "lin_vel_y": [], "cmd_x": [], "root_pos_z": [],
               "push_flag": [], "dones": []}
    push_ids = torch.arange(1, N, device=dev)  # env 1..N-1 参与 push
    impulse = torch.tensor(PUSH_IMPULSE, device=dev, dtype=torch.float32)

    obs = env.get_observations()
    pin_command(build_cmd(0))
    t0 = time.time()
    for step in range(args_cli.steps):
        cmd_vec = build_cmd(step)
        pin_command(cmd_vec)  # 必须在 step 之前（观测在 step 内部读 command）

        do_push = (step > 0) and (step % PUSH_PERIOD == 0)
        if do_push:
            try:
                vel_w = robot.data.root_vel_w[push_ids].clone()
                # 交替左右方向，避免只往一边推造成定向偏置
                sign = 1.0 if (step // PUSH_PERIOD) % 2 == 0 else -1.0
                vel_w[:, 1] += sign * impulse       # 世界系 y = 侧向
                robot.write_root_velocity_to_sim(vel_w, env_ids=push_ids)
            except Exception as exc:
                print(f"[WARN] push 注入失败 step={step}: {exc}", flush=True)

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            records["lin_vel_x"].append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
            records["lin_vel_y"].append(robot.data.root_lin_vel_b[:, 1].cpu().numpy())
            records["root_pos_z"].append(robot.data.root_pos_w[:, 2].cpu().numpy())
            records["dones"].append(dones.cpu().numpy())
        records["cmd_x"].append(cmd_vec.cpu().numpy())
        records["push_flag"].append(np.full(N, float(do_push), np.float32))

        if step % 200 == 0:
            print(f"  step {step}/{args_cli.steps} ({time.time()-t0:.0f}s) "
                  f"ramp_cmd={float(cmd_vec[0]):.2f}", flush=True)

    arr = {k: np.stack(v, axis=0) for k, v in records.items()}  # (T,N,...)
    tag = os.path.basename(resume_path).replace(".pt", "")
    npz = os.path.join(args_cli.out, f"stress_{tag}.npz")
    np.savez_compressed(npz, dt=dt, ramp_up=RAMP_UP, ramp_dn=RAMP_DN,
                        v_lo=V_LO, v_hi=V_HI, push_period=PUSH_PERIOD,
                        push_vels=np.array(PUSH_VELS), push_impulse=np.array(PUSH_IMPULSE),
                        fall_z=FALL_Z, **arr)
    print(f"[INFO] saved -> {npz}  shape={arr['lin_vel_x'].shape}", flush=True)
    print("=== STRESS COLLECT DONE ===", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
