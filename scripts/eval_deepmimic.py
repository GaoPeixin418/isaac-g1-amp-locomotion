"""
DeepMimic 量化评估（对照 AMP 用）
=============================================================================
为什么不能复用 eval_gait.py：
   那个脚本每一步都要把 `base_velocity` 命令钉死在某个值上，靠指令分组。
   DeepMimic 环境**压根没有 command manager**（源码核验过：观测组里只有
   root/joint 状态和 6 项 ref_* 参考误差），所以那条路走不通。

 substitut 方案：
   DeepMimic 的速度来源是参考动作本身。所以分组依据改成
   **参考动作的瞬时前进速度** ref_vx，而不是人工指令 cmd_x。

   这样两边的对比仍然是同构的：
       AMP        —— 给定 cmd 速度，看 duty factor 怎么变
       DeepMimic  —— 跟着参考走到某个 ref 速度，看 duty factor 怎么变
   都是问同一个问题：步态形状是否随速度连续变化。

对外接口刻意与 eval_gait.py 保持一致（dt / mass / lin_vel_x / foot_z /
torque / joint_vel / cmd_x），这样 compare_gait.py 能原样读它。
cmd_x 这一列存的是参考速度，语义上是"此刻应该有多快"。

额外多存三列，只有 DeepMimic 才有：
   joint_err     关节角跟踪误差 (rad)    —— DeepMimic 的主场指标
   root_pos_err  根位置跟踪误差 (m)      —— 复现的几何保真度
   ref_vel_x     参考动作的瞬时前进速度
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="DeepMimic 定量评估")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac--Deepmimic-G1-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=6)
parser.add_argument("--steps", type=int, default=1500)
parser.add_argument("--out", type=str, default="/data/eval_out")
parser.add_argument("--tag", type=str, default="deepmimic")
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- AppLauncher 之后才能 import isaaclab 相关模块 -------------------------
import gymnasium as gym  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

import legged_lab.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    os.makedirs(args_cli.out, exist_ok=True)

    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = 42

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[INFO] Loading checkpoint: {resume_path}")

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path, map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    dt = float(base_env.step_dt)

    # ---- 参考帧从动画管理器取，而不是从观测向量里切 -----------------------
    # eval_gait.py 那套 command 接口这里没有；得自己抓 animation term。
    try:
        anim = base_env.animation_manager.get_term("animation")
        print("[INFO] 拿到 animation term")
    except Exception as exc:
        print(f"[ERROR] 取 animation term 失败: {exc}")
        raise SystemExit(1)

    body_names = list(robot.data.body_names)
    foot_ids = [i for i, nm in enumerate(body_names) if "ankle_roll" in nm]
    print(f"[INFO] foot bodies: {[body_names[i] for i in foot_ids]}")

    try:
        mass = float(robot.root_physx_view.get_masses()[0].sum())
    except Exception as exc:
        print(f"[WARN] 取质量失败({exc})，回退到 35kg")
        mass = 35.0
    print(f"[INFO] robot mass = {mass:.2f} kg, dt = {dt}")

    obs = env.get_observations()

    records = {
        "lin_vel_x": [], "lin_vel_y": [], "foot_z": [],
        "torque": [], "joint_vel": [],
        "cmd_x": [],        # 这里存的是参考速度（见下：用速率而非 x 分量）
        "joint_err": [], "root_pos_err": [], "ref_vel_x": [], "ref_vel_y": [],
        "ref_speed": [],
    }

    t0 = time.time()
    for step in range(args_cli.steps):
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            # 参考帧：buffer 的 index 0 是当前帧（num_steps=4）
            ref_dof = anim.get_dof_pos()[:, 0, :]              # (N, dof)
            ref_root_pos = anim.get_root_pos_w()[:, 0, :]      # (N, 3)
            ref_root_vel = anim.get_root_vel_w()[:, 0, :]      # (N, 3)

            records["lin_vel_x"].append(robot.data.root_lin_vel_w[:, 0].cpu().numpy())
            records["lin_vel_y"].append(robot.data.root_lin_vel_w[:, 1].cpu().numpy())
            records["foot_z"].append(
                robot.data.body_pos_w[:, foot_ids, 2].cpu().numpy().astype(np.float32)
            )
            records["torque"].append(robot.data.applied_torque.cpu().numpy().astype(np.float32))
            records["joint_vel"].append(robot.data.joint_vel.cpu().numpy().astype(np.float32))

            # 参考数据中 root_pos 是**局部坐标**（仿真时才加 env_origins），
            # robot.data.root_pos_w 含 env_origins，必须减掉才能比。
            local_root_pos = robot.data.root_pos_w - base_env.scene.env_origins
            records["root_pos_err"].append(
                (local_root_pos[:, :2] - ref_root_pos[:, :2]).norm(dim=-1).cpu().numpy()
            )
            records["joint_err"].append(
                (robot.data.joint_pos - ref_dof).abs().mean(dim=-1).cpu().numpy()
            )
            # 速度标签为什么用"速率"而不是 x 分量：
            #   30 段 clip 里混了转弯/侧步/倒退，根速度朝向各不相同，
            #   实测 x 分量均值是 **-0.249**（负数）。按 x 分量的话，
            #   "倒退" 和 "前进" 会互相抵消，"侧步" 会记成 0 速。
            #   而我们关心的是"此刻应该有多快"，与方向无关，所以用水平速率。
            vxy = ref_root_vel[:, :2]
            ref_speed = vxy.norm(dim=-1)
            records["ref_vel_x"].append(ref_root_vel[:, 0].cpu().numpy())
            records["ref_vel_y"].append(ref_root_vel[:, 1].cpu().numpy())
            records["ref_speed"].append(ref_speed.cpu().numpy())
            records["cmd_x"].append(ref_speed.cpu().numpy())

        if step % 300 == 0:
            print(f"  step {step}/{args_cli.steps}  ({time.time()-t0:.0f}s)")

    print(f"[INFO] rollout done in {time.time()-t0:.0f}s")

    arr = {k: np.stack(v, axis=0) for k, v in records.items()}
    mean_ref = float(np.nanmean(arr["ref_speed"]))
    npz = os.path.join(args_cli.out, f"gait_{args_cli.tag}.npz")
    np.savez_compressed(
        npz, dt=dt, mass=mass,
        velocities=np.full(args_cli.num_envs, mean_ref, dtype=np.float32),
        **arr,
    )
    print(f"[INFO] 平均参考速率 = {mean_ref:.3f} m/s（水平速率，非 x 分量）")
    print(f"[INFO] saved raw data -> {npz}")
    print("=== EVAL DEEPMIMIC DONE ===")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
