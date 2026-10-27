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

    # 关掉定时推机器人：要的是干净的步态，不是鲁棒性测试。
    # 注意不是改 interval_range_s —— 那个 param 不被这种 term 接受。
    # 正确做法是把推力幅值设为 0，等价于不推。
    if args_cli.no_push:
        try:
            env_cfg.events.push_robot.params["velocity_range"] = {
                "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)
            }
            print("[INFO] push_robot 推力已置零")
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

    # Isaac Lab 2.3 的 ArticulationData 没有 body_pos_b（只有世界坐标）。
    # 但足端"高度"就是世界 z，不受偏航角影响，所以直接用世界坐标的 z 分量。
    if not hasattr(robot.data, "body_pos_w"):
        print("[DEBUG] body attrs:", [a for a in dir(robot.data) if "body" in a or "_pos_" in a])
        raise SystemExit(1)

    has_contact = "contact_forces" in base_env.scene.sensors or (
        "contact_forces" in getattr(base_env.scene, "_sensors", {})
    )
    # 关键坑：ContactSensor.data.net_forces_w 只有**被配置的 body**（本例 2 个），
    # 不是 robot 的全部 30 个 link。必须用 sensor 自己的 body index 去取，
    # 用 robot 的全局 body index 会静默采到全 0（实测踩过）。
    sensor_ids = None
    if has_contact:
        csensor = base_env.scene["contact_forces"]
        found_ids, found_names = csensor.find_bodies(".*_ankle_roll_link")
        sensor_ids = torch.tensor(found_ids, device=base_env.device)
        print(f"[INFO] contact sensor body ids={found_ids} names={found_names}")

    # 机器人总质量（用于 CoT）。
    # 注意 get_masses() 返回 (N, num_links)，必须只取一个 env 再求和，
    # 否则会把 6 个环境的质量一起加起来（曾算出 206kg 的荒谬值）。
    try:
        mass = float(robot.root_physx_view.get_masses()[0].sum())
    except Exception as exc:
        print(f"[WARN] 取质量失败({exc})，回退到 35kg")
        mass = 35.0
    print(f"[INFO] robot mass = {mass:.2f} kg, dt = {dt}, num_links = {len(body_names)}")

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

    # ---- 关掉指令重采样与 standing env -------------------------------------
    # 踩过的坑（2026-10-21 23:25 发现）：
    # 训练配置 commands.base_velocity 里
    #   resampling_time_range = [10.0, 10.0]
    #   rel_standing_envs     = 0.02
    # 意味着每 10 秒会把指令重采样一次，且有 2% 的 env 会被强制设为零指令（站立）。
    # 6 个 env × 2% ≈ 1 个 env 会被选中 —— 实测 2.5 m/s 那一路被选中，
    # 前 10 秒 cmd 恒为 0，导致 vx≈0。
    # 当时误判为"2.5 m/s 存在启动暂态（策略缺陷）"，实为评估脚本污染。
    # 机器人静止是正确行为，被污染的其实是测量本身。
    #
    # 彻底修法：直接在 command term 上把这两个机制关掉，而不是靠 pin 抢顺序。
    def disable_command_sampling():
        try:
            term = base_env.command_manager.get_term("base_velocity")
            term.cfg.resampling_time_range = (1e9, 1e9)
            term.cfg.rel_standing_envs = 0.0
            print("[INFO] 已关闭 base_velocity 重采样与 standing env")
        except Exception as exc:
            print(f"[WARN] 关闭指令重采样失败({exc})，将仅依赖每步 pin")

    disable_command_sampling()

    obs = env.get_observations()
    pin_command()

    # joint_pos 必须记：2026-10-27 用户反馈"膝盖大部分时间弯着"，而历史 npz
    # 里只有 torque / foot_z / joint_vel，没有关节角，导致这个主诉无法量化。
    # 姿态类问题（蹲着走、躯干前倾、勾脚）全靠这一列，缺了就只能靠肉眼看视频。
    # root_pos_z 必须记：蹲姿的最终观测量是躯干高度，膝角可以被策略靠
    # "把屈曲转移到髋/踝"绕过，躯干高度绕不过去。
    records = {"lin_vel_x": [], "lin_vel_y": [], "foot_z": [], "torque": [],
               "joint_vel": [], "cmd_x": [], "joint_pos": [], "root_pos_z": []}
    if has_contact:
        records["foot_force_z"] = []

    t0 = time.time()
    for step in range(args_cli.steps):
        # pin 必须在 step 之前：观测是在 env.step() 内部读 command 生成的，
        # 放在 step 之后则策略当步看到的是被 resample 覆写过的指令。
        pin_command()
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

            records["lin_vel_x"].append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
            records["lin_vel_y"].append(robot.data.root_lin_vel_b[:, 1].cpu().numpy())
            records["foot_z"].append(
                robot.data.body_pos_w[:, foot_ids, 2].cpu().numpy().astype(np.float32)
            )
            records["torque"].append(robot.data.applied_torque.cpu().numpy().astype(np.float32))
            records["joint_vel"].append(robot.data.joint_vel.cpu().numpy().astype(np.float32))
            records["joint_pos"].append(robot.data.joint_pos.cpu().numpy().astype(np.float32))
            records["root_pos_z"].append(robot.data.root_pos_w[:, 2].cpu().numpy().astype(np.float32))
            if has_contact:
                csensor = base_env.scene["contact_forces"]
                records["foot_force_z"].append(
                    csensor.data.net_forces_w[:, sensor_ids, 2].cpu().numpy().astype(np.float32)
                )
            try:
                records["cmd_x"].append(
                    base_env.command_manager.get_command("base_velocity")[:, 0].cpu().numpy()
                )
            except Exception:
                records["cmd_x"].append(np.full(n, np.nan, dtype=np.float32))

        if step % 300 == 0:
            print(f"  step {step}/{args_cli.steps}  ({time.time()-t0:.0f}s)")

    print(f"[INFO] rollout done in {time.time()-t0:.0f}s")

    arr = {k: np.stack(v, axis=0) for k, v in records.items()}  # (T, N, ...)

    # ---- 指令纯净度自检 ---------------------------------------------------
    # 踩过坑才加的：现在就把污染的 env 点出来，而不是等到分析阶段才发现。
    # 判据：cmd_x 的均值偏离目标 > 0.02，或时间标准差 > 0.05（应为恒定量）。
    _cmd = arr["cmd_x"]
    _bad = []
    for i, v in enumerate(vel_list):
        _c = _cmd[:, i]
        _dev = float(abs(_c.mean() - v))
        _sd = float(_c.std())
        if _dev > 0.02 or _sd > 0.05:
            _bad.append(i)
            print(f"[ERROR] env#{i} 指令被污染：目标={v:.2f} "
                  f"实测 mean={_c.mean():.3f} std={_sd:.3f}")
            _seg = [f"{_c[int(k * 0.5 / dt):int((k + 1) * 0.5 / dt)].mean():.2f}"
                    for k in range(0, min(12, int(_c.shape[0] * dt / 0.5)))]
            print(f"        前 6 s 指令分段均值: {' '.join(_seg)}")
    if _bad:
        print("[ERROR] 以上 env 的数据不可用于速度相关指标，请修好后重采。")
    else:
        print(f"[INFO] 指令纯净度自检通过：{len(vel_list)} 个 env 全部恒定")

    tag = os.path.basename(resume_path).replace(".pt", "")
    npz = os.path.join(args_cli.out, f"gait_{tag}.npz")
    np.savez_compressed(npz, dt=dt, mass=mass, velocities=np.array(vel_list), **arr)
    print(f"[INFO] saved raw data -> {npz}")
    print("=== EVAL COLLECT DONE ===")

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
