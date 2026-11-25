# -*- coding: utf-8 -*-
"""受控对照：区分「策略不会做全向动作」还是「策略不会切换动作模式」。

背景（2026-11-25）：
  eval_coverage.py（从起步就恒定钉死一个指令）测出侧移/组合/前进中转全部可行，
  与 10-26 diag_omni.py「侧移/后退/原地转向 ≈0」冲突。但两者有三处口径差异：
  权重(model_4000 vs model_2999)、指令时序(行进中切换 vs 起步恒定)、task(Play vs 训练)。
  本脚本用**同一权重 model_2999 + 同一框架**，只改一个变量：指令时序。

设计（每个指令点成对出现，唯一差异是 mode）：
  constant 模式：step 0..END 恒定钉死目标指令（复现 eval_coverage.py）
  switch   模式：前 WARM 步钉死「前进 1.0」让策略稳定走起来，
                 WARM 之后切换到目标指令（复现 diag_omni.py 的脚本切换时序）
  稳态窗口对两种模式都在「切换后 +125 步」取，越过 11-20 测到的 0.9s 指令滞后。
  → 若 constant 可行而 switch 失效：证明是「模式切换/指令突变」问题，不是能力缺失；
    这正是 AMP 用阶跃指令训练 + 判别器风格先验锁死的表现（贴 2026 held-out velocity）。
  → 若两者都可行：说明 2999 相对 4000 全向能力确实恢复（权重差异）。

指令点选旧结论测过的确切值 + 本轮 v1 的可行点，做组内对照。
复用 eval_coverage.py 全部已验证机制：关 heading/resampling/standing、放开 ranges、
cmd 回读校验、逐 env 独立、跌倒只用 root_pos_z。

用法：
  python eval_coverage_switch.py --task LeggedLab-Isaac-AMP-G1-v0 --headless \
      --checkpoint <model_2999.pt> --out /data/eval_out
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="全向能力受控对照：恒定 vs 切换")
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-G1-v0")
parser.add_argument("--checkpoint", type=str, required=True)
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

# ---- 关键指令点（含旧结论测过的确切值）---------------------------------
POINTS = [
    ("lat_m04", 0.0, -0.4, 0.0),     # 旧 diag_omni 侧移 vy=-0.4，曾判"完全没执行"
    ("lat_p04", 0.0, 0.4, 0.0),
    ("back_m04", -0.4, 0.0, 0.0),    # 旧 diag_omni 后退 vx=-0.4，曾判"完全没执行"
    ("yaw_m074", 0.0, 0.0, -0.74),   # 旧 diag_omni 原地转 wz=-0.74，曾判"达成6%"
    ("yaw_p074", 0.0, 0.0, 0.74),
    ("lat_fwd", 1.0, -0.5, 0.0),     # v1 里可行的斜向侧移，作阳性对照
    ("mix", 1.0, 0.3, 0.3),           # v1 里可行的组合，作阳性对照
    ("fwd_1", 1.0, 0.0, 0.0),        # 前进对照
]

# 每个点两种 mode → 16 env
LABELS, MODES, GRID = [], [], []
for lab, vx, vy, wz in POINTS:
    for mode in ("constant", "switch"):
        LABELS.append(lab)
        MODES.append(mode)
        GRID.append([vx, vy, wz])
N = len(LABELS)

WARM = 150          # switch 模式前 150 步(3s) 钉死前进 1.0
SETTLE = 125        # 切换后丢 125 步(2.5s) 越过 0.9s 指令滞后
TOTAL = WARM + SETTLE + 150   # = 425 步，稳态窗口取最后 150 步


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    os.makedirs(args_cli.out, exist_ok=True)
    env_cfg.scene.num_envs = N
    env_cfg.seed = 42
    print(f"[SW] START N={N} total={TOTAL} warm={WARM} t={time.strftime('%H:%M:%S')}", flush=True)

    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    resume_path = retrieve_file_path(args_cli.checkpoint)
    print(f"[SW] Loading checkpoint: {resume_path}", flush=True)
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

    try:
        term = base_env.command_manager.get_term("base_velocity")
        term.cfg.resampling_time_range = (1e9, 1e9)
        term.cfg.rel_standing_envs = 0.0
        term.cfg.heading_command = False
        try:
            term.cfg.rel_heading_envs = 0.0
        except Exception:
            pass
        try:
            term.is_heading_env[:] = False
        except Exception:
            pass
        term.cfg.ranges.lin_vel_x = (-2.0, 4.0)
        term.cfg.ranges.lin_vel_y = (-2.0, 2.0)
        term.cfg.ranges.ang_vel_z = (-2.0, 2.0)
        print("[SW] 已关 resampling/standing/heading 并放开 ranges", flush=True)
    except Exception as exc:
        print(f"[SW][WARN] 关指令机制失败({exc})", flush=True)

    grid = torch.tensor(GRID, device=dev, dtype=torch.float32)          # (N,3)
    is_switch = torch.tensor([m == "switch" for m in MODES], device=dev)  # (N,)
    warm_cmd = torch.tensor([1.0, 0.0, 0.0], device=dev, dtype=torch.float32)

    def pin_command(step):
        try:
            cmd = base_env.command_manager.get_command("base_velocity")
            tgt = grid.clone()
            # switch 模式在 WARM 之前钉死前进 1.0
            if step < WARM:
                tgt[is_switch] = warm_cmd
            cmd[:, 0] = tgt[:, 0]
            cmd[:, 1] = tgt[:, 1]
            cmd[:, 2] = tgt[:, 2]
            return cmd.detach().clone()
        except Exception as exc:
            print(f"[SW][WARN] pin 失败 {exc}", flush=True)
            return grid

    records = {"lin_vel_x": [], "lin_vel_y": [], "ang_vel_z": [],
               "root_pos_z": [], "cmd_rb": []}
    obs = env.get_observations()
    pin_command(0)
    print(f"[SW] KIT_READY t={time.strftime('%H:%M:%S')}", flush=True)
    t0 = time.time()
    for step in range(TOTAL):
        cmd_rb = pin_command(step)
        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)
            records["lin_vel_x"].append(robot.data.root_lin_vel_b[:, 0].cpu().numpy())
            records["lin_vel_y"].append(robot.data.root_lin_vel_b[:, 1].cpu().numpy())
            records["ang_vel_z"].append(robot.data.root_ang_vel_b[:, 2].cpu().numpy())
            records["root_pos_z"].append(robot.data.root_pos_w[:, 2].cpu().numpy())
        records["cmd_rb"].append(cmd_rb[:, 0:3].cpu().numpy())
        if step % 100 == 0:
            print(f"[SW] step {step}/{TOTAL} ({time.time()-t0:.0f}s)", flush=True)

    arr = {k: np.stack(v, axis=0) for k, v in records.items()}
    tag = os.path.basename(resume_path).replace(".pt", "")
    npz = os.path.join(args_cli.out, f"coverage_switch_{tag}.npz")
    np.savez_compressed(
        npz, dt=dt, labels=np.array(LABELS, dtype=object),
        modes=np.array(MODES, dtype=object), grid=np.array(GRID, dtype=np.float32),
        warm=WARM, settle=SETTLE, total=TOTAL, **arr)
    print(f"[SW] saved -> {npz}  shape={arr['lin_vel_x'].shape}", flush=True)
    print("=== SWITCH COLLECT DONE ===", flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
