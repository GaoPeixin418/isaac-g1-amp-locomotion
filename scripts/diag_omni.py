"""
diag_omni.py - 全向脚本无渲染重放 + yaw/速度轨迹记录
=============================================================================
背景（2026-10-26）：amp4000_omni2.mp4 里用户观察到"20s 后站定、只转了约150°"。
录制日志显示 8 段脚本全部进入，所以要么是
  a) 后半段本来就是低速/原地（脚本编排问题），要么是
  b) heading 闭环没真正把机器人转过去（转向执行失败）。
区分 a/b 只能靠 yaw 曲线：每段目标转向 vs 实际达成转向。

用法（服务器）：
  /data/envs/isaaclab/bin/python /data/setup/diag_omni.py \
      --checkpoint /data/legged_lab/logs/rsl_rl/g1_amp/2026-10-19_16-37-21/model_4000.pt \
      --script "2,0,0,0; 8,1.0,0,0; 13,1.0,0,1.5708; 18,1.0,0,0; 22,0,-0.4,0; 26,-0.4,0,0; 30,0,0,-1.5708; 36,0,0,0" \
      --steps 1800
"""
import argparse
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", type=str, default="LeggedLab-Isaac-AMP-G1-Play-v0")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--script", type=str, required=True)
parser.add_argument("--steps", type=int, default=1800)
parser.add_argument("--ramp", type=float, default=1.0)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.utils.assets import retrieve_file_path  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
from rsl_rl.runners import AMPRunner  # noqa: E402

import legged_lab.tasks  # noqa: F401,E402


@hydra_task_config(args_cli.task, "rsl_rl_cfg_entry_point")
def main(env_cfg, agent_cfg):
    env_cfg.scene.num_envs = 1
    env_cfg.seed = 42
    env = gym.make(args_cli.task, cfg=env_cfg)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = AMPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(retrieve_file_path(args_cli.checkpoint), map_location=agent_cfg.device)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    try:
        policy_nn = runner.alg.policy
    except AttributeError:
        policy_nn = runner.alg.actor_critic

    base = env.unwrapped
    robot = base.scene["robot"]
    dt = float(base.step_dt)

    # ---- 复刻 patch_play_cmd 的 omni 逻辑（必须与录制路径一致） ----------
    script = []
    for p in args_cli.script.split(";"):
        p = p.strip()
        if not p:
            continue
        v = [float(x) for x in p.split(",")]
        script.append((v[0], v[1], v[2], v[3] if len(v) > 3 else 0.0))

    term = base.command_manager.get_term("base_velocity")
    term.cfg.heading_command = True
    term.cfg.ranges.ang_vel_z = (-1.0, 1.0)
    try:
        term.cfg.rel_heading_envs = 1.0
        term.is_heading_env[:] = True
    except Exception:
        pass

    cur = torch.zeros(2, dtype=torch.float32, device=base.device)
    seg = -1
    t = 0.0
    dev = base.device

    obs = env.get_observations()
    log = []
    seg_enter_yaw = None

    for step in range(args_cli.steps):
        t += dt
        si = len(script) - 1
        for i, s in enumerate(script):
            if t <= s[0]:
                si = i
                break
        if si != seg:
            seg = si
            yaw = robot.data.heading_w
            term.heading_target[:] = yaw + script[si][3]
            seg_enter_yaw = float(yaw[0]) if hasattr(yaw, "__len__") else float(yaw)
            print(f"[SEG] t={t:5.1f}s seg{si} target_v=({script[si][1]:+.2f},{script[si][2]:+.2f}) "
                  f"d_yaw={script[si][3]:+.2f} enter_yaw={np.degrees(seg_enter_yaw):+7.1f} deg")

        tgt = torch.tensor(script[si][1:3], dtype=torch.float32, device=dev)
        d = tgt - cur
        mx = args_cli.ramp * dt
        cur = cur + torch.clamp(d, -mx, mx)
        cmd = base.command_manager.get_command("base_velocity")
        cmd[:, 0] = cur[0].item()
        cmd[:, 1] = cur[1].item()

        with torch.inference_mode():
            actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            policy_nn.reset(dones)

        yaw = float(robot.data.heading_w[0])
        vx = float(robot.data.root_lin_vel_b[0, 0])
        vy = float(robot.data.root_lin_vel_b[0, 1])
        wz = float(robot.data.root_ang_vel_b[0, 2])
        log.append((t, yaw, vx, vy, wz, float(cmd[0, 0]), float(cmd[0, 1]),
                    float(cmd[0, 2]) if cmd.shape[1] > 2 else 0.0))

    # ---- 汇总：每秒一行 + 每段转向达成率 ----
    print("\n=== 每秒轨迹 (t, yaw_deg, vx, vy, wz_cmd) ===")
    arr = np.array(log)
    for k in range(0, len(arr), int(1.0 / dt)):
        r = arr[k]
        print(f"t={r[0]:5.1f} yaw={np.degrees(r[1]):+8.1f} vx={r[2]:+5.2f} vy={r[3]:+5.2f} "
              f"wz={r[4]:+5.2f} cmd=({r[5]:+.2f},{r[6]:+.2f},{r[7]:+.2f})")

    print("\n=== 段转向达成率 ===")
    t_arr = arr[:, 0]
    yaw_arr = np.degrees(arr[:, 1])
    for i, s in enumerate(script):
        if abs(s[3]) < 1e-6:
            continue
        t_in = script[i - 1][0] if i > 0 else 0.0
        t_out = s[0]
        y_in = yaw_arr[np.argmin(np.abs(t_arr - t_in))]
        y_out = yaw_arr[np.argmin(np.abs(t_arr - t_out))]
        got = y_out - y_in
        # wrap 到 [-180, 180]
        got = (got + 180) % 360 - 180
        want = np.degrees(s[3])
        print(f"seg{i}: 目标 {want:+.0f} deg / 实际 {got:+.0f} deg / 达成 {got/want*100:.0f}%")

    print("=== OMNI DIAG DONE ===")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
