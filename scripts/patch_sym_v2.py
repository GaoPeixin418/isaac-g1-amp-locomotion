#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
补丁 v2：根治「跛行 + 碎步蹲姿 + 转弯晃」（2026-10-27 晚）

v1 的实测结果（从零训 3000 iter，统一口径 @1.0 m/s）
-----------------------------------------------------------------------------
  对称性  legAI 42.6%(2400) -> 46.2%(2800) -> 48.7%(2999)   后期回升，没根治
  步态尺度 步频 150 -> 136（参考 95-103）  步长 0.79 -> 0.89（参考 1.17-1.27）
  姿态    膝均屈 20.0° -> 16.7°（参考 15.0°）

三个病的因果链（本机实测 + 机制推断）
-----------------------------------------------------------------------------
  蹲（膝一直弯、躯干压低）
      -> 腿的有效摆长变短 -> 步长只有参考的 65%
      -> 为了满足 1.0 m/s 只能靠提高步频凑 -> 碎步、"像快走"
  而 v1 的对称惩罚本身留了后门：
      W = EMA(tau^2)，时间窗 0.4 s ≈ 半个步态周期。
      步频越高，窗口内左右交替次数越多，tau^2 被平均得越接近 1
      -> "加快步频就能刷低不对称" -> 蹲 + 碎步被同时奖励。
  tau^2 还有第二个漏洞：等长收缩（关节不动、纯撑着）时 omega≈0，
      真实做功为 0，但 tau^2 很大，会把"站着撑"误判成"在使劲"。

v2 改什么（四项）
-----------------------------------------------------------------------------
1. leg_torque_asymmetry -> leg_work_asymmetry
   * 量从 tau^2 换成 |tau * omega|（机械功率 = 做功速率），堵住等长收缩漏洞
   * time_constant 0.4 s -> 1.5 s（>= 1 个完整步态周期，参考周期约 1.2 s），
     堵住"加快步频平均掉不对称"的后门
   * 权重 -1.0 -> -2.0（v1 后期压不住任务奖励）

2. 新增 base_height_bias（治蹲的主项，策略绕不过去）
   躯干高度是"蹲不蹲"的最终观测量 —— 只罚膝角的话，策略可以把屈曲
   从膝转移到髋/踝，膝角看着正常但人照样蹲着。
   target 0.765 m = 参考动作在 0.8-1.3 m/s 的实测 root 高度。
   deadband 0.05 m 内不罚，避免把步态冻死。

3. 新增 knee_bias（治蹲的姿态侧，工作点靠向参考 15°）
   罚的是 EMA 膝角的**工作点**偏离，不是瞬时值 —— 不限制屈伸幅度（ROM），
   只把"停在半弯"这个工作点拉回参考。deadband ±10°。

4. ang_vel_xy_l2 权重 -0.05 -> -0.2（治躯干晃动 / 转弯不稳）
   这是最便宜的平衡项：罚躯干俯仰+横滚角速度。

已知的归因边界（写清楚，别当消融用）
-----------------------------------------------------------------------------
一次改四项，成功时无法归因到单项。这是工程修复不是受控消融。

用法
-----------------------------------------------------------------------------
    python patch_sym_v2.py            # 打补丁（幂等，自动备份）
    python patch_sym_v2.py --revert   # 还原到 v2 之前
    python patch_sym_v2.py --check    # 只检查状态
=============================================================================
"""
import shutil
import sys
from pathlib import Path

REWARDS = Path(
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"
)
ENV_CFG = Path(
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
)

MARK_FUNC = "[PATCHED-SYM-V2]"
MARK_TERM = "[PATCHED-SYM-V2]"

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]
KNEE_JOINTS = ["left_knee_joint", "right_knee_joint"]

# 参考动作实测（ref_height_stats.py，0.8-1.3 m/s 档）
REF_BASE_HEIGHT = 0.765   # m
REF_KNEE_MEAN_DEG = 15.0  # deg

FUNC_CODE = '''


# ---------------------------------------------------------------------------
# [PATCHED-SYM-V2]
# ---------------------------------------------------------------------------
def leg_work_asymmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_leg_joints: int = 6,
    time_constant: float = 1.5,
) -> torch.Tensor:
    """Penalize left/right asymmetry in *mechanical work*, not raw torque.

    Two fixes over the v1 (tau^2, 0.4 s) version:

    1. |tau * omega| instead of tau^2.  tau^2 is large during isometric
       contraction (joint barely moving, just holding the body up) even though
       no mechanical work is done, so it over-counts "effort" in stance and
       leaves a loophole.
    2. time_constant >= one full gait cycle (~1.2 s at 1 m/s).  With a 0.4 s
       window the EMA averages over less than half a cycle, so stepping faster
       makes the two legs' windows look more alike -- the policy could lower
       this penalty simply by walking faster, which is exactly the shuffling
       gait we observed.
    """
    asset = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]  # (N, J)
    vel = asset.data.joint_vel[:, asset_cfg.joint_ids]       # (N, J)

    pw = (tau * vel).abs()  # mechanical power magnitude = work rate
    pw_l = pw[:, :num_leg_joints]
    pw_r = pw[:, num_leg_joints : 2 * num_leg_joints]
    sq = torch.stack((pw_l, pw_r), dim=0)  # (2, N, 6)

    dt = float(env.step_dt)
    decay = float(torch.exp(torch.tensor(-dt / time_constant)))

    ema = getattr(env, "_leg_work_asym_ema", None)
    if ema is None or ema.shape != sq.shape:
        ema = sq.clone()
    else:
        ema = ema * decay + sq * (1.0 - decay)
    fresh = (env.episode_length_buf == 0).view(1, -1, 1).to(ema.device)
    ema = torch.where(fresh, sq, ema)
    setattr(env, "_leg_work_asym_ema", ema)

    w_l, w_r = ema[0], ema[1]
    rel = (w_l - w_r).abs() / (w_l + w_r + 1.0)
    return rel.mean(dim=-1)


def base_height_bias(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_height: float = 0.765,
    deadband: float = 0.05,
) -> torch.Tensor:
    """Keep the torso at the reference height (anti-crouch, deadband outside).

    Torso height is the observable the policy cannot dodge: penalizing only
    the knee lets it move the flexion to hip/ankle and stay crouched while the
    knee angle looks fine.  Target comes from the reference motion itself
    (0.765 m at 0.8-1.3 m/s), not from a guess.
    """
    asset = env.scene[asset_cfg.name]
    rp = getattr(asset.data, "root_pos_w", None)
    if rp is None:
        rp = asset.data.root_link_pos_w
    h = rp[:, 2]
    err = (h - target_height).abs() - deadband
    return torch.clamp(err, min=0.0)


def knee_bias(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target: float = 0.262,
    deadband: float = 0.17,
    time_constant: float = 1.5,
) -> torch.Tensor:
    """Pull the knee *operating point* towards the reference (15 deg).

    Penalizes the EMA of the knee angle, not the instantaneous value, so the
    flexion/extension range (ROM) stays free -- only the "parked half-bent"
    operating point is corrected.  Deadband +/-10 deg covers the reference
    p10-p90 band (3.8-29.6 deg).
    """
    asset = env.scene[asset_cfg.name]
    q = asset.data.joint_pos[:, asset_cfg.joint_ids]  # (N, 2)

    dt = float(env.step_dt)
    decay = float(torch.exp(torch.tensor(-dt / time_constant)))

    ema = getattr(env, "_knee_bias_ema", None)
    if ema is None or ema.shape != q.shape:
        ema = q.clone()
    else:
        ema = ema * decay + q * (1.0 - decay)
    fresh = (env.episode_length_buf == 0).view(-1, 1).to(ema.device)
    ema = torch.where(fresh, q, ema)
    setattr(env, "_knee_bias_ema", ema)

    err = (ema - target).abs() - deadband
    return torch.clamp(err, min=0.0).mean(dim=-1)
'''

TERM_LEG_WORK = '''
    # [PATCHED-SYM-V2] 左右腿做功不对称（|tau*omega| 的周期级 EMA，不是瞬时 tau^2）
    leg_work_asymmetry = RewTerm(
        func=mdp.leg_work_asymmetry,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    %s,
                ],
            ),
            "num_leg_joints": 6,
            "time_constant": 1.5,
        },
    )
''' % (",\n                    ".join('"%s"' % n for n in LEG_JOINTS),)

TERM_HEIGHT = '''
    # [PATCHED-SYM-V2] 躯干高度靠向参考（治蹲的主项，策略无法靠转移屈曲躲避）
    base_height_bias = RewTerm(
        func=mdp.base_height_bias,
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "target_height": %.3f,
            "deadband": 0.05,
        },
    )
    # [PATCHED-SYM-V2] 膝角工作点靠向参考（只调工作点，不限制屈伸幅度）
    knee_bias = RewTerm(
        func=mdp.knee_bias,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    %s,
                ],
            ),
            "target": 0.262,
            "deadband": 0.17,
            "time_constant": 1.5,
        },
    )
''' % (REF_BASE_HEIGHT, ",\n                    ".join('"%s"' % n for n in KNEE_JOINTS),)

ANCHOR = "    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)"
OLD_ANGVEL = "    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.05)"
NEW_ANGVEL = "    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.2)"


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".presymv2")
    if not b.exists():
        shutil.copy2(p, b)
        print("  备份 -> %s" % b)


def revert():
    for p in (REWARDS, ENV_CFG):
        b = p.with_suffix(p.suffix + ".presymv2")
        if b.exists():
            shutil.copy2(b, p)
            print("已还原 %s" % p)
        else:
            print("!! %s 没有 v2 备份，跳过" % p)
    print("done")


def check():
    r = REWARDS.read_text(encoding="utf-8", errors="ignore")
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")
    print("rewards.py   函数: leg_work_asymmetry=%s base_height_bias=%s knee_bias=%s"
          % ("在" if "def leg_work_asymmetry" in r else "无",
             "在" if "def base_height_bias" in r else "无",
             "在" if "def knee_bias" in r else "无"))
    print("env_cfg.py   项: leg_work=%s height=%s knee=%s angvel0.2=%s"
          % ("在" if "leg_work_asymmetry = RewTerm" in c else "无",
             "在" if "base_height_bias = RewTerm" in c else "无",
             "在" if "knee_bias = RewTerm" in c else "无",
             "是" if NEW_ANGVEL in c else "否"))
    print("env_cfg.py   残留 v1: %s"
          % ("有（应替换）" if "leg_torque_asymmetry" in c else "无"))


def _replace_rewterm_block(text: str, name: str, new_block: str) -> str:
    """把 `    <name> = RewTerm(` 起到下一个顶层 `    )` 的整块替换掉。"""
    lines = text.split("\n")
    start = None
    for i, ln in enumerate(lines):
        if ln.strip().startswith(name + " = RewTerm("):
            start = i
            break
    if start is None:
        return None
    end = None
    for j in range(start + 1, len(lines)):
        if lines[j] == "    )":
            end = j
            break
    if end is None:
        return None
    return "\n".join(lines[:start]) + new_block.rstrip("\n") + "\n" + "\n".join(lines[end + 1:])


def apply():
    # --- 1) 奖励函数 ---
    r = REWARDS.read_text(encoding="utf-8", errors="ignore")
    if MARK_FUNC in r:
        print("== rewards.py 已有 v2 函数，跳过")
    else:
        backup(REWARDS)
        if not r.endswith("\n"):
            r += "\n"
        REWARDS.write_text(r + FUNC_CODE, encoding="utf-8")
        print("== rewards.py 已追加 3 个函数")

    # --- 2) 奖励项配置 ---
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")

    if "leg_torque_asymmetry = RewTerm(" in c:
        backup(ENV_CFG)
        c2 = _replace_rewterm_block(c, "leg_torque_asymmetry", TERM_LEG_WORK)
        if c2 is None:
            print("!! 定位 leg_torque_asymmetry 块失败，放弃")
            sys.exit(1)
        c = c2
        print("== env_cfg.py: leg_torque_asymmetry -> leg_work_asymmetry (w=-2.0, tc=1.5)")
    elif "leg_work_asymmetry = RewTerm(" in c:
        print("== env_cfg.py 已是 leg_work_asymmetry，跳过")
    else:
        print("!! 既没有 v1 也没有 v2 的对称项，检查配置")
        sys.exit(1)

    if "base_height_bias = RewTerm(" not in c:
        backup(ENV_CFG)
        if ANCHOR not in c:
            print("!! 找不到锚点，放弃")
            sys.exit(1)
        c = c.replace(ANCHOR, TERM_HEIGHT + "\n" + ANCHOR, 1)
        print("== env_cfg.py 已插入 base_height_bias + knee_bias")
    else:
        print("== env_cfg.py 已有 base_height_bias/knee_bias，跳过")

    if OLD_ANGVEL in c:
        backup(ENV_CFG)
        c = c.replace(OLD_ANGVEL, NEW_ANGVEL, 1)
        print("== env_cfg.py: ang_vel_xy_l2 -0.05 -> -0.2")
    elif NEW_ANGVEL in c:
        print("== env_cfg.py ang_vel_xy_l2 已是 -0.2，跳过")
    else:
        print("!! ang_vel_xy_l2 行没匹配上，请手工确认")

    ENV_CFG.write_text(c, encoding="utf-8")
    check()


def verify_syntax():
    import ast
    import py_compile

    ok = True
    for p in (REWARDS, ENV_CFG):
        try:
            py_compile.compile(str(p), doraise=True)
            print("  编译通过: %s" % p.name)
        except Exception as e:
            print("  !! 编译失败 %s: %r" % (p.name, e))
            ok = False

    names = [n.name for n in ast.parse(
        REWARDS.read_text(encoding="utf-8", errors="ignore")).body
        if isinstance(n, ast.FunctionDef)]
    for f in ("leg_work_asymmetry", "base_height_bias", "knee_bias"):
        if f in names:
            print("  函数已定义: %s" % f)
        else:
            print("  !! 缺函数 %s" % f)
            ok = False
    return ok


if __name__ == "__main__":
    args = sys.argv[1:]
    if "--revert" in args:
        revert()
    elif "--check" in args:
        check()
    else:
        apply()
        print("--- 校验 ---")
        ok = verify_syntax()
        print("\n结果: %s" % ("通过，可以启动训练" if ok else "!! 未通过，不要启动训练"))
        sys.exit(0 if ok else 1)
