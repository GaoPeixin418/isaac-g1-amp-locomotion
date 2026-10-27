#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
补丁：给 G1 AMP 任务加"左右腿出力不对称"惩罚（2026-10-27）

背景
-----------------------------------------------------------------------------
2026-10-26 的 finetune 加了 air_time_variance（罚两脚腾空**时长**的方差），
结果跛行从 36.7% 到 37.1%，纹丝不动。原因：时长和出力是两个物理量 ——
一条腿多使一倍的劲，完全可以做到和另一条腿落地一样久。

mirror loss（use_mirror_loss=True, coeff=0.1）一直开着也没治好，因为它约束的
是"策略函数在镜像观测下的输出对称性"，不是实际跑出来的轨迹对称性。

所以这一项直接罚**轨迹层面的出力不对称**。

关键设计：为什么不能比瞬时扭矩
-----------------------------------------------------------------------------
正常行走左右腿是反相的：一条腿支撑（扭矩大）、另一条摆动（扭矩小），
**瞬时值天然不相等**。直接罚 (tau_L - tau_R) 会逼出"双腿同步用力"的跳跃步态，
比跛行更糟。

所以这里比的是**一个步态周期内各自扭矩平方的均值** —— 用指数滑动平均近似
滑动时间窗，得到两腿各自的 RMS。RMS 与相位无关，正是人眼看到的
"这条腿一直在使劲"。

    rel = |m_L - m_R| / (m_L + m_R + 1.0)      ∈ [0, 1)
    m = EMA(tau^2)，时间常数 0.4 s ≈ 一个步态周期

已知风险（要监控）
-----------------------------------------------------------------------------
1. rel 是相对量：两腿扭矩一起降到 0 也能让 rel=0。理论上策略可以靠"双腿都
   不出力"作弊，但那样走不动，任务奖励会把它拉回来。监控方法：看 |tau| 的
   绝对值均值是否塌下去。
2. 和 air_time_variance / feet_clearance / mirror loss 同时开着，成功时无法
   归因到单项。这是工程修复不是消融。

用法
-----------------------------------------------------------------------------
    python patch_torque_sym.py            # 打补丁（幂等，自动备份）
    python patch_torque_sym.py --revert   # 还原
    python patch_torque_sym.py --check    # 只检查状态
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

MARK_FUNC = "[PATCHED-TORQUE-SYMMETRY-v1]"
MARK_TERM = "[PATCHED-TORQUE-SYMMETRY-v1] 左右腿出力不对称"

LEG_JOINTS = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
]

FUNC_CODE = '''


# ---------------------------------------------------------------------------
# [PATCHED-TORQUE-SYMMETRY-v1]
# ---------------------------------------------------------------------------
def leg_torque_asymmetry(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    num_leg_joints: int = 6,
    time_constant: float = 0.4,
) -> torch.Tensor:
    """Penalize left/right leg effort asymmetry (RMS over a gait-cycle window).

    Why not compare instantaneous torque: in walking the two legs are in
    anti-phase, so one leg is always in stance (large torque) while the other
    swings (small torque). Penalizing tau_L - tau_R directly would push the
    policy towards a synchronous hopping gait.

    Instead we compare each leg's own RMS (EMA of tau^2) over a ~0.4 s window,
    which is phase-independent and matches what "one leg works harder" looks
    like to the human eye.
    """
    asset = env.scene[asset_cfg.name]
    tau = asset.data.applied_torque[:, asset_cfg.joint_ids]  # (N, J)

    tau_l = tau[:, :num_leg_joints]
    tau_r = tau[:, num_leg_joints : 2 * num_leg_joints]
    sq = torch.stack((tau_l.square(), tau_r.square()), dim=0)  # (2, N, 6)

    dt = float(env.step_dt)
    decay = float(torch.exp(torch.tensor(-dt / time_constant)))

    ema = getattr(env, "_leg_torque_asym_ema", None)
    if ema is None or ema.shape != sq.shape:
        ema = sq.clone()
    else:
        ema = ema * decay + sq * (1.0 - decay)
    # environments that were just reset: drop the stale history
    fresh = (env.episode_length_buf == 0).view(1, -1, 1).to(ema.device)
    ema = torch.where(fresh, sq, ema)
    setattr(env, "_leg_torque_asym_ema", ema)

    m_l, m_r = ema[0], ema[1]
    rel = (m_l - m_r).abs() / (m_l + m_r + 1.0)  # 0 = symmetric, ->1 = single-leg
    return rel.mean(dim=-1)
'''

_JN = ", ".join('"%s"' % n for n in LEG_JOINTS)
TERM_CODE = '''
    # [PATCHED-TORQUE-SYMMETRY-v1] 左右腿出力不对称（直接罚"使劲多少"，不是"时长"）
    leg_torque_asymmetry = RewTerm(
        func=mdp.leg_torque_asymmetry,
        weight=-1.0,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    %s,
                ],
            ),
            "num_leg_joints": 6,
            "time_constant": 0.4,
        },
    )
''' % (",\n                    ".join('"%s"' % n for n in LEG_JOINTS),)

ANCHOR = "    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)"


def backup(p: Path):
    b = p.with_suffix(p.suffix + ".pretorque")
    if not b.exists():
        shutil.copy2(p, b)
        print("  备份 -> %s" % b)


def revert():
    for p in (REWARDS, ENV_CFG):
        b = p.with_suffix(p.suffix + ".pretorque")
        if b.exists():
            shutil.copy2(b, p)
            print("已还原 %s" % p)
        elif MARK_FUNC in p.read_text(encoding="utf-8", errors="ignore"):
            print("!! %s 有补丁标记但没有备份，不敢还原" % p)
    print("done")


def check():
    r = REWARDS.read_text(encoding="utf-8", errors="ignore")
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")
    print("rewards.py  函数: %s" % ("在" if MARK_FUNC in r else "无"))
    print("env_cfg.py  奖励项: %s" % ("在" if MARK_TERM in c else "无"))


def apply():
    # --- 1) 奖励函数 ---
    r = REWARDS.read_text(encoding="utf-8", errors="ignore")
    if MARK_FUNC in r:
        print("== rewards.py 已有补丁，跳过")
    else:
        backup(REWARDS)
        if not r.endswith("\n"):
            r += "\n"
        REWARDS.write_text(r + FUNC_CODE, encoding="utf-8")
        print("== rewards.py 已追加 leg_torque_asymmetry()")

    # --- 2) 奖励项配置 ---
    c = ENV_CFG.read_text(encoding="utf-8", errors="ignore")
    if MARK_TERM in c:
        print("== env_cfg.py 已有奖励项，跳过")
    elif ANCHOR not in c:
        print("!! 找不到插入锚点 %r，放弃（绝不破坏原文件）" % ANCHOR)
        sys.exit(1)
    else:
        backup(ENV_CFG)
        ENV_CFG.write_text(c.replace(ANCHOR, TERM_CODE + "\n" + ANCHOR, 1), encoding="utf-8")
        print("== env_cfg.py 已插入 leg_torque_asymmetry 奖励项 (weight=-1.0)")

    check()


def verify_syntax():
    """静态校验：能编译 + 函数确实定义了 + 奖励项确实插入了。

    注意：不能在这里 import isaaclab —— 裸 python 缺 pxr（Isaac Sim 运行时
    的模块），必须走 Kit 启动的 python。所以运行时验证留给 train.py 起来
    的前几十秒（mdp.leg_torque_asymmetry 解析不到会立刻报 AttributeError）。
    """
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

    tree = ast.parse(REWARDS.read_text(encoding="utf-8", errors="ignore"))
    names = [n.name for n in tree.body if isinstance(n, ast.FunctionDef)]
    if "leg_torque_asymmetry" in names:
        print("  函数已定义: leg_torque_asymmetry")
    else:
        print("  !! rewards.py 里找不到 leg_torque_asymmetry")
        ok = False

    if MARK_TERM in ENV_CFG.read_text(encoding="utf-8", errors="ignore"):
        print("  奖励项已插入: leg_torque_asymmetry (weight=-1.0)")
    else:
        print("  !! env_cfg.py 里找不到奖励项")
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
