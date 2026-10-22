#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""给 G1AmpRewards 增加「步态对称性」与「抬脚高度」两项奖励。

=============================================================================
为什么加这两项（2026-10-22 诊断结论）
=============================================================================
实测 model_4000 的左右腿对称性指数（SI = 2|L-R|/(L+R)）：

    档位      0.5     1.0     1.5     2.0     2.5     3.0
    抬升SI   16.1%    7.7%   27.7%   13.1%   11.1%   21.0%
    支撑SI   16.7%    3.1%   11.8%    0.0%    0.0%    0.0%

人类正常步态 SI < 5%。而且 6 档里有 5 档是「左脚抬升 < 右脚」，
方向高度一致 —— 是系统性偏置，不是噪声。

根本原因：`G1AmpRewards` 的 15 个奖励项里，**没有任何一项约束左右腿是否对称**。
优化器没有动力去修它。

为什么框架自带的 symmetry 没救回来：
`rsl_rl_ppo_cfg.py` 里确实开了
    symmetry_cfg = RslRlSymmetryCfg(use_data_augmentation=True,
                                     use_mirror_loss=True, mirror_loss_coeff=0.1)
但它约束的是  π(mirror(s)) = mirror(π(s))  —— 同一时刻的镜像等变性。
而跛行的本质是「一个周期内」左脚迈 0.14m、右脚迈 0.19m。
这个信息不在单帧状态里，所以 mirror loss 在数学上就抓不到。

（这与判别器的盲区同构：判别器只看 4 帧看不到 25 秒横向漂移；
  mirror loss 只看单帧看不到一个周期的左右不对称。）

=============================================================================
新增的两个奖励项
=============================================================================
1. air_time_variance_penalty  (weight -2.0)
   实现：torch.var(clip(last_air_time, max=0.5), dim=1)
       + torch.var(clip(last_contact_time, max=0.5), dim=1)
   它对「两只脚上次的腾空时长」和「支撑时长」分别求方差 ——
   **正是周期级的左右对称性**，恰好补上 mirror loss 的盲区。
   参考：GO2 flat_env_cfg.py 用 -1.0；此处翻倍到 -2.0，
   因为 λ=0.5 的 AMP 把任务奖励稀释了一半。

2. feet_clearance  (weight 1.0, target_height 0.1)
   奖励摆动脚抬到目标高度。实测支撑期脚底 z=0.035m，
   目标抬升 0.1 → 世界坐标 0.1（与 G1 官方 flat_env_cfg 完全一致）。
   针对「左脚抬升偏低 → 拖地 → 看起来瘸」。

=============================================================================
用法
=============================================================================
    python scripts/patch_amp_rewards.py [g1_amp_env_cfg.py 路径]

安全性：幂等 + 备份 .prerew + 找不到锚点不动文件 + 回读校验
"""
import shutil
import sys
from pathlib import Path

MARK = "# [PATCHED-GAIT-SYMMETRY]"
DEFAULT = (
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/"
    "config/g1/g1_amp_env_cfg.py"
)

IMPORT_ANCHOR = "import legged_lab.tasks.locomotion.amp.mdp as mdp"
IMPORT_ADD = (
    "\n# 步态对称性奖励需要 legged_lab 的 velocity.mdp（Isaac Lab 原版里没有这两个函数）\n"
    "import legged_lab.tasks.locomotion.velocity.mdp as mdp_vel"
)

REW_ANCHOR = "    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-50.0)"
REW_BLOCK = """    # ---- 步态对称性与抬脚高度（2026-10-22 新增）----
    {mark}
    # 左右脚「腾空时长 / 支撑时长」的方差 -> 直接惩罚跛行。
    # 与 mirror loss 的区别：这一项看的是周期级统计量，不是单帧镜像。
    air_time_variance = RewTerm(
        func=mdp_vel.air_time_variance_penalty,
        weight=-2.0,
        params={{"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_ankle_roll_link")}},
    )
    # 摆动脚抬到目标高度，避免拖地（G1 官方 flat 配置同参）
    feet_clearance = RewTerm(
        func=mdp_vel.feet_clearance,
        weight=1.0,
        params={{
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*ankle_roll.*"),
        }},
    )

""".format(mark=MARK) + REW_ANCHOR


def patch(path: Path) -> int:
    if not path.exists():
        print(f"!! 找不到 {path}")
        return 1

    src = path.read_text(encoding="utf-8")
    if MARK in src:
        print(f"== 已打过补丁，跳过: {path}")
        return 0

    if IMPORT_ANCHOR not in src:
        print(f"!! 没找到 import 锚点: {IMPORT_ANCHOR}")
        return 2
    if REW_ANCHOR not in src:
        print(f"!! 没找到奖励锚点: {REW_ANCHOR}")
        return 2

    new = src.replace(IMPORT_ANCHOR, IMPORT_ANCHOR + IMPORT_ADD, 1)
    new = new.replace(REW_ANCHOR, REW_BLOCK, 1)

    bak = path.with_suffix(path.suffix + ".prerew")
    if not bak.exists():
        shutil.copy2(path, bak)
        print(f"== 已备份 -> {bak}")

    path.write_text(new, encoding="utf-8")
    print(f"== 已打补丁 -> {path}")

    chk = path.read_text(encoding="utf-8")
    for key, desc in [
        ("import legged_lab.tasks.locomotion.velocity.mdp as mdp_vel", "import"),
        ("air_time_variance_penalty", "air_time_variance"),
        ("feet_clearance", "feet_clearance"),
        ("weight=-2.0", "weight -2.0"),
        ('"target_height": 0.1', "target_height 0.1"),
    ]:
        print(f"   [{'OK' if key in chk else 'MISS'}] {desc}")
    return 0


def main() -> int:
    p = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT)
    return patch(p)


if __name__ == "__main__":
    raise SystemExit(main())
