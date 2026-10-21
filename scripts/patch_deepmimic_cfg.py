#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
17 - 把 DeepMimic 的动作数据改成 AMP 用的那 30 段

为什么不用 hydra 命令行覆盖：
   motion_data_weights 是 30 个 key 的 dict，key 里带 '-' '_-__' 这类字符，
   hydra 的 OmegaConf 解析容易翻车。而且注释里说得很清楚 —— motion_data_dir
   和 weights 是配对的，改一个就得改另一个。用脚本同时改两处最稳。

公平性考量（这是整个实验的关键）：
   DeepMimic 默认跟的是「后踢」这种花活。拿它跟 AMP 比没有任何意义 ——
   两者看的根本不是同一批动作。
   改成 AMP 那 30 段走跑之后，两边才真正做到同源：
       同一个机器人 (G1 29DOF)
       同一批参考数据 (30 段 LAFAN1 走跑)
       同一个 seed / 同一套并行规模
   唯一的变量就是「奖励从哪来」。
"""
import os
import re
import shutil

CFG = (
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/"
    "deepmimic/config/g1/g1_deepmimic_env_cfg.py"
)
DATA_DIR = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run"

assert os.path.isfile(CFG), f"找不到配置: {CFG}"

# 备份：改源码前先留后路，跑完好还原
if not os.path.exists(CFG + ".bak"):
    shutil.copy(CFG, CFG + ".bak")
    print(f"[INFO] 已备份 -> {CFG}.bak")

# weights 的 key 必须是**不带后缀的 clip 名**（和 AMP 配置里的写法一致）
names = sorted(
    f[:-4] for f in os.listdir(DATA_DIR) if f.endswith(".pkl")
)
assert names, f"{DATA_DIR} 里没有 pkl"
print(f"[INFO] 共 {len(names)} 段动作")

src = open(CFG, encoding="utf-8").read()

# --- 1) motion_data_dir 指到 amp/walk_and_run --------------------------------
new_dir = (
    'self.motion_data.motion_dataset.motion_data_dir = os.path.join(\n'
    '            LEGGED_LAB_ROOT_DIR, "data", "MotionData", "g1_29dof", "amp", "walk_and_run"\n'
    '        )'
)
src, n1 = re.subn(
    r'self\.motion_data\.motion_dataset\.motion_data_dir\s*=\s*os\.path\.join\([^)]*\)',
    new_dir, src, flags=re.S,
)

# --- 2) weights 换成全部 30 段 ----------------------------------------------
new_w = (
    "self.motion_data.motion_dataset.motion_data_weights = {\n"
    + "".join(f'            "{n}": 1.0,\n' for n in names)
    + "        }"
)
src, n2 = re.subn(
    r'self\.motion_data\.motion_dataset\.motion_data_weights\s*=\s*\{[^}]*\}',
    new_w, src, flags=re.S,
)

open(CFG, "w", encoding="utf-8").write(src)
print(f"[INFO] motion_data_dir  替换 {n1} 处")
print(f"[INFO] motion_data_weights 替换 {n2} 处")

if n1 != 1 or n2 != 1:
    print("!! 替换次数异常，请人工检查该文件")
    raise SystemExit(1)

# --- 自检：把改完的结果打出来 ----------------------------------------------
print("\n===== 改后自检 =====")
for line in open(CFG, encoding="utf-8"):
    if "motion_data_dir" in line or "walk_and_run" in line:
        print("   ", line.rstrip())
print(f"    weights 条目数 = {len(names)}")
print("=== PATCH DONE ===")
