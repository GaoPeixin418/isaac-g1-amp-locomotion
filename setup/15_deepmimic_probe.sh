#!/bin/bash
# 15 - DeepMimic 兼容性验证
# =============================================================================
# 为什么先验证再训练：
#   DeepMimic 比 AMP 需要的字段多得多 —— 它要跟踪根位置 (root_pos)、根朝向
#   (root_rot)、末端 body 位置 (key_body_pos)，而 AMP 的判别器刻意只看 4 帧
#   关节状态，甚至刻意**不含**根的世界坐标。
#
#   如果 walk_and_run 这批数据在转换时没保留 root_pos，DeepMimic 根本跑不起来。
#   一次完整训练要 2 小时，先在енности 1 分钟把这事儿查清楚，值。
#
# 要回答的三个问题：
#   1) 服务器上到底有哪些 MotionData？AMP 和 DeepMimic 各自指向哪个目录？
#   2) walk_and_run 的 pkl 里有没有 DeepMimic 必需的字段？
#   3) DeepMimic 自带的参考数据长什么样（用来对照格式）？
# =============================================================================
source /data/envs/isaaclab/bin/activate 2>/dev/null || true
cd /data/legged_lab

echo "===== [1/4] MotionData 目录树 ====="
python - <<'PY'
import os
base = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData"
if not os.path.isdir(base):
    print("MISSING:", base)
else:
    for root, dirs, files in os.walk(base):
        depth = root.replace(base, "").count(os.sep)
        if depth > 3:
            continue
        pkls = [f for f in files if f.endswith(".pkl")]
        indent = "  " * depth
        print(f"{indent}{os.path.basename(root)}/   <- {len(pkls)} pkl")
        if pkls and depth <= 3:
            print(f"{indent}   e.g. {pkls[:3]}")
PY

echo ""
echo "===== [2/4] AMP 的 MotionData 配置原文 ====="
CFG=/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py
grep -n -A 12 "motion_data" "$CFG" 2>/dev/null | head -40

echo ""
echo "===== [3/4] walk_and_run 数据字段检查（决定性） ====="
python - <<'PY'
import glob, pickle
import numpy as np

def probe(path_pattern, label, limit=2):
    files = sorted(glob.glob(path_pattern))
    print(f"\n--- {label}: {len(files)} files ---")
    if not files:
        print("  !! 没有找到文件")
        return None
    need = ["fps", "root_pos", "root_rot", "dof_pos", "loop_mode", "key_body_pos"]
    ok = True
    for f in files[:limit]:
        try:
            with open(f, "rb") as fh:
                d = pickle.load(fh)
        except Exception as exc:
            print(f"  !! 加载失败 {f}: {exc}")
            ok = False
            continue
        print(f"  {os.path.basename(f)}")
        if isinstance(d, dict):
            for k, v in d.items():
                try:
                    shape = np.asarray(v).shape
                except Exception:
                    shape = f"scalar={v}"
                print(f"      {k:16s} {shape}")
            missing = [k for k in need if k not in d]
            if missing:
                print(f"      !! DeepMimic 必需但缺失: {missing}")
                ok = False
            else:
                print("      OK 全部必需字段齐备")
        else:
            print(f"      !! 顶层不是 dict，是 {type(d)}")
            ok = False
    return ok

import os
walk = probe(
    "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/amp/walk_and_run/*.pkl",
    "AMP 用的 walk_and_run",
)
probe(
    "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData/g1_29dof/deepmimic/*/*.pkl",
    "DeepMimic 自带参考数据",
)

print()
if walk:
    print("VERDICT: walk_and_run 可直接用于 DeepMimic")
else:
    print("VERDICT: walk_and_run 缺字段，需要改用 deepmimic 自带数据或重新转换")
PY

echo ""
echo "===== [4/4] 任务注册名核对 ====="
python - <<'PY'
import re
p = "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/g1/__init__.py"
try:
    txt = open(p).read()
    ids = re.findall(r'id="([^"]+)"', txt)
    print("注册的任务 ID：")
    for i in ids:
        print("   ", i)
    print()
    print("注意双连字符是真实存在还是笔误，以这里为准。")
except Exception as exc:
    print("读取失败:", exc)
PY

echo ""
echo "=== PROBE DONE ==="
