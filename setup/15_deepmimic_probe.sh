#!/bin/bash
# 15 - DeepMimic 兼容性验证
# =============================================================================
# 为什么先验证再训练：
#   DeepMimic 比 AMP 需要的字段多得多 —— 它要跟踪根位置 (root_pos)、根朝向
#   (root_rot)、末端 body 位置 (key_body_pos)，而 AMP 的判别器刻意只看 4 帧
#   关节状态，甚至刻意**不含**根的世界坐标。
#
#   如果 walk_and_run 这批数据在转换时没保留 root_pos，DeepMimic 根本跑不起来。
#   一次完整训练要 2 小时，花 1 分钟先查清楚，值。
#
# 要回答的四个问题：
#   1) MotionData 实际结构？clip 是目录还是文件？（决定 hydra override 怎么写）
#   2) AMP 到底配了哪些 clip？（决定 DeepMimic 该挑哪几段做对照）
#   3) 这些 pkl 里有没有 DeepMimic 必需的字段？（决定性）
#   4) 真实注册的任务 ID 是什么？
# =============================================================================
# conda activate 在这种非交互 shell 里不保证生效，直接用解释器绝对路径最稳
PY=/data/envs/isaaclab/bin/python
cd /data/legged_lab

BASE=/data/legged_lab/source/legged_lab/legged_lab/data/MotionData

echo "===== [1/4] MotionData 结构与 clip 名单 ====="
$PY - <<'PY'
import os

base = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData"
print("顶层结构：")
for root, dirs, files in os.walk(base):
    depth = root.replace(base, "").count(os.sep)
    if depth > 3:
        continue
    pkls = [f for f in files if f.endswith(".pkl")]
    indent = "  " * depth
    print(f"{indent}{os.path.basename(root)}/   <- {len(pkls)} pkl")
    if pkls:
        print(f"{indent}   e.g. {pkls[:4]}")

def listing(sub, limit=50):
    d = os.path.join(base, sub)
    print(f"\n--- {sub}")
    if not os.path.isdir(d):
        print("    !! 目录不存在")
        return
    items = sorted(os.listdir(d))
    print(f"    共 {len(items)} 项")
    for it in items[:limit]:
        p = os.path.join(d, it)
        if os.path.isdir(p):
            inner = [f for f in os.listdir(p) if f.endswith(".pkl")]
            print(f"      [DIR ] {it}   ({len(inner)} pkl)")
        else:
            print(f"      [FILE] {it}")
    if len(items) > limit:
        print(f"      ... 还有 {len(items) - limit} 项")

listing("g1_29dof/amp/walk_and_run")
listing("g1_29dof/deepmimic")
PY

echo ""
echo "===== [2/4] AMP 的完整 motion_data 配置 ====="
CFG=/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py
sed -n '118,170p' "$CFG" 2>/dev/null

echo ""
echo "===== [3/4] pkl 字段检查（决定性） ====="
$PY - <<'PY'
import glob, os, pickle
import numpy as np

needed = ["fps", "root_pos", "root_rot", "dof_pos", "loop_mode", "key_body_pos"]

def probe_one(path, verbose=True):
    with open(path, "rb") as fh:
        d = pickle.load(fh)
    if not isinstance(d, dict):
        return None, f"顶层不是 dict，是 {type(d)}"
    missing = [k for k in needed if k not in d]
    if verbose:
        print(f"    {os.path.basename(path)}")
        for k, v in d.items():
            try:
                shp = np.asarray(v).shape
            except Exception:
                shp = f"scalar={v}"
            mark = "" if k in needed else "  (AMP 额外字段)"
            print(f"        {k:16s} {shp}{mark}")
    return missing, None

def walk(patterns, label):
    files = sorted(glob.glob(patterns))
    print(f"\n--- {label}: 匹配到 {len(files)} 个 pkl")
    if not files:
        print("    !! 没有文件，glob 模式可能不对")
        return False
    all_ok = True
    # 每个 clip 目录抽查一个
    seen_dirs = set()
    for f in files:
        dn = os.path.dirname(f)
        if dn in seen_dirs:
            continue
        seen_dirs.add(dn)
        try:
            missing, err = probe_one(f)
        except Exception as exc:
            print(f"    !! 加载失败 {f}: {exc}")
            all_ok = False
            continue
        if err:
            print(f"    !! {f}: {err}")
            all_ok = False
        elif missing:
            print(f"        !! DeepMimic 必需但缺失: {missing}")
            all_ok = False
        else:
            print("        OK 全部必需字段齐备")
        if len(seen_dirs) >= 3:
            break
    return all_ok

base = "/data/legged_lab/source/legged_lab/legged_lab/data/MotionData"
ok1 = walk(base + "/g1_29dof/amp/walk_and_run/*/*.pkl", "AMP walk_and_run（子目录形式）")
if not ok1:
    walk(base + "/g1_29dof/amp/walk_and_run/*.pkl", "AMP walk_and_run（平铺形式）")
walk(base + "/g1_29dof/deepmimic/*/*.pkl", "DeepMimic 自带参考数据")

print()
print("VERDICT:", ("walk_and_run 可直接用于 DeepMimic" if ok1
      else "walk_and_run 平铺探测未见有效文件，需进一步确认结构或改用 deepmimic 自带数据"))
PY

echo ""
echo "===== [4/4] 任务注册名核对 ====="
$PY - <<'PY'
import re
for p in [
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/g1/__init__.py",
    "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/__init__.py",
]:
    try:
        ids = re.findall(r'id="([^"]+)"', open(p).read())
        print(f"{p.split('/')[-3]}:")
        for i in ids:
            print("   ", i)
    except Exception as exc:
        print("读取失败:", exc)
PY

echo ""
echo "=== PROBE DONE ==="
