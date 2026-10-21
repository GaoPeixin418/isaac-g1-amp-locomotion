#!/usr/bin/env bash
# =============================================================================
# 16 - 训练 DeepMimic 对照组
#
# 动作数据由谁负责：
#   不是本脚本，是 17_patch_deepmimic_cfg.py。
#
#   为什么不能像我最初设想的那样用 hydra 覆盖 —— 实测踩过：
#     Could not override 'env.motion_data.motion_dataset.motion_data_weights'
#     Key '...' is not in struct
#   原因是 motion_data_weights 是 dict，hydra 只允许覆盖**已经存在**的 key，
#   把整批 clip 换成另一批名字会直接报错退出进程（白等一次 Kit 启动）。
#   所以改成：源码层 patch（带 .bak 备份）+ 本脚本只管训练参数。
#
# 用法：
#   SMOKE=1 bash 16_train_deepmimic.sh         # 冒烟：64 envs x 50 iter
#   MAX_ITER=3000 bash 16_train_deepmimic.sh   # 正式训练
# =============================================================================
set -u

CONDA_PREFIX=/data/envs/isaaclab
PYTHON=/data/envs/isaaclab/bin/python
export PYTHON
export CONDA_PREFIX
export OMNI_KIT_ACCEPT_EULA=YES
export PIP_CACHE_DIR=/data/pip_cache
export PATH="$CONDA_PREFIX/bin:$PATH"

LAB=/data/legged_lab
cd "$LAB" || exit 1

TASK="${TASK:-LeggedLab-Isaac--Deepmimic-G1-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-3000}"

# 冒烟测试：确认链路通、checkpoint 能落盘
if [ "${SMOKE:-0}" = "1" ]; then
    NUM_ENVS=64
    MAX_ITER=50
    echo "[INFO] SMOKE 模式：64 envs x 50 iter"
fi

# 顺手确认一下数据在，避免又跑到一半才发现
DM_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/deepmimic/config/g1/g1_deepmimic_env_cfg.py"
if [ -f "$DM_CFG" ]; then
    n=$(grep -c '": 1.0,"' "$DM_CFG")
    echo "[INFO] 动作数据条目数 = $n（应由 17_patch_deepmimic_cfg.py 写入）"
    grep -n 'walk_and_run' "$DM_CFG" | head -2
fi

echo ""
echo "[$(date '+%m-%d %H:%M:%S')] DeepMimic 开训"
echo "  TASK     = $TASK"
echo "  NUM_ENVS = $NUM_ENVS"
echo "  MAX_ITER = $MAX_ITER"
echo ""

yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" 2>&1

echo "=== TRAIN DEEPMIMIC DONE $(date '+%m-%d %H:%M:%S') ==="
