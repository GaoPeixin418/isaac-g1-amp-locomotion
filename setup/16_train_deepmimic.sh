#!/usr/bin/env bash
# =============================================================================
# 16 - 训练 DeepMimic 对照组
#
# 为什么用 hydra override 而不是改源码：
#   08_train_resume.sh 当年是直接 sed 改 legged_lab 源码的默认值（踩过脏）。
#   DeepMimic 这轮只改两个字段 —— 动作数据目录和采样权重 —— 而 rsl_rl
#   train.py 本来就用 hydra 组装 env cfg，命令行覆盖即可，连源码都不用碰。
#   好处：跑完不留残留，换数据不用 revert。
#
# 用法：
#   MAX_ITER=3000 bash 16_train_deepmimic.sh                 # 正式训练
#   SMOKE=1 bash 16_train_deepmimic.sh                       # 冒烟（50 iter）
#   DM_DIR=... DM_WEIGHTS='{name: 1.0}' bash 16_train_deepmimic.sh
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

MOTION_ROOT="${MOTION_ROOT:-$LAB/source/legged_lab/legged_lab/data/MotionData}"
DM_DIR="${DM_DIR:-$MOTION_ROOT/g1_29dof/deepmimic}"
DM_WEIGHTS="${DM_WEIGHTS:-{G5_-__back_kick_stageii: 1.0}}"

# 冒烟测试：确认链路通、checkpoint 能落盘，几分钟出结论
if [ "${SMOKE:-0}" = "1" ]; then
    NUM_ENVS=64
    MAX_ITER=50
    echo "[INFO] SMOKE 模式：64 envs x 50 iter"
fi

echo "[$(date '+%m-%d %H:%M:%S')] DeepMimic 开训"
echo "  TASK        = $TASK"
echo "  NUM_ENVS    = $NUM_ENVS"
echo "  MAX_ITER    = $MAX_ITER"
echo "  动作数据目录 = $DM_DIR"
echo "  采样权重    = $DM_WEIGHTS"
echo ""

yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    env.motion_data.motion_dataset.motion_data_dir="$DM_DIR" \
    env.motion_data.motion_dataset.motion_data_weights="$DM_WEIGHTS" 2>&1

echo "=== TRAIN DEEPMIMIC DONE $(date '+%m-%d %H:%M:%S') ==="
