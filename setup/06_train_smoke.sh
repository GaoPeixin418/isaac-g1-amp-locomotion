#!/usr/bin/env bash
# =============================================================================
# 06 - AMP 训练体检（2 个迭代，只验证链路通不通，不追求收敛）
#
# 为什么要用真实训练而不是 demo 脚本做体检：
#   create_empty.py 那类 demo 会卡在 Kit 冷启动（实测 73 分钟无输出），
#   而真正要交的是训练能跑。用真实任务做体检 = 一次动作同时完成"验证"和"推进"。
#
# 任务：LeggedLab-Isaac-AMP-G1-v0
#   奖励 = track_lin_vel_xy_exp(速度跟踪) + AMP style reward
#   → velocity-conditioned AMP：能按速度指令在走/跑之间切换
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export PIP_CONSTRAINT=/data/constraints.txt
export PIP_CACHE_DIR=/data/pip_cache
export HF_HOME=/data/hf_cache
export OMNI_KIT_ACCEPT_EULA=YES

PY=/data/envs/isaaclab/bin/python
TASK=${TASK:-LeggedLab-Isaac-AMP-G1-v0}
NUM_ENVS=${NUM_ENVS:-64}
MAX_ITER=${MAX_ITER:-2}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

cd /data/legged_lab || exit 1
log "task=$TASK num_envs=$NUM_ENVS max_iterations=$MAX_ITER"

yes Yes | "$PY" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --seed 42 2>&1

log "=== TRAIN SMOKE DONE (exit=$?) ==="
