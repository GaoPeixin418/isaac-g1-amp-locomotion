#!/usr/bin/env bash
# =============================================================================
# 21 - 带步态对称性奖励的 AMP 重训
#
# 背景（2026-10-22 诊断）：
#   model_4000 的左右腿对称性指数 SI = 7.7%~27.7%（人类 <5%），
#   且 6 档里有 5 档是「左脚抬升 < 右脚」，方向一致 -> 系统性偏置。
#   根因：15 个奖励项里没有任何一项约束左右对称。
#   框架自带的 mirror loss 只约束单帧镜像等变性，抓不到周期内的不对称。
#
# 本脚本做两件事：
#   1. 打奖励补丁（air_time_variance_penalty + feet_clearance）
#   2. 训练。默认 2000 iter 先验证方向，再决定是否续训
#
# 两阶段用法：
#   bash 21_train_sym.sh                          # 阶段1: 2000 iter 验证
#   RESUME=1 CKPT=model_2000.pt MAX_ITER=6000 \
#       bash 21_train_sym.sh                      # 阶段2: 续训到 6000
#
# 其他参数：
#   NUM_ENVS=4096  MAX_ITER=2000  SEED=42  NO_PATCH=1(跳过打补丁)
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
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-2000}
SEED=${SEED:-42}
RESUME=${RESUME:-0}
CKPT=${CKPT:-}
# 续训时必须显式指定 run 目录。
# 用 .* 会命中「最近修改的」run，一旦别的操作碰过目录就会加载错权重。
LOAD_RUN=${LOAD_RUN:-2026-10-22_15-52-15}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

CFG=/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py

# ---- 1) 打奖励补丁 --------------------------------------------------------
if [ "${NO_PATCH:-0}" != "1" ]; then
    log "===== [1/2] 打步态对称性奖励补丁 ====="
    "$PY" /data/setup/patch_amp_rewards.py "$CFG" || {
        log "!! 补丁失败，中止"
        exit 1
    }
else
    log "===== [1/2] 跳过打补丁（NO_PATCH=1）====="
fi

# ---- 验证补丁确实在 ------------------------------------------------------
if ! grep -q "air_time_variance" "$CFG"; then
    log "!! 配置里没有 air_time_variance，训练会白跑，中止"
    exit 1
fi

# ---- 2) 训练 --------------------------------------------------------------
log "===== [2/2] 训练 ====="
log "  TASK=$TASK  NUM_ENVS=$NUM_ENVS  MAX_ITER=$MAX_ITER  SEED=$SEED"

cd /data/legged_lab || exit 1

EXTRA_ARGS=""
if [ "$RESUME" = "1" ]; then
    if [ -z "$CKPT" ]; then
        log "!! RESUME=1 但没给 CKPT，中止"
        exit 1
    fi
    EXTRA_ARGS="--resume --load_run $LOAD_RUN --checkpoint $CKPT"
    log "  续训: run=$LOAD_RUN  CKPT=$CKPT  -> 目标 MAX_ITER=$MAX_ITER"
    if [ ! -f "/data/legged_lab/logs/rsl_rl/g1_amp/$LOAD_RUN/$CKPT" ]; then
        log "!! 权重不存在: /data/legged_lab/logs/rsl_rl/g1_amp/$LOAD_RUN/$CKPT，中止"
        exit 1
    fi
fi

# shellcheck disable=SC2086
yes Yes | "$PY" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --seed "$SEED" \
    $EXTRA_ARGS 2>&1

log "=== TRAIN SYM DONE (exit=$?) ==="
log "最新 run 目录："
ls -td /data/legged_lab/logs/rsl_rl/g1_amp/*/ | head -3
