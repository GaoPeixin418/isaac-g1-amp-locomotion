#!/usr/bin/env bash
# =============================================================================
# 10 - 标准化步态评估包装器
#
# 为什么需要包装器：
#   eval_gait.py 产出的 npz 用 checkpoint 名命名（如 gait_model_2000.npz）。
#   三个模型分别来自两个 run 目录，名字能区分，但看不出哪个是 AMP 哪个是基线，
#   后面做对比分析时容易搞混。包装器多接一个 TAG 参数，跑完立即改名。
#
#   另外把 headless / Kit 环境变量 / 超时保护统一在这里，避免每次手敲漏参数。
#
# 用法：
#   RUN=2026-10-19_16-37-21 CKPT=model_2000.pt TAG=amp_2000 \
#       nohup bash /data/setup/10_eval_gait.sh > /data/logs/eval_amp_2000.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

PY=/data/envs/isaaclab/bin/python
TASK=${TASK:-LeggedLab-Isaac-AMP-G1-v0}
RUN=${RUN:?需要 RUN（run 目录名）}
CKPT=${CKPT:?需要 CKPT（如 model_2000.pt）}
TAG=${TAG:-${CKPT%.pt}}
STEPS=${STEPS:-1500}
VELS=${VELS:-0.5,1.0,1.5,2.0,2.5,3.0}
OUT=${OUT:-/data/eval_out}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

CKPT_PATH=/data/legged_lab/logs/rsl_rl/g1_amp/$RUN/$CKPT
if [ ! -f "$CKPT_PATH" ]; then
    log "!! checkpoint 不存在: $CKPT_PATH"
    ls -1 /data/legged_lab/logs/rsl_rl/g1_amp/$RUN/ | tail -8
    exit 1
fi

log "eval: tag=$TAG run=$RUN ckpt=$CKPT steps=$STEPS vels=$VELS"

cd /data/legged_lab || exit 1

timeout -s KILL 2400 yes Yes | "$PY" /data/setup/eval_gait.py \
    --task "$TASK" \
    --headless \
    --checkpoint "$CKPT_PATH" \
    --num_envs 6 \
    --steps "$STEPS" \
    --velocities "$VELS" \
    --out "$OUT" 2>&1

rc=$?
log "eval exit=$rc"

SRC="$OUT/gait_${CKPT%.pt}.npz"
DST="$OUT/gait_${TAG}.npz"
if [ -f "$SRC" ]; then
    mv -f "$SRC" "$DST"
    log "saved -> $DST"
else
    log "!! 未找到产出 $SRC"
    ls -la "$OUT"
fi

echo "=== EVAL DONE $TAG ==="
