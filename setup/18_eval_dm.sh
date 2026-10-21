#!/usr/bin/env bash
# =============================================================================
# 18 - DeepMimic 量化评估 + 录制
#
# 与 AMP 侧 10_eval_gait.sh 的差异，只有一个原因：
#   DeepMimic 环境没有 command manager（源码核验过），所以不能"钉死速度指令"。
#   评估脚本换成 eval_deepmimic.py，按参考动作自身的瞬时速度记录。
#
# 用法：
#   RUN=2026-10-21_18-44-12 CKPT=2999 bash 18_eval_dm.sh
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

RUN="${RUN:-2026-10-21_18-44-12}"
CKPT="${CKPT:-2999}"
TAG="${TAG:-deepmimic}"
NUM_ENVS="${NUM_ENVS:-6}"
STEPS="${STEPS:-1500}"
TASK="${TASK:-LeggedLab-Isaac--Deepmimic-G1-Play-v0}"

CKPT_PATH="/data/legged_lab/logs/rsl_rl/g1_deepmimic/${RUN}/model_${CKPT}.pt"

echo "[$(date '+%H:%M:%S')] 评估 DeepMimic  RUN=$RUN  CKPT=$CKPT"
ls -la "$CKPT_PATH" || { echo "[ERROR] checkpoint 不存在"; exit 1; }

"$PYTHON" scripts/eval_deepmimic.py \
    --task "$TASK" \
    --checkpoint "$CKPT_PATH" \
    --num_envs "$NUM_ENVS" \
    --steps "$STEPS" \
    --tag "$TAG" \
    --out /data/eval_out \
    --headless 2>&1

echo "=== EVAL DEEPMIMIC WRAPPER DONE $(date '+%H:%M:%S') ==="
