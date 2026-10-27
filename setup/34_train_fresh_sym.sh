#!/usr/bin/env bash
# =============================================================================
# 从零训：治跛行（2026-10-27）
#
# 与之前所有 run 的区别（三处，都是新变量）：
#   1. 起点：从头训，不 resume 任何旧权重
#      —— finetune 实测无效（36.7% -> 37.1%），且会继承已固化的出力偏差
#   2. 新奖励：leg_torque_asymmetry，直接罚左右腿"出力"不对称（不是时长）
#   3. 早停：不训到 5000，3000 iter 为止，按扭矩 AI 曲线取最好的 ckpt
#      —— 同一主 run 实测：3200it 16.7% / 4000it 35.3% / 4999it 36.7%，越训越跛
#
# 用法: EVAL_TIMEOUT=... bash 34_train_fresh_sym.sh
#       NUM_ENVS / MAX_ITER / SEED 可环境变量覆盖
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

TASK="${TASK:-LeggedLab-Isaac-AMP-G1-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-3000}"
SEED="${SEED:-42}"
LOG="${LOG:-/data/logs/train_fresh_sym.log}"

ENV_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
AGENT_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py"

# ---------------------------------------------------------------------------
# 闸 1：扭矩对称补丁必须在位（否则这一轮白跑 3 小时）
# ---------------------------------------------------------------------------
if ! grep -q "leg_torque_asymmetry" "$ENV_CFG"; then
    echo "!! 扭矩对称奖励项不在位，拒绝启动。先跑 patch_torque_sym.py"
    exit 1
fi

# ---------------------------------------------------------------------------
# 闸 2：λ 必须是 0.5（10-22 事故：残留 1.0 导致三个 run 全是纯 PPO）
# ---------------------------------------------------------------------------
LERP_NOW=$(grep -o "task_style_lerp=[0-9.]*" "$AGENT_CFG" | head -1 | cut -d= -f2)
if [ "$(awk -v v="$LERP_NOW" 'BEGIN{printf "%g", v}')" != "0.5" ]; then
    echo "!! task_style_lerp=$LERP_NOW，不是 0.5，拒绝启动"
    exit 1
fi

# ---------------------------------------------------------------------------
# 闸 3：不能在旧 run 目录里续训 —— 确认这次不带 --resume（脚本写死）
# ---------------------------------------------------------------------------
echo "[$(date '+%m-%d %H:%M:%S')] TASK=$TASK NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER SEED=$SEED"
echo "[$(date '+%m-%d %H:%M:%S')] lerp=$LERP_NOW  torque_sym=ON  mode=FRESH"

mkdir -p /data/logs
yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --seed "$SEED" 2>&1

echo "=== FRESH SYM TRAIN DONE $(date '+%m-%d %H:%M:%S') ==="
