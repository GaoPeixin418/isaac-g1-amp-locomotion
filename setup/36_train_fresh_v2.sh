#!/usr/bin/env bash
# =============================================================================
# 从零训 v2：根治「跛行 + 碎步蹲姿 + 转弯晃」（2026-10-27 晚）
#
# 与 34_train_fresh_sym.sh（v1）的区别
#   1. 对称项换成 leg_work_asymmetry：|tau*omega| 的周期级 EMA，tc 0.4->1.5 s，
#      权重 -1.0 -> -2.0（v1 实测后期回升，权重压不住）
#   2. 新增 base_height_bias：躯干高度靠向参考 0.765 m（治蹲，绕不过去）
#   3. 新增 knee_bias：膝角工作点靠向参考 15°（治蹲的姿态侧）
#   4. ang_vel_xy_l2 -0.05 -> -0.2（治躯干晃动 / 转弯不稳）
#   5. 配 watch_v2.py 自动早停：连续 3 个评估点综合分不改善就杀掉训练，
#      不再"训到底再看"（v1 实测 2400->2999 对称性反而变差）
#
# 用法: bash 36_train_fresh_v2.sh
#       MAX_ITER / SEED / NUM_ENVS 可环境变量覆盖
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
LOG="${LOG:-/data/logs/train_fresh_v2.log}"

ENV_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
AGENT_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py"

fail() { echo "!! $1"; exit 1; }

# --- 闸 1：v1 的对称项必须已经被替换掉（否则这一轮白跑） ---
grep -q "leg_work_asymmetry" "$ENV_CFG" || fail "leg_work_asymmetry 不在位，先跑 patch_sym_v2.py"
grep -q "leg_torque_asymmetry" "$ENV_CFG" && fail "还在用 v1 的 leg_torque_asymmetry，先跑 patch_sym_v2.py"

# --- 闸 2：治蹲两项必须在位 ---
grep -q "base_height_bias" "$ENV_CFG" || fail "base_height_bias 不在位"
grep -q "knee_bias" "$ENV_CFG" || fail "knee_bias 不在位"

# --- 闸 3：躯干晃动惩罚已加重 ---
grep -q "ang_vel_xy_l2, weight=-0.2" "$ENV_CFG" || fail "ang_vel_xy_l2 不是 -0.2"

# --- 闸 4：λ 必须是 0.5（10-22 事故：残留 1.0 导致三个 run 全是纯 PPO） ---
LERP_NOW=$(grep -o "task_style_lerp=[0-9.]*" "$AGENT_CFG" | head -1 | cut -d= -f2)
[ "$(awk -v v="$LERP_NOW" 'BEGIN{printf "%g", v}')" = "0.5" ] || fail "task_style_lerp=$LERP_NOW 不是 0.5"

echo "[$(date '+%m-%d %H:%M:%S')] TASK=$TASK NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER SEED=$SEED"
echo "[$(date '+%m-%d %H:%M:%S')] lerp=$LERP_NOW  work_sym=ON(w-2.0,tc1.5)  height=ON  knee=ON  angvel=-0.2  mode=FRESH"

mkdir -p /data/logs
yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --seed "$SEED" 2>&1

echo "=== FRESH V2 TRAIN DONE $(date '+%m-%d %H:%M:%S') ==="
