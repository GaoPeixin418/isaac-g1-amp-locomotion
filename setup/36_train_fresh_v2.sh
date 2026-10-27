#!/usr/bin/env bash
# =============================================================================
# 从零训 v2：根治「跛行 + 碎步蹲姿 + 转弯晃」（2026-10-27 晚）
#
# 与 34_train_fresh_sym.sh（v1）的区别
#   1. 对称项换成 leg_work_asymmetry：|tau*omega| 的周期级 EMA，tc 0.4->1.5 s，
#      权重 -1.0 -> -2.0（v1 实测后期回升，权重压不住）
#   2. 新增 base_height_bias：躯干高度靠向参考 0.765 m（治蹲，绕不过去）
#   3. 新增 knee_bias：膝角工作点靠向参考 15°（治蹲的姿态侧）
#   4. ~~ang_vel_xy_l2 -0.05 -> -0.2~~ **已回退（v2.1）**：实测该项 -0.14，
#      是 base_height(-0.005)/knee(-0.05) 的几十倍，且它罚躯干俯仰+横滚角速度，
#      而迈大步必然带来更大躯干摆动 -> 它在**加剧碎步**，与治步态尺度方向相反。
#   5. ~~feet_air_time threshold 0.4 -> 0.5~~ **已回退（v2.2）**：未验证的
#      多变量改动，先减变量；碎步若复发再单独上这个旋钮
#   6. 配 watch_v2.py 自动早停：连续 3 个评估点综合分不改善就杀掉训练，
#      不再"训到底再看"（v1 实测 2400->2999 对称性反而变差）
#   7. **v2.2**：work_asym 权重 -2.0 -> -1.0。v2/v2.1 两轮到 iter 800/1200
#      都没学会走路（v_act 0.05-0.07，v1 同量级 -1.0 在 iter800 已 0.97）。
#      ang_vel 已被 v2.1 回退排除 -> 主嫌疑是 -2.0 早期压制探索；且站着不动
#      |tau*omega|≈0 零惩罚，最优解成了"站着别动"。回到 v1 量级，保留
#      tc=1.5 + |tau*omega| 两个防 exploit 修复。
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

# --- 闸 3：v2.2 —— ang_vel 回退在位 + 对称权重 -1.0 + airtime 回 0.4 ---
grep -q "ang_vel_xy_l2, weight=-0.05" "$ENV_CFG" || fail "ang_vel_xy_l2 没回退到 -0.05（v2.1）"
grep -q "leg_torque_asymmetry" "$ENV_CFG" && fail "仍有 v1 的 leg_torque_asymmetry"
python - <<'PY' 2>/dev/null || fail "leg_work_asymmetry weight 不是 -1.0（先跑 patch_sym_v22.py）"
import sys
c = open("/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py", encoding="utf-8").read()
sys.exit(0 if "func=mdp.leg_work_asymmetry,\n        weight=-1.0," in c else 1)
PY
python - <<'PY' 2>/dev/null || fail "feet_air_time threshold 不是 0.4（先跑 patch_sym_v22.py）"
import sys
p = "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
c = open(p, encoding="utf-8").read()
i = c.find("feet_air_time = RewTerm(")
seg = c[i:c.find("    )", i)]
sys.exit(0 if '"threshold": 0.4' in seg else 1)
PY

# --- 闸 4：λ 必须是 0.5（10-22 事故：残留 1.0 导致三个 run 全是纯 PPO） ---
LERP_NOW=$(grep -o "task_style_lerp=[0-9.]*" "$AGENT_CFG" | head -1 | cut -d= -f2)
[ "$(awk -v v="$LERP_NOW" 'BEGIN{printf "%g", v}')" = "0.5" ] || fail "task_style_lerp=$LERP_NOW 不是 0.5"

echo "[$(date '+%m-%d %H:%M:%S')] TASK=$TASK NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER SEED=$SEED"
# angvel 打印实际值（v2.1 回退到 -0.05 后，写死的 "-0.2" 会误导人）
ANGVEL_NOW=$(grep -o "ang_vel_xy_l2, weight=[-0-9.]*" "$ENV_CFG" | head -1 | grep -o "[-0-9.]*$")
WORKW_NOW=$(python -c "
c=open('$ENV_CFG',encoding='utf-8').read()
i=c.find('leg_work_asymmetry = RewTerm(')
j=c.find('weight=',i)
print(c[j+7:c.find(',',j)].strip())" 2>/dev/null || echo "?")
AIRT_NOW=$(python -c "
c=open('$ENV_CFG',encoding='utf-8').read(); i=c.find('feet_air_time = RewTerm(')
print(c[i:c.find('    )',i)].split('\"threshold\":')[1].split(',')[0].strip())" 2>/dev/null || echo "?")
echo "[$(date '+%m-%d %H:%M:%S')] lerp=$LERP_NOW  work_sym=ON(w$WORKW_NOW,tc1.5)  height=ON  knee=ON  angvel=$ANGVEL_NOW  air_threshold=$AIRT_NOW  mode=FRESH"

mkdir -p /data/logs
yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --seed "$SEED" 2>&1

echo "=== FRESH V2 TRAIN DONE $(date '+%m-%d %H:%M:%S') ==="
