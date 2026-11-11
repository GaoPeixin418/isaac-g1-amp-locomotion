#!/usr/bin/env bash
# =============================================================================
# 从零训 v2.3：v2.2 + leg_work_asymmetry 早期 warmup（2026-11-11）
#
# 为什么有这一版（取证结论，见 docs/v2.2结案与v2.3方案.md）
#   v2.2 只跑到 iter 84/2400（5 分 15 秒，00:03 断电），**没有产生任何评估点**
#   => 它是"没跑"，不是"失败"。而 v1 的相变恰好发生在 400→800 之间
#   （v1 iter400 v_act=0.06 -> iter800 v_act=0.97），v2 在 iter800 就被判死刑。
#   所以真正的假设是：约束在 motor primitive 形成期就把探索压死了
#   （教训 3.6：约束必须在形成期施加；教训 3.9：惩罚项零点=站着不动）。
#
# 本脚本相对 36_train_fresh_v2.sh 的改动（只有一处）
#   1. leg_work_asymmetry 的返回值乘 0→1 线性斜坡，前 WARMUP 个控制步爬满
#      -> 前 500 iter 该项权重≈0（约等于 v1 的宽松条件），之后才是 -1.0
#   2. 默认 MAX_ITER 2400 -> 3000（与 v1 同训练量，避免又踩"杀早了"）
#   3. 日志独立 train_fresh_v23.log，不与 v2/v2.1/v2.2 混写
#   4. 新增闸 5：确认 warmup 补丁在位
#
# 两种跑法（同一个源码，靠环境变量切换）
#   bash 37_train_fresh_v23.sh                       # v2.3：warmup 12000 步
#   WARMUP=0 bash 37_train_fresh_v23.sh              # 纯 v2.2 复跑（对照组）
#   WARMUP=6000 bash 37_train_fresh_v23.sh           # 更短的斜坡（250 iter）
#   MAX_ITER=1200 bash 37_train_fresh_v23.sh         # 只看早期能不能起飞
#
# 用法: bash 37_train_fresh_v23.sh
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
WARMUP="${WARMUP:-12000}"
LOG="${LOG:-/data/logs/train_fresh_v23.log}"

export LEG_ASYM_WARMUP_STEPS="$WARMUP"

ENV_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
AGENT_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py"
REWARDS="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"

fail() { echo "!! $1"; exit 1; }

# --- 闸 1：v1 的对称项必须已被替换 ---
grep -q "leg_work_asymmetry" "$ENV_CFG" || fail "leg_work_asymmetry 不在位，先跑 patch_sym_v2.py"
grep -q "leg_torque_asymmetry" "$ENV_CFG" && fail "还在用 v1 的 leg_torque_asymmetry"

# --- 闸 2：治蹲两项在位（实测为 no-op，保留是为了与 v2.2 严格可比）---
grep -q "base_height_bias" "$ENV_CFG" || fail "base_height_bias 不在位"
grep -q "knee_bias" "$ENV_CFG" || fail "knee_bias 不在位"

# --- 闸 3：v2.2 的数值必须在位 ---
grep -q "ang_vel_xy_l2, weight=-0.05" "$ENV_CFG" || fail "ang_vel_xy_l2 没回退到 -0.05"
"$PYTHON" - <<'PY' 2>/dev/null || fail "leg_work_asymmetry weight 不是 -1.0（先跑 patch_sym_v22.py）"
import sys
c = open("/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py", encoding="utf-8").read()
sys.exit(0 if "func=mdp.leg_work_asymmetry,\n        weight=-1.0," in c else 1)
PY
"$PYTHON" - <<'PY' 2>/dev/null || fail "feet_air_time threshold 不是 0.4（先跑 patch_sym_v22.py）"
import sys
p = "/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
c = open(p, encoding="utf-8").read()
i = c.find("feet_air_time = RewTerm(")
seg = c[i:c.find("    )", i)]
sys.exit(0 if '"threshold": 0.4' in seg else 1)
PY

# --- 闸 4：λ 必须是 0.5（10-22 事故：残留 1.0 导致三个 run 全是纯 PPO）---
LERP_NOW=$(grep -o "task_style_lerp=[0-9.]*" "$AGENT_CFG" | head -1 | cut -d= -f2)
[ "$(awk -v v="$LERP_NOW" 'BEGIN{printf "%g", v}')" = "0.5" ] || fail "task_style_lerp=$LERP_NOW 不是 0.5"

# --- 闸 5（新）：warmup 补丁必须在位，且能被环境变量控制 ---
grep -q "PATCHED-SYM-V23-WARMUP" "$REWARDS" || fail "warmup 补丁不在位，先跑 patch_sym_v23.py"
grep -q "_v23_warm_scale(env)" "$REWARDS" || fail "return 没乘 warmup scale（补丁只打了一半）"
"$PYTHON" - <<'PY' 2>/dev/null || fail "warmup 常量没读到环境变量"
import sys
c = open("/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py", encoding="utf-8").read()
sys.exit(0 if "LEG_ASYM_WARMUP_STEPS" in c else 1)
PY

if [ "$WARMUP" = "0" ]; then
  echo "[$(date '+%m-%d %H:%M:%S')] ⚠️  WARMUP=0 -> 这是纯 v2.2 复跑（对照组），不是 v2.3"
fi

echo "[$(date '+%m-%d %H:%M:%S')] TASK=$TASK NUM_ENVS=$NUM_ENVS MAX_ITER=$MAX_ITER SEED=$SEED WARMUP=$WARMUP"
ANGVEL_NOW=$(grep -o "ang_vel_xy_l2, weight=[-0-9.]*" "$ENV_CFG" | head -1 | grep -o "[-0-9.]*$")
WORKW_NOW=$("$PYTHON" -c "
c=open('$ENV_CFG',encoding='utf-8').read()
i=c.find('leg_work_asymmetry = RewTerm(')
j=c.find('weight=',i)
print(c[j+7:c.find(',',j)].strip())" 2>/dev/null || echo "?")
echo "[$(date '+%m-%d %H:%M:%S')] lerp=$LERP_NOW  work_sym=ON(w$WORKW_NOW,tc1.5,warmup=$WARMUP)  angvel=$ANGVEL_NOW  mode=FRESH-v2.3"

mkdir -p /data/logs
yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --max_iterations "$MAX_ITER" \
    --seed "$SEED" 2>&1

echo "=== FRESH V2.3 TRAIN DONE $(date '+%m-%d %H:%M:%S') ==="
