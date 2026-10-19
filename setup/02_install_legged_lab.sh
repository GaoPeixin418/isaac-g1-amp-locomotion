#!/usr/bin/env bash
# =============================================================================
# 02 - 安装 legged_lab (AMP for Unitree G1) + rsl_rl AMP fork
# 前置：01_install_env.sh 已跑完，且 conda activate env_isaaclab
#
# 为什么选 legged_lab 做主线：
#   它的 AMP 是 **velocity-conditioned** 的（源码里有 |cmd_xy| < 0.06 的门控），
#   也就是能按速度指令在走/跑之间切换 —— 这正是面试官要的"拟人态走跑"。
#   而 humanoid_amp 是纯 style imitation（无速度指令），只能模仿固定动作。
#
# 预计耗时：30-50 分钟（git lfs pull 拉 G1 USD + motion data 占大头）
# =============================================================================
set -euo pipefail

WORK="${WORK:-/root/autodl-tmp}"
ENV_NAME="env_isaaclab"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
# 1. rsl_rl fork (feature/amp 分支) —— legged_lab 的 AMP 依赖它
# ---------------------------------------------------------------------------
log "Step 1/4: rsl_rl fork (feature/amp)"
if [ ! -d "${WORK}/rsl_rl" ]; then
    git clone -b feature/amp https://github.com/zitongbai/rsl_rl.git "${WORK}/rsl_rl" \
        || git clone -b feature/amp_3 https://github.com/zitongbai/rsl_rl.git "${WORK}/rsl_rl"
fi
cd "${WORK}/rsl_rl"
git log -1 --oneline
pip install -e .
python -c "import rsl_rl; print('rsl_rl from', rsl_rl.__file__)"

# ---------------------------------------------------------------------------
# 2. legged_lab
# ---------------------------------------------------------------------------
log "Step 2/4: cloning legged_lab"
if [ ! -d "${WORK}/legged_lab" ]; then
    git clone https://github.com/zitongbai/legged_lab "${WORK}/legged_lab"
fi
cd "${WORK}/legged_lab"
git log -1 --oneline

# ---------------------------------------------------------------------------
# 3. git lfs pull —— 拉 G1 USD 模型 + 已 retarget 好的 motion data
#    （大坑：国内服务器 GitHub LFS 经常超时。拉不动就用 03 的降级方案）
# ---------------------------------------------------------------------------
log "Step 3/4: git lfs pull (G1 USD + motion data, may take a while)"
git lfs install
git lfs pull --include="source/legged_lab/legged_lab/data/**" || {
    log "!! LFS pull failed/partial. Fallback: download motion data from HF mirror."
    log "!! run: python data/download_lafan1.py"
}

ls -lh source/legged_lab/legged_lab/data/ 2>/dev/null || true
ls -lh source/legged_lab/legged_lab/data/MotionData/ 2>/dev/null || true

# ---------------------------------------------------------------------------
# 4. 安装 + 冒烟
# ---------------------------------------------------------------------------
log "Step 4/4: pip install -e source/legged_lab"
pip install -e source/legged_lab

python - <<'PY'
import gymnasium as gym
import legged_lab  # noqa: F401  (触发任务注册)
ids = [i for i in gym.registry.keys() if "AMP" in i and "G1" in i]
print("AMP G1 tasks found:", ids)
if not ids:
    print("!! no AMP-G1 task registered -- check version compatibility")
PY

log "=== LEGGED_LAB OK ==="
log "next: 数据体检 -> python scripts/verify_motion.py"
