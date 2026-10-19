#!/usr/bin/env bash
# =============================================================================
# 03 - 降级/保底方案：humanoid_amp（pip 安装，最省事）
#
# 什么时候用：
#   1) legged_lab 与 Isaac Lab 2.3.2 有 API 冲突，调不动
#   2) git lfs 死活拉不出数据
#   3) 时间不够，先保一个"能看的 G1 拟人步态视频"
#
# 局限（提前说清楚）：
#   它的 AMP 是 **纯 style imitation**（task_reward_weight=0），没有速度指令，
#   训出来是"模仿一个固定的走/跑片段"，**不能按命令切换走跑**。
#   所以它是保底，不是主线。
# =============================================================================
set -euo pipefail

WORK="${WORK:-/root/autodl-tmp}"
ENV_NAME="env_isaaclab"
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${ENV_NAME}"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "Step 1/3: clone humanoid_amp"
if [ ! -d "${WORK}/humanoid_amp" ]; then
    git clone https://github.com/linden713/humanoid_amp "${WORK}/humanoid_amp"
fi
cd "${WORK}/humanoid_amp"
git log -1 --oneline

log "Step 2/3: pip install -e . (skrl 已由 01 脚本装好)"
pip install -e .

log "Step 3/3: sync skrl train/play scripts"
bash ./sync_skrl_scripts.sh || true

python - <<'PY'
import gymnasium as gym
import humanoid_amp  # noqa: F401
ids = [i for i in gym.registry.keys() if "G1-AMP" in i]
print("humanoid_amp tasks:", ids)
PY

log "=== HUMANOID_AMP OK ==="
log "train: python -m humanoid_amp.train --task Isaac-G1-AMP-Walk-Direct-v0 --headless --num_envs 4096"
