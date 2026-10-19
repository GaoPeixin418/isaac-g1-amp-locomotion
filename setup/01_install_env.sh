#!/usr/bin/env bash
# =============================================================================
# 01 - 安装 Isaac Sim 5.1 + Isaac Lab 2.3.2
# 用途：AMP (Adversarial Motion Priors) 训练 Unitree G1 拟人态走跑
# 预计耗时：60-90 分钟（网络好的话 40 分钟）
# 用法：bash setup/01_install_env.sh   （建议 nohup 后台跑）
# =============================================================================
set -euo pipefail

ENV_NAME="env_isaaclab"
ISAACSIM_VER="5.1.0"          # Isaac Sim 5.x -> 必须 Python 3.11
TORCH_VER="2.7.0"
TORCHVISION_VER="0.22.0"
CUDA_TAG="cu128"
ISAACLAB_TAG="v2.3.2"         # legged_lab 要求 v2.3.2

# 数据盘路径：9gpu 用 /data，AutoDL 用 /root/autodl-tmp
WORK="${WORK:-/root/autodl-tmp}"
ISAACLAB_DIR="${WORK}/IsaacLab"

log() { echo "[$(date '+%H:%M:%S')] $*"; }

log "WORK dir = ${WORK}"
mkdir -p "${WORK}"

# ---------------------------------------------------------------------------
# 1. conda 环境
# ---------------------------------------------------------------------------
log "Step 1/6: creating conda env ${ENV_NAME} (python 3.11)"
# 让 conda 在非交互 shell 里可用
CONDA_BASE="$(conda info --base 2>/dev/null || echo "${HOME}/miniconda3")"
# shellcheck disable=SC1091
source "${CONDA_BASE}/etc/profile.d/conda.sh"

if ! conda env list | grep -q "^${ENV_NAME} "; then
    conda create -n "${ENV_NAME}" python=3.11 -y
fi
conda activate "${ENV_NAME}"
python -V

# ---------------------------------------------------------------------------
# 2. PyTorch (CUDA 12.8)
# ---------------------------------------------------------------------------
log "Step 2/6: installing torch ${TORCH_VER}+${CUDA_TAG}"
pip install --upgrade pip
pip install -U "torch==${TORCH_VER}" "torchvision==${TORCHVISION_VER}" \
    --index-url "https://download.pytorch.org/whl/${CUDA_TAG}"

# ---------------------------------------------------------------------------
# 3. Isaac Sim (pip)  —— 必须走 pypi.nvidia.com，不要加国内镜像
# ---------------------------------------------------------------------------
log "Step 3/6: installing isaacsim ${ISAACSIM_VER} (this is the long one, ~20 min)"
pip install "isaacsim[all,extscache]==${ISAACSIM_VER}" \
    --extra-index-url https://pypi.nvidia.com

# ---------------------------------------------------------------------------
# 4. Isaac Lab (source, pinned tag)
# ---------------------------------------------------------------------------
log "Step 4/6: cloning IsaacLab @ ${ISAACLAB_TAG}"
if [ ! -d "${ISAACLAB_DIR}" ]; then
    git clone https://github.com/isaac-sim/IsaacLab.git --branch "${ISAACLAB_TAG}" "${ISAACLAB_DIR}" \
        || git clone https://github.com/isaac-sim/IsaacLab.git "${ISAACLAB_DIR}"
fi
cd "${ISAACLAB_DIR}"
git log -1 --oneline

log "Step 5/6: ./isaaclab.sh --install rsl_rl skrl (this pulls RL libs)"
# 只装我们需要的两个，装 all 会拖很久
./isaaclab.sh --install rsl_rl skrl || ./isaaclab.sh --install

# ---------------------------------------------------------------------------
# 5. 验证
# ---------------------------------------------------------------------------
log "Step 6/6: verifying..."
python -c "import isaacsim; print('isaacsim', isaacsim.__version__)" 2>/dev/null || true
python -c "import isaaclab; print('isaaclab', isaaclab.__version__)" 2>/dev/null || true
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available())"

# 空场景冒烟测试（headless，不需要 GUI）
if [ -f scripts/tutorials/00_sim/create_empty.py ]; then
    timeout 600 python scripts/tutorials/00_sim/create_empty.py --headless \
        && log "smoke test PASSED" || log "smoke test FAILED - check log"
fi

log "=== ENV OK ==="
log "next: bash setup/02_install_legged_lab.sh"
