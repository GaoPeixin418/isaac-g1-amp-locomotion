#!/usr/bin/env bash
# =============================================================================
# 01 - 安装 Isaac Sim 5.1 + Isaac Lab 2.3.2 （9gpu 实机适配版 v2）
# 机器: jq1.9gpu.com | RTX 3090 24G | Ubuntu 22.04 | /data 200G 数据盘
# 关键决策: conda env 用 -p 装到 /data/envs/isaaclab（系统盘只剩 26G，装不下 30G+ 的 isaacsim）
# =============================================================================
set -uo pipefail

CONDA=/miniconda3/bin/conda
ENV_PREFIX=/data/envs/isaaclab
ISAACSIM_VER=5.1.0
TORCH_VER=2.7.0
TORCHVISION_VER=0.22.0
ISAACLAB_TAG=v2.3.2
WORK=/data
ISAACLAB_DIR=$WORK/IsaacLab

export PIP_CACHE_DIR=$WORK/pip_cache
export HF_HOME=$WORK/hf_cache
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$WORK/logs"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---------------------------------------------------------------------------
log "Step 0/6: git-lfs + 基础依赖"
apt-get update -qq >/dev/null 2>&1 || log "apt update 失败(不影响主线)"
apt-get install -y -qq git-lfs >/dev/null 2>&1 || log "apt 装 git-lfs 失败，稍后兜底"
git lfs version || log "!! git-lfs 仍不可用，02 脚本里再处理"

# ---------------------------------------------------------------------------
log "Step 1/6: conda env -> $ENV_PREFIX (python 3.11)"
"$CONDA" create -y -q -p "$ENV_PREFIX" python=3.11
source /miniconda3/etc/profile.d/conda.sh
conda activate "$ENV_PREFIX"
python -V

# ---------------------------------------------------------------------------
log "Step 2/6: torch $TORCH_VER+cu128 (pytorch 官方 index)"
pip install -q -U "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" \
    --index-url https://download.pytorch.org/whl/cu128 || { log "!! torch 安装失败"; exit 1; }
python -c "import torch; print('torch', torch.__version__, '| cuda avail:', torch.cuda.is_available())"

# ---------------------------------------------------------------------------
log "Step 3/6: isaacsim[$ISAACSIM_VER] —— 最大的一步 (~20GB wheels)，走 nvidia index + 清华主源"
pip install "isaacsim[all,extscache]==$ISAACSIM_VER" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.nvidia.com || { log "!! isaacsim 安装失败"; exit 1; }
python -c "import isaacsim; print('isaacsim OK')" || { log "!! isaacsim import 失败"; exit 1; }

# ---------------------------------------------------------------------------
log "Step 4/6: clone IsaacLab @ $ISAACLAB_TAG"
if [ ! -d "$ISAACLAB_DIR" ]; then
    git clone --depth 1 --branch "$ISAACLAB_TAG" https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR" \
        || { log "!! clone IsaacLab 失败"; exit 1; }
fi
cd "$ISAACLAB_DIR"
git log -1 --oneline

# ---------------------------------------------------------------------------
log "Step 5/6: isaaclab.sh --install rsl_rl skrl"
./isaaclab.sh --install rsl_rl skrl || { log "!! isaaclab install 失败"; exit 1; }

# ---------------------------------------------------------------------------
log "Step 6/6: 冒烟测试（headless 空场景，验证 PhysX 能在 3090 上跑）"
if [ -f scripts/tutorials/00_sim/create_empty.py ]; then
    timeout 900 python scripts/tutorials/00_sim/create_empty.py --headless \
        && log "SMOKE TEST PASSED" || { log "!! smoke test FAILED"; exit 1; }
fi

log "=== ENV OK ==="
