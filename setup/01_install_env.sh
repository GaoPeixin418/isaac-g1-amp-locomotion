#!/usr/bin/env bash
# =============================================================================
# 01 - 安装 Isaac Sim 5.1 + Isaac Lab 2.3.2 （9gpu 实机适配版 v3）
# 机器: jq1.9gpu.com | RTX 3090 24G | Ubuntu 22.04 | 系统盘59G(剩26G) + 数据盘200G(/data)
#
# v3 修复记录（v2 踩的坑）：
#   1) conda 25.9 必须先 `conda tos accept`，否则 create 静默失败
#      -> 后果是 activate 失败，后续全跑在系统 Python 3.10 上
#   2) isaacsim 5.x 的 wheel 只有 cp311，Python 3.10 的 pip 只能看到 4.x
#      -> 所以必须用 env 的绝对路径 python/pip，绝不依赖 conda activate
#   3) env 用 -p 装到 /data（系统盘装不下 30G+ 的 isaacsim）
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

PY=$ENV_PREFIX/bin/python
PIP=$ENV_PREFIX/bin/pip

export PIP_CACHE_DIR=$WORK/pip_cache
export HF_HOME=$WORK/hf_cache
mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$WORK/logs"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { log "!! $*"; exit 1; }

# ---------------------------------------------------------------------------
log "Step 0/6: conda TOS + env(py3.11)"
"$CONDA" tos accept --override-channels \
    --channel https://repo.anaconda.com/pkgs/main \
    --channel https://repo.anaconda.com/pkgs/r >/dev/null 2>&1 || true
if [ ! -x "$PY" ]; then
    "$CONDA" create -y -q -p "$ENV_PREFIX" python=3.11 || die "conda create 失败"
fi
"$PY" -V || die "env python 不可用"

# ---------------------------------------------------------------------------
log "Step 1/6: git-lfs"
git lfs version >/dev/null 2>&1 || apt-get install -y -qq git-lfs >/dev/null 2>&1 || log "git-lfs 装不上，02 脚本再兜底"

# ---------------------------------------------------------------------------
log "Step 2/6: torch $TORCH_VER+cu128"
"$PIP" install -q -U pip
"$PIP" install "torch==$TORCH_VER" "torchvision==$TORCHVISION_VER" \
    --index-url https://download.pytorch.org/whl/cu128 || die "torch 安装失败"
"$PY" -c "import torch; print('torch', torch.__version__, '| cuda:', torch.cuda.is_available())" || die "torch import 失败"

# ---------------------------------------------------------------------------
log "Step 3/6: isaacsim[$ISAACSIM_VER] (~20GB, 最慢的一步)"
"$PIP" install "isaacsim[all,extscache]==$ISAACSIM_VER" \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --extra-index-url https://pypi.nvidia.com || die "isaacsim 安装失败"
"$PY" -c "import isaacsim; print('isaacsim OK')" || die "isaacsim import 失败"

# ---------------------------------------------------------------------------
log "Step 4/6: clone IsaacLab @ $ISAACLAB_TAG"
[ -d "$ISAACLAB_DIR" ] || git clone --depth 1 --branch "$ISAACLAB_TAG" \
    https://github.com/isaac-sim/IsaacLab.git "$ISAACLAB_DIR" || die "clone IsaacLab 失败"
cd "$ISAACLAB_DIR" || die "cd 失败"
git log -1 --oneline

# ---------------------------------------------------------------------------
log "Step 5/6: isaaclab.sh --install rsl_rl skrl"
./isaaclab.sh --install rsl_rl skrl || die "isaaclab install 失败"

# ---------------------------------------------------------------------------
log "Step 6/6: headless 冒烟测试"
if [ -f scripts/tutorials/00_sim/create_empty.py ]; then
    timeout 900 "$PY" scripts/tutorials/00_sim/create_empty.py --headless \
        && log "SMOKE TEST PASSED" || die "smoke test FAILED"
fi

log "=== ENV OK ==="
