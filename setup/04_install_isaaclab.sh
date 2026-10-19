#!/usr/bin/env bash
# =============================================================================
# 04 - 安装 IsaacLab v2.3.2（接在 01 的 Step 3 之后跑）
#
# 背景：01 脚本在 Step 3 的 `import isaacsim` 处失败了 ——
#   原因是 Omniverse Kit 首次启动会**交互式**问 EULA（Do you accept the EULA? (Yes/No)），
#   非交互环境 stdin 读到 EOF 就挂。解法是用 `yes Yes |` 喂 stdin。
#
# 关键环境技巧：
#   isaaclab.sh 的 extract_python_exe() 优先读 CONDA_PREFIX，
#   所以用 -p 建到 /data 的 env 只要导出 CONDA_PREFIX 就能被正确识别，
#   不需要真的 `conda activate`（activate 在脚本里失败是静默的，见 01 的踩坑记录）。
# =============================================================================
set -uo pipefail

WORK=/data
ENV_PREFIX=/data/envs/isaaclab
ISAACLAB_DIR=$WORK/IsaacLab

# --- 让 isaaclab.sh 找到正确的 python（它优先读 CONDA_PREFIX）---
export CONDA_PREFIX=$ENV_PREFIX
export ISAACLAB_PATH=$ISAACLAB_DIR
export PATH=$ENV_PREFIX/bin:$PATH
export PIP_CACHE_DIR=$WORK/pip_cache
export HF_HOME=$WORK/hf_cache
export OMNI_KIT_ACCEPT_EULA=YES

PY=$ENV_PREFIX/bin/python

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { log "!! $*"; exit 1; }

log "python: $($PY -V 2>&1) at $(command -v python)"

# ---------------------------------------------------------------------------
log "Step 4/6: IsaacLab repo"
[ -d "$ISAACLAB_DIR" ] || die "IsaacLab 未 clone，先跑 Step 4"
cd "$ISAACLAB_DIR" || die "cd 失败"
git log -1 --oneline

# ---------------------------------------------------------------------------
log "Step 5/6: isaaclab.sh --install rsl_rl skrl（EULA 用 yes 管道喂）"
if ! yes Yes | ./isaaclab.sh --install rsl_rl skrl; then
    die "isaaclab install 失败"
fi

log "verify isaaclab import"
if ! yes Yes | timeout 600 "$PY" -c "import isaaclab; print('isaaclab', isaaclab.__version__ if hasattr(isaaclab,'__version__') else 'OK')"; then
    die "isaaclab import 失败"
fi

# ---------------------------------------------------------------------------
log "Step 6/6: headless 冒烟测试"
if [ -f scripts/tutorials/00_sim/create_empty.py ]; then
    if ! yes Yes | timeout 900 "$PY" scripts/tutorials/00_sim/create_empty.py --headless; then
        die "smoke test FAILED"
    fi
    log "SMOKE TEST PASSED"
fi

log "=== ENV OK ==="
