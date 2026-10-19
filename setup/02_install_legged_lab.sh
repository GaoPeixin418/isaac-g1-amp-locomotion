#!/usr/bin/env bash
# =============================================================================
# 02 - 安装 legged_lab (AMP for Unitree G1) + rsl_rl AMP fork  【实机适配版 v2】
# 机器: jq1.9gpu.com | RTX 3090 24G | env 在 /data/envs/isaaclab
#
# v2 相对初版的修改（与 01 踩坑一致）：
#   1) 不用 conda activate（脚本里失败是静默的）→ 全程绝对路径 python/pip + CONDA_PREFIX
#   2) WORK 默认 /root/autodl-tmp → /data（这台机器的数据盘）
#   3) 加 PIP_CONSTRAINT 锁 torch==2.7.0+cu128（防依赖解析升级到 2.14 破坏 isaacsim 绑定）
#   4) git lfs 路径按实际仓库结构探测，不写死
#
# 为什么选 legged_lab 做主线：
#   它的 AMP 是 **velocity-conditioned** 的（源码有 |cmd_xy| < 0.06 门控），
#   能按速度指令在走/跑之间切换 —— 面试官要的"拟人态走跑"。
#   humanoid_amp 是纯 style imitation（无速度指令），只能模仿固定动作 → 保底。
#
# 预计耗时：20-40 分钟（git lfs pull 拉 G1 USD + motion data 占大头）
# =============================================================================
set -uo pipefail

WORK=/data
PY=/data/envs/isaaclab/bin/python
PIP=/data/envs/isaaclab/bin/pip

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export PATH=/data/envs/isaaclab/bin:$PATH
export PIP_CONSTRAINT=/data/constraints.txt   # torch==2.7.0+cu128 锁
export PIP_CACHE_DIR=/data/pip_cache
export HF_HOME=/data/hf_cache
export OMNI_KIT_ACCEPT_EULA=YES

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { log "!! $*"; exit 1; }

# ---------------------------------------------------------------------------
log "Step 1/5: git-lfs"
git lfs version >/dev/null 2>&1 || apt-get install -y -qq git-lfs >/dev/null 2>&1 || log "(git-lfs 装不上，Step 3 会失败)"

# ---------------------------------------------------------------------------
log "Step 2/5: rsl_rl fork (feature/amp —— AMP 依赖它，会覆盖 PyPI 的 rsl-rl-lib)"
if [ ! -d "$WORK/rsl_rl" ]; then
    git clone -b feature/amp https://github.com/zitongbai/rsl_rl.git "$WORK/rsl_rl" 2>&1 | tail -3 \
        || git clone -b feature/amp_3 https://github.com/zitongbai/rsl_rl.git "$WORK/rsl_rl" 2>&1 | tail -3 \
        || die "rsl_rl clone 失败"
fi
cd "$WORK/rsl_rl" || die "cd rsl_rl 失败"
git log -1 --oneline
"$PIP" install -e . || die "rsl_rl 安装失败"
"$PY" -c "import rsl_rl; print('rsl_rl from', rsl_rl.__file__)" || die "rsl_rl import 失败"

# ---------------------------------------------------------------------------
log "Step 3/5: legged_lab clone"
if [ ! -d "$WORK/legged_lab" ]; then
    git clone https://github.com/zitongbai/legged_lab "$WORK/legged_lab" 2>&1 | tail -3 || die "legged_lab clone 失败"
fi
cd "$WORK/legged_lab" || die "cd legged_lab 失败"
git log -1 --oneline

# ---------------------------------------------------------------------------
log "Step 4/5: git lfs pull (G1 USD + 已 retarget 的 motion data)"
git lfs install
git lfs pull 2>&1 | tail -10 || log "!! lfs pull 部分失败，检查 data 目录"
echo "--- data tree ---"
find . -maxdepth 5 -type d \( -name "data" -o -name "MotionData" -o -name "*g1*" \) 2>/dev/null | head -15
du -sh . 2>/dev/null

# ---------------------------------------------------------------------------
log "Step 5/5: pip install -e legged_lab + 任务注册检查"
"$PIP" install -e source/legged_lab || die "legged_lab 安装失败"

"$PY" -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__; print('torch', torch.__version__, 'LOCKED OK')" || die "torch 版本被动了！"

"$PY" <<'PYEOF'
import gymnasium as gym
import legged_lab  # noqa: F401  触发任务注册
ids = sorted(i for i in gym.registry.keys() if "AMP" in i)
print("AMP tasks:", ids[:20])
g1 = [i for i in ids if "G1" in i or "g1" in i]
print("AMP G1 tasks:", g1)
if not ids:
    print("!! 没有注册任何 AMP 任务 -- 版本兼容性问题")
PYEOF

log "=== LEGGED_LAB OK ==="
