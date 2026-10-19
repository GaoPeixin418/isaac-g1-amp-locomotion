#!/usr/bin/env bash
# =============================================================================
# 05 - 修复 isaaclab 主包安装
#
# 04 脚本的尸检报告：
#   1) isaaclab 主包依赖 flatdict，其 4.0.1 在 py3.11 + 新 setuptools 下构建 wheel 失败
#      -> isaaclab 主包没装上（其余 6 个扩展包都成功了，torch 2.7.0+cu128 完好）
#      解法（Poppy 部署时验证过）：pip install flatdict==4.0.1 --no-build-isolation
#   2) `isaaclab.sh --install rsl_rl skrl` 不接受 skrl 参数（Invalid argument）
#      -> skrl 直接 pip 装；rsl_rl 其实已经装好了
#   3) isaaclab_rl 依赖解析时 pip 想拉 torch 2.14.1+cu13（会破坏 isaacsim 5.1 的 torch 2.7 绑定）
#      -> 用 PIP_CONSTRAINT 锁死 torch 版本，这是标准保险
# =============================================================================
set -uo pipefail

PY=/data/envs/isaaclab/bin/python
PIP=/data/envs/isaaclab/bin/pip
export PIP_CACHE_DIR=/data/pip_cache
export HF_HOME=/data/hf_cache
export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

# 锁死 torch（防止任何依赖解析把它升级到 2.14+cu13）
cat > /data/constraints.txt <<'EOF'
torch==2.7.0+cu128
torchvision==0.22.0+cu128
EOF
export PIP_CONSTRAINT=/data/constraints.txt

# pip 源换腾讯云（阿里云文件下载实测只有 ~540KB/s，腾讯云快 5 倍以上）
mkdir -p /root/.config/pip
cat > /root/.config/pip/pip.conf <<'EOF'
[global]
index-url = https://mirrors.cloud.tencent.com/pypi/simple/
EOF

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
die() { log "!! $*"; exit 1; }

# ---------------------------------------------------------------------------
log "Step 1/5: flatdict==4.0.1 (setuptools 84 移除了 pkg_resources，先降到 80.9)"
"$PIP" install "setuptools==80.9.0" || die "setuptools 降级失败"
"$PIP" install flatdict==4.0.1 --no-build-isolation || die "flatdict 安装失败"

# ---------------------------------------------------------------------------
log "Step 2/5: isaaclab 主包 (editable, 依赖已缓存)"
cd /data/IsaacLab || die "cd 失败"
"$PIP" install -e source/isaaclab || die "isaaclab 主包安装失败"

# ---------------------------------------------------------------------------
log "Step 3/5: skrl (保底框架用, 直接 pip 不走 isaaclab.sh)"
"$PIP" install skrl || die "skrl 安装失败"

# ---------------------------------------------------------------------------
log "Step 4/5: import 三连验证"
"$PY" -c "import torch; assert torch.__version__.startswith('2.7.0'), torch.__version__; print('torch', torch.__version__, 'LOCKED OK')" || die "torch 版本被动了！"
"$PY" -c "import flatdict; print('flatdict OK')" || die "flatdict import 失败"
"$PY" -c "import isaaclab; print('isaaclab OK')" || die "isaaclab import 失败"
"$PY" -c "import skrl; print('skrl OK')" || die "skrl import 失败"
"$PY" -c "import rsl_rl; print('rsl_rl OK')" || die "rsl_rl import 失败"

# ---------------------------------------------------------------------------
log "Step 5/5: headless 冒烟测试"
SMOKE=scripts/tutorials/00_sim/create_empty.py
if [ -f "$SMOKE" ]; then
    yes Yes | timeout 900 "$PY" "$SMOKE" --headless || die "smoke test FAILED"
    log "SMOKE TEST PASSED"
else
    log "(smoke 脚本不存在，跳过——后续用训练脚本直接验证)"
fi

log "=== ENV OK ==="
