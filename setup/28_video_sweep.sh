#!/usr/bin/env bash
# =============================================================================
# 28 - λ 三臂同条件对比成片
#
# 为什么必须录这一段
# -----------------------------------------------------------------------------
# λ=0.75 在稳态速度误差 / SI_clear / 人参照带偏差上都赢了 0.5，
# **但风格分只有 0.390，比 0.5 的 0.721 低了 46%**。
#
# 而最终 demo 是给人看的。**"判别器觉得像不像" ≠ "人看着自然不自然"** ——
# 这两个是两把不同的尺子：
#   - 风格分 = 判别器输出，它只在 LAFAN1 的人类动作分布上有定义，
#     遇到 2.5/3.0 m/s 这种数据里没有的速度必然扣分
#   - SI_clear / 速度误差 / CoT = 客观物理量，不依赖任何参考数据
#
# 所以到底用哪个 λ，不能只靠表格决定 —— 得把三个并排放出来，
# 让人看到"0.75 是不是走得太机械"。
#
# 只录单速 1.2 m/s 的原因
# -----------------------------------------------------------------------------
# 实测 10 env 单速只要 77 秒，而 6 env 跑 6 档速度要 7.5 分钟以上还没完
# （相机跟随范围大 → 渲染贵）。这段时间优先换先看得到的东西。
#
# 产出
# -----------------------------------------------------------------------------
#   lam025_multi10_12.mp4 / lam050_multi10_12.mp4 / lam075_multi10_12.mp4
#
# 用法
# -----------------------------------------------------------------------------
#   nohup env EVAL_TIMEOUT=3600 bash /data/setup/28_video_sweep.sh \
#         > /data/logs/video_sweep.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
VIDEO_LEN=${VIDEO_LEN:-700}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

# ---- 0) 两个补丁必须在位 ---------------------------------------------------
log "===== [0/4] 校验补丁 ====="
"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令固定补丁失败，中止"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }

record() {  # $1=run $2=tag
    local run=$1 tag=$2
    local ckpt=model_1499.pt
    if [ ! -f "$LOGDIR/$run/$ckpt" ]; then
        log "!! 缺权重 $run/$ckpt，跳过"; return 1
    fi
    rm -f "$LOGDIR/$run/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $tag | run=$run 10 env × 1.2 m/s"
    PIN_VEL="1.2" \
        TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
        RUN="$run" CKPT="$ckpt" NUM_ENVS=10 VIDEO_LEN="$VIDEO_LEN" \
        bash /data/setup/07_mid_eval.sh
    if cp -f "$LOGDIR/$run/videos/play/rl-video-step-0.mp4" \
              "$OUT/${tag}_multi10_12.mp4" 2>/dev/null; then
        log "saved $OUT/${tag}_multi10_12.mp4"
    else
        log "!! 未生成 ${tag}（继续下一段）"
    fi
}

log "===== [1/4] λ=0.25 ====="
record 2026-10-22_21-19-29 lam025
log "===== [2/4] λ=0.50 ====="
record 2026-10-22_20-08-48 lam050
log "===== [3/4] λ=0.75 ====="
record 2026-10-22_22-30-36 lam075

log "===== [4/4] 收尾 ====="
ls -la "$OUT"/lam0*_multi10_12.mp4 2>/dev/null
echo "=== VIDEO SWEEP DONE ==="
