#!/usr/bin/env bash
# =============================================================================
# 32 - 前栽 A/B 补录（v5 同相机）：阶跃 vs 斜坡1.0
#
# 为什么重录：上午的 ab_pin12/ab_ramp12 是固定相机（v3）录的，
# 和 follow 相机（v5）的 ab_ramp05_12 视角不同，"前倾程度"没法公平比。
# 本脚本在同相机下补齐对照组，凑齐三向对比：
#   阶跃 vs RAMP_RATE=1.0 vs RAMP_RATE=0.5（回答"斜坡参数还要不要再调"）
#
# 用法：nohup bash /data/setup/32_video_ab_v5.sh > /data/logs/video_ab_v5.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
RUN=${RUN:-2026-10-19_16-37-21}
CKPT=${CKPT:-model_4000.pt}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令补丁失败，中止"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }

rec() {  # $1=输出名 $2=PIN_MODE $3=RAMP_RATE
    local out=$1 mode=$2 rate=$3
    if [ -f "$OUT/$out" ]; then
        log "== $out 已存在，跳过"; return 0
    fi
    rm -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $out | mode=$mode ramp=$rate"
    PIN_MODE="$mode" OMNI_SCRIPT="12,1.2,0,0" PIN_VEL="1.2" RAMP_RATE="$rate" \
        TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
        RUN="$RUN" CKPT="$CKPT" NUM_ENVS=1 VIDEO_LEN=600 \
        bash /data/setup/07_mid_eval.sh
    if cp -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4" "$OUT/$out" 2>/dev/null; then
        log "saved $OUT/$out"
    else
        log "!! 未生成 $out"
    fi
}

log "===== [1/2] 阶跃（对照组，v5 相机） ====="
rec ab_pin12_v5.mp4 pin 1.0

log "===== [2/2] 斜坡 1.0（实验组，v5 相机） ====="
rec ab_ramp12_v5.mp4 ramp 1.0

log "===== 收尾 ====="
ls -la "$OUT"/ab_*_v5.mp4 2>/dev/null
echo "=== VIDEO AB V5 DONE ==="
