#!/usr/bin/env bash
# =============================================================================
# 40 - v2.3 (model_1599) 跛行验收视频：只录 1.0 m/s 定速 12 s 一段
#
# 为什么照抄 35 而不是直接用 07
# -----------------------------------------------------------------------------
# 10-27 的 f2800_walk10.mp4 是用 35_video_check.sh 成功录出来的：
#   ① 先打 patch_play_cmd.py + patch_play_env.py 两个补丁
#   ② TASK 必须用 G1-Play-v0（关扰动 / 固定指令 / 相机一致）
# 11-16 那次卡死 32 min，是直接用 07_mid_eval.sh、NUM_ENVS=1、VIDEO_LEN=700，
# 日志停在 terrain_importer warning —— 缺补丁/缺 Play task 的嫌疑最大。
# 本脚本复用 35 已验证的路径，但只录一段 walk10，并加 15 min 硬超时。
#
# 用法：
#   nohup timeout -s KILL 1000 bash /data/setup/40_video_v23.sh \
#       > /data/logs/video_v23_40.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-900}

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
RUN=${RUN:-2026-11-16_16-49-16}
CKPT=${CKPT:-model_1599.pt}
TAG=${TAG:-v23_1599}
VIDEO_LEN=${VIDEO_LEN:-600}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

if [ ! -f "$LOGDIR/$RUN/$CKPT" ]; then
    log "!! 缺权重 $LOGDIR/$RUN/$CKPT"; exit 1
fi

log "===== [0/2] 校验两个补丁 ====="
"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令固定补丁失败，中止"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }

rm -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4"
log "===== [1/2] 录制 ${TAG}_walk10 | $RUN/$CKPT len=$VIDEO_LEN ====="
PIN_MODE=ramp OMNI_SCRIPT="12,1.0,0,0" PIN_VEL="1.0" RAMP_RATE="1.0" \
    TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
    RUN="$RUN" CKPT="$CKPT" NUM_ENVS=1 VIDEO_LEN="$VIDEO_LEN" \
    bash /data/setup/07_mid_eval.sh

log "===== [2/2] 另存成片 ====="
if cp -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4" "$OUT/${TAG}_walk10.mp4" 2>/dev/null; then
    log "saved $OUT/${TAG}_walk10.mp4"
    ls -lh "$OUT/${TAG}_walk10.mp4"
else
    log "!! 未生成视频（超时或卡死），看 /data/logs/video_v23_40.log"
fi
echo "=== VIDEO 40 DONE ==="
