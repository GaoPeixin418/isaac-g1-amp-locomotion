#!/usr/bin/env bash
# =============================================================================
# 20 - 固定指令视频（成片用）
#
# 为什么还要重录：
#   之前所有视频都是 play.py 原版录的，而它不固定指令。于是画面里实际发生的是
#     * resampling_time_range=[10,10]  -> 每 10 秒随机换一次速度
#     * lin_vel_x ∈ [-0.5, 3.0]        -> 一会儿慢走一会儿冲刺
#     * rel_heading_envs=1.0           -> 100% env 跟随机航向，一直在转弯
#     * rel_standing_envs=0.02         -> 有概率突然站着不动
#   观感就是"前几步不稳、后面走得急、路线乱"—— 是录制方式的问题，不是策略缺陷。
#   （固定指令下的稳态振荡实测 CV 仅 0.07~0.25，中高速档在人类量级内）
#
# 本脚本产出两段真正的成片素材：
#   1) 6 档并排：0.5 / 1.0 / 1.5 / 2.0 / 2.5 / 3.0 m/s，走直线
#   2) 单机匀速：1.2 m/s（中间值，用来证明"任意速度"不只是 6 个离散档）
#
# 前置：scripts/patch_play_cmd.py 需已放到 /data/setup/
#
# 用法：
#   nohup bash /data/setup/20_video_pinned.sh > /data/logs/video_pinned.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

PY=/data/envs/isaaclab/bin/python
PLAY=/data/legged_lab/scripts/rsl_rl/play.py
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
AMP_RUN=2026-10-19_16-37-21
CKPT=${CKPT:-model_4000.pt}
VIDEO_LEN=${VIDEO_LEN:-700}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

# ---- 0) 打补丁（幂等，已打过会跳过）--------------------------------------
log "===== [0/3] 给 play.py 打固定指令补丁 ====="
"$PY" /data/setup/patch_play_cmd.py "$PLAY" || {
    log "!! 打补丁失败，中止（避免录出随机指令的视频）"
    exit 1
}

# 每次录制前先清掉旧视频，避免误把上一次的成片当成这次的
rm -f "$LOGDIR/$AMP_RUN/videos/play/rl-video-step-0.mp4"

# ---- 1) 6 档并排，走直线 --------------------------------------------------
log "===== [1/3] 6 档并排 0.5→3.0 m/s (num_envs=6) ====="
PIN_VEL="0.5,1.0,1.5,2.0,2.5,3.0" \
    RUN=$AMP_RUN CKPT=$CKPT NUM_ENVS=6 VIDEO_LEN=$VIDEO_LEN \
    bash /data/setup/07_mid_eval.sh
cp -f "$LOGDIR/$AMP_RUN/videos/play/rl-video-step-0.mp4" \
      "$OUT/amp_4000_ladder6.mp4" 2>/dev/null \
  && log "saved amp_4000_ladder6.mp4" \
  || log "!! 6 档视频未生成"

# ---- 2) 单机匀速 1.2 m/s（中间值）----------------------------------------
log "===== [2/3] 单机匀速 1.2 m/s (num_envs=1) ====="
rm -f "$LOGDIR/$AMP_RUN/videos/play/rl-video-step-0.mp4"
PIN_VEL="1.2" \
    RUN=$AMP_RUN CKPT=$CKPT NUM_ENVS=1 VIDEO_LEN=$VIDEO_LEN \
    bash /data/setup/07_mid_eval.sh
cp -f "$LOGDIR/$AMP_RUN/videos/play/rl-video-step-0.mp4" \
      "$OUT/amp_4000_steady12.mp4" 2>/dev/null \
  && log "saved amp_4000_steady12.mp4" \
  || log "!! 单机视频未生成"

# ---- 3) 收尾 --------------------------------------------------------------
log "===== [3/3] 收尾 ====="
ls -la "$OUT"/amp_4000_*.mp4 2>/dev/null
log "提示：补丁保留在 play.py 里；想恢复原版用 "
log "      cp $PLAY.precmd $PLAY"
echo "=== VIDEO PINNED DONE ==="
