#!/usr/bin/env bash
# =============================================================================
# 14 - 单环境版本视频（成片主镜头用）
#
# 为什么还要录一版 1 env 的：
#   默认 4 envs 版本里，4 个机器人并排站在同一块场地（env_spacing=2.5m），
#   且各自的速度指令是随机采样的 —— 画面里同时有在走的、在跑的、在转向的，
#   观感上会放大"不稳"的印象，评审看不出重点。
#   num_envs=1 版本画面里只有一个机器人，是干净的成片素材。
#
# 用法：
#   nohup bash /data/setup/14_single_video.sh > /data/logs/single_video.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
AMP_RUN=2026-10-19_16-37-21
PPO_RUN=$(ls -1 "$LOGDIR" | grep -E '^2026-10-20_20-' | sort | tail -1)

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "===== [1/2] 单环境录制 AMP model_4000 ====="
cd /data/legged_lab
RUN=$AMP_RUN CKPT=model_4000.pt NUM_ENVS=1 VIDEO_LEN=700 \
    bash /data/setup/07_mid_eval.sh
cp -f "$LOGDIR/$AMP_RUN/videos/play/rl-video-step-0.mp4" \
      "$OUT/amp_4000_single.mp4" 2>/dev/null && log "saved amp_4000_single.mp4"

log "===== [2/2] 单环境录制 PPO 基线 ====="
CK_PPO=$(ls -1 "$LOGDIR/$PPO_RUN" | grep -E '^model_[0-9]+\.pt$' \
         | sort -t_ -k2 -n | tail -1)
log "ppo run=$PPO_RUN ckpt=$CK_PPO"
RUN=$PPO_RUN CKPT=$CK_PPO NUM_ENVS=1 VIDEO_LEN=700 \
    bash /data/setup/07_mid_eval.sh
cp -f "$LOGDIR/$PPO_RUN/videos/play/rl-video-step-0.mp4" \
      "$OUT/ppo_baseline_single.mp4" 2>/dev/null && log "saved ppo_baseline_single.mp4"

log "===== SINGLE VIDEO DONE ====="
echo "=== SINGLE VIDEO DONE $(date '+%m-%d %H:%M:%S') ==="
ls -la "$OUT/"
