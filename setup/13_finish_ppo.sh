#!/usr/bin/env bash
# =============================================================================
# 13 - 收尾链：等 λ=1.0 纯任务 PPO 基线训完 -> 评估 -> 补录最终模型视频
#
# 为什么要这条链：
#   chain2 只负责启动 PPO 基线训练，跑完没人接手。而这条链要在训练结束的
#   那一刻立刻接上，否则 GPU 就空转了。等的时间不计成本，但空转是纯浪费。
#
# 视频覆盖坑（重要）：
#   play.py 的 video_folder 是固定的 <run>/videos/play，同名文件会被覆盖。
#   上次录 model_2000 就把 16-37-21 里昨天的终版视频盖掉了。
#   所以本脚本严格按「先抢救旧视频 -> 再录新的 -> 录完立刻另存」的顺序做。
#
# 用法：
#   nohup bash /data/setup/13_finish_ppo.sh > /data/logs/finish_ppo.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
mkdir -p "$OUT"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- 0) 先抢救已录好的两段视频，避免后面被覆盖 ----
log "===== [0/5] 抢救已有视频 ====="
if [ -f "$LOGDIR/2026-10-19_16-37-21/videos/play/rl-video-step-0.mp4" ]; then
    cp -f "$LOGDIR/2026-10-19_16-37-21/videos/play/rl-video-step-0.mp4" \
          "$OUT/amp_model_2000.mp4"
    log "saved amp_model_2000.mp4"
fi
if [ -f "$LOGDIR/2026-10-20_18-02-56/videos/play/rl-video-step-0.mp4" ]; then
    cp -f "$LOGDIR/2026-10-20_18-02-56/videos/play/rl-video-step-0.mp4" \
          "$OUT/style_only_lerp0.mp4"
    log "saved style_only_lerp0.mp4"
fi

# ---- 1) 等 PPO 基线训练落 DONE ----
log "===== [1/5] 等待 PPO 基线训练完成 ====="
for i in $(seq 1 200); do
    if grep -q "TRAIN RESUME DONE" /data/logs/train_ppo_baseline.log 2>/dev/null; then
        log "PPO 基线训练已完成"
        break
    fi
    sleep 60
done

# PPO run 目录是脚本自动建的，名字带时间戳 —— 取最新的那个
PPO_RUN=$(ls -1 "$LOGDIR" | grep -E '^2026-10-20_' | sort | tail -1)
CK_PPO=$(ls -1 "$LOGDIR/$PPO_RUN" | grep -E '^model_[0-9]+\.pt$' \
         | sort -t_ -k2 -n | tail -1)
log "PPO run=$PPO_RUN ckpt=$CK_PPO"

# ---- 2) 评估 PPO 基线步态 ----
log "===== [2/5] eval PPO 基线 (LERP=1.0) ====="
RUN=$PPO_RUN CKPT=$CK_PPO TAG=ppo_baseline bash /data/setup/10_eval_gait.sh

# ---- 3) 补录 AMP model_4000 视频（阶梯里拟人度最优档）----
log "===== [3/5] video AMP model_4000 ====="
AMP_RUN=2026-10-19_16-37-21
RUN=$AMP_RUN CKPT=model_4000.pt NUM_ENVS=4 VIDEO_LEN=700 \
    bash /data/setup/07_mid_eval.sh
cp -f "$LOGDIR/$AMP_RUN/videos/play/rl-video-step-0.mp4" \
      "$OUT/amp_model_4000.mp4" 2>/dev/null && log "saved amp_model_4000.mp4"

# ---- 4) 录 PPO 基线视频 ----
log "===== [4/5] video PPO 基线 ====="
RUN=$PPO_RUN CKPT=$CK_PPO NUM_ENVS=4 VIDEO_LEN=700 \
    bash /data/setup/07_mid_eval.sh
cp -f "$LOGDIR/$PPO_RUN/videos/play/rl-video-step-0.mp4" \
      "$OUT/ppo_baseline.mp4" 2>/dev/null && log "saved ppo_baseline.mp4"

# ---- 5) 收尾清单 ----
log "===== [5/5] 收尾 ====="
ls -la "$OUT/"
echo "=== FINISH PPO DONE $(date '+%m-%d %H:%M:%S') ==="
ls -la /data/eval_out/*.npz
