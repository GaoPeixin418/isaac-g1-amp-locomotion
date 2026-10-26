#!/usr/bin/env bash
# =============================================================================
# 29 - 「干净 + 随机指令」成片：证明指令跟随能力
#
# 为什么这段是面试主卖点
# -----------------------------------------------------------------------------
# 钉死指令的视频（amp4000_clean_steady12 等）只能证明"这个速度走得稳"，
# 证明不了"听得懂指令"。而旧配置的随机指令版会被误读成"不稳"——
# 因为晃的来源是 PLAY 配置里的推搡+噪声，不是策略跟不上。
#
# 本脚本录的才是正确形态：
#   PIN_MODE=rand  → 每 10s 随机换速度（0.5~3.0 m/s 前进档）
#   patch_play_env → 无推搡、无外力、无观测噪声、无随机航向
#   rel_standing   → 0（不会有环境突然站着不动）
# 效果：10 个机器人各走各的速度、走直线，到点整齐地加减速 ——
# 这正是"对指令的跟随能力"的直接证据。
#
# 录制长度 1400 步 = 28s（默认 700 步只有 14s，只会碰上 1 次换速，
# 看不出跟随）。28s 里会发生 2~3 次切换，足够看出"换指令时稳不稳"。
#
# 产出
# -----------------------------------------------------------------------------
#   amp4000_rand10.mp4     旧主模型（λ=0.5, iter4000, 无对称奖励）
#   lam075_rand10.mp4      λ=0.75 臂（iter1500, 有对称奖励）
# 后者用来初步回答"λ=0.75 在随机指令下会不会因为风格分低而显得机械"。
#
# 用法
# -----------------------------------------------------------------------------
#   nohup env EVAL_TIMEOUT=3600 bash /data/setup/29_video_rand.sh \
#         > /data/logs/video_rand.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
VIDEO_LEN=${VIDEO_LEN:-1400}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

# ---- 0) 两个补丁必须在位 ---------------------------------------------------
log "===== [0/3] 校验补丁 ====="
"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令补丁失败，中止"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }

record() {  # $1=run $2=ckpt $3=outname
    local run=$1 ckpt=$2 out=$3
    if [ -f "$OUT/$out" ]; then
        log "== $out 已存在，跳过（断点续录）"; return 0
    fi
    if [ ! -f "$LOGDIR/$run/$ckpt" ]; then
        log "!! 缺权重 $run/$ckpt，跳过"; return 1
    fi
    rm -f "$LOGDIR/$run/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $out | run=$run ckpt=$ckpt 10 env × rand"
    PIN_MODE=rand \
        TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
        RUN="$run" CKPT="$ckpt" NUM_ENVS=10 VIDEO_LEN="$VIDEO_LEN" \
        bash /data/setup/07_mid_eval.sh
    if cp -f "$LOGDIR/$run/videos/play/rl-video-step-0.mp4" \
              "$OUT/$out" 2>/dev/null; then
        log "saved $OUT/$out"
    else
        log "!! 未生成 $out（继续下一段）"
    fi
}

log "===== [1/3] amp4000 随机指令 ====="
record 2026-10-19_16-37-21 model_4000.pt amp4000_rand10.mp4

log "===== [2/3] λ=0.75 随机指令 ====="
record 2026-10-22_22-30-36 model_1499.pt lam075_rand10.mp4

log "===== [3/3] 收尾 ====="
ls -la "$OUT"/*rand10*.mp4 2>/dev/null
echo "=== VIDEO RAND DONE ==="
