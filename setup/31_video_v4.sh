#!/usr/bin/env bash
# =============================================================================
# 31 - v4 补丁后的补录：相机跟随全向 + 斜坡参数对比 + 定速 1.0/2.0
#
# 对应 2026-10-26 用户反馈的四件事：
#   1. 全向视频 11 s 后机器人出画  -> v4 相机跟随，重录 omni2
#   2. 斜坡后仍有轻微前栽          -> RAMP_RATE=0.5 与已有 1.0 做 A/B
#   3. "固定 2.0 m/s 会不会更好"   -> 录 walk20_follow 实证
#      （数据集探测：|v-2.0|<0.2 只占 5% 帧且全是跑/变速帧，
#        预期不会更好；1.0-1.2 才是覆盖最好的行走速度）
#   4. 候选最佳展示速度            -> 录 walk10_follow（数据集 0.8-1.3 占 32%）
#
# 用法
#   nohup env EVAL_TIMEOUT=3600 bash /data/setup/31_video_v4.sh \
#         > /data/logs/video_v4.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
RUN=${RUN:-2026-10-19_16-37-21}      # λ=0.5 真 AMP 主 run
CKPT=${CKPT:-model_4000.pt}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

log "===== [0/4] 打 v4 补丁（指令控制 + 相机跟随） ====="
"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令补丁失败，中止"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }

rec() {  # $1=输出名 $2=PIN_MODE $3=步数 $4=OMNI_SCRIPT $5=PIN_VEL $6=RAMP_RATE
    local out=$1 mode=$2 len=$3 script=$4 pvel=$5 rate=$6
    if [ -f "$OUT/$out" ]; then
        log "== $out 已存在，跳过（断点续录）"; return 0
    fi
    if [ ! -f "$LOGDIR/$RUN/$CKPT" ]; then
        log "!! 缺权重 $LOGDIR/$RUN/$CKPT"; return 1
    fi
    rm -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $out | mode=$mode len=$len pin=$pvel ramp=$rate"
    PIN_MODE="$mode" OMNI_SCRIPT="$script" PIN_VEL="$pvel" RAMP_RATE="$rate" \
        TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
        RUN="$RUN" CKPT="$CKPT" NUM_ENVS=1 VIDEO_LEN="$len" \
        bash /data/setup/07_mid_eval.sh
    if cp -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4" \
              "$OUT/$out" 2>/dev/null; then
        log "saved $OUT/$out"
    else
        log "!! 未生成 $out"
    fi
}

OMNI="2,0,0,0; 8,1.0,0,0; 13,1.0,0,1.5708; 18,1.0,0,0; 22,0,-0.4,0; 26,-0.4,0,0; 30,0,0,-1.5708; 36,0,0,0"

log "===== [1/4] 全向行走 v2（相机跟随，1800 步 = 36 s） ====="
rec amp4000_omni2.mp4 omni 1800 "$OMNI" "1.0" "1.0"

log "===== [2/4] 斜坡 A/B：RAMP_RATE=0.5（对照已有 ab_ramp12 的 1.0） ====="
rec ab_ramp05_12.mp4 ramp 600 "12,1.2,0,0" "1.2" "0.5"

log "===== [3/4] 定速 1.0 m/s（数据集覆盖最好的行走速度段） ====="
rec walk10_follow.mp4 ramp 600 "12,1.0,0,0" "1.0" "1.0"

log "===== [4/4] 定速 2.0 m/s（验证'高速指令更自然'猜想） ====="
rec walk20_follow.mp4 ramp 600 "12,2.0,0,0" "2.0" "1.0"

log "===== 收尾 ====="
ls -la "$OUT"/amp4000_omni2.mp4 "$OUT"/ab_ramp05_12.mp4 \
       "$OUT"/walk10_follow.mp4 "$OUT"/walk20_follow.mp4 2>/dev/null
echo "=== VIDEO V4 DONE ==="
