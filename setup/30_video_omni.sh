#!/usr/bin/env bash
# =============================================================================
# 30 - 单环境全向行走成片 + 「指令斜坡」A/B 验证
#
# 为什么这两件事放一个脚本
# -----------------------------------------------------------------------------
# 面试官的反馈：(1) 多环境看不出细节，改用单环境；(2) 步幅不对、一顿一顿。
# 同时我们自己发现所有视频开头都"猛地向前栽一下"。
#
# 本脚本一次给出两块证据：
#   A. 全向行走（前进 / 转向 / 侧移 / 后退 / 原地转）单机器人成片
#      —— 这是面试官点名要的镜头
#   B. pin vs ramp 的 A/B（各 10 s，同一模型同一速度）
#      —— 验证"前栽是阶跃指令逼出来的，斜坡能治"这个机制判断
#
# 指令脚本（OMNI_SCRIPT，段末累计秒数, vx, vy, d_heading[rad]）
# -----------------------------------------------------------------------------
#   2,  0,    0,    0        起步前站定 2 s（斜坡从 0 升速，不会前栽）
#   8,  1.0,  0,    0        前进
#   13, 1.0,  0,    1.5708   边走边左转 90°
#   18, 1.0,  0,    0        继续前进
#   22, 0,   -0.4,  0        向右侧移
#   26, -0.4, 0,    0        后退
#   30, 0,    0,   -1.5708   原地右转 90°
#   36, 0,    0,    0        停
#
# 用法
# -----------------------------------------------------------------------------
#   nohup env EVAL_TIMEOUT=3600 bash /data/setup/30_video_omni.sh \
#         > /data/logs/video_omni.log 2>&1 &
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
CKPT=${CKPT:-model_4000.pt}          # 主模型：训得最久、风格分 0.721
RAMP_RATE=${RAMP_RATE:-1.0}          # 线速度变化率上限 m/s^2

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

# ---- 0) 两个补丁必须在位 ---------------------------------------------------
log "===== [0/3] 校验补丁 ====="
"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令补丁失败，中止"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }

rec() {  # $1=输出名 $2=PIN_MODE $3=步数 $4=OMNI_SCRIPT
    local out=$1 mode=$2 len=$3 script=$4
    if [ -f "$OUT/$out" ]; then
        log "== $out 已存在，跳过（断点续录）"; return 0
    fi
    if [ ! -f "$LOGDIR/$RUN/$CKPT" ]; then
        log "!! 缺权重 $LOGDIR/$RUN/$CKPT"; return 1
    fi
    rm -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $out | mode=$mode len=$len"
    PIN_MODE="$mode" OMNI_SCRIPT="$script" PIN_VEL="${PIN_VEL:-1.2}" \
    RAMP_RATE="$RAMP_RATE" \
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

log "===== [1/3] 单环境全向行走（1800 步 = 36 s） ====="
rec amp4000_omni1.mp4 omni 1800 "$OMNI"

log "===== [2/3] A/B：pin（阶跃，预期前栽） ====="
rec ab_pin12.mp4 pin 500 "10,1.2,0,0"

log "===== [3/3] A/B：ramp（斜坡，预期不栽） ====="
rec ab_ramp12.mp4 ramp 500 "10,1.2,0,0"

log "===== 收尾 ====="
ls -la "$OUT"/amp4000_omni1.mp4 "$OUT"/ab_*.mp4 2>/dev/null
echo "=== VIDEO OMNI DONE ==="
