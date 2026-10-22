#!/usr/bin/env bash
# =============================================================================
# 22 - 干净版成片：固定指令 + 无扰动 + 多环境
#
# 与 20_video_pinned.sh 的区别
# -----------------------------------------------------------------------------
# 20 只做了「钉住指令」，但 G1AmpEnvCfg_PLAY 当时还漏了三条展示模式必关项
# （push_robot / base_external_force_torque / 观测噪声），所以 15:41 录出来的
# amp_4000_ladder6.mp4 里机器人仍每 5 秒被随机推一下 → 观感"晃"。
# 本脚本在「指令固定 + 配置干净」两个补丁都在位的前提下重录。
#
# 为什么主打"固定指令"而不是"随机切换指令"
# -----------------------------------------------------------------------------
# 1. 随机切换会把「对指令的正确响应」误读成「不稳」
# 2. 固定指令下才能横向可比：同一条指令，谁走得直、谁步态对称，一眼可见
# 3. 任意速度的证明不靠随机：6 档并排 + 一个非整数档（1.2 m/s）就够了
#
# 产出
# -----------------------------------------------------------------------------
#   amp4000_clean_ladder6.mp4    6 env，0.5/1.0/1.5/2.0/2.5/3.0 m/s 走直线
#   amp4000_clean_steady12.mp4   1 env，1.2 m/s（非整数档，证明连续可控）
#   sym_clean_ladder3.mp4        3 env，0.5/1.0/1.5 m/s（对称奖励版，步态对比）
#
# 用法：
#   nohup bash /data/setup/22_video_clean.sh > /data/logs/video_clean.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos

OLD_RUN=2026-10-19_16-37-21
NEW_RUN=${NEW_RUN:-2026-10-22_18-54-54}
NEW_CKPT=${NEW_CKPT:-model_2200.pt}
VIDEO_LEN=${VIDEO_LEN:-700}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

# ---- 0) 两个补丁都必须到位，否则拒绝录制 ---------------------------------
log "===== [0/4] 校验补丁 ====="
"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令固定补丁失败，中止（宁可不录，也不录随机指令）"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }

record() {  # $1=run $2=ckpt $3=num_envs $4=PIN_VEL $5=outname
    local run=$1 ckpt=$2 nenv=$3 pin=$4 out=$5
    rm -f "$LOGDIR/$run/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $out | run=$run ckpt=$ckpt envs=$nenv pin=$pin"
    PIN_VEL="$pin" \
        TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
        RUN="$run" CKPT="$ckpt" NUM_ENVS="$nenv" VIDEO_LEN="$VIDEO_LEN" \
        bash /data/setup/07_mid_eval.sh
    if cp -f "$LOGDIR/$run/videos/play/rl-video-step-0.mp4" "$OUT/$out" 2>/dev/null; then
        log "saved $OUT/$out"
    else
        log "!! 未生成 $out"
    fi
}

# ---- 1) 旧模型 6 档并排 ----------------------------------------------------
log "===== [1/4] 旧 model_4000：6 档并排 0.5→3.0 m/s ====="
record "$OLD_RUN" model_4000.pt 6 "0.5,1.0,1.5,2.0,2.5,3.0" amp4000_clean_ladder6.mp4

# ---- 2) 旧模型 单机 1.2 m/s ------------------------------------------------
log "===== [2/4] 旧 model_4000：单机 1.2 m/s ====="
record "$OLD_RUN" model_4000.pt 1 "1.2" amp4000_clean_steady12.mp4

# ---- 3) 新模型（对称奖励）3 档并排 ----------------------------------------
log "===== [3/4] 新 $NEW_CKPT：3 档并排 0.5/1.0/1.5 m/s ====="
record "$NEW_RUN" "$NEW_CKPT" 3 "0.5,1.0,1.5" sym_clean_ladder3.mp4

# ---- 4) 新模型 单机 1.0 m/s ------------------------------------------------
log "===== [4/4] 新 $NEW_CKPT：单机 1.0 m/s ====="
record "$NEW_RUN" "$NEW_CKPT" 1 "1.0" sym_clean_steady10.mp4

log "===== 收尾 ====="
ls -la "$OUT"/*clean*.mp4 2>/dev/null
echo "=== VIDEO CLEAN DONE ==="
