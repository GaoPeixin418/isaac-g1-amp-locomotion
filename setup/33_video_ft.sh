#!/usr/bin/env bash
# =============================================================================
# 33 - finetune（对称性修复）后的验收录制
#
# 前提：08_train_resume.sh MODE=resume 把 2026-10-19_16-37-21 续训到 6500 iter
# 产出：
#   ft_walk10.mp4   1.0 m/s 定速 —— 跛行是否消失的核心验证（对比 walk10_follow）
#   ft_omni3.mp4    画方形全向 —— 只用"前进+边走边转"（达成率 82%），
#                   删除侧移/后退/原地转向（diag_omni 实测三者全失效）
#
# 用法：
#   nohup env EVAL_TIMEOUT=3600 bash /data/setup/33_video_ft.sh \
#         > /data/logs/video_ft.log 2>&1 &
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
CKPT=${CKPT:-model_6499.pt}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

if [ ! -f "$LOGDIR/$RUN/$CKPT" ]; then
    log "!! 缺权重 $LOGDIR/$RUN/$CKPT（finetune 未完成？）"; exit 1
fi

"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令补丁失败"; exit 1; }
"$PY" /data/setup/patch_play_env.py || { log "!! PLAY 配置补丁失败"; exit 1; }

rec() {
    local out=$1 mode=$2 len=$3 script=$4 pvel=$5 rate=$6
    if [ -f "$OUT/$out" ]; then log "== $out 已存在，跳过"; return 0; fi
    rm -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $out | ckpt=$CKPT mode=$mode len=$len"
    PIN_MODE="$mode" OMNI_SCRIPT="$script" PIN_VEL="$pvel" RAMP_RATE="$rate" \
        TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
        RUN="$RUN" CKPT="$CKPT" NUM_ENVS=1 VIDEO_LEN="$len" \
        bash /data/setup/07_mid_eval.sh
    if cp -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4" "$OUT/$out" 2>/dev/null; then
        log "saved $OUT/$out"
    else
        log "!! 未生成 $out"
    fi
}

log "===== [1/2] ft_walk10：1.0 m/s 定速 12 s（跛行验收） ====="
rec ft_walk10.mp4 ramp 600 "12,1.0,0,0" "1.0" "1.0"

# 画方形：4 条边，每边 6 s 前进 1.0 m/s，段首 +90°（行走中转，达成 82%）
# 净转向 360°，总时长 28 s。无原地/侧移/后退段。
SQUARE="2,0,0,0; 8,1.0,0,1.5708; 14,1.0,0,1.5708; 20,1.0,0,1.5708; 26,1.0,0,1.5708; 28,0,0,0"
log "===== [2/2] ft_omni3：画方形全向 28 s（1400 步） ====="
rec ft_omni3.mp4 omni 1400 "$SQUARE" "1.0" "1.0"

log "===== 收尾 ====="
ls -la "$OUT"/ft_walk10.mp4 "$OUT"/ft_omni3.mp4 2>/dev/null
echo "=== VIDEO FT DONE ==="
