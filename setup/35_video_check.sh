#!/usr/bin/env bash
# =============================================================================
# 35 - 通用验收录制（参数化 RUN/CKPT/TAG），用于训练中途抽检
#
# 产出：
#   <TAG>_walk10.mp4   1.0 m/s 定速 12 s —— 跛行验收
#                      （与 walk10_follow / ft_walk10 同指令同相机，可直接 A/B）
#   <TAG>_omni.mp4     画方形全向 27 s —— 转向改成渐进式
#
# 与 33 的区别（全向不稳的两处修法）
# ---------------------------------------------------------------------------
# 1. 朝向目标不再阶跃。33 里每段起手 heading 直接跳 +90°，闭环会以最大角速度
#    硬拧，看起来就是"一直在找平衡"。这里改成 3 s 内分 4 小步递进，
#    等效转向速率 30°/s（人类常速转弯约 45-90°/s，这里取偏慢的一档求稳）。
# 2. 每条边转向结束后留 3 s 纯直走，让姿态收敛再进入下一次转向。
#
# 用法：
#   nohup env RUN=2026-10-27_17-53-43 CKPT=model_1200.pt TAG=f1200 \
#         bash /data/setup/35_video_check.sh > /data/logs/video_check.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos
RUN=${RUN:?需要 RUN=2026-10-27_xx-xx-xx}
CKPT=${CKPT:?需要 CKPT=model_xxxx.pt}
TAG=${TAG:?需要 TAG=输出文件名前缀}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

if [ ! -f "$LOGDIR/$RUN/$CKPT" ]; then
    log "!! 缺权重 $LOGDIR/$RUN/$CKPT"; exit 1
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

log "===== [1/2] ${TAG}_walk10：1.0 m/s 定速 12 s ====="
rec "${TAG}_walk10.mp4" ramp 600 "12,1.0,0,0" "1.0" "1.0"

# 画方形，转向渐进：每条边 3 s 转 90°（4 小步）+ 3 s 直走
# heading 目标 = 0 -> 90° -> 180° -> 270° -> 360°，净转向 360°
P=1.5708   # 90°
SQUARE="2,0,0,0; \
3,1.0,0,0; 3.75,1.0,0,0.3927; 4.5,1.0,0,0.7854; 5.25,1.0,0,1.1781; 6,1.0,0,1.5708; \
9,1.0,0,1.5708; \
9.75,1.0,0,1.9635; 10.5,1.0,0,2.3562; 11.25,1.0,0,2.7489; 12,1.0,0,3.1416; \
15,1.0,0,3.1416; \
15.75,1.0,0,3.5343; 16.5,1.0,0,3.9270; 17.25,1.0,0,4.3197; 18,1.0,0,4.7124; \
21,1.0,0,4.7124; \
21.75,1.0,0,5.1051; 22.5,1.0,0,5.4978; 23.25,1.0,0,5.8905; 24,1.0,0,6.2832; \
27,0,0,0"

log "===== [2/2] ${TAG}_omni：画方形全向 27 s（渐进转向） ====="
rec "${TAG}_omni.mp4" omni 1400 "$SQUARE" "1.0" "1.0"

log "===== 收尾 ====="
ls -la "$OUT/${TAG}_walk10.mp4" "$OUT/${TAG}_omni.mp4" 2>/dev/null
echo "=== VIDEO CHECK DONE ==="
