#!/usr/bin/env bash
# =============================================================================
# 25 - amp2000 vs amp4000 同条件对比成片
#
# 为什么要专门录这一段
# -----------------------------------------------------------------------------
# Q2 里有个直觉：「amp2000 风格分 0.775 > amp4000 的 0.651，是不是 2000 更好？」
# 但风格分是判别器原始输出，不走 lerp 加权；而 4000 在动作平滑 / 转向跟踪上
# 明显更好。两个 checkpoint 谁更适合当主 demo，不能只看一条曲线 ——
# 得在**完全相同的条件**下并排放出来让人看。
#
# 同条件的四条约束（缺一不可）
# -----------------------------------------------------------------------------
#   1. 同一个 run（2026-10-19_16-37-21，λ=0.5 真 AMP），只有 checkpoint 不同
#   2. 同样的 env 数、同样的帧数、同样的速度指令
#   3. 指令固定补丁（patch_play_cmd.py）在位 —— 不随机重采样、不走曲线
#   4. PLAY 配置补丁（patch_play_env.py）在位 —— 无推搡、无外力、无观测噪声
#
# 为什么用 10 个环境
# -----------------------------------------------------------------------------
# 单机 1 个 env 只是「一次采样」，换个初始姿态就换一个结果。
# 10 个 env 同时跑同一条 1.2 m/s 指令，步态一致性 / 有没有瘸
# 一眼能看出来，样本量也支撑得住。
# 另外再录 6 档速度并排（0.5→3.0），证明不是只会走一个速度。
#
# 产出
# -----------------------------------------------------------------------------
#   amp2000_multi10_12.mp4   10 env，1.2 m/s 走直线（非整数档）
#   amp4000_multi10_12.mp4   10 env，1.2 m/s 走直线  ← 与上面横向比
#   amp2000_ladder6.mp4      6 env，0.5/1.0/1.5/2.0/2.5/3.0 m/s
#   amp4000_ladder6.mp4      6 env，同上                ← 与上面横向比
#
# 用法
# -----------------------------------------------------------------------------
#   nohup env EVAL_TIMEOUT=3600 bash /data/setup/25_video_compare.sh \
#         > /data/logs/video_compare.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES
export EVAL_TIMEOUT=${EVAL_TIMEOUT:-3600}

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
OUT=/data/videos

# 两个 checkpoint 出自同一个 run → 保证除训练步数外无其他变量
RUN=${RUN:-2026-10-19_16-37-21}
VIDEO_LEN=${VIDEO_LEN:-700}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$OUT"

# ---- 0) 两个补丁必须都在位，否则拒绝录制 ---------------------------------
log "===== [0/5] 校验补丁 ====="
"$PY" /data/setup/patch_play_cmd.py /data/legged_lab/scripts/rsl_rl/play.py || {
    log "!! 指令固定补丁失败，中止"; exit 1; }
"$PY" /data/setup/patch_play_env.py || {
    log "!! PLAY 配置补丁失败，中止"; exit 1; }
for m in model_2000.pt model_4000.pt; do
    [ -f "$LOGDIR/$RUN/$m" ] || { log "!! 缺权重 $RUN/$m，中止"; exit 1; }
done
log "  权重齐全，λ 快照 = $(grep -h 'task_style_lerp' "$LOGDIR/$RUN/params/agent.yaml" | tr -d ' ')"

record() {  # $1=ckpt $2=num_envs $3=PIN_VEL $4=outname
    local ckpt=$1 nenv=$2 pin=$3 out=$4
    rm -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4"
    log "--- 录制 $out | ckpt=$ckpt envs=$nenv pin=$pin"
    PIN_VEL="$pin" \
        TASK=LeggedLab-Isaac-AMP-G1-Play-v0 \
        RUN="$RUN" CKPT="$ckpt" NUM_ENVS="$nenv" VIDEO_LEN="$VIDEO_LEN" \
        bash /data/setup/07_mid_eval.sh
    if cp -f "$LOGDIR/$RUN/videos/play/rl-video-step-0.mp4" "$OUT/$out" 2>/dev/null; then
        log "saved $OUT/$out"
    else
        log "!! 未生成 $out（继续下一段，不中断）"
    fi
}

# ---- 1-2) 主对比：10 env × 1.2 m/s ---------------------------------------
log "===== [1/5] amp2000 | 10 env × 1.2 m/s ====="
record model_2000.pt 10 "1.2" amp2000_multi10_12.mp4

log "===== [2/5] amp4000 | 10 env × 1.2 m/s ====="
record model_4000.pt 10 "1.2" amp4000_multi10_12.mp4

# ---- 3-4) 速度覆盖：6 档并排 ---------------------------------------------
log "===== [3/5] amp2000 | 6 档 0.5→3.0 m/s ====="
record model_2000.pt 6 "0.5,1.0,1.5,2.0,2.5,3.0" amp2000_ladder6.mp4

log "===== [4/5] amp4000 | 6 档 0.5→3.0 m/s ====="
record model_4000.pt 6 "0.5,1.0,1.5,2.0,2.5,3.0" amp4000_ladder6.mp4

log "===== [5/5] 收尾 ====="
ls -la "$OUT"/amp2000_*.mp4 "$OUT"/amp4000_*.mp4 2>/dev/null
echo "=== VIDEO COMPARE DONE ==="
