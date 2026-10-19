#!/usr/bin/env bash
# =============================================================================
# 07 - 中期评估：用中间 checkpoint 提前录视频
#
# 为什么要在训练没跑完时就做这件事：
#   AMP 的 style 奖励高 ≠ 视频里的步态好看。如果等到 5000 iter 跑完才发现
#   「分数不错但走得像机器人」，就白白浪费了 5 小时。提前用中间 checkpoint
#   录一段视频，能把「能不能看」这个最关键的风险点提前 3 小时暴露出来。
#
#   另一个收益：这段 1600 iter 的视频本身就是「训练中间产物」，
#   和最终 5000 iter 的对比能直接放进报告，展示风格是怎么一步步学到的。
#
# 安全检查：主训练在跑，play 用小 num_envs 避免抢显存
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

PY=/data/envs/isaaclab/bin/python
TASK=${TASK:-LeggedLab-Isaac-AMP-G1-v0}
RUN=${RUN:-2026-10-19_16-37-21}
CKPT=${CKPT:-model_1600.pt}
NUM_ENVS=${NUM_ENVS:-4}
VIDEO_LEN=${VIDEO_LEN:-800}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

cd /data/legged_lab || exit 1

# retrieve_file_path() 只做 os.path.isfile 检查，不做拼接 → 必须给绝对路径
CKPT_PATH=/data/legged_lab/logs/rsl_rl/g1_amp/$RUN/$CKPT
if [ ! -f "$CKPT_PATH" ]; then
    log "!! checkpoint 不存在: $CKPT_PATH"
    ls -1 /data/legged_lab/logs/rsl_rl/g1_amp/$RUN/ | tail -5
    exit 1
fi
log "play: task=$TASK ckpt=$CKPT_PATH num_envs=$NUM_ENVS video_len=$VIDEO_LEN"

timeout -s KILL 1200 yes Yes | "$PY" scripts/rsl_rl/play.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --video \
    --video_length "$VIDEO_LEN" \
    --checkpoint "$CKPT_PATH" 2>&1

log "=== PLAY DONE (exit=$?) ==="
echo "=== 视频目录 ==="
find /data/legged_lab/logs/rsl_rl/g1_amp -name "*.mp4" -newermt "-20 minutes" 2>/dev/null
