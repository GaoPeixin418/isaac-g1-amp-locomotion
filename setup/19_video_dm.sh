#!/usr/bin/env bash
# =============================================================================
# 19 - DeepMimic 视频录制（Debug 模式：策略机器人 vs 半透明参考机器人）
#
# 为什么值得单独录一版：
#   legged_lab 给 DeepMimic 准备了 -Debug-v0 变体，会在场景里摆一个
#   半透明参考机器人，跟策略机器人并排跑。录出来能直接看到
#   "策略 vs 标准答案" 的差距 —— 这个我们自己去实现代价很高。
#   Debug 模式同时会把动作数据的时间轴可视化出来。
#
# 用法：
#   bash 19_video_dm.sh                       # Debug 模式（默认，最直观）
#   TASK=LeggedLab-Isaac--Deepmimic-G1-Play-v0 bash 19_video_dm.sh   # 干净版
# =============================================================================
set -u

CONDA_PREFIX=/data/envs/isaaclab
PYTHON=/data/envs/isaaclab/bin/python
export CONDA_PREFIX
export PYTHON
export OMNI_KIT_ACCEPT_EULA=YES
export PIP_CACHE_DIR=/data/pip_cache
export PATH="$CONDA_PREFIX/bin:$PATH"

TASK="${TASK:-LeggedLab-Isaac--Deepmimic-G1-Debug-v0}"
RUN="${RUN:-2026-10-21_18-44-12}"
CKPT_ABS="${CKPT_ABS:-/data/legged_lab/logs/rsl_rl/g1_deepmimic/2026-10-21_18-44-12/model_2999.pt}"
NUM_ENVS="${NUM_ENVS:-1}"
VIDEO_LEN="${VIDEO_LEN:-600}"

OUTDIR=/data/legged_lab/logs/rsl_rl/g1_deepmimic/$RUN/videos/play

echo "[$(date '+%m-%d %H:%M:%S')] 录 DeepMimic  TASK=$TASK  envs=$NUM_ENVS  len=$VIDEO_LEN"
ls -la "$CKPT_ABS" || { echo "[ERROR] checkpoint 不存在"; exit 1; }

cd /data/legged_lab || exit 1

yes Yes | "$PYTHON" scripts/rsl_rl/play.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --video \
    --video_length "$VIDEO_LEN" \
    --load_run "$RUN" \
    --checkpoint "$CKPT_ABS" 2>&1

echo "=== DM VIDEO DONE $(date '+%m-%d %H:%M:%S') ==="

# 立刻另存，否则下一段会把 rl-video-step-0.mp4 覆盖掉
# （踩过：AMP 终版视频就是这么被 model_2000 那次录制盖掉的）
if [ -f "$OUTDIR/rl-video-step-0.mp4" ]; then
    SUFFIX="debug"; [ "$TASK" = "LeggedLab-Isaac--Deepmimic-G1-Play-v0" ] && SUFFIX="play"
    mkdir -p /data/videos
    cp "$OUTDIR/rl-video-step-0.mp4" "/data/videos/deepmimic_${SUFFIX}.mp4"
    echo "已另存 -> /data/videos/deepmimic_${SUFFIX}.mp4"
    ls -la /data/videos/deepmimic_${SUFFIX}.mp4
else
    echo "[WARN] 没找到录出来的视频"
    ls -la "$OUTDIR" 2>&1 | tail -5
fi
