#!/usr/bin/env bash
# =============================================================================
# 终版视频录制（iter 4999）
#
# 用法：
#   VIDEO_LEN=600 NUM_ENVS=4 bash /data/setup/09_final_video.sh
#
# 产物：<run>/videos/play/rl-video-step-0.mp4
# =============================================================================
set -u

CONDA_PREFIX=/data/envs/isaaclab
PYTHON=/data/envs/isaaclab/bin/python
export CONDA_PREFIX
export PYTHON
export OMNI_KIT_ACCEPT_EULA=YES
export PIP_CACHE_DIR=/data/pip_cache
export PATH="$CONDA_PREFIX/bin:$PATH"

TASK="${TASK:-LeggedLab-Isaac-AMP-G1-v0}"
RUN="${RUN:-2026-10-19_16-37-21}"
CKPT_ABS="${CKPT_ABS:-/data/legged_lab/logs/rsl_rl/g1_amp/2026-10-19_16-37-21/model_4999.pt}"
NUM_ENVS="${NUM_ENVS:-4}"
VIDEO_LEN="${VIDEO_LEN:-600}"

echo "[$(date '+%m-%d %H:%M:%S')] 录视频 ckpt=$CKPT_ABS len=$VIDEO_LEN envs=$NUM_ENVS"

cd /data/legged_lab || exit 1

# 注意：Isaac Lab 的 retrieve_file_path 只做 os.path.isfile，
# 不会把 --load_run 拼到 --checkpoint 前面 → --checkpoint 必须给绝对路径
yes Yes | "$PYTHON" scripts/rsl_rl/play.py \
    --task "$TASK" \
    --headless \
    --num_envs "$NUM_ENVS" \
    --video \
    --video_length "$VIDEO_LEN" \
    --load_run "$RUN" \
    --checkpoint "$CKPT_ABS" 2>&1

echo "=== VIDEO DONE $(date '+%m-%d %H:%M:%S') ==="
echo "产物目录：/data/legged_lab/logs/rsl_rl/g1_amp/$RUN/videos/play/"
