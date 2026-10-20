#!/usr/bin/env bash
# =============================================================================
# 11 - 一键连跑：三组步态评估 + 两段对比视频
#
# 为什么合成一个脚本：
#   每步都要等人盯着的话，Kit 启动的几分钟空档就白白浪费了。串行一条命令跑完，
#   中途我去做分析，最后一次把结果全拉回来。
#
# ⚠️ 2026-10-20 更正：task_style_lerp 的语义与直觉相反
#   源码 rsl_rl/modules/amp.py:203
#       r = task_style_lerp * r_task + (1 - task_style_lerp) * r_style
#   即 λ 是「任务」的权重，不是「风格」的权重。所以：
#       LERP=0.0 -> 纯风格（纯模仿），没有任何任务奖励
#       LERP=0.5 -> AMP 正式配置，任务/风格各半
#       LERP=1.0 -> 纯任务（= 标准 PPO），没有任何风格奖励
#   本脚本第 3 步评的 2026-10-20_18-02-56 是 LERP=0.0，
#   所以它是「纯风格消融」，不是 PPO 基线 —— 标签必须为 style_only。
#   真正的 PPO 基线要 LERP=1.0 单独训（见 12_train_ppo_baseline.sh）。
#
# 顺序（按重要性排，最重要的先跑）：
#   1) AMP 风格峰值 model_2000   <- 决定最终交付用哪个模型
#   2) AMP 训练终点 model_4999   <- 对照，验证"后 3000 轮到底有没有用"
#   3) 纯风格消融（LERP=0）      <- 验证"光模仿人、不看任务会怎样"
#   4) 两段视频                  <- 视觉证据
#
# 用法：
#   nohup bash /data/setup/11_run_all_evals.sh > /data/logs/all_evals.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

AMP_RUN=${AMP_RUN:-2026-10-19_16-37-21}
PPO_RUN=${PPO_RUN:-2026-10-20_18-02-56}
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# 自动挑最新 checkpoint：rsl_rl 终止时存的是 (N-1)，写死文件名容易踩空
pick_ckpt() {
    local dir="$1"; shift
    local want="$1"
    if [ -f "$LOGDIR/$dir/$want" ]; then
        echo "$want"; return
    fi
    log "!! $want 不存在，改为自动挑最新"
    ls -1 "$LOGDIR/$dir" | grep -E '^model_[0-9]+\.pt$' \
        | sort -t_ -k2 -n | tail -1
}

CK_AMP2000=$(pick_ckpt "$AMP_RUN" model_2000.pt)
CK_AMP4999=$(pick_ckpt "$AMP_RUN" model_4999.pt)
CK_PPO=$(pick_ckpt "$PPO_RUN" model_2999.pt)
log "ckpt: amp2000=$CK_AMP2000 amp4999=$CK_AMP4999 ppo=$CK_PPO"

log "===== [1/5] eval AMP model_2000 ====="
RUN=$AMP_RUN CKPT=$CK_AMP2000 TAG=amp_2000 bash /data/setup/10_eval_gait.sh

log "===== [2/5] eval AMP model_4999 ====="
RUN=$AMP_RUN CKPT=$CK_AMP4999 TAG=amp_4999 bash /data/setup/10_eval_gait.sh

log "===== [3/5] eval 纯风格消融 (LERP=0) ====="
RUN=$PPO_RUN CKPT=$CK_PPO TAG=style_only bash /data/setup/10_eval_gait.sh

log "===== [4/5] video 纯风格消融 ====="
RUN=$PPO_RUN CKPT=$CK_PPO NUM_ENVS=4 VIDEO_LEN=700 bash /data/setup/07_mid_eval.sh

log "===== [5/5] video AMP model_2000 ====="
RUN=$AMP_RUN CKPT=$CK_AMP2000 NUM_ENVS=4 VIDEO_LEN=700 bash /data/setup/07_mid_eval.sh

log "===== ALL EVALS DONE ====="
echo "=== ALL EVALS DONE $(date '+%m-%d %H:%M:%S') ==="
ls -la /data/eval_out/
