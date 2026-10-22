#!/usr/bin/env bash
# =============================================================================
# 明早开工：无缝续训 / 调 style 权重
#
# 前提：/data 数据盘已保留全部环境（无需重装，省 5.8 小时）
# 用法见本脚本底部注释
# =============================================================================
set -u

CONDA_PREFIX=/data/envs/isaaclab
PYTHON=/data/envs/isaaclab/bin/python
export PYTHON
export CONDA_PREFIX
export OMNI_KIT_ACCEPT_EULA=YES
export PIP_CACHE_DIR=/data/pip_cache
export PATH="$CONDA_PREFIX/bin:$PATH"

LAB=/data/legged_lab
cd "$LAB" || exit 1

MODE="${MODE:-resume}"          # resume | fresh | sweep
TASK="${TASK:-LeggedLab-Isaac-AMP-G1-v0}"
RUN="${RUN:-2026-10-19_16-37-21}"
NUM_ENVS="${NUM_ENVS:-4096}"
MAX_ITER="${MAX_ITER:-7000}"    # resume 模式下的目标总数
LERP="${LERP:-0.5}"             # task_style_lerp（λ）
# ⚠️ 2026-10-22 更正：λ 的语义与直觉相反！
#    r = λ * r_task + (1-λ) * r_style   （rsl_rl/modules/amp.py:203）
#    λ=0.0 -> 纯模仿，无任务奖励 -> 站着不动
#    λ=0.5 -> 标准 AMP          <- 默认
#    λ=1.0 -> 纯任务 = 普通 PPO，风格奖励系数 0
#    即：λ 越大越"重任务"，越小越"重风格"。
#    原注释写的"0.3 保守 / 0.7 重风格"是反的，也是 10-22 事故的思想根源。
SCALE="${SCALE:-2.0}"           # style_reward_scale
LOG=/data/logs/train_resume.log

echo "[$(date '+%m-%d %H:%M:%S')] MODE=$MODE LERP=$LERP SCALE=$SCALE MAX_ITER=$MAX_ITER"

# ---------------------------------------------------------------------------
# 改 task_style_lerp：直接改 legged_lab 源码里的默认值
# 路径固定，改完可用 git diff 自查
# ---------------------------------------------------------------------------
CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py"
DEFAULT_LERP=0.5
DEFAULT_SCALE=2.0

if [ "$MODE" = "sweep" ] || [ "$LERP" != "$DEFAULT_LERP" ] || [ "$SCALE" != "$DEFAULT_SCALE" ]; then
    echo "[INFO] patch $CFG  task_style_lerp=$LERP style_reward_scale=$SCALE"
    # 备份到独立的 .thisrun 文件，**不覆盖 .bak**：
    # .bak 里可能残留着上一次的旧值（曾经的 0.0），拿来还原就是新事故
    cp "$CFG" "$CFG.thisrun"
    DID_BACKUP=1
    sed -i "s/task_style_lerp=[0-9.]*/task_style_lerp=$LERP/g" "$CFG"
    sed -i "s/style_reward_scale=[0-9.]*/style_reward_scale=$SCALE/g" "$CFG"
    grep -n "amp_discriminator=RslRlAmpCfg.AMPDiscriminatorCfg" -A 2 "$CFG"
fi

# ⚠️ 本脚本会**永久修改 legged_lab 源码里的 cfg**（不是本次 run 的副本）。
#    10-22 事故：这里改完 1.0 忘还原，之后 3 个 run 全是纯 PPO，4 小时白烧。
#    所以挂一个 trap：不管正常结束、报错还是被 Ctrl-C，都把 cfg 还原回去。
#    DID_BACKUP 标志很重要：只有**本次**真的备份过才还原。
#    否则会拿 .bak 里上一次残留的旧值（曾经是 λ=0.0）覆盖当前值，制造新事故。
DID_BACKUP=0
restore_cfg() {
    if [ "$DID_BACKUP" = "1" ] && [ -f "$CFG.thisrun" ]; then
        cp -f "$CFG.thisrun" "$CFG"
        rm -f "$CFG.thisrun"
        echo "[INFO] 已还原 cfg: $(grep -o 'task_style_lerp=[0-9.]*' "$CFG")"
    fi
}
trap restore_cfg EXIT

# ---------------------------------------------------------------------------
case "$MODE" in
resume)
    echo "[INFO] 从 $RUN 最新 checkpoint 续训到 $MAX_ITER iter"
    yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
        --task "$TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITER" \
        --resume \
        --load_run "$RUN" 2>&1
    ;;
fresh)
    echo "[INFO] 从头训 $MAX_ITER iter（新 SaveInterval 会开新 run 目录）"
    yes Yes | "$PYTHON" scripts/rsl_rl/train.py \
        --task "$TASK" \
        --headless \
        --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITER" 2>&1
    ;;
*)
    echo "!! 未知 MODE=$MODE（可选 resume / fresh）"
    exit 1
    ;;
esac

echo "=== TRAIN RESUME DONE $(date '+%m-%d %H:%M:%S') ==="
