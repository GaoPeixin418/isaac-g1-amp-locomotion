#!/usr/bin/env bash
# =============================================================================
# 23 - 真正的 AMP + 对称性奖励重训（λ 显式锁死 + 启动后复核）
#
# 为什么废弃 21_train_sym.sh
# -----------------------------------------------------------------------------
# 21 只校验了「奖励补丁在不在」，没校验 λ。
# 而 cfg 在 10-20 晚上为录 PPO 基线被改成 task_style_lerp=1.0 后一直没还原，
# 于是 15-52-15 / 18-54-54 两个 run 训的都是
#   r = 1.0*task + 0.0*style = 纯 PPO，风格奖励根本没参与。
# 日志里 Episode_Reward/style 照样在打印（独立记录），所以从日志看不出来。
# → 那两个 run 的结论（SI 3.7% / 横漂 0.03m / 高速崩）全部作废：
#   「换奖励 + 换 λ」两个变量同时动，无法归因。
#
# 本脚本的防复发机制（三道闸）
# -----------------------------------------------------------------------------
#   闸1  LERP 必须显式给，脚本不提供"悄悄的默认值"
#   闸2  patch_lerp.py 写完后回读验证，失败就 exit 1，训练根本起不来
#   闸3  训练启动 150s 后，从**本次新建的 run 目录**的 params/agent.yaml
#        再验一次 λ，不一致直接杀掉（宁可没结果，也不要错结果）
#
# 用法
# -----------------------------------------------------------------------------
#   nohup env LERP=0.5 MAX_ITER=4000 bash /data/setup/23_train_amp_sym.sh \
#         > /data/logs/train23.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export PIP_CONSTRAINT=/data/constraints.txt
export PIP_CACHE_DIR=/data/pip_cache
export HF_HOME=/data/hf_cache
export OMNI_KIT_ACCEPT_EULA=YES

PY=/data/envs/isaaclab/bin/python
TASK=${TASK:-LeggedLab-Isaac-AMP-G1-v0}
NUM_ENVS=${NUM_ENVS:-4096}
MAX_ITER=${MAX_ITER:-4000}
SEED=${SEED:-42}
RESUME=${RESUME:-0}
CKPT=${CKPT:-}
LOAD_RUN=${LOAD_RUN:-}
# λ 不给就拒绝运行 —— 上次事故的根因就是"没人记得现在是几"
LERP=${LERP:-}

CFG=/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py
AGENT_CFG=/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

# ---- 闸1：λ 必须显式给 ----------------------------------------------------
if [ -z "$LERP" ]; then
    log "!! 必须显式给 LERP（0.3/0.5/0.7 …）。不给就跑 = 重蹈 10-22 覆辙"
    exit 1
fi

# ---- 闸2：写 + 回读验证 ---------------------------------------------------
log "===== [闸1+2/4] 锁死 λ=$LERP ====="
"$PY" /data/setup/patch_lerp.py "$LERP" "$AGENT_CFG" || {
    log "!! λ 写入/验证失败，训练中止（宁可不跑，也不跑错的）"
    exit 1
}

# ---- 奖励补丁 -------------------------------------------------------------
if [ "${NO_PATCH:-0}" != "1" ]; then
    log "===== [2/4] 打步态对称性奖励补丁 ====="
    "$PY" /data/setup/patch_amp_rewards.py "$CFG" || { log "!! 补丁失败，中止"; exit 1; }
else
    log "===== [2/4] 跳过奖励补丁（NO_PATCH=1）====="
fi
grep -q "air_time_variance" "$CFG" || { log "!! 配置里没有 air_time_variance，中止"; exit 1; }

# ---- 3) 训练（后台起，方便后面复核 + 流式输出）---------------------------
log "===== [3/4] 训练 λ=$LERP MAX_ITER=$MAX_ITER SEED=$SEED ====="
cd /data/legged_lab || exit 1

EXTRA_ARGS=""
if [ "$RESUME" = "1" ]; then
    [ -z "$CKPT" ] && { log "!! RESUME=1 但没给 CKPT"; exit 1; }
    EXTRA_ARGS="--resume --load_run $LOAD_RUN --checkpoint $CKPT"
    [ -f "$LOGDIR/$LOAD_RUN/$CKPT" ] || { log "!! 权重不存在 $LOAD_RUN/$CKPT"; exit 1; }
    log "  续训: run=$LOAD_RUN CKPT=$CKPT"
fi

# 记录训练前的 run 列表，后面用差集找出本次新建的 run 目录
ls -1 "$LOGDIR" > /tmp/train23_runs_before.txt 2>/dev/null

# shellcheck disable=SC2086
nohup bash -c "yes Yes | $PY scripts/rsl_rl/train.py --task $TASK --headless \
    --num_envs $NUM_ENVS --max_iterations $MAX_ITER --seed $SEED $EXTRA_ARGS" \
    > /tmp/train23_raw.log 2>&1 &
TRAIN_PID=$!
log "  训练进程 PID=$TRAIN_PID"

# 边等边把子进程的日志流出来，nohup 日志里能实时看到进度
tail -f --pid="$TRAIN_PID" /tmp/train23_raw.log 2>/dev/null &
TAIL_PID=$!

# ---- 闸3：启动后从新 run 目录复核 ----------------------------------------
sleep 150
NEW_RUN=$(comm -13 <(sort -u /tmp/train23_runs_before.txt) <(ls -1 "$LOGDIR" | sort -u) | head -1)
if [ -z "$NEW_RUN" ]; then
    log "!! 150s 内没找到新建的 run 目录，可能训练没起来，杀掉"
    pkill -9 -f "train.py --task $TASK" 2>/dev/null
    exit 1
fi
ACTUAL=$(grep -h "task_style_lerp" "$LOGDIR/$NEW_RUN/params/agent.yaml" 2>/dev/null |
         tr -d ' ' | tail -1)
log "===== [闸3/4] 复核 run=$NEW_RUN  实际 λ: ${ACTUAL:-<读不到>} ====="
if [ "$ACTUAL" != "task_style_lerp:$LERP" ]; then
    log "!! 实际 λ($ACTUAL) 与预期($LERP) 不符，杀掉训练"
    pkill -9 -f "train.py --task $TASK" 2>/dev/null
    exit 1
fi
log "== λ 复核通过，本次确实是 AMP（λ=$LERP），训练继续"

wait $TRAIN_PID
RC=$?
kill $TAIL_PID 2>/dev/null
log "===== [4/4] 训练结束 rc=$RC  run=$NEW_RUN ====="
ls -1t "$LOGDIR/$NEW_RUN"/*.pt 2>/dev/null | head -5
echo "=== TRAIN 23 DONE ==="
