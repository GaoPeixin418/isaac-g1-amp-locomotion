#!/usr/bin/env bash
# =============================================================================
# 26 - λ 扫描（方案 B：三臂 × 1500 iter，训完立刻评估）
#
# 为什么必须做这次扫描
# -----------------------------------------------------------------------------
# λ 是 r = λ·r_task + (1−λ)·r_style 里唯一的连续旋钮。
# 我们手上的端点实测：λ=0（站着不动）+ λ=1（纯 PPO，10-20 已录过）。
# 中间到底是什么形状，只有一个 λ=0.5 的点 —— 而那个点是框架默认值，
# 不是我们选的。"默认"不等于"最优"，所以要扫。
#
# 为什么是 1500 iter 而不是 4000
# -----------------------------------------------------------------------------
# 10-22 的 TB 曲线实测：线速度跟踪在 1500 iter 就已饱和
# （1500: 0.658 / 2000: 0.670 / 4000: 0.650），
# 再训下去涨的是动作平滑和转向跟踪，不影响"哪个 λ 更好"这个排序判断。
# 1500 × 3 臂 = 3.7h；4000 × 3 = 9.8h。省 6 小时买同样的排序结论。
#
# 为什么固定 seed=42 / 4096 env / 同一份奖励表
# -----------------------------------------------------------------------------
# 三臂之间只允许 λ 一个变量动。seed 固定后，三个 run 看到的
# 初始网络、指令采样序列、判别器初始化完全一致，差异只能来自 λ。
#
# 每臂训完立刻评估（不等三臂全训完）
# -----------------------------------------------------------------------------
# 万一中间要中断，已经训完的臂也有数据；更重要的是能逐个看趋势。
#
# 五指标判据（缺一不可，单一指标会误导）
# -----------------------------------------------------------------------------
#   1. vel_err      六档速度跟踪误差（任务完成度，硬门槛）
#   2. vmax         稳态下真正能跟住的最高速度（不能只会慢走）
#   3. style        TB 里的判别器风格分（拟人度）
#   4. SI_clear     步态对称性指数（治"瘸"）
#   5. drift        25 秒横向漂移（治"飘"）
# 先用 1+2 淘汰不合格的，再在合格者里取 3 最高，4+5 作为平局裁决。
#
# 用法
# -----------------------------------------------------------------------------
#   nohup bash /data/setup/26_sweep_lambda.sh > /data/logs/sweep_lambda.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

PY=/data/envs/isaaclab/bin/python
LOGDIR=/data/legged_lab/logs/rsl_rl/g1_amp
EVAL_OUT=/data/eval_out
MAX_ITER=${MAX_ITER:-1500}
SEED=${SEED:-42}
NUM_ENVS=${NUM_ENVS:-4096}

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }
mkdir -p "$EVAL_OUT"
rm -f /data/eval_out/sweep_runs.txt

# λ=0.5 放第一个：它是最可能的赢家，也最先验证对称奖励在真 λ 下会不会崩。
# 只要这一臂在 1500 iter 就高速崩，后面两臂不用看就知道要改奖励。
for PAIR in "0.50:lam050" "0.25:lam025" "0.75:lam075"; do
    L=${PAIR%%:*}
    TAG=${PAIR##*:}

    log "########## 臂 $TAG  λ=$L ##########"
    ls -1 "$LOGDIR" > /tmp/sw_before_$TAG.txt 2>/dev/null

    # ---- 训练（23 脚本内含三道闸：λ 必须显式 / 写后回读 / 启动后复核）----
    LERP=$L MAX_ITER=$MAX_ITER SEED=$SEED NUM_ENVS=$NUM_ENVS \
        bash /data/setup/23_train_amp_sym.sh
    TRAIN_RC=$?
    log "  训练返回 rc=$TRAIN_RC"

    NEW_RUN=$(comm -13 <(sort -u /tmp/sw_before_$TAG.txt) \
                       <(ls -1 "$LOGDIR" | sort -u) | head -1)
    if [ -z "$NEW_RUN" ]; then
        log "!! 没找到本臂新建的 run 目录，跳过评估"
        continue
    fi
    echo "$TAG $L $NEW_RUN" >> /data/eval_out/sweep_runs.txt
    log "  本臂 run = $NEW_RUN"

    # ---- 评估 1：步态 + 速度跟踪（6 env × 6 档 × 1500 步）----------------
    log "  [评估1/3] 步态指标"
    RUN=$NEW_RUN CKPT=model_$((MAX_ITER - 1)).pt TAG=$TAG \
        OUT=$EVAL_OUT STEPS=1500 \
        bash /data/setup/10_eval_gait.sh > /tmp/sw_eval_$TAG.log 2>&1
    log "  eval rc=$?  -> $(ls -1 $EVAL_OUT/gait_$TAG.npz 2>/dev/null || echo 未产出)"

    # ---- 评估 2：对称性（读上一步的 npz）---------------------------------
    log "  [评估2/3] 对称性"
    if [ -f "$EVAL_OUT/gait_$TAG.npz" ]; then
        "$PY" /data/setup/analyze_symmetry.py "$EVAL_OUT/gait_$TAG.npz" 2>&1 | tail -12
    fi

    # ---- 评估 3：风格分（直接从 tfevents 读，不需要再起 Isaac Sim）-------
    log "  [评估3/3] 风格分（TB 实测）"
    "$PY" /data/setup/dump_tb.py "$LOGDIR/$NEW_RUN" \
        --iters "$MAX_ITER" --win 200 --only style 2>&1 | tail -6
done

# ---- 汇总：三臂并排 -------------------------------------------------------
log "########## 汇总 ##########"
PAIRS=""
while read -r tag lerpv run; do
    [ -f "$EVAL_OUT/gait_$tag.npz" ] && PAIRS="$PAIRS $tag=$EVAL_OUT/gait_$tag.npz"
done < /data/eval_out/sweep_runs.txt

if [ -n "$PAIRS" ]; then
    # shellcheck disable=SC2086
    "$PY" /data/setup/compare_gait.py $PAIRS 2>&1 | tail -40
else
    log "!! 没有任何臂产出 npz，汇总跳过"
fi

log "===== 各臂 run 目录 ====="
cat /data/eval_out/sweep_runs.txt 2>/dev/null
echo "=== SWEEP LAMBDA DONE ==="
