#!/usr/bin/env bash
# =============================================================================
# 41 - v2.3 续训：model_1599 -> 总 3000 iter（与 v1 训练量对齐）
#                  2026-11-18 开机窗口（约 2h，17:21 自动关机）
#
# ★ 两个会致命的语义，已固化在下面，别"顺手改"：
#
# 1) --max_iterations 是【增量】不是【总数】
#    /data/rsl_rl/rsl_rl/runners/amp_runner.py:65-66
#        start_it = self.current_learning_iteration   # resume 后 = 1599
#        total_it = start_it + num_learning_iterations
#    所以 1599 + MAX_ITER = 目标总数。要到 3000 必须传 1401。
#    传 1600 -> 3199（上次 watchdog 日志 1679/3199 就是这么来的），会被关机切断。
#    setup/08_train_resume.sh 注释写的"目标总数"是错的，别信。
#
# 2) resume 一律 WARMUP=0（满额）
#    env._v23_reward_calls 挂在 env 上，进程重启从 0 重来。若仍传 12000，
#    会重新爬 500 iter 斜坡 = 把已过形成期的策略又松绑一次，与实验意图相反。
#
# 崩溃自愈：Isaac Sim 5.1 曾在 iter217 触发 carb.tasking Mutex 断言（偶发竞态），
#   用「300s 无新 Learning iteration」判卡死 -> kill -9 -> resume 接上。
#   达标时【先 kill 训练进程再 break】（修 39 只 break 不 kill、导致 attempt2
#   又续到 3199 的 bug）。
#
# 用法：  nohup bash /data/setup/41_resume_v23_to3000.sh \
#             > /data/logs/launch_41.log 2>&1 &
# =============================================================================
set -u

CONDA_PREFIX=/data/envs/isaaclab
PYTHON=/data/envs/isaaclab/bin/python
export PYTHON CONDA_PREFIX
export OMNI_KIT_ACCEPT_EULA=YES
export PIP_CACHE_DIR=/data/pip_cache
export PATH="$CONDA_PREFIX/bin:$PATH"

LAB=/data/legged_lab
cd "$LAB" || exit 1

TASK="${TASK:-LeggedLab-Isaac-AMP-G1-v0}"
NUM_ENVS="${NUM_ENVS:-4096}"
SRC_RUN="${SRC_RUN:-2026-11-16_16-49-16}"
SRC_CKPT="${SRC_CKPT:-model_1599.pt}"
START_ITER=1599
TARGET_TOTAL="${TARGET_TOTAL:-3000}"
MAX_ITER=$((TARGET_TOTAL - START_ITER))       # = 1401（增量！）
SEED="${SEED:-42}"
STALL_SEC="${STALL_SEC:-300}"
MAX_RESTART="${MAX_RESTART:-3}"

TRAIN_LOG="${TRAIN_LOG:-/data/logs/train_41_resume.log}"
RUNROOT=$LAB/logs/rsl_rl/g1_amp
ENV_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
AGENT_CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py"
REWARDS="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py"

fail() { echo "!! $1"; exit 1; }

# ---------------------------- 前置闸 ----------------------------
if ps aux | grep -E "rsl_rl/[t]rain\.py" | grep -v grep >/dev/null; then
    ps aux | grep -E "rsl_rl/[t]rain\.py" | grep -v grep
    fail "已有训练进程，先清场"
fi
[ -f "$RUNROOT/$SRC_RUN/$SRC_CKPT" ] || fail "缺起点 $RUNROOT/$SRC_RUN/$SRC_CKPT"
grep -q "leg_work_asymmetry" "$ENV_CFG" || fail "leg_work_asymmetry 不在位"
"$PYTHON" - <<'PY' || fail "work_asym 权重不是 -1.0"
import sys
c = open("/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py", encoding="utf-8").read()
sys.exit(0 if "func=mdp.leg_work_asymmetry,\n        weight=-1.0," in c else 1)
PY
grep -q "ang_vel_xy_l2, weight=-0.05" "$ENV_CFG" || fail "angvel 不是 -0.05"
LERP=$(grep -o "task_style_lerp=[0-9.]*" "$AGENT_CFG" | head -1 | cut -d= -f2)
[ "$(awk -v v="$LERP" 'BEGIN{printf "%g", v}')" = "0.5" ] || fail "lerp=$LERP 不是 0.5"
grep -q "PATCHED-SYM-V23-WARMUP" "$REWARDS" || fail "warmup 补丁不在位"
nvidia-smi -L >/dev/null 2>&1 || fail "GPU 不可见（无卡模式？）"

echo "=================================================================="
echo " v2.3 续训  $SRC_RUN/$SRC_CKPT (iter=$START_ITER)"
echo " --max_iterations=$MAX_ITER  [增量] -> 目标总数 $TARGET_TOTAL"
echo " WARMUP=0 (满额)  ENVS=$NUM_ENVS  SEED=$SEED"
echo " 训练日志 $TRAIN_LOG"
echo " 卡死判定 ${STALL_SEC}s 无新 iter；最多重启 ${MAX_RESTART} 次"
echo "=================================================================="
[ "$MAX_ITER" -gt 0 ] || fail "MAX_ITER=$MAX_ITER 非正，目标应 > $START_ITER"

mkdir -p /data/logs /data/progress
: > "$TRAIN_LOG"

last_iter() {
    grep -o "Learning iteration [0-9]*" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]*$"
}
kill_train() {
    for p in $(ps aux | grep "rsl_rl/[t]rain\.py" | grep -v grep | awk '{print $2}'); do
        kill -9 "$p" 2>/dev/null
    done
}

attempt=0
while [ "$attempt" -le "$MAX_RESTART" ]; do
    attempt=$((attempt + 1))
    touch /tmp/v41_mark

    # 全部 attempt 都从同一个起点 resume：起点已过形成期，warmup 恒为 0
    export LEG_ASYM_WARMUP_STEPS=0
    echo "[$(date '+%H:%M:%S')] === attempt $attempt: RESUME $SRC_RUN/$SRC_CKPT WARMUP=0 ==="
    # 只传 --resume --load_run（与 setup/08_train_resume.sh 验证过的写法一致）。
    # 不传 --load_checkpoint：它是 hydra config 字段(默认正则 model_.*.pt)，
    # 不是本 train.py 的 CLI 参数；get_checkpoint_path 会自动选该 run 下最新的
    # model_1599.pt（目录里最大编号就是 1599，无歧义）。
    nohup "$PYTHON" scripts/rsl_rl/train.py \
        --task "$TASK" --headless --num_envs "$NUM_ENVS" \
        --max_iterations "$MAX_ITER" --seed "$SEED" \
        --resume --load_run "$SRC_RUN" \
        >> "$TRAIN_LOG" 2>&1 &
    TPID=$!
    disown $TPID 2>/dev/null
    echo "[$(date '+%H:%M:%S')] train pid=$TPID"

    # 等新 run 目录（resume 会新建 log_dir，ckpt 存这里）
    NEWRUN=""
    for _ in $(seq 1 60); do
        sleep 5
        NEWRUN=$(find "$RUNROOT" -maxdepth 1 -mindepth 1 -type d -newer /tmp/v41_mark 2>/dev/null \
            | grep -v "_FAKE_" | head -1)
        [ -n "$NEWRUN" ] && break
    done
    [ -n "$NEWRUN" ] || { echo "!! 5min 内没等到新 run 目录"; tail -25 "$TRAIN_LOG"; break; }
    echo "$NEWRUN" > /data/progress/v41_run_dir.txt
    echo "[$(date '+%H:%M:%S')] 新 run 目录 = $NEWRUN"

    # ---- 守护循环 ----
    # 注意 amp_runner 的 for 循环是 0-indexed，最后一个 it = total_it-1 = 2999，
    # 日志分子可能永远到不了 3000。达标阈值用 TARGET_TOTAL-5 容忍这个 off-by-one。
    DONE_AT=$((TARGET_TOTAL - 5))
    killed_by_us=0
    last_n=$(grep -c "Learning iteration" "$TRAIN_LOG" 2>/dev/null || echo 0)
    stall=0
    while kill -0 $TPID 2>/dev/null; do
        sleep 30
        n=$(grep -c "Learning iteration" "$TRAIN_LOG" 2>/dev/null || echo 0)
        if [ "$n" = "$last_n" ]; then stall=$((stall + 30)); else stall=0; last_n=$n; fi
        if [ "$stall" -ge "$STALL_SEC" ]; then
            echo "[$(date '+%H:%M:%S')] !! 卡死 ${stall}s 无新 iter -> kill -9"
            kill -9 "$TPID" 2>/dev/null; kill_train; sleep 5; killed_by_us=1; break
        fi
        li=$(last_iter)
        if [ -n "$li" ] && [ "$li" -ge "$DONE_AT" ]; then break; fi
    done

    li=$(last_iter)
    echo "[$(date '+%H:%M:%S')] attempt $attempt 结束，最后 iter=${li:-?}/$TARGET_TOTAL (killed_by_us=$killed_by_us)"
    if [ -n "$li" ] && [ "$li" -ge "$DONE_AT" ]; then
        echo "[$(date '+%H:%M:%S')] 已达目标 -> kill 训练进程后退出（修 39 只 break 不 kill 的 bug）"
        kill -9 "$TPID" 2>/dev/null; kill_train
        break
    fi
    # 进程是自己退出的（不是我们 kill 的）：若已接近目标，视为正常跑完，不要 resume；
    # 否则才当成崩溃，进入下一次 resume。避免"跑完了又续一段"。
    if [ "$killed_by_us" -eq 0 ] && [ -n "$li" ] && [ "$li" -ge "$((TARGET_TOTAL - 100))" ]; then
        echo "[$(date '+%H:%M:%S')] 训练自然退出且已接近目标，视为完成，不 resume"
        break
    fi
    echo "[$(date '+%H:%M:%S')] 未达标且非正常完成，准备 resume（WARMUP 恒为 0）"
    sleep 5
done

echo "=== 41 RESUME DONE $(date '+%m-%d %H:%M:%S')  last_iter=$(last_iter) ==="
echo "=== run dir: $(cat /data/progress/v41_run_dir.txt 2>/dev/null) ==="
