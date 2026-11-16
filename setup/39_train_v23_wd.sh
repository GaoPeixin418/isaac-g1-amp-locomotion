#!/usr/bin/env bash
# =============================================================================
# 39_train_v23_wd.sh —— v2.3 + 崩溃自动续训（watchdog）  2026-11-16
#
# 为什么要 watchdog
#   2026-11-11 的真 v2.3 首跑在 iter 217 崩了，原因是 Isaac Sim 5.1 内部断言：
#     carb.tasking/Mutex.cpp(103): Assertion (m_recursive) failed: Recursion not allowed
#   这是 Kit 任务调度器的竞态 bug，不是我们的配置错误，也不是 OOM
#   （OOM 会先报 CUDA error，且当日另一轮同配置跑满 1200 iter 无事）。
#   => 重跑可能再崩，必须能自动接上，否则一小时又白烧。
#
# 两个关键设计
#   1. **卡死判定不靠退出码**：那个断言对话框会弹 "Press ABORT/RETRY/IGNORE"
#      阻塞等 stdin（我们的 stdin 是 `yes Yes`，对不上选项），进程可能永远不退。
#      所以用「日志连续 STALL_SEC 秒没有新的 Learning iteration」判活埋，然后 kill -9。
#   2. **resume 时 WARMUP 置 0**：`env._v23_reward_calls` 是挂在 env 上的计数器，
#      进程重启后从 0 重来。若仍传 12000，会重新爬 500 iter 斜坡，
#      相当于把已过形成期的策略又松绑一次（与实验意图相反）。
#      resume 一律 LEG_ASYM_WARMUP_STEPS=0（满额），并在日志记明。
#
# 时间预算（2026-11-16 用户开机到 19:00）
#   1600 iter × 3.01 s ≈ 80 min + Kit 启动 1-2 min ≈ 85 min
#   判定点 iter 400 / 800 / 1200 / 1600
#
# 用法:  bash 39_train_v23_wd.sh
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
MAX_ITER="${MAX_ITER:-1600}"
SEED="${SEED:-42}"
WARMUP="${WARMUP:-12000}"
STALL_SEC="${STALL_SEC:-300}"      # 5 分钟无新 iter = 卡死
MAX_RESTART="${MAX_RESTART:-4}"

TRAIN_LOG="${TRAIN_LOG:-/data/logs/train_v23_wd.log}"
WATCH_LOG="${WATCH_LOG:-/data/logs/watch_v23_wd.log}"
CSV="${CSV:-/data/progress/v23_wd_curve.csv}"
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
[ -f "$ENV_CFG" ] || fail "cfg 不存在"
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
echo " v2.3 + watchdog   WARMUP=$WARMUP  MAX_ITER=$MAX_ITER  ENVS=$NUM_ENVS  SEED=$SEED"
echo " 训练日志 $TRAIN_LOG"
echo " 监控日志 $WATCH_LOG"
echo " 曲线     $CSV"
echo " 卡死判定 ${STALL_SEC}s 无新 iter；最多重启 ${MAX_RESTART} 次"
echo "=================================================================="

mkdir -p /data/logs /data/progress
# 监控日志每次清空：下面要靠它判断「监控是否已判死」，旧记录会造成误判
: > "$WATCH_LOG"
: > "$TRAIN_LOG"

# ---------------------------- 工具函数 ----------------------------
last_iter() {
    grep -o "Learning iteration [0-9]*" "$TRAIN_LOG" 2>/dev/null | tail -1 | grep -o "[0-9]*$"
}

start_watch() {   # $1 = run dir
    local dir="$1"
    if [ -z "$dir" ]; then return; fi
    # 杀掉旧监控（避免两个 watch 抢写同一 CSV）
    for p in $(ps aux | grep "watch_[v]2\.py" | grep -v grep | awk '{print $2}'); do kill -9 "$p"; done
    nohup env WATCH_RUN="$dir" WATCH_CSV="$CSV" WATCH_EVERY=400 \
        NEVER_WALK_ITER=1200 EARLY_MIN_ITER=1200 \
        "$PYTHON" -u /data/setup/watch_v2.py >> "$WATCH_LOG" 2>&1 &
    echo "[$(date '+%H:%M:%S')] 监控 -> $dir (pid=$!)"
}

wait_run_dir() {  # $1 = mark file
    local mark="$1" d=""
    for _ in $(seq 1 60); do
        sleep 5
        d=$(find "$RUNROOT" -maxdepth 1 -mindepth 1 -type d -newer "$mark" 2>/dev/null \
            | grep -v "_FAKE_" | head -1)
        [ -n "$d" ] && { echo "$d"; return 0; }
    done
    return 1
}

CURRENT_RUN=""
attempt=0

while [ "$attempt" -le "$MAX_RESTART" ]; do
    attempt=$((attempt + 1))
    touch /tmp/v23wd_mark

    if [ "$attempt" -eq 1 ]; then
        MODE=fresh
        RESUME_ARGS=""
        export LEG_ASYM_WARMUP_STEPS="$WARMUP"
        echo "[$(date '+%H:%M:%S')] === attempt $attempt: FRESH  WARMUP=$WARMUP ==="
        nohup "$PYTHON" scripts/rsl_rl/train.py \
            --task "$TASK" --headless --num_envs "$NUM_ENVS" \
            --max_iterations "$MAX_ITER" --seed "$SEED" \
            >> "$TRAIN_LOG" 2>&1 &
    else
        # resume：从最新 run 的最新 ckpt 接上；warmup 满额（见头部说明 2）
        [ -n "$CURRENT_RUN" ] || fail "没有可 resume 的 run 目录"
        export LEG_ASYM_WARMUP_STEPS=0
        echo "[$(date '+%H:%M:%S')] === attempt $attempt: RESUME from $CURRENT_RUN (WARMUP=0 满额) ==="
        nohup "$PYTHON" scripts/rsl_rl/train.py \
            --task "$TASK" --headless --num_envs "$NUM_ENVS" \
            --max_iterations "$MAX_ITER" \
            --resume --load_run "$(basename "$CURRENT_RUN")" \
            >> "$TRAIN_LOG" 2>&1 &
    fi
    TPID=$!
    disown $TPID 2>/dev/null
    echo "[$(date '+%H:%M:%S')] train pid=$TPID"

    NEWRUN=$(wait_run_dir /tmp/v23wd_mark) || {
        echo "!! 60x5s 内没等到 run 目录，训练可能起不来"
        tail -20 "$TRAIN_LOG"
        break
    }
    CURRENT_RUN="$NEWRUN"
    echo "[$(date '+%H:%M:%S')] run 目录 = $CURRENT_RUN"
    start_watch "$CURRENT_RUN"

    # ---- 守护循环：判卡死 ----
    last_n=$(grep -c "Learning iteration" "$TRAIN_LOG" 2>/dev/null || echo 0)
    stall=0
    while kill -0 $TPID 2>/dev/null; do
        sleep 30
        n=$(grep -c "Learning iteration" "$TRAIN_LOG" 2>/dev/null || echo 0)
        if [ "$n" = "$last_n" ]; then
            stall=$((stall + 30))
        else
            stall=0
            last_n=$n
        fi
        if [ "$stall" -ge "$STALL_SEC" ]; then
            echo "[$(date '+%H:%M:%S')] !! 卡死 ${stall}s 无新 iter -> kill -9"
            for p in $TPID $(ps aux | grep "rsl_rl/[t]rain\.py" | grep -v grep | awk '{print $2}'); do
                kill -9 "$p" 2>/dev/null
            done
            sleep 5
            break
        fi
        # 正常达标退出
        li=$(last_iter)
        if [ -n "$li" ] && [ "$li" -ge "$MAX_ITER" ]; then break; fi
    done

    li=$(last_iter)
    echo "[$(date '+%H:%M:%S')] attempt $attempt 结束，最后 iter=${li:-?}/$MAX_ITER"
    if [ -n "$li" ] && [ "$li" -ge "$MAX_ITER" ]; then
        echo "[$(date '+%H:%M:%S')] 已达目标 $MAX_ITER，退出守护"
        break
    fi
    # 监控已判死（never-walked / 综合分早停）就不要 resume，否则会无限循环：
    # 那些情况是「配置学不会」，不是「进程崩了」，重启解决不了。
    if grep -q "NEVER-WALKED STOP\|早停：已发 kill" "$WATCH_LOG" 2>/dev/null; then
        echo "[$(date '+%H:%M:%S')] !! 监控判定该配置学不会走路，停止守护（不 resume）"
        break
    fi
    sleep 5
done

for p in $(ps aux | grep "watch_[v]2\.py" | grep -v grep | awk '{print $2}'); do kill -9 "$p" 2>/dev/null; done
echo "=== V2.3 WATCHDOG DONE $(date '+%m-%d %H:%M:%S')  last_iter=$(last_iter) ==="
