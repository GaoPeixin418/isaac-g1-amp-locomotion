#!/usr/bin/env bash
# =============================================================================
# 38_run_ctrl_v22.sh —— 对照组：纯 v2.2 配置（WARMUP=0）  2026-11-11
#
# 为什么先跑这一组
#   v2.2 历史 run（2026-10-27_23-56-50）只跑到 iter 84/2400 就断电，
#   v22_curve.csv 只有表头 = **从未产生任何评估点**。
#   所以"v2.2 这套配置到底能不能学会走路"这件事至今**没有实测证据**，
#   有的只是 v2（work_asym -2.0）在 iter800 被判死、v2.1 在 iter1200 仍 0.07。
#   本组把 WARMUP 设 0（退化成 v2.2），从零训到 1600，给它一个公平机会。
#
# 与 v2.3 的关系（同一个源码，只差一个环境变量）
#   WARMUP=0     -> 本脚本，纯 v2.2
#   WARMUP=12000 -> setup/37 默认，v2.3 带 warmup
#   两组可直接对比，因为 weight(-1.0)/tc(1.5)/angvel(-0.05)/airtime(0.4) 全部一致。
#
# 为什么 MAX_ITER=1600 而不是 2400/3000
#   v1 的相变发生在 iter 400→800（v_act 0.06 → 0.97），不是缓慢爬升。
#   1600 给 4 个评估点（400/800/1200/1600），既覆盖相变窗口，
#   又给"晚熟"留了 2 倍余量；判死时约 1h45m 收手，不占掉整天。
#
# 判定阈值（**预注册**：启动前写死，不允许事后换指标）
#   iter  400  v_act >= 0.20  -> 有戏（v1 同点仅 0.06，此时还不必紧张）
#   iter  800  v_act >= 0.70  -> 起飞（v1 同点 0.97）
#   iter 1200  v_act <  0.20  -> 判死；never-walk 早停会自动杀掉
#   iter 1600  走满仍 <0.20   -> 证伪"v2.2 配置可训" => v2.3 warmup 才有意义
#
# 用法
#   bash 38_run_ctrl_v22.sh            # 启动（后台，约 5 秒返回）
#   bash 38_run_ctrl_v22.sh --check    # 只做前置校验，不启动（无卡模式也能跑）
#   MAX_ITER=2400 bash 38_run_ctrl_v22.sh   # 想给更多时间就改这个
# =============================================================================
set -u

LAB=/data/legged_lab
PYTHON=/data/envs/isaaclab/bin/python
RUNROOT=$LAB/logs/rsl_rl/g1_amp

MAX_ITER="${MAX_ITER:-1600}"
NUM_ENVS="${NUM_ENVS:-4096}"
SEED="${SEED:-42}"
WATCH_EVERY="${WATCH_EVERY:-400}"      # 必须是 save_interval(200) 的倍数
NEVER_WALK_ITER="${NEVER_WALK_ITER:-1200}"

TRAIN_LOG="${TRAIN_LOG:-/data/logs/train_ctrl_v22.log}"
WATCH_LOG="${WATCH_LOG:-/data/logs/watch_ctrl_v22.log}"
CSV="${CSV:-/data/progress/ctrl_v22_curve.csv}"

CHECK_ONLY=0
[ "${1:-}" = "--check" ] && CHECK_ONLY=1

fail() { echo "!! $1"; exit 1; }

echo "=================================================================="
echo " 启动器（实际参数以 WARMUP 值为准）"
echo " WARMUP=$WARMUP  MAX_ITER=$MAX_ITER  ENVS=$NUM_ENVS  SEED=$SEED"
echo "   WARMUP=0     -> 纯 v2.2 对照组"
echo "   WARMUP=12000 -> v2.3（500 iter 斜坡）"
echo " 训练日志 $TRAIN_LOG"
echo " 监控日志 $WATCH_LOG"
echo " 曲线     $CSV"
echo "=================================================================="

# --- 前置 1：不能有别的训练进程（10-27 踩过：双进程抢 GPU、日志混写）---
if ps aux | grep -E "rsl_rl/[t]rain\.py" | grep -v grep >/dev/null; then
    echo "!! 已有训练进程在跑："
    ps aux | grep -E "rsl_rl/[t]rain\.py" | grep -v grep
    fail "先确认并杀掉旧进程，再启动"
fi
echo "[1/5] 无残留训练进程          OK"

# --- 前置 2：工具链在位 ---
for f in /data/setup/37_train_fresh_v23.sh /data/setup/watch_v2.py \
         /data/setup/eval_gait.py /data/setup/gait_metrics2.py; do
    [ -f "$f" ] || fail "缺少 $f"
done
echo "[2/5] 工具链齐全              OK"

# --- 前置 3：v2.2 数值（若这里不对，对照组就名不副实）---
CFG="$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py"
grep -q "leg_work_asymmetry" "$CFG" || fail "leg_work_asymmetry 不在位"
"$PYTHON" - <<'PY' || fail "work_asym 权重不是 -1.0"
import sys
c = open("/data/legged_lab/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/g1_amp_env_cfg.py", encoding="utf-8").read()
sys.exit(0 if "func=mdp.leg_work_asymmetry,\n        weight=-1.0," in c else 1)
PY
grep -q "ang_vel_xy_l2, weight=-0.05" "$CFG" || fail "ang_vel_xy_l2 不是 -0.05"
LERP=$(grep -o "task_style_lerp=[0-9.]*" \
    "$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/config/g1/agents/rsl_rl_ppo_cfg.py" \
    | head -1 | cut -d= -f2)
[ "$(awk -v v="$LERP" 'BEGIN{printf "%g", v}')" = "0.5" ] || fail "task_style_lerp=$LERP 不是 0.5"
echo "[3/5] v2.2 数值核对            OK (work_asym=-1.0, angvel=-0.05, lerp=0.5)"

# --- 前置 4：warmup 补丁在位（WARMUP=0 时会短路成 1.0，但补丁必须能读到环境变量）---
grep -q "PATCHED-SYM-V23-WARMUP" \
    "$LAB/source/legged_lab/legged_lab/tasks/locomotion/amp/mdp/rewards.py" \
    || fail "warmup 补丁不在位，先跑 patch_sym_v23.py"
echo "[4/5] warmup 补丁在位          OK"

# --- 前置 5：GPU 可见（无卡模式下这里必然失败，正是我们想提前报出来的）---
if ! nvidia-smi -L >/dev/null 2>&1; then
    echo "[5/5] GPU                      !! 不可见"
    fail "当前是无卡模式，Isaac Sim 无法启动。请在控制台切到有卡模式后重跑本脚本"
fi
nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader
echo "[5/5] GPU 可见                 OK"

if [ "$CHECK_ONLY" = "1" ]; then
    echo "=================================================================="
    echo "--check 完成：五道前置全部通过，切卡后去掉 --check 即可启动"
    exit 0
fi

# ============================ 启动 ============================
mkdir -p /data/logs /data/progress
touch /tmp/ctrl_v22_mark

echo "[$(date '+%m-%d %H:%M:%S')] 启动训练（后台）..."
nohup env WARMUP=0 MAX_ITER="$MAX_ITER" NUM_ENVS="$NUM_ENVS" SEED="$SEED" \
    bash /data/setup/37_train_fresh_v23.sh > "$TRAIN_LOG" 2>&1 &
TPID=$!
disown $TPID 2>/dev/null
echo "[$(date '+%m-%d %H:%M:%S')] train pid=$TPID"

# 等新 run 目录出现再启监控：10-27 踩过 —— 监控先起会抓到上一轮的目录，
# 把旧模型数据写进新曲线（latest_run 取 mtime 最新）。
echo "等待新 run 目录出现（Isaac 冷启动可能 1-8 分钟）..."
NEWRUN=""
for i in $(seq 1 180); do
    sleep 5
    if ! kill -0 $TPID 2>/dev/null; then
        echo "!! 训练进程已退出，最后 25 行："
        tail -25 "$TRAIN_LOG"
        fail "训练启动失败，看 $TRAIN_LOG"
    fi
    NEWRUN=$(find "$RUNROOT" -maxdepth 1 -mindepth 1 -type d -newer /tmp/ctrl_v22_mark 2>/dev/null | head -1)
    [ -n "$NEWRUN" ] && break
done
[ -n "$NEWRUN" ] || fail "15 分钟内没等到 run 目录，训练可能卡在 Kit 启动，看 $TRAIN_LOG"
echo "[$(date '+%m-%d %H:%M:%S')] run 目录 = $NEWRUN"

echo "启动监控（每 ${WATCH_EVERY} iter 评估一次，never-walk 阈值 ${NEVER_WALK_ITER}）..."
nohup env WATCH_RUN="$NEWRUN" WATCH_CSV="$CSV" WATCH_EVERY="$WATCH_EVERY" \
    NEVER_WALK_ITER="$NEVER_WALK_ITER" EARLY_MIN_ITER="$NEVER_WALK_ITER" \
    "$PYTHON" -u /data/setup/watch_v2.py > "$WATCH_LOG" 2>&1 &
WPID=$!
disown $WPID 2>/dev/null

echo "=================================================================="
echo " 已启动"
echo "   train pid=$TPID  watch pid=$WPID"
echo "   run 目录 $NEWRUN"
echo ""
echo " 看进度："
echo "   tail -f $TRAIN_LOG | grep -E 'Learning iteration'"
echo "   column -s, -t $CSV"
echo "   tail -20 $WATCH_LOG"
echo ""
echo " 停止："
echo "   pkill -f 'rsl_rl/[t]rain.py'; pkill -f 'watch_[v]2.py'"
echo "=================================================================="
