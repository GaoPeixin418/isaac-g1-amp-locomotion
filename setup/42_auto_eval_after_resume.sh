#!/usr/bin/env bash
# =============================================================================
# 42 - 续训结束后自动重测（11 档密集速度扫描）
#                  2026-11-18 开机窗口，17:21 自动关机
#
# 为什么要有这个：续训约 16:40 才跑满 3000，而机器 17:21 自动关机。
#   如果等人工发现"训练结束了"再手动起评测，任何调度偏差都可能让这次
#   同训练量对比拿不到数据。挂一个后台守护，训练一结束自动出 npz，最稳。
#
# 等什么信号（两个都满足才动，避免和 watchdog 的 kill 抢 GPU）：
#   ① launch_41.log 出现 "41 RESUME DONE"
#   ② 没有 rsl_rl/train.py 进程（GPU 确实释放了）
#
# 取哪个 ckpt：新 run 目录里编号最大的 model_*.pt。
#   注意 amp_runner 只在 it%200==0 存盘(1600..2800)，循环结束才补存末次
#   model_2999.pt。41 的 watchdog 在 iter2995 达标就 kill -9，可能截断末次存档，
#   因此这里自适应取"最大编号"，并在日志里写明最终用的是哪个，不硬编码 2999。
#
# 用法：  nohup bash /data/setup/42_auto_eval_after_resume.sh \
#             > /data/logs/auto_eval_42.log 2>&1 &
# =============================================================================
set -u

RUN="${RUN:-2026-11-18_15-40-01}"
RUNROOT=/data/legged_lab/logs/rsl_rl/g1_amp
DONE_LOG=/data/logs/launch_41.log
TAG="${TAG:-v23_resumed_sweep11}"
VELS="${VELS:-0.5,0.75,1.0,1.25,1.5,1.75,2.0,2.25,2.5,2.75,3.0}"
MAX_WAIT_MIN="${MAX_WAIT_MIN:-80}"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

log "===== 等待续训结束 (最多 ${MAX_WAIT_MIN}min) ====="
log "  run=$RUN  done_log=$DONE_LOG"
waited=0
while [ "$waited" -lt $((MAX_WAIT_MIN * 60)) ]; do
    done_flag=0
    grep -q "41 RESUME DONE" "$DONE_LOG" 2>/dev/null && done_flag=1
    nproc=$(ps aux | grep "rsl_rl/[t]rain\.py" | grep -v grep | wc -l)
    if [ "$done_flag" = "1" ] && [ "$nproc" = "0" ]; then
        log "训练结束信号已确认（DONE=1, train进程=0）"
        break
    fi
    sleep 30
    waited=$((waited + 30))
done
if [ "$waited" -ge $((MAX_WAIT_MIN * 60)) ]; then
    log "!! 超时 ${MAX_WAIT_MIN}min 未等到结束信号，放弃自动评测"
    tail -10 "$DONE_LOG"
    exit 1
fi
# 额外给 GPU/文件系统 10 秒缓冲
sleep 10

# ---- 取最大编号 ckpt ----
CKPT=$(ls -1 "$RUNROOT/$RUN"/model_*.pt 2>/dev/null | sort -V | tail -1)
if [ -z "$CKPT" ]; then
    log "!! $RUNROOT/$RUN 下没有任何 ckpt"; exit 1
fi
CKPT_NAME=$(basename "$CKPT")
log "最终评测 ckpt = $CKPT_NAME"
ls -la "$RUNROOT/$RUN"/model_*.pt | sort -V | tail -8

# ---- 跑 11 档扫描（与 1599 那份完全同参数，可直接横比）----
log "===== 启动 11 档评测 tag=$TAG ====="
RUN="$RUN" CKPT="$CKPT_NAME" TAG="$TAG" VELS="$VELS" STEPS=1500 \
    bash /data/setup/10_eval_gait.sh

NPZ=/data/eval_out/gait_${TAG}.npz
if [ -f "$NPZ" ]; then
    log "OK 产出 $NPZ ($(du -h "$NPZ" | cut -f1))"
else
    log "!! 未产出 $NPZ"
fi
echo "$CKPT_NAME" > /data/progress/v42_ckpt_used.txt
log "===== 42 AUTO EVAL DONE ====="
echo "=== AUTO EVAL 42 DONE $(date '+%m-%d %H:%M:%S') ckpt=$CKPT_NAME ==="
