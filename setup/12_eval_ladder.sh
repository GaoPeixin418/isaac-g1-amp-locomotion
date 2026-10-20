#!/usr/bin/env bash
# =============================================================================
# 12 - checkpoint 阶梯评估
#
# 为什么补这一组：
#   只比 model_2000 和 model_4999 两个点，得到的是"互有胜负"，没法下结论。
#   但 Kit 缓存热了之后每组评估只要 ~2 分钟，成本极低 —— 那就把
#   1000/3000/4000 也评了，把"步态指标随训练怎么变"画成一条曲线。
#
#   这能直接回答两个问题：
#     1) 风格分从 0.797 掉到 0.602，是真的变差了，还是判别器变严了？
#        -> 若步态指标同期持平/变好，说明掉的是"判别器的尺子"，不是动作本身
#     2) 该用哪个 checkpoint 交付？
#        -> 看指标在哪一轮进入平台期
#
# 用法：
#   nohup bash /data/setup/12_eval_ladder.sh > /data/logs/eval_ladder.log 2>&1 &
# =============================================================================
set -uo pipefail

export CONDA_PREFIX=/data/envs/isaaclab
export ISAACLAB_PATH=/data/IsaacLab
export OMNI_KIT_ACCEPT_EULA=YES

AMP_RUN=${AMP_RUN:-2026-10-19_16-37-21}
log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

for CK in model_1000.pt model_3000.pt model_4000.pt; do
    TAG=amp_${CK%.pt}
    log "===== ladder: $TAG ====="
    RUN=$AMP_RUN CKPT=$CK TAG=$TAG bash /data/setup/10_eval_gait.sh
done

log "===== LADDER DONE ====="
echo "=== LADDER DONE $(date '+%m-%d %H:%M:%S') ==="
ls -la /data/eval_out/
