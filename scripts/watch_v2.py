#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
监控器 v2：盯训练 + 自动早停（2026-10-27 晚）

v1（watch_limp.py）只盯对称性一个量，结果 v1 训练把对称性刷低了、代价是
碎步 + 蹲姿。这一版盯**四个**量，并且任何一个被"作弊式优化"都会在综合分
里被其它项抓住：

  legAI    左右腿出力不对称（逐关节 AI 的均值，绝不用合成值）
  cadence  步频（双腿 steps/min）—— 盯碎步
  knee     膝均屈角（deg）        —— 盯蹲姿
  height   躯干高度（m）          —— 盯蹲姿（策略绕不过去的最终观测量）
  v_act    实际速度               —— 硬门槛，没在走则这一整点作废

为什么要有早停
-----------------------------------------------------------------------------
用户明确要求"衡量训练迭代步数和模型效果的关系，不需要一直训练"。
上一轮实测：对称性 2400(42.6) -> 2800(46.2) -> 2999(48.7)，后期是**变差**的。
多训的 600 iter 不但没用还在倒退。所以这一版在综合分连续 PATIENCE 个评估点
不再改善时，直接把训练进程杀掉并打印最佳 ckpt。

综合分（越低越好，各项都是"相对参考的偏差%"）
-----------------------------------------------------------------------------
  score = legAI
        + |cadence - 100| / 100 * 100
        + |knee - 15|    / 15   * 100
        + |height - 0.765| / 0.765 * 100
  （参考值都来自 ref_height_stats.py / gait_metrics2.py 的实测，不是拍脑袋）

用法
-----------------------------------------------------------------------------
    WATCH_EVERY=300 EARLY_MIN_ITER=1500 EARLY_PATIENCE=3 \
      nohup /data/envs/isaaclab/bin/python -u /data/setup/watch_v2.py \
      > /data/logs/watch_v2.log 2>&1 &
"""
import glob
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, "/data/setup")
from gait_metrics2 import gait_from_footz, joint_stats  # noqa: E402
import numpy as np  # noqa: E402

EVERY = int(os.environ.get("WATCH_EVERY", "300"))
SLEEP = int(os.environ.get("WATCH_SLEEP", "120"))
CSV = os.environ.get("WATCH_CSV", "/data/progress/v2_curve.csv")
RUN_ROOT = "/data/legged_lab/logs/rsl_rl/g1_amp"
EVAL = "/data/setup/eval_gait.py"
EVAL_OUT = "/data/eval_out"

# 参考值（0.8-1.3 m/s 档实测）
REF_CADENCE = 100.0
REF_KNEE_DEG = 15.0
REF_HEIGHT = 0.765

# 早停
EARLY_MIN_ITER = int(os.environ.get("EARLY_MIN_ITER", "1500"))
EARLY_PATIENCE = int(os.environ.get("EARLY_PATIENCE", "3"))
EARLY_MARGIN = float(os.environ.get("EARLY_MARGIN", "0.98"))  # 需改善 >2% 才算改善

HEADER = ("iter,time,leg_ai,hip_ai,knee_ai,cadence,stride_m,knee_deg,height_m,"
          "v_cmd,v_act,score,best_score,ckpt\n")


def latest_run():
    # WATCH_RUN 必须支持显式指定：10-27 踩过 —— 监控在训练刚启动、新 run 目录
    # 还没有 model_*.pt 时启动，latest_run() 会抓到上一轮的目录，把旧模型的数据
    # 写进新曲线里。锁死目录就不会串台。
    forced = os.environ.get("WATCH_RUN", "").strip()
    if forced:
        return forced
    ds = sorted(glob.glob(os.path.join(RUN_ROOT, "*")), key=os.path.getmtime)
    return ds[-1] if ds else None


def ckpt_iters(run):
    out = {}
    for p in glob.glob(os.path.join(run, "model_*.pt")):
        m = re.search(r"model_(\d+)\.pt$", p)
        if m:
            out[int(m.group(1))] = p
    return out


def done_iters():
    if not os.path.exists(CSV):
        return set()
    s = set()
    with open(CSV, encoding="utf-8") as f:
        for line in f:
            p = line.strip().split(",")
            if p and p[0].isdigit():
                s.add(int(p[0]))
    return s


def newest_npz(after_ts):
    c = [p for p in glob.glob(os.path.join(EVAL_OUT, "gait_*.npz"))
         if os.path.getmtime(p) > after_ts - 5]
    return max(c, key=os.path.getmtime) if c else None


def measure(npz):
    """返回 dict，或 None。口径与 gait_metrics2 完全一致。"""
    d = np.load(npz, allow_pickle=True)
    tq = d["torque"]
    T, N, D = tq.shape
    if D < 12:
        return None
    n0 = int(T * 0.3)
    dt = float(d["dt"])
    fps = 1.0 / dt

    # --- 对称性：逐关节 AI 的均值（合成值会被反向不对称抵消，见 MEMORY 3.7）---
    rms = np.sqrt(np.mean(tq[n0:] ** 2, axis=(0, 1)))
    per = []
    for k in range(6):
        a, b = float(rms[k]), float(rms[k + 6])
        den = 0.5 * (abs(a) + abs(b))
        per.append(abs(a - b) / den * 100 if den > 1e-9 else float("nan"))
    leg_ai = float(np.mean(per))

    # --- 速度（水平速率，不用 x 分量：转弯/侧移会抵消）---
    v_cmd = float(np.mean(d["velocities"]))
    v_act = float(np.mean(np.hypot(d["lin_vel_x"][n0:], d["lin_vel_y"][n0:])))

    # --- 步态尺度 ---
    fz = d["foot_z"][n0:]
    r = gait_from_footz(fz.reshape(-1, N * fz.shape[2]), fps, v_act) or {}
    cad = float(r.get("cadence", float("nan")))
    stride = float(r.get("stride_m", float("nan")))

    # --- 姿态 ---
    knee_deg = float("nan")
    if "joint_pos" in d.files:
        js = joint_stats(np.degrees(d["joint_pos"][n0:]))
        knee_deg = float(js["knee_mean"])
    height = float(np.mean(d["root_pos_z"][n0:])) if "root_pos_z" in d.files else float("nan")

    return {"leg_ai": leg_ai, "hip_ai": float(per[0]), "knee_ai": float(per[3]),
            "cadence": cad, "stride": stride, "knee_deg": knee_deg,
            "height": height, "v_cmd": v_cmd, "v_act": v_act}


def score_of(m):
    s = m["leg_ai"]
    if np.isfinite(m["cadence"]):
        s += abs(m["cadence"] - REF_CADENCE) / REF_CADENCE * 100
    if np.isfinite(m["knee_deg"]):
        s += abs(m["knee_deg"] - REF_KNEE_DEG) / REF_KNEE_DEG * 100
    if np.isfinite(m["height"]):
        s += abs(m["height"] - REF_HEIGHT) / REF_HEIGHT * 100
    return s


def kill_training():
    """杀掉训练进程。正则写成 [t]rain 是为了不匹配执行命令的 shell 自己
    （10-27 踩过：pkill -f watch_limp.py 把执行它的 shell 一起杀了，SSH 断连）。"""
    p = subprocess.run(["bash", "-lc", "pkill -f 'rsl_rl/[t]rain.py'; echo killed=$?"],
                       capture_output=True, text=True)
    print("[watch] 早停：已发 kill -> %s %s" % (p.stdout.strip(), p.stderr.strip()), flush=True)


def main():
    once = "--once" in sys.argv
    os.makedirs(os.path.dirname(CSV), exist_ok=True)
    if not os.path.exists(CSV):
        with open(CSV, "w", encoding="utf-8") as f:
            f.write(HEADER)

    best_score, best_iter = float("inf"), -1
    no_improve = 0
    print("[watch-v2] start every=%d min_iter=%d patience=%d"
          % (EVERY, EARLY_MIN_ITER, EARLY_PATIENCE), flush=True)

    while True:
        run = latest_run()
        if run is None:
            print("[watch-v2] 没有 run 目录，等待", flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue
        cks = ckpt_iters(run)
        done = done_iters()
        todo = sorted(i for i in cks if i >= EVERY and i % EVERY == 0 and i not in done)
        if not todo:
            print("[watch-v2] %s 暂无新 ckpt（最新 %s，已测 %d）"
                  % (time.strftime("%H:%M:%S"), max(cks) if cks else "-", len(done)), flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue

        it = todo[0]
        ck = cks[it]
        print("[watch-v2] %s 评估 iter=%d" % (time.strftime("%H:%M:%S"), it), flush=True)
        ts = time.time()
        p = subprocess.run(
            ["/data/envs/isaaclab/bin/python", EVAL,
             "--task", "LeggedLab-Isaac-AMP-G1-v0",
             "--checkpoint", ck,
             "--num_envs", "1", "--steps", "600",
             "--velocities", "1.0", "--headless"],
            capture_output=True, text=True, cwd="/data/legged_lab",
        )
        if p.returncode != 0:
            print("[watch-v2] eval 失败 rc=%d\n%s"
                  % (p.returncode, ((p.stdout or "")[-600:] + (p.stderr or "")[-600:])), flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue

        npz = newest_npz(ts)
        if npz is None:
            print("[watch-v2] 没找到新 npz", flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue

        m = measure(npz)
        if m is None:
            if once:
                return
            time.sleep(SLEEP)
            continue

        # 硬门槛：没在走，这一整点作废（iter400 那次 13.3% 是站着不动的假象）
        walking = m["v_act"] >= 0.7 * max(m["v_cmd"], 1e-6)
        sc = score_of(m) if walking else float("inf")

        if walking and sc < best_score:
            improved = sc < best_score * EARLY_MARGIN
            if improved:
                no_improve = 0
            else:
                no_improve += 1
            if sc < best_score:
                best_score, best_iter = sc, it
        elif walking:
            no_improve += 1

        with open(CSV, "a", encoding="utf-8") as f:
            f.write("%d,%s,%.1f,%.1f,%.1f,%.1f,%.3f,%.1f,%.3f,%.2f,%.2f,%s,%s,%s\n" % (
                it, time.strftime("%m-%d %H:%M"), m["leg_ai"], m["hip_ai"], m["knee_ai"],
                m["cadence"], m["stride"], m["knee_deg"], m["height"],
                m["v_cmd"], m["v_act"],
                ("%.1f" % sc) if np.isfinite(sc) else "NA",
                ("%.1f" % best_score) if np.isfinite(best_score) else "NA",
                os.path.basename(ck)))

        flag = "" if walking else "  <<< 没在走，本点作废"
        print("[watch-v2] iter=%d legAI=%.1f%% cad=%.0f stride=%.2f knee=%.1f h=%.3f "
              "v_act=%.2f score=%s best=%s(it%d) stall=%d%s"
              % (it, m["leg_ai"], m["cadence"], m["stride"], m["knee_deg"], m["height"],
                 m["v_act"], ("%.1f" % sc) if np.isfinite(sc) else "NA",
                 ("%.1f" % best_score) if np.isfinite(best_score) else "NA",
                 best_iter, no_improve, flag), flush=True)

        if walking and it >= EARLY_MIN_ITER and no_improve >= EARLY_PATIENCE:
            print("[watch-v2] 连续 %d 个评估点未改善（最佳 iter=%d score=%.1f）-> 早停"
                  % (no_improve, best_iter, best_score), flush=True)
            kill_training()
            print("[watch-v2] EARLY STOP AT iter=%d  best=%d" % (it, best_iter), flush=True)
            return

        if once:
            return
        time.sleep(20)


if __name__ == "__main__":
    main()
