#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
=============================================================================
监控器：盯住正在跑的训练，每 EVERY 个 iter 自动测一次"真实"出力不对称

为什么需要它
-----------------------------------------------------------------------------
训练日志里 rsl_rl 会打印 Episode_Reward/leg_torque_asymmetry，那是**奖励项本身**
的数值，不是我们真正关心的指标。两者必须互相验证：
  * 奖励项数值低 但 扭矩 AI 高  -> 奖励项没起作用（权重太小/被 exploit）
  * 奖励项数值低 且 扭矩 AI 低  -> 真治好了
只信其中一个就是 10-26 那次"指标说满分、肉眼说跛行"的翻版。

它做什么
-----------------------------------------------------------------------------
1. 扫描最新 run 目录里的 model_*.pt
2. 对 iter 为 EVERY 整数倍、且还没测过的 ckpt，跑一次无渲染 eval（1 env / 1.0 m/s）
3. 用与 scripts/analyze_leg_effort.py **完全相同**的口径算扭矩不对称
   （rms = sqrt(mean(t^2)), 腿合力 = sqrt(mean(rms[0:6]^2)), AI = 200|L-R|/(L+R)）
4. 追加写 /data/progress/limp_curve.csv

口径一致性是硬要求：跨 ckpt 比较时算法必须一模一样，否则趋势是假的。

用法
-----------------------------------------------------------------------------
    nohup /data/envs/isaaclab/bin/python watch_limp.py > /data/logs/watch_limp.log 2>&1 &
    # 只跑一轮就退出（调试用）:
    /data/envs/isaaclab/bin/python watch_limp.py --once
"""
import glob
import os
import re
import subprocess
import sys
import time

EVERY = int(os.environ.get("WATCH_EVERY", "400"))
SLEEP = int(os.environ.get("WATCH_SLEEP", "300"))
CSV = "/data/progress/limp_curve.csv"
RUN_ROOT = "/data/legged_lab/logs/rsl_rl/g1_amp"
EVAL = "/data/setup/eval_gait.py"
EVAL_OUT = "/data/eval_out"
LOG = "/data/logs/watch_limp.log"

LEG_L = slice(0, 6)
LEG_R = slice(6, 12)


def latest_run():
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
            parts = line.strip().split(",")
            if parts and parts[0].isdigit():
                s.add(int(parts[0]))
    return s


def newest_npz(after_ts):
    cands = [p for p in glob.glob(os.path.join(EVAL_OUT, "gait_*.npz"))
             if os.path.getmtime(p) > after_ts - 5]
    return max(cands, key=os.path.getmtime) if cands else None


def measure(npz):
    """算扭矩不对称 + 速度跟踪。

    速度这一列是硬要求：iter 400 那次量出 13.3% 看着很好，结果实测速度只有
    0.06 m/s —— 它根本没在走。站着不动当然左右对称。没有速度列，这条曲线
    会把"还没学会走路"误报成"治好了"。
    """
    import numpy as np
    d = np.load(npz, allow_pickle=True)
    tq = d["torque"]
    T, N, D = tq.shape
    if D < 12:
        return None
    n0 = int(T * 0.3)
    rows = []
    for j in range(N):
        t = tq[n0:, j, :]
        rms = np.sqrt(np.mean(t ** 2, axis=0))
        legL = float(np.sqrt(np.mean(rms[LEG_L] ** 2)))
        legR = float(np.sqrt(np.mean(rms[LEG_R] ** 2)))
        kneeL, kneeR = float(rms[3]), float(rms[9])
        ai = lambda a, b: abs(a - b) / (0.5 * (abs(a) + abs(b))) * 100 if (abs(a) + abs(b)) > 1e-9 else float("nan")
        rows.append((legL, legR, ai(legL, legR), kneeL, kneeR, ai(kneeL, kneeR)))
    legL = sum(r[0] for r in rows) / len(rows)
    legR = sum(r[1] for r in rows) / len(rows)
    legai = sum(r[2] for r in rows) / len(rows)
    kneeai = sum(r[5] for r in rows) / len(rows)

    vcmd = float(np.mean(d["velocities"]))
    vact = float(np.mean([
        np.mean(np.hypot(d["lin_vel_x"][n0:, j], d["lin_vel_y"][n0:, j]))
        for j in range(d["lin_vel_x"].shape[1])
    ]))
    return legL, legR, legai, kneeai, vcmd, vact


def main():
    once = "--once" in sys.argv
    os.makedirs("/data/progress", exist_ok=True)
    HEADER = "iter,time,leg_ai_pct,knee_ai_pct,rms_L,rms_R,rms_mean,v_cmd,v_act,ckpt\n"
    need_hdr = True
    if os.path.exists(CSV):
        with open(CSV, encoding="utf-8") as f:
            need_hdr = not f.readline().startswith("iter,time,leg_ai_pct")
    if need_hdr:
        with open(CSV, "w", encoding="utf-8") as f:
            f.write(HEADER)

    print("[watch] start  run=%s  every=%d  sleep=%d" % (latest_run(), EVERY, SLEEP), flush=True)
    while True:
        run = latest_run()
        if run is None:
            print("[watch] 没有 run 目录，等待", flush=True)
            time.sleep(SLEEP)
            continue
        cks = ckpt_iters(run)
        done = done_iters()
        todo = sorted(i for i in cks if i >= EVERY and i % EVERY == 0 and i not in done)
        if not todo:
            print("[watch] %s 暂无新 ckpt（已有 %d 个，最新 %s）" %
                  (time.strftime("%H:%M:%S"), len(cks), max(cks) if cks else "-"), flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue

        it = todo[0]
        ck = cks[it]
        print("[watch] %s 评估 iter=%d  %s" % (time.strftime("%H:%M:%S"), it, ck), flush=True)
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
            tail = (p.stdout or "")[-800:] + (p.stderr or "")[-800:]
            print("[watch] eval 失败 rc=%d\n%s" % (p.returncode, tail), flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue

        npz = newest_npz(ts)
        if npz is None:
            print("[watch] 没找到新 npz", flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue

        m = measure(npz)
        if m is None:
            print("[watch] npz 数据不足", flush=True)
            if once:
                return
            time.sleep(SLEEP)
            continue

        legL, legR, legai, kneeai, vcmd, vact = m
        with open(CSV, "a", encoding="utf-8") as f:
            f.write("%d,%s,%.1f,%.1f,%.1f,%.1f,%.1f,%.2f,%.2f,%s\n" % (
                it, time.strftime("%m-%d %H:%M"), legai, kneeai,
                legL, legR, 0.5 * (legL + legR), vcmd, vact, os.path.basename(ck)))
        flag = "  <<< 没在走，AI 无效" if vact < 0.4 * max(vcmd, 1e-6) else ""
        print("[watch] iter=%d  legAI=%.1f%%  kneeAI=%.1f%%  rms %.1f/%.1f  "
              "v_cmd=%.2f v_act=%.2f%s" % (it, legai, kneeai, legL, legR, vcmd, vact, flag),
              flush=True)

        if once:
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
