# -*- coding: utf-8 -*-
"""分析 eval_stress.py 产出的 npz：连续斜坡跟踪 + 外力推鲁棒性。

口径（启动前写死，避免事后挑指标）：
  【两个必须先剔除的污染源，2026-11-20 实测踩到】
   ① 启动暂态：step 0 附近策略冷启动前栽，v_act 出现 -1.76 的假值（约前 60 步）
   ② episode timeout：约 step 1000(20s) 全局 done + reset，重置瞬间速度出现尖峰
      （step 1000 的 v=2.10、cmd=0.5 就是这么来的），**不是跌倒也不是掉速**
   → ramp 段只在 [warm, timeout_step) 内统计；这两段之外的数据作废
  【RAMP env0】
   - 跟踪误差 MAE = mean|v_act - cmd|
   - 滞后 lag：使 corr(cmd(t-lag), v_act(t)) 最大的 lag（秒），正值=策略滞后于指令
   - 上下行不对称：上行段(0.5→3.0)与下行段(3.0→0.5)的 MAE 之差
   - 断点：连续 |err|>0.5 的最长持续秒数（掉速/失步）
   - 与 11 档离散扫描的 ρ 对照：连续斜坡 ρ 若接近 0.998 则"任意速度"坐实
  【PUSH env1..6】
   - 跌倒判据**只用躯干高度 z<fall_z**（dones 含 timeout，会虚高，不能用）
   - 每次推动后(period 内)测量：躯干最低高度、速度恢复到 cmd±15% 所需步数、
     侧向速度峰值；按 (目标速度, 冲量强度) 分组报跌倒率与恢复步数

用法：
  python scripts/analyze_stress.py <stress.npz> --csv out.csv
"""
import argparse
import csv
import os

import numpy as np


def longest_true(mask):
    best = cur = 0
    for v in mask:
        cur = cur + 1 if v else 0
        best = max(best, cur)
    return best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    d = np.load(args.npz, allow_pickle=True)
    dt = float(d["dt"])
    T, N = d["lin_vel_x"].shape
    vx = d["lin_vel_x"]        # (T,N) 机体前向速度
    vy = d["lin_vel_y"]
    cmd = d["cmd_x"]
    z = d["root_pos_z"]
    push = d["push_flag"][:, 0].astype(bool)
    dones = d["dones"].astype(bool)
    ramp_up = int(d["ramp_up"]); ramp_dn = int(d["ramp_dn"])
    period = int(d["push_period"])
    fall_z = float(d["fall_z"])
    pvels = d["push_vels"]; pimp = d["push_impulse"]

    print("=== 数据 %s  T=%d N=%d dt=%.3f 时长=%.1fs ===" %
          (os.path.basename(args.npz), T, N, dt, T * dt))

    # 找全局 timeout 步（多数 env 同时 done 的那一步），作为 ramp 统计右边界
    global_done = np.where(dones.sum(axis=1) >= N // 2)[0]
    tmo = int(global_done[0]) if len(global_done) else T
    warm = int(2.5 / dt)   # 丢启动 2.5s 暂态（实测前栽约 1s，留余量）
    print("== 污染剔除 ==")
    print("  启动暂态: 丢弃 step < %d (%.1fs)" % (warm, warm * dt))
    print("  episode timeout: 全局 done 于 step %d (%.1fs)，ramp 只统计 [%d,%d)" %
          (tmo, tmo * dt, warm, tmo))

    # ---------------- RAMP (env 0) ----------------
    c0, v0 = cmd[:, 0], vx[:, 0]
    # 统计窗口：丢启动 + 截掉 timeout reset 尖峰
    cs, vs = c0[warm:tmo], v0[warm:tmo]
    err = vs - cs
    mae = float(np.mean(np.abs(err)))
    rho = float(np.corrcoef(cs, vs)[0, 1])
    # 滞后：cmd 相对 v 平移 lag 步后相关最大
    best_lag, best_r = 0, -2
    for lag in range(0, int(1.0 / dt) + 1):  # 0~1s
        cc = cs[:len(cs) - lag] if lag else cs
        vv = vs[lag:] if lag else vs
        r = float(np.corrcoef(cc, vv)[0, 1])
        if r > best_r:
            best_r, best_lag = r, lag
    up = slice(warm, ramp_up); dn = slice(ramp_up, min(ramp_up + ramp_dn, tmo))
    mae_up = float(np.mean(np.abs((v0 - c0)[up])))
    mae_dn = float(np.mean(np.abs((v0 - c0)[dn])))
    gap = longest_true(np.abs(v0[warm:tmo] - c0[warm:tmo]) > 0.5) * dt
    print("\n== 连续斜坡 (env0) ==")
    print("  ρ(cmd,v_act)        = %.4f   (11档离散扫描 2999 是 0.9983)" % rho)
    print("  MAE                 = %.3f m/s" % mae)
    print("  滞后 lag            = %.2f s (corr=%.4f)" % (best_lag * dt, best_r))
    print("  上行 MAE / 下行 MAE = %.3f / %.3f  (差 %+.3f)" %
          (mae_up, mae_dn, mae_dn - mae_up))
    print("  最大掉速 |err|>0.5 持续 = %.2f s" % gap)
    print("  cmd 范围 %.2f→%.2f→%.2f；v_act 实测 %.2f→%.2f" %
          (c0[0], c0[ramp_up], c0[-1], v0.min(), v0.max()))

    # ---------------- PUSH (env 1..6) ----------------
    print("\n== 外力推鲁棒性 (env1..6) ==")
    print("  跌倒判据: 只看 root_pos_z < %.2f（dones 含 timeout，已排除）；恢复=速度回到 cmd±15%%" % fall_z)
    rows = []
    for k in range(N - 1):
        e = k + 1
        tgt = float(pvels[k]); imp = float(pimp[k])
        # 推动事件只取"完整观测窗口"内的（避开启动段与 timeout reset）
        events = [s for s in np.where(push)[0] if warm <= s < tmo - period]
        n_evt = len(events)
        falls = 0; rec_steps = []; zmin = []
        for s in events:
            seg = slice(s, min(s + period, tmo))
            zz = z[seg, e]; vv = vx[seg, e]
            fell = bool(np.any(zz < fall_z))   # 只用躯干高度，不用 dones
            falls += fell
            zmin.append(float(zz.min()))
            # 恢复：从推动后第 3 步起，连续 15 步都在 cmd±15% 内视为恢复
            band = 0.15 * tgt
            ok = np.abs(vv - tgt) <= band
            rec = None
            for i in range(3, len(ok) - 15):
                if np.all(ok[i:i + 15]):
                    rec = i; break
            rec_steps.append(rec if rec is not None else period)
        fall_rate = falls / max(n_evt, 1)
        med_rec = float(np.median(rec_steps)) * dt if rec_steps else float("nan")
        rows.append(dict(env=e, v_cmd=tgt, impulse=imp, n_push=n_evt,
                         fall_rate=round(fall_rate, 3),
                         recover_s=round(med_rec, 2),
                         z_min=round(float(np.min(zmin)), 3) if zmin else None,
                         vy_peak=round(float(np.max(np.abs(vy[:, e]))), 3)))
        print("  env%d v=%.1f 冲量=%.1f m/s: 推%2d次 跌倒率=%.0f%% "
              "恢复=%.2fs 最低躯干高=%.3f |vy|峰=%.2f" %
              (e, tgt, imp, n_evt, fall_rate * 100, med_rec,
               float(np.min(zmin)) if zmin else -1, float(np.max(np.abs(vy[:, e])))))

    # 基线（未推的 env0）作对照 —— 同样只统计 [warm,tmo)，避免启动前栽假象
    print("\n  [对照] env0(连续斜坡,无推) [%d,%d) 最低躯干高=%.3f" %
          (warm, tmo, float(z[warm:tmo, 0].min())))

    if args.csv:
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        # 同时把 ramp 汇总写到一个 txt 旁边
        with open(args.csv.replace(".csv", "_ramp.txt"), "w", encoding="utf-8") as f:
            f.write("rho=%.4f mae=%.3f lag_s=%.2f mae_up=%.3f mae_dn=%.3f gap_s=%.2f\n" %
                    (rho, mae, best_lag * dt, mae_up, mae_dn, gap))
        print("CSV ->", args.csv)


if __name__ == "__main__":
    main()
