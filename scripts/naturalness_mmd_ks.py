# -*- coding: utf-8 -*-
"""无裁判自然度指标：关节角分布 MMD / KS（绕开自训练判别器的开卷打分）。

=============================================================================
为什么要做这个（公平性负债）
=============================================================================
项目里所有"风格分"来自 AMP 判别器，而判别器是在**策略自生成数据**上训练的，
给策略自己打分等于开卷考试 —— 只能组内比，不能用来证明"像人"。

本脚本换一个完全不依赖裁判的客观量：把策略 rollout 的 29 维关节角分布，
直接与 LAFAN1 重定向到 G1 的真人参考动作分布比：
  * KS (Kolmogorov-Smirnov)：逐关节一维分布的最大 CDF 差，0~1，越小越像
  * MMD (Maximum Mean Discrepancy)：RBF 核下的多变量分布距离，带 bootstrap 显著性

=============================================================================
口径陷阱（必须先处理，否则又是一个不可比数字）
=============================================================================
参考数据 30 个 clip **全是转弯 / 侧步 / 变速**（实测 |v| 0.17–2.91 m/s），
没有匀速直走；而策略是定速直走 rollout。直接拿"全参考"比"定速 1.0"
会把动作模式差异算成自然度差异。因此：
  1) 主口径 = 速度匹配：策略 @v 只与参考中 |v| 落在 [v-0.35, v+0.35] 的帧比
  2) 同时报"全参考"口径作对照，明确它是上界、不是结论
  3) 只比关节角位置（dof_pos / joint_pos），不比速度/时序 —— 前者是风格，
     后者受动作模式主导（MEMORY 3.5 的教训：物理量与主诉要对齐）

关节分组（G1 29 DOF，与 MEMORY 3.5 一致）：
  [0:6] 左腿  [6:12] 右腿  → 步态风格主要看这 12 个；其余含躯干/手臂，
  在行走 AMP 里幅度小，单独报一档避免被"手臂几乎不动"稀释。

用法：
  python scripts/naturalness_mmd_ks.py \
      --npz <sweep11.npz> --ref-dir <walk_and_run/ 含30个pkl> \
      --ckpt-tag v23_2999 --csv out.csv
"""
import argparse
import glob
import os
import pickle
import sys

import numpy as np

# ---- 手写 KS，避免依赖 scipy（本地 venv 未装 scipy 也能跑）-----------------
def ks_stat(a, b):
    """一维两样本 KS 统计量：经验 CDF 的最大绝对差。"""
    a = np.sort(np.asarray(a, float).ravel())
    b = np.sort(np.asarray(b, float).ravel())
    allv = np.concatenate([a, b])
    # 在合并样本的所有取值点上评估两条 CDF
    ca = np.searchsorted(a, allv, side="right") / len(a)
    cb = np.searchsorted(b, allv, side="right") / len(b)
    return float(np.max(np.abs(ca - cb)))


def mmd_rbf(x, y, n_boot=0, rng=None):
    """MMD^2 (无偏估计) + RBF 核，带宽用 median heuristic。
    返回 (mmd2, p_boot 或 None)。n_boot>0 时做置换检验给 p 值。"""
    x = np.asarray(x, float); y = np.asarray(y, float)
    nx, ny = len(x), len(y)
    # 子采样控制 O(n^2)：每边最多 600 帧
    if rng is None:
        rng = np.random.default_rng(0)
    if nx > 600: x = x[rng.choice(nx, 600, replace=False)]; nx = 600
    if ny > 600: y = y[rng.choice(ny, 600, replace=False)]; ny = 600

    z = np.vstack([x, y])
    # median heuristic 带宽
    d2 = np.sum((z[:, None, :] - z[None, :, :]) ** 2, axis=2)
    med = np.median(d2[d2 > 0]) if np.any(d2 > 0) else 1.0
    gamma = 1.0 / (2.0 * med + 1e-9)

    def K(u, v):
        return np.exp(-gamma * np.sum((u[:, None, :] - v[None, :, :]) ** 2, axis=2))

    kxx, kyy, kxy = K(x, x), K(y, y), K(x, y)
    # 无偏：去掉对角
    mmd2 = (kxx.sum() - np.trace(kxx)) / (nx * (nx - 1)) \
         + (kyy.sum() - np.trace(kyy)) / (ny * (ny - 1)) \
         - 2.0 * kxy.mean()

    p = None
    if n_boot > 0:
        # 置换检验：打乱 x/y 标签重算，看真实值在零分布里的分位
        pooled = np.vstack([x, y]); n = nx + ny
        null = np.empty(n_boot)
        for i in range(n_boot):
            idx = rng.permutation(n)
            xa, ya = pooled[idx[:nx]], pooled[idx[nx:]]
            ka, kb, kab = K(xa, xa), K(ya, ya), K(xa, ya)
            null[i] = (ka.sum()-np.trace(ka))/(nx*(nx-1)) \
                    + (kb.sum()-np.trace(kb))/(ny*(ny-1)) - 2.0*kab.mean()
        p = float((np.sum(null >= mmd2) + 1) / (n_boot + 1))
    return float(max(mmd2, 0.0)), p


def load_ref(ref_dir):
    """读全部 pkl，返回 (dof[N,29], speed[N])。speed 用 root_pos 差分水平速率。"""
    fs = sorted(glob.glob(os.path.join(ref_dir, "*.pkl")))
    if not fs:
        raise SystemExit("没找到参考 pkl: " + ref_dir)
    dofs, speeds = [], []
    for f in fs:
        d = pickle.load(open(f, "rb"))
        dp = np.asarray(d["dof_pos"], float)       # (T,29)
        fps = float(np.asarray(d["fps"]).ravel()[0])
        rp = np.asarray(d["root_pos"], float)       # (T,3)
        v = np.linalg.norm(np.diff(rp[:, :2], axis=0), axis=1) * fps  # (T-1,)
        # 让 speed 与 dof 对齐：最后一帧沿用前一帧速度
        v = np.concatenate([v, v[-1:]])
        dofs.append(dp); speeds.append(v)
    return np.concatenate(dofs, 0), np.concatenate(speeds, 0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", required=True, help="sweep11 npz，含 joint_pos(T,N,29) 与 velocities(N)")
    ap.add_argument("--ref-dir", required=True, help="含 30 个 pkl 的目录")
    ap.add_argument("--ckpt-tag", default="ckpt")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--match-bw", type=float, default=0.35, help="速度匹配半宽 m/s")
    ap.add_argument("--boot", type=int, default=200, help="MMD 置换检验次数，0=关闭")
    ap.add_argument("--warmup-drop", type=float, default=0.3, help="丢弃前多少比例的暂态帧")
    args = ap.parse_args()

    ref_dof, ref_speed = load_ref(args.ref_dir)
    print("参考: %d 帧, |v| 范围 %.2f–%.2f, 均速 %.2f" %
          (len(ref_dof), ref_speed.min(), ref_speed.max(), ref_speed.mean()))

    d = np.load(args.npz, allow_pickle=True)
    jp = d["joint_pos"]            # (T, N, 29)
    vels = d["velocities"]
    T, N, D = jp.shape
    n0 = int(T * args.warmup_drop)
    print("策略: %s  T=%d N=%d D=%d  丢前 %d 帧" % (args.ckpt_tag, T, N, D, n0))

    rng = np.random.default_rng(42)
    leg = list(range(0, 12))        # 12 个腿部关节（风格主体）
    rows = []
    for j, v in enumerate(vels):
        pol = jp[n0:, j, :]         # (T', 29)
        lo, hi = v - args.match_bw, v + args.match_bw
        m = (ref_speed >= lo) & (ref_speed <= hi)
        ref_m = ref_dof[m]
        # 匹配不足就放宽到全参考，但标注
        matched = len(ref_m) >= 200
        if not matched:
            ref_m = ref_dof
        # 逐关节 KS（腿 12 个）
        ks_leg = [ks_stat(pol[:, k], ref_m[:, k]) for k in leg]
        ks_all = [ks_stat(pol[:, k], ref_dof[:, k]) for k in range(D)]
        mmd2_m, p_m = mmd_rbf(pol[:, leg], ref_m[:, leg], args.boot, rng)
        mmd2_f, p_f = mmd_rbf(pol[:, leg], ref_dof[:, leg], 0, rng)
        rows.append(dict(v=float(v), n_ref=int(len(ref_m)), matched=matched,
                         ks_leg_mean=float(np.mean(ks_leg)), ks_leg_max=float(np.max(ks_leg)),
                         ks_all_mean=float(np.mean(ks_all)),
                         mmd2_match=mmd2_m, mmd2_p=p_m, mmd2_fullref=mmd2_f))
        print("  v=%.2f  匹配帧=%4d%s  KS_leg mean/max=%.3f/%.3f  "
              "KS_all=%.3f  MMD2=%.4f p=%s  (全参考 MMD2=%.4f)" %
              (v, len(ref_m), "" if matched else "(放宽)", np.mean(ks_leg),
               np.max(ks_leg), np.mean(ks_all), mmd2_m,
               "%.3f" % p_m if p_m is not None else "-", mmd2_f))

    print("\n== 汇总（速度匹配口径为主）==")
    print("KS_leg_mean 全档均值 %.3f 范围 %.3f–%.3f" %
          (np.mean([r["ks_leg_mean"] for r in rows]),
           min(r["ks_leg_mean"] for r in rows), max(r["ks_leg_mean"] for r in rows)))
    print("KS 随速度单调? " + ("是(越高越不像人)" if
          all(rows[i]["ks_leg_mean"] <= rows[i+1]["ks_leg_mean"] for i in range(len(rows)-1))
          else "否(非单调)"))

    if args.csv:
        import csv
        os.makedirs(os.path.dirname(os.path.abspath(args.csv)), exist_ok=True)
        with open(args.csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader(); w.writerows(rows)
        print("CSV ->", args.csv)


if __name__ == "__main__":
    main()
