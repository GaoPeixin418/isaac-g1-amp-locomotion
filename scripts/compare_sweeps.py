#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""对比两份速度扫描 npz（训练前 vs 训练后），回答两个预注册问题。

用法:
    python compare_sweeps.py <before.npz> <after.npz>

预注册的两个问题（见 docs/速度扫描结论_2026-11-18.md §7）：
  Q1 高速掉速是否随训练量缓解？ -> 高速段 (>=2.0) 平均 |err| 的前后变化
  Q2 legAI 是否继续改善，还是像 v1 后期回升？ -> 各档 legAI 的前后变化

口径与 analyze_sweep.py / full_eval.py 完全一致（复用同一函数）。
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import analyze_sweep as az   # 复用 per_env_metrics / 口径兼容层


def load_rows(path):
    d = np.load(path, allow_pickle=True)
    T = d["lin_vel_x"].shape[0]
    n0 = int(T * 0.3)
    N = d["lin_vel_x"].shape[1]
    return [az.per_env_metrics(d, j, n0) for j in range(N)], d


def main():
    if len(sys.argv) < 3:
        print("用法: compare_sweeps.py <before.npz> <after.npz>")
        return 1
    p_before, p_after = sys.argv[1], sys.argv[2]
    rb, db = load_rows(p_before)
    ra, da = load_rows(p_after)
    print("before =", os.path.basename(p_before))
    print("after  =", os.path.basename(p_after))
    vb = db["velocities"]; va = da["velocities"]
    if not np.allclose(vb, va):
        print("!! 两份 npz 的速度档不一致，不可比")
        print("  before:", np.round(vb, 2).tolist())
        print("  after :", np.round(va, 2).tolist())
        return 2
    print("速度档一致: %d 档 %s" % (len(vb), np.round(vb, 2).tolist()))

    # 逐档对比表
    print("\n== 逐档对比（after - before）==")
    print("%-6s | %-15s | %-15s | %-15s" %
          ("v_cmd", "v_act 前→后", "err 前→后", "legAI 前→后"))
    for b, a in zip(rb, ra):
        print("%-6.2f | %6.3f → %6.3f | %+6.3f → %+6.3f | %5.1f → %5.1f (%+5.1f)" %
              (b["v_cmd"], b["v_act"], a["v_act"], b["err_actual"], a["err_actual"],
               b["legAI"], a["legAI"], a["legAI"] - b["legAI"]))

    # 只用指令中心准的档
    gb = [r for r in rb if r["cmd_dev"] <= 0.03]
    ga = [r for r in ra if r["cmd_dev"] <= 0.03]
    key = lambda r: round(r["v_cmd"], 2)
    gbd = {key(r): r for r in gb}; gad = {key(r): r for r in ga}
    common = sorted(set(gbd) & set(gad))

    def seg(rs, lo, hi):
        return [r for r in rs if lo <= r["v_cmd"] <= hi]

    print("\n== 预注册问题的裁决 ==")
    # Q1 高速掉速
    for name, lo, hi in [("低速段 ≤1.5", 0.0, 1.5), ("高速段 ≥2.0", 2.0, 99.0)]:
        bseg = [gbd[k] for k in common if lo <= k <= hi]
        aseg = [gad[k] for k in common if lo <= k <= hi]
        if not bseg:
            continue
        eb = np.mean([abs(r["err_actual"]) for r in bseg])
        ea = np.mean([abs(r["err_actual"]) for r in aseg])
        print("Q1 %s 平均|err|: %.3f → %.3f m/s (%+.3f, %s)" %
              (name, eb, ea, ea - eb,
               "缓解" if ea < eb - 0.02 else ("恶化" if ea > eb + 0.02 else "基本不变")))
    # Q2 legAI
    bmid = [gbd[k] for k in common if 0.9 <= k <= 1.1]
    amid = [gad[k] for k in common if 0.9 <= k <= 1.1]
    if bmid:
        lb = np.mean([r["legAI"] for r in bmid]); la = np.mean([r["legAI"] for r in amid])
        print("Q2 @1.0 m/s legAI: %.1f%% → %.1f%% (%+.1f%%, %s)" %
              (lb, la, la - lb,
               "改善" if la < lb - 2 else ("回升(越训越不对称)" if la > lb + 2 else "基本不变")))
    allb = [gbd[k] for k in common]; alla = [gad[k] for k in common]
    print("Q2 全档 legAI 均值: %.1f%% → %.1f%% (%+.1f%%)" %
          (np.mean([r["legAI"] for r in allb]), np.mean([r["legAI"] for r in alla]),
           np.mean([r["legAI"] for r in alla]) - np.mean([r["legAI"] for r in allb])))

    # ρ 前后
    for tag, rs in [("before", gb), ("after", ga)]:
        c = np.array([r["cmd_mean"] for r in rs]); a = np.array([r["v_act"] for r in rs])
        if len(rs) >= 3:
            print("ρ %s = %.4f, MAE = %.3f m/s" %
                  (tag, np.corrcoef(c, a)[0, 1], np.mean(np.abs(a - c))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
