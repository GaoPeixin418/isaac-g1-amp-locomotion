#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""步态左右对称性诊断：从 eval_gait.py 产出的 npz 直接算 SI 与周期稳定性。

=============================================================================
为什么单独写一个脚本（2026-10-22）
=============================================================================
之前所有步态指标都塞在 compare_gait.py 里，只报了 duty / 步频 / 步长 / CoT，
**没有一项看左右腿是否一样**。于是「走路有点瘸」这个反馈，
我只能靠肉眼判断，拿不出数字。

本脚本补上这一课。三个指标（全部只用 foot_z，不依赖任何参考动作）：

  1. SI_clear   抬升对称指数 = 2|L-R|/(L+R)，L/R 为逐步抬升高度的中位数
  2. SI_stance  支撑对称指数 = 同上，用逐步支撑时长
  3. CV_period  周期变异系数，左右腿分别算（看哪条腿更"乱"）

人类参照：SI < 5%，步周期 CV ≈ 0.05~0.10。
（文献量级参照，非精确引用；硬结论是本项目自身的数值与左右方向一致性。）

=============================================================================
关键实现细节（踩过坑）
=============================================================================
* **必须逐步分解后取中位数。** 第一版用 z.max()-z.min() 全局极值算抬升，
  得到 0.5 档 72.7% / 2.0 档 70.7% 的离谱数值——被单步异常污染。
  逐步切分（触地事件之间算一步）再取中位数，才降到可信的 7.7%~27.7%。
* **触地判定用高度而不是接触力**：阈值 = 该脚高度的 15 分位 + 3 cm。
  自适应下限，避免不同模型足端基准不可比。

=============================================================================
用法
=============================================================================
    python scripts/analyze_symmetry.py backup/eval_out/gait_amp_model_4000.npz
    python scripts/analyze_symmetry.py backup/eval_out/*.npz        # 多模型对比
    python scripts/analyze_symmetry.py x.npz --steady-from 10       # 只统计 10s 后
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np

SI_THRESHOLD = 5.0  # 人类正常上限（%）


def split_strides(foot_z: np.ndarray, dt: float) -> np.ndarray | None:
    """把单只脚的高度序列切成逐步，返回 (n_steps, 3)：[支撑时长, 抬升高度, 步周期]。"""
    thr = np.percentile(foot_z, 15) + 0.03
    stance = (foot_z < thr).astype(int)
    change = np.diff(stance, prepend=stance[0])
    land = np.where(change > 0)[0]

    rows = []
    for k in range(len(land) - 1):
        a, b = land[k], land[k + 1]
        seg, st = foot_z[a:b], stance[a:b]
        swing, support = seg[st == 0], seg[st == 1]
        if len(swing) > 2 and len(support) > 2:
            rows.append((st.sum() * dt, swing.max() - support.min(), (b - a) * dt))
    return np.array(rows) if len(rows) > 2 else None


def symmetry_index(a: float, b: float) -> float:
    """SI = 2|a-b|/(a+b) × 100%。"""
    s = a + b
    return 2 * abs(a - b) / s * 100 if s > 1e-9 else 0.0


def analyze(path: Path, steady_from: float = 10.0, label: str | None = None) -> dict | None:
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    vels = d["velocities"]
    tag = label or path.stem

    print(f"\n=== {tag} ===")
    print(f"{'档位':>5} | {'SI_clear':>9} | {'SI_stance':>10} | {'CV_L':>7} | {'CV_R':>7} | {'判定'}")
    print("-" * 68)

    rows = []
    for i, v in enumerate(vels):
        start = int(steady_from / dt)
        left = d["foot_z"][start:, i, 0]
        right = d["foot_z"][start:, i, 1]

        sl, sr = split_strides(left, dt), split_strides(right, dt)
        if sl is None or sr is None:
            print(f"{v:5.1f} | 数据不足以切分步态")
            continue

        clear_l, clear_r = np.median(sl[:, 1]), np.median(sr[:, 1])
        stance_l, stance_r = np.median(sl[:, 0]), np.median(sr[:, 0])
        cv_l = sl[:, 2].std() / sl[:, 2].mean()
        cv_r = sr[:, 2].std() / sr[:, 2].mean()

        si_c = symmetry_index(clear_l, clear_r)
        si_s = symmetry_index(stance_l, stance_r)
        bad = si_c > SI_THRESHOLD or si_s > SI_THRESHOLD

        rows.append(
            dict(v=v, si_clear=si_c, si_stance=si_s, cv_l=cv_l, cv_r=cv_r,
                 clear_l=clear_l, clear_r=clear_r, stance_l=stance_l, stance_r=stance_r)
        )
        print(
            f"{v:5.1f} | {si_c:8.1f}% | {si_s:9.1f}% | {cv_l:7.3f} | {cv_r:7.3f} | "
            f"{'✗ 不对称' if bad else '✓'}"
        )

    if not rows:
        return None

    si_cs = np.array([r["si_clear"] for r in rows])
    si_ss = np.array([r["si_stance"] for r in rows])
    # 方向一致性：谁抬得低
    lower_left = sum(1 for r in rows if r["clear_l"] < r["clear_r"])

    print("-" * 68)
    print(f"SI_clear  平均 {si_cs.mean():.1f}%  最大 {si_cs.max():.1f}%  "
          f"超标 {int((si_cs > SI_THRESHOLD).sum())}/{len(rows)} 档")
    print(f"SI_stance 平均 {si_ss.mean():.1f}%  最大 {si_ss.max():.1f}%  "
          f"超标 {int((si_ss > SI_THRESHOLD).sum())}/{len(rows)} 档")
    print(f"左脚抬升更低: {lower_left}/{len(rows)} 档  "
          f"{'(方向一致 -> 系统性偏置)' if lower_left >= len(rows) - 1 or lower_left <= 1 else '(方向不一致 -> 更像随机波动)'}")

    return dict(tag=tag, rows=rows, si_clear_mean=float(si_cs.mean()),
                si_clear_max=float(si_cs.max()), si_stance_mean=float(si_ss.mean()),
                lower_left=lower_left, n=len(rows))


def main() -> int:
    ap = argparse.ArgumentParser(description="步态左右对称性诊断")
    ap.add_argument("npz", nargs="+", help="npz 文件或通配符")
    ap.add_argument("--steady-from", type=float, default=10.0, help="从第几秒开始统计（默认10）")
    args = ap.parse_args()

    paths: list[Path] = []
    for pat in args.npz:
        hits = [Path(p) for p in glob.glob(pat)]
        paths.extend(hits if hits else [Path(pat)])
    paths = [p for p in paths if p.exists()]
    if not paths:
        print("!! 没有找到任何 npz")
        return 1

    results = [r for p in paths if (r := analyze(p, args.steady_from))]

    if len(results) > 1:
        print("\n" + "=" * 68)
        print("=== 汇总（越小越对称）===")
        print(f"{'模型':<28} | {'SI_clear均值':>12} | {'SI_clear最大':>12} | {'左低档数':>8}")
        print("-" * 68)
        for r in sorted(results, key=lambda x: x["si_clear_mean"]):
            print(f"{r['tag']:<28} | {r['si_clear_mean']:11.1f}% | "
                  f"{r['si_clear_max']:11.1f}% | {r['lower_left']}/{r['n']:>6}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
