#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
从 Isaac Lab 的训练日志里解析每轮迭代的指标，导出 CSV + 里程碑摘要。

背景
----
Isaac Lab 的训练日志是给人看的：每轮迭代打印一个带 ANSI 颜色和大量对齐空格的块，
5000 轮下来十几 MB，既没法直接画图也没法放进报告。这个脚本把它压成结构化的时间序列，
供后续画训练曲线、做 AMP vs PPO 对照时使用。

用法
----
    python scripts/parse_train_log.py <train_main.log> [--out <csv路径>]

输出
----
1. CSV：每行一轮迭代，列为各指标（列名与日志中的字段名一致）
2. stdout：每 500 轮的里程碑摘要表，便于快速核对趋势

设计要点
--------
- 日志里有 ANSI 转义符（`\\x1b[1m` 之类），必须先剥掉再匹配
- 日志文件里混有被错误解码的中文乱码（原始 UTF-8 被按 GBK 解码），用 errors="replace" 兜底
- "Iteration time: 4.26s" 这类带单位的数值，正则要容忍尾部单位
"""
import argparse
import csv
import re
import sys
from pathlib import Path

ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
ITER_RE = re.compile(r"Learning iteration (\d+)/(\d+)")
KV_RE = re.compile(r"^\s*(.+?):\s*(-?\d[\d.eE+\-]*)s?\s*$")

# 只保留对分析和汇报有价值的字段，其余丢弃（日志里有上百行无关输出）
FIELDS = [
    "Mean reward",
    "Mean episode length",
    "Mean action noise std",
    "Mean value loss",
    "Mean surrogate loss",
    "Mean amp/disc_loss loss",
    "Mean amp/disc_grad_penalty loss",
    "Mean amp/disc_score loss",
    "Mean amp/disc_demo_score loss",
    "Mean AMP total reward",
    "Mean AMP style reward",
    "Episode_Reward/style",
    "Episode_Reward/track_lin_vel_xy_exp",
    "Episode_Reward/track_ang_vel_z_exp",
    "Episode_Reward/feet_air_time",
    "Episode_Reward/feet_slide",
    "Episode_Reward/termination_penalty",
    "Episode_Reward/dof_torques_l2",
    "Metrics/base_velocity/error_vel_xy",
    "Metrics/base_velocity/error_vel_yaw",
    "Episode_Termination/time_out",
    "Episode_Termination/base_height",
    "Episode_Termination/bad_orientation",
]

SUMMARY_FIELDS = [
    ("Episode_Reward/style", "风格分"),
    ("Metrics/base_velocity/error_vel_xy", "速度误差 m/s"),
    ("Episode_Termination/time_out", "跑满全程 %"),
    ("Episode_Termination/base_height", "摔倒-重心低 %"),
    ("Episode_Termination/bad_orientation", "摔倒-姿态崩 %"),
    ("Mean AMP style reward", "AMP风格奖励"),
    ("Mean amp/disc_score loss", "判别器分(策略)"),
    ("Mean amp/disc_demo_score loss", "判别器分(参考)"),
    ("Mean reward", "总回报"),
]


def parse_log(path: Path) -> list[dict]:
    rows: list[dict] = []
    cur: dict | None = None
    with path.open(encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = ANSI_RE.sub("", raw).rstrip("\n")
            m = ITER_RE.search(line)
            if m:
                if cur is not None:
                    rows.append(cur)
                cur = {"iteration": int(m.group(1)), "max_iteration": int(m.group(2))}
                continue
            if cur is None:
                continue
            kv = KV_RE.match(line)
            if kv:
                key = kv.group(1).strip()
                if key in FIELDS:
                    try:
                        cur[key] = float(kv.group(2))
                    except ValueError:
                        cur[key] = kv.group(2)
    if cur is not None:
        rows.append(cur)
    return rows


def write_csv(rows: list[dict], out: Path) -> None:
    cols = ["iteration", "max_iteration"] + FIELDS
    with out.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def summarize(rows: list[dict], every: int = 500) -> None:
    if not rows:
        print("!! 没解析到任何迭代，检查日志路径或格式")
        return
    print(f"解析到 {len(rows)} 轮，最后一轮 iteration = {rows[-1]['iteration']}")
    header = "iter".ljust(6) + "".join(name.ljust(14) for _, name in SUMMARY_FIELDS)
    print(header)
    print("-" * len(header))
    targets = list(range(0, rows[-1]["iteration"] + 1, every))
    if rows[-1]["iteration"] not in targets:
        targets.append(rows[-1]["iteration"])
    for t in targets:
        row = next((r for r in rows if r["iteration"] == t), None)
        if row is None:
            continue
        line = str(t).ljust(6)
        for key, _ in SUMMARY_FIELDS:
            v = row.get(key)
            line += (f"{v:.4f}" if isinstance(v, float) else "-").ljust(14)
        print(line)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", help="训练日志路径（如 backup/server/<快照>/data/logs/train_main.log）")
    ap.add_argument("--out", default=None, help="CSV 输出路径")
    ap.add_argument("--every", type=int, default=500, help="摘要采样间隔")
    args = ap.parse_args()

    path = Path(args.log)
    if not path.exists():
        print(f"!! 找不到日志: {path}")
        return 1

    rows = parse_log(path)
    out = Path(args.out) if args.out else path.parent.parent / "train_curve.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    write_csv(rows, out)
    print(f">> CSV 已写入: {out}  ({len(rows)} 行)")
    summarize(rows, args.every)
    return 0


if __name__ == "__main__":
    sys.exit(main())
