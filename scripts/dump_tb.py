#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""把 rsl_rl 的 tfevents 标量抽成"按 iter 对齐的小表"。

=============================================================================
为什么要这个脚本（2026-10-22）
=============================================================================
用户问「amp_2000 是不是比 amp_4000 好，毕竟 2000 的风格分更高？」。
要回答这个问题必须拿同一条曲线在两个 iter 上的值，而不是凭印象。

train log 的尾巴只有最后一屏，tfevents 才是全量数据。
本脚本直接读 events 文件（不需要起 tensorboard 服务），
输出「标签 -> 指定 iter 处的值 / 指定窗口内的均值」。

用法
=============================================================================
    python dump_tb.py <events文件或目录> [--iters 1000,2000,3000,4000]
                     [--win 200] [--only Reward,Episode,Policy]
                     [--csv out.csv]

--iters 给多个 iter，各取 [iter-win, iter) 窗口的均值，等于
「训练到这一带时的稳定水平」，比单帧值更能代表模型状态。
"""
from __future__ import annotations

import argparse
import statistics
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator


def load(path: Path) -> dict[str, list[tuple[int, float]]]:
    ea = event_accumulator.EventAccumulator(
        str(path), size_guidance={event_accumulator.SCALARS: 0}
    )
    ea.Reload()
    out: dict[str, list[tuple[int, float]]] = {}
    for tag in ea.Tags().get("scalars", []):
        out[tag] = [(e.step, e.value) for e in ea.Scalars(tag)]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--iters", default="1000,2000,3000,4000")
    ap.add_argument("--win", type=int, default=200)
    ap.add_argument("--only", default="")
    ap.add_argument("--csv", default="")
    args = ap.parse_args()

    p = Path(args.path)
    if p.is_dir():
        files = sorted(p.glob("events.out.tfevents.*"), key=lambda f: f.stat().st_mtime)
        # 续训会产生 0/1 两个 events 文件，全部读进来按 step 合并
        data: dict[str, list[tuple[int, float]]] = {}
        for f in files:
            for tag, rows in load(f).items():
                data.setdefault(tag, []).extend(rows)
    else:
        data = load(p)

    iters = [int(x) for x in args.iters.split(",") if x.strip()]
    filters = [s for s in args.only.split(",") if s]

    print(f"# source={p}")
    header = ["tag"] + [f"@{it}" for it in iters]
    print(" | ".join(f"{h:>38s}" if i == 0 else f"{h:>10s}"
                     for i, h in enumerate(header)))
    print("-" * (38 + 12 * len(iters)))

    csv_rows = []
    for tag in sorted(data):
        if filters and not any(f in tag for f in filters):
            continue
        rows = sorted(data[tag], key=lambda r: r[0])
        if not rows:
            continue
        cells = []
        for it in iters:
            lo, hi = it - args.win, it
            vals = [v for s, v in rows if lo <= s < hi]
            if not vals:
                vals = [v for s, v in rows if s <= hi][-args.win:]
            cells.append(statistics.mean(vals) if vals else float("nan"))
        line_tag = tag[:38]
        print(f"{line_tag:>38s} | " + " ".join(f"{c:10.4f}" for c in cells))
        csv_rows.append([tag] + cells)

    if args.csv:
        Path(args.csv).write_text(
            "tag," + ",".join(str(i) for i in iters) + "\n"
            + "\n".join(",".join(str(x) for x in r) for r in csv_rows),
            encoding="utf-8",
        )
        print(f"# csv -> {args.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
