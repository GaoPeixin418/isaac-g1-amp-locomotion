#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
judge_ctrl.py —— 按预注册阈值判对照组结局（A/B/C）

阈值来源：docs/对照组v22预注册协议.md 第三节，**训练启动前写死**，不允许事后改。

用法（本地或服务器都能跑，只读 csv）：
    python judge_ctrl.py /data/progress/ctrl_v22_curve.csv
    python judge_ctrl.py backup/server/xxx/data/progress/ctrl_v22_curve.csv
"""
import csv
import sys

TRIGGER = 0.20   # iter 400 起，低于此算"没在走"
FLY = 0.70       # iter 800 达标算起飞
DEAD = 0.20      # iter 1200 仍低于此判死


def row_of(rows, it):
    for r in rows:
        if int(r["iter"]) == it:
            return r
    return None


def main(path):
    with open(path, encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r.get("iter", "").isdigit()]
    if not rows:
        print("曲线还没有数据（只有表头）-> 训练未产生评估点")
        return 2

    print("%-6s %-8s %-8s %-8s %-8s %-8s" % ("iter", "v_cmd", "v_act", "leg_ai", "cadence", "knee"))
    for r in rows:
        print("%-6s %-8.2f %-8.3f %-8.1f %-8s %-8s" % (
            r["iter"], float(r["v_cmd"] or 0), float(r["v_act"] or 0),
            float(r["leg_ai"] or float("nan")),
            r.get("cadence", "-"), r.get("knee_deg", "-")))

    r400, r800, r1200 = row_of(rows, 400), row_of(rows, 800), row_of(rows, 1200)
    v = lambda r: float(r["v_act"]) if r else None  # noqa: E731

    print("\n" + "=" * 58)
    checks = [(400, r400, TRIGGER, "有戏"), (800, r800, FLY, "起飞"),
              (1200, r1200, DEAD, "判死线")]
    for it, r, th, label in checks:
        if r is None:
            print("iter %-5d %-8s 未评估（还没跑到）" % (it, label))
        else:
            ok = "PASS" if v(r) >= th else "FAIL"
            print("iter %-5d %-8s v_act=%.3f 阈值>=%.2f -> %s" % (it, label, v(r), th, ok))

    if r800 and v(r800) >= FLY:
        last = rows[-1]
        ai = float(last["leg_ai"])
        print("\n>>> 结局 A：对照组学会走路")
        print("    legAI = %.1f%% -> %s" % (
            ai, "已达根治线(<=15%)，v2.3 可跳过" if ai <= 15 else
            "仍偏高，下一步权重 -1.0 -> -1.5（只加 50%，绝不翻倍）"))
        return 0
    if r1200 and v(r1200) < DEAD:
        print("\n>>> 结局 B：判死，证伪『v2.2 配置可训』")
        print("    下一步：bash /data/setup/37_train_fresh_v23.sh   (WARMUP=12000)")
        return 1
    if r400 and v(r400) < TRIGGER and (r800 is None or v(r800) < TRIGGER):
        print("\n>>> 趋势：早期没起来，等 iter 1200 定论（别提前杀，v1 在 400 时也只有 0.06）")
        return 2
    print("\n>>> 结局 C：模糊区间，看趋势决定续训还是转 B")
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else "/data/progress/ctrl_v22_curve.csv"))
