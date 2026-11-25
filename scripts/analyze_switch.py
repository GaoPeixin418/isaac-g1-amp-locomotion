# -*- coding: utf-8 -*-
"""受控对照分析：同一指令点下 constant vs switch 的稳态跟踪对比。

稳态窗口：两种模式都取最后一段（切换后 +SETTLE 步之后），保证越过指令滞后、
比较的是"最终能不能稳定执行"，而不是切换瞬间的响应。
物理量口径同 analyze_coverage.py：沿指令方向投影 / 跌倒只用 z / 逐 env 独立。

用法： python scripts/analyze_switch.py <coverage_switch_*.npz> --csv out.csv
"""
import argparse
import numpy as np


def main(npz_path, csv_path):
    d = np.load(npz_path, allow_pickle=True)
    dt = float(d["dt"])
    labels = d["labels"]
    modes = d["modes"]
    grid = d["grid"]
    warm = int(d["warm"])
    settle = int(d["settle"])
    total = int(d["total"])
    vx, vy, wz = d["lin_vel_x"], d["lin_vel_y"], d["ang_vel_z"]
    z = d["root_pos_z"]
    rb = d["cmd_rb"]
    N = vx.shape[1]

    s0 = warm + settle        # 切换后 +125 步
    s = slice(s0, total)
    fall_z = 0.40

    rows = []
    for i in range(N):
        lab, mode = str(labels[i]), str(modes[i])
        cvx, cvy, cwz = [float(x) for x in grid[i]]
        speed = float(np.hypot(cvx, cvy))
        rb_err = float(np.max(np.abs(rb[s, i, :].mean(axis=0) - grid[i])))
        vax, vay, waz = (float(vx[s, i].mean()), float(vy[s, i].mean()),
                         float(wz[s, i].mean()))
        zmin = float(z[s, i].min())
        fell = bool((z[s, i] < fall_z).mean() > 0.05)

        if speed > 1e-3:
            u = grid[i, :2] / speed
            proj = float(vax * u[0] + vay * u[1])
            ratio = proj / speed
        else:
            proj = float("nan")
            ratio = (waz / cwz) if abs(cwz) > 1e-3 else 0.0

        cls = "跌倒" if fell else ("可行" if 0.7 <= abs(ratio) <= 1.3 else
              ("部分" if abs(ratio) >= 0.3 else "失效"))
        rows.append(dict(label=lab, mode=mode, cmd_vx=cvx, cmd_vy=cvy, cmd_wz=cwz,
                         act_vx=vax, act_vy=vay, act_wz=waz, ratio=ratio,
                         z_min=zmin, fell=int(fell), rb_err=rb_err, cls=cls))

    cols = ["label", "mode", "cmd_vx", "cmd_vy", "cmd_wz", "act_vx", "act_vy",
            "act_wz", "ratio", "z_min", "fell", "rb_err", "cls"]
    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(
                    (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                    for c in cols) + "\n")
        print(f"[OK] CSV -> {csv_path}")

    # 成对对比表：把每个 label 的 constant / switch 放一起
    print(f"\n=== 受控对照（稳态窗口 [{s0},{total}), dt={dt}, "
          f"切换点={warm}, settle={settle}）===")
    print(f"  {'label':<10}{'cmd':<18}{'constant':>22}{'switch':>22}  判定")
    by = {(r["label"], r["mode"]): r for r in rows}
    for lab in dict.fromkeys(str(x) for x in labels):
        c, s_ = by.get((lab, "constant")), by.get((lab, "switch"))
        if not c or not s_:
            continue
        cmd = f"({c['cmd_vx']:.2f},{c['cmd_vy']:.2f},{c['cmd_wz']:.2f})"
        cs = f"{c['ratio']:+.2f}/{c['cls']}"
        ss = f"{s_['ratio']:+.2f}/{s_['cls']}"
        # 机制判读
        gap = abs(c["ratio"]) - abs(s_["ratio"])
        if gap > 0.35:
            verdict = "切换后退化 → 模式切换/指令突变问题"
        elif gap < -0.35:
            verdict = "切换后更好 → 需查证"
        else:
            verdict = "两种时序一致"
        print(f"  {lab:<10}{cmd:<18}{cs:>22}{ss:>22}  {verdict}")

    max_rb = max(r["rb_err"] for r in rows)
    print(f"\n[校验] cmd 回读最大偏差 = {max_rb:.4f} "
          f"({'OK' if max_rb < 0.05 else '⚠ 指令被覆盖'})")
    n_const_ok = sum(1 for r in rows if r["mode"] == "constant" and r["cls"] == "可行")
    n_sw_ok = sum(1 for r in rows if r["mode"] == "switch" and r["cls"] == "可行")
    print(f"[汇总] constant 可行 {n_const_ok}/{N//2}  ·  switch 可行 {n_sw_ok}/{N//2}")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    main(a.npz, a.csv)
