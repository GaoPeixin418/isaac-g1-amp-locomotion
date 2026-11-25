# -*- coding: utf-8 -*-
"""指令可行域扫描分析：把每个指令点的稳态跟踪误差算清楚，输出能力边界表。

物理量口径（全部对齐项目里踩过的坑，见 MEMORY.md）：
  * 用水平速率 norm(v_xy) 与沿指令方向的投影，不用 x 分量（会互相抵消）
  * 丢启动暂态（前 2.5 s = 125 步），只在稳态窗口统计
  * 跌倒只用 root_pos_z < fall_z，不用 dones（混 episode timeout）
  * 逐 env / 逐指令点独立算，绝不跨点合成（合成会掩盖失效点）
  * cmd 回读校验：pin 是否被 command manager 覆盖（覆盖则该点作废，不硬凑）

判据（写死，可证伪）：
  track_ratio = 沿指令方向的实测速度 / 指令速率（线性点）
  yaw_ratio   = 实测角速度 / 指令角速度（纯转向点）
  "可行" = 主比值 ∈ [0.7, 1.3] 且未跌倒；"部分" ∈ [0.3,0.7)；"失效" < 0.3

用法：
  python scripts/analyze_coverage.py <coverage_*.npz> --csv out.csv
"""
import argparse
import numpy as np


def analyze(npz_path, csv_path):
    d = np.load(npz_path, allow_pickle=True)
    dt = float(d["dt"])
    labels = d["labels"]                 # (N,) object
    grid = d["grid"]                     # (N,3) vx,vy,wz
    vx = d["lin_vel_x"]                  # (T,N)
    vy = d["lin_vel_y"]
    wz = d["ang_vel_z"]
    z = d["root_pos_z"]
    cmd_rb = d["cmd_rb"]                 # (T,N,3)
    T, N = vx.shape

    warm = int(2.5 / dt)                 # 125 步启动暂态
    warm = min(warm, T // 3)
    fall_z = 0.40
    s = slice(warm, T)

    rows = []
    for i in range(N):
        lab = str(labels[i])
        cvx, cvy, cwz = [float(x) for x in grid[i]]
        speed_cmd = float(np.hypot(cvx, cvy))

        # cmd 回读：稳态窗口均值，检查 pin 是否生效
        rb = cmd_rb[s, i, :].mean(axis=0)
        rb_err = float(np.max(np.abs(rb - grid[i])))

        # 实测稳态速度矢量（时间均值，避免瞬时抖动）
        vax = float(vx[s, i].mean())
        vay = float(vy[s, i].mean())
        waz = float(wz[s, i].mean())
        zmin = float(z[s, i].min())
        fell = bool((z[s, i] < fall_z).mean() > 0.05)

        if speed_cmd > 1e-3:
            u = grid[i, :2] / speed_cmd
            proj = float(vax * u[0] + vay * u[1])       # 沿指令方向投影
            track = proj / speed_cmd                     # 主比值
            lateral = float(np.hypot(vax - proj * u[0], vay - proj * u[1]))
            main = track
        else:                                            # 纯转向 / 站立
            proj = float("nan")
            lateral = float(np.hypot(vax, vay))
            track = float("nan")
            main = (waz / cwz) if abs(cwz) > 1e-3 else 0.0

        # 站立点特殊：比值无意义，用"是否静止"判
        if lab == "stand":
            cls = "静止" if lateral < 0.15 and abs(waz) < 0.2 else "漂移/自发运动"
        elif fell:
            cls = "跌倒"
        elif abs(main) >= 0.7 and abs(main) <= 1.3:
            cls = "可行"
        elif abs(main) >= 0.3:
            cls = "部分"
        else:
            cls = "失效"

        rows.append(dict(
            env=i, label=lab,
            cmd_vx=cvx, cmd_vy=cvy, cmd_wz=cwz,
            act_vx=vax, act_vy=vay, act_wz=waz,
            proj=proj, track_ratio=main, lateral=lateral,
            z_min=zmin, fell=int(fell), cmd_readback_err=rb_err, cls=cls,
        ))

    # 写 CSV
    cols = ["env", "label", "cmd_vx", "cmd_vy", "cmd_wz",
            "act_vx", "act_vy", "act_wz", "proj", "track_ratio",
            "lateral", "z_min", "fell", "cmd_readback_err", "cls"]
    if csv_path:
        with open(csv_path, "w", encoding="utf-8") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(
                    (f"{r[c]:.4f}" if isinstance(r[c], float) else str(r[c]))
                    for c in cols) + "\n")
        print(f"[OK] CSV -> {csv_path}")

    # 控制台摘要
    print(f"\n=== 指令可行域扫描（T={T}, N={N}, 稳态窗口[{warm},{T}), dt={dt}）===\n")
    print(f"  {'label':<10}{'cmd(vx,vy,wz)':<22}{'act_vx':>8}{'act_vy':>8}{'act_wz':>8}"
          f"{'ratio':>8}{'z_min':>8}{'rbErr':>8}  cls")
    for r in rows:
        cmd = f"({r['cmd_vx']:.2f},{r['cmd_vy']:.2f},{r['cmd_wz']:.2f})"
        print(f"  {r['label']:<10}{cmd:<22}{r['act_vx']:>8.3f}{r['act_vy']:>8.3f}"
              f"{r['act_wz']:>8.3f}{r['track_ratio']:>8.2f}{r['z_min']:>8.3f}"
              f"{r['cmd_readback_err']:>8.3f}  {r['cls']}")

    # 分组统计（逐组，不合成总能力分）
    print("\n=== 分组汇总（各组独立，绝不求全局平均）===")
    groups: dict[str, list] = {}
    for r in rows:
        groups.setdefault(r["label"], []).append(r)
    for g, rs in groups.items():
        n_ok = sum(1 for r in rs if r["cls"] == "可行")
        n_fail = sum(1 for r in rs if r["cls"] in ("失效", "跌倒"))
        ratios = [r["track_ratio"] for r in rs
                  if np.isfinite(r["track_ratio"]) and abs(r["cmd_wz"]) < 1e-3]
        mean_r = float(np.mean(ratios)) if ratios else float("nan")
        print(f"  {g:<10} n={len(rs):>2}  可行={n_ok}  失效/跌倒={n_fail}  "
              f"线速点平均比值={mean_r:.2f}")

    # pin 生效校验
    max_rb = max(r["cmd_readback_err"] for r in rows)
    print(f"\n[校验] cmd 回读最大偏差 = {max_rb:.4f} "
          f"({'OK: pin 生效' if max_rb < 0.05 else '⚠ 指令被覆盖，结果作废'})")
    return rows


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    analyze(a.npz, a.csv)
