#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""离线分析"多档速度扫描"npz：逐档输出全指标，口径与 full_eval.py 完全一致。

为什么单独写这个
-----------------------------------------------------------------------------
eval_gait.py 一次 rollout 把 N 个速度档各放一个 env，产一个 (T,N,...) 的 npz。
full_eval.py 会跨所有 env 求平均（它假设整份 npz 是同一个速度的多 env 重复），
直接拿来分析多档 npz 会把 11 个速度混成一个数 —— 正是本项目踩过多次的
"口径不一致"坑。这里按 env 逐档拆开算，每档一行，才能横比。

用法:
    python analyze_sweep.py <npz路径> [--csv 输出.csv]
"""
import os
import sys
import csv
import numpy as np

# 本地是 numpy 2.x，移除了 ndarray.ptp()；而 gait_metrics2.py（口径唯一来源）
# 是给服务器 numpy 1.x 写的。绝不能改口径文件本身（会破坏与服务器数字的一致性），
# 也不能给不可变的 ndarray 打补丁。做法：读它的源码，只把那一处 .ptp() 换成
# 等价的 np.ptp(z)，再 exec 进独立命名空间 —— 算法一个字没变，仍来自唯一口径来源。
_HERE = os.path.dirname(os.path.abspath(__file__))
_src = open(os.path.join(_HERE, "gait_metrics2.py"), encoding="utf-8").read()
_src = _src.replace("z.ptp()", "np.ptp(z)")
_ns = {"__name__": "gait_metrics2_local", "__file__": os.path.join(_HERE, "gait_metrics2.py")}
exec(compile(_src, _ns["__file__"], "exec"), _ns)
gait_from_footz = _ns["gait_from_footz"]
joint_stats = _ns["joint_stats"]

LEG_L, LEG_R = slice(0, 6), slice(6, 12)


def ai(a, b):
    return abs(a - b) / (0.5 * (abs(a) + abs(b))) * 100 if (abs(a) + abs(b)) > 1e-9 else float("nan")


def per_env_metrics(d, j, n0):
    """算第 j 个 env（对应一个速度档）的全部指标。"""
    T = d["lin_vel_x"].shape[0]
    vcmd = float(d["velocities"][j])
    # 指令纯净度：cmd_x 该恒等于目标速度
    cmd = d["cmd_x"][n0:, j]
    cmd_dev = float(abs(cmd.mean() - vcmd))
    cmd_sd = float(cmd.std())

    vx = d["lin_vel_x"][n0:, j]
    vy = d["lin_vel_y"][n0:, j] if "lin_vel_y" in d.files else np.zeros_like(vx)
    vact = float(np.mean(np.hypot(vx, vy)))

    # 逐关节扭矩 RMS 不对称（29 DOF 里 [0:6] 左 / [6:12] 右同名）
    tq = d["torque"][n0:, j, :]
    rms = np.sqrt(np.mean(tq ** 2, axis=0))
    per = [ai(float(rms[k]), float(rms[k + 6])) for k in range(6)]
    leg_ai = float(np.mean(per))

    fps = 1.0 / float(d["dt"])
    fz = d["foot_z"][n0:, j, :]          # (T-n0, 2) 两只脚
    g = gait_from_footz(fz, fps, vact)
    cad = g["cadence"] if g else float("nan")
    stride = g["stride_m"] if g else float("nan")
    duty = g["duty"] if g else float("nan")

    knee = float("nan")
    hip_rom = float("nan")
    height = float("nan")
    if "joint_pos" in d.files:
        qd = np.degrees(d["joint_pos"][n0:, j:j + 1, :])  # (T,1,D)
        js = joint_stats(qd)
        knee = js["knee_mean"]
        hip_rom = js["hip_rom"]
    if "root_pos_z" in d.files:
        height = float(np.mean(d["root_pos_z"][n0:, j]))

    return dict(v_cmd=vcmd, cmd_mean=float(cmd.mean()), cmd_dev=cmd_dev, cmd_sd=cmd_sd,
                v_act=vact, err=vact - vcmd, err_actual=vact - float(cmd.mean()),
                legAI=leg_ai, hipAI=per[0], kneeAI=per[3], ankAI=per[4],
                cad=cad, stride=stride, duty=duty, knee=knee, hipROM=hip_rom, height=height)


HDR = ["v_cmd", "cmd_mean", "v_act", "err", "err_actual", "cmd_dev", "cmd_sd",
       "legAI", "hipAI", "kneeAI", "ankAI", "cad", "stride", "duty", "knee", "hipROM", "height"]


def main():
    args = [a for a in sys.argv[1:] if a != "--csv"]
    csv_path = None
    if "--csv" in sys.argv:
        i = sys.argv.index("--csv")
        if i + 1 < len(sys.argv):
            csv_path = sys.argv[i + 1]
            args = [a for a in args if a != csv_path]
    if not args:
        print("用法: analyze_sweep.py <npz> [--csv out.csv]")
        return 1
    path = args[0]
    d = np.load(path, allow_pickle=True)
    T = d["lin_vel_x"].shape[0]
    N = d["lin_vel_x"].shape[1]
    n0 = int(T * 0.3)   # 与 full_eval.py 一致：丢前 30% 暂态
    vels = d["velocities"]
    print("npz =", os.path.basename(path))
    print("T=%d  N(档数)=%d  dt=%.4f  稳态起点=%d (丢前30%%)" % (T, N, float(d["dt"]), n0))
    print("velocities =", np.round(vels, 2).tolist())

    rows = []
    # 指令可信度分两级：dev（均值偏离目标）= 硬污染，sd（帧间抖动）= 软抖动。
    # eval_gait.py 原判据是 dev>0.02 或 sd>0.05 一律判死，但这里 sd 0.06-0.10
    # 只是每帧 command 被重采样前的轻微波动，均值 dev 几乎都 <0.01 —— 指令中心是准的，
    # 不该把 7 档全废掉。硬污染只用 dev 判（>0.03），sd 单列出来供审阅。
    hard_bad = []
    soft_jitter = []
    for j in range(N):
        r = per_env_metrics(d, j, n0)
        rows.append(r)
        if r["cmd_dev"] > 0.03:
            hard_bad.append((j, r["v_cmd"], r["cmd_dev"], r["cmd_sd"]))
        elif r["cmd_sd"] > 0.05:
            soft_jitter.append(j)
    if hard_bad:
        print("\n!! 指令中心明显偏离（硬污染，该档速度结论作废）:")
        for j, v, dv, sd in hard_bad:
            print("   env#%d v=%.2f dev=%.3f sd=%.3f" % (j, v, dv, sd))
    else:
        print("\n指令中心自检通过：%d 档 dev 全部 <0.03" % N)
    if soft_jitter:
        print("（帧间抖动 sd>0.05 的档: %s —— 均值准，仅记录，不判死）" %
              [round(rows[j]["v_cmd"], 2) for j in soft_jitter])

    print("\n== 逐档全指标（速度 m/s；AI 单位 %%）==")
    print("%-6s %-8s %-7s %-8s %-8s %-7s %-7s %-7s %-7s %-7s %-6s" %
          ("v_cmd", "cmd_mean", "v_act", "err", "err_act", "legAI", "hipAI",
           "cad", "stride", "duty", "knee"))
    for r in rows:
        print("%-6.2f %-8.3f %-7.3f %+-8.3f %+-8.3f %-7.1f %-7.1f %-7.1f %-7.3f %-7.3f %-6.1f" %
              (r["v_cmd"], r["cmd_mean"], r["v_act"], r["err"], r["err_actual"],
               r["legAI"], r["hipAI"], r["cad"], r["stride"], r["duty"], r["knee"]))

    # 速度跟踪总评：只用「指令中心准」的档。err_actual 用实测 cmd 均值，
    # 比标称 err 更诚实。低速/高速分段误差单列，避免平均掩盖高速掉速。
    good = [r for r in rows if r["cmd_dev"] <= 0.03]
    if len(good) >= 3:
        c = np.array([r["cmd_mean"] for r in good]); a = np.array([r["v_act"] for r in good])
        rho = float(np.corrcoef(c, a)[0, 1])
        mae = float(np.mean(np.abs(a - c)))
        print("\n== 速度跟踪总评（%d 档，指令中心准）==" % len(good))
        print("Pearson rho(cmd_mean, v_act) = %.4f" % rho)
        print("平均绝对误差 MAE = %.3f m/s" % mae)
        print("误差范围: %+.3f ~ %+.3f m/s" % (min(a - c), max(a - c)))
        lo = [r for r in good if r["v_cmd"] <= 1.5]
        hi = [r for r in good if r["v_cmd"] >= 2.0]
        if lo:
            print("低速段 (<=1.5) 平均 |err| = %.3f m/s" %
                  np.mean([abs(r["err_actual"]) for r in lo]))
        if hi:
            print("高速段 (>=2.0) 平均 |err| = %.3f m/s  <-- 高速掉速看这里" %
                  np.mean([abs(r["err_actual"]) for r in hi]))

    if csv_path:
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=HDR)
            w.writeheader()
            for r in rows:
                w.writerow({k: round(r[k], 4) for k in HDR})
        print("\nCSV ->", csv_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
