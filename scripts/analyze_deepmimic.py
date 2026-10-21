"""
19 - DeepMimic 步态分析（按瞬时参考速度分箱）
=============================================================================
为什么不能直接用 compare_gait.py：
   那个脚本按 **env 分组**，因为 AMP 侧每个 env 的速度指令是钉死的常量。
   DeepMimic 没有指令，速度来自参考动作本身，同一个 env 在 30 秒里会从
   走过渡到跑，速度一直在变。按 env 分组等于把所有速度混在一起平均，
   那样 duty factor 曲线就是一条平线，什么都看不出来。

 substitut 方案：滑动窗口 + 按窗口平均参考速度分箱
   1) 用 2.5s 窗口、1s 步长滑动，每步算一个 (参考速度, 步态指标) 样本
   2) 把样本按参考速度分箱到 0.5/1.0/.../3.0 六档
   3) 输出的列名与 compare_gait.py 完全一致，可以并排放进同一张表

这样两边问的是同一个问题：
       AMP        —— 我命令它跑这么快，它的步态长什么样
       DeepMimic  —— 参考动作让它跑这么快，它的步态长什么样
   都是"步态形状是否随速度连续变化"。

多余的三项只有 DeepMimic 有（它的主场指标）：
   joint_err    关节角跟踪误差（度）  —— 姿势复现精度
   root_pos_err 根位置跟踪误差（m）   —— 轨迹复现精度
   ref_follow   参考速度 vs 实际速度  —— 它有没有真的跟上参考
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from analyze_gait import detect_strides  # noqa: E402

WIN_SEC = 2.5
HOP_SEC = 1.0
WARMUP_SEC = 5.0
BINS = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
BIN_TOL = 0.35


def windowed_samples(npz_path: Path):
    """切成 (参考速度, 指标) 样本流。"""
    d = np.load(npz_path, allow_pickle=True)
    dt = float(d["dt"])
    mass = float(d["mass"])
    T, N = d["lin_vel_x"].shape

    W = int(WIN_SEC / dt)
    HOP = int(HOP_SEC / dt)
    W0 = int(WARMUP_SEC / dt)

    foot_z = d["foot_z"]
    # 支撑阈值对整个序列算一次，保证窗口间口径一致（和 compare_gait.py 同算法）
    thr = np.percentile(foot_z, 15, axis=(0, 1)) + 0.03

    samples = []
    for i in range(N):
        for s in range(W0, T - W, HOP):
            seg_z = foot_z[s:s + W, i, :]
            stance = seg_z < thr

            ref_v = float(d["ref_speed"][s:s + W, i].mean())
            vx = d["lin_vel_x"][s:s + W, i]
            vy = d["lin_vel_y"][s:s + W, i]

            per_foot = []
            for k in range(foot_z.shape[2]):
                cnt, duty, _, _ = detect_strides(stance[:, k], dt)
                if cnt >= 1:
                    per_foot.append((cnt / (W * dt), duty))

            power = np.abs(d["torque"][s:s + W, i, :] * d["joint_vel"][s:s + W, i, :]).sum(axis=1)
            energy = float(power.sum() * dt)
            dist = float(vx.mean()) * W * dt

            samples.append(dict(
                v=ref_v,
                duty=float(np.mean([p[1] for p in per_foot])) if per_foot else float("nan"),
                freq=float(np.mean([p[0] for p in per_foot])) if per_foot else float("nan"),
                act_v=float(vx.mean()),
                drift=float(np.abs(vy).mean()),
                energy=energy,
                dist=dist,
                joint_err=float(d["joint_err"][s:s + W, i].mean()),
                root_err=float(d["root_pos_err"][s:s + W, i].mean()),
            ))
    return d, dt, mass, samples


def fmt(v, nd=3):
    return "nan" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{nd}f}"


def main() -> int:
    if len(sys.argv) < 2:
        print("用法: python analyze_deepmimic.py <npz路径>")
        return 1

    d, dt, mass, samples = windowed_samples(Path(sys.argv[1]))
    vs = np.array([s["v"] for s in samples])

    print(f"=== DeepMimic 步态分析（滑动窗口分箱） ===")
    print(f"样本数 = {len(samples)}   dt = {dt}")
    print(f"参考速度范围 = {vs.min():.2f} ~ {vs.max():.2f} m/s  "
          f"（均值 {vs.mean():.2f}）")
    print("分位: " + " ".join(f"{q}%={np.percentile(vs, q):.2f}" for q in (5, 25, 50, 75, 95)))

    print("\n### 按瞬时参考速度分箱")
    print("| 参考速度 (m/s) | 支撑占比 | 步频 (1/s) | 步长 (m) | CoT | 实际速度 | 侧向漂移 | 关节误差(°) | 根位置误差(m) | 样本数 |")
    print("|" + "|".join(["---"] * 10) + "|")

    rows = []
    for b in BINS:
        sel = [s for s in samples if abs(s["v"] - b) <= BIN_TOL]
        if not sel:
            print(f"| {b:.1f} | - | - | - | - | - | - | - | - | 0 |")
            continue
        duty = np.nanmean([s["duty"] for s in sel])
        freq = np.nanmean([s["freq"] for s in sel])
        act_v = np.mean([s["act_v"] for s in sel])
        drift = np.mean([s["drift"] for s in sel])
        je = np.mean([s["joint_err"] for s in sel])
        re_ = np.mean([s["root_err"] for s in sel])
        # CoT：先把每窗口的 e/(mgd) 算好再平均，避免小位移窗口把均值拉爆
        cots = [s["energy"] / (mass * 9.81 * abs(s["dist"]))
                for s in sel if abs(s["dist"]) > 0.05]
        cot = float(np.mean(cots)) if cots else float("nan")
        stride_len = 2.0 * act_v / freq if freq and not np.isnan(freq) and freq > 1e-6 else float("nan")
        print(f"| {b:.1f} | {fmt(duty)} | {fmt(freq,2)} | {fmt(stride_len,2)} | {fmt(cot,2)} | "
              f"{fmt(act_v)} | {fmt(drift)} | {fmt(np.degrees(je),2)} | {fmt(re_)} | {len(sel)} |")
        rows.append((b, duty))

    ok = [(b, du) for b, du in rows if not np.isnan(du)]
    if len(ok) >= 3:
        xs = [b for b, _ in ok]
        ys = [du for _, du in ok]
        rx = np.argsort(np.argsort(xs))
        ry = np.argsort(np.argsort(ys))
        rho = np.corrcoef(rx, ry)[0, 1]
        print(f"\n支撑占比 vs 参考速度 秩相关 = {rho:.2f}（应≈-1 才说明存在走→跑过渡）")

    # DeepMimic 主场：它本来就是干这个的，这两个指标是它的定义性目标
    all_je = [s["joint_err"] for s in samples]
    all_re = [s["root_err"] for s in samples]
    print(f"\n全程平均关节跟踪误差 = {np.degrees(np.mean(all_je)):.2f} °")
    print(f"全程平均根位置跟踪误差 = {np.mean(all_re):.3f} m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
