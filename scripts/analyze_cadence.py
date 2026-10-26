"""步频 / 步长 / 支撑占比 诊断：把"走得匆忙、一顿一顿"翻译成可证伪量。

用法（服务器上）：
    /data/envs/isaaclab/bin/python /data/setup/analyze_cadence.py

口径（与项目约定一致，别再踩坑）：
  freq      = 单脚每秒着地次数 = gait cycle 频率 (Hz)
  stride    = v / freq          （一个 cycle 走的距离，不能乘 2）
  step      = stride / 2
  cadence   = 2 * freq * 60     （双腿 steps/min，文献口径）
  duty      = 单脚着地时长 / cycle 时长

人类参照（成人，腿长 0.90 m）——文献锚点，自洽校验 v = cadence/60 * stride/2：
  v(m/s)   0.8    1.0    1.2    1.4    1.6
  cadence  95     102    110    116    122   (steps/min)
  stride   1.01   1.18   1.31   1.45   1.57  (m)
动力相似缩放到 G1（腿长 L_G1）：
  同 Froude -> v_h = v * sqrt(L_HU / L_G1)
  频率 ∝ 1/sqrt(L) -> freq_G1 = freq_h * sqrt(L_HU / L_G1)
"""

import glob
import os

import numpy as np

G = 9.81
L_G1 = 0.68  # G1 大腿 0.32 + 小腿 0.32 + 足踝，腿长取 0.68 m
L_HU = 0.90

# 人类锚点
V_H = np.array([0.8, 1.0, 1.2, 1.4, 1.6])
CAD_H = np.array([95.0, 102.0, 110.0, 116.0, 122.0])  # steps/min
FREQ_H = CAD_H / 120.0  # gait cycle Hz


def freq_human_at(v_g1):
    """同 Froude 数下，缩放到 G1 腿长的期望步频 (Hz)。"""
    v_h = v_g1 * np.sqrt(L_HU / L_G1)
    f_h = np.interp(v_h, V_H, FREQ_H, left=FREQ_H[0], right=FREQ_H[-1])
    return f_h * np.sqrt(L_HU / L_G1)


def step_len_human_at(v_g1):
    """同 Froude 数下，缩放到 G1 腿长的期望步长 (m，单步)。"""
    v_h = v_g1 * np.sqrt(L_HU / L_G1)
    stride_h = v_h / np.interp(v_h, V_H, FREQ_H, left=FREQ_H[0], right=FREQ_H[-1])
    stride_g1 = stride_h * (L_G1 / L_HU)  # 长度 ∝ 腿长
    return stride_g1 / 2.0


def contact_runs(force, thr):
    """返回 (onset_idx, offset_idx) 配对，只保留完整着地段。"""
    c = force > thr
    if c.sum() == 0 or (~c).sum() == 0:
        return np.array([]), np.array([])
    on = np.where(~c[:-1] & c[1:])[0] + 1
    off = np.where(c[:-1] & ~c[1:])[0] + 1
    if c[0]:
        off = off[1:] if len(off) and off[0] < 1 else off
        on = np.concatenate([[0], on]) if not c[0] == False else on
    # 逐段配对：每段 onset 后紧跟的第一个 offset
    pairs_on, pairs_off = [], []
    j = 0
    for i in on:
        while j < len(off) and off[j] <= i:
            j += 1
        if j < len(off):
            pairs_on.append(i)
            pairs_off.append(off[j])
            j += 1
    return np.array(pairs_on), np.array(pairs_off)


def analyze(path, warmup=0.3):
    d = np.load(path, allow_pickle=True)
    dt = float(d["dt"])
    vels = d["velocities"]
    lvx, lvy = d["lin_vel_x"], d["lin_vel_y"]
    fz = d["foot_force_z"]
    n = int(len(lvx) * warmup)

    rows = []
    for j in range(len(vels)):
        vx, vy = lvx[n:, j], lvy[n:, j]
        v = float(np.mean(np.hypot(vx, vy)))
        ff = fz[n:, j, :]
        # 阈值：用该 env 力信号的分位数，避免整体量纲差异把阈值带偏
        p90 = np.percentile(ff, 90)
        thr = max(0.15 * p90, 1e-3)

        freqs, dutys = [], []
        for k in range(2):
            on, off = contact_runs(ff[:, k], thr)
            if len(on) < 3:
                continue
            dur = (on[1:] - on[:-1]) * dt
            stance = (off[:-1] - on[:-1]) * dt
            ok = (dur > 0.15) & (dur < 2.0) & (stance > 0.02) & (stance < dur)
            if ok.sum() < 2:
                continue
            freqs.append(1.0 / np.mean(dur[ok]))
            dutys.append(np.mean(stance[ok] / dur[ok]))
        if not freqs:
            rows.append((vels[j], v, np.nan, np.nan, np.nan, np.nan))
            continue
        f = float(np.mean(freqs))
        duty = float(np.mean(dutys))
        stride = v / f if f > 0 else np.nan
        f_ref = freq_human_at(v)
        rows.append((vels[j], v, f, stride, duty, f / f_ref))
    return rows


def main():
    files = sorted(glob.glob("/data/eval_out/gait_*.npz"))
    keep = ["amp2000", "amp_2000", "amp_4000_clean", "amp_model_4000", "lam025", "lam050",
            "lam075", "ppo_baseline", "deepmimic"]
    hdr = f"{'模型':<16}{'指令':>6}{'实测v':>7}{'步频Hz':>8}{'cadence':>9}" \
          f"{'步长m':>8}{'人步长':>8}{'步频比':>8}{'duty':>7}"
    print(hdr)
    print("-" * 80)
    for p in files:
        b = os.path.basename(p)[5:-4]
        if not any(k in b for k in keep):
            continue
        try:
            rows = analyze(p)
        except Exception as e:
            print(f"{b:<16} 解析失败: {e}")
            continue
        for cmd, v, f, st, duty, ratio in rows:
            if np.isnan(f):
                print(f"{b:<16}{cmd:>6.1f}{v:>7.2f}   -- 接触检测失败 --")
                continue
            sl = st / 2.0
            sl_ref = step_len_human_at(v)
            flag = "" if ratio < 1.15 else ("  碎步" if ratio < 1.5 else "  严重碎步")
            print(f"{b:<16}{cmd:>6.1f}{v:>7.2f}{f:>8.2f}{2*f*60:>9.0f}{sl:>8.2f}"
                  f"{sl_ref:>8.2f}{ratio:>8.2f}{duty:>7.2f}{flag}")
    print("-" * 80)
    print("步频比 = 实测步频 / 同 Froude 数下人类步频（已按腿长缩放）")
    print("  =1 与人一致；>1.15 碎步（匆忙）；>1.5 严重碎步")
    print("duty: 走 ~0.6, 跑 ~0.3, 站立 ~1.0")


if __name__ == "__main__":
    main()
