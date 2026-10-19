"""快速肉眼检查 npz 里每个数组的形状与取值，用于定位采集 bug"""
import sys
import numpy as np

data = np.load(sys.argv[1], allow_pickle=True)
for k in data.files:
    a = data[k]
    print(f"{k:14s} shape={str(a.shape):18s} dtype={a.dtype}", end="  ")
    if a.size > 0 and a.dtype.kind == "f":
        fin = a[np.isfinite(a)]
        print(f"min={fin.min():.4f} max={fin.max():.4f} mean={fin.mean():.4f}")
    else:
        print(a)

print("\n--- 每个 env 的接触力分布 ---")
if "foot_force_z" in data.files:
    fz = np.abs(data["foot_force_z"])
    for i in range(fz.shape[1]):
        vals = fz[10:, i, :]
        print(f"  env{i}: 非零比例(|fz|>1)={np.mean(vals>1):.3f}  "
              f"|>20={np.mean(vals>20):.3f}  |>100={np.mean(vals>100):.3f}  max={vals.max():.1f}")

print("\n--- 每个 env 的足端高度 ---")
if "foot_z" in data.files:
    z = data["foot_z"]
    for i in range(z.shape[1]):
        print(f"  env{i}: min={z[:,i,:].min():.4f} max={z[:,i,:].max():.4f} "
              f"mean={z[:,i,:].mean():.4f}")

print("\n--- 实际前进速度 ---")
vx = data["lin_vel_x"]
for i in range(vx.shape[1]):
    print(f"  env{i}: mean={vx[:,i].mean():.3f} m/s   cmd={data['velocities'][i]}")
