import re
import sys

WANT = [200, 600, 1000, 1400, 1800, 2000, 3000, 4000, 5000]

for path in sys.argv[1:]:
    pairs = []
    cur = None
    with open(path, "r", errors="ignore") as fh:
        for line in fh:
            m = re.search(r"Learning iteration (\d+)", line)
            if m:
                cur = int(m.group(1))
                continue
            m2 = re.search(r"error_vel_xy:\s*([\d.]+)", line)
            if m2 and cur is not None:
                pairs.append((cur, float(m2.group(1))))
                cur = None
    best = {}
    for it, err in pairs:
        if it in WANT:
            best[it] = err
    print(f"--- {path.split('/')[-1]}  (共 {len(pairs)} 条)")
    for it in WANT:
        if it in best:
            print(f"    iter {it:>5} -> {best[it]:.4f}")
