#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
下载已 retarget 到 Unitree G1 的动捕数据（国内服务器请用 HF 镜像）。

用法：
    export HF_ENDPOINT=https://hf-mirror.com          # 国内机器务必设
    python data/download_lafan1.py --out /root/autodl-tmp/motions

数据源（按优先级）：
  1. lvhaidong/LAFAN1_Retargeting_Dataset
     - 宇树官方 LAFAN1 retarget 结果的第三方镜像（官方源已下架）
     - 结构大致是 LAFAN1/g1/*.csv，走/跑/转身/跳舞都有
  2. ember-lab-berkeley/AMASS_Retargeted_for_G1
     - AMASS retarget 到 G1 的结果，动作库更大，可选补充

注意：宇树官方 CSV 的关节顺序必须和训练用的 G1 USD 一致，
     下完后一定要跑 scripts/verify_motion.py 体检，别直接开训。
"""
import argparse
import os
import sys

# 国内服务器默认走镜像
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

REPOS = {
    "lafan1": "lvhaidong/LAFAN1_Retargeting_Dataset",
    "amass": "ember-lab-berkeley/AMASS_Retargeted_for_G1",
}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./motions", help="下载到哪个目录")
    ap.add_argument("--only", choices=list(REPOS.keys()) + ["all"], default="lafan1")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("需要 huggingface_hub，先装：  pip install -U huggingface_hub hf_transfer")
        return 1

    os.makedirs(args.out, exist_ok=True)
    targets = [args.only] if args.only != "all" else list(REPOS.keys())

    for key in targets:
        repo = REPOS[key]
        dst = os.path.join(args.out, key)
        print(f"\n>>> downloading {repo} -> {dst}")
        try:
            snapshot_download(
                repo_id=repo,
                repo_type="dataset",
                local_dir=dst,
                max_workers=args.workers,
            )
            print(f"    OK: {dst}")
        except Exception as e:  # noqa: BLE001
            print(f"    FAILED: {e}")
            print("    国内机器如果超时，确认已设 HF_ENDPOINT=https://hf-mirror.com")
            return 1

    # 简单列一下拿到了什么
    print("\n=== 下载结果 ===")
    for key in targets:
        dst = os.path.join(args.out, key)
        for root, _dirs, files in os.walk(dst):
            csvs = [f for f in files if f.endswith((".csv", ".npz", ".pkl"))]
            if csvs:
                print(f"{key}: {root}  ({len(csvs)} files)  e.g. {csvs[:5]}")

    print("\n下一步：python scripts/verify_motion.py --dir <下载目录>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
