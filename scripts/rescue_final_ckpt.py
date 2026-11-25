# -*- coding: utf-8 -*-
# 路线 B：服务器放手前，抢救 v2.3 最终模型的「可复现最小集」到本地。
# 带 md5 闭环校验：服务器先算 md5 -> SFTP 拉回 -> 本地重算 -> 逐一比对。
#
# 用法： python scripts/rescue_final_ckpt.py
# 依赖： scripts/remote.py (paramiko)，.secrets/ssh.json
#
# 为什么不止拉 model_2999.pt：
#   复现/续训需要当时的 env.yaml / agent.yaml（网络结构与观测维度）、
#   exported/policy.onnx（部署用导出策略）、git/rsl_rl.diff（代码改动），
#   以及续训起点 model_1600.pt。只留一个 .pt 无法可靠复现。
import hashlib
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import load_conf, connect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
DST = ROOT / "backup" / "checkpoints" / "v23_final_2999"
RUN = "/data/legged_lab/logs/rsl_rl/g1_amp/2026-11-18_15-40-01"

# (服务器相对路径, 本地落地文件名)
FILES = [
    ("model_2999.pt", "model_2999.pt"),
    ("params/env.yaml", "env.yaml"),
    ("params/agent.yaml", "agent.yaml"),
    ("exported/policy.onnx", "policy.onnx"),
    ("git/rsl_rl.diff", "rsl_rl.diff"),
    ("model_1600.pt", "model_1600_resume_start.pt"),
]


def md5_of(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    cli = connect(load_conf())
    try:
        # 1) 服务器端一次性算全部 md5
        rpaths = [f"{RUN}/{rel}" for rel, _ in FILES]
        _in, out, err = cli.exec_command("md5sum " + " ".join(rpaths), timeout=60)
        txt = out.read().decode("utf-8", "replace")
        etxt = err.read().decode("utf-8", "replace").strip()
        if etxt:
            print("[server stderr]", etxt)
        remote_md5 = {}
        for line in txt.splitlines():
            parts = line.split()
            if len(parts) == 2:
                remote_md5[parts[1]] = parts[0]

        DST.mkdir(parents=True, exist_ok=True)
        sftp = cli.open_sftp()
        results = []
        all_ok = True
        for rel, lname in FILES:
            rp = f"{RUN}/{rel}"
            lp = DST / lname
            r_md5 = remote_md5.get(rp)
            if r_md5 is None:
                print(f"[MISS] {rel}: 服务器无 md5（文件缺失？）")
                results.append((rel, "MISSING", "", "", False))
                all_ok = False
                continue
            sftp.get(rp, str(lp))
            l_md5 = md5_of(lp)
            size = lp.stat().st_size
            ok = (l_md5 == r_md5)
            all_ok = all_ok and ok
            results.append((rel, lname, size, l_md5, ok))
            print(f"[{'OK' if ok else 'FAIL'}] {rel:22s} -> {lname:28s} "
                  f"{size/1024/1024:7.2f} MB  md5={l_md5}  {'==' if ok else '!= server ' + r_md5}")
        sftp.close()

        # 2) 写一份机器可读清单（本地留档，云端靠 docs 记录）
        manifest = DST / "RESCUE_MANIFEST.txt"
        lines = [
            "v2.3 最终模型抢救清单 (路线 B)",
            f"server run: {RUN}",
            "server host: jq1.9gpu.com:12250  (RTX 4090, /data)",
            "",
            "文件 | 本地名 | 字节 | md5 | 校验",
        ]
        for rel, lname, size, m, ok in results:
            lines.append(f"{rel} | {lname} | {size} | {m} | {'OK' if ok else 'FAIL'}")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")

        print()
        print("=== 结果 ===")
        print(f"落地目录: {DST}")
        print(f"清单文件: {manifest}")
        print("ALL MD5 VERIFIED" if all_ok else "!! 有文件 md5 不一致或缺失")
        return 0 if all_ok else 1
    finally:
        cli.close()


if __name__ == "__main__":
    sys.exit(main())
