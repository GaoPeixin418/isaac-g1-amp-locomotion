"""把服务器上的文件拉回本地。

用法：
    python scripts/fetch_any.py <remote_path> [remote_path2 ...]
本地路径自动映射为 backup/<远程末级目录名>/<文件名>，
例如 /data/eval_out/gait.npz -> backup/eval_out/gait.npz

注意：不要加 -d 之类的短参数 —— PowerShell 会把它们解释成自己的参数吃掉。
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import load_conf, connect  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
cli = connect(load_conf())
sftp = cli.open_sftp()
for remote in sys.argv[1:]:
    # 不要用 Path() 解析远端路径 —— Windows 的 pathlib 会把 "/data/x" 变成
    # "\data\x"，stat 直接 ENOENT。远端路径一律用纯字符串处理。
    remote = remote.replace("\\", "/")
    parts = [p for p in remote.split("/") if p]
    fname = parts[-1]
    parent = parts[-2] if len(parts) >= 2 else "server"
    local = ROOT / "backup" / parent / fname
    local.parent.mkdir(parents=True, exist_ok=True)
    size = sftp.stat(remote).st_size
    print(f"download {remote}  {size/1024/1024:.2f} MB")
    sftp.get(remote, str(local))
    print("  ->", local)
sftp.close()
cli.close()
print("DONE")
