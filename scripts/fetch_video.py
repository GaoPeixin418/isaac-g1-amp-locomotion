"""把服务器上的 play 视频拉回本地 backup/video/"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from remote import load_conf, connect  # noqa: E402

REMOTE = "/data/legged_lab/logs/rsl_rl/g1_amp/2026-10-19_16-37-21/videos/play"
LOCAL = Path(__file__).resolve().parent.parent / "backup" / "video"

cli = connect(load_conf())
sftp = cli.open_sftp()
try:
    files = sftp.listdir(REMOTE)
except IOError as exc:
    print("LIST FAIL", exc)
    files = []
print("remote files:", files)

LOCAL.mkdir(parents=True, exist_ok=True)
for name in files:
    if not name.endswith(".mp4"):
        continue
    src = f"{REMOTE}/{name}"
    size = sftp.stat(src).st_size
    dst = LOCAL / name
    if dst.exists() and dst.stat().st_size == size:
        print(f"skip (same size) {name} {size}")
        continue
    print(f"download {name}  {size/1024/1024:.2f} MB ...")
    sftp.get(src, str(dst))
    print("  ->", dst)
sftp.close()
cli.close()
print("DONE")
