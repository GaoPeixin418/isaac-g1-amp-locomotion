"""纯 stdlib 解析 mp4 的 mvhd(时长/timescale) 与 tkhd(宽高)，不依赖 ffmpeg。"""
import struct
import sys
from pathlib import Path


def walk(f, end, depth=0):
    """遍历 box，遇到 moov/trak 容器递归。"""
    while f.tell() < end:
        head = f.read(8)
        if len(head) < 8:
            return
        size, typ = struct.unpack(">I4s", head)
        typ = typ.decode("latin-1")
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr = 16
        else:
            hdr = 8
        if size == 0:
            size = end - f.tell() + hdr
        start = f.tell()
        body_end = start + size - hdr

        if typ == "mvhd":
            ver = f.read(1)[0]
            f.read(3)
            if ver == 1:
                f.read(8); f.read(8)
                scale = struct.unpack(">I", f.read(4))[0]
                dur = struct.unpack(">Q", f.read(8))[0]
            else:
                f.read(4); f.read(4)
                scale = struct.unpack(">I", f.read(4))[0]
                dur = struct.unpack(">I", f.read(4))[0]
            print(f"  duration = {dur/scale:.2f} s  (timescale={scale})")
        elif typ == "tkhd":
            ver = f.read(1)[0]
            f.read(3)
            if ver == 1:
                f.read(8); f.read(8); f.read(4); f.read(2); f.read(2); f.read(8)
            else:
                f.read(4); f.read(4); f.read(4); f.read(2); f.read(2); f.read(8)
            f.read(4)  # reserved
            f.read(36)  # matrix
            w = struct.unpack(">I", f.read(4))[0] >> 16
            h = struct.unpack(">I", f.read(4))[0] >> 16
            print(f"  track size = {w}x{h}")
        elif typ in ("moov", "trak", "mdia"):
            if typ in ("moov", "trak"):
                f.read(4)  # mvhd/tkhd 已在上面处理容器头
            walk(f, body_end, depth + 1)

        f.seek(body_end)


for p in sys.argv[1:]:
    path = Path(p)
    size = path.stat().st_size
    print(f"{path.name}  ({size/1024/1024:.2f} MB)")
    with path.open("rb") as f:
        walk(f, size)
