# -*- coding: utf-8 -*-
"""从 mp4 抽指定秒的帧存 png。用法: python extract_frames.py video.mp4 5 15 30"""
import sys

import imageio.v3 as iio

video = sys.argv[1]
fps_meta = iio.immeta(video).get("fps", 50)
for t in [float(x) for x in sys.argv[2:]]:
    idx = int(t * fps_meta)
    frame = iio.imread(video, index=idx, plugin="pyav") if False else None
    # imageio ffmpeg 插件按帧号读
    frame = iio.imread(video, index=idx, plugin="FFMPEG")
    out = video.rsplit(".", 1)[0] + f"_t{int(t):02d}.png"
    iio.imwrite(out, frame)
    print("saved", out)
