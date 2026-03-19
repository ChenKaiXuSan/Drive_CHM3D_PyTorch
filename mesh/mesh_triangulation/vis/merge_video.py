#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/merge_video.py
Project: /workspace/code/triangulation/vis
Created Date: Tuesday October 14th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday October 14th 2025 12:53:44 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import cv2
from pathlib import Path
import logging

logger = logging.getLogger(__name__)
import numpy as np
from typing import Iterable, Sequence, Union, Tuple, Optional


def merge_frames_to_video(
    frame_dir: str, output_video_path: str, fps: int = 30
) -> None:
    """
    将指定目录下的图像帧合并为视频。
    frame_dir: 图像帧目录，假设命名格式为 frame_0000.png, frame_0001.png, ...
    output_video_path: 输出视频路径
    fps: 视频帧率
    """

    frame_files = sorted(Path(frame_dir).glob("frame_*.png"))
    if not frame_files:
        raise ValueError(f"No frames found in directory: {frame_dir}")

    # 读取第一张图片以获取尺寸信息
    first_frame = cv2.imread(str(frame_files[0]))
    height, width, _ = first_frame.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # 使用 mp4v 编码
    video_writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

    for frame_file in frame_files:
        frame = cv2.imread(str(frame_file))
        if frame is None:
            logger.warning(f"Could not read frame {frame_file}, skipping.")
            continue
        video_writer.write(frame)

    video_writer.release()
    logger.info(f"Video saved to {output_video_path}")
