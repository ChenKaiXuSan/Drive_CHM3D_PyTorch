#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/bundle_adjustment/visualization/merge.py
Project: /workspace/code/bundle_adjustment/visualization
Created Date: Sunday December 7th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Sunday December 7th 2025 4:32:17 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)

def _ensure_rgb_uint8(img: np.ndarray) -> np.ndarray:
    """把任意读出的图像变成 (H, W, 3) uint8"""
    if img is None:
        raise ValueError("Got None image")

    img = np.asarray(img)

    # 灰度图 (H, W)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)

    # (H, W, C)
    elif img.ndim == 3:
        if img.shape[2] == 1:
            img = np.concatenate([img] * 3, axis=-1)
        elif img.shape[2] == 4:
            # 丢掉 alpha 通道
            img = img[:, :, :3]
        elif img.shape[2] == 3:
            pass
        else:
            raise ValueError(f"Unsupported channel number: {img.shape}")

    else:
        raise ValueError(f"Unsupported image shape: {img.shape}")

    # 转成 uint8
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    return img


def merge_frame_to_video(save_path: Path, flag: str, fps: int = 30) -> None:
    frame_dir = save_path / flag
    out_path = save_path / "video"
    out_path.mkdir(exist_ok=True, parents=True)

    frames = sorted(frame_dir.glob("*"), key=lambda x: int(x.stem.split("_")[0]))
    if not frames:
        raise RuntimeError(f"No frames found in {frame_dir}")

    output_file = out_path / f"{flag}.mp4"

    # 先读一帧，确定尺寸
    first_img = cv2.imread(str(frames[0]), cv2.IMREAD_UNCHANGED)
    first_img = _ensure_rgb_uint8(first_img)
    H, W = first_img.shape[:2]

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_file), fourcc, fps, (W, H))
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {output_file}")

    # 先写入第一帧
    writer.write(first_img)

    for f in tqdm(frames[1:], desc=f"Save {flag}"):
        img = cv2.imread(str(f), cv2.IMREAD_UNCHANGED)
        img = _ensure_rgb_uint8(img)

        # 如果尺寸不一致，resize 到第一帧大小
        if img.shape[0] != H or img.shape[1] != W:
            img = cv2.resize(img, (W, H), interpolation=cv2.INTER_LINEAR)

        writer.write(img)

    writer.release()
    print(f"Video saved to {output_file}")
